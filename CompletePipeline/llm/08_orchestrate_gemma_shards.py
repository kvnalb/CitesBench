#!/usr/bin/env python3
"""
Shard cached RDD papers across multiple dedicated Gemma endpoints and optionally
launch one worker process per endpoint.

This script reuses 07_run_rdd_bandwidth_coarse_reviews.py as the worker. The
orchestrator itself does not call any LLMs; it only:
    1. loads the selected-paper manifest
    2. filters to papers with local extracted text
    3. balances papers across endpoints using greedy bin packing on text size
    4. writes per-shard manifests and launch commands
    5. optionally launches and monitors one worker per endpoint
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gemma_lifecycle_utils import start_endpoint_names, stop_endpoint_names, terminate_pid


def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_SELECTED_JSONL = (
    ROOT / "OutputNew" / "Empirics" / "rdd_bandwidth_2018_2020__gemma4_dedicated_stage1" / "prefetch_selected_papers.jsonl"
)
DEFAULT_FULLTEXT_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview" / "rdd_bandwidth_2018_2020__gemma4_dedicated_stage1" / "fulltext"
DEFAULT_PDF_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview" / "rdd_bandwidth_2018_2020__gemma4_dedicated_stage1" / "pdf"
DEFAULT_RUNNER = Path(__file__).resolve().parent / "07_run_rdd_bandwidth_coarse_reviews.py"
DEFAULT_RDD_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
DEFAULT_OPENREVIEW_CSV = ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview" / "openreview_yearly_submissions.csv"
DEFAULT_KEY_FILE = ROOT / "key.txt"
DEFAULT_DECISION_HEAD_MODELS = "deepseek-v3.1"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    pieces: list[str] = []
    for ch in value:
        pieces.append(ch if ch.isalnum() else "_")
    return "".join(pieces).strip("_").lower()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def parse_endpoint_names(values: list[str], file_path: Path | None) -> list[str]:
    names: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned:
            names.append(cleaned)
    if file_path is not None:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                names.append(cleaned)
    unique_names = list(dict.fromkeys(names))
    if not unique_names:
        raise ValueError("At least one endpoint name must be provided.")
    return unique_names


def load_completed_paper_ids(output_root: Path) -> set[str]:
    completed: set[str] = set()
    if not output_root.exists():
        return completed
    for path in output_root.glob("**/papers/*/paper_result.json"):
        completed.add(path.parent.name)
    return completed


def load_assigned_paper_ids(selection_root: Path) -> set[str]:
    assigned: set[str] = set()
    if not selection_root.exists():
        return assigned
    for path in selection_root.glob("shard_*/selected_papers.jsonl"):
        for row in load_jsonl(path):
            assigned.add(str(row["paper_id"]))
    return assigned


def count_worker_artifacts(output_dir: Path) -> dict[str, int]:
    if not output_dir.exists():
        return {"paper_results": 0, "failures": 0, "coarse_reviews": 0, "deepseek_decisions": 0, "total": 0}
    paper_results = sum(1 for _ in output_dir.glob("papers/*/paper_result.json"))
    failures = sum(1 for _ in output_dir.glob("papers/*/failure.json"))
    coarse_reviews = sum(1 for _ in output_dir.glob("papers/*/coarse_review.json"))
    deepseek_decisions = sum(1 for _ in output_dir.glob("papers/*/deepseek_decision.json"))
    return {
        "paper_results": paper_results,
        "failures": failures,
        "coarse_reviews": coarse_reviews,
        "deepseek_decisions": deepseek_decisions,
        "total": paper_results + failures + coarse_reviews + deepseek_decisions,
    }


def load_candidate_papers(
    *,
    selected_jsonl: Path,
    fulltext_dir: Path,
    max_papers: int | None,
    skip_completed_output_root: Path | None,
    skip_assigned_selection_root: Path | None,
) -> list[dict[str, Any]]:
    selected_rows = load_jsonl(selected_jsonl)
    completed_ids = load_completed_paper_ids(skip_completed_output_root) if skip_completed_output_root else set()
    assigned_ids = load_assigned_paper_ids(skip_assigned_selection_root) if skip_assigned_selection_root else set()

    candidates: list[dict[str, Any]] = []
    for row in selected_rows:
        paper_id = str(row["paper_id"])
        if paper_id in completed_ids:
            continue
        if paper_id in assigned_ids:
            continue
        fulltext_path = fulltext_dir / f"{paper_id}.txt"
        if not fulltext_path.exists():
            continue
        enriched = dict(row)
        enriched["fulltext_path"] = str(fulltext_path)
        enriched["fulltext_bytes"] = fulltext_path.stat().st_size
        candidates.append(enriched)

    candidates.sort(key=lambda row: (-int(row["fulltext_bytes"]), int(row["year"]), str(row["paper_id"])))
    if max_papers is not None:
        candidates = candidates[:max_papers]
    return candidates


def build_balanced_shards(
    *,
    endpoint_names: list[str],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shards = [
        {
            "index": idx + 1,
            "endpoint_name": endpoint_name,
            "papers": [],
            "total_fulltext_bytes": 0,
        }
        for idx, endpoint_name in enumerate(endpoint_names)
    ]

    for row in candidates:
        target = min(
            shards,
            key=lambda shard: (
                int(shard["total_fulltext_bytes"]),
                len(shard["papers"]),
                int(shard["index"]),
            ),
        )
        target["papers"].append(row)
        target["total_fulltext_bytes"] += int(row["fulltext_bytes"])

    for shard in shards:
        shard["n_papers"] = len(shard["papers"])
        shard["years"] = dict(sorted(Counter(int(row["year"]) for row in shard["papers"]).items()))
        shard["paper_ids"] = [str(row["paper_id"]) for row in shard["papers"]]
    return shards


def build_worker_command(
    *,
    python_executable: Path,
    runner_path: Path,
    shard: dict[str, Any],
    shard_selection_dir: Path,
    shard_output_dir: Path,
    key_file: Path,
    rdd_csv: Path,
    openreview_csv: Path,
    pdf_dir: Path,
    fulltext_dir: Path,
    decision_head_models: str,
    committee_bias: str,
    timeout_seconds: float,
    max_consecutive_failures: int,
    head_max_tokens: int,
    head_temperature: float,
    head_top_p: float,
    max_content_chars: int,
    section_char_limit: int,
    intro_max_chars: int,
    method_max_chars: int,
    conclusion_max_chars: int,
    stage: str,
    overwrite: bool,
) -> list[str]:
    command = [
        str(python_executable),
        str(runner_path),
        "--key-file",
        str(key_file),
        "--rdd-csv",
        str(rdd_csv),
        "--openreview-csv",
        str(openreview_csv),
        "--committee-model",
        f"together_ai/{shard['endpoint_name']}",
        "--committee-bias",
        committee_bias,
        "--decision-head-models",
        decision_head_models,
        "--max-parallel-papers",
        "1",
        "--timeout-seconds",
        str(timeout_seconds),
        "--max-consecutive-failures",
        str(max_consecutive_failures),
        "--stage",
        stage,
        "--selection-dir",
        str(shard_selection_dir),
        "--output-dir",
        str(shard_output_dir),
        "--pdf-dir",
        str(pdf_dir),
        "--fulltext-dir",
        str(fulltext_dir),
        "--head-max-tokens",
        str(head_max_tokens),
        "--head-temperature",
        str(head_temperature),
        "--head-top-p",
        str(head_top_p),
        "--max-content-chars",
        str(max_content_chars),
        "--section-char-limit",
        str(section_char_limit),
        "--intro-max-chars",
        str(intro_max_chars),
        "--method-max-chars",
        str(method_max_chars),
        "--conclusion-max-chars",
        str(conclusion_max_chars),
    ]
    if overwrite:
        command.append("--overwrite")
    for paper_id in shard["paper_ids"]:
        command.extend(["--paper-id", paper_id])
    return command


def build_orchestrator_summary(shards: list[dict[str, Any]], launch: bool) -> dict[str, Any]:
    years = Counter()
    total_bytes = 0
    for shard in shards:
        years.update(Counter(int(row["year"]) for row in shard["papers"]))
        total_bytes += int(shard["total_fulltext_bytes"])
    return {
        "created_at_utc": now_utc(),
        "launch_requested": launch,
        "n_shards": len(shards),
        "n_papers": sum(int(shard["n_papers"]) for shard in shards),
        "years": dict(sorted(years.items())),
        "total_fulltext_bytes": total_bytes,
    }


def terminate_running_processes(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for proc in processes:
        process = proc["process"]
        if process.poll() is not None:
            continue
        results.append(
            {
                "shard_index": proc["shard_index"],
                "pid": process.pid,
                "result": terminate_pid(int(process.pid)),
            }
        )
    return results


def stop_endpoints_safely(
    *,
    key_file: Path,
    endpoint_names: list[str],
    reason: str,
) -> dict[str, Any]:
    try:
        endpoint_records = stop_endpoint_names(key_file, endpoint_names)
        error = None
    except Exception as exc:
        endpoint_records = []
        error = str(exc)
    return {
        "stopped_at_utc": now_utc(),
        "reason": reason,
        "endpoint_names": endpoint_names,
        "endpoint_records": endpoint_records,
        "error": error,
    }


def monitor_processes(
    *,
    processes: list[dict[str, Any]],
    status_path: Path,
    poll_seconds: float,
    manage_endpoints: bool,
    key_file: Path,
    endpoint_names: list[str],
    stale_minutes: float,
) -> dict[str, Any]:
    started = time.time()
    endpoint_stop_result: dict[str, Any] | None = None
    shutdown_reason: str | None = None
    progress_state: dict[str, dict[str, Any]] = {}
    for proc in processes:
        counts = count_worker_artifacts(Path(str(proc["output_dir"])))
        progress_state[str(proc["shard_index"])] = {
            "artifact_total": int(counts["total"]),
            "last_progress_utc": now_utc(),
            "last_progress_epoch": time.time(),
        }

    while True:
        running = 0
        finished = 0
        failed = 0
        stale_workers: list[str] = []
        failed_workers: list[str] = []
        status_rows: list[dict[str, Any]] = []
        for proc in processes:
            returncode = proc["process"].poll()
            shard_index = str(proc["shard_index"])
            output_dir = Path(str(proc["output_dir"]))
            artifact_counts = count_worker_artifacts(output_dir)
            proc_progress = progress_state[shard_index]
            if int(artifact_counts["total"]) > int(proc_progress["artifact_total"]):
                proc_progress["artifact_total"] = int(artifact_counts["total"])
                proc_progress["last_progress_utc"] = now_utc()
                proc_progress["last_progress_epoch"] = time.time()
            minutes_since_progress = round((time.time() - float(proc_progress["last_progress_epoch"])) / 60.0, 2)

            if returncode is None:
                running += 1
                state = "running"
                if stale_minutes > 0 and minutes_since_progress >= stale_minutes:
                    stale_workers.append(shard_index)
            else:
                finished += 1
                state = "completed" if returncode == 0 else "failed"
                if returncode != 0:
                    failed += 1
                    failed_workers.append(shard_index)
            status_rows.append(
                {
                    "shard_index": shard_index,
                    "endpoint_name": proc["endpoint_name"],
                    "pid": proc["process"].pid,
                    "state": state,
                    "returncode": returncode,
                    "log_path": proc["log_path"],
                    "output_dir": proc["output_dir"],
                    "artifact_counts": artifact_counts,
                    "last_progress_utc": proc_progress["last_progress_utc"],
                    "minutes_since_progress": minutes_since_progress,
                }
            )

        status_payload = {
            "updated_at_utc": now_utc(),
            "elapsed_minutes": round((time.time() - started) / 60.0, 2),
            "running": running,
            "finished": finished,
            "failed": failed,
            "managed_endpoints": manage_endpoints,
            "stale_minutes": stale_minutes,
            "shutdown_reason": shutdown_reason,
            "endpoint_stop_result": endpoint_stop_result,
            "workers": status_rows,
        }
        write_json(status_path, status_payload)

        if failed_workers:
            shutdown_reason = f"worker_failed:{','.join(failed_workers)}"
            pid_results = terminate_running_processes(processes)
            endpoint_stop_result = stop_endpoints_safely(
                key_file=key_file,
                endpoint_names=endpoint_names,
                reason=shutdown_reason,
            ) if manage_endpoints else None
            status_payload["shutdown_reason"] = shutdown_reason
            status_payload["pid_stop_results"] = pid_results
            status_payload["endpoint_stop_result"] = endpoint_stop_result
            write_json(status_path, status_payload)
            return status_payload

        if stale_workers:
            shutdown_reason = f"stale_no_progress:{','.join(stale_workers)}"
            pid_results = terminate_running_processes(processes)
            endpoint_stop_result = stop_endpoints_safely(
                key_file=key_file,
                endpoint_names=endpoint_names,
                reason=shutdown_reason,
            ) if manage_endpoints else None
            status_payload["shutdown_reason"] = shutdown_reason
            status_payload["pid_stop_results"] = pid_results
            status_payload["endpoint_stop_result"] = endpoint_stop_result
            write_json(status_path, status_payload)
            return status_payload

        if finished == len(processes):
            if manage_endpoints:
                endpoint_stop_result = stop_endpoints_safely(
                    key_file=key_file,
                    endpoint_names=endpoint_names,
                    reason="all_workers_finished",
                )
                status_payload["endpoint_stop_result"] = endpoint_stop_result
                write_json(status_path, status_payload)
            return status_payload
        time.sleep(poll_seconds)


def aggregate_finished_shards(output_root: Path) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    failed_shards: list[dict[str, Any]] = []
    for shard_dir in sorted(output_root.glob("shard_*")):
        summary_path = shard_dir / "summary.json"
        if summary_path.exists():
            summary_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
        else:
            failed_shards.append({"shard_dir": str(shard_dir), "error": "summary.json missing"})

    total_completed = 0
    total_failed = 0
    total_minutes = 0.0
    total_committee_cost = 0.0
    total_deepseek_cost = 0.0
    years = Counter()

    for row in summary_rows:
        total_completed += int(row.get("n_completed") or 0)
        total_failed += int(row.get("n_failed") or 0)
        total_minutes += float(row.get("elapsed_minutes") or 0.0)
        metrics = row.get("metrics") or {}
        total_committee_cost += float(metrics.get("committee_cost_usd") or 0.0)
        total_deepseek_cost += float(metrics.get("deepseek_estimated_cost_usd") or 0.0)
        years.update(Counter({int(k): int(v) for k, v in (metrics.get("years") or {}).items()}))

    return {
        "created_at_utc": now_utc(),
        "n_shards_finished": len(summary_rows),
        "n_shards_missing_summary": len(failed_shards),
        "n_completed": total_completed,
        "n_failed": total_failed,
        "years": dict(sorted(years.items())),
        "committee_cost_usd": round(total_committee_cost, 6),
        "deepseek_estimated_cost_usd": round(total_deepseek_cost, 6),
        "total_estimated_cost_usd": round(total_committee_cost + total_deepseek_cost, 6),
        "sum_worker_elapsed_minutes": round(total_minutes, 2),
        "missing_shards": failed_shards,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard cached RDD papers across Gemma endpoints and optionally launch one worker per endpoint.")
    parser.add_argument("--endpoint-name", action="append", default=[], help="Dedicated Together endpoint name, without the together_ai/ prefix. Repeat for multiple endpoints.")
    parser.add_argument("--endpoint-file", type=Path, default=None, help="Optional file with one endpoint name per line.")
    parser.add_argument("--selected-jsonl", type=Path, default=DEFAULT_SELECTED_JSONL)
    parser.add_argument("--fulltext-dir", type=Path, default=DEFAULT_FULLTEXT_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--runner-path", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--rdd-csv", type=Path, default=DEFAULT_RDD_CSV)
    parser.add_argument("--openreview-csv", type=Path, default=DEFAULT_OPENREVIEW_CSV)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--decision-head-models", default=DEFAULT_DECISION_HEAD_MODELS)
    parser.add_argument("--committee-bias", choices=("plain", "positive"), default="plain")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--stage", choices=("both", "committee_only", "decision_only"), default="both")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=8,
        help="Passed to each worker. A worker exits non-zero after this many consecutive paper failures.",
    )
    parser.add_argument("--head-max-tokens", type=int, default=1200)
    parser.add_argument("--head-temperature", type=float, default=0.0)
    parser.add_argument("--head-top-p", type=float, default=0.9)
    parser.add_argument("--max-content-chars", type=int, default=9000)
    parser.add_argument("--section-char-limit", type=int, default=1800)
    parser.add_argument("--intro-max-chars", type=int, default=4000)
    parser.add_argument("--method-max-chars", type=int, default=8000)
    parser.add_argument("--conclusion-max-chars", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--manage-endpoints",
        action="store_true",
        help="Start endpoints before launch and stop them when workers finish, fail, or stall.",
    )
    parser.add_argument(
        "--no-start-endpoints",
        action="store_true",
        help="With --manage-endpoints, skip endpoint startup and only manage shutdown.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=15.0,
        help="With --manage-endpoints, stop endpoints and kill workers after this many minutes with no output progress.",
    )
    parser.add_argument("--skip-completed-output-root", type=Path, default=None)
    parser.add_argument("--skip-assigned-selection-root", type=Path, default=None)
    parser.add_argument("--run-slug", default=None)
    parser.add_argument("--selection-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint_names = parse_endpoint_names(args.endpoint_name, args.endpoint_file.resolve() if args.endpoint_file else None)
    run_slug = args.run_slug or f"rdd_gemma_shards__{len(endpoint_names)}x__{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    selection_root = (args.selection_root or (ROOT / "OutputNew" / "LLMOutput" / run_slug)).resolve()
    output_root = (args.output_root or (ROOT / "OutputNew" / "Empirics" / run_slug)).resolve()
    selection_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_papers(
        selected_jsonl=args.selected_jsonl.resolve(),
        fulltext_dir=args.fulltext_dir.resolve(),
        max_papers=args.max_papers,
        skip_completed_output_root=args.skip_completed_output_root.resolve() if args.skip_completed_output_root else None,
        skip_assigned_selection_root=args.skip_assigned_selection_root.resolve() if args.skip_assigned_selection_root else None,
    )
    if not candidates:
        raise ValueError("No cached candidate papers matched the requested filters.")

    shards = build_balanced_shards(endpoint_names=endpoint_names, candidates=candidates)
    orchestrator_manifest = {
        **build_orchestrator_summary(shards, args.launch),
        "run_slug": run_slug,
        "selected_jsonl": str(args.selected_jsonl.resolve()),
        "fulltext_dir": str(args.fulltext_dir.resolve()),
        "pdf_dir": str(args.pdf_dir.resolve()),
        "runner_path": str(args.runner_path.resolve()),
        "python_executable": str(args.python_executable),
        "rdd_csv": str(args.rdd_csv.resolve()),
        "openreview_csv": str(args.openreview_csv.resolve()),
        "key_file": str(args.key_file.resolve()),
        "decision_head_models": args.decision_head_models,
        "committee_bias": args.committee_bias,
        "timeout_seconds": args.timeout_seconds,
        "max_consecutive_failures": args.max_consecutive_failures,
        "stage": args.stage,
        "manage_endpoints": args.manage_endpoints,
        "no_start_endpoints": args.no_start_endpoints,
        "stale_minutes": args.stale_minutes,
        "selection_root": str(selection_root),
        "output_root": str(output_root),
        "skip_completed_output_root": str(args.skip_completed_output_root.resolve()) if args.skip_completed_output_root else None,
        "skip_assigned_selection_root": str(args.skip_assigned_selection_root.resolve()) if args.skip_assigned_selection_root else None,
        "shards": [],
    }

    launch_script_lines = ["#!/bin/zsh", "set -euo pipefail", ""]

    for shard in shards:
        shard_name = f"shard_{int(shard['index']):02d}"
        shard_selection_dir = selection_root / shard_name
        shard_output_dir = output_root / shard_name
        shard_selection_dir.mkdir(parents=True, exist_ok=True)
        shard_output_dir.mkdir(parents=True, exist_ok=True)

        command = build_worker_command(
            python_executable=args.python_executable,
            runner_path=args.runner_path.resolve(),
            shard=shard,
            shard_selection_dir=shard_selection_dir,
            shard_output_dir=shard_output_dir,
            key_file=args.key_file.resolve(),
            rdd_csv=args.rdd_csv.resolve(),
            openreview_csv=args.openreview_csv.resolve(),
            pdf_dir=args.pdf_dir.resolve(),
            fulltext_dir=args.fulltext_dir.resolve(),
            decision_head_models=args.decision_head_models,
            committee_bias=args.committee_bias,
            timeout_seconds=args.timeout_seconds,
            max_consecutive_failures=args.max_consecutive_failures,
            head_max_tokens=args.head_max_tokens,
            head_temperature=args.head_temperature,
            head_top_p=args.head_top_p,
            max_content_chars=args.max_content_chars,
            section_char_limit=args.section_char_limit,
            intro_max_chars=args.intro_max_chars,
            method_max_chars=args.method_max_chars,
            conclusion_max_chars=args.conclusion_max_chars,
            stage=args.stage,
            overwrite=args.overwrite,
        )

        shard_manifest = {
            "created_at_utc": now_utc(),
            "run_slug": run_slug,
            "shard_name": shard_name,
            "endpoint_name": shard["endpoint_name"],
            "n_papers": shard["n_papers"],
            "years": shard["years"],
            "total_fulltext_bytes": shard["total_fulltext_bytes"],
            "paper_ids": shard["paper_ids"],
            "selection_dir": str(shard_selection_dir),
            "output_dir": str(shard_output_dir),
            "command": command,
        }
        write_json(shard_selection_dir / "shard_manifest.json", shard_manifest)
        write_jsonl(shard_selection_dir / "selected_papers.jsonl", shard["papers"])
        launch_script_lines.append(f"# {shard_name} -> {shard['endpoint_name']}")
        launch_script_lines.append(shlex.join(command))
        launch_script_lines.append("")

        orchestrator_manifest["shards"].append(
            {
                "shard_name": shard_name,
                "endpoint_name": shard["endpoint_name"],
                "n_papers": shard["n_papers"],
                "years": shard["years"],
                "total_fulltext_bytes": shard["total_fulltext_bytes"],
                "selection_dir": str(shard_selection_dir),
                "output_dir": str(shard_output_dir),
            }
        )

    write_json(selection_root / "orchestrator_manifest.json", orchestrator_manifest)
    (selection_root / "launch_commands.sh").write_text("\n".join(launch_script_lines), encoding="utf-8")

    print(
        f"Prepared {len(shards)} shards covering {len(candidates)} cached papers. "
        f"Manifest: {selection_root / 'orchestrator_manifest.json'}"
    )

    if not args.launch:
        return

    processes: list[dict[str, Any]] = []
    endpoint_start_result: list[dict[str, Any]] | None = None
    try:
        if args.manage_endpoints and not args.no_start_endpoints:
            endpoint_start_result = start_endpoint_names(args.key_file.resolve(), endpoint_names)
            write_json(
                output_root / "endpoint_start_result.json",
                {
                    "started_at_utc": now_utc(),
                    "endpoint_names": endpoint_names,
                    "endpoint_records": endpoint_start_result,
                },
            )

        for shard_row in orchestrator_manifest["shards"]:
            shard_name = str(shard_row["shard_name"])
            shard_manifest_path = selection_root / shard_name / "shard_manifest.json"
            shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            log_path = output_root / f"{shard_name}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                shard_manifest["command"],
                cwd=str(ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(
                {
                    "shard_index": shard_name,
                    "endpoint_name": shard_row["endpoint_name"],
                    "process": proc,
                    "log_path": str(log_path),
                    "output_dir": str(shard_row["output_dir"]),
                    "log_handle": log_handle,
                }
            )
            print(
                f"Launched {shard_name} on {shard_row['endpoint_name']} "
                f"with {shard_row['n_papers']} papers. Log: {log_path}"
            )

        status_payload = monitor_processes(
            processes=processes,
            status_path=output_root / "orchestrator_status.json",
            poll_seconds=args.poll_seconds,
            manage_endpoints=args.manage_endpoints,
            key_file=args.key_file.resolve(),
            endpoint_names=endpoint_names,
            stale_minutes=args.stale_minutes,
        )
        aggregate_payload = aggregate_finished_shards(output_root)
        write_json(output_root / "aggregate_summary.json", aggregate_payload)
        print(json.dumps({"status": status_payload, "aggregate": aggregate_payload}, indent=2))
        if status_payload.get("shutdown_reason"):
            raise SystemExit(2)
    finally:
        active_processes = [proc for proc in processes if proc["process"].poll() is None]
        if args.manage_endpoints and active_processes:
            emergency_shutdown = {
                "created_at_utc": now_utc(),
                "reason": "orchestrator_finally_active_processes",
                "pid_stop_results": terminate_running_processes(processes),
                "endpoint_stop_result": stop_endpoints_safely(
                    key_file=args.key_file.resolve(),
                    endpoint_names=endpoint_names,
                    reason="orchestrator_finally_active_processes",
                ),
            }
            write_json(output_root / "emergency_shutdown.json", emergency_shutdown)
        for proc in processes:
            try:
                proc["log_handle"].close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
