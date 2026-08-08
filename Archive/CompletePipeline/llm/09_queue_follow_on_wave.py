#!/usr/bin/env python3
"""
Queue a follow-on Gemma shard run behind an in-flight endpoint and manage the
endpoint lifecycle.

Behavior:
    1. computes the remaining cached papers not already completed or assigned
    2. writes a queued selection manifest
    3. watches the upstream wave for progress and shuts it down if it stalls
    4. launches the follow-on run once the upstream wave completes
    5. watches the follow-on run and stops the endpoint when the queue empties
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from gemma_lifecycle_utils import (
    count_paper_results,
    is_orchestrator_complete,
    now_utc,
    read_endpoint_names,
    read_orchestrator_status,
    start_endpoint_names,
    stop_endpoint_names,
    terminate_pid,
    write_json,
)


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
DEFAULT_WAIT_OUTPUT_ROOT = ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave2_incremental"
DEFAULT_ENDPOINT_FILE = ROOT / "OutputNew" / "LLMOutput" / "gemma10_cached_stage1" / "ready_endpoint_pool08.txt"
DEFAULT_QUEUE_RUN_SLUG = "gemma_ready8_wave3_followon_queue"
DEFAULT_FOLLOW_ON_RUN_SLUG = "gemma_ready8_wave3_followon"
DEFAULT_ORCHESTRATOR = Path(__file__).resolve().parent / "08_orchestrate_gemma_shards.py"
DEFAULT_WORKER_PYTHON = Path(".venv-coarse/bin/python")
DEFAULT_KEY_FILE = ROOT / "key.txt"
DEFAULT_PROBE_OUTPUT_ROOT = ROOT / "OutputNew" / "Empirics" / "rdd_bandwidth_2018_2020__gemma4_dedicated_probe10_single_instance"
DEFAULT_TRACK_ROOTS = [
    ROOT / "OutputNew" / "Empirics" / "gemma_ready7_wave1_cached_v2",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave2_incremental",
]
DEFAULT_ASSIGNED_SELECTION_ROOTS = [
    ROOT / "OutputNew" / "LLMOutput" / "gemma_ready7_wave1_cached_v2",
    ROOT / "OutputNew" / "LLMOutput" / "gemma_ready8_wave2_incremental",
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def load_assigned_paper_ids(selection_roots: list[Path]) -> set[str]:
    assigned: set[str] = set()
    for root in selection_roots:
        if not root.exists():
            continue
        for path in root.glob("shard_*/selected_papers.jsonl"):
            for row in load_jsonl(path):
                assigned.add(str(row["paper_id"]))
    return assigned


def load_completed_paper_ids(output_roots: list[Path]) -> set[str]:
    completed: set[str] = set()
    for root in output_roots:
        if not root.exists():
            continue
        for path in root.glob("**/papers/*/paper_result.json"):
            completed.add(path.parent.name)
    return completed


def load_remaining_rows(
    *,
    selected_jsonl: Path,
    fulltext_dir: Path,
    assigned_selection_roots: list[Path],
    completed_output_roots: list[Path],
) -> list[dict[str, Any]]:
    selected_rows = load_jsonl(selected_jsonl)
    assigned_ids = load_assigned_paper_ids(assigned_selection_roots)
    completed_ids = load_completed_paper_ids(completed_output_roots)

    remaining_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        paper_id = str(row["paper_id"])
        if paper_id in assigned_ids or paper_id in completed_ids:
            continue
        fulltext_path = fulltext_dir / f"{paper_id}.txt"
        if not fulltext_path.exists():
            continue
        remaining_rows.append(row)

    remaining_rows.sort(key=lambda row: (int(row["year"]), str(row["paper_id"])))
    return remaining_rows


def summarize_output_root(output_root: Path) -> dict[str, Any]:
    status = read_orchestrator_status(output_root)
    payload = {
        "output_root": str(output_root),
        "paper_results": count_paper_results(output_root),
        "status_path": str(output_root / "orchestrator_status.json"),
        "status": status,
    }
    summary_path = output_root / "aggregate_summary.json"
    if summary_path.exists():
        payload["aggregate_summary_path"] = str(summary_path)
        try:
            payload["aggregate_summary"] = load_json(summary_path)
        except json.JSONDecodeError:
            payload["aggregate_summary"] = None
    return payload


def build_follow_on_command(
    *,
    orchestrator_path: Path,
    endpoint_file: Path,
    selected_jsonl: Path,
    run_slug: str,
    worker_python: Path,
    skip_completed_output_root: Path | None,
    committee_bias: str,
) -> list[str]:
    command = [
        str(worker_python.resolve()),
        str(orchestrator_path),
        "--endpoint-file",
        str(endpoint_file),
        "--selected-jsonl",
        str(selected_jsonl),
        "--python-executable",
        str(worker_python),
        "--run-slug",
        run_slug,
        "--committee-bias",
        committee_bias,
        "--launch",
        "--poll-seconds",
        "60",
    ]
    if skip_completed_output_root is not None:
        command.extend(["--skip-completed-output-root", str(skip_completed_output_root)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue a follow-on cached Gemma shard run and track progress.")
    parser.add_argument("--selected-jsonl", type=Path, default=DEFAULT_SELECTED_JSONL)
    parser.add_argument("--fulltext-dir", type=Path, default=DEFAULT_FULLTEXT_DIR)
    parser.add_argument("--wait-output-root", type=Path, default=DEFAULT_WAIT_OUTPUT_ROOT)
    parser.add_argument("--endpoint-file", type=Path, default=DEFAULT_ENDPOINT_FILE)
    parser.add_argument("--orchestrator-path", type=Path, default=DEFAULT_ORCHESTRATOR)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--queue-run-slug", default=DEFAULT_QUEUE_RUN_SLUG)
    parser.add_argument("--follow-on-run-slug", default=DEFAULT_FOLLOW_ON_RUN_SLUG)
    parser.add_argument("--committee-bias", choices=("plain", "positive"), default="plain")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stale-minutes", type=float, default=15.0)
    parser.add_argument("--track-output-root", type=Path, action="append", default=[])
    parser.add_argument("--assigned-selection-root", type=Path, action="append", default=[])
    parser.add_argument("--completed-output-root", type=Path, action="append", default=[])
    return parser.parse_args()


def has_failed_worker(status: dict[str, Any] | None) -> bool:
    if not status:
        return False
    return any(str(row.get("state")) == "failed" for row in (status.get("workers") or []))


def running_worker_pids(status: dict[str, Any] | None) -> list[int]:
    pids: list[int] = []
    if not status:
        return pids
    for row in status.get("workers") or []:
        if str(row.get("state")) == "running" and isinstance(row.get("pid"), int):
            pids.append(int(row["pid"]))
    return pids


def build_progress_state(output_root: Path) -> dict[str, Any]:
    return {
        "output_root": str(output_root),
        "completed_papers": count_paper_results(output_root),
        "last_progress_utc": now_utc(),
        "last_progress_epoch": time.time(),
    }


def update_progress(progress: dict[str, Any], output_root: Path) -> None:
    completed = count_paper_results(output_root)
    if completed > int(progress["completed_papers"]):
        progress["completed_papers"] = completed
        progress["last_progress_utc"] = now_utc()
        progress["last_progress_epoch"] = time.time()


def shutdown_wave(
    *,
    key_file: Path,
    endpoint_names: list[str],
    status: dict[str, Any] | None,
) -> dict[str, Any]:
    pid_results = []
    for pid in running_worker_pids(status):
        pid_results.append({"pid": pid, "result": terminate_pid(pid)})
    endpoint_results = stop_endpoint_names(key_file, endpoint_names)
    return {
        "shutdown_at_utc": now_utc(),
        "pid_results": pid_results,
        "endpoint_results": endpoint_results,
    }


def main() -> None:
    args = parse_args()
    queue_selection_root = (ROOT / "OutputNew" / "LLMOutput" / args.queue_run_slug).resolve()
    queue_output_root = (ROOT / "OutputNew" / "Empirics" / args.queue_run_slug).resolve()
    follow_on_selection_root = (ROOT / "OutputNew" / "LLMOutput" / args.follow_on_run_slug).resolve()
    follow_on_output_root = (ROOT / "OutputNew" / "Empirics" / args.follow_on_run_slug).resolve()
    queue_selection_root.mkdir(parents=True, exist_ok=True)
    queue_output_root.mkdir(parents=True, exist_ok=True)

    assigned_selection_roots = [path.resolve() for path in (args.assigned_selection_root or DEFAULT_ASSIGNED_SELECTION_ROOTS)]
    completed_output_roots = [path.resolve() for path in (args.completed_output_root or [DEFAULT_PROBE_OUTPUT_ROOT])]
    tracked_roots = [path.resolve() for path in (args.track_output_root or DEFAULT_TRACK_ROOTS)]
    queue_endpoint_names = read_endpoint_names(args.endpoint_file.resolve())

    remaining_rows = load_remaining_rows(
        selected_jsonl=args.selected_jsonl.resolve(),
        fulltext_dir=args.fulltext_dir.resolve(),
        assigned_selection_roots=assigned_selection_roots,
        completed_output_roots=completed_output_roots,
    )
    if not remaining_rows:
        raise ValueError("No remaining cached papers to queue.")

    queued_selected_jsonl = queue_selection_root / "selected_papers.jsonl"
    write_jsonl(queued_selected_jsonl, remaining_rows)
    years = Counter(int(row["year"]) for row in remaining_rows)

    follow_on_command = build_follow_on_command(
        orchestrator_path=args.orchestrator_path.resolve(),
        endpoint_file=args.endpoint_file.resolve(),
        selected_jsonl=queued_selected_jsonl,
        run_slug=args.follow_on_run_slug,
        worker_python=args.worker_python,
        skip_completed_output_root=completed_output_roots[0] if completed_output_roots else None,
        committee_bias=args.committee_bias,
    )

    queue_manifest = {
        "created_at_utc": now_utc(),
        "queue_run_slug": args.queue_run_slug,
        "follow_on_run_slug": args.follow_on_run_slug,
        "queued_selected_jsonl": str(queued_selected_jsonl),
        "n_remaining_papers": len(remaining_rows),
        "years": dict(sorted(years.items())),
        "wait_output_root": str(args.wait_output_root.resolve()),
        "endpoint_file": str(args.endpoint_file.resolve()),
        "endpoint_names": queue_endpoint_names,
        "committee_bias": args.committee_bias,
        "assigned_selection_roots": [str(path) for path in assigned_selection_roots],
        "completed_output_roots": [str(path) for path in completed_output_roots],
        "tracked_output_roots": [str(path) for path in tracked_roots],
        "follow_on_command": follow_on_command,
        "stale_minutes": float(args.stale_minutes),
    }
    write_json(queue_selection_root / "queue_manifest.json", queue_manifest)

    launched = (follow_on_selection_root / "orchestrator_manifest.json").exists() or (follow_on_output_root / "orchestrator_status.json").exists()
    launch_pid: int | None = None
    shutdown_result: dict[str, Any] | None = None
    stale_minutes = float(args.stale_minutes)
    wait_progress = build_progress_state(args.wait_output_root.resolve())
    follow_progress = build_progress_state(follow_on_output_root)

    while True:
        phase = "monitoring_follow_on" if launched else "waiting_for_endpoint"
        tracked = tracked_roots + ([follow_on_output_root] if launched else [])
        tracked_payload = {root.name: summarize_output_root(root) for root in tracked}

        wait_status = read_orchestrator_status(args.wait_output_root.resolve())
        wait_complete = is_orchestrator_complete(wait_status)
        update_progress(wait_progress, args.wait_output_root.resolve())
        wait_minutes_since_progress = round((time.time() - float(wait_progress["last_progress_epoch"])) / 60.0, 2)

        if not launched and has_failed_worker(wait_status):
            shutdown_result = shutdown_wave(
                key_file=args.key_file.resolve(),
                endpoint_names=queue_endpoint_names,
                status=wait_status,
            )
            phase = "wait_wave_failed_shutdown"
        elif not launched and not wait_complete and wait_minutes_since_progress >= stale_minutes:
            shutdown_result = shutdown_wave(
                key_file=args.key_file.resolve(),
                endpoint_names=queue_endpoint_names,
                status=wait_status,
            )
            phase = "wait_wave_stale_shutdown"
        elif not launched and wait_complete:
            # Keep the endpoint warm if the follow-on can start immediately, but
            # ensure the endpoint is up in case it was manually stopped.
            start_endpoint_names(args.key_file.resolve(), queue_endpoint_names)
            launch_log_path = queue_output_root / "follow_on_launch.log"
            with launch_log_path.open("w", encoding="utf-8") as handle:
                proc = subprocess.Popen(
                    follow_on_command,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            launched = True
            launch_pid = proc.pid
            phase = "monitoring_follow_on"
            tracked = tracked_roots + [follow_on_output_root]
            tracked_payload = {root.name: summarize_output_root(root) for root in tracked}

        follow_on_status = read_orchestrator_status(follow_on_output_root) if launched else None
        if launched:
            update_progress(follow_progress, follow_on_output_root)
        follow_minutes_since_progress = round((time.time() - float(follow_progress["last_progress_epoch"])) / 60.0, 2)
        follow_on_complete = launched and is_orchestrator_complete(follow_on_status)

        if launched and has_failed_worker(follow_on_status):
            shutdown_result = shutdown_wave(
                key_file=args.key_file.resolve(),
                endpoint_names=queue_endpoint_names,
                status=follow_on_status,
            )
            phase = "follow_on_failed_shutdown"
        elif launched and not follow_on_complete and follow_minutes_since_progress >= stale_minutes:
            shutdown_result = shutdown_wave(
                key_file=args.key_file.resolve(),
                endpoint_names=queue_endpoint_names,
                status=follow_on_status,
            )
            phase = "follow_on_stale_shutdown"
        elif follow_on_complete:
            shutdown_result = shutdown_wave(
                key_file=args.key_file.resolve(),
                endpoint_names=queue_endpoint_names,
                status=follow_on_status,
            )

        queue_status = {
            "updated_at_utc": now_utc(),
            "phase": "completed" if follow_on_complete else phase,
            "queue_run_slug": args.queue_run_slug,
            "follow_on_run_slug": args.follow_on_run_slug,
            "queue_selection_root": str(queue_selection_root),
            "queue_output_root": str(queue_output_root),
            "queued_selected_jsonl": str(queued_selected_jsonl),
            "n_remaining_papers": len(remaining_rows),
            "years": dict(sorted(years.items())),
            "stale_minutes": stale_minutes,
            "wait_output_root": str(args.wait_output_root.resolve()),
            "wait_complete": wait_complete,
            "wait_progress": {
                "completed_papers": int(wait_progress["completed_papers"]),
                "last_progress_utc": wait_progress["last_progress_utc"],
                "minutes_since_progress": wait_minutes_since_progress,
            },
            "follow_on_selection_root": str(follow_on_selection_root),
            "follow_on_output_root": str(follow_on_output_root),
            "follow_on_launched": launched,
            "follow_on_launch_pid": launch_pid,
            "follow_on_progress": {
                "completed_papers": int(follow_progress["completed_papers"]),
                "last_progress_utc": follow_progress["last_progress_utc"],
                "minutes_since_progress": follow_minutes_since_progress,
            },
            "follow_on_command": follow_on_command,
            "shutdown_result": shutdown_result,
            "tracked_runs": tracked_payload,
        }
        write_json(queue_output_root / "queue_status.json", queue_status)

        if phase.endswith("_shutdown") or follow_on_complete:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
