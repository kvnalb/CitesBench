#!/usr/bin/env python3
"""
Summarize pairwise model sweep runs and rank them by quality, speed, and cost.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any
from urllib import request

from _abstract_review_common import TOGETHER_API_URL, now_utc, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "LLMOutput"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pairwise benchmark runs.")
    parser.add_argument(
        "--runs",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit run directories to include. If omitted, use --glob under LLMOutput/.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Optional glob under LLMOutput/, e.g. 'pairwise_consensus10_*'.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable summary JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional path for markdown summary.",
    )
    return parser.parse_args()


def discover_runs(args: argparse.Namespace) -> list[Path]:
    runs: list[Path] = []
    if args.runs:
        runs.extend(path.resolve() for path in args.runs)
    if args.glob:
        runs.extend(sorted(DEFAULT_OUTPUT_ROOT.glob(args.glob)))
    deduped = []
    seen = set()
    for path in runs:
        if path in seen:
            continue
        if not path.is_dir():
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fetch_live_pricing() -> dict[str, dict[str, float]]:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        return {}
    req = request.Request(
        "https://api.together.xyz/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"},
        method="GET",
    )
    with request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pricing = {}
    for row in payload:
        model_id = row.get("id") or row.get("name")
        model_pricing = row.get("pricing") or {}
        if not model_id:
            continue
        pricing[model_id] = {
            "input": float(model_pricing.get("input") or 0.0),
            "output": float(model_pricing.get("output") or 0.0),
        }
    return pricing


def collect_usage(run_dir: Path) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    empty_raw_calls = 0
    elapsed_seconds: list[float] = []
    total_calls = 0
    winners: dict[str, int] = {"A": 0, "B": 0, "Tie": 0}

    for filename in ("judgments.jsonl", "judgments_reask.jsonl"):
        path = run_dir / filename
        if not path.exists():
            continue
        for row in load_jsonl(path):
            winner = str(row.get("final_winner") or "Tie")
            if winner in winners:
                winners[winner] += 1
            for call in row.get("calls", []):
                usage = call.get("usage") or {}
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                if not str(call.get("raw_text") or "").strip():
                    empty_raw_calls += 1
                elapsed = call.get("elapsed_seconds")
                if isinstance(elapsed, (int, float)):
                    elapsed_seconds.append(float(elapsed))
                total_calls += 1

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "empty_raw_calls": empty_raw_calls,
        "total_calls": total_calls,
        "avg_seconds_per_call": statistics.mean(elapsed_seconds) if elapsed_seconds else None,
        "median_seconds_per_call": statistics.median(elapsed_seconds) if elapsed_seconds else None,
        "total_model_seconds": sum(elapsed_seconds) if elapsed_seconds else None,
        "winner_counts": winners,
    }


def build_row(run_dir: Path, pricing: dict[str, dict[str, float]]) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    evaluation = summary["evaluation"]
    config = summary["config"]
    model_id = config["model"]
    usage = collect_usage(run_dir)
    model_pricing = pricing.get(model_id, {"input": 0.0, "output": 0.0})
    estimated_cost = (
        usage["prompt_tokens"] / 1_000_000 * model_pricing["input"]
        + usage["completion_tokens"] / 1_000_000 * model_pricing["output"]
    )
    return {
        "run_dir": str(run_dir),
        "model": model_id,
        "model_label": config.get("model_label") or model_id,
        "prompt_strength": config.get("runtime", {}).get("prompt_strength"),
        "max_tokens": config.get("runtime", {}).get("max_tokens"),
        "reask_max_tokens": config.get("reask", {}).get("max_tokens"),
        "rank_spearman_rho": evaluation.get("rank_spearman_rho"),
        "pairwise_accuracy": evaluation.get("pairwise_accuracy"),
        "decisive_pairwise_accuracy": evaluation.get("decisive_pairwise_accuracy"),
        "topk_overlap": evaluation.get("topk_overlap") or {},
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "estimated_cost_usd": estimated_cost,
        "avg_seconds_per_call": usage["avg_seconds_per_call"],
        "median_seconds_per_call": usage["median_seconds_per_call"],
        "total_model_seconds": usage["total_model_seconds"],
        "empty_raw_calls": usage["empty_raw_calls"],
        "total_calls": usage["total_calls"],
        "winner_counts": usage["winner_counts"],
    }


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple:
        return (
            -(row.get("decisive_pairwise_accuracy") or -1.0),
            -(row.get("pairwise_accuracy") or -1.0),
            -(row.get("rank_spearman_rho") or -1.0),
            row.get("estimated_cost_usd") or math.inf,
            row.get("avg_seconds_per_call") or math.inf,
            row.get("model_label") or row.get("model") or "",
        )

    return sorted(rows, key=key)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Pairwise Model Sweep",
        "",
        f"Generated: {now_utc()}",
        "",
        "| Model | Rho | Pair Acc | Decisive Acc | Cost (USD) | Avg Sec/Call | Empty Calls | Winners (A/B/Tie) |",
        "|------|-----:|---------:|-------------:|-----------:|-------------:|------------:|-------------------|",
    ]
    for row in rows:
        winners = row["winner_counts"]
        lines.append(
            "| "
            f"{row['model_label']} | "
            f"{(row['rank_spearman_rho'] if row['rank_spearman_rho'] is not None else float('nan')):.4f} | "
            f"{(row['pairwise_accuracy'] if row['pairwise_accuracy'] is not None else float('nan')):.4f} | "
            f"{(row['decisive_pairwise_accuracy'] if row['decisive_pairwise_accuracy'] is not None else float('nan')):.4f} | "
            f"{row['estimated_cost_usd']:.4f} | "
            f"{(row['avg_seconds_per_call'] if row['avg_seconds_per_call'] is not None else float('nan')):.2f} | "
            f"{row['empty_raw_calls']} | "
            f"{winners['A']}/{winners['B']}/{winners['Tie']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dirs = discover_runs(args)
    if not run_dirs:
        raise SystemExit("No run directories found.")
    pricing = fetch_live_pricing()
    rows = sort_rows([build_row(run_dir, pricing) for run_dir in run_dirs])
    payload = {"generated_at_utc": now_utc(), "rows": rows}

    if args.output_json:
        write_json(args.output_json, payload)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(rows), encoding="utf-8")

    print(render_markdown(rows))


if __name__ == "__main__":
    main()
