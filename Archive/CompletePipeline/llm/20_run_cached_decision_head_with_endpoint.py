#!/usr/bin/env python3
"""Run a cached decision-head sweep with a managed Together endpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
DEFAULT_KEY_FILE = ROOT / "key.txt"
DEFAULT_RUNNER = Path(__file__).resolve().parent / "12_eval_cached_decision_heads.py"
DEFAULT_PYTHON = ROOT / ".venv-coarse" / "bin" / "python"
DEFAULT_ENDPOINT_NAME = "thedatainnovati_6e25/google/gemma-2-9b-it-e9d6e73e"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def count_completed_calls(output_dir: Path, endpoint_name: str) -> int:
    call_root = output_dir / "calls" / slugify(endpoint_name)
    if not call_root.exists():
        return 0
    return sum(1 for _ in call_root.glob("*.json"))


def slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python_executable.resolve()),
        str(args.runner_path.resolve()),
        "--key-file",
        str(args.key_file.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--models",
        args.endpoint_name,
        "--prompt-variant",
        args.prompt_variant,
        "--max-papers",
        str(args.max_papers),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-retries",
        str(args.max_retries),
        "--max-workers",
        str(args.max_workers),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached decision-head sweep with managed endpoint lifecycle.")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--runner-path", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "OutputNew" / "Empirics" / "decision_head_positive_bias_gemma2_9b_all_20260421",
    )
    parser.add_argument("--prompt-variant", choices=("plain", "positive_bias"), default="positive_bias")
    parser.add_argument("--max-papers", type=int, default=1_000_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stale-minutes", type=float, default=20.0)
    parser.add_argument("--no-start-endpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "managed_run_status.json"
    log_path = output_dir / "managed_run.log"
    command = build_command(args)
    endpoint_names = [args.endpoint_name]
    endpoint_start_result = None
    endpoint_stop_result = None
    proc: subprocess.Popen[str] | None = None
    started = time.time()
    last_call_count = count_completed_calls(output_dir, args.endpoint_name)
    last_progress_epoch = time.time()
    last_progress_utc = now_utc()
    shutdown_reason = None

    def write_status(state: str, returncode: int | None = None) -> None:
        write_json(
            status_path,
            {
                "updated_at_utc": now_utc(),
                "state": state,
                "returncode": returncode,
                "pid": proc.pid if proc else None,
                "elapsed_minutes": round((time.time() - started) / 60.0, 2),
                "endpoint_name": args.endpoint_name,
                "endpoint_start_result": endpoint_start_result,
                "endpoint_stop_result": endpoint_stop_result,
                "shutdown_reason": shutdown_reason,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "command": command,
                "completed_call_files": count_completed_calls(output_dir, args.endpoint_name),
                "last_progress_utc": last_progress_utc,
                "minutes_since_progress": round((time.time() - last_progress_epoch) / 60.0, 2),
            },
        )

    try:
        write_status("starting_endpoint")
        if not args.no_start_endpoint:
            endpoint_start_result = start_endpoint_names(args.key_file.resolve(), endpoint_names)

        write_status("running")
        with log_path.open("a", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                returncode = proc.poll()
                current_call_count = count_completed_calls(output_dir, args.endpoint_name)
                if current_call_count > last_call_count:
                    last_call_count = current_call_count
                    last_progress_epoch = time.time()
                    last_progress_utc = now_utc()

                if returncode is not None:
                    write_status("completed" if returncode == 0 else "failed", returncode)
                    if returncode != 0:
                        shutdown_reason = f"runner_returncode:{returncode}"
                        raise SystemExit(returncode)
                    break

                minutes_since_progress = (time.time() - last_progress_epoch) / 60.0
                if args.stale_minutes > 0 and minutes_since_progress >= args.stale_minutes:
                    shutdown_reason = f"stale_no_call_progress:{minutes_since_progress:.1f}m"
                    terminate_pid(proc.pid)
                    write_status("stale_stopped", proc.poll())
                    raise SystemExit(2)

                write_status("running")
                time.sleep(args.poll_seconds)
    finally:
        if proc is not None and proc.poll() is None:
            terminate_pid(proc.pid)
        endpoint_stop_result = stop_endpoint_names(args.key_file.resolve(), endpoint_names)
        write_status("stopped", proc.poll() if proc else None)


if __name__ == "__main__":
    main()
