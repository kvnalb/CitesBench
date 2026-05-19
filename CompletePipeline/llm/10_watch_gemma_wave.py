#!/usr/bin/env python3
"""Monitor a Gemma shard wave and stop endpoints on completion or stall."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from gemma_lifecycle_utils import (
    count_worker_results,
    load_json,
    now_utc,
    pid_exists,
    read_orchestrator_status,
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
DEFAULT_KEY_FILE = ROOT / "key.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a Gemma wave and stop endpoints on completion or stall.")
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stale-minutes", type=float, default=15.0)
    return parser.parse_args()


def build_initial_worker_state(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for shard in manifest.get("shards") or []:
        shard_name = str(shard["shard_name"])
        output_dir = Path(str(shard["output_dir"]))
        completed = count_worker_results(output_dir)
        state[shard_name] = {
            "endpoint_name": str(shard["endpoint_name"]),
            "assigned_papers": int(shard["n_papers"]),
            "output_dir": str(output_dir),
            "completed_papers": completed,
            "last_progress_utc": now_utc(),
            "last_progress_epoch": time.time(),
            "terminal_reason": None,
            "endpoint_stopped": False,
            "stop_result": None,
            "last_seen_pid": None,
            "pid_alive": None,
        }
    return state


def main() -> None:
    args = parse_args()
    selection_root = args.selection_root.resolve()
    output_root = args.output_root.resolve()
    status_path = output_root / "lifecycle_status.json"
    manifest = load_json(selection_root / "orchestrator_manifest.json")
    worker_state = build_initial_worker_state(manifest)
    stale_seconds = float(args.stale_minutes) * 60.0

    while True:
        orchestrator_status = read_orchestrator_status(output_root)
        worker_rows = {
            str(row["shard_index"]): row
            for row in (orchestrator_status.get("workers") or [])
        } if orchestrator_status else {}

        all_terminal = True
        status_rows: list[dict[str, Any]] = []
        for shard_name, state in worker_state.items():
            worker_row = worker_rows.get(shard_name) or {}
            output_dir = Path(str(state["output_dir"]))
            completed = count_worker_results(output_dir)
            if completed > int(state["completed_papers"]):
                state["completed_papers"] = completed
                state["last_progress_utc"] = now_utc()
                state["last_progress_epoch"] = time.time()

            pid = worker_row.get("pid")
            if isinstance(pid, int):
                state["last_seen_pid"] = pid
                state["pid_alive"] = pid_exists(pid)

            worker_state_name = str(worker_row.get("state") or "unknown")
            minutes_since_progress = round((time.time() - float(state["last_progress_epoch"])) / 60.0, 2)

            action: str | None = None
            if state["terminal_reason"] is None:
                if worker_state_name == "completed":
                    action = "completed_stop"
                elif worker_state_name == "failed":
                    action = "failed_stop"
                elif worker_state_name == "running" and minutes_since_progress >= float(args.stale_minutes):
                    action = "stale_stop"

            if action is not None:
                pid_stop_result = None
                if action == "stale_stop" and isinstance(state["last_seen_pid"], int):
                    pid_stop_result = terminate_pid(int(state["last_seen_pid"]))
                stop_result = stop_endpoint_names(args.key_file.resolve(), [str(state["endpoint_name"])])
                state["endpoint_stopped"] = True
                state["stop_result"] = {
                    "action": action,
                    "pid_stop_result": pid_stop_result,
                    "endpoint_records": stop_result,
                    "stopped_at_utc": now_utc(),
                }
                state["terminal_reason"] = action

            if state["terminal_reason"] is None:
                all_terminal = False

            status_rows.append(
                {
                    "shard_name": shard_name,
                    "endpoint_name": state["endpoint_name"],
                    "assigned_papers": state["assigned_papers"],
                    "completed_papers": state["completed_papers"],
                    "remaining_papers": max(0, int(state["assigned_papers"]) - int(state["completed_papers"])),
                    "worker_state": worker_state_name,
                    "last_seen_pid": state["last_seen_pid"],
                    "pid_alive": state["pid_alive"],
                    "last_progress_utc": state["last_progress_utc"],
                    "minutes_since_progress": minutes_since_progress,
                    "terminal_reason": state["terminal_reason"],
                    "endpoint_stopped": state["endpoint_stopped"],
                    "stop_result": state["stop_result"],
                }
            )

        lifecycle_status = {
            "updated_at_utc": now_utc(),
            "selection_root": str(selection_root),
            "output_root": str(output_root),
            "stale_minutes": float(args.stale_minutes),
            "orchestrator_status_path": str(output_root / "orchestrator_status.json"),
            "workers": status_rows,
        }
        write_json(status_path, lifecycle_status)

        if all_terminal:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
