#!/usr/bin/env python3
"""Utility helpers for monitoring Gemma shard waves and managing endpoints."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
TOGETHER_CLI = ROOT / ".venv-coarse" / "bin" / "together"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def lifecycle_dry_run_enabled() -> bool:
    value = os.environ.get("GEMMA_LIFECYCLE_DRY_RUN", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_orchestrator_status(output_root: Path) -> dict[str, Any] | None:
    status_path = output_root / "orchestrator_status.json"
    if not status_path.exists():
        return None
    try:
        return load_json(status_path)
    except json.JSONDecodeError:
        return None


def is_orchestrator_complete(status: dict[str, Any] | None) -> bool:
    if not status:
        return False
    if int(status.get("running") or 0) != 0:
        return False
    workers = status.get("workers") or []
    return all(str(worker.get("state")) != "running" for worker in workers)


def count_paper_results(output_root: Path) -> int:
    if not output_root.exists():
        return 0
    return sum(1 for _ in output_root.glob("**/papers/*/paper_result.json"))


def count_worker_results(worker_output_dir: Path) -> int:
    if not worker_output_dir.exists():
        return 0
    return sum(1 for _ in worker_output_dir.glob("papers/*/paper_result.json"))


def read_endpoint_names(endpoint_file: Path) -> list[str]:
    names: list[str] = []
    for line in endpoint_file.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            names.append(cleaned)
    return list(dict.fromkeys(names))


def _dry_run_action_log_path() -> Path | None:
    value = os.environ.get("GEMMA_LIFECYCLE_ACTION_LOG", "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def log_lifecycle_action(action: str, payload: dict[str, Any]) -> None:
    path = _dry_run_action_log_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "logged_at_utc": now_utc(),
        "action": action,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _together_env(key_file: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["TOGETHER_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
    return env


def run_together(key_file: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(TOGETHER_CLI), *args],
        cwd=str(ROOT),
        env=_together_env(key_file),
        check=True,
        capture_output=True,
        text=True,
    )


def list_dedicated_endpoints(key_file: Path) -> list[dict[str, Any]]:
    result = run_together(key_file, ["endpoints", "list", "--mine", "--type", "dedicated", "--json"])
    return json.loads(result.stdout)


def resolve_endpoint_records(key_file: Path, endpoint_names: list[str]) -> list[dict[str, Any]]:
    wanted = set(endpoint_names)
    endpoints = list_dedicated_endpoints(key_file)
    records = [row for row in endpoints if str(row.get("name")) in wanted]
    found = {str(row.get("name")) for row in records}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Could not resolve Together endpoints: {missing}")
    return records


def start_endpoint_names(key_file: Path, endpoint_names: list[str]) -> list[dict[str, Any]]:
    if lifecycle_dry_run_enabled():
        records = [
            {
                "id": f"dry-run::{name}",
                "name": name,
                "state": "STARTED",
                "dry_run": True,
            }
            for name in endpoint_names
        ]
        log_lifecycle_action("start_endpoint_names", {"endpoint_names": endpoint_names, "records": records})
        return records
    records = resolve_endpoint_records(key_file, endpoint_names)
    started: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("state")) != "STARTED":
            run_together(key_file, ["endpoints", "start", str(record["id"]), "--wait"])
            refreshed = resolve_endpoint_records(key_file, [str(record["name"])])[0]
            started.append(refreshed)
        else:
            started.append(record)
    return started


def stop_endpoint_names(key_file: Path, endpoint_names: list[str]) -> list[dict[str, Any]]:
    if lifecycle_dry_run_enabled():
        records = [
            {
                "id": f"dry-run::{name}",
                "name": name,
                "state": "STOPPED",
                "dry_run": True,
            }
            for name in endpoint_names
        ]
        log_lifecycle_action("stop_endpoint_names", {"endpoint_names": endpoint_names, "records": records})
        return records
    records = resolve_endpoint_records(key_file, endpoint_names)
    stopped: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("state")) != "STOPPED":
            run_together(key_file, ["endpoints", "stop", str(record["id"]), "--wait"])
            refreshed = resolve_endpoint_records(key_file, [str(record["name"])])[0]
            stopped.append(refreshed)
        else:
            stopped.append(record)
    return stopped


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_pid(pid: int, *, grace_seconds: float = 10.0) -> str:
    if lifecycle_dry_run_enabled():
        result = "dry_run"
        log_lifecycle_action("terminate_pid", {"pid": pid, "result": result, "grace_seconds": grace_seconds})
        return result
    if pid <= 0:
        return "invalid_pid"
    if not pid_exists(pid):
        return "not_running"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "not_running"
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not pid_exists(pid):
            return "terminated"
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    return "killed"
