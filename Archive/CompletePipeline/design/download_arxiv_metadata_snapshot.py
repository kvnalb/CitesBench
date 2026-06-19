#!/usr/bin/env python3
"""
Download a local copy of the arXiv metadata snapshot.

This script uses the Hugging Face mirror of the arXiv metadata snapshot because
it is directly downloadable without a Kaggle login in the current environment.
The matcher script supports both Parquet mirror shards and Kaggle-style JSONL
files, so the downloaded mirror can be swapped out later without changing the
matching workflow.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "arXiv" / "dump" / "hf_snapshot"
DEFAULT_REPO_ID = "librarian-bots/arxiv-metadata-snapshot"
DEFAULT_ALLOW_PATTERNS = ["*.parquet", "README.md", "dataset_infos.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the mirrored arXiv metadata snapshot for local matching."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Dataset repo id to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory for the downloaded snapshot.",
    )
    parser.add_argument(
        "--include-readme-only",
        action="store_true",
        help="Download only the dataset README and metadata manifests.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of files even if they already exist locally.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent download workers for snapshot_download.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = ["README.md", "dataset_infos.json"] if args.include_readme_only else DEFAULT_ALLOW_PATTERNS
    repo_files = list_repo_files(repo_id=args.repo_id, repo_type="dataset")

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        allow_patterns=allow_patterns,
        force_download=args.force_download,
        max_workers=args.max_workers,
    )

    parquet_files = sorted(str(path.relative_to(args.output_dir)) for path in args.output_dir.rglob("*.parquet"))
    total_bytes = sum(path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file())
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "snapshot_path": snapshot_path,
        "output_dir": str(args.output_dir),
        "allow_patterns": allow_patterns,
        "repo_file_count": len(repo_files),
        "downloaded_parquet_file_count": len(parquet_files),
        "downloaded_parquet_files": parquet_files[:200],
        "total_downloaded_bytes": total_bytes,
    }
    write_json(args.output_dir / "download_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
