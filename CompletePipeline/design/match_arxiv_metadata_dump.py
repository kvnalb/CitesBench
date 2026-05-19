#!/usr/bin/env python3
"""
Match local ICLR papers against a local arXiv metadata dump.

Supported dump formats:
- Parquet shards, such as the Hugging Face mirror of the arXiv metadata snapshot
- JSONL or JSONL.GZ files in the Kaggle-style arXiv metadata schema

This first-pass matcher is intentionally conservative: it only generates
automatic matches for exact normalized title matches found in the local dump.
That gives a high-precision identifier layer which can later be supplemented
with slower API-based matching for the unmatched remainder if needed.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from fetch_arxiv_metadata import load_input_rows, normalize_title, title_similarity, token_jaccard, write_json

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_INPUT_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
DEFAULT_RAW_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "arXiv"
DEFAULT_DUMP_PATH = DEFAULT_RAW_DIR / "dump" / "hf_snapshot"
EXCLUDED_DUMP_FILENAMES = {"download_manifest.json", "dataset_infos.json"}

PARQUET_COLUMNS = [
    "id",
    "title",
    "authors",
    "authors_parsed",
    "categories",
    "comments",
    "journal-ref",
    "doi",
    "license",
    "abstract",
    "versions",
    "update_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match local ICLR papers against a local arXiv metadata dump."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input paper-level CSV. Defaults to the year-specific RDD sample.",
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=DEFAULT_DUMP_PATH,
        help="Path to a local arXiv dump directory or file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where dump-match outputs are written.",
    )
    parser.add_argument(
        "--paper-id-col",
        default="paper_id",
        help="Paper identifier column in the input CSV.",
    )
    parser.add_argument(
        "--title-col",
        default="title",
        help="Title column in the input CSV.",
    )
    parser.add_argument(
        "--year-col",
        default="year",
        help="Year column in the input CSV.",
    )
    parser.add_argument(
        "--source-id-col",
        default="source_id",
        help="Optional source-id column in the input CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of input rows to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Row batch size when scanning Parquet shards.",
    )
    parser.add_argument(
        "--progress-every-files",
        type=int,
        default=10,
        help="Print a progress line every N dump files scanned.",
    )
    return parser.parse_args()


def build_title_index(input_rows: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    title_index: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in input_rows.itertuples(index=False):
        title_index[normalize_title(row.title)].append(
            {
                "paper_id": row.paper_id,
                "source_id": row.source_id,
                "input_year": int(row.year) if pd.notna(row.year) else None,
                "input_title": row.title,
            }
        )
    return title_index


def dump_file_type(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return "parquet"
    if suffixes.endswith(".jsonl") or suffixes.endswith(".json"):
        return "jsonl"
    if suffixes.endswith(".jsonl.gz") or suffixes.endswith(".json.gz"):
        return "jsonl_gz"
    raise ValueError(f"Unsupported dump file type for {path}")


def list_dump_files(dump_path: Path) -> list[Path]:
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump path does not exist: {dump_path}")
    if dump_path.is_file():
        return [dump_path]

    files: list[Path] = []
    patterns = ["*.parquet", "*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz"]
    for pattern in patterns:
        files.extend(dump_path.rglob(pattern))
    files = sorted(
        path
        for path in files
        if path.is_file()
        and path.name not in EXCLUDED_DUMP_FILENAMES
        and ".cache" not in path.parts
    )
    if not files:
        raise FileNotFoundError(f"No supported dump files found under {dump_path}")
    return files


def iter_parquet_rows(path: Path, batch_size: int) -> Iterable[dict[str, object]]:
    parquet_file = pq.ParquetFile(path)
    available_cols = [column for column in PARQUET_COLUMNS if column in parquet_file.schema.names]
    for batch in parquet_file.iter_batches(columns=available_cols, batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


def iter_jsonl_rows(path: Path) -> Iterable[dict[str, object]]:
    opener = gzip.open if "".join(path.suffixes).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def parse_version_year(versions: object, update_date: object) -> int | None:
    if isinstance(versions, list) and versions:
        first = versions[0]
        created = first.get("created") if isinstance(first, dict) else None
        if created:
            try:
                return parsedate_to_datetime(str(created)).year
            except (TypeError, ValueError, OverflowError):
                pass
    if update_date is not None:
        text = str(update_date)
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
    return None


def normalize_dump_row(raw_row: dict[str, object], dump_file: Path) -> dict[str, object]:
    arxiv_id = str(raw_row.get("id") or "").strip()
    title = str(raw_row.get("title") or "").strip()
    authors = str(raw_row.get("authors") or "").strip() or None
    authors_parsed = raw_row.get("authors_parsed")
    categories = str(raw_row.get("categories") or "").strip() or None
    comments = str(raw_row.get("comments") or "").strip() or None
    journal_ref = str(raw_row.get("journal-ref") or "").strip() or None
    doi = str(raw_row.get("doi") or "").strip() or None
    license_value = str(raw_row.get("license") or "").strip() or None
    abstract = str(raw_row.get("abstract") or "").strip() or None
    update_date = str(raw_row.get("update_date") or "").strip() or None
    versions = raw_row.get("versions")
    version_year = parse_version_year(versions=versions, update_date=update_date)
    return {
        "arxiv_id": arxiv_id or None,
        "arxiv_title": title,
        "arxiv_authors": authors,
        "arxiv_authors_parsed_json": json.dumps(authors_parsed, ensure_ascii=False) if authors_parsed is not None else None,
        "arxiv_categories": categories,
        "arxiv_comments": comments,
        "arxiv_journal_ref": journal_ref,
        "arxiv_doi": doi,
        "arxiv_license": license_value,
        "arxiv_abstract": abstract,
        "arxiv_update_date": update_date,
        "arxiv_first_version_year": version_year,
        "arxiv_abs_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "arxiv_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
        "dump_file": str(dump_file),
    }


def score_exact_match(input_title_norm: str, candidate_title_norm: str, input_year: int | None, candidate_year: int | None) -> dict[str, object]:
    similarity = title_similarity(input_title_norm, candidate_title_norm)
    jaccard = token_jaccard(input_title_norm, candidate_title_norm)
    exact = input_title_norm == candidate_title_norm and input_title_norm != ""
    year_diff = None if input_year is None or candidate_year is None else abs(input_year - candidate_year)
    penalty = 0.0 if year_diff is None else max(0, year_diff - 1) * 0.03
    score = (1.0 if exact else 0.0) + similarity + (0.2 * jaccard) - penalty
    return {
        "title_similarity": similarity,
        "token_jaccard": jaccard,
        "exact_title_match": exact,
        "near_exact_title_match": similarity >= 0.96 and jaccard >= 0.90,
        "published_year": candidate_year,
        "year_diff": year_diff,
        "candidate_score": score,
    }


def best_row_sort_key(row: dict[str, object]) -> tuple[float, float, str]:
    year_diff = row.get("year_diff")
    year_rank = float("inf") if year_diff is None or pd.isna(year_diff) else float(year_diff)
    score = float(row.get("candidate_score") or 0.0)
    arxiv_id = str(row.get("arxiv_id") or "")
    return (score, -year_rank, arxiv_id)


def scan_dump(
    dump_files: list[Path],
    title_index: dict[str, list[dict[str, object]]],
    batch_size: int,
    progress_every_files: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate_rows: list[dict[str, object]] = []
    scanned_rows = 0
    matched_rows = 0

    for file_number, dump_file in enumerate(dump_files, start=1):
        file_type = dump_file_type(dump_file)
        row_iter = (
            iter_parquet_rows(dump_file, batch_size=batch_size)
            if file_type == "parquet"
            else iter_jsonl_rows(dump_file)
        )
        for raw_row in row_iter:
            scanned_rows += 1
            normalized = normalize_dump_row(raw_row, dump_file=dump_file)
            title_norm = normalize_title(normalized["arxiv_title"])
            input_matches = title_index.get(title_norm)
            if not input_matches:
                continue

            matched_rows += 1
            for input_row in input_matches:
                metrics = score_exact_match(
                    input_title_norm=title_norm,
                    candidate_title_norm=title_norm,
                    input_year=input_row["input_year"],
                    candidate_year=normalized["arxiv_first_version_year"],
                )
                candidate_rows.append(
                    {
                        "paper_id": input_row["paper_id"],
                        "source_id": input_row["source_id"],
                        "input_year": input_row["input_year"],
                        "input_title": input_row["input_title"],
                        "input_title_norm": title_norm,
                        "match_source": "dump_exact_title",
                        **normalized,
                        **metrics,
                    }
                )

        if progress_every_files > 0 and (
            file_number == 1 or file_number % progress_every_files == 0 or file_number == len(dump_files)
        ):
            print(
                (
                    f"[dump-scan] files={file_number}/{len(dump_files)} "
                    f"scanned_rows={scanned_rows} candidate_rows={len(candidate_rows)}"
                ),
                flush=True,
            )

    scan_summary = {
        "n_dump_files": len(dump_files),
        "n_scanned_rows": scanned_rows,
        "n_title_hits_in_dump": matched_rows,
        "n_candidate_rows": len(candidate_rows),
    }
    return candidate_rows, scan_summary


def build_best_matches(
    input_rows: pd.DataFrame,
    candidate_rows: list[dict[str, object]],
) -> pd.DataFrame:
    if candidate_rows:
        candidate_df = pd.DataFrame(candidate_rows)
        best_candidate_df = (
            candidate_df.sort_values(
                by=["candidate_score", "year_diff", "arxiv_id"],
                ascending=[False, True, True],
                na_position="last",
            )
            .groupby("paper_id", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
    else:
        candidate_df = pd.DataFrame()
        best_candidate_df = pd.DataFrame(columns=["paper_id"])

    base_rows = input_rows.rename(columns={"title": "input_title", "year": "input_year"}).copy()
    base_rows["input_title_norm"] = base_rows["input_title"].map(normalize_title)
    best_df = base_rows.merge(best_candidate_df, on="paper_id", how="left", suffixes=("", "_matched"))

    best_df["matched"] = best_df["arxiv_id"].notna()
    best_df["match_status"] = best_df["matched"].map({True: "exact_title_match", False: "no_match_in_dump"})
    best_df["best_query_attempt"] = "local_dump_exact_title"
    best_df["attempt_status"] = "local_dump_scanned"
    best_df["total_entries_seen"] = best_df["matched"].astype(int)

    return best_df, candidate_df


def write_outputs(
    args: argparse.Namespace,
    input_rows: pd.DataFrame,
    dump_files: list[Path],
    best_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    scan_summary: dict[str, object],
) -> None:
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.raw_dir / "arxiv_dump_best_matches.csv"
    candidate_path = args.raw_dir / "arxiv_dump_candidate_matches.csv"
    enriched_path = args.raw_dir / f"{args.input_csv.stem}_with_arxiv_dump_match.csv"
    summary_path = args.raw_dir / "arxiv_dump_match_summary.json"

    best_df.to_csv(best_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)

    enriched_df = input_rows.merge(
        best_df.drop(columns=["input_title", "input_year", "source_id"], errors="ignore"),
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    enriched_df.to_csv(enriched_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv),
        "dump_path": str(args.dump_path),
        "raw_dir": str(args.raw_dir),
        "n_input_rows": int(len(input_rows)),
        "n_dump_files": int(len(dump_files)),
        "matched_input_rows": int(best_df["matched"].sum()) if not best_df.empty else 0,
        "unmatched_input_rows": int((~best_df["matched"]).sum()) if not best_df.empty else int(len(input_rows)),
        **scan_summary,
    }
    write_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    input_rows = load_input_rows(
        input_csv=args.input_csv,
        paper_id_col=args.paper_id_col,
        title_col=args.title_col,
        year_col=args.year_col,
        source_id_col=args.source_id_col,
        limit=args.limit,
    )
    dump_files = list_dump_files(args.dump_path)
    title_index = build_title_index(input_rows)
    candidate_rows, scan_summary = scan_dump(
        dump_files=dump_files,
        title_index=title_index,
        batch_size=args.batch_size,
        progress_every_files=args.progress_every_files,
    )
    best_df, candidate_df = build_best_matches(input_rows=input_rows, candidate_rows=candidate_rows)
    write_outputs(
        args=args,
        input_rows=input_rows,
        dump_files=dump_files,
        best_df=best_df,
        candidate_df=candidate_df,
        scan_summary=scan_summary,
    )


if __name__ == "__main__":
    main()
