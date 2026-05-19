#!/usr/bin/env python3
"""
Summarize whether matched arXiv preprints were posted before or after the
OpenReview submission timestamp.

This script uses the OpenReview-enriched arXiv match file for the local ICLR
RDD sample. It recovers the first arXiv posting timestamp directly from the
local metadata dump's `versions` field, compares it to OpenReview's
`openreview_tcdate_at` submission timestamp, and writes year-by-year summary
tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)

DEFAULT_INPUT_CSV = (
    ROOT
    / "rawdata"
    / "Design"
    / "OpenReview"
    / "arxiv_dump_combined_best_matches_with_openreview_yearly_submissions.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "Design" / "iclr_local_rdd"

MODERN_ARXIV_ID_RE = re.compile(r"^(\d{4})\.(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="OpenReview-enriched arXiv match file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are written.",
    )
    return parser.parse_args()


def parse_first_version_timestamp(versions: object, update_date: object) -> pd.Timestamp | pd.NaT:
    if isinstance(versions, (list, tuple)) and len(versions) > 0:
        first = versions[0]
        if isinstance(first, dict):
            created = first.get("created")
            if created:
                ts = pd.to_datetime(created, utc=True, errors="coerce")
                if pd.notna(ts):
                    return ts

    if update_date is not None and str(update_date).strip():
        return pd.to_datetime(update_date, utc=True, errors="coerce")

    return pd.NaT


def normalize_arxiv_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    match = MODERN_ARXIV_ID_RE.match(text)
    if match:
        prefix, suffix = match.groups()
        return f"{prefix}.{suffix.ljust(5, '0')}"
    return text


def parse_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NaT
    return pd.to_datetime(text, utc=True, errors="coerce")


def load_first_arxiv_posted_at(df: pd.DataFrame) -> pd.DataFrame:
    matched = df.loc[df["matched"].fillna(False)].copy()
    matched = matched.loc[matched["dump_file"].notna() & matched["arxiv_id"].notna()].copy()
    matched["arxiv_id_norm"] = matched["arxiv_id"].map(normalize_arxiv_id)

    file_to_ids: dict[str, set[str]] = {}
    for dump_file, arxiv_id in matched[["dump_file", "arxiv_id_norm"]].itertuples(index=False):
        file_to_ids.setdefault(str(dump_file), set()).add(str(arxiv_id))

    rows: list[dict[str, object]] = []
    for dump_file, ids in sorted(file_to_ids.items()):
        dump_path = Path(dump_file)
        if not dump_path.exists():
            continue
        dump_df = pd.read_parquet(dump_path, columns=["id", "versions", "update_date"])
        dump_df["arxiv_id_norm"] = dump_df["id"].map(normalize_arxiv_id)
        subset = dump_df.loc[dump_df["arxiv_id_norm"].isin(ids)].copy()
        if subset.empty:
            continue
        subset["arxiv_first_posted_at"] = subset.apply(
            lambda row: parse_first_version_timestamp(row["versions"], row["update_date"]),
            axis=1,
        )
        rows.extend(
            subset[["arxiv_id_norm", "arxiv_first_posted_at"]]
            .rename(columns={"arxiv_id_norm": "arxiv_id_norm"})
            .to_dict(orient="records")
        )

    first_posted = pd.DataFrame(rows).drop_duplicates(subset=["arxiv_id_norm"])
    return first_posted


def build_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["openreview_submitted_at"] = working["openreview_tcdate_at"].map(parse_timestamp)
    working["arxiv_first_posted_at"] = working["arxiv_first_posted_at"].map(parse_timestamp)

    valid = working.loc[
        working["matched"].fillna(False)
        & working["openreview_submitted_at"].notna()
        & working["arxiv_first_posted_at"].notna()
    ].copy()

    valid["timing_exact"] = "same_moment"
    valid.loc[
        valid["arxiv_first_posted_at"] < valid["openreview_submitted_at"],
        "timing_exact",
    ] = "before_openreview"
    valid.loc[
        valid["arxiv_first_posted_at"] > valid["openreview_submitted_at"],
        "timing_exact",
    ] = "after_openreview"

    valid["arxiv_first_posted_date"] = valid["arxiv_first_posted_at"].dt.date
    valid["openreview_submitted_date"] = valid["openreview_submitted_at"].dt.date
    valid["timing_by_day"] = "same_day"
    valid.loc[
        valid["arxiv_first_posted_date"] < valid["openreview_submitted_date"],
        "timing_by_day",
    ] = "before_openreview"
    valid.loc[
        valid["arxiv_first_posted_date"] > valid["openreview_submitted_date"],
        "timing_by_day",
    ] = "after_openreview"

    yearly = (
        valid.groupby("input_year", as_index=False)
        .agg(
            n_matched_with_dates=("paper_id", "size"),
            n_before_exact=("timing_exact", lambda s: int((s == "before_openreview").sum())),
            n_after_exact=("timing_exact", lambda s: int((s == "after_openreview").sum())),
            n_same_moment=("timing_exact", lambda s: int((s == "same_moment").sum())),
            n_before_day=("timing_by_day", lambda s: int((s == "before_openreview").sum())),
            n_after_day=("timing_by_day", lambda s: int((s == "after_openreview").sum())),
            n_same_day=("timing_by_day", lambda s: int((s == "same_day").sum())),
        )
        .rename(columns={"input_year": "year"})
        .sort_values("year")
        .reset_index(drop=True)
    )

    yearly["pct_before_exact"] = 100.0 * yearly["n_before_exact"] / yearly["n_matched_with_dates"]
    yearly["pct_after_exact"] = 100.0 * yearly["n_after_exact"] / yearly["n_matched_with_dates"]
    yearly["pct_same_moment"] = 100.0 * yearly["n_same_moment"] / yearly["n_matched_with_dates"]
    yearly["pct_before_day"] = 100.0 * yearly["n_before_day"] / yearly["n_matched_with_dates"]
    yearly["pct_after_day"] = 100.0 * yearly["n_after_day"] / yearly["n_matched_with_dates"]
    yearly["pct_same_day"] = 100.0 * yearly["n_same_day"] / yearly["n_matched_with_dates"]

    return valid, yearly


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False, dtype={"paper_id": "string", "arxiv_id": "string"})
    df["arxiv_id_norm"] = df["arxiv_id"].map(normalize_arxiv_id)
    first_posted = load_first_arxiv_posted_at(df)
    merged = df.merge(first_posted, on="arxiv_id_norm", how="left")

    paper_level, yearly = build_summary(merged)

    paper_level_path = args.output_dir / "arxiv_vs_openreview_timing_paper_level.csv"
    yearly_path = args.output_dir / "arxiv_vs_openreview_timing_by_year.csv"

    paper_level.to_csv(paper_level_path, index=False)
    yearly.to_csv(yearly_path, index=False)

    print(f"Wrote {paper_level_path}")
    print(f"Wrote {yearly_path}")


if __name__ == "__main__":
    main()
