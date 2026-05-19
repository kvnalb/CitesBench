#!/usr/bin/env python3
"""
Plot a combined accepted/rejected histogram of arXiv posting delays relative to
OpenReview pdate, with milestone summaries.

The histogram x-axis is:

    arXiv first post date - OpenReview pdate

For accepted papers, OpenReview pdate is the acceptance date. For rejected
papers, pdate is only available in some years and should be interpreted as the
final decision date where present.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)

MPLCONFIGDIR = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SAMPLE_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
DEFAULT_TIMING_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "arxiv_vs_openreview_timing_paper_level.csv"
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "Design" / "iclr_local_rdd"

CONFERENCE_START_DATES = {
    2018: "2018-04-30",
    2019: "2019-05-06",
    2020: "2020-04-26",
    2021: "2021-05-03",
    2022: "2022-04-25",
    2023: "2023-05-01",
    2024: "2024-05-07",
    2025: "2025-04-24",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=DEFAULT_SAMPLE_CSV,
        help="Paper-level sample with accepted/rejected labels.",
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=DEFAULT_TIMING_CSV,
        help="Paper-level arXiv/OpenReview timing file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=44,
        help="Number of histogram bins in the plotted range.",
    )
    parser.add_argument(
        "--clip-lower-quantile",
        type=float,
        default=0.01,
        help="Lower quantile used to set the plotted x-range.",
    )
    parser.add_argument(
        "--clip-upper-quantile",
        type=float,
        default=0.99,
        help="Upper quantile used to set the plotted x-range.",
    )
    return parser.parse_args()


def parse_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NaT
    return pd.to_datetime(text, utc=True, errors="coerce")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    repair_mask = series.notna() & parsed.isna()
    if repair_mask.any():
        parsed.loc[repair_mask] = series.loc[repair_mask].map(parse_timestamp)
    return parsed


def load_data(sample_csv: Path, timing_csv: Path) -> pd.DataFrame:
    sample = pd.read_csv(
        sample_csv,
        low_memory=False,
        usecols=["paper_id", "year", "accepted", "decision"],
    )
    timing = pd.read_csv(
        timing_csv,
        low_memory=False,
        usecols=[
            "paper_id",
            "input_year",
            "arxiv_first_posted_at",
            "openreview_submitted_at",
            "openreview_pdate_at",
        ],
    )
    df = sample.merge(timing, on="paper_id", how="inner")
    df["year"] = df["year"].astype(int)
    df["arxiv_first_posted_at"] = parse_timestamp_series(df["arxiv_first_posted_at"])
    df["openreview_submitted_at"] = parse_timestamp_series(df["openreview_submitted_at"])
    df["openreview_pdate_at"] = parse_timestamp_series(df["openreview_pdate_at"])
    df["conference_start_at"] = parse_timestamp_series(df["year"].map(CONFERENCE_START_DATES))

    df = df.loc[
        df["accepted"].isin([0.0, 1.0])
        & df["arxiv_first_posted_at"].notna()
        & df["openreview_submitted_at"].notna()
        & df["conference_start_at"].notna()
    ].copy()

    df["decision_label"] = np.where(df["accepted"] == 1.0, "Accepted", "Rejected")
    df["submission_delay_days"] = (
        df["arxiv_first_posted_at"] - df["openreview_submitted_at"]
    ).dt.total_seconds() / 86400.0
    df["conference_delay_days"] = (
        df["arxiv_first_posted_at"] - df["conference_start_at"]
    ).dt.total_seconds() / 86400.0

    df = df.loc[df["openreview_pdate_at"].notna()].copy()
    df["pdate_delay_days"] = (
        df["arxiv_first_posted_at"] - df["openreview_pdate_at"]
    ).dt.total_seconds() / 86400.0
    df["conference_offset_from_pdate_days"] = (
        df["conference_start_at"] - df["openreview_pdate_at"]
    ).dt.total_seconds() / 86400.0
    return df


def pct_before(series: pd.Series) -> float:
    return 100.0 * (series < 0).mean()


def pct_after(series: pd.Series) -> float:
    return 100.0 * (series >= 0).mean()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["decision_label", "accepted"], as_index=False)
        .agg(
            n_papers=("paper_id", "size"),
            mean_pdate_delay_days=("pdate_delay_days", "mean"),
            median_pdate_delay_days=("pdate_delay_days", "median"),
            p10_pdate_delay_days=("pdate_delay_days", lambda s: s.quantile(0.10)),
            p90_pdate_delay_days=("pdate_delay_days", lambda s: s.quantile(0.90)),
            submission_before_pct=("submission_delay_days", pct_before),
            submission_after_pct=("submission_delay_days", pct_after),
            acceptance_before_pct=("pdate_delay_days", pct_before),
            acceptance_after_pct=("pdate_delay_days", pct_after),
            conference_before_pct=("conference_delay_days", pct_before),
            conference_after_pct=("conference_delay_days", pct_after),
            median_conference_offset_days=("conference_offset_from_pdate_days", "median"),
        )
        .sort_values("accepted", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def format_box(row: pd.Series) -> str:
    return (
        f"{row['decision_label']}\n"
        f"n={int(row['n_papers'])}\n"
        f"submission: {row['submission_before_pct']:.1f}% / {row['submission_after_pct']:.1f}%\n"
        f"accept/decision: {row['acceptance_before_pct']:.1f}% / {row['acceptance_after_pct']:.1f}%\n"
        f"conference: {row['conference_before_pct']:.1f}% / {row['conference_after_pct']:.1f}%"
    )


def plot_hist(df: pd.DataFrame, summary: pd.DataFrame, output_path: Path, bins: int, lower_q: float, upper_q: float) -> None:
    colors = {"Accepted": "#4C72B0", "Rejected": "#C44E52"}
    x_min = float(df["pdate_delay_days"].quantile(lower_q))
    x_max = float(df["pdate_delay_days"].quantile(upper_q))
    edges = np.linspace(x_min, x_max, bins + 1)

    accepted = df.loc[df["decision_label"] == "Accepted", "pdate_delay_days"]
    rejected = df.loc[df["decision_label"] == "Rejected", "pdate_delay_days"]
    conference_line = float(df["conference_offset_from_pdate_days"].median())

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.hist(
        accepted,
        bins=edges,
        alpha=0.55,
        color=colors["Accepted"],
        label="Accepted",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        rejected,
        bins=edges,
        alpha=0.55,
        color=colors["Rejected"],
        label="Rejected",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        accepted,
        bins=edges,
        histtype="step",
        color=colors["Accepted"],
        linewidth=1.4,
    )
    ax.hist(
        rejected,
        bins=edges,
        histtype="step",
        color=colors["Rejected"],
        linewidth=1.4,
    )

    ax.axvline(0, color="#2F2F2F", linestyle="--", linewidth=1.2, label="OR pdate")
    ax.axvline(conference_line, color="#2F6B3B", linewidth=1.6, label="Conference start")
    ax.grid(alpha=0.18)
    ax.set_xlim(x_min, x_max)
    ax.set_title("arXiv Posting Relative to OpenReview pdate by Decision")
    ax.set_xlabel("Delay in days: arXiv first post minus OpenReview pdate")
    ax.set_ylabel("Paper count")
    ax.legend(frameon=False, loc="upper center", ncol=4)

    accepted_row = summary.loc[summary["decision_label"] == "Accepted"].iloc[0]
    rejected_row = summary.loc[summary["decision_label"] == "Rejected"].iloc[0]

    ax.text(
        0.02,
        0.97,
        format_box(accepted_row),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#1F355F",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.0},
    )
    ax.text(
        0.98,
        0.97,
        format_box(rejected_row),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#7A2E33",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.0},
    )
    ax.text(
        0.5,
        0.03,
        f"Green line = median conference-start offset from pdate ({conference_line:.1f} days)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#2F6B3B",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(sample_csv=args.sample_csv, timing_csv=args.timing_csv)
    summary = build_summary(df)

    paper_level_path = args.output_dir / "arxiv_overlay_delay_pdate_by_decision_paper_level.csv"
    summary_path = args.output_dir / "arxiv_overlay_delay_pdate_by_decision_summary.csv"
    fig_path = args.output_dir / "fig_arxiv_overlay_delay_pdate_by_decision.png"

    df.to_csv(paper_level_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_hist(
        df=df,
        summary=summary,
        output_path=fig_path,
        bins=args.bins,
        lower_q=args.clip_lower_quantile,
        upper_q=args.clip_upper_quantile,
    )

    print(f"Wrote {paper_level_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
