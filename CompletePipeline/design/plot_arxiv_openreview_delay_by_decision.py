#!/usr/bin/env python3
"""
Plot the delay between a chosen OpenReview or conference anchor date and first
arXiv posting by decision status.

The delay is defined as:

    arXiv first posted date - anchor date

so positive values indicate papers first appearing on arXiv after the chosen
anchor date, while negative values indicate papers already on arXiv beforehand.
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

ANCHOR_CONFIGS = {
    "submission": {
        "label": "OpenReview submission",
        "slug": "openreview_submission",
        "column": "openreview_submitted_at",
        "title": "Delay Between OpenReview Submission and First arXiv Posting by Decision",
        "xlabel": "Delay in days: arXiv first post minus OpenReview submission",
        "share_after_label": "share_posted_after_submission",
        "share_before_label": "share_posted_before_submission",
        "before_text": "before OR submission",
        "after_text": "after OR submission",
    },
    "decision": {
        "label": "OpenReview decision",
        "slug": "openreview_decision",
        "column": "openreview_pdate_at",
        "title": "Delay Between OpenReview Decision Date and First arXiv Posting by Decision",
        "xlabel": "Delay in days: arXiv first post minus OpenReview decision date",
        "share_after_label": "share_posted_after_decision",
        "share_before_label": "share_posted_before_decision",
        "before_text": "before OR decision",
        "after_text": "after OR decision",
    },
    "conference": {
        "label": "conference start",
        "slug": "conference_start",
        "column": "conference_start_at",
        "title": "Delay Between ICLR Conference Start and First arXiv Posting by Decision",
        "xlabel": "Delay in days: arXiv first post minus conference start date",
        "share_after_label": "share_posted_after_conference",
        "share_before_label": "share_posted_before_conference",
        "before_text": "before conf.",
        "after_text": "after conf.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=DEFAULT_SAMPLE_CSV,
        help="Paper-level sample with decision labels.",
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
        default=45,
        help="Number of histogram bins inside the plotted range.",
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
    parser.add_argument(
        "--anchor-date",
        choices=sorted(ANCHOR_CONFIGS),
        default="submission",
        help="Which anchor date to compare against the first arXiv posting date.",
    )
    return parser.parse_args()


def load_delay_data(sample_csv: Path, timing_csv: Path, anchor_date: str) -> pd.DataFrame:
    anchor_cfg = ANCHOR_CONFIGS[anchor_date]
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
    merged = sample.merge(timing, on="paper_id", how="inner")
    merged["arxiv_first_posted_at"] = pd.to_datetime(merged["arxiv_first_posted_at"], utc=True, errors="coerce")
    merged["openreview_submitted_at"] = pd.to_datetime(merged["openreview_submitted_at"], utc=True, errors="coerce")
    merged["openreview_pdate_at"] = merged["openreview_pdate_at"].map(
        lambda x: pd.to_datetime(str(x).strip(), utc=True, errors="coerce") if pd.notna(x) else pd.NaT
    )
    merged["conference_start_at"] = merged["year"].map(CONFERENCE_START_DATES)
    merged["conference_start_at"] = pd.to_datetime(merged["conference_start_at"], utc=True, errors="coerce")
    anchor_col = anchor_cfg["column"]
    merged["anchor_at"] = merged[anchor_col]
    merged = merged.loc[
        merged["accepted"].isin([0.0, 1.0])
        & merged["arxiv_first_posted_at"].notna()
        & merged["anchor_at"].notna()
    ].copy()
    merged["delay_days"] = (
        merged["arxiv_first_posted_at"] - merged["anchor_at"]
    ).dt.total_seconds() / 86400.0
    merged["decision_label"] = np.where(merged["accepted"] == 1.0, "Accepted", "Rejected")
    merged["anchor_date_type"] = anchor_date
    return merged


def build_summary(df: pd.DataFrame, anchor_date: str) -> pd.DataFrame:
    anchor_cfg = ANCHOR_CONFIGS[anchor_date]
    summary = (
        df.groupby(["decision_label", "accepted"], as_index=False)
        .agg(
            n_papers=("paper_id", "size"),
            mean_delay_days=("delay_days", "mean"),
            median_delay_days=("delay_days", "median"),
            p10_delay_days=("delay_days", lambda s: s.quantile(0.10)),
            p25_delay_days=("delay_days", lambda s: s.quantile(0.25)),
            p75_delay_days=("delay_days", lambda s: s.quantile(0.75)),
            p90_delay_days=("delay_days", lambda s: s.quantile(0.90)),
            min_delay_days=("delay_days", "min"),
            max_delay_days=("delay_days", "max"),
            share_posted_after_anchor=("delay_days", lambda s: 100.0 * (s > 0).mean()),
            share_posted_before_anchor=("delay_days", lambda s: 100.0 * (s < 0).mean()),
        )
        .sort_values("accepted", ascending=False)
        .reset_index(drop=True)
    )
    summary["anchor_date_type"] = anchor_date
    summary = summary.rename(
        columns={
            "share_posted_after_anchor": anchor_cfg["share_after_label"],
            "share_posted_before_anchor": anchor_cfg["share_before_label"],
        }
    )
    return summary


def format_stats_text(
    row: pd.Series,
    left_out: int,
    right_out: int,
    anchor_date: str,
) -> str:
    anchor_cfg = ANCHOR_CONFIGS[anchor_date]
    return (
        f"n={int(row['n_papers'])}\n"
        f"mean={row['mean_delay_days']:.1f}d\n"
        f"median={row['median_delay_days']:.1f}d\n"
        f"{anchor_cfg['before_text']}: {row[anchor_cfg['share_before_label']]:.1f}%\n"
        f"{anchor_cfg['after_text']}: {row[anchor_cfg['share_after_label']]:.1f}%\n"
        f"outside view: {left_out + right_out}"
    )


def plot_histograms(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    bins: int,
    lower_q: float,
    upper_q: float,
    anchor_date: str,
) -> None:
    anchor_cfg = ANCHOR_CONFIGS[anchor_date]
    x_min = float(df["delay_days"].quantile(lower_q))
    x_max = float(df["delay_days"].quantile(upper_q))
    edges = np.linspace(x_min, x_max, bins + 1)

    colors = {"Accepted": "#4C72B0", "Rejected": "#C44E52"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, sharey=True)

    for ax, label in zip(axes, ["Accepted", "Rejected"]):
        subset = df.loc[df["decision_label"] == label].copy()
        shown = subset.loc[subset["delay_days"].between(x_min, x_max, inclusive="both")].copy()
        left_out = int((subset["delay_days"] < x_min).sum())
        right_out = int((subset["delay_days"] > x_max).sum())
        row = summary.loc[summary["decision_label"] == label].iloc[0]

        ax.hist(
            shown["delay_days"],
            bins=edges,
            color=colors[label],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(0, color="#2F2F2F", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.axvline(row["median_delay_days"], color="#2F2F2F", linewidth=1.4, alpha=0.85)
        ax.axvline(row["mean_delay_days"], color="#2F2F2F", linestyle=":", linewidth=1.6, alpha=0.85)
        ax.set_title(label)
        ax.grid(alpha=0.18)
        ax.text(
            0.03,
            0.97,
            format_stats_text(row, left_out=left_out, right_out=right_out, anchor_date=anchor_date),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
        )
        ax.text(
            0.97,
            0.97,
            f"left tail={left_out}\nright tail={right_out}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
        )

    fig.suptitle(anchor_cfg["title"])
    fig.supxlabel(anchor_cfg["xlabel"])
    fig.supylabel("Paper count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor_cfg = ANCHOR_CONFIGS[args.anchor_date]
    delay_df = load_delay_data(
        sample_csv=args.sample_csv,
        timing_csv=args.timing_csv,
        anchor_date=args.anchor_date,
    )
    summary_df = build_summary(delay_df, anchor_date=args.anchor_date)

    paper_level_path = args.output_dir / f"{anchor_cfg['slug']}_delay_by_decision_paper_level.csv"
    summary_path = args.output_dir / f"{anchor_cfg['slug']}_delay_by_decision_summary.csv"
    fig_path = args.output_dir / f"fig_{anchor_cfg['slug']}_delay_by_decision_hist.png"

    delay_df.to_csv(paper_level_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_histograms(
        delay_df,
        summary_df,
        output_path=fig_path,
        bins=args.bins,
        lower_q=args.clip_lower_quantile,
        upper_q=args.clip_upper_quantile,
        anchor_date=args.anchor_date,
    )

    print(f"Wrote {paper_level_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
