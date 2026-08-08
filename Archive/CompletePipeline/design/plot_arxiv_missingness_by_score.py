#!/usr/bin/env python3
"""
Plot arXiv match missingness by paper-level score bins.

This joins the arXiv-queried ICLR analytic sample to the arXiv combined match
file and summarizes how the arXiv missing share varies with paper-level mean
reviewer rating. The main use case is to diagnose whether the missing papers
are concentrated in particular parts of the score distribution, while keeping
the year-specific RDD cutoffs and bandwidths visible for comparison.
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
DEFAULT_SCORE_SUPPORT_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "paper_level_all_years.csv"
DEFAULT_ARXIV_MATCH_CSV = ROOT / "OutputNew" / "rawdata" / "Design" / "arXiv" / "arxiv_dump_combined_best_matches.csv"
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "Design" / "iclr_local_rdd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=DEFAULT_SAMPLE_CSV,
        help="Paper-level CSV used for arXiv missingness summaries.",
    )
    parser.add_argument(
        "--score-support-csv",
        type=Path,
        default=DEFAULT_SCORE_SUPPORT_CSV,
        help="Paper-level CSV used only to set the full score range on yearly plots.",
    )
    parser.add_argument(
        "--arxiv-match-csv",
        type=Path,
        default=DEFAULT_ARXIV_MATCH_CSV,
        help="arXiv combined match CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.5,
        help="Width of the mean-rating bins.",
    )
    parser.add_argument(
        "--score-col",
        default="mean_rating",
        help="Score column to bin. Defaults to paper-level mean rating.",
    )
    return parser.parse_args()


def build_score_bins(score: pd.Series, bin_width: float) -> np.ndarray:
    score_min = float(score.min())
    score_max = float(score.max())
    lower = np.floor(score_min / bin_width) * bin_width
    upper = np.ceil(score_max / bin_width) * bin_width
    return np.arange(lower, upper + bin_width, bin_width)


def add_bins(df: pd.DataFrame, score_col: str, bin_width: float) -> pd.DataFrame:
    bins = build_score_bins(df[score_col], bin_width)
    out = df.copy()
    out["score_bin"] = pd.cut(
        out[score_col],
        bins=bins,
        include_lowest=True,
        right=False,
    )
    out["score_bin_left"] = out["score_bin"].map(lambda x: float(x.left) if pd.notna(x) else np.nan)
    out["score_bin_right"] = out["score_bin"].map(lambda x: float(x.right) if pd.notna(x) else np.nan)
    out["score_bin_mid"] = out[["score_bin_left", "score_bin_right"]].mean(axis=1)
    out["score_bin_label"] = out["score_bin"].map(
        lambda x: f"[{x.left:.1f}, {x.right:.1f})" if pd.notna(x) else None
    )
    return out


def summarize_missingness(df: pd.DataFrame, group_cols: list[str], score_col: str) -> pd.DataFrame:
    group_df = df.dropna(subset=group_cols).copy()
    summary = (
        group_df.groupby(group_cols, as_index=False, observed=True)
        .agg(
            n_papers=("paper_id", "size"),
            n_missing=("missing_arxiv", "sum"),
            n_matched=("matched_arxiv", "sum"),
            accept_rate=("accepted", "mean"),
            mean_score=(score_col, "mean"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    for col in ["score_bin_left", "score_bin_right", "score_bin_mid"]:
        if col in summary.columns:
            summary[col] = summary[col].astype(float)
    summary["score_bin_label"] = summary.apply(
        lambda row: f"[{row['score_bin_left']:.1f}, {row['score_bin_right']:.1f})",
        axis=1,
    )
    summary["missing_share"] = summary["n_missing"] / summary["n_papers"]
    summary["matched_share"] = summary["n_matched"] / summary["n_papers"]
    return summary


def get_year_design_params(sample_df: pd.DataFrame) -> pd.DataFrame:
    params = (
        sample_df.groupby("year", as_index=False)
        .agg(
            cutoff=("cutoff", "first"),
            bandwidth=("bandwidth", "first"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    params["cutoff"] = params["cutoff"].astype(float)
    params["bandwidth"] = params["bandwidth"].astype(float)
    params["bandwidth_left"] = params["cutoff"] - params["bandwidth"]
    params["bandwidth_right"] = params["cutoff"] + params["bandwidth"]
    return params


def get_axis_bounds(score_df: pd.DataFrame, score_col: str) -> tuple[float, float]:
    score_min = float(score_df[score_col].min())
    score_max = float(score_df[score_col].max())
    lower = float(np.floor(score_min))
    upper = float(np.ceil(score_max))
    return lower, upper


def plot_overall(summary: pd.DataFrame, score_col: str, output_path: Path) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )

    x = summary["score_bin_mid"].astype(float)
    x_min = float(summary["score_bin_left"].min())
    x_max = float(summary["score_bin_right"].max())

    ax_top.plot(x, 100 * summary["missing_share"], color="#C44E52", linewidth=2.0)
    ax_top.scatter(x, 100 * summary["missing_share"], color="#C44E52", s=55, zorder=3)
    for _, row in summary.iterrows():
        ax_top.text(
            float(row["score_bin_mid"]),
            100 * float(row["missing_share"]) + 1.4,
            f"n={int(row['n_papers'])}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4D4D4D",
        )
    ax_top.set_ylabel("% missing in arXiv")
    ax_top.set_ylim(0, max(100 * summary["missing_share"].max() * 1.20, 10))
    ax_top.set_title("arXiv Missing Share by Mean Rating Bin")
    ax_top.set_xlim(x_min, x_max)
    ax_top.grid(alpha=0.2)

    width = float((summary["score_bin_right"] - summary["score_bin_left"]).iloc[0]) * 0.42
    ax_bottom.bar(
        x - width / 2,
        summary["n_matched"],
        width=width,
        color="#4C72B0",
        alpha=0.85,
        label="Matched in arXiv",
    )
    ax_bottom.bar(
        x + width / 2,
        summary["n_missing"],
        width=width,
        color="#DD8452",
        alpha=0.85,
        label="Missing in arXiv",
    )
    ax_bottom.set_xlabel(score_col.replace("_", " "))
    ax_bottom.set_ylabel("Paper count")
    ax_bottom.set_xlim(x_min, x_max)
    ax_bottom.grid(alpha=0.2)
    ax_bottom.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_by_year(
    summary: pd.DataFrame,
    score_col: str,
    output_path: Path,
    year_design: pd.DataFrame,
    x_bounds: tuple[float, float],
) -> None:
    years = sorted(summary["year"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    design_map = year_design.set_index("year").to_dict("index")
    x_min, x_max = x_bounds

    for ax, year in zip(axes.flat, years):
        subset = summary.loc[summary["year"] == year].copy()
        x = subset["score_bin_mid"].astype(float)
        params = design_map[int(year)]

        ax.axvspan(
            params["bandwidth_left"],
            params["bandwidth_right"],
            color="#55A868",
            alpha=0.12,
            zorder=0,
        )
        ax.axvline(params["cutoff"], color="#2F6B3B", linewidth=1.8, zorder=1)
        ax.axvline(
            params["bandwidth_left"],
            color="#55A868",
            linewidth=1.2,
            linestyle="--",
            zorder=1,
        )
        ax.axvline(
            params["bandwidth_right"],
            color="#55A868",
            linewidth=1.2,
            linestyle="--",
            zorder=1,
        )
        ax.plot(x, 100 * subset["missing_share"], color="#C44E52", linewidth=1.8)
        ax.scatter(
            x,
            100 * subset["missing_share"],
            s=18 + 2.5 * np.sqrt(subset["n_papers"].astype(float)),
            color="#C44E52",
            alpha=0.9,
        )
        ax.set_title(f"{year}  c={params['cutoff']:.2f}, h={params['bandwidth']:.2f}")
        ax.set_ylim(0, max(100 * summary["missing_share"].max() * 1.15, 10))
        ax.set_xlim(x_min, x_max)
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.95,
            f"[{params['bandwidth_left']:.2f}, {params['bandwidth_right']:.2f}]",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#2F6B3B",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )

    for ax in axes.flat[len(years) :]:
        ax.axis("off")

    fig.suptitle("arXiv Missing Share by Mean Rating Bin and Year")
    fig.supxlabel(score_col.replace("_", " "))
    fig.supylabel("% missing in arXiv")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(args.sample_csv, low_memory=False)
    score_support_df = pd.read_csv(args.score_support_csv, low_memory=False)
    arxiv_df = pd.read_csv(args.arxiv_match_csv, low_memory=False)

    if args.score_col not in sample_df.columns:
        raise ValueError(f"Score column `{args.score_col}` not found in {args.sample_csv}")
    if args.score_col not in score_support_df.columns:
        raise ValueError(f"Score column `{args.score_col}` not found in {args.score_support_csv}")

    merged = sample_df.merge(
        arxiv_df[["paper_id", "matched", "match_status"]],
        on="paper_id",
        how="left",
    )
    if merged["matched"].isna().any():
        raise ValueError("Some sample rows did not merge to the arXiv match table.")

    merged["matched_arxiv"] = merged["matched"].fillna(False).astype(int)
    merged["missing_arxiv"] = 1 - merged["matched_arxiv"]
    year_design = get_year_design_params(sample_df)
    x_bounds = get_axis_bounds(score_support_df, score_col=args.score_col)

    binned = add_bins(merged, score_col=args.score_col, bin_width=args.bin_width)
    overall_summary = summarize_missingness(
        binned,
        group_cols=["score_bin_left", "score_bin_right", "score_bin_mid"],
        score_col=args.score_col,
    )
    yearly_summary = summarize_missingness(
        binned,
        group_cols=["year", "score_bin_left", "score_bin_right", "score_bin_mid"],
        score_col=args.score_col,
    )
    yearly_summary = yearly_summary.merge(year_design, on="year", how="left")

    overall_path = args.output_dir / "arxiv_missing_share_by_score_bin.csv"
    yearly_path = args.output_dir / "arxiv_missing_share_by_score_bin_by_year.csv"
    fig_overall_path = args.output_dir / "fig_arxiv_missing_share_by_score_bin.png"
    fig_year_path = args.output_dir / "fig_arxiv_missing_share_by_score_bin_by_year.png"

    overall_summary.to_csv(overall_path, index=False)
    yearly_summary.to_csv(yearly_path, index=False)
    plot_overall(overall_summary, score_col=args.score_col, output_path=fig_overall_path)
    plot_by_year(
        yearly_summary,
        score_col=args.score_col,
        output_path=fig_year_path,
        year_design=year_design,
        x_bounds=x_bounds,
    )

    print(f"Wrote {overall_path}")
    print(f"Wrote {yearly_path}")
    print(f"Wrote {fig_overall_path}")
    print(f"Wrote {fig_year_path}")


if __name__ == "__main__":
    main()
