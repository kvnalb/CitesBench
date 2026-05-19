#!/usr/bin/env python3
"""
Acceptance probability by paper-level mean rating, one panel per year (2018-2020).
Reproduces the upstream fig_acceptance_vs_score_by_year.png filtered to sample years.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
ALL_PAPERS = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "paper_level_all_years.csv"
RDD_SAMPLE = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

YEARS = [2018, 2019, 2020]

# ---------- load data ----------
df = pd.read_csv(ALL_PAPERS)
df = df[df["year"].isin(YEARS)].dropna(subset=["mean_rating"])

rdd = pd.read_csv(RDD_SAMPLE, usecols=["year", "cutoff", "bandwidth"])
rdd = rdd[rdd["year"].isin(YEARS)].drop_duplicates(subset=["year"])
cutoff_map = dict(zip(rdd["year"], rdd["cutoff"]))
bandwidth_map = dict(zip(rdd["year"], rdd["bandwidth"]))

# ---------- plot ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, year in zip(axes, YEARS):
    subset = df[df["year"] == year]
    grouped = (
        subset.groupby("mean_rating", as_index=False)
        .agg(accept_rate=("accepted", "mean"), n_papers=("paper_id", "size"))
        .sort_values("mean_rating")
    )

    cutoff = cutoff_map[year]
    bandwidth = bandwidth_map[year]
    size_scale = 20 + 12 * np.sqrt(grouped["n_papers"].astype(float))

    # bandwidth window
    ax.axvspan(cutoff - bandwidth, cutoff + bandwidth,
               color="#7F7F7F", alpha=0.12, zorder=0)

    # acceptance curve
    ax.plot(grouped["mean_rating"], grouped["accept_rate"],
            color="#4C72B0", alpha=0.6, zorder=2)
    ax.scatter(grouped["mean_rating"], grouped["accept_rate"],
               s=size_scale, color="#4C72B0", alpha=0.85, edgecolors="white",
               linewidth=0.3, zorder=3)

    # cutoff line
    ax.axvline(cutoff, color="#C44E52", linestyle="--", linewidth=1.2, zorder=4)

    # RDD sample count
    in_window = subset[
        (subset["mean_rating"] >= cutoff - bandwidth)
        & (subset["mean_rating"] <= cutoff + bandwidth)
    ]

    ax.set_title(f"ICLR {year}  (n = {len(subset):,})", fontsize=12)
    ax.text(0.03, 0.96,
            f"c = {cutoff:.2f}\nh = {bandwidth:.2f}\nRDD sample: {len(in_window):,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(1, 10)
    ax.set_xlabel("Mean reviewer rating", fontsize=10)
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Acceptance rate", fontsize=10)

fig.suptitle("Acceptance Probability by Paper-Level Mean Rating — ICLR 2018–2020",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / "acceptance_vs_score_by_year.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "acceptance_vs_score_by_year.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'acceptance_vs_score_by_year.png'}")
