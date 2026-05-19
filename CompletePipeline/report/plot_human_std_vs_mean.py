#!/usr/bin/env python3
"""
Scatter of reviewer disagreement (SD) vs paper mean, colored by actual decision.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
RDD_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

YEARS = [2018, 2019, 2020]

# ---------- load ----------
df = pd.read_csv(RDD_CSV)
df = df[df["year"].isin(YEARS)].copy()
df = df.dropna(subset=["mean_rating", "std_rating"])

accepted = df[df["accepted"] == 1.0]
rejected = df[df["accepted"] == 0.0]

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(rejected["mean_rating"], rejected["std_rating"],
           c="#D65F5F", alpha=0.35, s=18, edgecolors="none",
           label=f"Reject (n={len(rejected):,})", zorder=2)
ax.scatter(accepted["mean_rating"], accepted["std_rating"],
           c="#2CA02C", alpha=0.45, s=18, edgecolors="none",
           label=f"Accept (n={len(accepted):,})", zorder=3)

# median cutoff reference line
median_cutoff = df.groupby("year")["cutoff"].first().median()
ax.axvline(median_cutoff, color="grey", linewidth=0.8, linestyle="--",
           alpha=0.6, label=f"Median cutoff = {median_cutoff:.2f}", zorder=1)

ax.set_xlabel("Mean human reviewer rating", fontsize=11)
ax.set_ylabel("Std. dev. of human reviewer ratings (per paper)", fontsize=11)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.set_xlim(1, 10)
ax.set_ylim(-0.1, df["std_rating"].max() * 1.05)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.15)

n = len(df)
ax.set_title(
    f"Reviewer Disagreement vs. Paper Mean — ICLR 2018–2020  (n = {n:,})",
    fontsize=12, fontweight="bold",
)

fig.tight_layout()
fig.savefig(PLOT_DIR / "human_std_vs_mean.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "human_std_vs_mean.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'human_std_vs_mean.png'}")
print(f"Median std: {df['std_rating'].median():.2f}   Max std: {df['std_rating'].max():.2f}")
