#!/usr/bin/env python3
"""
Figure 1: Two-panel histogram of human-reviewer mean scores for ICLR 2018-2020.
  Panel A: Pooled across years.
  Panel B: By year.

Figure 2: Pooled histogram split by accept (green) / reject (red).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
ALL_PAPERS = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "paper_level_all_years.csv"
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

YEARS = [2018, 2019, 2020]
BINS = [x * 0.5 for x in range(0, 23)]

# ---------- load data ----------
df = pd.read_csv(ALL_PAPERS)
df = df[df["year"].isin(YEARS)].copy()
df = df.dropna(subset=["mean_rating"])

# ---------- Figure 1: two-panel (pooled + by year) ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.6]})

# Panel A: pooled
ax = axes[0]
ax.hist(
    df["mean_rating"], bins=BINS,
    color="#4878CF", edgecolor="white", linewidth=0.5, alpha=0.85,
)
ax.set_xlabel("Mean human-reviewer rating", fontsize=10)
ax.set_ylabel("Number of papers", fontsize=10)
ax.set_xlim(1, 10)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.set_title(f"(A)  All years pooled  (n = {len(df):,})", fontsize=11)

# Panel B: by year, dodged
ax = axes[1]
bin_edges = np.array(BINS)
bin_width = bin_edges[1] - bin_edges[0]
n_years = len(YEARS)
bar_width = bin_width / (n_years + 0.5)
colors = ["#4878CF", "#6ACC65", "#D65F5F"]
for i, year in enumerate(YEARS):
    subset = df[df["year"] == year]["mean_rating"]
    counts, _ = np.histogram(subset, bins=bin_edges)
    offsets = bin_edges[:-1] + i * bar_width
    ax.bar(
        offsets, counts, width=bar_width,
        color=colors[i], edgecolor="white", linewidth=0.3, alpha=0.85,
        label=f"{year} (n={len(subset):,})", align="edge",
    )
ax.set_xlabel("Mean human-reviewer rating", fontsize=10)
ax.set_xlim(1, 10)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.legend(fontsize=9, loc="upper right")
ax.set_title("(B)  By year", fontsize=11)

fig.suptitle(
    "Distribution of Mean Human-Reviewer Ratings — ICLR 2018–2020",
    fontsize=13, fontweight="bold", y=1.02,
)
fig.tight_layout()
fig.savefig(PLOT_DIR / "mean_score_histogram.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "mean_score_histogram.png", bbox_inches="tight", dpi=300)
print(f"Saved figure 1 to {PLOT_DIR / 'mean_score_histogram.png'}")

# ---------- Figure 2: accept/reject histogram + acceptance rate by score ----------
df_accept = df[df["accepted"] == 1.0]["mean_rating"]
df_reject = df[df["accepted"] == 0.0]["mean_rating"]

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A: stacked histogram
ax2a.hist(
    [df_reject, df_accept], bins=BINS, stacked=True,
    color=["#D65F5F", "#6ACC65"], edgecolor="white", linewidth=0.5, alpha=0.85,
    label=[f"Reject (n={len(df_reject):,})", f"Accept (n={len(df_accept):,})"],
)
ax2a.set_xlabel("Mean human-reviewer rating", fontsize=10)
ax2a.set_ylabel("Number of papers", fontsize=10)
ax2a.set_xlim(1, 10)
ax2a.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax2a.legend(fontsize=9, loc="upper right")
ax2a.set_title(f"(A)  Score distribution by decision  (n = {len(df):,})", fontsize=11)

# Panel B: acceptance rate by score bin (RDD-style)
bin_edges = np.array(BINS)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
counts_all, _ = np.histogram(df["mean_rating"], bins=bin_edges)
counts_acc, _ = np.histogram(df_accept, bins=bin_edges)
mask = counts_all >= 5  # only plot bins with enough papers
acc_rate = np.where(mask, counts_acc / counts_all, np.nan)

ax2b.scatter(bin_centers[mask], acc_rate[mask], s=counts_all[mask] * 0.4,
             color="#4878CF", alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
ax2b.plot(bin_centers[mask], acc_rate[mask],
          color="#4878CF", linewidth=1.5, alpha=0.5, zorder=2)
ax2b.axhline(0.5, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)

# shade typical cutoff region
median_cutoff = df.merge(
    pd.read_csv(ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv",
                usecols=["year", "cutoff"]).drop_duplicates(),
    on="year", how="left",
)["cutoff"].median()
ax2b.axvline(median_cutoff, color="#C44E52", linewidth=1.5, linestyle="--",
             label=f"Median cutoff = {median_cutoff:.2f}", zorder=4)

ax2b.set_xlabel("Mean human-reviewer rating", fontsize=10)
ax2b.set_ylabel("Acceptance rate", fontsize=10)
ax2b.set_xlim(1, 10)
ax2b.set_ylim(-0.05, 1.05)
ax2b.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax2b.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax2b.legend(fontsize=9, loc="upper left")
ax2b.set_title("(B)  Acceptance rate by mean score", fontsize=11)

fig2.suptitle(
    "Human-Reviewer Ratings and Acceptance — ICLR 2018–2020",
    fontsize=13, fontweight="bold", y=1.02,
)
fig2.tight_layout()
fig2.savefig(PLOT_DIR / "mean_score_histogram_by_decision.pdf", bbox_inches="tight", dpi=300)
fig2.savefig(PLOT_DIR / "mean_score_histogram_by_decision.png", bbox_inches="tight", dpi=300)
print(f"Saved figure 2 to {PLOT_DIR / 'mean_score_histogram_by_decision.png'}")
