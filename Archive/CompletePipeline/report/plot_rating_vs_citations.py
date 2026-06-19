#!/usr/bin/env python3
"""
Scatter of mean human review score vs citations for accepted papers
(RDD sample, ICLR 2018-2020). Citations from OpenAlex (matched via arXiv).
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DATA = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv"
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

YEARS = [2018, 2019, 2020]

df = pd.read_csv(DATA, low_memory=False)
df = df[df["in_year_specific_rdd_sample"] == True]
df = df[df["year"].isin(YEARS)]

accepted_total = df[df["accepted"] == 1]
n_accepted = len(accepted_total)

acc = accepted_total[accepted_total["openalex_cited_by_count"].notna()].copy()
acc["citations"] = acc["openalex_cited_by_count"].astype(float)
n_cites = len(acc)

per_year = {y: (int((accepted_total["year"] == y).sum()),
               int((acc["year"] == y).sum())) for y in YEARS}

x = acc["mean_rating"].to_numpy()
y = acc["citations"].to_numpy()
log_y = np.log1p(y)

pearson = stats.pearsonr(x, log_y)[0]
spearman = stats.spearmanr(x, y)[0]
median_citations = float(np.median(y))

year_colors = {2018: "#4878CF", 2019: "#6ACC65", 2020: "#D65F5F"}

fig, ax = plt.subplots(figsize=(9, 6))

for yr in YEARS:
    sub = acc[acc["year"] == yr]
    ax.scatter(sub["mean_rating"], sub["citations"],
               c=year_colors[yr], alpha=0.55, s=22, edgecolors="none",
               label=f"{yr} (n={len(sub):,})")

bins = np.arange(np.floor(x.min()), np.ceil(x.max()) + 0.5, 0.5)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_medians = []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (x >= lo) & (x < hi)
    bin_medians.append(np.median(y[mask]) if mask.sum() >= 5 else np.nan)
ax.plot(bin_centers, bin_medians, color="#222222", linewidth=1.6,
        marker="o", markersize=5, label="Median citations (0.5-rating bins)",
        zorder=5)

ax.set_yscale("symlog", linthresh=1)
ax.set_ylim(-0.5, max(y) * 1.1)
ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
ax.set_xlabel("Mean human reviewer rating", fontsize=11)
ax.set_ylabel("Citations (OpenAlex, symlog scale)", fontsize=11)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.grid(alpha=0.15, which="both")

stats_text = (
    f"Accepted papers (RDD 2018–2020): n = {n_accepted:,}\n"
    f"  w/ OpenAlex citation data:     n = {n_cites:,}  ({100*n_cites/n_accepted:.1f}%)\n"
    f"  2018: {per_year[2018][1]:>3}/{per_year[2018][0]:>3}   "
    f"2019: {per_year[2019][1]:>3}/{per_year[2019][0]:>3}   "
    f"2020: {per_year[2020][1]:>3}/{per_year[2020][0]:>3}\n"
    f"Pearson r (rating, log1p cites) = {pearson:+.3f}\n"
    f"Spearman ρ (rating, cites)      = {spearman:+.3f}\n"
    f"Median citations                = {median_citations:.0f}"
)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#999999", alpha=0.9))

ax.legend(fontsize=9, loc="lower right")
ax.set_title(
    "Mean Human Review Rating vs. OpenAlex Citations — Accepted ICLR 2018–2020",
    fontsize=12, fontweight="bold",
)

fig.tight_layout()
fig.savefig(PLOT_DIR / "rating_vs_citations.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "rating_vs_citations.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'rating_vs_citations.png'}")
print(f"Accepted: {n_accepted}   w/ citations: {n_cites}")
print(f"Pearson(rating, log1p cites) = {pearson:+.3f}   Spearman(rating, cites) = {spearman:+.3f}")
