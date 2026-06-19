#!/usr/bin/env python3
"""
Mean human review score vs citations for accepted papers — with paper-level topic FEs.

Topics: k-means (k=20, seed=42) on L2-normalized SPECTER2 abstract embeddings
(matches Code/Empirics/04_reviews_predict_cites.R). Plots Frisch-Waugh-Lovell
residualized citations vs residualized rating after partialling out year + topic FEs.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DATA = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv"
EMB = ROOT / "OutputNew" / "Empirics" / "embeddings" / "abstracts_specter2_2018_2023.csv"
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

YEARS = [2018, 2019, 2020]
N_TOPICS = 20
SEED = 42


def residualize(y: np.ndarray, fe_cols: list[pd.Series]) -> np.ndarray:
    """Return y residuals after removing additive group means for each FE
    (iterated projection — same as demean)."""
    resid = y.astype(float).copy()
    for _ in range(50):
        old = resid.copy()
        for fe in fe_cols:
            means = pd.Series(resid).groupby(fe.to_numpy()).transform("mean").to_numpy()
            resid = resid - means
        if np.max(np.abs(resid - old)) < 1e-10:
            break
    return resid


df = pd.read_csv(DATA, low_memory=False)
df = df[df["in_year_specific_rdd_sample"] == True]
df = df[df["year"].isin(YEARS) & (df["accepted"] == 1)]
df = df.dropna(subset=["mean_rating", "openalex_cited_by_count"]).copy()
df["citations"] = df["openalex_cited_by_count"].astype(float)
df["lcites"] = np.log1p(df["citations"])

print(f"Accepted w/ citations (2018-2020): {len(df):,}")

emb = pd.read_csv(EMB)
emb_cols = [c for c in emb.columns if c.startswith("emb_")]
emb = emb[["paper_id"] + emb_cols]

merged = df.merge(emb, on="paper_id", how="left")
has_emb_mask = merged[emb_cols[0]].notna()
print(f"  w/ SPECTER2 embeddings:           {int(has_emb_mask.sum()):,}")

emb_mat = merged.loc[has_emb_mask, emb_cols].to_numpy(dtype=float)
emb_mat = emb_mat / np.linalg.norm(emb_mat, axis=1, keepdims=True)

km = KMeans(n_clusters=N_TOPICS, random_state=SEED, n_init=5, max_iter=50).fit(emb_mat)
merged.loc[has_emb_mask, "topic"] = km.labels_

d = merged[has_emb_mask].copy()
d["topic"] = d["topic"].astype(int)
d["year"] = d["year"].astype(int)

n_plot = len(d)
print(f"  in residualized plot (topic + year FE): {n_plot:,}")
print(f"  topic sizes (top 10): {d['topic'].value_counts().head(10).to_dict()}")

lcites_resid = residualize(d["lcites"].to_numpy(), [d["year"], d["topic"]])
rating_resid = residualize(d["mean_rating"].to_numpy(), [d["year"], d["topic"]])

slope, intercept, r, pval, _ = stats.linregress(rating_resid, lcites_resid)
pearson = stats.pearsonr(d["mean_rating"], d["lcites"])[0]
spearman = stats.spearmanr(d["mean_rating"], d["citations"])[0]

year_colors = {2018: "#4878CF", 2019: "#6ACC65", 2020: "#D65F5F"}

fig, ax = plt.subplots(figsize=(9, 6))

for yr in YEARS:
    mask = (d["year"] == yr).to_numpy()
    ax.scatter(rating_resid[mask], lcites_resid[mask],
               c=year_colors[yr], alpha=0.55, s=22, edgecolors="none",
               label=f"{yr} (n={mask.sum():,})")

xline = np.linspace(rating_resid.min(), rating_resid.max(), 100)
ax.plot(xline, intercept + slope * xline, color="#222222", linewidth=1.6,
        label=f"OLS: slope = {slope:+.3f}", zorder=5)

ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5, zorder=1)
ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5, zorder=1)

ax.set_xlabel("Mean human reviewer rating  (residualized on year + topic FE)", fontsize=11)
ax.set_ylabel("log(1 + citations)  (residualized on year + topic FE)", fontsize=11)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.grid(alpha=0.15)

stats_text = (
    f"Accepted papers 2018–2020:              n = {len(df):,}\n"
    f"  w/ OpenAlex citations:                n = {len(df):,}\n"
    f"  w/ SPECTER2 embeddings (in plot):     n = {n_plot:,}\n"
    f"Topics: k-means (k={N_TOPICS}) on SPECTER2\n"
    f"Residualized slope          = {slope:+.3f}  (p = {pval:.1e})\n"
    f"Residualized Pearson r      = {r:+.3f}\n"
    f"Raw Pearson (rating, lcites) = {pearson:+.3f}\n"
    f"Raw Spearman (rating, cites) = {spearman:+.3f}"
)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#999999", alpha=0.9))

ax.legend(fontsize=9, loc="lower right")
ax.set_title(
    "Rating vs. Citations — After Partialling Out Year + Topic FE  (ICLR 2018–2020, accepted)",
    fontsize=11, fontweight="bold",
)

fig.tight_layout()
fig.savefig(PLOT_DIR / "rating_vs_citations_topicFE.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "rating_vs_citations_topicFE.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'rating_vs_citations_topicFE.png'}")
print(f"Residualized slope={slope:+.3f}  r={r:+.3f}  Raw Pearson={pearson:+.3f}")
