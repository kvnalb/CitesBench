#!/usr/bin/env python3
"""
Two-panel figure:
  (A) Scatter of human mean rating vs LLM committee rating, colored by actual decision.
      Stats box: n, Pearson r, Spearman rho, MAE, Bias.
  (B) Density overlay of human mean vs LLM committee scores.
"""

from __future__ import annotations
from pathlib import Path
import json
import sqlite3
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DATA = ROOT / "OutputNew" / "Empirics" / "human_vs_llm_committee_scores_20260421" / "human_vs_llm_committee_scores.csv"
DB = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
EMPIRICS = ROOT / "OutputNew" / "Empirics"
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

RUNS = [
    "gemma_ready7_wave1_cached_v2",
    "gemma_ready8_wave2_incremental",
    "gemma_ready8_wave3_single_managed",
]


def parse_num(text):
    if not text:
        return None
    m = re.match(r"(\d+)", str(text))
    return int(m.group(1)) if m else None


def collect_raw_scores(paper_ids: set[str]) -> tuple[list[float], list[float]]:
    # Raw human reviewer ratings
    conn = sqlite3.connect(str(DB))
    pid_list = list(paper_ids)
    placeholders = ",".join(["?"] * len(pid_list))
    rows = conn.execute(
        f"SELECT r.rating FROM REVIEW r WHERE r.paper_id IN ({placeholders})",
        pid_list,
    ).fetchall()
    conn.close()
    raw_human = [parse_num(r[0]) for r in rows]
    raw_human = [r for r in raw_human if r is not None]

    # Raw persona ratings from LLM pipeline
    raw_llm = []
    for run in RUNS:
        run_dir = EMPIRICS / run
        for search_root in [run_dir] + sorted(run_dir.glob("shard_*")):
            papers = search_root / "papers"
            if not papers.is_dir():
                continue
            for p in papers.iterdir():
                if p.name not in paper_ids:
                    continue
                persona_dir = p / "persona_reviews"
                if not persona_dir.is_dir():
                    continue
                for f in persona_dir.glob("*.json"):
                    d = json.loads(f.read_text())
                    r = d.get("rating")
                    if r is not None:
                        raw_llm.append(float(r))
    return raw_human, raw_llm

# ---------- load ----------
df = pd.read_csv(DATA)
df = df.dropna(subset=["human_rating_mean", "llm_rating"])

accepted_mask = df["decision"].astype(str).str.lower().str.startswith("accept")
accepted = df[accepted_mask]
rejected = df[~accepted_mask]

# ---------- stats ----------
h = df["human_rating_mean"].to_numpy()
l = df["llm_rating"].to_numpy()
pearson_r = stats.pearsonr(h, l)[0]
spearman_rho = stats.spearmanr(h, l)[0]
mae = float(np.mean(np.abs(l - h)))
bias = float(np.mean(l - h))  # LLM - human
n = len(df)

# ---------- raw scores ----------
raw_human, raw_llm = collect_raw_scores(set(df["paper_id"].tolist()))
print(f"Raw human reviews: {len(raw_human)}, raw LLM persona reviews: {len(raw_llm)}")

# ---------- plot ----------
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.28, wspace=0.22)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# ── Panel A: Scatter ──
ax1.scatter(rejected["human_rating_mean"], rejected["llm_rating"],
            c="#D65F5F", alpha=0.35, s=18, edgecolors="none",
            label=f"Reject (n={len(rejected):,})", zorder=2)
ax1.scatter(accepted["human_rating_mean"], accepted["llm_rating"],
            c="#2CA02C", alpha=0.45, s=18, edgecolors="none",
            label=f"Accept (n={len(accepted):,})", zorder=3)

# diagonal
ax1.plot([1, 10], [1, 10], color="grey", linewidth=1.0,
         linestyle="--", alpha=0.7, label="LLM = human", zorder=1)

# stats box
stats_text = (
    f"n = {n:,}\n"
    f"Pearson r = {pearson_r:.3f}\n"
    f"Spearman ρ = {spearman_rho:.3f}\n"
    f"MAE = {mae:.2f}\n"
    f"Bias (LLM − human) = {bias:+.2f}"
)
ax1.text(0.03, 0.97, stats_text, transform=ax1.transAxes, fontsize=9,
         va="top", ha="left", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="#999999", alpha=0.9))

ax1.set_xlabel("Human mean rating", fontsize=11)
ax1.set_ylabel("LLM committee rating", fontsize=11)
ax1.set_xlim(1, 10)
ax1.set_ylim(1, 10)
ax1.set_aspect("equal")
ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax1.legend(fontsize=9, loc="lower right")
ax1.grid(alpha=0.15)
ax1.set_title("(A)  Paper-level score relationship (aggregated)", fontsize=11)

# ── Panel B: Density of aggregated means ──
x = np.linspace(1, 10, 400)
kde_h = gaussian_kde(h, bw_method=0.25)
kde_l = gaussian_kde(l, bw_method=0.25)
d_h = kde_h(x)
d_l = kde_l(x)

ax2.fill_between(x, d_h, alpha=0.3, color="#4878CF")
ax2.plot(x, d_h, color="#4878CF", linewidth=1.5,
         label=f"Human mean (μ={h.mean():.2f}, σ={h.std():.2f})")

ax2.fill_between(x, d_l, alpha=0.3, color="#FF9933")
ax2.plot(x, d_l, color="#FF9933", linewidth=1.5,
         label=f"LLM committee (μ={l.mean():.2f}, σ={l.std():.2f})")

ax2.set_xlabel("Rating", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_xlim(1, 10)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(alpha=0.15)
ax2.set_title("(B)  Paper-level aggregate score distributions", fontsize=11)

# ── Panel C: Density of raw (individual) scores ──
raw_h = np.array(raw_human, dtype=float)
raw_l = np.array(raw_llm, dtype=float)
kde_rh = gaussian_kde(raw_h, bw_method=0.25)
kde_rl = gaussian_kde(raw_l, bw_method=0.25)
d_rh = kde_rh(x)
d_rl = kde_rl(x)

ax3.fill_between(x, d_rh, alpha=0.3, color="#4878CF")
ax3.plot(x, d_rh, color="#4878CF", linewidth=1.5,
         label=f"Human reviews  n={len(raw_h):,}  (μ={raw_h.mean():.2f}, σ={raw_h.std():.2f})")

ax3.fill_between(x, d_rl, alpha=0.3, color="#FF9933")
ax3.plot(x, d_rl, color="#FF9933", linewidth=1.5,
         label=f"LLM personas  n={len(raw_l):,}  (μ={raw_l.mean():.2f}, σ={raw_l.std():.2f})")

ax3.set_xlabel("Rating", fontsize=11)
ax3.set_ylabel("Density", fontsize=11)
ax3.set_xlim(1, 10)
ax3.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax3.legend(fontsize=8.5, loc="upper right")
ax3.grid(alpha=0.15)
ax3.set_title("(C)  Raw individual score distributions (no aggregation)", fontsize=11)

fig.suptitle("Human Review Scores vs. Final LLM Committee Scores — ICLR 2018–2020",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / "human_vs_llm_scatter_density.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "human_vs_llm_scatter_density.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'human_vs_llm_scatter_density.png'}")
print(f"n={n}  Pearson={pearson_r:.3f}  Spearman={spearman_rho:.3f}  MAE={mae:.3f}  Bias={bias:+.3f}")
