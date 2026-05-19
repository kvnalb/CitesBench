#!/usr/bin/env python3
"""
Four histograms of within-paper pairwise review-text similarity
(balanced overlap @ cosine threshold 0.5):
  (A) Human reviewer pairs             (human × human, within paper)
  (B) LLM persona pairs                (LLM × LLM, within paper)
  (C) Cross pairs                      (every human × every LLM persona)
  (D) Per human reviewer: max across LLM personas in same paper
      (best-match LLM for each human reviewer)
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DATA_DIR = ROOT / "OutputNew" / "Empirics" / "review_similarity_within_paper"
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"


def load_scores(path: Path) -> np.ndarray:
    vals = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("balanced") is not None:
                vals.append(float(row["balanced"]))
    return np.array(vals)


def load_cross_max_per_human(path: Path) -> np.ndarray:
    """For each (paper_id, human_src), take the max balanced score across LLM personas."""
    best: dict[tuple[str, str], float] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("balanced") is None:
                continue
            key = (row["paper_id"], row["human_src"])
            val = float(row["balanced"])
            if key not in best or val > best[key]:
                best[key] = val
    return np.array(list(best.values()))


human = load_scores(DATA_DIR / "human_pairs.jsonl")
llm = load_scores(DATA_DIR / "llm_pairs.jsonl")
cross = load_scores(DATA_DIR / "cross_pairs.jsonl")
best_llm_per_human = load_cross_max_per_human(DATA_DIR / "cross_pairs.jsonl")

print(f"Human pairs:         {len(human):,}  (mean={human.mean():.3f}, median={np.median(human):.3f})")
print(f"LLM pairs:           {len(llm):,}  (mean={llm.mean():.3f}, median={np.median(llm):.3f})")
print(f"Cross pairs:         {len(cross):,}  (mean={cross.mean():.3f}, median={np.median(cross):.3f})")
print(f"Best-LLM per human:  {len(best_llm_per_human):,}  "
      f"(mean={best_llm_per_human.mean():.3f}, median={np.median(best_llm_per_human):.3f})")

bins = np.linspace(0, 1, 41)

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
(ax1, ax2), (ax3, ax4) = axes


def panel(ax, vals, color, title_letter, title, legend_loc, show_ylabel):
    ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(vals.mean(), color="#222222", linestyle="-", linewidth=1.0,
               label=f"Mean = {vals.mean():.3f}")
    ax.axvline(np.median(vals), color="#222222", linestyle="--", linewidth=1.0,
               label=f"Median = {np.median(vals):.3f}")
    ax.set_xlabel("Balanced overlap score (cosine ≥ 0.5)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Number of pairs", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9, loc=legend_loc)
    ax.set_title(f"({title_letter})  {title}  (n = {len(vals):,})",
                 fontsize=11, fontweight="bold")


panel(ax1, human, "#4878CF", "A", "Human reviewer pairs", "upper right", True)
panel(ax2, llm, "#FF9933", "B", "LLM persona pairs", "upper left", False)
panel(ax3, cross, "#8E6FBF", "C", "Human × LLM cross pairs", "upper right", True)
panel(ax4, best_llm_per_human, "#2CA02C", "D",
      "Per human reviewer: best LLM match", "upper right", False)

fig.suptitle(
    "Within-Paper Pairwise Review Similarity — ICLR 2018–2020",
    fontsize=13, fontweight="bold", y=1.00,
)
fig.tight_layout()
fig.savefig(PLOT_DIR / "pairwise_review_similarity.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "pairwise_review_similarity.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'pairwise_review_similarity.png'}")
