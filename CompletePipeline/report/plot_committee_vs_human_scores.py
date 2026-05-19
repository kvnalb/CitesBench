#!/usr/bin/env python3
"""
Scatter plot of committee weighted-average rating vs human mean reviewer rating.
Papers colored green (accepted) or red (rejected) by actual ICLR decision.
"""

from __future__ import annotations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"
EMPIRICS = ROOT / "OutputNew" / "Empirics"

RUNS = [
    "gemma_ready7_wave1_cached_v2",
    "gemma_ready8_wave2_incremental",
    "gemma_ready8_wave3_single_managed",
]


def collect_papers(run_dir: Path) -> list[dict]:
    rows = []
    for search_root in [run_dir] + sorted(run_dir.glob("shard_*")):
        papers = search_root / "papers"
        if not papers.is_dir():
            continue
        for p in papers.iterdir():
            cr = p / "coarse_review.json"
            pr = p / "paper_result.json"
            if not cr.exists() or not pr.exists():
                continue
            c = json.loads(cr.read_text())
            r = json.loads(pr.read_text())
            rows.append({
                "paper_id": p.name,
                "year": r.get("year"),
                "mean_rating": r.get("mean_rating"),
                "accepted": r.get("accepted"),
                "committee_rating": c.get("rating"),
            })
    return rows


# ---------- collect data ----------
all_rows = []
seen = set()
for run in RUNS:
    for row in collect_papers(EMPIRICS / run):
        if row["paper_id"] not in seen:
            seen.add(row["paper_id"])
            all_rows.append(row)

human = np.array([r["mean_rating"] for r in all_rows])
committee = np.array([r["committee_rating"] for r in all_rows])
accepted = np.array([r["accepted"] == 1.0 for r in all_rows])

print(f"Total papers: {len(all_rows)}")
print(f"Accepted: {accepted.sum()}, Rejected: {(~accepted).sum()}")

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(7, 6.5))

# rejected first (behind), then accepted
ax.scatter(
    human[~accepted], committee[~accepted],
    c="#D65F5F", alpha=0.35, s=18, edgecolors="none",
    label=f"Reject (n={int((~accepted).sum()):,})", zorder=2,
)
ax.scatter(
    human[accepted], committee[accepted],
    c="#2CA02C", alpha=0.45, s=18, edgecolors="none",
    label=f"Accept (n={int(accepted.sum()):,})", zorder=3,
)

# diagonal
lims = [1, 10]
ax.plot(lims, lims, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)

ax.set_xlabel("Mean human-reviewer rating", fontsize=11)
ax.set_ylabel("Committee weighted-average rating", fontsize=11)
ax.set_xlim(1, 10)
ax.set_ylim(1, 10)
ax.set_aspect("equal")
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.15)

corr = np.corrcoef(human, committee)[0, 1]
ax.set_title(
    f"Committee vs. Human Ratings — ICLR 2018–2020  (n = {len(all_rows):,}, r = {corr:.2f})",
    fontsize=12, fontweight="bold",
)

fig.tight_layout()
fig.savefig(PLOT_DIR / "committee_vs_human_scores.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "committee_vs_human_scores.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'committee_vs_human_scores.png'}")
