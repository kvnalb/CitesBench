#!/usr/bin/env python3
"""
Grouped bar chart: committee recommendation (4 bins) x actual decision (accept/reject).
Shows % of papers in each of the 8 cells.
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

RECO_ORDER = ["reject", "borderline reject", "borderline accept", "strong accept"]


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
                "accepted": r.get("accepted"),
                "recommendation": c.get("recommendation", ""),
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

n_total = len(all_rows)

# ---------- count 4x2 ----------
counts = {reco: {"Accept": 0, "Reject": 0} for reco in RECO_ORDER}
for row in all_rows:
    reco = row["recommendation"].strip().lower()
    decision = "Accept" if row["accepted"] == 1.0 else "Reject"
    if reco in counts:
        counts[reco][decision] += 1

# ---------- plot ----------
x = np.arange(len(RECO_ORDER))
bar_width = 0.35

accept_pcts = [100 * counts[r]["Accept"] / n_total for r in RECO_ORDER]
reject_pcts = [100 * counts[r]["Reject"] / n_total for r in RECO_ORDER]
accept_counts = [counts[r]["Accept"] for r in RECO_ORDER]
reject_counts = [counts[r]["Reject"] for r in RECO_ORDER]

fig, ax = plt.subplots(figsize=(9, 5))

bars_reject = ax.bar(x - bar_width / 2, reject_pcts, bar_width,
                     color="#D65F5F", alpha=0.85, label="Actual: Reject", edgecolor="white")
bars_accept = ax.bar(x + bar_width / 2, accept_pcts, bar_width,
                     color="#2CA02C", alpha=0.85, label="Actual: Accept", edgecolor="white")

# annotate counts and percentages
for bar, count, pct in zip(bars_reject, reject_counts, reject_pcts):
    if pct > 0.5:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5)

for bar, count, pct in zip(bars_accept, accept_counts, accept_pcts):
    if pct > 0.5:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels([r.title() for r in RECO_ORDER], fontsize=10)
ax.set_xlabel("Committee recommendation", fontsize=11)
ax.set_ylabel("% of all papers", fontsize=11)
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", alpha=0.2)

ax.set_title(
    f"Committee Recommendation vs. Actual Decision — ICLR 2018–2020  (n = {n_total:,})",
    fontsize=12, fontweight="bold",
)

fig.tight_layout()
fig.savefig(PLOT_DIR / "recommendation_vs_decision.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "recommendation_vs_decision.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'recommendation_vs_decision.png'}")
