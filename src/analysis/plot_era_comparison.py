"""
Figures for the era comparison (ICLR 2018-2020 vs 2025).

Three panels, in the order the argument runs:
  A  recall@k across all k — the headline
  B  rating decile -> mean citation percentile — the same claim as a dose-response
  C  citation distributions — the age confound, shown rather than hidden

Panel C is the honest one: 2025 papers have ~18 months of citations against 6-8
years, so the outcome itself is coarser there. That attenuates any correlation
mechanically and is a live alternative to the leakage reading of panel A.

Run: python src/analysis/plot_era_comparison.py [--tier-a-only]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figstyle as fs
import compare_eras as ce

OUT_PNG = "outputs/era_comparison.png"
OUT_PDF = "outputs/era_comparison.pdf"

C_1820, C_2025 = fs.BLUE, fs.ORANGE
KGRID = np.arange(0.02, 0.51, 0.02)


def recall_curve(d):
    rng = np.random.default_rng(ce.SEED)
    r, c = d["rating"].to_numpy(), d["cite_pct"].to_numpy()
    return np.array([ce.topk_recall(r, c, k, rng)[0] for k in KGRID]) * 100


def panel_a(ax, d1, d2):
    ax.plot(KGRID * 100, KGRID * 100, ls=(0, (4, 3)), lw=1.4, color=fs.MUTED, zorder=1)
    ax.annotate("random", (40, 36), color=fs.MUTED, fontsize="small",
                rotation=26, ha="left")
    ends = []
    for d, col, lbl in [(d1, C_1820, "2018–2020"), (d2, C_2025, "2025")]:
        y = recall_curve(d)
        ax.plot(KGRID * 100, y, lw=2.6, color=col, solid_capstyle="round", zorder=3)
        ends.append([y[-1], lbl, col])
    fs.label_ends(ax, ends, KGRID[-1] * 100, min_gap=5)
    ax.set_xlabel("% of papers selected")
    fs.axis_note(ax, "% of true top-k captured")
    ax.set_xlim(0, 78); ax.set_ylim(0, 72)
    ax.set_xticks(range(0, 51, 10))
    ax.set_title("Recall@k", pad=34)
    fs.clean(ax)


def panel_b(ax, d1, d2):
    ends = []
    for d, col, lbl in [(d1, C_1820, "2018–2020"), (d2, C_2025, "2025")]:
        q = pd.qcut(d["rating"].rank(method="first"), 10, labels=False)
        g = d.groupby(q)["cite_pct"].agg(["mean", "sem"])
        ax.errorbar(np.arange(1, 11), g["mean"] * 100, yerr=g["sem"] * 100, lw=2.6,
                    color=col, marker="o", ms=5, capsize=2, elinewidth=1)
        ends.append([g["mean"].iloc[-1] * 100, lbl, col])
    ax.axhline(50, ls=(0, (4, 3)), lw=1.4, color=fs.MUTED, zorder=1)
    fs.label_ends(ax, ends, 10, min_gap=6)
    ax.set_xlabel("committee rating decile")
    fs.axis_note(ax, "mean citation percentile")
    ax.set_xlim(0.5, 14.5); ax.set_xticks([1, 5, 10])
    ax.set_title("Dose–response", pad=34)
    fs.clean(ax)


def panel_c(ax, d1, d2):
    for d, col, lbl in [(d1, C_1820, "2018–2020"), (d2, C_2025, "2025")]:
        v = d["s2_citations"].clip(lower=0) + 1
        ax.hist(v, bins=np.logspace(0, 4, 34), weights=np.ones(len(v)) / len(v) * 100,
                histtype="step", lw=2.2, color=col)
        ax.axvline(v.median(), color=col, ls=(0, (2, 2)), lw=1.4)
        ax.annotate(f"median {v.median():.0f}", (v.median(), ax.get_ylim()[1] * 0.92),
                    color=col, fontsize="small", ha="center",
                    bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xscale("log")
    ax.set_xlabel("citations (log)")
    fs.axis_note(ax, "% of papers")
    ax.set_title("The age confound", pad=34)
    fs.clean(ax)


def main(tier_a_only=False):
    tiers = ["A"] if tier_a_only else ["A", "B"]
    ev = pd.read_csv(ce.EVAL_1820, low_memory=False)
    acc = ev[ev["decision"].str.startswith("Accept", na=False)]
    d1 = ce.load_era(acc[["paper_id", "year", "committee_rating"]],
                     ce.TIER_1820, "committee_rating", tiers)
    r25 = pd.read_csv(ce.RATE_2025)
    d2 = ce.load_era(r25[["paper_id", "year", "rating"]], ce.TIER_2025, "rating", tiers)

    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panel_a(axes[0], d1, d2)
    panel_b(axes[1], d1, d2)
    panel_c(axes[2], d1, d2)

    rho1 = stats.spearmanr(d1["rating"], d1["cite_pct"])[0]
    rho2 = stats.spearmanr(d2["rating"], d2["cite_pct"])[0]
    fs.title_block(fig, "The committee ranks 2018–2020 papers far better than 2025 "
                        f"(ρ {rho1:.2f} vs {rho2:.2f})")
    fig.subplots_adjust(left=0.045, right=0.975, top=0.775, bottom=0.135, wspace=0.34)

    os.makedirs("outputs", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}  rho {rho1:.3f} / {rho2:.3f}  (n={len(d1):,} / {len(d2):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
