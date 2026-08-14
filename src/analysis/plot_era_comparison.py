"""
Figures for the era comparison (ICLR 2018-2020 vs 2025).

All three human/LLM selectors are drawn in both eras, because the committee's
era-to-era fall means nothing on its own — the 2025 outcome is measured over ~18
months against 6-8 years, so *any* selector should look worse there. The human
reviewers and the area chairs are the control: they face the identical outcome in
each era, so whatever the citation window costs, it costs them too. If only the
committee falls, the window is not the explanation.

Encoding: colour = era, line style = selector (solid LLM, dashed reviewers,
dotted area chairs).

  A  recall@k across all k — the decision-relevant view
  B  own-score decile -> mean citation percentile — the same claim as dose-response
  C  citation distributions — the age confound, shown rather than hidden

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
import did_leakage as dl

OUT_PNG = "outputs/era_comparison.png"
OUT_PDF = "outputs/era_comparison.pdf"

C_1820, C_2025 = fs.BLUE, fs.ORANGE
KGRID = np.arange(0.02, 0.51, 0.02)

# colour carries the era, style carries the selector
SELECTORS = [("rating", "LLM committee", "-", 2.6),
             ("mean_rating", "Human reviewers", (0, (5, 2)), 1.9),
             ("ac", "Area chairs", (0, (1.4, 1.8)), 1.9)]


def recall_curve(d, col):
    rng = np.random.default_rng(ce.SEED)
    r, c = d[col].to_numpy(float), d["cite_pct"].to_numpy()
    return np.array([ce.topk_recall(r, c, k, rng)[0] for k in KGRID]) * 100


def panel_a(ax, eras):
    ax.plot(KGRID * 100, KGRID * 100, ls=(0, (4, 3)), lw=1.4, color=fs.MUTED, zorder=1)
    ax.annotate("random", (40, 36), color=fs.MUTED, fontsize="small",
                rotation=26, ha="left")
    ends = []
    for d, colr, era in eras:
        for col, lbl, style, lw in SELECTORS:
            y = recall_curve(d, col)
            ax.plot(KGRID * 100, y, lw=lw, ls=style, color=colr,
                    solid_capstyle="round", zorder=3)
            if col == "rating":                       # label the committee only
                ends.append([y[-1], era, colr])
    fs.label_ends(ax, ends, KGRID[-1] * 100, min_gap=5)
    ax.set_xlabel("% of papers selected")
    fs.axis_note(ax, "% of true top-k captured")
    ax.set_xlim(0, 78); ax.set_ylim(0, 72)
    ax.set_xticks(range(0, 51, 10))
    ax.set_title("Recall@k", pad=34)
    fs.clean(ax)


def panel_b(ax, eras):
    ends = []
    for d, colr, era in eras:
        for col, lbl, style, lw in SELECTORS:
            # decile on the selector's OWN score, so each is graded on its own ranking
            q = pd.qcut(d[col].rank(method="first"), 10, labels=False)
            g = d.groupby(q)["cite_pct"].agg(["mean", "sem"])
            ax.errorbar(np.arange(1, 11), g["mean"] * 100, yerr=g["sem"] * 100,
                        lw=lw, ls=style, color=colr,
                        marker="o" if col == "rating" else None, ms=4,
                        capsize=2, elinewidth=0.9)
            if col == "rating":
                ends.append([g["mean"].iloc[-1] * 100, era, colr])
    ax.axhline(50, ls=(0, (4, 3)), lw=1.4, color=fs.MUTED, zorder=1)
    fs.label_ends(ax, ends, 10, min_gap=6)
    ax.set_xlabel("decile of the selector's own score")
    fs.axis_note(ax, "mean citation percentile")
    ax.set_xlim(0.5, 14.5); ax.set_xticks([1, 5, 10])
    ax.set_title("Dose–response", pad=34)
    fs.clean(ax)


def panel_c(ax, eras):
    for d, col, lbl in [(e[0], e[1], e[2]) for e in eras]:
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
    # dl.load already joins the human score and the area chair tier onto both eras
    d1, d2 = dl.load(tier_a_only)
    eras = [(d1, C_1820, "2018–2020"), (d2, C_2025, "2025")]

    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    panel_a(axes[0], eras)
    panel_b(axes[1], eras)
    panel_c(axes[2], eras)

    r = {e: {c: stats.spearmanr(d[c], d["cite_pct"])[0] for c, _, _, _ in SELECTORS}
         for d, _, e in eras}
    fs.title_block(
        fig,
        "Only the committee falls between eras — the humans it is judged against do not",
        f"ρ vs citation percentile.  2018–2020: committee {r['2018–2020']['rating']:.2f}, "
        f"reviewers {r['2018–2020']['mean_rating']:.2f}, area chairs "
        f"{r['2018–2020']['ac']:.2f}   ·   2025: committee {r['2025']['rating']:.2f}, "
        f"reviewers {r['2025']['mean_rating']:.2f}, area chairs {r['2025']['ac']:.2f}")

    handles = [plt.Line2D([], [], color=fs.MUTED, ls=s, lw=lw, label=lbl)
               for _, lbl, s, lw in SELECTORS]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize="small",
               bbox_to_anchor=(0.985, 1.0))
    fig.subplots_adjust(left=0.045, right=0.975, top=0.745, bottom=0.13, wspace=0.34)

    os.makedirs("outputs", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}  (n={len(d1):,} / {len(d2):,})")
    for e, v in r.items():
        print(f"  {e}: " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
