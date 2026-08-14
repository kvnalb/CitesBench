"""
Figures for the provisional era comparison (2018-2020 vs 2025).

Four panels, in the order the argument runs:
  A  recall@k across all k — the headline; how much of the true top-k the committee finds
  B  rating decile -> mean citation percentile — the same claim as a dose-response
  C  rating distributions — rules out "2025 ratings are more compressed" as the cause
  D  citation distributions — shows the age confound rather than hiding it

Panel D is the honest one: 2025 papers have ~18 months of citations against 6-8
years, so the outcome itself is coarser there. That attenuates any correlation
mechanically and is a live alternative to the leakage reading of panel A.

Run: python src/analysis/plot_era_comparison.py [--tier-a-only]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import compare_eras as ce

OUT_PNG = "outputs/era_comparison.png"
OUT_PDF = "outputs/era_comparison.pdf"

# validated categorical pair (light surface): worst adjacent CVD dE 24.7 protan
C_1820, C_2025 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
KGRID = np.arange(0.02, 0.51, 0.02)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 9, "legend.frameon": False,
    })


def recall_curve(d, rating_col="rating"):
    rng = np.random.default_rng(ce.SEED)
    r, c = d[rating_col].to_numpy(), d["cite_pct"].to_numpy()
    return np.array([ce.topk_recall(r, c, k, rng)[0] for k in KGRID])


def panel_a(ax, d1, d2):
    ax.plot(KGRID * 100, KGRID * 100, ls=(0, (4, 3)), lw=1.2, color=INK2, zorder=1)
    ax.annotate("random", (37, 33), color=INK2, fontsize=8, rotation=27, ha="left")
    for d, col, lbl in [(d1, C_1820, "2018–2020"), (d2, C_2025, "2025")]:
        y = recall_curve(d) * 100
        ax.plot(KGRID * 100, y, lw=2, color=col, zorder=3)
        ax.annotate(f"  {lbl}", (KGRID[-1] * 100, y[-1]), color=col, fontsize=9,
                    va="center", ha="left", fontweight="bold")
    ax.set_xlabel("% of accepted papers selected (k)")
    ax.set_ylabel("% of true top-k captured")
    ax.set_title("A  Recall@k — how much of the true top-k the committee finds", loc="left")
    ax.set_xlim(0, 66); ax.set_ylim(0, 72)
    ax.grid(True, alpha=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_b(ax, d1, d2):
    for d, col, lbl in [(d1, C_1820, "2018–2020"), (d2, C_2025, "2025")]:
        q = pd.qcut(d["rating"].rank(method="first"), 10, labels=False)
        g = d.groupby(q)["cite_pct"].agg(["mean", "sem"])
        x = np.arange(1, 11)
        ax.errorbar(x, g["mean"] * 100, yerr=g["sem"] * 100, lw=2, color=col,
                    marker="o", ms=5, capsize=2, elinewidth=1, label=lbl)
    ax.axhline(50, ls=(0, (4, 3)), lw=1.2, color=INK2, zorder=1)
    ax.set_xlabel("committee rating decile (1 = lowest)")
    ax.set_ylabel("mean citation percentile")
    ax.set_title("B  Dose–response: does a higher rating mean more citations?", loc="left")
    ax.set_xticks(range(1, 11)); ax.grid(True, alpha=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_c(ax, d1, d2):
    for d, col, lbl in [(d1, C_1820, "2018–2020"), (d2, C_2025, "2025")]:
        v = d["rating"]
        bins = np.arange(v.min() - 0.125, v.max() + 0.375, 0.25)
        ax.hist(v, bins=bins, weights=np.ones(len(v)) / len(v) * 100,
                histtype="step", lw=2, color=col, label=lbl)
    ax.set_xlabel("committee rating")
    ax.set_ylabel("% of papers")
    ax.set_title("C  Rating spread — 2025 is narrower (a third candidate cause)", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.annotate("2018–2020  sd 0.51, range 4.75–8.0\n2025          sd 0.38, range 5.0–7.0\n\n"
                "But clipping 2018–2020 to 5–7\nmoves ρ only 0.49 → 0.47,\n"
                "so range restriction is not\nwhat drives the gap.",
                xy=(0.985, 0.62), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.6, color=INK2, linespacing=1.5)
    ax.grid(True, alpha=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_d(ax, d1, d2):
    for d, col, lbl in [(d1, C_1820, "2018–2020"), (d2, C_2025, "2025")]:
        v = d["s2_citations"].clip(lower=0) + 1
        ax.hist(v, bins=np.logspace(0, 4, 40),
                weights=np.ones(len(v)) / len(v) * 100,
                histtype="step", lw=2, color=col, label=lbl)
        ax.axvline(v.median(), color=col, ls=(0, (2, 2)), lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("citations + 1  (log scale; dashed = median)")
    ax.set_ylabel("% of papers")
    ax.set_title("D  The confound: 2025 has ~18 months of citations, not 6–8 years",
                 loc="left")
    ax.grid(True, alpha=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(tier_a_only=False):
    tiers = ["A"] if tier_a_only else ["A", "B"]
    ev = pd.read_csv(ce.EVAL_1820, low_memory=False)
    acc = ev[ev["decision"].str.startswith("Accept", na=False)]
    d1 = ce.load_era(acc[["paper_id", "year", "committee_rating"]],
                     ce.TIER_1820, "committee_rating", tiers)
    r25 = pd.read_csv(ce.RATE_2025)
    d2 = ce.load_era(r25[["paper_id", "year", "rating"]],
                     ce.TIER_2025, "rating", tiers)

    style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    panel_a(axes[0, 0], d1, d2)
    panel_b(axes[0, 1], d1, d2)
    panel_c(axes[1, 0], d1, d2)
    panel_d(axes[1, 1], d1, d2)

    rho1 = stats.spearmanr(d1["rating"], d1["cite_pct"])[0]
    rho2 = stats.spearmanr(d2["rating"], d2["cite_pct"])[0]
    fig.suptitle("Can the LLM committee rank accepted ICLR papers by citation impact?",
                 x=0.008, y=0.983, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.008, 0.951,
             f"Spearman ρ {rho1:.2f} (2018–2020, n={len(d1):,})  vs  "
             f"{rho2:.2f} (2025, n={len(d2):,}).   Accepted papers only, both eras.   "
             f"Same instrument: gemma-4-31B-it, 8 LLM calls.",
             ha="left", fontsize=8.6, color=INK)
    fig.text(0.008, 0.925,
             f"PROVISIONAL — citation coverage is still unequal "
             f"({len(d1)/len(acc):.0%} vs {len(d2)/len(r25):.0%}); the title-match fetch "
             f"closes it. Magnitudes will move, direction is unlikely to.",
             ha="left", fontsize=8.2, color=INK2, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG} and {OUT_PDF}  (n={len(d1):,} / {len(d2):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
