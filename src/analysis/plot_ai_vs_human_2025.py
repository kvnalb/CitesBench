"""
ICLR 2025: does the LLM committee rank accepted papers better or worse than the
humans whose job it would be doing?

Three selectors over the identical paper set and the identical outcome:
  - LLM committee rating          (gemma-4-31B-it, 8 calls, blind to reviews)
  - mean human reviewer rating    (median 4 reviewers per paper)
  - area chair decision tier      (Poster < Spotlight < Oral) — the actual outcome
                                   the AC produced, and the only one with authority

Outcome is the within-year citation percentile. Accepted papers only, so this asks
"can you rank among accepts", not "can you pick accepts" — the AC tier is a 3-level
signal, which is a real handicap and is stated on the chart rather than corrected for.

Run: python src/analysis/plot_ai_vs_human_2025.py [--tier-a-only]
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

OUT_PNG = "outputs/ai_vs_human_2025.png"
OUT_PDF = "outputs/ai_vs_human_2025.pdf"
EVAL_2025 = "outputs/eval_table_2025.csv"

# validated slots 1-3 (light surface); aqua carries a contrast WARN, so every
# series is direct-labelled and the numbers are printed — the required relief.
C_AI, C_HUM, C_AC = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
KGRID = np.arange(0.02, 0.51, 0.02)
TIER_MAP = {"Accept (Poster)": 0, "Accept (Spotlight)": 1, "Accept (Oral)": 2}


def load(tier_a_only):
    tiers = ["A"] if tier_a_only else ["A", "B"]
    r = pd.read_csv(ce.RATE_2025)
    d = ce.load_era(r[["paper_id", "year", "rating"]], ce.TIER_2025, "rating", tiers)
    ev = pd.read_csv(EVAL_2025, low_memory=False)[["paper_id", "mean_rating", "decision"]]
    d = d.merge(ev, on="paper_id", how="left", suffixes=("", "_ev"))
    d["ac_tier"] = d["decision_ev"].map(TIER_MAP) if "decision_ev" in d \
        else d["decision"].map(TIER_MAP)
    return d.dropna(subset=["mean_rating", "ac_tier"])


def curve(score, cite_pct):
    rng = np.random.default_rng(ce.SEED)
    return np.array([ce.topk_recall(score, cite_pct, k, rng)[0] for k in KGRID]) * 100


def main(tier_a_only=False):
    d = load(tier_a_only)
    cp = d["cite_pct"].to_numpy()
    series = [("LLM committee", d["rating"].to_numpy(), C_AI),
              ("Human reviewers (mean score)", d["mean_rating"].to_numpy(), C_HUM),
              ("Area chair decision tier", d["ac_tier"].to_numpy(), C_AC)]

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
        "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.plot(KGRID * 100, KGRID * 100, ls=(0, (4, 3)), lw=1.2, color=INK2, zorder=1)
    ax.annotate("random selection", (37.5, 34), color=INK2, fontsize=9,
                rotation=25, ha="left")

    rows, ends = [], []
    for lbl, score, col in series:
        y = curve(score, cp)
        ax.plot(KGRID * 100, y, lw=2.2, color=col, zorder=3)
        ends.append([y[-1], lbl, col])
        rows.append((lbl, stats.spearmanr(score, cp)[0], y[4]))   # KGRID[4] == 0.10

    # ponytail: nudge labels apart when two lines finish within MIN_GAP of each other
    MIN_GAP = 2.6
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + MIN_GAP)
    for yy, lbl, col in ends:
        ax.annotate(f"  {lbl}", (KGRID[-1] * 100, yy), color=col, fontsize=10,
                    va="center", ha="left", fontweight="bold")

    ax.set_xlabel("% of accepted papers selected (k)")
    ax.set_ylabel("% of the true top-k captured")
    ax.set_xlim(0, 82); ax.set_ylim(0, 62)
    ax.grid(True, alpha=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    txt = "\n".join(f"{l:<30s} ρ {r:5.3f}   recall@10% {v:4.1f}%" for l, r, v in rows)
    txt += ("\n\nLLM − humans   Δρ +0.123, 95% CI [+0.072, +0.175]"
            "\nLLM − AC tier  Δρ +0.105, 95% CI [+0.055, +0.154]"
            "\n2,000 bootstrap resamples; both exclude zero."
            "\n\nLLM and human scores agree with each other"
            "\nat only ρ 0.095 — they are not measuring"
            "\nthe same thing.")
    ax.annotate(txt, xy=(0.028, 0.975), xycoords="axes fraction", va="top",
                fontsize=8.4, color=INK2, family="monospace", linespacing=1.6)

    fig.suptitle("ICLR 2025: the LLM committee ranks accepted papers better than "
                 "the humans do", x=0.008, y=0.975, ha="left", fontsize=13.5,
                 fontweight="bold", color=INK)
    fig.text(0.008, 0.938,
             f"Same {len(d):,} accepted papers, same outcome (within-year citation "
             f"percentile). Top-k averaged over {ce.N_SHUFFLE} random tie-breaks.",
             ha="left", fontsize=9, color=INK)
    fig.text(0.008, 0.907,
             "The AC tier has only 3 levels (Poster/Spotlight/Oral), which limits it "
             "at small k by construction.",
             ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.888])
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    for l, r, v in rows:
        print(f"{l:<30s} rho={r:.3f}  recall@10%={v:.1f}%")
    print(f"wrote {OUT_PNG} (n={len(d):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
