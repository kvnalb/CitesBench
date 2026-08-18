"""
ICLR 2025: does the LLM committee rank accepted papers better or worse than the
humans whose job it would be doing?

Three selectors over the identical paper set and the identical outcome:
  - LLM committee rating          (gemma-4-31B-it, 8 calls, blind to the reviews)
  - mean human reviewer rating    (median 4 reviewers per paper)
  - area chair decision tier      (Poster < Spotlight < Oral) — the outcome with
                                   actual authority over the paper

Outcome is the within-year citation percentile. Accepted papers only, so this asks
"can you rank among accepts", not "can you pick accepts" — the AC tier is a 3-level
signal, a real handicap that is stated on the chart rather than corrected for.

Run: python src/analysis/plot_ai_vs_human_2025.py [--tier-a-only]
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
from figures import figstyle as fs
import compare_eras as ce

OUT_PNG = "outputs/ai_vs_human_2025.png"
OUT_PDF = "outputs/ai_vs_human_2025.pdf"
EVAL_2025 = "outputs/eval_table_2025.csv"

KGRID = np.arange(0.02, 0.51, 0.02)
TIER_MAP = {"Accept (Poster)": 0, "Accept (Spotlight)": 1, "Accept (Oral)": 2}
N_BOOT = 2000


def load(tier_a_only):
    tiers = ["A"] if tier_a_only else ["A", "B"]
    r = pd.read_csv(ce.RATE_2025)
    d = ce.load_era(r[["paper_id", "year", "rating"]], ce.TIER_2025, "rating", tiers)
    ev = pd.read_csv(EVAL_2025, low_memory=False)[["paper_id", "mean_rating", "decision"]]
    d = d.merge(ev, on="paper_id", how="left", suffixes=("", "_ev"))
    col = "decision_ev" if "decision_ev" in d else "decision"
    d["ac_tier"] = d[col].map(TIER_MAP)
    return d.dropna(subset=["mean_rating", "ac_tier"])


def curve(score, cite_pct):
    rng = np.random.default_rng(ce.SEED)
    return np.array([ce.topk_recall(score, cite_pct, k, rng)[0] for k in KGRID]) * 100


def boot_delta(a, b, cp, rng):
    """Bootstrap CI on the difference of two Spearman rhos over the same papers."""
    n = len(cp)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = rng.integers(0, n, n)
        diffs[i] = stats.spearmanr(a[ix], cp[ix])[0] - stats.spearmanr(b[ix], cp[ix])[0]
    return diffs.mean(), *np.percentile(diffs, [2.5, 97.5])


def main(tier_a_only=False):
    d = load(tier_a_only)
    cp = d["cite_pct"].to_numpy()
    series = [("LLM committee", d["rating"].to_numpy(), fs.BLUE),
              ("Human reviewers", d["mean_rating"].to_numpy(), fs.ORANGE),
              ("Area chair tier", d["ac_tier"].to_numpy(), fs.AQUA)]

    fs.apply()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(KGRID * 100, KGRID * 100, ls=(0, (4, 3)), lw=1.4, color=fs.MUTED, zorder=1)
    ax.annotate("random", (44, 40), color=fs.MUTED, fontsize="small",
                rotation=23, ha="left")

    ends, rows = [], []
    for lbl, score, col in series:
        y = curve(score, cp)
        ax.plot(KGRID * 100, y, lw=2.6, color=col, solid_capstyle="round", zorder=3)
        rho = stats.spearmanr(score, cp)[0]
        ends.append([y[-1], f"{lbl}   ρ {rho:.2f}", col])
        rows.append((lbl, rho, y[4]))                 # KGRID[4] == 0.10
    fs.label_ends(ax, ends, KGRID[-1] * 100, min_gap=3.0)

    ax.set_xlabel("% of papers selected")
    fs.axis_note(ax, "% of true top-k captured")
    ax.set_xlim(0, 76); ax.set_ylim(0, 62)
    ax.set_xticks(range(0, 51, 10))    # no ticks past where the data stops
    fs.clean(ax)

    fs.title_block(fig, "Ranking ICLR 2025 papers by citation impact")
    fig.subplots_adjust(left=fs.LEFT, right=0.80, top=0.845, bottom=0.115)

    os.makedirs("outputs", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    for l, r, v in rows:
        print(f"{l:<18s} rho={r:.3f}  recall@10%={v:.1f}%")
    print(f"wrote {OUT_PNG} (n={len(d):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
