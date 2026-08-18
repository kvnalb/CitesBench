"""
Figures for the era comparison (ICLR 2018-2020 vs 2025).

Layout is small multiples, not one crowded panel. Era x selector is a 2x3 crossing;
drawing all six series together and splitting them by colour AND line style made
four of them illegible in the 2025 bunch. So the panel carries the era and colour
carries the selector, nothing carries two things at once, and no panel holds more
than three lines.

  A  slope: rho for each selector, 2018-2020 -> 2025. The whole argument in one
     panel — only the committee falls.
  B  recall@k, 2018-2020        } shared y-axis, so the two eras are
  C  recall@k, 2025             } directly comparable by eye
  D  the committee's lead over the humans on all six measures, both eras

The humans are the control: they face the identical outcome within each era, so
whatever the shorter citation window costs a selector, it costs them too. That is
why they belong in the figure at all.

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
from figures import figstyle as fs
import compare_eras as ce
import did_leakage as dl

OUT_PNG = "outputs/era_comparison.png"
OUT_PDF = "outputs/era_comparison.pdf"

KGRID = np.arange(0.02, 0.51, 0.02)
# colour = selector, and nothing else
SELECTORS = [("rating", "LLM committee", fs.BLUE),
             ("mean_rating", "Human reviewers", fs.ORANGE),
             ("ac", "Area chairs", fs.AQUA)]
ERA_1820, ERA_2025 = "2018–2020", "2025"


def recall_curve(d, col):
    rng = np.random.default_rng(ce.SEED)
    r, c = d[col].to_numpy(float), d["cite_pct"].to_numpy()
    return np.array([ce.topk_recall(r, c, k, rng)[0] for k in KGRID]) * 100


def panel_slope(ax, d1, d2):
    """One line per selector across the two eras. Three lines, no overlap."""
    x = [0, 1]
    ends = []
    for col, lbl, colr in SELECTORS:
        y = [stats.spearmanr(d[col], d["cite_pct"])[0] for d in (d1, d2)]
        ax.plot(x, y, lw=2.6, color=colr, marker="o", ms=7, solid_capstyle="round",
                zorder=3, clip_on=False)
        # above the point, not left of it — left collided with the y tick labels
        ax.annotate(f"{y[0]:.2f}", (0, y[0]), xytext=(0, 11),
                    textcoords="offset points", ha="center", va="bottom",
                    color=colr, fontsize="small", fontweight="bold")
        ends.append([y[1], f"{lbl}  {y[1]:.2f}", colr])
    fs.label_ends(ax, ends, 1, min_gap=0.035)
    ax.set_xlim(-0.08, 2.05); ax.set_ylim(0, 0.56)   # right half is label space
    ax.set_xticks(x); ax.set_xticklabels([ERA_1820, ERA_2025])
    fs.axis_note(ax, "Spearman ρ vs citation percentile")
    ax.set_title("Ranking skill across eras", pad=34)
    fs.clean(ax)


def panel_recall(ax, d, era, n, show_note):
    ax.plot(KGRID * 100, KGRID * 100, ls=(0, (4, 3)), lw=1.4, color=fs.MUTED, zorder=1)
    ends = []
    for col, lbl, colr in SELECTORS:
        y = recall_curve(d, col)
        ax.plot(KGRID * 100, y, lw=2.6, color=colr, solid_capstyle="round", zorder=3)
        ends.append([y[-1], lbl, colr])
    fs.label_ends(ax, ends, KGRID[-1] * 100, min_gap=5.5)
    ax.annotate("random", (34, 31), color=fs.MUTED, fontsize="small",
                rotation=24, ha="left")
    ax.set_xlabel("% of papers selected")
    if show_note:
        fs.axis_note(ax, "% of true top-k captured")
    ax.set_xlim(0, 96); ax.set_ylim(0, 72)
    ax.set_xticks(range(0, 51, 10))
    ax.set_title(f"Recall@k · {era}  (n={n:,})", pad=34)
    fs.clean(ax)


DID_CSV = "outputs/metric_suite_did.csv"
METRIC_NAMES = {"recall_at_10": "Recall@10%", "spearman_rho": "Spearman ρ",
                "auc_top10": "Top-decile AUC", "kendall_tau_b": "Kendall τ-b",
                "ndcg_at_10": "NDCG@10%", "somers_d": "Somers' D"}


def panel_advantage(ax, control="human reviewers"):
    """Dumbbell: how far the committee leads the humans, per measure, per era.

    Era is encoded by marker fill rather than colour — colour already means
    selector everywhere else in this figure, and reusing it here would make the
    same hue mean two different things."""
    if not os.path.exists(DID_CSV):
        ax.set_axis_off()
        ax.set_title("Advantage by measure — run metric_suite.py", pad=34)
        return
    d = pd.read_csv(DID_CSV)
    d = d[d["control"] == control].copy()
    d["name"] = d["metric"].map(METRIC_NAMES)
    d = d.sort_values("adv_1820")
    y = np.arange(len(d))

    for yy, a, b in zip(y, d["adv_1820"], d["adv_2025"]):
        ax.plot([b, a], [yy, yy], color="#c9c9c4", lw=2.4, zorder=1,
                solid_capstyle="round")
    ax.scatter(d["adv_2025"], y, s=68, color="white", edgecolor=fs.ORANGE,
               linewidth=2.2, zorder=3)
    ax.scatter(d["adv_1820"], y, s=68, color=fs.BLUE, edgecolor="white",
               linewidth=1.2, zorder=3)
    ax.axvline(0, color=fs.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)

    ax.set_yticks(y); ax.set_yticklabels(d["name"])
    ax.set_xlim(-0.03, 0.46); ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("committee's lead over human reviewers")
    ax.annotate("2018–2020", (d["adv_1820"].max(), len(d) - 0.35), color=fs.BLUE,
                fontsize="small", fontweight="bold", ha="center", va="bottom")
    ax.annotate("2025", (d["adv_2025"].min(), len(d) - 0.35), color=fs.ORANGE,
                fontsize="small", fontweight="bold", ha="center", va="bottom")
    ax.set_title("The lead shrinks on every measure", pad=34)
    fs.clean(ax)
    ax.yaxis.grid(False); ax.xaxis.grid(True)


def main(tier_a_only=False):
    d1, d2 = dl.load(tier_a_only)          # already carries mean_rating and ac

    fs.apply()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_slope(axes[0, 0], d1, d2)
    panel_advantage(axes[0, 1])
    panel_recall(axes[1, 0], d1, ERA_1820, len(d1), True)
    panel_recall(axes[1, 1], d2, ERA_2025, len(d2), False)

    fs.title_block(fig, "Performance comparison across eras")
    fig.subplots_adjust(left=0.075, right=0.955, top=0.855, bottom=0.058,
                        wspace=0.30, hspace=0.40)

    os.makedirs("outputs", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}  (n={len(d1):,} / {len(d2):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
