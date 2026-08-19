"""
The candidate panels, one standalone figure each.

Every panel reads outputs/eval_table.csv through spec.py and is drawn at ICLR's
5.5in text width in the same style as the numbered exhibits, so any panel can be
promoted into the paper without being redrawn.

No titles, decks or source lines. Axis labels and legends only; the filename
carries the panel's identity and the caption is written in the LaTeX document.

WHY SELECTION PROBABILITY RATHER THAN A SLATE. Every panel that depends on which
papers a regime picked uses each paper's probability of selection over
spec.N_SHUFFLE tie orderings, not its membership in one of them. With 74-80% of
the single-call slate supplied by the tie-break, a single ordering would draw
curves that are substantially row order. Weighting by probability makes each
curve an expectation over everything the regime is indifferent between, which is
the same estimand Figure 2's brackets report.

The one exception is the overlap panel: expected overlap is not the product of
two marginal probabilities, so that panel resamples slates jointly.

Run: python src/figures/panels.py            (all panels)
     python src/figures/panels.py capture    (one panel)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs   # noqa: E402

PANEL_DIR = "outputs/figures/panels"
OUT_CSV = "outputs/figures/panels_metrics.csv"

N_JOINT = 50          # joint slate resamples for the overlap panel
# At 5.5in the bundle's own 7pt tick size is already small; "x-small" was for the
# oversized contact sheet and is unreadable at a single-panel width.
SMALL = "small"


# --------------------------------------------------------------- computation
def selection_probability(et, regime):
    """P(paper is selected) over spec.N_SHUFFLE tie orderings, per year."""
    prob = pd.Series(0.0, index=et["paper_id"])
    n_ord = 1
    for year in spec.YEARS:
        pool = et[et["year"] == year]
        n = spec.n_for(et, year)
        k = 0
        for k, sel in enumerate(spec.select_with_ties(pool, regime, n)):
            prob.loc[sel] += 1.0
        n_ord = k + 1
    return prob / n_ord


def joint_slates(et, regimes, k=N_JOINT):
    """k jointly-drawn slates per regime, so overlap counts are not assumed
    independent across regimes."""
    out = {r.key: [set() for _ in range(k)] for r in regimes}
    for year in spec.YEARS:
        pool = et[et["year"] == year]
        n = spec.n_for(et, year)
        for r in regimes:
            for i, sel in enumerate(spec.select_with_ties(pool, r, n, k)):
                out[r.key][i] |= set(sel)
            if r.score is None:                    # one slate, reused
                out[r.key] = [out[r.key][0]] * k
    return out


def bins_within_year(et, col, q):
    """Quantile bin within year. Rank first: citations tie heavily at low counts
    and qcut alone collapses to fewer bins than asked for."""
    out = pd.Series(np.nan, index=et.index)
    for year in spec.YEARS:
        m = (et["year"] == year) & et[col].notna()
        if m.sum() < q:
            continue
        out.loc[m] = pd.qcut(et.loc[m, col].rank(method="first"), q,
                             labels=False) + 1
    return out


# -------------------------------------------------------------- page 1: EDA
def p1_pool(ax, et):
    acc = [int(((et.year == y) & et.accepted).sum()) for y in spec.YEARS]
    rej = [int(((et.year == y) & ~et.accepted).sum()) for y in spec.YEARS]
    x = np.arange(len(spec.YEARS))
    ax.bar(x, rej, 0.6, color=fs.NEUTRAL, label="rejected", zorder=3)
    ax.bar(x, acc, 0.6, bottom=rej, color=fs.BLUE, label="accepted", zorder=3)
    for xi, (a, r) in enumerate(zip(acc, rej)):
        ax.annotate(f"{a/(a+r):.0%}", (xi, a + r), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=SMALL,
                    color=fs.INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(spec.YEARS, fontsize=SMALL)
    ax.legend(frameon=False, fontsize=SMALL, loc="upper left")

    ax.set_ylabel("Submissions", fontsize=SMALL, color=fs.MUTED)

def p2_outcome_dist(ax, et):
    for lab, sub, col in [("accepted", et[et.accepted], fs.BLUE),
                          ("rejected", et[~et.accepted], fs.NEUTRAL)]:
        v = np.log1p(sub[spec.OUTCOME].dropna())
        ax.hist(v, bins=45, color=col, alpha=0.75, label=lab, zorder=3)
    ax.set_xlabel("log(1 + citations)", fontsize=SMALL, color=fs.MUTED)
    ax.legend(frameon=False, fontsize=SMALL)

    ax.set_ylabel("Papers", fontsize=SMALL, color=fs.MUTED)

def p3_coverage(ax, et):
    x = np.arange(len(spec.YEARS)); w = 0.36
    for i, (lab, mask, col) in enumerate([("accepted", et.accepted, fs.BLUE),
                                          ("rejected", ~et.accepted, fs.NEUTRAL)]):
        v = [et[(et.year == y) & mask][spec.OUTCOME].notna().mean()
             for y in spec.YEARS]
        ax.bar(x + (i - 0.5) * w, v, w, color=col, label=lab, zorder=3)
    for xi, y in enumerate(spec.YEARS):
        s = et[et.year == y]
        d = abs(s[s.accepted][spec.OUTCOME].notna().mean()
                - s[~s.accepted][spec.OUTCOME].notna().mean()) * 100
        ax.annotate(f"{d:.1f}pp", (xi, 1.0), xytext=(0, 3), fontsize=SMALL,
                    textcoords="offset points", ha="center", color=fs.INK)
    ax.set_ylim(0.85, 1.06); ax.set_xticks(x)
    ax.set_xticklabels(spec.YEARS, fontsize=SMALL)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])   # else 0.875 renders as "88%"
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=SMALL, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), ncol=2)

    ax.set_ylabel("Share with a citation match", fontsize=SMALL, color=fs.MUTED)

def p4_score_profile(ax, et):
    """Sorted score, min-max scaled. Flat runs are ties; the cutoff is marked."""
    pool = et[et.year == 2020]
    n = spec.n_for(et, 2020)
    for r in spec.REGIMES:
        if r.score is None:
            continue
        s = pool[r.score].dropna().sort_values(ascending=False).to_numpy()
        y = (s - s.min()) / (s.max() - s.min())
        ax.plot(np.arange(len(y)) / len(y), y, lw=1.8, color=r.color,
                label=f"{r.label.split(' (')[0]} ({pool[r.score].nunique()} values)")
    ax.axvline(n / len(pool), color=fs.INK, ls=(0, (3, 2)), lw=1.1, zorder=5)
    ax.annotate("cutoff", (n / len(pool), 0.98), xytext=(4, 0), fontsize=SMALL,
                textcoords="offset points", color=fs.INK)
    ax.set_xlabel("paper rank (share of the 2020 pool)", fontsize=SMALL, color=fs.MUTED)
    ax.legend(frameon=False, fontsize=SMALL, loc="lower left")

    ax.set_ylabel("Score (min–max scaled)", fontsize=SMALL, color=fs.MUTED)

def p5_completeness(ax, et):
    cols = [("citations", spec.OUTCOME), ("field label", "field"),
            ("human score", "mean_rating"), ("council score", "committee_rating"),
            ("single-call score", "single_call_rating")]
    y = np.arange(len(cols)); h = 0.36
    for i, (lab, mask, col) in enumerate([("accepted", et.accepted, fs.BLUE),
                                          ("rejected", ~et.accepted, fs.NEUTRAL)]):
        ax.barh(y + (i - 0.5) * h, [et[mask][c].notna().mean() for _, c in cols],
                h, color=col, label=lab, zorder=3)
    for yi, (_, c) in enumerate(cols):
        d = abs(et[et.accepted][c].notna().mean()
                - et[~et.accepted][c].notna().mean()) * 100
        ax.annotate(f"{d:.1f}pp", (1.02, yi), fontsize=SMALL, va="center",
                    color=fs.INK if d > 5 else fs.MUTED,
                    fontweight="bold" if d > 5 else "normal")
    ax.legend(frameon=False, fontsize=SMALL, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), ncol=2)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cols], fontsize=SMALL)
    ax.set_xlim(0, 1.18); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v <= 1 else "")


def p6_score_vs_cites(ax, et):
    d = et.dropna(subset=["mean_rating", spec.OUTCOME])
    ax.hexbin(d.mean_rating, np.log1p(d[spec.OUTCOME]), gridsize=28,
              cmap="Blues", mincnt=1, linewidths=0)
    b = d.groupby(d.mean_rating.round())[spec.OUTCOME].median()
    ax.plot(b.index, np.log1p(b.values), color=fs.VERMILLION, lw=2, marker="o", ms=3)
    ax.set_xlabel("mean human review score", fontsize=SMALL, color=fs.MUTED)


# ---------------------------------------------------- page 2: main + robustness    ax.set_ylabel("log(1 + citations)", fontsize=SMALL, color=fs.MUTED)

def pA_recall_curve(ax, et, probs):
    ks = np.geomspace(0.004, 1.0, 90)
    rows = []
    for r in spec.HEADLINE:
        # spec.recall_at aggregates per year then averages, matching every other
        # exhibit. Pooling the three years instead moves the numbers materially.
        ys = [spec.recall_at(et, probs[r.key], k) for k in ks]
        ax.plot(ks, ys, lw=2, color=r.color, label=r.label.split(" (")[0])
        rows += [{"panel": "A", "regime": r.label, "k": k, "recall": y}
                 for k, y in zip(ks, ys)]
    ax.set_xscale("log"); ax.set_xlabel("k (top k% by citations)", fontsize=SMALL,
                                        color=fs.MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v*100:g}%")
    return rows

    ax.set_ylabel("Recall of the true top k%", fontsize=SMALL, color=fs.MUTED)

def pB_survival(ax, et, probs):
    grid = np.geomspace(1, 3000, 60)
    for r in spec.HEADLINE:
        d = et.dropna(subset=[spec.OUTCOME])
        w = probs[r.key].reindex(d.paper_id).fillna(0).to_numpy()
        c = d[spec.OUTCOME].to_numpy()
        ax.plot(grid, [(w * (c > g)).sum() / w.sum() for g in grid],
                lw=2, color=r.color, label=r.label.split(" (")[0])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("citations", fontsize=SMALL, color=fs.MUTED)
    ax.legend(frameon=False, fontsize=SMALL, loc="lower left")

    ax.set_ylabel("P(citations > x)", fontsize=SMALL, color=fs.MUTED)

def pC_capture(ax, et, probs):
    d = et.dropna(subset=[spec.OUTCOME]).sort_values(spec.OUTCOME, ascending=False)
    tot = d[spec.OUTCOME].sum()
    for r in spec.HEADLINE:
        w = probs[r.key].reindex(d.paper_id).fillna(0).to_numpy()
        ax.plot(np.cumsum(w), np.cumsum(w * d[spec.OUTCOME].to_numpy()) / tot,
                lw=2, color=r.color, label=r.label.split(" (")[0])
    ax.set_xlabel("papers admitted (ordered by true citations)", fontsize=SMALL,
                  color=fs.MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    ax.set_ylabel("Share of all citations captured", fontsize=SMALL, color=fs.MUTED)

def pDE_rate_by_bin(ax, et, probs, col, q, xlab):
    b = bins_within_year(et, col, q)
    base = et.accepted.mean()
    ax.axhline(base, color=fs.MUTED, ls=(0, (4, 3)), lw=1.1, zorder=2)
    for r in spec.HEADLINE:
        p = probs[r.key].reindex(et.paper_id).to_numpy()
        ax.plot(range(1, q + 1), [p[(b == i).to_numpy()].mean() for i in range(1, q + 1)],
                marker="o", ms=3, lw=1.8, color=r.color, label=r.label.split(" (")[0])
    ax.set_xlabel(xlab, fontsize=SMALL, color=fs.MUTED)
    ax.set_ylabel("Selection rate", fontsize=SMALL, color=fs.MUTED)
    ax.set_xticks(range(1, q + 1)); ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def pF_overlap(ax, et):
    js = joint_slates(et, spec.HEADLINE)
    keys = [r.key for r in spec.HEADLINE]
    regions, counts = [], []
    for mask in range(1, 8):
        inc = [keys[i] for i in range(3) if mask >> i & 1]
        exc = [k for k in keys if k not in inc]
        vals = []
        for i in range(N_JOINT):
            s = set.intersection(*[js[k][i] for k in inc])
            for k in exc:
                s -= js[k][i]
            vals.append(len(s))
        regions.append("+".join(k.split("_")[-1][:4] for k in inc))
        counts.append(np.mean(vals))
    order = np.argsort(counts)[::-1]
    ax.bar(range(7), [counts[i] for i in order], 0.62, color=fs.BLUE, zorder=3)
    for xi, i in enumerate(order):
        ax.annotate(f"{counts[i]:,.0f}", (xi, counts[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=SMALL)
    ax.set_xticks(range(7))
    ax.set_xticklabels([regions[i] for i in order], fontsize=SMALL, rotation=35,
                       ha="right")

    ax.set_ylabel("Papers", fontsize=SMALL, color=fs.MUTED)

def pG_metric_sensitivity(ax, res):
    """Lift over the area chairs, metric by metric. The ordering IS the finding."""
    metrics = ["recall_at_1", "recall_at_5", "recall_at_10", "median_citations",
               "mean_log_citations"]
    labels = ["recall@1%", "recall@5%", "recall@10%", "median cites", "mean log"]
    ac = res.set_index(["regime", "metric"])["value"]
    x = np.arange(len(metrics)); w = 0.36
    for i, r in enumerate([spec.BY_KEY["llm_council"], spec.BY_KEY["llm_single"]]):
        v = [(ac[(r.label, m)] - ac[(spec.BY_KEY["human_ac"].label, m)])
             / abs(ac[(spec.BY_KEY["human_ac"].label, m)]) for m in metrics]
        ax.bar(x + (i - 0.5) * w, v, w, color=r.color,
               label=r.label.split(" (")[0], zorder=3)
    ax.axhline(0, color=fs.INK, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=SMALL, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.legend(frameon=False, fontsize=SMALL)

    ax.set_ylabel("Lift over the area chairs", fontsize=SMALL, color=fs.MUTED)

def pH_per_year(ax, et, probs):
    """The fragility cut: the council's mean-log edge lives in 2018 alone."""
    x = np.arange(len(spec.YEARS)); w = 0.26
    for i, r in enumerate(spec.HEADLINE):
        v = []
        for year in spec.YEARS:
            d = et[(et.year == year)].dropna(subset=[spec.OUTCOME])
            wt = probs[r.key].reindex(d.paper_id).fillna(0).to_numpy()
            v.append((wt * np.log1p(d[spec.OUTCOME].to_numpy())).sum() / wt.sum())
        ax.bar(x + (i - 1) * w, v, w, color=r.color, label=r.label.split(" (")[0],
               zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(spec.YEARS, fontsize=SMALL)
    ax.set_ylim(4.0, 5.45)               # headroom for the legend
    ax.legend(frameon=False, fontsize=SMALL, ncol=3, loc="upper center")

    ax.set_ylabel("Mean log(1 + citations)", fontsize=SMALL, color=fs.MUTED)

def pI_resolution(ax, et):
    x = np.arange(len(spec.YEARS)); w = 0.26
    for i, r in enumerate(spec.HEADLINE):
        v = [0.0 if r.score is None else
             spec.resolution(et[et.year == y][r.score], spec.n_for(et, y))[1]
             / spec.n_for(et, y) for y in spec.YEARS]
        ax.bar(x + (i - 1) * w, v, w, color=r.color, label=r.label.split(" (")[0],
               zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(spec.YEARS, fontsize=SMALL)
    ax.set_ylim(0, 1.25)                 # headroom for the legend
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=SMALL, ncol=3, loc="upper center")


# ------------------------------------------------------------------- assembly
# Panel key -> (callable taking one Axes, height in inches). Height varies because
# a horizontal bar chart with five categories needs less than a scatter.    ax.set_ylabel("Share of slate set by tie-break", fontsize=SMALL, color=fs.MUTED)

def panel_specs(et, probs, g):
    return {
        "pool":            (lambda ax: p1_pool(ax, et), 2.8),
        "outcome_dist":    (lambda ax: p2_outcome_dist(ax, et), 2.8),
        "coverage":        (lambda ax: p3_coverage(ax, et), 3.0, 0.13, 0.52, 0.34),
        "score_profile":   (lambda ax: p4_score_profile(ax, et), 3.0),
        "completeness":    (lambda ax: p5_completeness(ax, et), 2.9, 0.24, 0.45, 0.34),
        "score_vs_cites":  (lambda ax: p6_score_vs_cites(ax, et), 3.0),
        "recall_curve":    (lambda ax: pA_recall_curve(ax, et, probs), 3.0),
        "survival":        (lambda ax: pB_survival(ax, et, probs), 3.0),
        "capture":         (lambda ax: pC_capture(ax, et, probs), 3.0),
        "rate_by_citation_decile":
            (lambda ax: pDE_rate_by_bin(ax, et, probs, spec.OUTCOME, 10,
                                        "True citation decile (within year)"), 3.0),
        "rate_by_score_quintile":
            (lambda ax: pDE_rate_by_bin(ax, et, probs, "mean_rating", 5,
                                        "Human score quintile (within year)"), 3.0),
        "overlap":         (lambda ax: pF_overlap(ax, et), 3.1, 0.13),
        "metric_lift":     (lambda ax: pG_metric_sensitivity(ax, g), 3.2, 0.15, 0.85),
        "mean_log_by_year": (lambda ax: pH_per_year(ax, et, probs), 2.8),
        "tie_share":       (lambda ax: pI_resolution(ax, et), 2.8),
    }


def build(only=None):
    os.makedirs(PANEL_DIR, exist_ok=True)
    et = spec.read_eval_table()
    probs = {r.key: selection_probability(et, r) for r in spec.HEADLINE}

    # Panel G's metrics come from spec.metric_over_orderings — the same weights and
    # the same per-year-then-average rule Figure 2 uses, so the two cannot disagree.
    g = pd.DataFrame([{"regime": r.label, "metric": m,
                       "value": spec.metric_over_orderings(et, r, m)[0]}
                      for r in spec.HEADLINE
                      for m in ["recall_at_1", "recall_at_5", "recall_at_10",
                                "median_citations", "mean_log_citations"]])
    g.to_csv(OUT_CSV, index=False)

    specs = panel_specs(et, probs, g)
    if only:
        missing = [k for k in only if k not in specs]
        if missing:
            raise SystemExit(f"unknown panel(s) {missing}; known: {sorted(specs)}")
        specs = {k: specs[k] for k in only}

    written = []
    for key, sp in specs.items():
        draw, height = sp[0], sp[1]
        left = sp[2] if len(sp) > 2 else 0.13
        bottom = sp[3] if len(sp) > 3 else 0.52
        top = sp[4] if len(sp) > 4 else 0.12
        fs.apply()                       # 5.5in, ICLR type sizes
        fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN, height))
        draw(ax)
        fs.clean(ax, xgrid=(key in ("score_vs_cites", "capture")))
        fs.frame(fig, top_in=top, bottom_in=bottom, left=left, right=0.98)
        base = os.path.join(PANEL_DIR, key)
        fig.savefig(base + ".pdf")
        fig.savefig(base + ".png", dpi=200)
        plt.close(fig)
        written.append(base + ".pdf")

    for w in written:
        print(f"-> {w}")
    print(f"-> {OUT_CSV}")
    return g


def demo():
    g = build()
    p = g.pivot(index="regime", columns="metric", values="value")
    AC, CO = spec.BY_KEY["human_ac"].label, spec.BY_KEY["llm_council"].label

    # The metric ordering is what the recall curve and the lift panel exist to
    # show. If the council's edge stops shrinking as k widens, both lose their point.
    assert (p.loc[CO, "recall_at_1"] - p.loc[AC, "recall_at_1"]
            > p.loc[CO, "recall_at_10"] - p.loc[AC, "recall_at_10"]), \
        "council edge should be largest at the top of the distribution"

    # Panel metrics and Figure 2 come from different code paths over the same data
    # and must agree exactly — an earlier version pooled the years where Figure 2
    # averaged them, giving a median of 111.0 against 123.2 with no error raised.
    f2 = pd.read_csv("outputs/figures/fig2_headline.csv")
    for _, row in f2.iterrows():
        got = p.loc[row.regime, row.metric]
        assert abs(got - row.value) < 1e-6, (
            f"panel metrics disagree with Figure 2 on {row.regime}/{row.metric}: "
            f"{got} vs {row.value}")

    n = len([f for f in os.listdir(PANEL_DIR) if f.endswith(".pdf")])
    assert n == 15, f"expected 15 panels, found {n}"
    print(f"\nok — {n} panels, each 5.5in; metrics match Figure 2")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        build(sys.argv[1:])
    else:
        demo()
