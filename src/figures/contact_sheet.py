"""
Contact sheet: every candidate panel, drawn small, on two pages.

This exists to be chosen from, not published. Page 1 characterises the sample.
Page 2 carries the candidate main analyses and the robustness cuts. Nothing here
is a new analysis — every panel reads outputs/eval_table.csv through spec.py, so
a panel promoted to a paper exhibit will not move when it is redrawn at full size.

WHY SELECTION PROBABILITY RATHER THAN A SLATE. Every panel that depends on which
papers a regime picked uses each paper's probability of selection over
spec.N_SHUFFLE tie orderings, not its membership in one of them. With 74-80% of
the single-call slate supplied by the tie-break, a single ordering would draw
curves that are substantially row order. Weighting by probability makes each
curve an expectation over everything the regime is indifferent between, which is
the same estimand Figure 2's brackets report.

The one exception is the overlap panel: expected overlap is not the product of
two marginal probabilities, so that panel resamples slates jointly.

Run: python src/figures/contact_sheet.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs   # noqa: E402

OUT_P1_PDF = "outputs/figures/contact_sheet_p1_sample.pdf"
OUT_P1_PNG = "outputs/figures/contact_sheet_p1_sample.png"
OUT_P2_PDF = "outputs/figures/contact_sheet_p2_analysis.pdf"
OUT_P2_PNG = "outputs/figures/contact_sheet_p2_analysis.png"
OUT_CSV = "outputs/figures/contact_sheet_panels.csv"

N_JOINT = 50          # joint slate resamples for the overlap panel
SMALL = "x-small"


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
    ax.bar(x, rej, 0.6, color="#d8d8d3", label="rejected", zorder=3)
    ax.bar(x, acc, 0.6, bottom=rej, color=fs.BLUE, label="accepted", zorder=3)
    for xi, (a, r) in enumerate(zip(acc, rej)):
        ax.annotate(f"{a/(a+r):.0%}", (xi, a + r), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=SMALL,
                    color=fs.INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(spec.YEARS, fontsize=SMALL)
    ax.legend(frameon=False, fontsize=SMALL, loc="upper left")
    ax.set_title("1. Pool composition (n pinned to accepts)", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


def p2_outcome_dist(ax, et):
    for lab, sub, col in [("accepted", et[et.accepted], fs.BLUE),
                          ("rejected", et[~et.accepted], "#b9b9b3")]:
        v = np.log1p(sub[spec.OUTCOME].dropna())
        ax.hist(v, bins=45, color=col, alpha=0.75, label=lab, zorder=3)
    ax.set_xlabel("log(1 + citations)", fontsize=SMALL, color=fs.MUTED)
    ax.legend(frameon=False, fontsize=SMALL)
    ax.set_title("2. The outcome is heavy-tailed in both arms", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


def p3_coverage(ax, et):
    x = np.arange(len(spec.YEARS)); w = 0.36
    for i, (lab, mask, col) in enumerate([("accepted", et.accepted, fs.BLUE),
                                          ("rejected", ~et.accepted, "#b9b9b3")]):
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
    ax.legend(frameon=False, fontsize=SMALL, loc="lower left")
    ax.set_title("3. Outcome coverage, by decision (the differential)",
                 fontsize=SMALL, color=fs.INK, fontweight="bold")


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
    ax.set_title("4. Score profile — flat runs are ties", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


def p5_completeness(ax, et):
    cols = [("citations", spec.OUTCOME), ("field label", "field"),
            ("human score", "mean_rating"), ("council score", "committee_rating"),
            ("single-call score", "single_call_rating")]
    y = np.arange(len(cols)); h = 0.36
    for i, (lab, mask, col) in enumerate([("accepted", et.accepted, fs.BLUE),
                                          ("rejected", ~et.accepted, "#b9b9b3")]):
        ax.barh(y + (i - 0.5) * h, [et[mask][c].notna().mean() for _, c in cols],
                h, color=col, label=lab, zorder=3)
    for yi, (_, c) in enumerate(cols):
        d = abs(et[et.accepted][c].notna().mean()
                - et[~et.accepted][c].notna().mean()) * 100
        ax.annotate(f"{d:.1f}pp", (1.02, yi), fontsize=SMALL, va="center",
                    color=fs.INK if d > 5 else fs.MUTED,
                    fontweight="bold" if d > 5 else "normal")
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cols], fontsize=SMALL)
    ax.set_xlim(0, 1.18); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v <= 1 else "")
    ax.legend(frameon=False, fontsize=SMALL, loc="lower left")
    ax.set_title("5. Completeness by decision — field is the outlier",
                 fontsize=SMALL, color=fs.INK, fontweight="bold")


def p6_score_vs_cites(ax, et):
    d = et.dropna(subset=["mean_rating", spec.OUTCOME])
    ax.hexbin(d.mean_rating, np.log1p(d[spec.OUTCOME]), gridsize=28,
              cmap="Blues", mincnt=1, linewidths=0)
    b = d.groupby(d.mean_rating.round())[spec.OUTCOME].median()
    ax.plot(b.index, np.log1p(b.values), color=fs.ORANGE, lw=2, marker="o", ms=3)
    ax.set_xlabel("mean human review score", fontsize=SMALL, color=fs.MUTED)
    ax.set_title("6. Raw signal: human score vs outcome", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


# ---------------------------------------------------- page 2: main + robustness
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
    ax.legend(frameon=False, fontsize=SMALL, loc="lower right")
    ax.set_title("A. Recall@k — the metric-choice problem, removed",
                 fontsize=SMALL, color=fs.INK, fontweight="bold")
    return rows


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
    ax.set_title("B. Survival of the selected set — overlap, then split",
                 fontsize=SMALL, color=fs.INK, fontweight="bold")


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
    ax.legend(frameon=False, fontsize=SMALL, loc="lower right")
    ax.set_title("C. Share of all citations captured", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


def pDE_rate_by_bin(ax, et, probs, col, q, xlab, title):
    b = bins_within_year(et, col, q)
    base = et.accepted.mean()
    ax.axhline(base, color=fs.MUTED, ls=(0, (4, 3)), lw=1.1, zorder=2)
    for r in spec.HEADLINE:
        p = probs[r.key].reindex(et.paper_id).to_numpy()
        ax.plot(range(1, q + 1), [p[(b == i).to_numpy()].mean() for i in range(1, q + 1)],
                marker="o", ms=3, lw=1.8, color=r.color, label=r.label.split(" (")[0])
    ax.set_xlabel(xlab, fontsize=SMALL, color=fs.MUTED)
    ax.set_xticks(range(1, q + 1)); ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(title, fontsize=SMALL, color=fs.INK, fontweight="bold")


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
    ax.set_title("F. Who picked what (ac / coun / sing)", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


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
    ax.set_title("G. Lift over the area chairs, by metric", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


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
    ax.set_title("H. Mean log by year — where the edge lives", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


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
    ax.set_title("I. Share of the slate decided by the tie-break", fontsize=SMALL,
                 color=fs.INK, fontweight="bold")


# ------------------------------------------------------------------- assembly
def build():
    os.makedirs("outputs/figures", exist_ok=True)
    et = spec.read_eval_table()
    probs = {r.key: selection_probability(et, r) for r in spec.HEADLINE}
    t1 = spec.read_table1()
    allrow = t1[t1.year.astype(str) == "all"].iloc[0]
    fs.apply("paper")

    # ------------------------------------------------------------- page 1
    fig1, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    p1_pool(ax[0, 0], et); p2_outcome_dist(ax[0, 1], et); p3_coverage(ax[0, 2], et)
    p4_score_profile(ax[1, 0], et); p5_completeness(ax[1, 1], et)
    p6_score_vs_cites(ax[1, 2], et)
    for a in ax.ravel():
        fs.clean(a)
    fs.title_block(
        fig1, "Contact sheet 1 of 2 — the sample",
        f"ICLR 2018-2020, all {int(allrow.submissions):,} submissions, accepts and "
        f"rejects. Outcome: Semantic Scholar citations, tier {'+'.join(spec.TIERS)}, "
        f"{allrow.cite_coverage:.1%} coverage.\nPanels are candidates, drawn small to "
        "be chosen from. Nothing here is a new analysis.", y=0.975)
    fs.source(fig1, y=0.012, text=(
        "Source: outputs/eval_table.csv via src/figures/spec.py. Panel 5 is the "
        "reason field normalization is excluded: the field label is 40% missing and "
        "its gap by decision is 6.0 pp, against 3.9 pp for the outcome itself."))
    fig1.subplots_adjust(left=0.05, right=0.97, top=0.85, bottom=0.09,
                         wspace=0.26, hspace=0.36)
    fig1.savefig(OUT_P1_PDF); fig1.savefig(OUT_P1_PNG, dpi=170); plt.close(fig1)

    # ------------------------------------------------------------- page 2
    # eval_results predates the single-call regime being registered, so panel G's
    # numbers come from spec.weighted_metric — the same weights and the same
    # per-year-then-average rule every other panel and Figure 2 use.
    rows = [{"regime": r.label, "metric": m,
             "value": spec.metric_over_orderings(et, r, m)[0]}
            for r in spec.HEADLINE
            for m in ["recall_at_1", "recall_at_5", "recall_at_10",
                      "median_citations", "mean_log_citations"]]
    g = pd.DataFrame(rows)

    fig2, ax = plt.subplots(3, 3, figsize=(15, 12.4))
    panel_rows = pA_recall_curve(ax[0, 0], et, probs)
    pB_survival(ax[0, 1], et, probs)
    pC_capture(ax[0, 2], et, probs)
    pDE_rate_by_bin(ax[1, 0], et, probs, spec.OUTCOME, 10,
                    "true citation decile (within year)",
                    "D. Selection rate by true citation decile")
    pDE_rate_by_bin(ax[1, 1], et, probs, "mean_rating", 5,
                    "human score quintile (within year)",
                    "E. Selection rate by human score quintile")
    pF_overlap(ax[1, 2], et)
    pG_metric_sensitivity(ax[2, 0], g)
    pH_per_year(ax[2, 1], et, probs)
    pI_resolution(ax[2, 2], et)
    for a in ax.ravel():
        fs.clean(a)
    fs.title_block(
        fig2, "Contact sheet 2 of 2 — main analyses (A-F) and robustness (G-I)",
        "Every panel weights papers by their probability of selection over 200 tie "
        "orderings, so no curve depends on one arbitrary ordering.\nF resamples "
        "slates jointly, because expected overlap is not the product of two "
        "marginal probabilities.", y=0.985)
    fs.source(fig2, y=0.010, text=(
        "Source: outputs/eval_table.csv via src/figures/spec.py, mode=raw. "
        "G recomputes its metrics from the same weights as every other panel: "
        "eval_results.csv predates the single-call regime being registered.\n"
        "H is the uncomfortable one — pooling hides that the council's mean-log "
        "edge sits in 2018 alone."))
    fig2.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.06,
                         wspace=0.26, hspace=0.48)
    fig2.savefig(OUT_P2_PDF); fig2.savefig(OUT_P2_PNG, dpi=170); plt.close(fig2)

    out = pd.concat([pd.DataFrame(panel_rows),
                     g.assign(panel="G")], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(g.pivot(index="regime", columns="metric", values="value").to_string())
    print(f"\n-> {OUT_P1_PNG}\n-> {OUT_P2_PNG}\n-> {OUT_CSV}")
    return g


def demo():
    g = build()
    p = g.pivot(index="regime", columns="metric", values="value")
    AC, CO = spec.BY_KEY["human_ac"].label, spec.BY_KEY["llm_council"].label
    # The metric ordering is what panel A and panel G exist to show. If the
    # council's edge stops shrinking as k widens, both panels lose their point.
    assert (p.loc[CO, "recall_at_1"] - p.loc[AC, "recall_at_1"]
            > p.loc[CO, "recall_at_10"] - p.loc[AC, "recall_at_10"]), \
        "council edge should be largest at the top of the distribution"
    for f in (OUT_P1_PNG, OUT_P2_PNG):
        assert os.path.getsize(f) > 50_000, f"{f} looks empty"

    # Panel G and Figure 2 must agree to the last decimal. They are computed by
    # different code paths on the same data, and an earlier version of this file
    # pooled the years where Figure 2 averaged them, giving a median of 111.0
    # against 123.2 without any error. That is the drift spec.py exists to stop.
    f2 = pd.read_csv("outputs/figures/fig2_headline.csv")
    for _, row in f2.iterrows():
        got = p.loc[row.regime, row.metric]
        assert abs(got - row.value) < 1e-6, (
            f"panel G disagrees with Figure 2 on {row.regime}/{row.metric}: "
            f"{got} vs {row.value}")
    print("\nok — 15 panels on 2 pages; panel G matches Figure 2 exactly")


if __name__ == "__main__":
    demo()
