"""
Figure 3: where each regime's selection comes from.

Table 2 says the council and the area chairs are indistinguishable on average. This
asks the obvious follow-up — whether they are indistinguishable *everywhere*, or
whether they trade off across the distribution and cancel.

Two panels, both showing the share of papers each regime selects:

  left    by human score quintile, within year — where the regimes disagree with
          each other on the input they both had
  right   by TRUE citation decile, within year — where they disagree with the
          outcome. This is the panel that matters: decile 10 is the papers that
          turned out to matter most, and selection rate there is recall.

The dashed line is each year's accept rate (~33%), the rate a regime that carried no
information at all would hit in every bin. Height above that line is skill, and the
SHAPE across bins is the mechanism.

WHY THESE AXES AND NOT AUTHOR COVARIATES. The obvious heterogeneity cut — team size,
h-index, institution, industry — is unusable here. Those covariates resolve for 71%
of papers, and the 71% is selected on the outcome: papers WITH covariates have a
41.6% accept rate and median 42 citations, papers without have 12.9% and median 2, a
28.7 pp gap. Splitting on them would condition the comparison on a variable that
already encodes the answer, which is the same error as excluding "memorized" papers
whose exclusion criterion correlates with citations. Field is no better: 40% missing
and levels from n=70 to n=1,749.

The two axes used here are 99.7% (mean_rating) and 96.3% (citations) covered, and
neither is a proxy for the thing being measured.

Deciles are computed WITHIN year: a 2018 paper has had two more years to accumulate
citations, so a pooled decile would mostly sort by year.

Run: python src/figures/fig3_heterogeneity.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs

OUT_PDF = "outputs/figures/fig3_heterogeneity.pdf"
OUT_PNG = "outputs/figures/fig3_heterogeneity.png"
OUT_CSV = "outputs/figures/fig3_heterogeneity.csv"

YEARS = list(spec.YEARS)
SERIES = spec.HEADLINE
# Axis labels only. No title, deck or source line: captions belong in the LaTeX
# document where the author controls them, not baked into the PDF.
PANELS = [("score_q", "mean_rating", 5, "Human score quintile (within year)",
           "Selection rate"),
          ("cite_d", spec.OUTCOME, 10, "True citation decile (within year)",
           "Selection rate")]


def bins_within_year(et, col, q):
    """Quantile bin within year, so a bin means the same thing across years."""
    out = pd.Series(np.nan, index=et.index)
    for yr in YEARS:
        m = (et.year == yr) & et[col].notna()
        if m.sum() < q:
            continue
        # rank first: citations are heavily tied at low counts and qcut alone
        # collapses to fewer bins than asked for
        r = et.loc[m, col].rank(method="first")
        out.loc[m] = pd.qcut(r, q, labels=False) + 1
    return out


def build():
    os.makedirs("outputs/figures", exist_ok=True)
    et = spec.read_eval_table()
    for key, col, q, _, _ in PANELS:
        et[key] = bins_within_year(et, col, q)

    # Selection PROBABILITY per paper, not membership in one arbitrary slate.
    # With 77% of the single-call slate decided by the tie-break, a single ordering
    # would draw a curve that is mostly an artifact of row order. Averaging the
    # membership indicator over spec.N_SHUFFLE orderings gives each paper its
    # probability of selection, and the bin mean is then an expected rate.
    for r in SERIES:
        prob = pd.Series(0.0, index=et.paper_id)
        n_ord = 0
        for yr in YEARS:
            p = et[et.year == yr]
            for k, sel in enumerate(spec.select_with_ties(p, r, spec.n_for(et, yr))):
                prob.loc[sel] += 1.0
            n_ord = k + 1
        et[r.key] = (prob / n_ord).to_numpy()

    recs = []
    for key, _, q, _, _ in PANELS:
        for b in range(1, q + 1):
            d = et[et[key] == b]
            row = {"panel": key, "bin": b, "n": len(d)}
            for r in SERIES:
                row[r.label] = d[r.key].mean()
            recs.append(row)
    res = pd.DataFrame(recs)
    res.to_csv(OUT_CSV, index=False)
    base = et.accepted.mean()

    fs.apply(ncols=2)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    for ax, (key, _, q, xlab, note) in zip(axes, PANELS):
        sub = res[res.panel == key]
        ax.axhline(base, color=fs.MUTED, ls=(0, (4, 3)), lw=1.2, zorder=2)
        for r in SERIES:
            ax.plot(sub.bin, sub[r.label], marker="o", ms=5, lw=2,
                    color=r.color, label=r.label, zorder=3)
        ax.annotate(f"accept rate ({base:.0%})", (q, base), xytext=(0, -4),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=plt.rcParams["font.size"] * 0.8, color=fs.INK, zorder=6,
                    path_effects=[patheffects.withStroke(linewidth=2.2,
                                                         foreground="white")])
        ax.set_xlabel(xlab, fontsize="small", color=fs.MUTED)
        ax.set_xticks(range(1, q + 1))
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylabel(note)
        fs.clean(ax)
    axes[0].legend(frameon=False, fontsize="small", loc="upper left")

    fs.frame(fig, top_in=0.10, bottom_in=0.46, left=0.09, right=0.99,
             wspace=0.32)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    with pd.option_context("display.width", 200):
        print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {OUT_PDF}\n-> {OUT_PNG}\n-> {OUT_CSV}")
    return res, base


def demo():
    res, base = build()
    cd = res[res.panel == "cite_d"].set_index("bin")
    top = cd.loc[10]
    AC, CO = spec.BY_KEY["human_ac"].label, spec.BY_KEY["llm_council"].label
    SC = spec.BY_KEY["llm_single"].label
    assert (top[[AC, CO]] > base).all(), \
        "both should beat the base rate in the top decile"
    assert (cd[CO].iloc[-1] > cd[CO].iloc[0]), "no gradient"
    # the mechanism: council ahead in the top decile, area chairs ahead in the middle.
    # If this crossover disappears, Figure 3 no longer explains Table 2's null and the
    # title is wrong — so it fails here rather than redrawing with a stale claim.
    assert top[CO] > top[AC], "council should lead at the top"
    mid = cd.loc[6:9]
    assert (mid[AC] > mid[CO]).all(), \
        "area chairs should lead deciles 6-9"
    sq = res[res.panel == "score_q"]
    assert len(sq) == 5 and len(cd) == 10
    print(f"\nok — top citation decile: AC {top[AC]:.0%}, "
          f"council {top[CO]:.0%}, single {top[SC]:.0%} "
          f"(base rate {base:.0%})")


if __name__ == "__main__":
    demo()
