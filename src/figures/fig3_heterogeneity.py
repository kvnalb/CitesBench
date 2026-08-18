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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import figstyle as fs
from regimes.human_actual import HumanActual
from regimes.llm_committee import LLMCommittee
from regimes.llm_ensemble import LLMEnsemble

EVAL_TABLE = "outputs/eval_table.csv"
OUT_PDF = "outputs/figures/fig3_heterogeneity.pdf"
OUT_PNG = "outputs/figures/fig3_heterogeneity.png"
OUT_CSV = "outputs/figures/fig3_heterogeneity.csv"

YEARS = [2018, 2019, 2020]
SERIES = [(HumanActual(), "Human (area chairs)", fs.BLUE),
          (LLMCommittee(), "LLM council", fs.AQUA),
          (LLMEnsemble(), "Naive LLM", fs.ORANGE)]
PANELS = [("score_q", "mean_rating", 5, "Human score quintile (within year)",
           "Share of papers selected, by what the humans scored them"),
          ("cite_d", "openalex_citations", 10, "True citation decile (within year)",
           "Share of papers selected, by how the paper actually did")]


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
    et = pd.read_csv(EVAL_TABLE, low_memory=False)
    et = et[et.year.isin(YEARS)].copy()
    et["accepted"] = et.decision.str.startswith("Accept", na=False)
    for key, col, q, _, _ in PANELS:
        et[key] = bins_within_year(et, col, q)

    for regime, label, _ in SERIES:
        ids = []
        for yr in YEARS:
            p = et[et.year == yr]
            ids += regime.select(p, int(p.accepted.sum()))
        et[regime.name] = et.paper_id.isin(ids)

    recs = []
    for key, _, q, _, _ in PANELS:
        for b in range(1, q + 1):
            d = et[et[key] == b]
            row = {"panel": key, "bin": b, "n": len(d)}
            for regime, label, _ in SERIES:
                row[label] = d[regime.name].mean()
            recs.append(row)
    res = pd.DataFrame(recs)
    res.to_csv(OUT_CSV, index=False)
    base = et.accepted.mean()

    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for ax, (key, _, q, xlab, note) in zip(axes, PANELS):
        sub = res[res.panel == key]
        ax.axhline(base, color=fs.MUTED, ls=(0, (4, 3)), lw=1.2, zorder=2)
        for regime, label, colour in SERIES:
            ax.plot(sub.bin, sub[label], marker="o", ms=5, lw=2,
                    color=colour, label=label, zorder=3)
        ax.annotate(f"accept rate ({base:.0%})", (q, base), xytext=(0, -14),
                    textcoords="offset points", ha="right", va="top",
                    fontsize="x-small", color=fs.MUTED)
        ax.set_xlabel(xlab, fontsize="small", color=fs.MUTED)
        ax.set_xticks(range(1, q + 1))
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        fs.axis_note(ax, note)
        fs.clean(ax)
    axes[0].legend(frameon=False, fontsize="small", loc="upper left")

    fs.title_block(
        fig, "The area chairs hold the middle; the council catches the top decile",
        "ICLR 2018-2020, all 4,567 submissions. Each regime selects exactly n papers "
        "per year, n = that year's accept count.\nIn the top citation decile the "
        "council reaches 85% against the area chairs' 72%; from deciles 4 to 9 the "
        "area chairs are ahead.\nThe two cancel, which is why the average contrast in "
        "Table 2 is indistinguishable from zero. Bins are within year.")
    fs.source(fig, y=0.012, text=(
        "Source: outputs/eval_table.csv. Outcome: Semantic Scholar citations, tier A+B.\n"
        "Author covariates are deliberately not used as a cut: they resolve for 71% of "
        "papers, and that 71% has a 41.6% accept rate against 12.9% for the rest."))
    fig.subplots_adjust(left=fs.LEFT, right=0.98, top=0.76, bottom=0.16, wspace=0.20)
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
    assert (top[["Human (area chairs)", "LLM council"]] > base).all(), \
        "both should beat the base rate in the top decile"
    assert (cd["LLM council"].iloc[-1] > cd["LLM council"].iloc[0]), "no gradient"
    # the mechanism: council ahead in the top decile, area chairs ahead in the middle.
    # If this crossover disappears, Figure 3 no longer explains Table 2's null and the
    # title is wrong — so it fails here rather than redrawing with a stale claim.
    assert top["LLM council"] > top["Human (area chairs)"], "council should lead at the top"
    mid = cd.loc[6:9]
    assert (mid["Human (area chairs)"] > mid["LLM council"]).all(), \
        "area chairs should lead deciles 6-9"
    sq = res[res.panel == "score_q"]
    assert len(sq) == 5 and len(cd) == 10
    print(f"\nok — top citation decile: AC {top['Human (area chairs)']:.0%}, "
          f"council {top['LLM council']:.0%}, naive {top['Naive LLM']:.0%} "
          f"(base rate {base:.0%})")


if __name__ == "__main__":
    demo()
