"""
Figure 2: the headline comparison — the 9-call council against a 1-call baseline
and the area chairs, on median citations and mean log(1 + citations).

Everything is computed here from outputs/eval_table.csv. The earlier version read
its point estimates from outputs/eval_results.csv while computing its intervals
itself, which had two consequences worth naming:

  1. `eval_results.csv` keys two incommensurable scales off one `value` column —
     mode='raw' holds citation counts (184.0), mode='normalized' holds percentile
     ranks (0.75) — so an unfiltered read averages them into plausible nonsense.
     That already cost one wrong conclusion during planning, silently.
  2. The point came from ONE arbitrary tie ordering while the bracket spanned 200,
     so the point did not sit inside its own interval in any principled place.

Both go away by computing the point and the interval from the same 200 orderings.
The figure now depends on one input, and spec.py declares what that input means.

WHY THE BARS ARE NOT CONFIDENCE INTERVALS. They are the identified set. When a
regime's scores are coarse, it does not pick out one slate — it picks out every
slate consistent with its ranking, and it is indifferent between them. In 2020 the
single-call baseline ranks 147 papers strictly above its cutoff and is indifferent
over 813 more, from which the harness must draw the remaining 540. Reporting a
single number there would read that indifference as a decision the model made.

So: the point is the mean over N_SHUFFLE orderings, and the bracket is the full
range. The council's bracket is narrow because the council actually resolves the
decision; the single call's is wide because it mostly does not. That contrast is
the finding, not a nuisance to be averaged away.

Rejected: a deterministic tie-break on paper_id or a secondary score. It makes a
false attribution reproducible, which is worse than leaving it visible.

Run: python src/figures/fig2_headline.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs   # noqa: E402
from metrics import compute_metrics        # noqa: E402
from baselines import random_baseline      # noqa: E402

OUT_PDF = "outputs/figures/fig2_headline.pdf"
OUT_PNG = "outputs/figures/fig2_headline.png"
OUT_CSV = "outputs/figures/fig2_headline.csv"

METRICS = [("median_citations", "Median citations of the selected papers"),
           ("mean_log_citations", "Mean log(1 + citations)")]

# Display may wrap a label across two lines; the CSV always carries spec's
# canonical string, so exhibits stay joinable on `regime`.
WRAP = {"Human (area chairs)": "Human\n(area chairs)",
        "LLM council (9 calls)": "LLM council\n(9 calls)",
        "LLM single call (1 call)": "LLM single call\n(1 call)"}


def metric_over_orderings(et, regime, metric):
    """Point, interval and resolution for one regime on one metric.

    Each of N_SHUFFLE orderings produces a slate per year; the metric is averaged
    across years within an ordering, so the spread reported is the spread of the
    year-averaged headline rather than of any single year.
    """
    per_year_streams = []
    for year in spec.YEARS:
        pool = et[et["year"] == year]
        n = spec.n_for(et, year)
        vals = [compute_metrics(sel, pool, spec.MODE)[metric]
                for sel in spec.select_with_ties(pool, regime, n)]
        per_year_streams.append(vals)

    # A decision-based regime yields one slate, not N_SHUFFLE of them.
    width = min(len(v) for v in per_year_streams)
    across = np.array([v[:width] for v in per_year_streams], dtype=float).mean(axis=0)
    point, lo, hi = spec.point_and_interval(across)
    if width == 1:                      # no ties to be indifferent over
        lo = hi = np.nan
    return point, lo, hi


def build():
    os.makedirs("outputs/figures", exist_ok=True)
    et = spec.read_eval_table()

    # Random baseline: regime-independent, so computed once per year and averaged.
    rand = {m: float(np.mean([random_baseline(et[et.year == y], spec.n_for(et, y),
                                              spec.MODE)[m] for y in spec.YEARS]))
            for m, _ in METRICS}

    recs = []
    for metric, _ in METRICS:
        for r in spec.HEADLINE:
            point, lo, hi = metric_over_orderings(et, r, metric)
            # resolution, averaged over years, so the caption can quote it
            sup = np.mean([spec.resolution(et[et.year == y][r.score],
                                           spec.n_for(et, y))[1] / spec.n_for(et, y)
                           for y in spec.YEARS]) if r.score else 0.0
            recs.append({"metric": metric, "regime": r.label, "key": r.key,
                         "value": point, "tie_lo": lo, "tie_hi": hi,
                         "random": rand[metric],
                         "tie_broken": r.score is not None,
                         "share_tie_broken": sup})
    res = pd.DataFrame(recs)
    res.to_csv(OUT_CSV, index=False)

    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (metric, unit) in zip(axes, METRICS):
        sub = res[res.metric == metric].set_index("key").loc[[r.key for r in spec.HEADLINE]]
        x = np.arange(len(spec.HEADLINE))
        ax.bar(x, sub.value, width=0.62, color=[r.color for r in spec.HEADLINE], zorder=3)

        # The identified set, drawn as a range through the bar rather than a
        # symmetric error bar, because it is neither symmetric nor a standard error.
        for xi, (_, row) in zip(x, sub.iterrows()):
            if not np.isnan(row.tie_lo):
                ax.vlines(xi, row.tie_lo, row.tie_hi, color=fs.INK, lw=1.6, zorder=5)
                ax.hlines([row.tie_lo, row.tie_hi], xi - 0.11, xi + 0.11,
                          color=fs.INK, lw=1.6, zorder=5)

        rnd = sub["random"].iloc[0]
        ax.axhline(rnd, color=fs.MUTED, ls=(0, (4, 3)), lw=1.2, zorder=4)
        ax.annotate("random baseline", (len(spec.HEADLINE) - 0.45, rnd), va="bottom",
                    ha="right", fontsize="x-small", color=fs.MUTED)

        ax.set_xticks(x)
        ax.set_xticklabels([WRAP.get(r.label, r.label) for r in spec.HEADLINE],
                           fontsize="small")
        fs.axis_note(ax, unit)
        fs.clean(ax)
        # Two decimals on the log panel: at 4.79 vs 4.79 the point IS that they
        # are the same number.
        fmt = ((lambda v: f"{v:.2f}") if metric == "mean_log_citations"
               else (lambda v: f"{v:,.0f}"))
        for xi, (_, row) in zip(x, sub.iterrows()):
            top = row.value if np.isnan(row.tie_hi) else max(row.value, row.tie_hi)
            ax.annotate(fmt(row.value), (xi, top), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize="small",
                        fontweight="bold", color=fs.INK)
        ax.set_ylim(0, np.nanmax([sub.value.max(), sub.tie_hi.max()]) * 1.22)

    t1 = spec.read_table1()
    allrow = t1[t1.year.astype(str) == "all"].iloc[0]
    single = res[res.key == "llm_single"].iloc[0]

    fs.title_block(
        fig, "The council resolves the decision; one call mostly does not",
        f"ICLR 2018-2020, all {int(allrow.submissions):,} submissions. Every regime "
        "selects exactly n papers, n = that year's actual accept count.\n"
        "Bars are the mean across years and across 200 tie-break orderings; brackets "
        "span the full range of those orderings.\nBrackets are the identified set, "
        "NOT confidence intervals — a wide one means the regime was indifferent.")
    fs.source(
        fig, y=0.012,
        text=f"Source: outputs/eval_table.csv via src/figures/spec.py (mode={spec.MODE}). "
             f"Outcome: Semantic Scholar citations, tier {'+'.join(spec.TIERS)}, "
             f"{allrow.cite_coverage:.1%} of the pool, "
             f"{allrow.coverage_differential_pp:.1f} pp accept/reject differential.\n"
             f"The single-call baseline has {single.share_tie_broken:.0%} of its slate "
             "supplied by the tie-break rather than by its own score; the council has "
             f"{res[res.key == 'llm_council'].iloc[0].share_tie_broken:.0%}.")
    fig.subplots_adjust(left=fs.LEFT, right=0.98, top=0.72, bottom=0.20, wspace=0.22)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    print(res.to_string(index=False))
    print(f"\n-> {OUT_PDF}\n-> {OUT_PNG}\n-> {OUT_CSV}")
    return res


def demo():
    res = build()
    med = res[res.metric == "median_citations"].set_index("key")

    # The area chairs made every one of their own decisions, so they have no
    # identified set. Anything else would be a fabricated interval.
    assert np.isnan(med.loc["human_ac", "tie_lo"]), "ACs should have no tie interval"

    # Resolution ordering is the figure's mechanism and must hold before any
    # claim about the levels is worth reading.
    assert (med.loc["llm_single", "share_tie_broken"]
            > med.loc["llm_council", "share_tie_broken"]), "resolution not ordered"

    # The point must sit inside its own identified set — the bug this rewrite fixed.
    for key in ["llm_council", "llm_single"]:
        lo, hi, v = (med.loc[key, "tie_lo"], med.loc[key, "tie_hi"],
                     med.loc[key, "value"])
        assert lo <= v <= hi, f"{key}: point {v} outside its interval [{lo}, {hi}]"

    assert (med["value"] > med["random"]).all(), "a regime failed to beat random"

    print(f"\nok — median citations: "
          + ", ".join(f"{k} {med.loc[k, 'value']:.1f}" for k in med.index)
          + f"; single-call slate {med.loc['llm_single', 'share_tie_broken']:.0%} "
            "tie-broken vs council "
            f"{med.loc['llm_council', 'share_tie_broken']:.0%}")


if __name__ == "__main__":
    demo()
