"""
Figure 2: the headline comparison — council vs a naive single-prompt LLM vs the
area chairs, on median citations and mean log(1+citations).

Point estimates come from outputs/eval_results.csv, which run_eval.py writes. That
file keys two incommensurable scales off one `value` column — mode='raw' holds
citation counts (184.0) and mode='normalized' holds percentiles (0.75) — so reading
it without filtering silently averages them into plausible nonsense (their mean,
92.4, looks like a real citation count). We filter to raw and then ASSERT that only
one mode survived, because that averaging already cost one wrong conclusion during
planning and produced no error when it did.

WHY THE ERROR BARS ARE NOT CONFIDENCE INTERVALS. They are tie-break intervals, and
they exist because the naive LLM scores take six distinct values across 4,508
papers. In 2020 exactly TWO papers sit strictly above its selection cutoff; the
other 685 of 687 are drawn from a 1,330-way tie at the cutoff. So its "selection"
is mostly whatever order the rows happened to be in — shuffling ties moves its 2018
median citations from 63.0 to 109.5, a 74% swing from nothing but sort order.

A bare bar would hide that behind a single confident number. Each bar therefore
carries the min-max over N_SHUFFLE tie orderings. The council's interval is tight
(178-200 on the same test) and the naive one is wide, which is the honest picture:
one of these is a measurement and the other is close to a coin flip.

The naive bar is a PLACEHOLDER. LLM2 (ensemble, 13 distinct values, ~50% of its
slate tie-broken) is the least bad of three weak options — LLM1 is ~100% tie-broken
— and it should be replaced by the single-call baseline from #34, which produces a
real score distribution on the council's own schema.

Run: python src/figures/fig2_headline.py
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

EVAL_RESULTS = "outputs/eval_results.csv"
EVAL_TABLE = "outputs/eval_table.csv"
OUT_PDF = "outputs/figures/fig2_headline.pdf"
OUT_PNG = "outputs/figures/fig2_headline.png"
OUT_CSV = "outputs/figures/fig2_headline.csv"

YEARS = [2018, 2019, 2020]
N_SHUFFLE = 200
SEED = 0

# (regime object, label, score column or None for a decision-based regime, colour)
PANEL = [
    (HumanActual(), "Human\n(area chairs)", None, fs.BLUE),
    (LLMCommittee(), "LLM council\n(9 calls)", "committee_rating", fs.AQUA),
    (LLMEnsemble(), "Naive LLM\n(1 prompt)", "llm_mean_rating", fs.ORANGE),
]
METRICS = [("median_citations", "Median citations of the selected papers"),
           ("mean_log_citations", "Mean log(1 + citations)")]


def load_points():
    """Per-regime, per-metric point estimate and baselines, averaged over years."""
    d = pd.read_csv(EVAL_RESULTS)
    d = d[(d["mode"] == "raw") & d.year.isin(YEARS)]
    assert d["mode"].nunique() == 1, "raw filter failed — see the docstring"
    return d.groupby(["regime", "metric"])[
        ["value", "random_value", "ideal_value"]].mean().reset_index()


def tie_interval(regime, col, metric, et, rng):
    """min-max of the metric over N_SHUFFLE tie orderings, averaged across years.

    A decision-based regime (the area chairs) has no score to tie on — its selected
    set is fixed — so it gets a degenerate interval rather than a fake one.
    """
    if col is None:
        return None
    vals = []
    for _ in range(N_SHUFFLE):
        per_year = []
        for yr in YEARS:
            pool = et[et.year == yr]
            n = int(pool.accepted.sum())
            shuffled = pool.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
            sel = regime.select(shuffled, n)
            c = pool[pool.paper_id.isin(sel)].openalex_citations.dropna()
            per_year.append(c.median() if metric == "median_citations"
                            else np.log1p(c).mean())
        vals.append(np.mean(per_year))
    return float(np.min(vals)), float(np.max(vals))


def build():
    os.makedirs("outputs/figures", exist_ok=True)
    pts = load_points()
    et = pd.read_csv(EVAL_TABLE, low_memory=False)
    et = et[et.year.isin(YEARS)].copy()
    et["accepted"] = et.decision.str.startswith("Accept", na=False)
    rng = np.random.default_rng(SEED)

    recs = []
    for metric, _ in METRICS:
        for regime, label, col, colour in PANEL:
            row = pts[(pts.regime == regime.name) & (pts.metric == metric)]
            assert len(row) == 1, f"{regime.name}/{metric} not in {EVAL_RESULTS}"
            lo_hi = tie_interval(regime, col, metric, et, rng)
            recs.append({
                "metric": metric, "regime": regime.name, "label": label.replace("\n", " "),
                "value": float(row.value.iloc[0]),
                "random": float(row.random_value.iloc[0]),
                "ideal": float(row.ideal_value.iloc[0]),
                "tie_lo": lo_hi[0] if lo_hi else np.nan,
                "tie_hi": lo_hi[1] if lo_hi else np.nan,
                "tie_broken": col is not None,
            })
    res = pd.DataFrame(recs)
    res.to_csv(OUT_CSV, index=False)

    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (metric, unit) in zip(axes, METRICS):
        sub = res[res.metric == metric].set_index("regime").loc[[p[0].name for p in PANEL]]
        x = np.arange(len(PANEL))
        colours = [p[3] for p in PANEL]
        ax.bar(x, sub.value, width=0.62, color=colours, zorder=3)

        # tie-break spread, drawn as a range through the bar rather than a symmetric
        # error bar, because it is not symmetric and is not a standard error
        for xi, (_, r) in zip(x, sub.iterrows()):
            if not np.isnan(r.tie_lo):
                ax.vlines(xi, r.tie_lo, r.tie_hi, color=fs.INK, lw=1.6, zorder=5)
                ax.hlines([r.tie_lo, r.tie_hi], xi - 0.11, xi + 0.11,
                          color=fs.INK, lw=1.6, zorder=5)

        rnd = sub["random"].iloc[0]
        ax.axhline(rnd, color=fs.MUTED, ls=(0, (4, 3)), lw=1.2, zorder=4)
        ax.annotate("random baseline", (len(PANEL) - 0.45, rnd), va="bottom", ha="right",
                    fontsize="x-small", color=fs.MUTED)

        ax.set_xticks(x)
        ax.set_xticklabels([p[1] for p in PANEL], fontsize="small")
        fs.axis_note(ax, unit)
        fs.clean(ax)
        # label above the bracket, not the bar, or the two collide. Two decimals on
        # the log panel: at 4.79 vs 4.79 the point IS that they are the same number.
        fmt = (lambda v: f"{v:.2f}") if metric == "mean_log_citations" else (lambda v: f"{v:,.0f}")
        for xi, (_, r) in zip(x, sub.iterrows()):
            top = r.value if np.isnan(r.tie_hi) else max(r.value, r.tie_hi)
            ax.annotate(fmt(r.value), (xi, top), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize="small",
                        fontweight="bold", color=fs.INK)
        ax.set_ylim(0, np.nanmax([sub.value.max(), sub.tie_hi.max()]) * 1.22)

    fs.title_block(
        fig, "The council matches the area chairs; one naive prompt is far behind",
        "ICLR 2018-2020, all 4,567 submissions. Every regime selects exactly n papers, "
        "n = that year's actual accept count.\nThe council leads on median citations "
        "under every tie-break ordering, but on mean log the two are the same number "
        "(4.79 vs 4.79).\nBars are the mean across years; brackets span 200 tie-break "
        "orderings and are NOT confidence intervals.")
    fs.source(fig, y=0.012, text="Source: outputs/eval_results.csv (mode=raw), outputs/eval_table.csv. "
                   "Outcome: Semantic Scholar citations, tier A+B, 96.3% of the pool.\n"
                   "The naive LLM bar is a placeholder: its scores take 13 distinct values, "
                   "so ~50% of its slate is decided by tie-break rather than by score.")
    fig.subplots_adjust(left=fs.LEFT, right=0.98, top=0.72, bottom=0.20, wspace=0.22)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    print(res.to_string(index=False))
    print(f"\n-> {OUT_PDF}\n-> {OUT_PNG}\n-> {OUT_CSV}")
    return res


def demo():
    res = build()
    med = res[res.metric == "median_citations"].set_index("regime")
    council = med.loc["LLM Committee (Gemma)", "value"]
    human = med.loc["Human (AC decisions)", "value"]
    naive = med.loc["LLM2 (ensemble)", "value"]
    assert council > human > naive, (council, human, naive)
    assert (med["value"] > med["random"]).all(), "a regime failed to beat random"
    # the point of the figure: the naive interval is wide, the council's is not
    w = lambda r: med.loc[r, "tie_hi"] - med.loc[r, "tie_lo"]
    assert w("LLM2 (ensemble)") > w("LLM Committee (Gemma)"), "tie spread not ordered"
    assert np.isnan(med.loc["Human (AC decisions)", "tie_lo"]), "ACs have no ties"
    # median: the AC point sits BELOW the council's worst tie ordering, so the lead
    # survives tie-break uncertainty. mean log: it sits INSIDE, so it does not.
    assert human < med.loc["LLM Committee (Gemma)", "tie_lo"], "median lead not robust"
    ml = res[res.metric == "mean_log_citations"].set_index("regime")
    h, lo, hi = (ml.loc["Human (AC decisions)", "value"],
                 ml.loc["LLM Committee (Gemma)", "tie_lo"],
                 ml.loc["LLM Committee (Gemma)", "tie_hi"])
    assert lo < h < hi, f"expected the mean-log difference to be indistinguishable ({h}, {lo}, {hi})"
    print(f"ok — council {council:.1f} > AC {human:.1f} > naive {naive:.1f}; "
          f"tie spread naive {w('LLM2 (ensemble)'):.1f} vs council "
          f"{w('LLM Committee (Gemma)'):.1f}")


if __name__ == "__main__":
    demo()
