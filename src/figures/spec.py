"""
Every design choice the paper's exhibits depend on, declared once.

Before this module `YEARS` was declared in four files, `SEED` in two, and the
regime list was assembled three different ways with three different label
strings. Joining fig2's CSV to fig3's on `regime` returned nothing, because one
said "LLM council (9 calls)" and the other said "LLM council". Copies drift;
this file exists so there are no copies.

The rationale lives here beside the values rather than in a FIGURES.md, because
a separate document rots and a docstring on the constant cannot be read without
seeing the constant.


WHY THESE VALUES
----------------

`YEARS = (2018, 2019, 2020)`
    The primary sample. All 4,567 submissions, accepts and rejects. 2025 is an
    appendix arm and is deliberately not reachable from this module: its
    `citation_pct_rank` is ranked within year with no field split while this
    era's is ranked within field x year, its citation column holds OpenAlex
    counts while this era's holds S2 under an OpenAlex-era name, and its
    rejected papers have 40% citation coverage against 98.5% for accepts. Any
    exhibit that spans both eras has to reconcile those three things explicitly.

`MODE = "raw"`
    Raw citation counts, not field x year percentile ranks. Field labels cover
    59.7% of the pool and the gap correlates with the decision (63.7% of accepts
    carry a label against 57.7% of rejects), so normalized mode silently drops
    1,930 papers along an axis that is not independent of the outcome. It also
    rescales n, which makes the two modes different selection problems on
    different populations. `read_eval_results` enforces the filter because an
    unfiltered `pivot_table` over both modes once averaged a median of 184.0
    with a percentile of 0.75 and reported 92.4.

`TIERS = ("A", "B")`
    A is an arXiv or DOI id match. B is title similarity >= 0.95 inside a year
    window with a shared author surname. C is excluded. This rule gives a 3.9 pp
    accept/reject coverage differential; the OpenAlex pull it replaced ran 26.3
    pp, which is disqualifying for a benchmark whose whole claim is about
    selecting from a mixed pool.

`N_SHUFFLE = 200`, `SEED = 0`
    Tie orderings averaged for a point estimate. See `select_with_ties`.

Unmatched papers stay NaN everywhere and are never imputed as zero. A paper we
could not find is not a paper with no citations.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figures import figstyle as fs  # noqa: E402

# ----------------------------------------------------------------- constants
YEARS = (2018, 2019, 2020)
SEED = 0
N_SHUFFLE = 200          # tie orderings averaged for a point estimate
N_BOOT = 400             # bootstrap draws for Table 2
MODE = "raw"
TIERS = ("A", "B")

# n per year is the actual accept count, pinned so every regime solves the same
# problem. Asserted against the table in check_inputs.py rather than trusted.
N_PINNED = {2018: 337, 2019: 502, 2020: 687}

# ---------------------------------------------------------------------- paths
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_abs = lambda rel: os.path.join(REPO, rel)

EVAL_TABLE = _abs("outputs/eval_table.csv")
EVAL_RESULTS = _abs("outputs/eval_results.csv")
CITATIONS = _abs("outputs/citations.csv")
FIG_DIR = _abs("outputs/figures")
TABLE1_CSV = os.path.join(FIG_DIR, "table1_sample.csv")

# The outcome column. Named for OpenAlex, holds S2 counts. Renaming it touches
# 122 references across 26 files and is its own PR (#38, out of scope); every
# exhibit goes through this constant so the eventual rename is one line here.
OUTCOME = "s2_citations"


# --------------------------------------------------------------- regime table
class Regime:
    """A selection regime as the exhibits see it.

    `score` is the column ranked to build the slate, or None for a regime that
    is not score-based. `HumanActual` reads a decision already made, so it has
    no score, no ties, and no interval — it is a point by construction.
    """

    def __init__(self, key, label, color, score, note):
        self.key, self.label, self.color = key, label, color
        self.score, self.note = score, note

    def __repr__(self):
        return f"Regime({self.key!r})"


# CSV output always uses `.label`. A figure may wrap it across two lines for
# layout; the data never carries a different string. This is the rule that
# stops fig2 and fig3 from disagreeing about what a regime is called.
#
# Colour: Okabe-Ito slots, assigned here rather than in figstyle so the palette
# and its use stay separable. Measured OKLab dE x100 for these four, worst case
# across protan/deutan/tritan simulation:
#
#     AC / council            24.0        council / single call   11.5
#     AC / single call        10.5        AC / diagnostic         12.1
#
# Worst pair overall is 8.2 (single call vs the diagnostic purple, protan),
# against a target of 8. The three headline regimes are the canonical
# blue/vermillion/bluish-green trio, which is the best-separated triple in the
# palette and the one Okabe & Ito recommend when only three are needed.
REGIMES = [
    Regime("human_ac", "Human (area chairs)", fs.BLUE, None,
           "the decision actually made; no score, so no ties"),
    Regime("llm_council", "LLM council (9 calls)", fs.VERMILLION, "committee_rating",
           "Gemma-4-31B, 9-call slim_coarse pipeline"),
    Regime("llm_single", "LLM single call (1 call)", fs.BLUISHGREEN, "single_call_rating",
           "same model, same schema, one call — the council's control"),
    Regime("human_score", "Human (mean review score)", fs.REDDISHPURPLE, "mean_rating",
           "top-n by mean reviewer rating, ignoring the AC; diagnostic only"),
]

# Values that may appear in an exhibit's `regime` column without naming a regime.
# Table 2 reports a council-minus-AC contrast as its own row; it is a derived
# quantity, not a selector, and check_inputs must not read it as label drift.
RESERVED_LABELS = {"contrast"}

BY_KEY = {r.key: r for r in REGIMES}
# The regimes that carry the paper's argument. Fig 2 and Table 2 use this subset;
# human_score is a diagnostic that belongs in the appendix.
# The area chairs, the 9-call council, and the 1-call control. The canonical
# blue / vermillion / bluish-green trio, worst CVD separation 11.5 (OKLab dE x100,
# worst of protan/deutan/tritan).
#
# The GENAI_REVIEW persona ratings (llm_neutral / positive / negative / mean) are
# deliberately absent from this module and from eval_table. They came with
# data/gen_review.db rather than from this project's pipeline, so their prompts,
# model and inputs are not ours to describe or defend. The regime classes are in
# Archive/regimes_gen_review/.
HEADLINE = [BY_KEY["human_ac"], BY_KEY["llm_council"], BY_KEY["llm_single"]]


# ------------------------------------------------------------------- readers
def read_eval_table():
    """The pool. 2018-2020, accepts and rejects, one row per paper."""
    et = pd.read_csv(EVAL_TABLE, low_memory=False)
    et = et[et["year"].isin(YEARS)].copy()
    assert not et["paper_id"].duplicated().any(), "duplicate paper_id in eval_table"
    et["accepted"] = et["decision"].str.startswith("Accept", na=False)
    return et


def read_eval_results():
    """Regime x year x metric results, filtered to one mode.

    The assert is the point of the function. `eval_results.csv` stacks raw
    citation counts and normalized percentile ranks in the same `value` column
    under different `mode` values, so any groupby that forgets the filter
    averages 184.0 with 0.75. That produced a convincing false finding once.
    """
    d = pd.read_csv(EVAL_RESULTS)
    before = d["mode"].nunique()
    d = d[d["mode"] == MODE]
    assert len(d), f"no rows with mode == {MODE!r} in {EVAL_RESULTS}"
    assert d["mode"].nunique() == 1, "mode filter let more than one mode through"
    if before > 1:
        d = d.drop(columns=["mode"])
    return d


def read_table1():
    """Table 1, so captions quote numbers rather than hardcoding them."""
    if not os.path.exists(TABLE1_CSV):
        raise SystemExit(
            f"{TABLE1_CSV} missing — run python src/figures/table1_sample.py first")
    return pd.read_csv(TABLE1_CSV)


def n_for(et, year):
    """n for a year: the accept count, checked against the pinned value."""
    n = int(et[(et["year"] == year) & et["accepted"]].shape[0])
    assert n == N_PINNED[year], f"{year}: accept count {n} != pinned {N_PINNED[year]}"
    return n


# --------------------------------------------------------- ties as an estimand
def resolution(scores, n):
    """How much of a slate the regime actually decided.

    Returns (own, supplied, tied_at_cutoff). `own` is the papers ranked strictly
    above the cutoff score — decisions the regime made. `supplied` is the
    remainder of the slate, filled from among papers tied at the cutoff by
    whatever the tie-break happens to be.

    A regime with 813 papers tied at the 2020 cutoff did not select 687 papers.
    It selected 147 and expressed indifference over 813. Reporting the point
    estimate without this number reads that indifference as a decision.
    """
    s = pd.Series(scores).dropna()
    if len(s) < n:
        return len(s), 0, 0
    cut = s.nlargest(n).iloc[-1]
    own = int((s > cut).sum())
    return own, n - own, int((s == cut).sum())


def select_with_ties(pool, regime, n, n_shuffle=N_SHUFFLE, seed=SEED):
    """Every slate the regime is indifferent between, as `n_shuffle` orderings.

    Yields `n_shuffle` lists of paper_id. A deterministic tie-break by paper_id
    or a secondary score was the first proposal here and it is wrong: when 813
    papers tie it invents 540 decisions from row order and attributes them to
    the regime, which makes a false attribution reproducible. Randomizing over
    ties leaves the indifference visible, so the spread across orderings is the
    regime's resolution rather than noise to be averaged away.

    A regime with no score (`HumanActual`) yields its single slate once —
    there is nothing to be indifferent about.
    """
    if regime.score is None:
        yield pool.loc[pool["accepted"], "paper_id"].tolist()
        return

    d = pool.dropna(subset=[regime.score])
    scores = d[regime.score].to_numpy(dtype=float)
    ids = d["paper_id"].to_numpy()
    rng = np.random.default_rng(seed)
    for _ in range(n_shuffle):
        order = np.lexsort((rng.random(len(scores)), -scores))
        yield ids[order[:n]].tolist()


def point_and_interval(values):
    """Point estimate and identified set from one metric over tie orderings.

    The point is the mean across orderings. The interval is the full range, and
    it is published as identification rather than sampling noise: with a coarse
    score the regime does not pick out one number, it picks out a set, and the
    width of that set is a property of the regime's resolution.
    """
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.min()), float(v.max())


def metric_over_orderings(et, regime, metric, n_shuffle=N_SHUFFLE):
    """Point, interval and resolution for one regime on one metric.

    THE AGGREGATION RULE LIVES HERE, and every exhibit calls this rather than
    reimplementing it. Two decisions it fixes:

    1. Mean of the per-year values, never a pooled statistic over the three years
       stacked together. They differ materially — pooling the median gives 111.0
       where the mean of per-year medians gives 123.2 — because the years differ
       in size and in citation age.

    2. Mean over tie orderings of the per-ordering metric, NOT the metric of a
       probability-weighted pool. For a metric linear in the selection indicator
       (recall, mean log) the two are identical; for the median they are not,
       because E[median] is not the median of the expectation. Computing the
       median the weighted way gave 123.0 against this function's 123.2 with no
       error raised, which is the kind of quiet disagreement spec.py exists to
       prevent.

    Returns (point, lo, hi). A regime with no score yields one slate, so its
    interval is NaN rather than a fabricated width.
    """
    from metrics import compute_metrics       # local: avoids a circular import

    streams = []
    for year in YEARS:
        pool = et[et["year"] == year]
        n = n_for(et, year)
        streams.append([compute_metrics(sel, pool, MODE)[metric]
                        for sel in select_with_ties(pool, regime, n, n_shuffle)])

    width = min(len(v) for v in streams)
    across = np.array([v[:width] for v in streams], dtype=float).mean(axis=0)
    point, lo, hi = point_and_interval(across)
    return (point, np.nan, np.nan) if width == 1 else (point, lo, hi)


def recall_at(et, prob, k):
    """Expected recall of the true top-k fraction, mean of the per-year values.

    Recall is linear in the selection indicator, so weighting papers by their
    selection probability is exactly equal to averaging over tie orderings, and
    far cheaper. That equivalence does not hold for the median — see
    metric_over_orderings.
    """
    per_year = []
    for year in YEARS:
        d = et[et["year"] == year].dropna(subset=[OUTCOME])
        m = max(1, int(round(k * len(d))))
        top = d.nlargest(m, OUTCOME)["paper_id"]
        per_year.append(prob.reindex(top).fillna(0).sum() / m)
    return float(np.mean(per_year))


def fingerprint():
    """Everything a cached artifact must be invalidated on."""
    return repr((YEARS, SEED, N_SHUFFLE, N_BOOT, MODE, TIERS, OUTCOME,
                 [(r.key, r.label, r.score) for r in REGIMES]))


if __name__ == "__main__":
    et = read_eval_table()
    print(f"{len(et):,} papers, {sorted(et.year.unique())}, mode={MODE}\n")
    print(f"{'regime':<30}{'n scored':>10}{'distinct':>10}"
          f"{'own':>8}{'supplied':>10}{'tied':>8}")
    for year in YEARS:
        p = et[et["year"] == year]
        n = n_for(et, year)
        print(f"-- {year}  n={n}")
        for r in REGIMES:
            if r.score is None:
                print(f"  {r.label:<28}{n:>10}{'—':>10}{n:>8}{0:>10}{0:>8}")
                continue
            s = p[r.score]
            own, sup, tied = resolution(s, n)
            print(f"  {r.label:<28}{int(s.notna().sum()):>10}"
                  f"{int(s.nunique()):>10}{own:>8}{sup:>10}{tied:>8}")
