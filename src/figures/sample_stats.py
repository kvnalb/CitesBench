"""
Sample statistics and data coverage for the 2018-2020 pool.

ONE INPUT: outputs/paper_master.parquet. That is the point of the master table —
every number in both exhibits comes from the same 4,567 rows, so a coverage figure
quoted in the text and a mean quoted in the table cannot disagree.

TWO TABLES, DIFFERENT ROW LISTS. The first draft used one variable list for both and
that was the mistake: coverage is a property of the SOURCE, not of the variable.
Five OpenAlex author variables all showed 71.5% because they are one indicator asked
five times, and four publication-record variables all showed 96.8% because they are
`notna()` over the S2-matched set. 24 rows carried 11 distinct values.

  sample_stats     one row per variable you would put in a regression.
  sample_coverage  one row per SOURCE, with the variables it feeds. Ten rows, ten
                   distinct numbers, no repetition.

AUTHOR VARIABLES COME FROM S2, NOT OPENALEX. Both sources describe the same
construct. S2 resolves 96.7% of papers against OpenAlex's 71.5%, and its
accept/reject availability gap is +2.8 pp against +26.3 pp. The OpenAlex columns are
still in the master table and still feed the three institution flags, because S2
affiliations run 34.3% and cannot replace them. Everything else author-related uses
S2.

Note the h-index levels are not an artifact of either source: OpenAlex gives max
h-index mean 57.8 / median 50.0 and S2 gives 60.5 / 52.5 on a larger sample. Two
independent resolutions agreeing is the reason to trust the level. It is high
because it is the most senior author on the team, read in 2026, and machine learning
seniors are heavily cited.

WHAT IS DELIBERATELY NOT HERE.
  - `s2_reference_count`: bibliography length, not an outcome, and S2 counts the
    version it indexed, which for an accepted paper is the camera-ready.
  - `has_tldr`: records whether an optional OpenReview form field was filled in.
  - `has_doi` / `has_oa_pdf` / `has_journal`: identical coverage to the venue flag,
    which is kept because 94.0% vs 24.9% is the informative one.
  - `log(1 + citations)` in the coverage table: identical to citations by definition.

Run: python src/figures/sample_stats.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs  # noqa: E402

MASTER = "outputs/paper_master.parquet"

STATS_CSV = "outputs/figures/sample_stats.csv"
STATS_TEX = "outputs/figures/sample_stats.tex"
STATS_PDF = "outputs/figures/sample_stats.pdf"
STATS_PNG = "outputs/figures/sample_stats.png"
COV_CSV = "outputs/figures/sample_coverage.csv"
COV_TEX = "outputs/figures/sample_coverage.tex"
COV_PDF = "outputs/figures/sample_coverage.pdf"
COV_PNG = "outputs/figures/sample_coverage.png"

TITLE_STATS = "Sample characteristics, ICLR 2018-2020"
TITLE_COV = ("Data availability by source, and its dependence on the "
             "decision")

# (column, label, kind). kind: "n" numeric, "b" binary share, "c" integer count.
PANELS = [
    ("Outcome", [
        ("s2_citations", "Citations", "n"),
        ("log_citations", "log(1 + citations)", "n"),
    ]),
    ("Human review", [
        ("mean_rating", "Mean review score", "n"),
        ("rating_std", "Review score SD", "n"),
        ("n_reviews", "Number of reviews", "c"),
        ("mean_confidence", "Mean reviewer confidence", "n"),
    ]),
    ("LLM regimes", [
        ("committee_rating", "Council rating (9 calls)", "n"),
        ("single_call_rating", "Single-call rating", "n"),
        ("deepseek_p_accept", "Decision head P(accept)", "n"),
    ]),
    ("Authors", [
        ("n_s2_authors", "Team size", "c"),
        ("s2_first_author_h_index", "First-author h-index", "n"),
        ("s2_max_h_index", "Max h-index on team", "n"),
        ("s2_sum_author_paper_count", "Team papers published", "c"),
    ]),
    ("Institutions", [
        ("share_authors_with_institution", "Authors with an institution", "b"),
        ("top_institution_flag", "Top institution on team", "b"),
        ("industry_flag", "Industry affiliation on team", "b"),
        ("us_team_flag", "US affiliation on team", "b"),
    ]),
    ("Submission", [
        ("abstract_words", "Abstract length (words)", "c"),
        ("n_keywords", "Number of keywords", "c"),
    ]),
    ("Publication record", [
        ("has_venue", "Published at a recorded venue", "b"),
    ]),
]

# (indicator column, source, what it feeds). One row per source, so every number in
# this table is a distinct fact about the data rather than the same fact restated.
SOURCES = [
    ("abstract", "OpenReview submission", "Abstract, decision, year"),
    ("keywords", "OpenReview keywords", "Keyword count"),
    ("mean_rating", "OpenReview reviews", "Score, disagreement, count"),
    ("mean_confidence", "OpenReview confidence", "Reviewer confidence"),
    ("committee_rating", "Our council run", "Council rating"),
    ("single_call_rating", "Our single-call run", "Single-call rating"),
    ("s2_primary_field", "S2 paper record", "Field, venue, team, h-index"),
    ("s2_citations", "S2 citations, tier A+B", "The outcome"),
    ("n_author_records", "OpenAlex author record", "Country and team flags"),
    ("institutions", "OpenAlex institution", "Named institution, industry"),
]


def prepare():
    d = pd.read_parquet(MASTER)
    d = d[d.year.isin(spec.YEARS)].copy()
    assert d.paper_id.is_unique, "paper_master is not one row per paper"

    d["log_citations"] = np.log1p(d[spec.OUTCOME])
    # OpenReview's abstract is the submitted text. S2's copy is the published
    # version and is missing wherever S2 did not match.
    d["abstract_words"] = d.abstract.fillna("").str.split().str.len().replace(0, np.nan)
    d["n_keywords"] = (d.keywords.fillna("").str.count(",") + 1).where(d.keywords.notna())
    # Defined over the S2-matched set: absent means "no venue recorded", not
    # "unknown", so 0/1 there and NaN outside it.
    d["has_venue"] = d.s2_venue_type.notna().astype(float).where(d.s2_primary_field.notna())
    return d


def welch(a, b):
    """Difference in means and its Welch t: unequal n, unequal variance, which is
    what an accept/reject split always is. 1,526 accepts against 3,041 rejects, and
    citation variance among accepts is far larger."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    diff = a.mean() - b.mean()
    return diff, (diff / se if se > 0 else np.nan)


def stats_rows(d):
    out = []
    for panel, items in PANELS:
        for col, label, kind in items:
            s = pd.to_numeric(d[col], errors="coerce")
            acc, rej = s[d.accepted], s[~d.accepted]
            diff, t = welch(acc, rej)
            out.append({"panel": panel, "variable": label, "column": col, "kind": kind,
                        "n": int(s.notna().sum()), "mean": s.mean(), "sd": s.std(),
                        "median": s.median(), "mean_accepted": acc.mean(),
                        "mean_rejected": rej.mean(), "difference": diff, "welch_t": t})
    return pd.DataFrame(out)


def coverage_rows(d):
    out = []
    for col, source, feeds in SOURCES:
        have = d[col].notna()
        a, r = have[d.accepted].mean(), have[~d.accepted].mean()
        out.append({"source": source, "indicator_column": col, "feeds": feeds,
                    "coverage": have.mean(), "coverage_accepted": a,
                    "coverage_rejected": r, "differential_pp": 100 * (a - r)})
    c = pd.DataFrame(out).sort_values("coverage", ascending=False, ignore_index=True)
    assert c.coverage.round(4).is_unique, \
        "two sources share a coverage figure — they are the same source"
    return c


def _fmt(v, kind):
    if pd.isna(v):
        return "—"
    if kind == "b":
        return f"{100 * v:.1f}"
    if kind == "c":
        return f"{v:,.1f}"
    return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.1f}"


def _minus(s):
    return s.replace("-", "−")


def _tex_num(s):
    """A leading hyphen in a maths column must be a real minus in LaTeX too."""
    return s.replace("−", "-").replace("-", "$-$")


def render(st, cv):
    os.makedirs("outputs/figures", exist_ok=True)
    fs.apply()

    body, rules = [], []
    for panel, _ in PANELS:
        rules.append(len(body))
        body.append([panel] + [""] * 8)
        for _, w in st[st.panel == panel].iterrows():
            body.append([f"   {w.variable}", f"{w.n:,}"]
                        + [_minus(_fmt(w[k], w.kind)) for k in
                           ["mean", "sd", "median", "mean_accepted", "mean_rejected",
                            "difference"]]
                        + [_minus("—" if pd.isna(w.welch_t) else f"{w.welch_t:.1f}")])
    fig = fs.table(
        header=[[("", 0, 0), ("N", 1, 1), ("Full sample", 2, 4), ("By decision", 5, 8)],
                ["", "", "Mean", "SD", "p50", "Accepted", "Rejected", "Diff.", "t"]],
        body=body, align="lrrrrrrrr",
        colw=[3.20, 0.70, 0.72, 0.64, 0.80, 1.02, 0.90, 0.80, 0.72],
        rules=tuple(rules[1:]),
        note="Shares in percent. Diff. = accepted minus rejected, t = Welch.")
    fs.add_title(fig, TITLE_STATS)
    fig.savefig(STATS_PDF)
    fig.savefig(STATS_PNG, dpi=220)
    plt.close(fig)

    body = [[w.source, w.feeds, f"{100 * w.coverage:.1f}",
             f"{100 * w.coverage_accepted:.1f}", f"{100 * w.coverage_rejected:.1f}",
             _minus(f"{w.differential_pp:+.1f}")] for _, w in cv.iterrows()]
    fig = fs.table(
        header=[[("Source", 0, 0), ("Feeds", 1, 1), ("Non-missing (%)", 2, 4),
                 ("Diff.", 5, 5)],
                ["", "", "All", "Accepted", "Rejected", "(pp)"]],
        body=body, align="llrrrr",
        colw=[2.55, 3.05, 0.68, 0.88, 0.84, 0.70],
        note="Availability of each source, not the values it carries.")
    fs.add_title(fig, TITLE_COV)
    fig.savefig(COV_PDF)
    fig.savefig(COV_PNG, dpi=220)
    plt.close(fig)


def latex(st, cv):
    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"& N & Mean & SD & p50 & Accepted & Rejected & Diff. & $t$ \\", r"\midrule"]
    for i, (panel, _) in enumerate(PANELS):
        if i:
            L.append(r"\addlinespace")
        L.append(r"\multicolumn{9}{l}{\textit{%s}} \\" % panel)
        for _, w in st[st.panel == panel].iterrows():
            cells = [f"{w.n:,}"] + [_tex_num(_fmt(w[k], w.kind)) for k in
                                    ["mean", "sd", "median", "mean_accepted",
                                     "mean_rejected", "difference"]] \
                + [_tex_num("—" if pd.isna(w.welch_t) else f"{w.welch_t:.1f}")]
            L.append(r"\quad %s & %s \\" % (w.variable, " & ".join(cells)))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(STATS_TEX, "w").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Source & Feeds & All & Accepted & Rejected & Diff. (pp) \\", r"\midrule"]
    for _, w in cv.iterrows():
        L.append(r"%s & %s & %.1f & %.1f & %.1f & %s \\" % (
            w.source, w.feeds, 100 * w.coverage, 100 * w.coverage_accepted,
            100 * w.coverage_rejected, _tex_num(f"{w.differential_pp:+.1f}")))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(COV_TEX, "w").write("\n".join(L) + "\n")


def build():
    d = prepare()
    st, cv = stats_rows(d), coverage_rows(d)
    st.to_csv(STATS_CSV, index=False)
    cv.to_csv(COV_CSV, index=False)
    render(st, cv)
    latex(st, cv)

    print(st[["variable", "n", "mean", "median", "mean_accepted", "mean_rejected",
              "difference", "welch_t"]].to_string(index=False,
                                                  float_format=lambda v: f"{v:,.3f}"))
    print()
    print(cv[["source", "coverage", "coverage_accepted", "coverage_rejected",
              "differential_pp"]].to_string(index=False,
                                            float_format=lambda v: f"{v:,.3f}"))
    print(f"\n-> {STATS_PDF} / {STATS_PNG} / {STATS_TEX} / {STATS_CSV}")
    print(f"-> {COV_PDF} / {COV_PNG} / {COV_TEX} / {COV_CSV}")
    return d, st, cv


def demo():
    d, st, cv = build()

    assert len(d) == 4567, f"pool is {len(d)}, expected 4,567"
    assert int(d.accepted.sum()) == sum(spec.N_PINNED.values()), \
        "accept count does not match the pinned n"

    # A row of dashes is a typo in a column name, not a finding.
    for _, w in st.iterrows():
        assert w.n > 0, f"{w.column}: no observations"
        assert not pd.isna(w["sd"]) and w["sd"] > 0, f"{w.column}: does not vary"

    # Sanity on the split: if this fails the accept flag is inverted somewhere.
    sc = st[st.column == "mean_rating"].iloc[0]
    assert sc.difference > 0 and sc.welch_t > 10, \
        f"accepted papers should score higher: diff {sc.difference}, t {sc.welch_t}"

    # The author variables must come from S2, whose coverage beats OpenAlex by ~25
    # points. Reverting to the OpenAlex columns would silently reintroduce a 26pp
    # availability gap on the axis this benchmark measures.
    s2 = float(cv.loc[cv.indicator_column == "s2_primary_field", "coverage"].iloc[0])
    oa = float(cv.loc[cv.indicator_column == "n_author_records", "coverage"].iloc[0])
    assert s2 > oa + 0.20, f"S2 {s2:.3f} should beat OpenAlex {oa:.3f} by 20+ points"
    assert st.loc[st.column == "n_s2_authors", "n"].iloc[0] > 4000, \
        "team size should be the S2 column, not the OpenAlex one"

    # Coverage must agree with the master table, which computes it by another route.
    ref = pd.read_csv("outputs/paper_master_coverage.csv").set_index("column")
    for _, w in cv.iterrows():
        b = float(ref.loc[w.indicator_column, "differential_pp"])
        assert abs(w.differential_pp - b) < 1e-6, \
            f"{w.indicator_column}: {w.differential_pp} here vs {b} in paper_master"

    print(f"\nok — {len(d):,} papers, {int(d.accepted.sum()):,} accepted; "
          f"{len(st)} variables in {len(PANELS)} panels, "
          f"{len(cv)} sources spanning {100 * cv.coverage.min():.1f}% to "
          f"{100 * cv.coverage.max():.1f}% coverage")


if __name__ == "__main__":
    demo()
