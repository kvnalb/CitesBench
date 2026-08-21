"""
Sample statistics for the 2018-2020 pool, from outputs/paper_master.parquet.

Two tables, because they answer two different questions and one wide table answers
neither well.

  sample_stats     what each variable looks like: N, mean, SD, median, and the
                   accepted / rejected means with their difference and Welch t.
  sample_coverage  how much of each variable exists, and whether its availability
                   depends on the decision.

WHY COVERAGE GETS ITS OWN TABLE. This benchmark measures "does this regime select
better from a pool of accepts and rejects". Any variable whose availability differs
by decision carries the decision inside it. So the differential is reported for
every variable, with no variable dropped for having a large one. Several are large:
author records run +26 pp, institutions +43 pp, and the columns describing where a
paper was eventually published run higher still. Those numbers are the evidence, and
which variables to use is the reader's call, not this script's.

VARIABLES MEASURED AFTER THE DECISION are marked with a dagger rather than removed.
A paper's DOI, its journal, its open-access PDF and its authors' current h-indices
are all things acceptance can cause. They are shown because the size of their
accept/reject gap is itself informative about how much acceptance changes.

Run: python src/figures/sample_stats.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs  # noqa: E402

# The parquet, not the CSV: abstract / keywords / tldr are the text columns
# build_paper_master.py strips from the CSV, and the text-length rows need them.
MASTER = "outputs/paper_master.parquet"

STATS_CSV = "outputs/figures/sample_stats.csv"
STATS_TEX = "outputs/figures/sample_stats.tex"
STATS_PDF = "outputs/figures/sample_stats.pdf"
STATS_PNG = "outputs/figures/sample_stats.png"
COV_CSV = "outputs/figures/sample_coverage.csv"
COV_TEX = "outputs/figures/sample_coverage.tex"
COV_PDF = "outputs/figures/sample_coverage.pdf"
COV_PNG = "outputs/figures/sample_coverage.png"

DAGGER = "†"

# (column, label, kind, post_treatment). kind: "n" numeric, "b" binary share,
# "c" integer count. A derived column is built in prepare().
PANELS = [
    ("Outcome", [
        ("openalex_citations", "Citations", "n", False),
        ("log_citations", "log(1 + citations)", "n", False),
        ("s2_reference_count", "References made", "n", False),
    ]),
    ("Human review", [
        ("mean_rating", "Mean review score", "n", False),
        ("rating_std", "Review score SD", "n", False),
        ("n_reviews", "Number of reviews", "c", False),
        ("mean_confidence", "Mean reviewer confidence", "n", False),
    ]),
    ("LLM regimes", [
        ("committee_rating", "Council rating (9 calls)", "n", False),
        ("single_call_rating", "Single-call rating", "n", False),
        ("deepseek_p_accept", "Decision head P(accept)", "n", False),
    ]),
    ("Authors and institutions", [
        ("team_size", "Team size", "c", False),
        ("share_authors_with_institution", "Authors with an institution", "b", False),
        ("top_institution_flag", "Top institution on team", "b", False),
        ("industry_flag", "Industry affiliation on team", "b", False),
        ("us_team_flag", "US affiliation on team", "b", False),
        ("first_author_h_index", "First-author h-index" + DAGGER, "n", True),
        ("max_h_index", "Max h-index on team" + DAGGER, "n", True),
    ]),
    ("Submission text", [
        ("abstract_words", "Abstract length (words)", "c", False),
        ("n_keywords", "Number of keywords", "c", False),
        ("has_tldr", "Has a TL;DR", "b", False),
    ]),
    ("Publication record" + DAGGER, [
        ("has_doi", "Has a DOI" + DAGGER, "b", True),
        ("has_oa_pdf", "Has an open-access PDF" + DAGGER, "b", True),
        ("has_venue", "Published at a recorded venue" + DAGGER, "b", True),
        ("has_journal", "Published in a journal" + DAGGER, "b", True),
    ]),
]


def prepare():
    d = pd.read_parquet(MASTER)
    d = d[d.year.isin(spec.YEARS)].copy()
    assert d.paper_id.is_unique, "paper_master is not one row per paper"

    d["log_citations"] = np.log1p(d[spec.OUTCOME])
    # Text length from OpenReview's abstract, which is the submitted text. S2's copy
    # is the published version and is missing for anything S2 did not match.
    d["abstract_words"] = d.abstract.fillna("").str.split().str.len().replace(0, np.nan)
    d["n_keywords"] = (d.keywords.fillna("").str.count(",") + 1).where(d.keywords.notna())
    d["has_tldr"] = d.tldr.notna().astype(float)
    # These four are defined for every paper S2 matched: absent means "no", not
    # "unknown", so they are 0/1 over the matched set rather than NaN.
    matched = d.s2_primary_field.notna()
    for col, src in [("has_doi", "s2_doi"), ("has_oa_pdf", "s2_oa_pdf_url"),
                     ("has_venue", "s2_venue_type"), ("has_journal", "s2_journal")]:
        d[col] = d[src].notna().astype(float).where(matched)
    return d


def welch(a, b):
    """Difference in means and its Welch t. Unequal n and unequal variance, which
    is what an accept/reject split always is."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    diff = a.mean() - b.mean()
    return diff, (diff / se if se > 0 else np.nan)


def collect(d):
    """One record per variable, for both tables."""
    out = []
    for panel, items in PANELS:
        for col, label, kind, post in items:
            s, acc, rej = d[col], d.accepted, ~d.accepted
            diff, t = welch(d.loc[acc, col], d.loc[rej, col])
            out.append({
                "panel": panel, "variable": label, "column": col,
                "kind": kind, "post_treatment": post,
                "n": int(s.notna().sum()),
                "mean": s.mean(), "sd": s.std(), "median": s.median(),
                "mean_accepted": d.loc[acc, col].mean(),
                "mean_rejected": d.loc[rej, col].mean(),
                "difference": diff, "welch_t": t,
                "coverage": s.notna().mean(),
                "coverage_accepted": d.loc[acc, col].notna().mean(),
                "coverage_rejected": d.loc[rej, col].notna().mean(),
            })
    r = pd.DataFrame(out)
    r["differential_pp"] = 100 * (r.coverage_accepted - r.coverage_rejected)
    return r


def _fmt(v, kind):
    if pd.isna(v):
        return "—"
    if kind == "b":
        return f"{100 * v:.1f}"
    if kind == "c":
        return f"{v:,.1f}"
    return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.1f}"


def _minus(s):
    return s.replace("-", "\u2212")


def _tex_num(s):
    """A leading hyphen in a maths column must be a real minus in LaTeX too."""
    return s.replace("\u2212", "-").replace("-", "$-$")


def render(r):
    os.makedirs("outputs/figures", exist_ok=True)
    fs.apply()

    # ---- summary statistics
    body, rules = [], []
    for panel, _ in PANELS:
        rules.append(len(body))
        body.append([panel] + [""] * 8)
        for _, w in r[r.panel == panel].iterrows():
            body.append([f"   {w.variable}", f"{w.n:,}"]
                        + [_minus(_fmt(w[k], w.kind)) for k in
                           ["mean", "sd", "median", "mean_accepted", "mean_rejected",
                            "difference"]]
                        + [_minus("—" if pd.isna(w.welch_t) else f"{w.welch_t:.1f}")])
    fig = fs.table(
        header=[[("", 0, 0), ("N", 1, 1), ("Full sample", 2, 4),
                 ("By decision", 5, 8)],
                ["", "", "Mean", "SD", "p50", "Accepted", "Rejected", "Diff.",
                 "t"]],
        body=[r + [""] if len(r) == 8 else r for r in
              [b if len(b) > 8 else b + [""] * (9 - len(b)) for b in body]],
        align="lrrrrrrrr",
        colw=[3.35, 0.70, 0.72, 0.64, 0.80, 0.92, 0.90, 0.80, 0.72],
        rules=tuple(rules[1:]),
        note=("Shares in percent. Diff. = accepted minus rejected, t = Welch. "
              f"{DAGGER} measured after the decision."))
    fig.savefig(STATS_PDF)
    fig.savefig(STATS_PNG, dpi=220)
    plt.close(fig)

    # ---- coverage
    body, rules = [], []
    for panel, _ in PANELS:
        rules.append(len(body))
        body.append([panel, "", "", "", ""])
        for _, w in r[r.panel == panel].iterrows():
            body.append([f"   {w.variable}", f"{100 * w.coverage:.1f}",
                         f"{100 * w.coverage_accepted:.1f}",
                         f"{100 * w.coverage_rejected:.1f}",
                         _minus(f"{w.differential_pp:+.1f}")])
    fig = fs.table(
        header=[[("", 0, 0), ("Non-missing (%)", 1, 3), ("Differential", 4, 4)],
                ["", "All", "Accepted", "Rejected", "(pp)"]],
        body=body, align="lrrrr",
        colw=[3.2, 0.85, 0.95, 0.95, 1.0],
        rules=tuple(rules[1:]),
        note=("Availability of each variable, not its value. "
              f"{DAGGER} measured after the decision."))
    fig.savefig(COV_PDF)
    fig.savefig(COV_PNG, dpi=220)
    plt.close(fig)


def latex(r):
    def block(cols, header, fmt):
        L = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", header,
             r"\midrule"]
        for panel, _ in PANELS:
            L.append(r"\addlinespace" if L[-1] != r"\midrule" else "")
            L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" %
                     (len(cols) + 1, panel.replace(DAGGER, r"$\dagger$")))
            for _, w in r[r.panel == panel].iterrows():
                cells = " & ".join(fmt(w, c) for c in cols)
                L.append(r"\quad %s & %s \\" %
                         (w.variable.replace(DAGGER, r"$\dagger$"), cells))
        L += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(x for x in L if x)

    tex = block(["n", "mean", "sd", "median", "mean_accepted", "mean_rejected",
                 "difference", "welch_t"],
                r"& N & Mean & SD & p50 & Accepted & Rejected & Diff. & $t$ \\",
                lambda w, c: (f"{w.n:,}" if c == "n"
                              else _tex_num("—" if pd.isna(w.welch_t)
                                            else f"{w.welch_t:.1f}") if c == "welch_t"
                              else _tex_num(_fmt(w[c], w.kind))))
    open(STATS_TEX, "w").write(tex + "\n")

    tex = block(["coverage", "coverage_accepted", "coverage_rejected",
                 "differential_pp"],
                r"& All & Accepted & Rejected & Diff. (pp) \\",
                lambda w, c: (_tex_num(f"{w.differential_pp:+.1f}")
                              if c == "differential_pp" else f"{100 * w[c]:.1f}"))
    open(COV_TEX, "w").write(tex + "\n")


def build():
    d = prepare()
    r = collect(d)
    r.to_csv(STATS_CSV, index=False)
    r[["panel", "variable", "column", "coverage", "coverage_accepted",
       "coverage_rejected", "differential_pp", "post_treatment"]].to_csv(
        COV_CSV, index=False)
    render(r)
    latex(r)

    show = r[["variable", "n", "mean", "median", "mean_accepted", "mean_rejected",
              "difference", "welch_t", "coverage", "differential_pp"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\n-> {STATS_PDF} / {STATS_PNG} / {STATS_TEX} / {STATS_CSV}")
    print(f"-> {COV_PDF} / {COV_PNG} / {COV_TEX} / {COV_CSV}")
    return d, r


def demo():
    d, r = build()

    assert len(d) == 4567, f"pool is {len(d)}, expected 4,567"
    assert int(d.accepted.sum()) == sum(spec.N_PINNED.values()), \
        "accept count does not match the pinned n"

    # Every variable in PANELS must actually exist and vary. A row of dashes is a
    # typo in a column name, not a finding.
    for _, w in r.iterrows():
        assert w.n > 0, f"{w.column}: no observations"
        assert not pd.isna(w["sd"]) and w["sd"] > 0, f"{w.column}: does not vary"

    # The sanity check on the split: accepted papers must have higher review scores.
    # If this fails the accept flag is inverted somewhere.
    sc = r[r.column == "mean_rating"].iloc[0]
    assert sc.difference > 0 and sc.welch_t > 10, \
        f"accepted papers should score higher: diff {sc.difference}, t {sc.welch_t}"

    # Coverage differentials must match the master table, which computes them from
    # the same file by a different route.
    cov = pd.read_csv("outputs/paper_master_coverage.csv").set_index("column")
    for col in ["openalex_citations", "mean_rating", "team_size", "max_h_index"]:
        a = float(r.loc[r.column == col, "differential_pp"].iloc[0])
        b = float(cov.loc[col, "differential_pp"])
        assert abs(a - b) < 1e-6, f"{col}: {a} here vs {b} in paper_master_coverage"

    big = r.loc[r.differential_pp.abs().idxmax()]
    print(f"\nok — {len(d):,} papers, {int(d.accepted.sum()):,} accepted, "
          f"{len(r)} variables in {len(PANELS)} panels; "
          f"largest availability gap {big.differential_pp:+.1f}pp ({big.column})")


if __name__ == "__main__":
    demo()
