"""
Coverage of author-side metadata for the 2018-2020 sample: institutions,
affiliations, countries, author h-index, and field labels.

WHY THIS IS AN AUDIT AND NOT A RESULT. Every covariate here is a candidate
control or a candidate heterogeneity cut, and each one resolves for a different
and non-random subset of the pool. A covariate whose coverage correlates with the
decision cannot be used as a cut without conditioning on a collider: comparing
"papers from top institutions" to the rest partly compares "papers OpenAlex could
resolve" to the rest, and resolution is easier for papers that got published.

So the number that decides whether a field is usable is not its coverage but the
gap in its coverage between accepted and rejected papers. That gap is the last
column of every table below, and it is why `field` is excluded from the exhibits
(see src/figures/spec.py) despite covering 60% of the pool.

Reported at two levels, because they answer different questions:

  paper level   can this paper enter an analysis that needs the covariate
  author level  of the author records we do have, how many resolve

Run: python src/audit/metadata_coverage.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec  # noqa: E402

AUTHOR_IDS = "outputs/paper_author_ids.csv"
AUTHOR_COVARIATES = "outputs/paper_author_covariates.csv"
AUTHOR_STATS = "outputs/author_stats.csv"
AUTHOR_NAMES_OR = "outputs/paper_author_names_openreview.csv"
FIELDS = "outputs/paper_fields.csv"

OUT_DIR = "outputs/audits"
OUT_CSV = "outputs/audits/metadata_coverage.csv"
OUT_MD = "outputs/audits/metadata_coverage.md"

# A gap this wide on the accept/reject axis makes a covariate unusable as a cut,
# because the cut is then partly a cut on the decision itself.
GAP_WARN_PP = 5.0


def paper_level(et):
    """One row per metadata field: coverage overall, by decision, and the gap."""
    ai = pd.read_csv(AUTHOR_IDS, low_memory=False)
    cov = pd.read_csv(AUTHOR_COVARIATES, low_memory=False)
    fields = pd.read_csv(FIELDS, low_memory=False).rename(columns={"id": "paper_id"})
    names = pd.read_csv(AUTHOR_NAMES_OR, low_memory=False)

    ai = ai[ai.paper_id.isin(set(et.paper_id))]
    g = ai.groupby("paper_id")

    have = {
        # --- identity
        "OpenAlex work matched (any author row)": set(ai.paper_id),
        "author names from OpenReview": set(names.paper_id),
        "OpenAlex author id, all authors":
            set(g.apply(lambda d: d.author_id.notna().all(), include_groups=False)
                 .loc[lambda s: s].index),
        # --- affiliation
        "institution, at least one author":
            set(g.institution_id.apply(lambda s: s.notna().any())
                 .loc[lambda s: s].index),
        "institution, first author":
            set(ai[(ai.author_position == ai.groupby("paper_id")
                    .author_position.transform("min")) & ai.institution_id.notna()]
                .paper_id),
        "institution, every author":
            set(g.institution_id.apply(lambda s: s.notna().all())
                 .loc[lambda s: s].index),
        "country, at least one author":
            set(g.country.apply(lambda s: s.notna().any()).loc[lambda s: s].index),
        # --- derived covariates
        "team size": set(cov.loc[cov.team_size.notna(), "paper_id"]),
        "first-author h-index": set(cov.loc[cov.first_author_h_index.notna(), "paper_id"]),
        "top-institution flag": set(cov.loc[cov.top_institution_flag.notna(), "paper_id"]),
        "industry flag": set(cov.loc[cov.industry_flag.notna(), "paper_id"]),
        # --- field
        "field label": set(fields.paper_id),
    }

    acc, rej = et[et.accepted], et[~et.accepted]
    rows = []
    for name, ids in have.items():
        a = acc.paper_id.isin(ids).mean()
        r = rej.paper_id.isin(ids).mean()
        rows.append({"level": "paper", "field": name,
                     "n_covered": int(et.paper_id.isin(ids).sum()),
                     "n_total": len(et),
                     "coverage": et.paper_id.isin(ids).mean(),
                     "coverage_accepted": a, "coverage_rejected": r,
                     "gap_pp": abs(a - r) * 100})
    return pd.DataFrame(rows), ai


def author_level(ai):
    """Of the author records we have, how many resolve to an institution."""
    rows = []
    for name, col in [("OpenAlex author id", "author_id"),
                      ("author name", "author_name"),
                      ("institution id", "institution_id"),
                      ("institution name", "institution_name"),
                      ("country", "country")]:
        rows.append({"level": "author", "field": name,
                     "n_covered": int(ai[col].notna().sum()),
                     "n_total": len(ai),
                     "coverage": ai[col].notna().mean(),
                     "coverage_accepted": float("nan"),
                     "coverage_rejected": float("nan"),
                     "gap_pp": float("nan")})
    return pd.DataFrame(rows)


def by_year(et, ai):
    rows = []
    for year in spec.YEARS:
        y = et[et.year == year]
        matched = y.paper_id.isin(set(ai.paper_id))
        inst = y.paper_id.isin(set(ai.loc[ai.institution_id.notna(), "paper_id"]))
        rows.append({"year": year, "papers": len(y),
                     "openalex_matched": matched.mean(),
                     "any_institution": inst.mean(),
                     "authors_per_matched_paper":
                         len(ai[ai.paper_id.isin(y.paper_id)]) / max(matched.sum(), 1)})
    return pd.DataFrame(rows)


def top_institutions(ai, n=12):
    c = ai.institution_name.value_counts().head(n)
    return pd.DataFrame({"institution": c.index, "author_rows": c.values,
                         "share_of_resolved": c.values / ai.institution_id.notna().sum()})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    et = spec.read_eval_table()

    paper, ai = paper_level(et)
    author = author_level(ai)
    years = by_year(et, ai)
    insts = top_institutions(ai)

    out = pd.concat([paper, author], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)

    def pct(v):
        return "—" if pd.isna(v) else f"{v:.1%}"

    L = [f"# Author metadata coverage — ICLR {spec.YEARS[0]}–{spec.YEARS[-1]}", "",
         f"{len(et):,} papers, {int(et.accepted.sum()):,} accepted / "
         f"{int((~et.accepted).sum()):,} rejected. Generated by "
         "`python src/audit/metadata_coverage.py`.", "",
         "The last column is the one that matters. A covariate whose coverage "
         "differs by decision cannot be used as a cut without the cut partly being "
         f"on the decision itself; anything above {GAP_WARN_PP:.0f} pp is flagged.",
         "", "## Paper level", "",
         "| field | covered | of | coverage | accepted | rejected | gap (pp) | |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in paper.itertuples():
        flag = " ⚠" if r.gap_pp > GAP_WARN_PP else ""
        L.append(f"| {r.field} | {r.n_covered:,} | {r.n_total:,} | {pct(r.coverage)} | "
                 f"{pct(r.coverage_accepted)} | {pct(r.coverage_rejected)} | "
                 f"{r.gap_pp:.1f} |{flag} |")

    L += ["", "## Author level", "",
          f"{len(ai):,} author records across "
          f"{ai.paper_id.nunique():,} papers with an OpenAlex match.", "",
          "| field | resolved | of | coverage |", "|---|---:|---:|---:|"]
    for r in author.itertuples():
        L.append(f"| {r.field} | {r.n_covered:,} | {r.n_total:,} | {pct(r.coverage)} |")

    L += ["", "## By year", "",
          "| year | papers | OpenAlex matched | any institution | authors / matched paper |",
          "|---|---:|---:|---:|---:|"]
    for r in years.itertuples():
        L.append(f"| {r.year} | {r.papers:,} | {pct(r.openalex_matched)} | "
                 f"{pct(r.any_institution)} | {r.authors_per_matched_paper:.1f} |")

    L += ["", "## Most frequent resolved institutions", "",
          "| institution | author rows | share of resolved |", "|---|---:|---:|"]
    for r in insts.itertuples():
        L.append(f"| {r.institution} | {r.author_rows:,} | {r.share_of_resolved:.1%} |")

    open(OUT_MD, "w").write("\n".join(L) + "\n")

    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(paper.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()
        print(author.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()
        print(years.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {OUT_CSV}\n-> {OUT_MD}")
    return paper, author


if __name__ == "__main__":
    p, a = main()
    worst = p.loc[p.gap_pp.idxmax()]
    print(f"\nwidest accept/reject gap: {worst.field} at {worst.gap_pp:.1f} pp")
