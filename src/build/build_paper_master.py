"""
One row per submission: the master paper table for sample statistics (issue #43).

WHY. A sample-statistics table currently pulls from six places, and every ad-hoc
pull re-derives the joins slightly differently. This script does the join once so
the coverage of any attribute is one `notna().mean()` and its accept/reject
differential is two lines. `outputs/paper_master_coverage.csv` precomputes both.

SPINE. outputs/eval_table.csv, which already defines the 2018-2020 population and
passes src/figures/check_inputs.py. Every merge is validate="1:1" and the row count
is asserted unchanged, so no join can fan out or drop a paper.

TWO DEVIATIONS FROM THE ISSUE, both because the data does not support the plan:

  1. The REVIEW sub-scores (soundness / presentation / contribution) are 100% blank
     on the 2018-2020 slice — they are ICLR 2024 form fields. So are `correctness`
     and the two `*_novelty_and_significance` columns, and SUBMISSION's
     `primary_area` and `code_of_ethics`. Writing them would add eight columns that
     are entirely null. What this era does carry is `confidence`, `main_review` and
     `binocular_score`, and those are what gets aggregated.
  2. GENAI_REVIEW is not read. Its persona reviews arrived inside data/gen_review.db
     rather than from this project's pipeline, so we cannot state what prompt, model
     or input text produced them; they were removed from the analysis in aceee1d and
     do not come back through this table.

S2 METADATA (issue #44). src/fetch/fetch_s2_metadata.py enriches the paper IDs that
src/fetch/fetch_citations_s2_v2.py already resolved. Two upstream conditions are
handled here rather than swallowed:

  - 26 S2 IDs are claimed by 52 of our papers (`collision` in the tiered match
    file). Joining through them would hand one paper's abstract and field labels to
    a different submission, so collisions are dropped from this join. They keep
    their citation counts; only the new metadata is withheld.
  - 2 IDs come back under a different canonical paperId because S2 merged the
    record. Those rows describe the wrong paper and are dropped too.

READ THE COVERAGE FILE BEFORE USING ANY s2_* COLUMN AS A COVARIATE. Several are
post-treatment by construction and their differentials say so out loud:
s2_venue_type 68.8pp, s2_pubtypes 38.3pp and s2_journal 26.5pp mostly encode
"did this get published somewhere", which acceptance causes. s2_doi and
s2_oa_pdf_url run NEGATIVE (-12.2pp, -10.1pp) because rejected papers go on to
journals that mint DOIs. None of these describe the submission as reviewers saw it.

The s2_* columns are descriptive. s2_*_h_index in particular is read at fetch time
in 2026 and is post-treatment for a 2018 submission: acceptance raised it. It is not
a pre-determined covariate. The outcome variable is untouched; `s2_citations_refetched`
sits beside `openalex_citations` for drift inspection and never replaces it.

OUTPUTS. The text columns make a single CSV unwieldy, so:
  outputs/paper_master.parquet  everything, including review and abstract text
  outputs/paper_master.csv      the same rows without the text columns, for reading
  outputs/paper_master_coverage.csv  one row per column: coverage and the differential

Nothing downstream is repointed. Existing exhibits keep reading eval_table.csv.

Run: python src/build/build_paper_master.py
"""
import os
import sqlite3

import pandas as pd

os.makedirs("outputs", exist_ok=True)

DB = "data/gen_review.db"
SPINE_CSV = "outputs/eval_table.csv"
AUTHOR_IDS_CSV = "outputs/paper_author_ids.csv"
AUTHOR_COVARIATES_CSV = "outputs/paper_author_covariates.csv"

OUT_PARQUET = "outputs/paper_master.parquet"
OUT_CSV = "outputs/paper_master.csv"
OUT_COVERAGE_CSV = "outputs/paper_master_coverage.csv"

S2_MATCH_CSV = "outputs/s2_citations_v2_authored_tiered.csv"
S2_METADATA_CSV = "outputs/s2_paper_metadata.csv"

TEXT_COLS = ["abstract", "tldr", "keywords", "review_text",
             "s2_abstract", "s2_tldr"]
SEP = "\n\n=== REVIEW BREAK ===\n\n"


def _blank_to_na(s):
    """OpenReview exports absent fields as '' or 'None', not NULL."""
    s = s.astype("string").str.strip()
    return s.mask(s.isin(["", "None", "nan", "N/A"]))


def submission_attrs(con, ids):
    d = pd.read_sql("SELECT id AS paper_id, abstract, tldr, keywords FROM SUBMISSION", con)
    d = d[d.paper_id.isin(ids)].drop_duplicates("paper_id")
    for c in ["abstract", "tldr", "keywords"]:
        d[c] = _blank_to_na(d[c])
    return d


def review_attrs(con, ids):
    """Collapse REVIEW to one row per paper. Sub-scores are skipped: see the
    module docstring. `confidence` parses like `rating` does in build_eval_table."""
    r = pd.read_sql(
        "SELECT paper_id, confidence, main_review, binocular_score FROM REVIEW", con)
    r = r[r.paper_id.isin(ids)].copy()
    r["confidence_num"] = _blank_to_na(r.confidence).str.extract(r"^(\d+)")[0].astype(float)
    r["main_review"] = _blank_to_na(r.main_review)

    g = r.groupby("paper_id")
    out = g.agg(
        mean_confidence=("confidence_num", "mean"),
        confidence_std=("confidence_num", "std"),
        mean_binocular_score=("binocular_score", "mean"),
        n_review_texts=("main_review", "count"),
    ).reset_index()
    out["review_text"] = g.main_review.apply(lambda s: SEP.join(s.dropna()) or pd.NA).values
    return out


def author_attrs(ids):
    a = pd.read_csv(AUTHOR_IDS_CSV)
    a = a[a.paper_id.isin(ids)].copy()
    g = a.groupby("paper_id")
    out = g.agg(
        n_author_records=("author_id", "size"),
        share_authors_with_institution=("institution_name", lambda s: s.notna().mean()),
    ).reset_index()
    out["institutions"] = g.institution_name.apply(
        lambda s: "; ".join(sorted(set(s.dropna()))) or pd.NA).values
    out["countries"] = g.country.apply(
        lambda s: "; ".join(sorted(set(s.dropna()))) or pd.NA).values
    first = (a[a.author_position == "first"].drop_duplicates("paper_id")
             [["paper_id", "author_name"]].rename(columns={"author_name": "first_author"}))
    return out.merge(first, on="paper_id", how="left", validate="1:1")


def s2_metadata_attrs(ids):
    """S2 metadata keyed back to our paper_id, with collisions and redirects out.

    Returns None when the fetch has not been run, so this script still works on a
    clean checkout before src/fetch/fetch_s2_metadata.py has been called.
    """
    if not os.path.exists(S2_METADATA_CSV):
        print(f"note: {S2_METADATA_CSV} absent — run src/fetch/fetch_s2_metadata.py")
        return None

    key = pd.read_csv(S2_MATCH_CSV, usecols=["paper_id", "s2_paper_id", "collision"])
    key = key[key.paper_id.isin(ids) & key.s2_paper_id.notna() & ~key.collision]
    key = key.drop(columns="collision")
    assert key.paper_id.is_unique and key.s2_paper_id.is_unique, \
        "collisions should already be gone; the join key is not 1:1"

    meta = pd.read_csv(S2_METADATA_CSV)
    meta = meta[meta.s2_paper_id == meta.s2_paper_id_returned]   # drop S2 redirects
    meta = meta.drop(columns=["s2_paper_id_returned", "s2_title"])
    return key.merge(meta, on="s2_paper_id", how="inner",
                     validate="1:1").drop(columns="s2_paper_id")


def build():
    spine = pd.read_csv(SPINE_CSV)
    assert spine.paper_id.is_unique, "spine is not one row per paper"
    n = len(spine)
    ids = set(spine.paper_id)
    spine["accepted"] = spine.decision.str.lower().str.contains("accept")

    con = sqlite3.connect(DB)
    parts = [submission_attrs(con, ids), review_attrs(con, ids)]
    con.close()

    # team_size lives in the covariates file already; the author roster gives its
    # own count, so keep both under distinct names rather than picking a winner.
    cov = pd.read_csv(AUTHOR_COVARIATES_CSV)
    parts += [author_attrs(ids), cov[cov.paper_id.isin(ids)].drop_duplicates("paper_id")]

    s2 = s2_metadata_attrs(ids)
    if s2 is not None:
        parts.append(s2)

    d = spine
    for p in parts:
        d = d.merge(p, on="paper_id", how="left", validate="1:1")
        assert len(d) == n, f"merge changed the row count: {len(d)} != {n}"

    d.to_parquet(OUT_PARQUET, index=False)
    d.drop(columns=[c for c in TEXT_COLS if c in d]).to_csv(OUT_CSV, index=False)

    cover = pd.DataFrame({
        "column": d.columns,
        "coverage": [d[c].notna().mean() for c in d.columns],
        "coverage_accepted": [d.loc[d.accepted, c].notna().mean() for c in d.columns],
        "coverage_rejected": [d.loc[~d.accepted, c].notna().mean() for c in d.columns],
    })
    cover["differential_pp"] = 100 * (cover.coverage_accepted - cover.coverage_rejected)
    cover["is_text"] = cover.column.isin(TEXT_COLS)
    cover.to_csv(OUT_COVERAGE_CSV, index=False)

    print(f"{n} rows x {len(d.columns)} columns\n")
    print(cover.to_string(index=False,
                          formatters={c: "{:.3f}".format for c in
                                      ["coverage", "coverage_accepted", "coverage_rejected"]}))
    print(f"\n-> {OUT_PARQUET}\n-> {OUT_CSV}\n-> {OUT_COVERAGE_CSV}")
    return d, cover


def demo():
    d, cover = build()
    spine = pd.read_csv(SPINE_CSV)

    assert len(d) == len(spine), "row count drifted from the spine"
    assert set(d.paper_id) == set(spine.paper_id), "paper_id set drifted from the spine"
    assert d.paper_id.is_unique, "master table is not one row per paper"

    # A column that is entirely null is a silent join failure, not a finding.
    empty = cover[cover.coverage == 0].column.tolist()
    assert not empty, f"columns are 100% null: {empty}"

    # Known coverage, from src/audit/metadata_coverage.py. These are the numbers a
    # summary table quotes, so a join that quietly halves one should fail here.
    checks = {"openalex_citations": 0.963, "n_author_records": 0.715,
              "institutions": 0.395}
    for col, want in checks.items():
        got = float(cover.loc[cover.column == col, "coverage"].iloc[0])
        assert abs(got - want) < 0.01, f"{col} coverage {got:.3f}, expected ~{want}"

    diff = float(cover.loc[cover.column == "openalex_citations", "differential_pp"].iloc[0])
    assert abs(diff - 3.9) < 0.5, f"citation coverage differential {diff:.1f}pp, expected ~3.9"

    # The point of the S2 pull: a field label whose coverage does not depend on the
    # decision. paper_fields.csv has a 6.0pp gap on the same axis the benchmark
    # measures, which is why field normalisation is excluded.
    #
    # S2 labels every paper it matched, so it adds no labelling gap of its own (the
    # fetch script asserts 0.0pp on the fetched set). What survives here is the
    # MATCH gap: an unmatched paper has no field, and the match rate is 98.9% for
    # accepted papers against 95.0% for rejected. So the bound is not an invented
    # threshold, it is the differential the outcome variable already carries. S2
    # fields may inherit that gap; they must not exceed it, because then they would
    # be introducing a selection axis rather than passing one through.
    if "s2_primary_field" in d:
        row = cover[cover.column == "s2_primary_field"].iloc[0]
        cite = float(cover.loc[cover.column == "openalex_citations",
                               "differential_pp"].iloc[0])
        assert row.coverage > 0.95, f"s2 field coverage only {row.coverage:.1%}"
        assert abs(row.differential_pp) <= abs(cite) + 0.1, (
            f"s2 field differential {row.differential_pp:.1f}pp exceeds the citation "
            f"match differential {cite:.1f}pp — it is adding a gap, not inheriting one")
        old = cover[cover.column == "field"].iloc[0]
        print(f"\nfield label: paper_fields {old.coverage:.1%} coverage at "
              f"{old.differential_pp:+.1f}pp -> s2 {row.coverage:.1%} at "
              f"{row.differential_pp:+.1f}pp (citation match gap {cite:+.1f}pp)")

    print(f"\nok — {len(d)} papers, {len(d.columns)} columns, "
          f"largest coverage differential "
          f"{cover.differential_pp.abs().max():.1f}pp "
          f"({cover.loc[cover.differential_pp.abs().idxmax(), 'column']})")


if __name__ == "__main__":
    demo()
