"""
Build the single flat evaluation table used by all downstream eval scripts.

Joins: SUBMISSION + REVIEW (aggregated) + citations + paper_fields
       + committee/decision-head LLM run (consistent gpt-oss-20b decision head)
Computes: field×year citation percentile rank, N accepts per year

Output: outputs/eval_table.csv
"""
import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

os.makedirs("outputs", exist_ok=True)
DB = "data/gen_review.db"
OUT = "outputs/eval_table.csv"
DECISION_HEAD_CSV = "outputs/all_paper_results_consistent_gptoss20b.csv"
SINGLE_CALL_CSV = "outputs/single_call_iclr1820_gemma.csv"

con = sqlite3.connect(DB)

papers = pd.read_sql(
    "SELECT id AS paper_id, title, when_submitted AS year, decision "
    "FROM SUBMISSION WHERE when_submitted IN (2018,2019,2020)",
    con,
)

reviews_raw = pd.read_sql(
    "SELECT paper_id, rating FROM REVIEW",
    con,
)
llm_raw = pd.read_sql("SELECT paper_id, type, rating FROM GENAI_REVIEW", con)
con.close()

# Parse "6: Marginally above acceptance threshold" → 6.0
reviews_raw["rating_num"] = reviews_raw["rating"].str.extract(r"^(\d+)").astype(float)
reviews = (
    reviews_raw.groupby("paper_id")["rating_num"]
    .agg(mean_rating="mean", rating_std="std", n_reviews="count")
    .reset_index()
)
# single-review papers have NaN std (ddof=1); treat as zero disagreement
reviews["rating_std"] = reviews["rating_std"].fillna(0)
llm_raw["rating_num"] = llm_raw["rating"].str.extract(r"^(\d+)").astype(float)
llm_scores = llm_raw.pivot_table(
    index="paper_id", columns="type", values="rating_num", aggfunc="mean"
).rename(columns={"neutral": "llm_neutral_rating", "positive": "llm_positive_rating",
                  "negative": "llm_negative_rating"}).reset_index()
llm_scores["llm_mean_rating"] = llm_scores[
    ["llm_neutral_rating", "llm_positive_rating", "llm_negative_rating"]
].mean(axis=1)

# Canonical citation table (src/build/build_citations.py) — S2, tiered, one rule
# for both eras. Replaces the old OpenAlex pull, whose coverage was 89.0% for
# accepted papers and 62.7% for rejected: a 26.3 point differential on the very
# axis this project measures. S2 tier A+B runs 0.8. Unmatched papers stay NaN and
# are never imputed as zero.
#
# The column keeps the name `openalex_citations` for now. That is a lie the repo
# already tells (dashboard.py has been assigning S2 counts to it since July), and
# renaming it touches 121 references across 22 files including the leakage probes
# — a separate mechanical PR. `citation_source` records the truth meanwhile.
CITATIONS_CSV = "outputs/citations.csv"
if not os.path.exists(CITATIONS_CSV):
    sys.exit(f"{CITATIONS_CSV} missing — run python src/build/build_citations.py first")
_cit = pd.read_csv(CITATIONS_CSV)
citations = _cit.rename(columns={"citations": "openalex_citations",
                                 "source": "citation_source",
                                 "tier": "citation_tier"})[
    ["paper_id", "openalex_citations", "citation_source", "citation_tier"]
]
citations["status"] = "found"     # the table only carries matched papers

fields_path = "outputs/paper_fields.csv"
if os.path.exists(fields_path):
    fields = pd.read_csv(fields_path)[["id", "field"]].rename(columns={"id": "paper_id"})
else:
    fields = pd.DataFrame(columns=["paper_id", "field"])
    print("Warning: paper_fields.csv not found — field column will be empty")

if os.path.exists(DECISION_HEAD_CSV):
    decision_head = pd.read_csv(DECISION_HEAD_CSV)[
        ["paper_id", "committee_rating", "deepseek_p_accept", "decision_head_model"]
    ]
else:
    decision_head = pd.DataFrame(
        columns=["paper_id", "committee_rating", "deepseek_p_accept", "decision_head_model"]
    )
    print(f"Warning: {DECISION_HEAD_CSV} not found — run "
          "src/build/build_consistent_decision_head.py first. "
          "committee_rating/deepseek_p_accept will be empty.")

# Single-call baseline (src/probes/run_single_call_baseline.py): the 1-call control
# for the 9-call council. The runner appends, so a paper that failed and later
# succeeded has both rows — keep the last successful row per paper, never a mean.
if os.path.exists(SINGLE_CALL_CSV):
    _sc = pd.read_csv(SINGLE_CALL_CSV, low_memory=False)
    _sc = _sc[_sc["rating"].notna()].drop_duplicates("paper_id", keep="last")
    single_call = _sc[["paper_id", "rating"]].rename(
        columns={"rating": "single_call_rating"})
else:
    single_call = pd.DataFrame(columns=["paper_id", "single_call_rating"])
    print(f"Warning: {SINGLE_CALL_CSV} not found — single_call_rating will be empty.")


df = (
    papers
    .merge(reviews, on="paper_id", how="left", validate="1:1")
    .merge(citations, on="paper_id", how="left", validate="1:1")
    .merge(fields, on="paper_id", how="left", validate="1:1")
    .merge(llm_scores, on="paper_id", how="left", validate="1:1")
    .merge(decision_head, on="paper_id", how="left", validate="1:1")
    .merge(single_call, on="paper_id", how="left", validate="1:1")
)
assert len(df) == len(papers), (
    f"merge fanned the table out: {len(papers)} papers in, {len(df)} rows out")

# The canonical table only contains matched papers, so a missing row means
# "no usable match" and must stay NaN — never zero.
df.loc[df["status"] != "found", "openalex_citations"] = np.nan

# field×year citation percentile rank (0–1) among papers with citations.
#
# COMPUTED BUT NOT USED AS A HEADLINE OUTCOME. `paper_fields.csv` covers 2,726 of
# 4,567 papers, and the coverage is not independent of the decision: 63.7% of
# accepted papers carry a label against 57.7% of rejected ones. That leaves this
# column defined for 57.7% of the pool with an 8.1 pp accept/reject gap of its own,
# on the same axis the benchmark measures. 1,749 of the 2,726 labels are
# `theory_methods`, so most of the normalization happens inside one catch-all.
#
# The column stays because the leakage scripts read it. Every paper exhibit goes
# through src/figures/spec.py, which pins MODE = "raw"; run_eval.py no longer
# writes the normalized arm unless asked. Fixing this means classifying the 1,841
# unlabeled papers, which is a project, not a patch.
#
# `groupby` drops NaN keys, so unlabeled papers keep NaN here rather than landing
# in a bucket they do not belong to.
df["citation_pct_rank"] = np.nan
for (field, year), grp in df.groupby(["field", "year"]):
    mask = df["field"].eq(field) & df["year"].eq(year) & df["openalex_citations"].notna()
    vals = df.loc[mask, "openalex_citations"]
    df.loc[mask, "citation_pct_rank"] = vals.rank(pct=True)

df.to_csv(OUT, index=False)

# Summary
n_accepts = df[df["decision"].str.startswith("Accept", na=False)].groupby("year").size()
print(f"Wrote {len(df):,} papers to {OUT}")
print("\nN accepts per year (used as N for all regimes):")
print(n_accepts.to_string())
print("\nField distribution:")
print(df["field"].value_counts().to_string())
print(f"\nCitation coverage: {df['openalex_citations'].notna().sum():,} / {len(df):,} papers")
print(f"Citation source: {df['citation_source'].dropna().unique()}")
print("Tier mix:", df["citation_tier"].value_counts(dropna=False).to_dict())
_acc = df["decision"].str.startswith("Accept", na=False)
_cov = df.groupby(_acc)["openalex_citations"].apply(lambda x: x.notna().mean())
if len(_cov) == 2:
    print(f"Coverage by decision: accepted {_cov[True]:.1%}  rejected {_cov[False]:.1%}"
          f"  differential {abs(_cov[True]-_cov[False])*100:.1f} pp  (OpenAlex was 26.3)")
