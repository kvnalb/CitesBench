"""
Build the single flat evaluation table used by all downstream eval scripts.

Joins: SUBMISSION + REVIEW (aggregated) + citations + paper_fields
Computes: field×year citation percentile rank, N accepts per year

Output: outputs/eval_table.csv
"""
import os
import sqlite3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

os.makedirs("outputs", exist_ok=True)
DB = "data/gen_review.db"
OUT = "outputs/eval_table.csv"

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
con.close()

# Parse "6: Marginally above acceptance threshold" → 6.0
reviews_raw["rating_num"] = reviews_raw["rating"].str.extract(r"^(\d+)").astype(float)
reviews = (
    reviews_raw.groupby("paper_id")["rating_num"]
    .agg(mean_rating="mean", rating_std="std", n_reviews="count")
    .reset_index()
)

citations = pd.read_csv("output/citations_2018_2020.csv")[
    ["paper_id", "openalex_citations", "status"]
]

fields_path = "outputs/paper_fields.csv"
if os.path.exists(fields_path):
    fields = pd.read_csv(fields_path)[["id", "field"]].rename(columns={"id": "paper_id"})
else:
    fields = pd.DataFrame(columns=["paper_id", "field"])
    print("Warning: paper_fields.csv not found — field column will be empty")

df = (
    papers
    .merge(reviews, on="paper_id", how="left")
    .merge(citations, on="paper_id", how="left")
    .merge(fields, on="paper_id", how="left")
)

# Only use citations where OpenAlex found the paper
df.loc[df["status"] != "found", "openalex_citations"] = np.nan

# field×year citation percentile rank (0–1) among papers with citations
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
