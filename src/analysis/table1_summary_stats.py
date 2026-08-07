"""
Table 1: summary statistics for the full ICLR 2018-2020 corpus and the RDD sample.

Prints the numbers that populate outputs/table1_summary_stats.tex. Pure recompute,
no API calls.

Run: python src/analysis/table1_summary_stats.py
"""
import os
import sqlite3

import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)


def load_full():
    ev = pd.read_csv("outputs/eval_table.csv")
    s2 = pd.read_csv("outputs/s2_citations_full.csv")
    gate = s2["s2_citations"].notna() & (
        (s2["method"] == "arxiv_batch") | (s2["title_sim"].fillna(0) >= 0.9))
    ev = ev.merge(s2.loc[gate, ["paper_id", "s2_citations"]].drop_duplicates("paper_id"),
                  on="paper_id", how="left")
    ax = pd.read_csv("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv", low_memory=False)
    ax["n_authors"] = ax["arxiv_authors"].str.split(",").apply(
        lambda x: len(x) if isinstance(x, list) else np.nan)
    ev = ev.merge(ax[["paper_id", "n_authors"]].drop_duplicates("paper_id"),
                  on="paper_id", how="left")
    con = sqlite3.connect("data/gen_review.db")
    rev = pd.read_sql("select paper_id, rating from REVIEW", con)
    rev["rnum"] = rev["rating"].str.extract(r"^(\d+)").astype(float)
    agg = rev.groupby("paper_id")["rnum"].agg(["min", "max"]).reset_index()
    ev = ev.merge(agg, on="paper_id", how="left")
    ev["accepted"] = ev["decision"].str.startswith("Accept", na=False)
    return ev


def load_rdd():
    ax = pd.read_csv("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv", low_memory=False)
    dm = ax[(ax["year"] <= 2020) & ax["in_year_specific_rdd_sample"].astype(bool)
            & ax["openalex_matched"].astype(bool)].copy()
    dm["n_authors"] = dm["arxiv_authors"].str.split(",").apply(
        lambda x: len(x) if isinstance(x, list) else np.nan)
    ev = pd.read_csv("outputs/eval_table.csv")
    s2 = pd.read_csv("outputs/s2_citations_full.csv")
    gate = s2["s2_citations"].notna() & (
        (s2["method"] == "arxiv_batch") | (s2["title_sim"].fillna(0) >= 0.9))
    ev2 = ev.merge(s2.loc[gate, ["paper_id", "s2_citations"]].drop_duplicates("paper_id"),
                   on="paper_id", how="left")
    dm = dm.merge(ev2[["paper_id", "field", "committee_rating", "s2_citations"]],
                  on="paper_id", how="left")
    con = sqlite3.connect("data/gen_review.db")
    rev = pd.read_sql("select paper_id, rating from REVIEW", con)
    rev["rnum"] = rev["rating"].str.extract(r"^(\d+)").astype(float)
    agg = rev.groupby("paper_id")["rnum"].agg(["min", "max"]).reset_index()
    dm = dm.merge(agg, on="paper_id", how="left")
    dm["accepted"] = dm["accepted"].astype(int)
    dm = dm.rename(columns={"openalex_cited_by_count": "openalex_citations"})
    return dm


def report(df, label):
    n = len(df)
    print(f"\n=== {label}: N={n} ===")
    print("by year:", df["year"].value_counts().sort_index().to_dict())
    print("by field:", df["field"].value_counts(dropna=False).to_dict())
    print(f"n_authors: mean {df.n_authors.mean():.2f} sd {df.n_authors.std():.2f} "
          f"cov {df.n_authors.notna().sum()}/{n}")
    print(f"OA citations: mean {df.openalex_citations.mean():.2f} "
          f"median {df.openalex_citations.median()} sd {df.openalex_citations.std():.2f}")
    print(f"S2 citations: mean {df.s2_citations.mean():.2f} "
          f"median {df.s2_citations.median()} sd {df.s2_citations.std():.2f} "
          f"cov {df.s2_citations.notna().sum()}/{n}")
    print(f"log1p OA: mean {np.log1p(df.openalex_citations).mean():.3f} "
          f"sd {np.log1p(df.openalex_citations).std():.3f}")
    print(f"mean_rating: mean {df.mean_rating.mean():.3f} sd {df.mean_rating.std():.3f}")
    print(f"min rating avg: {df['min'].mean():.3f}  max rating avg: {df['max'].mean():.3f} "
          f"cov {df['min'].notna().sum()}/{n}")
    print(f"committee_rating: mean {df.committee_rating.mean():.3f} "
          f"sd {df.committee_rating.std():.3f} cov {df.committee_rating.notna().sum()}/{n}")
    print(f"accepted: {int(df.accepted.sum())} ({df.accepted.mean():.4f})")
    print(f"n_reviews mean: {df.n_reviews.mean():.2f}")


if __name__ == "__main__":
    report(load_full(), "FULL CORPUS")
    report(load_rdd(), "RDD SAMPLE")
