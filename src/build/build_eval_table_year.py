"""
Build an eval table for a single submission year — used for the out-of-sample test.

The 2018-2020 corpus is contaminated for any model trained after 2021: the papers, the
decisions and the citation trajectories are all in the training data. The out-of-sample
design pairs a model with a known cutoff against papers that postdate it:

  paper year   submitted   decisions   Llama-3-70B (cutoff Dec 2023) could have seen
  2018-2020    2017-2019   2018-2020   papers and outcomes
  2024         Sep 2023    Jan 2024    preprints, but not outcomes
  2025         Sep 2024    Jan 2025    neither

Citations come from OpenAlex by arXiv DOI (identifier lookups are free and unmetered;
title search is not), using the arXiv IDs from src/resolve_arxiv_ids.py. Every row carries
`citations_fetched_at` — accept and reject are always measured on the same day, which the
old pipeline did not guarantee.

`Withdrawn` is kept as its own category rather than folded into Reject. It does not exist
in 2018-2020 and is 1,482 papers in 2024, 2,945 in 2025 — mostly post-review strategic
exits, which are neither accepts nor comparable rejects.

Output: outputs/eval_table_<year>.csv

Run: python src/build_eval_table_year.py --year 2025
"""
import os
import re
import time
import argparse
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

os.makedirs("outputs", exist_ok=True)
DB = "data/gen_review.db"
MAILTO = open("OpenAlex.txt").read().strip() if os.path.exists("OpenAlex.txt") else ""
OA_URL = "https://api.openalex.org/works"


def decision_class(d):
    d = str(d or "")
    if d.startswith("Accept"):
        return "accept"
    if d.startswith("Withdraw"):
        return "withdrawn"
    if d.startswith("Desk"):
        return "desk_reject"
    if d.startswith("Reject"):
        return "reject"
    return "other"


def fetch_openalex(arxiv_ids, chunk=50):
    """Free identifier lookups (1 credit / 50 works). Returns arxiv_id -> record."""
    out, t0 = {}, time.time()
    for i in range(0, len(arxiv_ids), chunk):
        ch = arxiv_ids[i:i + chunk]
        f = "doi:" + "|".join(f"https://doi.org/10.48550/arxiv.{a}" for a in ch)
        try:
            r = requests.get(OA_URL, params={
                "filter": f, "per-page": chunk, "mailto": MAILTO,
                "select": "id,doi,title,publication_year,cited_by_count,type"}, timeout=60)
        except requests.RequestException:
            time.sleep(2); continue
        if r.status_code != 200:
            print(f"  OpenAlex {r.status_code} at chunk {i // chunk + 1}: {r.text[:80]}")
            if r.status_code == 429:
                break                       # out of credits — stop, keep what we have
            continue
        for w in r.json().get("results", []):
            aid = (w.get("doi") or "").lower().split("arxiv.")[-1]
            out[aid] = {"openalex_id": w["id"], "citations": w["cited_by_count"],
                        "oa_year": w["publication_year"], "oa_type": w["type"]}
        if (i // chunk) % 20 == 0:
            print(f"  {i + len(ch):,}/{len(arxiv_ids):,} looked up "
                  f"({time.time() - t0:.0f}s)", flush=True)
        time.sleep(0.15)
    return out


def build(year):
    con = sqlite3.connect(DB)
    papers = pd.read_sql(
        "SELECT id AS paper_id, title, abstract, keywords, when_submitted AS year, decision "
        f"FROM SUBMISSION WHERE when_submitted = {int(year)}", con)
    reviews = pd.read_sql(
        "SELECT r.paper_id, r.rating FROM REVIEW r JOIN SUBMISSION s ON r.paper_id = s.id "
        f"WHERE s.when_submitted = {int(year)}", con)
    con.close()

    # Same parse as build_eval_table.py. Rows whose `rating` is a decision string
    # (the AC decision note, scraped into REVIEW) become NaN and drop out of the count.
    reviews["rating_num"] = reviews["rating"].str.extract(r"^(\d+)").astype(float)
    agg = (reviews.dropna(subset=["rating_num"]).groupby("paper_id")["rating_num"]
           .agg(mean_rating="mean", rating_std="std", n_reviews="count").reset_index())

    df = papers.merge(agg, on="paper_id", how="left")
    df["decision_class"] = df["decision"].map(decision_class)
    df["accepted"] = df["decision_class"].eq("accept").astype(int)

    res = pd.read_csv("outputs/arxiv_resolution.csv", low_memory=False,
                      dtype={"arxiv_id": str, "paper_id": str})
    res = res[res["matched"].eq(1)][["paper_id", "arxiv_id", "match_rule", "title_sim"]]
    df = df.merge(res, on="paper_id", how="left")
    have = df["arxiv_id"].dropna().unique().tolist()
    print(f"{len(df):,} submissions in {year}; {len(have):,} with an arXiv ID "
          f"({len(have)/len(df):.1%})")

    recs = fetch_openalex(have)
    stamp = datetime.now(timezone.utc).date().isoformat()
    df["arxiv_id_l"] = df["arxiv_id"].str.lower()
    oa = pd.DataFrame([{"arxiv_id_l": k, **v} for k, v in recs.items()])
    df = df.merge(oa, on="arxiv_id_l", how="left") if len(oa) else df.assign(
        openalex_id=np.nan, citations=np.nan, oa_year=np.nan, oa_type=np.nan)
    df["citations_fetched_at"] = stamp
    df = df.drop(columns=["arxiv_id_l"])

    # Percentile rank within year, over papers with a citation count. No field split:
    # field labels are 60% complete and unvalidated, so making the outcome depend on
    # them costs coverage for nothing.
    df["citation_pct_rank"] = df["citations"].rank(pct=True)

    out = f"outputs/eval_table_{year}.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(df["decision_class"].value_counts().to_string())
    print(f"\ncitation coverage: {df['citations'].notna().sum():,}/{len(df):,} "
          f"({df['citations'].notna().mean():.1%}), fetched {stamp}")
    cov = df.groupby("decision_class")["citations"].agg(
        n="size", have=lambda s: s.notna().mean(), median="median")
    print(cov.to_string())
    print(f"\nreviews: {df['n_reviews'].notna().sum():,} papers with parsed ratings, "
          f"mean {df['mean_rating'].mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    build(ap.parse_args().year)
