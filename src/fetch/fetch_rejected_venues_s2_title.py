"""
Fetch venues for rejected ICLR papers that have NO arXiv id, via Semantic Scholar
title match (/graph/v1/paper/search/match).

Skips papers already covered by: paper_venues.csv (non-repository),
rejected_venues_s2.csv (real venue), oa_title_match_venues.csv.

Output: outputs/rejected_venues_s2_title.csv
  paper_id, s2_venue, s2_venue_type, s2_venue_id, s2_citations, s2_title, title_sim

Records title_sim (difflib ratio on normalized titles) so the match threshold
(recommend >= 0.9) can be tuned downstream without refetching.

Resumable: skips already-fetched paper_ids on restart.
Run from repo root: python src/fetch/fetch_rejected_venues_s2_title.py
"""
import csv
import os
import re
import time
from difflib import SequenceMatcher

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OUT = "outputs/rejected_venues_s2_title.csv"
FIELDS = ["paper_id", "s2_venue", "s2_venue_type", "s2_venue_id",
          "s2_citations", "s2_title", "title_sim"]

S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
HEADERS = {"x-api-key": S2_KEY} if S2_KEY else {}
SLEEP = 1.0 if S2_KEY else 3.1  # unauthenticated: 100 req / 5 min


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def build_input():
    et = pd.read_csv("outputs/eval_table.csv")
    cit = pd.read_csv("output/citations_2018_2020.csv")
    oa = pd.read_csv("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv")

    rejected = et[~et["decision"].str.startswith("Accept", na=False)]
    rejected = rejected[rejected["decision"] != "Invite to Workshop Track"].copy()

    # arxiv-id papers are handled by fetch_rejected_venues_s2.py
    doi_map = cit[["paper_id", "doi"]].drop_duplicates("paper_id")
    rejected = rejected.merge(doi_map, on="paper_id", how="left")
    has_arxiv = rejected["doi"].astype(str).str.contains(
        r"arxiv[./]\d{4}\.\d{4,5}", case=False, regex=True)
    rdd_ids = set(oa[["paper_id", "arxiv_id"]].dropna()["paper_id"])
    rejected = rejected[~has_arxiv & ~rejected["paper_id"].isin(rdd_ids)]

    covered = set()
    pv = pd.read_csv("outputs/paper_venues.csv")
    covered |= set(pv[pv["venue_name"].notna()
                      & (pv["venue_type"] != "repository")]["paper_id"])
    s2 = pd.read_csv("outputs/rejected_venues_s2.csv")
    s2v = s2["s2_venue"].fillna("")
    covered |= set(s2[~s2v.isin(["NOT_FOUND", "ERROR", "arXiv.org"])
                      & (s2v.str.strip() != "")]["paper_id"])
    if os.path.exists("outputs/oa_title_match_venues.csv"):
        covered |= set(pd.read_csv("outputs/oa_title_match_venues.csv")["paper_id"])

    return rejected[~rejected["paper_id"].isin(covered)][
        ["paper_id", "title"]].reset_index(drop=True)


def fetch_match(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    for attempt in range(3):
        try:
            r = requests.get(
                url, headers=HEADERS,
                params={"query": title,
                        "fields": "title,year,venue,publicationVenue,citationCount"},
                timeout=15)
            if r.status_code == 404:  # no match
                return {}
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            data = r.json().get("data", [])
            return data[0] if data else {}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    return {}


if __name__ == "__main__":
    papers = build_input()

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["paper_id"].unique())

    todo = papers[~papers["paper_id"].isin(done)]
    print(f"Rejected papers without arxiv_id, uncovered: {len(papers)}")
    print(f"Already done: {len(done)}, to fetch: {len(todo)} "
          f"(~{len(todo) * SLEEP / 60:.0f} min)")

    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=FIELDS)
        if new_file:
            w.writeheader()

        for i, row in enumerate(todo.itertuples(), 1):
            try:
                d = fetch_match(row.title)
            except Exception as e:
                print(f"  SKIP {row.paper_id}: {e}")
                w.writerow({"paper_id": row.paper_id, "s2_venue": "ERROR"})
                fout.flush()
                time.sleep(SLEEP)
                continue

            if not d:
                w.writerow({"paper_id": row.paper_id, "s2_venue": "NOT_FOUND"})
                fout.flush()
                time.sleep(SLEEP)
                continue

            pv = d.get("publicationVenue") or {}
            sim = SequenceMatcher(None, norm(row.title), norm(d.get("title"))).ratio()
            w.writerow({
                "paper_id": row.paper_id,
                "s2_venue": d.get("venue") or pv.get("name") or "",
                "s2_venue_type": pv.get("type", ""),
                "s2_venue_id": pv.get("id", ""),
                "s2_citations": d.get("citationCount", ""),
                "s2_title": (d.get("title") or "")[:120],
                "title_sim": round(sim, 3),
            })
            fout.flush()
            time.sleep(SLEEP)

            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ({i / len(todo) * 100:.0f}%)")

    df = pd.read_csv(OUT)
    real = df[(df["title_sim"].fillna(0) >= 0.9)
              & ~df["s2_venue"].fillna("").isin(["", "NOT_FOUND", "ERROR", "arXiv.org"])]
    print(f"\nDone. {len(df)} fetched; {len(real)} with real venue at sim>=0.9.")
    print(real["s2_venue"].value_counts().head(12))
