"""
Full-corpus Semantic Scholar citation refetch.

Why: OpenAlex matched ~99% of our corpus to arXiv-preprint records and misses
citations to the published versions — median 2.9x undercount, and differential
by acceptance (3.5x accepted vs 2.0x rejected). See
outputs/citation_source_comparison.md. S2 merges preprint+published versions
and also indexes OpenReview-only submissions (via MAG/OpenReview crawl).

Method, per paper:
  1. arXiv-matched papers -> S2 batch endpoint by ARXIV id (500/request).
  2. non-arXiv papers already title-matched in outputs/rejected_venues_s2_title.csv
     -> reuse (marked method=title_cached).
  3. remaining papers -> S2 /paper/search/match by title, one call each,
     recording title_sim and s2_year so match quality is tunable downstream.

Output: outputs/s2_citations_full.csv (incremental, resumable)
  paper_id, method, s2_citations, s2_title, title_sim, s2_year, s2_venue

Run: python src/fetch_citations_s2_full.py [--limit N]
"""
import os
import re
import sys
import time
import argparse
from difflib import SequenceMatcher

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OUT_CSV = "outputs/s2_citations_full.csv"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
FIELDS = "citationCount,title,year,venue"
S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
HEADERS = {"x-api-key": S2_KEY} if S2_KEY else {}
SLEEP = 1.0 if S2_KEY else 3.1  # unauthenticated: 100 req / 5 min


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def sim(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def csv_row(paper_id, method, res, title_sim=""):
    if res is None:
        return f"{paper_id},{method},,,,,\n"
    venue = str(res.get("venue") or "").replace(",", ";")
    title = str(res.get("title") or "").replace(",", ";")
    return (f"{paper_id},{method},{res.get('citationCount')},{title},"
            f"{title_sim},{res.get('year') or ''},{venue}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="title-match at most N (smoke)")
    args = parser.parse_args()

    ev = pd.read_csv("outputs/eval_table.csv")
    ax = pd.read_csv("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv",
                     low_memory=False)[["paper_id", "arxiv_id_canonical"]]
    df = ev.merge(ax, on="paper_id", how="left")

    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV)["paper_id"])
    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0

    with open(OUT_CSV, "a") as fout:
        if write_header:
            fout.write("paper_id,method,s2_citations,s2_title,title_sim,s2_year,s2_venue\n")

        # 1. arXiv batch
        ax_todo = df[df["arxiv_id_canonical"].notna() & ~df["paper_id"].isin(done)]
        print(f"arXiv batch: {len(ax_todo)} papers")
        for start in range(0, len(ax_todo), 500):
            chunk = ax_todo.iloc[start:start + 500]
            ids = ["ARXIV:" + str(a) for a in chunk["arxiv_id_canonical"]]
            for attempt in range(6):
                r = requests.post(BATCH_URL, params={"fields": FIELDS},
                                  json={"ids": ids}, headers=HEADERS, timeout=120)
                if r.status_code == 200:
                    break
                print(f"  batch HTTP {r.status_code}, retry in {20*(attempt+1)}s")
                time.sleep(20 * (attempt + 1))
            else:
                sys.exit("ERROR: batch failed repeatedly — rerun to resume.")
            for row, res in zip(chunk.itertuples(), r.json()):
                s = f"{sim(row.title, res['title']):.3f}" if res else ""
                fout.write(csv_row(row.paper_id, "arxiv_batch", res, s))
            fout.flush()
            print(f"  batch {start//500 + 1} written")
            time.sleep(SLEEP)

        # 2. reuse earlier title-match results for non-arXiv papers
        no_ax = df[df["arxiv_id_canonical"].isna() & ~df["paper_id"].isin(done)]
        cached = {}
        if os.path.exists("outputs/rejected_venues_s2_title.csv"):
            prev = pd.read_csv("outputs/rejected_venues_s2_title.csv")
            cached = {r.paper_id: r for r in prev.itertuples()}
        n_cached = 0
        for row in no_ax.itertuples():
            c = cached.get(row.paper_id)
            if c is None or pd.isna(c.s2_title):
                continue
            res = {"citationCount": c.s2_citations, "title": c.s2_title,
                   "year": "", "venue": c.s2_venue if pd.notna(c.s2_venue) else ""}
            fout.write(csv_row(row.paper_id, "title_cached", res, f"{c.title_sim:.3f}"))
            n_cached += 1
        fout.flush()
        print(f"title_cached: {n_cached} reused")

        # 3. live title match for the rest
        done2 = done | set(pd.read_csv(OUT_CSV)["paper_id"])
        rest = df[df["arxiv_id_canonical"].isna() & ~df["paper_id"].isin(done2)]
        if args.limit:
            rest = rest.head(args.limit)
        print(f"title match (live): {len(rest)} papers, ~{len(rest)*SLEEP/60:.0f} min")
        for i, row in enumerate(rest.itertuples()):
            try:
                r = requests.get(MATCH_URL, params={"query": row.title, "fields": FIELDS},
                                 headers=HEADERS, timeout=30)
                if r.status_code == 429:
                    time.sleep(60)
                    r = requests.get(MATCH_URL, params={"query": row.title, "fields": FIELDS},
                                     headers=HEADERS, timeout=30)
                res = (r.json().get("data") or [None])[0] if r.status_code == 200 else None
            except requests.RequestException as e:
                print(f"  SKIP {row.paper_id}: {e}")
                res = None
            s = f"{sim(row.title, res['title']):.3f}" if res else ""
            fout.write(csv_row(row.paper_id, "title_match", res, s))
            fout.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(rest)}")
            time.sleep(SLEEP)

    # summary
    out = pd.read_csv(OUT_CSV)
    ok = out[out["s2_citations"].notna() & (out["title_sim"].fillna(0) >= 0.9)]
    print(f"\nTotal rows: {len(out)}; matched at sim>=0.9: {len(ok)} "
          f"({len(ok)/len(df):.1%} of corpus)")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
