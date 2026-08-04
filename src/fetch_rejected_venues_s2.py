"""
Fetch where rejected ICLR papers were eventually published, via Semantic Scholar.

Uses arxiv IDs (from citations CSV + OpenAlex RDD CSV) to look up S2's
merged proceedings record, which carries the correct venue/conference name.

Output: outputs/rejected_venues_s2.csv
  paper_id, arxiv_id, s2_venue, s2_venue_type, s2_venue_id, s2_citations, s2_title

Resumable: skips already-fetched paper_ids on restart.
Run from repo root: python src/fetch_rejected_venues_s2.py
"""
import os
import re
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OUT = "outputs/rejected_venues_s2.csv"

S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
HEADERS = {"x-api-key": S2_KEY} if S2_KEY else {}
SLEEP = 1.0 if S2_KEY else 3.1   # 1 req/s with key, ~20 req/min without


def build_input():
    et = pd.read_csv("outputs/eval_table.csv")
    cit = pd.read_csv("output/citations_2018_2020.csv")
    oa = pd.read_csv("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv")

    rejected = et[~et["decision"].str.startswith("Accept", na=False)].copy()

    def extract_arxiv(doi):
        if not isinstance(doi, str):
            return None
        m = re.search(r"arxiv[./](\d{4}\.\d{4,5})", doi, re.IGNORECASE)
        return m.group(1) if m else None

    doi_map = cit[["paper_id", "doi"]].drop_duplicates("paper_id")
    rejected = rejected.merge(doi_map, on="paper_id", how="left")
    rejected["arxiv_id"] = rejected["doi"].apply(extract_arxiv)

    rdd_ids = oa[["paper_id", "arxiv_id"]].dropna().drop_duplicates("paper_id")
    rejected = rejected.merge(rdd_ids, on="paper_id", how="left", suffixes=("", "_rdd"))
    rejected["arxiv_id"] = rejected["arxiv_id"].fillna(rejected["arxiv_id_rdd"])

    return rejected[rejected["arxiv_id"].notna()][
        ["paper_id", "arxiv_id", "year", "openalex_citations"]
    ].reset_index(drop=True)


def fetch_s2(arxiv_id):
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    for attempt in range(3):
        try:
            r = requests.get(
                url,
                headers=HEADERS,
                params={"fields": "title,year,venue,publicationVenue,citationCount"},
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(10)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(5)


if __name__ == "__main__":
    papers = build_input()

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["paper_id"].unique())

    todo = papers[~papers["paper_id"].isin(done)]
    print(f"Rejected papers with arxiv_id: {len(papers)}")
    print(f"Already done: {len(done)}, to fetch: {len(todo)}")
    print(f"API key: {'yes' if S2_KEY else 'no (slow mode, ~{:.0f} min)'.format(len(todo)*SLEEP/60)}")

    with open(OUT, "a") as fout:
        if not done:
            fout.write("paper_id,arxiv_id,s2_venue,s2_venue_type,s2_venue_id,"
                       "s2_citations,s2_title\n")

        for i, row in enumerate(todo.itertuples(), 1):
            try:
                d = fetch_s2(row.arxiv_id)
            except Exception as e:
                print(f"  SKIP {row.arxiv_id}: {e}")
                fout.write(f"{row.paper_id},{row.arxiv_id},ERROR,,,, \n")
                fout.flush()
                time.sleep(SLEEP)
                continue

            if not d or not isinstance(d, dict) or "paperId" not in d:
                fout.write(f"{row.paper_id},{row.arxiv_id},NOT_FOUND,,,, \n")
                fout.flush()
                time.sleep(SLEEP)
                continue

            pv = d.get("publicationVenue") or {}
            venue = (d.get("venue") or pv.get("name") or "").replace(",", " ")
            venue_type = pv.get("type", "")
            venue_id = (pv.get("id") or "")
            s2_cites = d.get("citationCount", "")
            s2_title = (d.get("title") or "").replace(",", " ")[:80]

            fout.write(f"{row.paper_id},{row.arxiv_id},{venue},{venue_type},"
                       f"{venue_id},{s2_cites},{s2_title}\n")
            fout.flush()
            time.sleep(SLEEP)

            if i % 50 == 0:
                pct = i / len(todo) * 100
                print(f"  {i}/{len(todo)} ({pct:.0f}%) — last: {venue[:40] or 'arXiv'}")

    # Summary
    df = pd.read_csv(OUT)
    df = df[df["s2_venue"] != "ERROR"]
    non_arxiv = df[~df["s2_venue"].str.contains("arXiv|preprint", case=False, na=True)
                   & df["s2_venue"].notna() & (df["s2_venue"] != "")]
    print(f"\nDone. {len(df)} papers fetched.")
    print(f"Published elsewhere (non-arXiv): {len(non_arxiv)} ({len(non_arxiv)/len(df):.0%})")
    print("\nTop venues for rejected papers:")
    print(non_arxiv["s2_venue"].value_counts().head(12))
