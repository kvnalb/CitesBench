"""
Fetch author-level covariates and venue data from OpenAlex for all eval_table papers.

Pass 1: work endpoint → per-paper author IDs + institutions
  output: outputs/paper_author_ids.csv

Pass 2: author endpoint → h_index, cited_by_count, works_count per unique author
  output: outputs/author_stats.csv

Pass 3: work endpoint (locations) → published venue per paper
  output: outputs/paper_venues.csv

All outputs are incrementally written and resumable.
Run from repo root: python src/fetch_author_stats.py
"""
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

EMAIL = open("OpenAlex.txt").read().strip()
HEADERS = {"User-Agent": f"mailto:{EMAIL}"}
SLEEP = 0.12  # ~8 req/s polite pool

CITATIONS = "output/citations_2018_2020.csv"
PASS1_OUT = "outputs/paper_author_ids.csv"
PASS2_OUT = "outputs/author_stats.csv"
PASS3_OUT = "outputs/paper_venues.csv"


def get_json(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


# ── Pass 1: work endpoint → author IDs per paper ──────────────────────────────

def run_pass1():
    cit = pd.read_csv(CITATIONS)
    papers = cit[cit["status"] == "found"][["paper_id", "openalex_id"]].dropna()

    # Resume: skip already-fetched paper_ids
    done = set()
    if os.path.exists(PASS1_OUT):
        done = set(pd.read_csv(PASS1_OUT)["paper_id"].unique())

    todo = papers[~papers["paper_id"].isin(done)]
    print(f"Pass 1: {len(done)} already done, {len(todo)} to fetch")

    with open(PASS1_OUT, "a") as fout:
        if not done:
            fout.write("paper_id,openalex_work_id,author_position,author_id,"
                       "author_name,institution_id,institution_name,country\n")

        for i, row in enumerate(todo.itertuples(), 1):
            work_id = row.openalex_id.split("/")[-1]
            try:
                data = get_json(f"https://api.openalex.org/works/{work_id}",
                                params={"mailto": EMAIL,
                                        "select": "id,authorships"})
            except Exception as e:
                print(f"  SKIP {work_id}: {e}")
                # Write a sentinel so we don't retry on resume
                fout.write(f"{row.paper_id},{work_id},ERROR,,,,,\n")
                fout.flush()
                time.sleep(SLEEP)
                continue

            authorships = data.get("authorships", [])
            if not authorships:
                fout.write(f"{row.paper_id},{work_id},0,,,,, \n")
                fout.flush()
            else:
                for a in authorships:
                    pos = a.get("author_position", "")
                    author = a.get("author") or {}
                    author_id = (author.get("id") or "").split("/")[-1]
                    author_name = (author.get("display_name") or "").replace(",", " ")
                    insts = a.get("institutions") or []
                    inst_id = (insts[0].get("id") or "").split("/")[-1] if insts else ""
                    inst_name = (insts[0].get("display_name") or "").replace(",", " ") if insts else ""
                    countries = a.get("countries", [])
                    country = countries[0] if countries else ""
                    fout.write(f"{row.paper_id},{work_id},{pos},{author_id},"
                               f"{author_name},{inst_id},{inst_name},{country}\n")
                fout.flush()

            time.sleep(SLEEP)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} works fetched")

    print("Pass 1 complete.")


# ── Pass 2: author endpoint → stats per unique author ─────────────────────────

def run_pass2():
    if not os.path.exists(PASS1_OUT):
        print("Pass 1 output not found — run pass 1 first.")
        return

    p1 = pd.read_csv(PASS1_OUT)
    all_ids = p1["author_id"].dropna().unique()
    # Filter out sentinel/error rows
    all_ids = [a for a in all_ids if a and a not in ("ERROR", "")]

    done = set()
    if os.path.exists(PASS2_OUT):
        done = set(pd.read_csv(PASS2_OUT)["author_id"].unique())

    todo = [a for a in all_ids if a not in done]
    print(f"Pass 2: {len(done)} already done, {len(todo)} to fetch")

    with open(PASS2_OUT, "a") as fout:
        if not done:
            fout.write("author_id,display_name,works_count,cited_by_count,"
                       "h_index,i10_index,last_institution_id,last_institution_name,"
                       "last_institution_country\n")

        for i, author_id in enumerate(todo, 1):
            try:
                data = get_json(
                    f"https://api.openalex.org/authors/{author_id}",
                    params={"mailto": EMAIL,
                            "select": "id,display_name,works_count,cited_by_count,"
                                      "summary_stats,last_known_institutions"})
            except Exception as e:
                print(f"  SKIP {author_id}: {e}")
                fout.write(f"{author_id},ERROR,,,,,,,\n")
                fout.flush()
                time.sleep(SLEEP)
                continue

            name = data.get("display_name", "").replace(",", " ")
            works = data.get("works_count", "")
            cites = data.get("cited_by_count", "")
            stats = data.get("summary_stats", {})
            h = stats.get("h_index", "")
            i10 = stats.get("i10_index", "")
            insts = data.get("last_known_institutions", [])
            inst_id = insts[0].get("id", "").split("/")[-1] if insts else ""
            inst_name = insts[0].get("display_name", "").replace(",", " ") if insts else ""
            inst_country = insts[0].get("country_code", "") if insts else ""

            fout.write(f"{author_id},{name},{works},{cites},{h},{i10},"
                       f"{inst_id},{inst_name},{inst_country}\n")
            fout.flush()
            time.sleep(SLEEP)

            if i % 200 == 0:
                print(f"  {i}/{len(todo)} authors fetched")

    print("Pass 2 complete.")


# ── Pass 3: locations → published venue per paper ─────────────────────────────

def run_pass3():
    cit = pd.read_csv(CITATIONS)
    papers = cit[cit["status"] == "found"][["paper_id", "openalex_id"]].dropna()

    done = set()
    if os.path.exists(PASS3_OUT):
        done = set(pd.read_csv(PASS3_OUT)["paper_id"].unique())

    todo = papers[~papers["paper_id"].isin(done)]
    print(f"Pass 3: {len(done)} already done, {len(todo)} to fetch")

    with open(PASS3_OUT, "a") as fout:
        if not done:
            fout.write("paper_id,openalex_work_id,venue_name,venue_type,"
                       "venue_is_oa,venue_url\n")

        for i, row in enumerate(todo.itertuples(), 1):
            work_id = row.openalex_id.split("/")[-1]
            try:
                data = get_json(f"https://api.openalex.org/works/{work_id}",
                                params={"mailto": EMAIL,
                                        "select": "id,locations"})
            except Exception as e:
                print(f"  SKIP {work_id}: {e}")
                fout.write(f"{row.paper_id},{work_id},ERROR,,,\n")
                fout.flush()
                time.sleep(SLEEP)
                continue

            locs = data.get("locations") or []
            # Pick first non-repository location; fall back to first repository
            published = next(
                (l for l in locs if (l.get("source") or {}).get("type") != "repository"),
                None
            )
            if published is None and locs:
                published = locs[0]

            if published:
                src = published.get("source") or {}
                venue_name = (src.get("display_name") or "").replace(",", " ")
                venue_type = src.get("type", "")
                venue_is_oa = published.get("is_oa", "")
                venue_url = (published.get("landing_page_url") or "").replace(",", " ")
            else:
                venue_name = venue_type = venue_is_oa = venue_url = ""

            fout.write(f"{row.paper_id},{work_id},{venue_name},{venue_type},"
                       f"{venue_is_oa},{venue_url}\n")
            fout.flush()
            time.sleep(SLEEP)

            if i % 100 == 0:
                print(f"  {i}/{len(todo)} venues fetched")

    print("Pass 3 complete.")


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary():
    if not os.path.exists(PASS1_OUT) or not os.path.exists(PASS2_OUT):
        return
    p1 = pd.read_csv(PASS1_OUT)
    p2 = pd.read_csv(PASS2_OUT)
    papers_covered = p1[p1["author_id"].notna() & (p1["author_id"] != "ERROR")]["paper_id"].nunique()
    authors = len(p2[p2["display_name"] != "ERROR"])
    with_h = p2["h_index"].notna().sum()
    with_inst = p2["last_institution_name"].notna().sum()
    print(f"\nSummary:")
    print(f"  Papers with author data: {papers_covered}")
    print(f"  Unique authors fetched:  {authors}")
    print(f"  With h_index:            {with_h} ({with_h/max(authors,1):.0%})")
    print(f"  With institution:        {with_inst} ({with_inst/max(authors,1):.0%})")


if __name__ == "__main__":
    run_pass1()
    run_pass2()
    run_pass3()
    print_summary()
