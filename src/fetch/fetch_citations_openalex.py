"""
Fetch OpenAlex citation counts for ICLR submissions.

For each paper in the DB, searches OpenAlex by title and accepts the top result
if the returned title is sufficiently similar (default: ≥0.85). Records the
openalex_id, DOI, and cited_by_count.

Resumable: skips paper_ids already present in the output CSV.

Usage:
    python src/fetch/fetch_citations_openalex.py [--years 2018 2019 2020] [--threshold 0.85]
"""
import argparse
import sqlite3
import time
import urllib.request
import urllib.parse
import json
import os
from difflib import SequenceMatcher

import pandas as pd

EMAIL  = open("OpenAlex.txt").read().strip()


def fresh_output_path(years: list[int]) -> str:
    """Always return a new filename; never clobber an existing file."""
    base = f"output/citations_{'_'.join(map(str, years))}"
    path = f"{base}.csv"
    i = 1
    while os.path.exists(path):
        path = f"{base}_v{i}.csv"
        i += 1
    return path
HEADERS = {"User-Agent": f"research/1.0 ({EMAIL})"}


def search_openalex(title: str) -> dict | None:
    """Return the top OpenAlex work for a title search, or None on failure."""
    params = urllib.parse.urlencode({
        "search":   title,
        "per-page": 1,
        "select":   "id,doi,display_name,cited_by_count",
        "mailto":   EMAIL,
    })
    url = f"https://api.openalex.org/works?{params}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            data = json.loads(urllib.request.urlopen(req, timeout=15).read())
            results = data.get("results", [])
            return results[0] if results else None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt * 10
                time.sleep(wait)
            else:
                return None
        except Exception:
            return None
    return None


def title_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020])
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Minimum title similarity to accept a match")
    args = parser.parse_args()

    con = sqlite3.connect("data/gen_review.db")
    papers = pd.read_sql(
        f"SELECT id as paper_id, title, when_submitted as year FROM SUBMISSION "
        f"WHERE when_submitted IN ({','.join('?'*len(args.years))})",
        con, params=args.years,
    )
    con.close()
    print(f"Papers to process: {len(papers)}")

    os.makedirs("output", exist_ok=True)
    output = fresh_output_path(args.years)
    print(f"Writing to {output}")

    with open(output, "w") as f:
        f.write("paper_id,title,year,doi,openalex_citations,openalex_id,status\n")

        for i, row in enumerate(papers.itertuples(), 1):
            result = search_openalex(row.title)

            if result is None:
                status = "not_found"
                doi = openalex_id = openalex_citations = ""
            else:
                oa_title = result.get("display_name") or ""
                sim = title_sim(row.title, oa_title)
                if sim >= args.threshold:
                    status = "found"
                    doi             = result.get("doi") or ""
                    openalex_id     = result.get("id") or ""
                    openalex_citations = result.get("cited_by_count", "")
                else:
                    status = "not_found"
                    doi = openalex_id = openalex_citations = ""

            # escape title for CSV (wrap in quotes, escape internal quotes)
            safe_title = '"' + row.title.replace('"', '""') + '"'
            f.write(f"{row.paper_id},{safe_title},{row.year},{doi},"
                    f"{openalex_citations},{openalex_id},{status}\n")
            f.flush()

            if i % 100 == 0:
                print(f"  {i}/{len(papers)} processed")

            time.sleep(0.12)  # ~8 req/s — polite pool limit

    final = pd.read_csv(output)
    found = (final.status == "found").sum()
    print(f"\nDone. {found}/{len(final)} found ({found/len(final):.1%})")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
