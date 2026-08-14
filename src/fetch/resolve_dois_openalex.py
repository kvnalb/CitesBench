"""
Resolve paper titles to DOIs via OpenAlex, so S2 can be queried in batches.

Why: S2's unauthenticated pool is saturated — measured 3 of every 4 requests
returning 429, which with the backoff makes a title match cost ~30s. The batch
endpoint takes 500 identifiers per call, but only identifiers. OpenAlex title
search is not throttled the same way (polite pool, ~0.5s/call), so it can supply
the identifiers that turn ~2,000 throttled S2 calls into a handful of batch calls.

What this does NOT do is trust OpenAlex. Its top-1 result is frequently the wrong
paper: searching a 2025 ICLR title returned the 2020 BYOL paper and, separately,
LeCun 1998. So a candidate is accepted only when the title matches at >= 0.95 AND
at least one author surname overlaps ours. Three candidates are considered rather
than one, which costs nothing extra per call.

Downstream, these DOIs are tagged `doi_oa_title` rather than `doi`, so assign_tiers
cannot promote them to tier A on the strength of "it has an identifier". They must
earn tier B on title, year and author overlap like any other fuzzy match — and the
year window is what rejects the BYOL and LeNet cases.

MEASURED RESULT — this does not work for the population it was written for.
Run against the 694 ICLR 2025 accepted papers that S2 had not yet matched:
**3 verified DOIs out of 596 (0.5%), and 91% returned zero OpenAlex results at
all.** Not a verification-threshold problem — author coverage was 692/694 and the
accepted matches were all title_sim 1.0 with >=2 shared authors. OpenAlex simply
does not index these papers. They are 2025 ICLR papers with no arXiv preprint and
no publisher DOI, i.e. they exist on OpenReview and nowhere else. S2 does carry
them (~93% match rate on the same population via title search), because it crawls
OpenReview and OpenAlex does not.

So the shortcut is kept for a population that has DOIs — earlier ICLR years, or
any corpus that is actually published — and is useless for recent OpenReview-only
work. The throttled S2 title path remains the only route for that tail.

Output: outputs/openalex_doi_resolution{_suffix}.csv (incremental, resumable)

Run: python src/fetch/resolve_dois_openalex.py --eval-table outputs/eval_table.csv
"""
import argparse
import csv
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_citations_s2_v2 import last_names, sim, load_inputs

OUT_CSV = "outputs/openalex_doi_resolution.csv"
API = "https://api.openalex.org/works"
EMAIL = open("OpenAlex.txt").read().strip()
SELECT = "id,doi,display_name,publication_year,authorships"
SLEEP = 0.15                      # polite pool allows ~10/s; stay well under
MIN_SIM = 0.95
COLS = ["paper_id", "oa_id", "doi", "oa_title", "title_sim", "author_overlap",
        "oa_year", "n_candidates", "accepted"]


def best_candidate(title, known):
    """Top-3 by relevance, keep the best that clears title AND author checks."""
    try:
        r = requests.get(API, params={"search": title, "per-page": 3,
                                      "select": SELECT, "mailto": EMAIL}, timeout=25)
    except requests.RequestException:
        return None, 0
    if r.status_code != 200:
        return None, 0
    results = r.json().get("results", []) or []
    best, best_key = None, None
    for w in results:
        s = sim(title, w.get("display_name") or "")
        names = last_names([(a.get("author") or {}).get("display_name", "")
                            for a in (w.get("authorships") or [])])
        ov = len(names & known) if known else 0
        if s < MIN_SIM or ov < 1:
            continue
        key = (s, ov)
        if best_key is None or key > best_key:
            best, best_key = dict(w, _sim=s, _ov=ov), key
    return best, len(results)


def main(eval_table, out_csv, limit=None):
    os.makedirs("outputs", exist_ok=True)
    ev, _, _, known = load_inputs(eval_table=eval_table)

    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv)["paper_id"])
    todo = ev[~ev["paper_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(done):,} resolved, {len(todo):,} to go (~{len(todo)*SLEEP/60:.0f} min)")

    new = not os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    hits = 0
    for i, r in enumerate(todo.itertuples(), 1):
        cand, n = best_candidate(r.title, known.get(r.paper_id, set()))
        row = dict(zip(COLS, [r.paper_id] + [""] * 7 + [0]))
        row["n_candidates"] = n
        if cand:
            row.update({
                "oa_id": cand["id"], "oa_title": (cand.get("display_name") or "")[:200],
                "doi": (cand.get("doi") or "").replace("https://doi.org/", ""),
                "title_sim": round(cand["_sim"], 4), "author_overlap": cand["_ov"],
                "oa_year": cand.get("publication_year") or "", "accepted": 1,
            })
            hits += row["doi"] != ""
        w.writerow(row)
        fh.flush()
        time.sleep(SLEEP)
        if i % 100 == 0:
            print(f"  {i:,}/{len(todo):,}  verified DOIs: {hits:,}", flush=True)
    fh.close()
    print(f"Done. {hits:,} verified DOIs -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-table", default="outputs/eval_table.csv")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    main(a.eval_table, a.out, a.limit)
