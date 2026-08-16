"""
Recompute author_overlap on rows that were fetched before we had good author lists.

The overlap is computed at fetch time from whatever author lists existed then, so
rows written before outputs/paper_author_names_openreview.csv landed carry a stale
0 and get demoted to tier C — for want of evidence, not for being wrong.

Refetching one row at a time would cost hours against a throttled API. It does not
have to: every matched row already stores its s2_paper_id, and the batch endpoint
takes 500 ids per call. So ~2,000 rows cost 4 calls, not 2,000.

Writes a NEW file rather than editing in place — the fetcher may still be appending
to the input, and two writers on one CSV is how you lose a corpus.

Run: python src/build/backfill_author_overlap.py --in outputs/s2_citations_v2.csv
"""
import argparse
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fetch"))
from fetch_citations_s2_v2 import last_names, load_inputs, HEADERS, BATCH_URL

SLEEP = 3.1
CHUNK = 500


def s2_authors(ids):
    """paperId -> {surname}. One call per 500 ids."""
    out = {}
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        for attempt in range(5):
            try:
                r = requests.post(BATCH_URL, params={"fields": "authors.name"},
                                  json={"ids": chunk}, headers=HEADERS, timeout=90)
            except requests.RequestException:
                time.sleep(SLEEP * (attempt + 2) * 2)
                continue
            if r.status_code == 200:
                for pid, item in zip(chunk, r.json()):
                    if item:
                        out[pid] = last_names([a.get("name", "")
                                               for a in (item.get("authors") or [])])
                break
            time.sleep(SLEEP * (attempt + 2) * 2)
        print(f"  batch {i // CHUNK + 1}: {len(out):,} resolved", flush=True)
        time.sleep(SLEEP)
    return out


def main(in_csv, eval_table, out_csv):
    df = pd.read_csv(in_csv)
    _, _, _, known = load_inputs(eval_table=eval_table)

    have = df["s2_paper_id"].notna() & df["s2_paper_id"].astype(str).ne("")
    ids = df.loc[have, "s2_paper_id"].astype(str).unique().tolist()
    print(f"{len(df):,} rows, {len(ids):,} with an S2 id -> "
          f"{(len(ids) + CHUNK - 1) // CHUNK} batch calls")

    auth = s2_authors(ids)

    old = pd.to_numeric(df["author_overlap"], errors="coerce").fillna(0)
    new = []
    for r in df.itertuples():
        ka = known.get(r.paper_id, set())
        sa = auth.get(str(r.s2_paper_id), set())
        new.append(len(ka & sa) if (ka and sa) else "")
    df["author_overlap"] = new
    df["n_known_authors"] = [len(known.get(p, set())) for p in df["paper_id"]]
    df.to_csv(out_csv, index=False)

    nw = pd.to_numeric(df["author_overlap"], errors="coerce").fillna(0)
    print(f"\nauthor_overlap > 0:  before {(old > 0).sum():,}  ->  after {(nw > 0).sum():,}"
          f"   (+{int((nw > 0).sum() - (old > 0).sum()):,})")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="outputs/s2_citations_v2.csv")
    ap.add_argument("--eval-table", default="outputs/eval_table.csv")
    ap.add_argument("--out")
    a = ap.parse_args()
    main(a.in_csv, a.eval_table, a.out or a.in_csv.replace(".csv", "_authored.csv"))
