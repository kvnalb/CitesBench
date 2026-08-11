"""
Download ICLR 2025 camera-ready PDFs from the official proceedings site.

This replaces the OpenReview route, which is capped at 26 PDFs per rolling hour —
~147 hours for one year. proceedings.iclr.cc serves the same papers with no auth and
no per-hour quota, so the same job takes about an hour.

The proceedings index lists exactly 3,703 papers for 2025, which matches the accepted
population in outputs/samples/slim_2025_papers.csv exactly. Papers are keyed there by
an opaque content hash, not by OpenReview forum id, and the abstract pages carry no
OpenReview link — so the join is on normalised title. That matches 98.6% outright with
zero duplicate titles; the rest fall back to a similarity match, and anything still
unresolved is written out rather than silently dropped.

Two things worth knowing about what this gives you:
  - These are camera-ready PDFs. The archive's 2018-2020 PDFs came from OpenReview,
    which serves the latest revision — usually also camera-ready for accepted papers,
    so the provenance is comparable, but it is not guaranteed identical.
  - The mapping file is the audit trail for the join. Keep it: a title-based join is
    the weakest link in this path, and the report needs to state its match rate.

Deliberately polite: one request at a time with a delay. There is no published rate
limit, which is a reason to be careful rather than a licence to hammer a volunteer-run
conference server.

Outputs:
  data/pdf_2025/{paper_id}.pdf                     named by OpenReview forum id
  outputs/iclr2025_proceedings_map.csv             paper_id, hash, title, match_method
  outputs/iclr2025_proceedings_fetch.log           one line per paper

Run: python src/fetch/fetch_iclr_proceedings_pdfs.py
     python src/fetch/fetch_iclr_proceedings_pdfs.py --map-only   # build the join, no download
"""
import os
import re
import sys
import time
import argparse
from difflib import SequenceMatcher

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INDEX_URL = "https://proceedings.iclr.cc/paper_files/paper/2025"
PDF_URL = "https://proceedings.iclr.cc/paper_files/paper/2025/file/{hash}-Paper-Conference.pdf"
SAMPLE_CSV = "outputs/samples/slim_2025_papers.csv"
PDF_DIR = "data/pdf_2025"
MAP_CSV = "outputs/iclr2025_proceedings_map.csv"
LOG = "outputs/iclr2025_proceedings_fetch.log"

DELAY_SECONDS = 1.0        # polite; there is no published limit
FUZZY_FLOOR = 0.93         # below this a title match is not trustworthy
UA = "CitesBench/1.0 (academic research; kunalb@berkeley.edu)"

LINK_RE = re.compile(
    r'href="/paper_files/paper/2025/hash/([a-f0-9]+)-Abstract-Conference\.html">([^<]+)</a>'
)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_index():
    r = requests.get(INDEX_URL, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    pairs = LINK_RE.findall(r.text)
    if not pairs:
        sys.exit("ERROR: parsed 0 papers from the proceedings index — layout changed?")
    return pd.DataFrame(pairs, columns=["hash", "proc_title"])


def build_map():
    """Join our frozen population to proceedings hashes on title. Exact first, then
    similarity for the remainder — a fuzzy join is the weak link here, so the method
    used for every row is recorded."""
    papers = pd.read_csv(SAMPLE_CSV)
    idx = load_index()
    print(f"proceedings index: {len(idx)} papers; our population: {len(papers)}")

    idx["k"] = idx.proc_title.map(norm)
    papers["k"] = papers.title.map(norm)
    exact = dict(zip(idx.k, idx.hash))

    rows, unresolved = [], []
    remaining = idx[~idx.k.isin(set(papers.k))]
    for r in papers.itertuples(index=False):
        h = exact.get(r.k)
        if h:
            rows.append((r.paper_id, h, r.title, "exact"))
            continue
        best, score = None, 0.0
        for c in remaining.itertuples(index=False):
            s = SequenceMatcher(None, r.k, c.k).ratio()
            if s > score:
                best, score = c.hash, s
        if best and score >= FUZZY_FLOOR:
            rows.append((r.paper_id, best, r.title, f"fuzzy:{score:.3f}"))
        else:
            unresolved.append((r.paper_id, r.title, round(score, 3)))

    m = pd.DataFrame(rows, columns=["paper_id", "hash", "title", "match_method"])
    m.to_csv(MAP_CSV, index=False)
    print(f"mapped {len(m)}/{len(papers)} -> {MAP_CSV}")
    print(m.match_method.str.split(":").str[0].value_counts().to_string())
    if unresolved:
        print(f"UNRESOLVED ({len(unresolved)}) — not downloaded, listed in the log:")
        for pid, t, s in unresolved[:5]:
            print(f"  {pid}  best={s}  {t[:70]}")
    return m, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--delay", type=float, default=DELAY_SECONDS)
    args = ap.parse_args()
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    m, unresolved = build_map()
    with open(LOG, "a") as flog:
        for pid, t, s in unresolved:
            flog.write(f"{pid}\tunresolved\tbest_ratio={s}\n")
    if args.map_only:
        return

    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    ok = skip = fail = 0
    t0 = time.time()
    with open(LOG, "a") as flog:
        for i, r in enumerate(m.itertuples(index=False), 1):
            path = os.path.join(PDF_DIR, f"{r.paper_id}.pdf")
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                skip += 1
                continue
            try:
                resp = sess.get(PDF_URL.format(hash=r.hash), timeout=120)
                if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
                    raise ValueError(f"HTTP {resp.status_code}, {len(resp.content)}B")
                tmp = path + ".part"      # rename-on-complete so a kill can't leave a
                with open(tmp, "wb") as f:  # truncated file a rerun would call done
                    f.write(resp.content)
                os.replace(tmp, path)
                ok += 1
                flog.write(f"{r.paper_id}\tok {len(resp.content)}\n")
            except Exception as e:
                fail += 1
                flog.write(f"{r.paper_id}\tfail {type(e).__name__}: {str(e)[:100]}\n")
            flog.flush()
            if i % 50 == 0 or i == len(m):
                el = time.time() - t0
                gb = sum(os.path.getsize(os.path.join(PDF_DIR, x))
                         for x in os.listdir(PDF_DIR) if x.endswith(".pdf")) / 1e9
                print(f"[{i}/{len(m)}] ok={ok} skip={skip} fail={fail} {gb:.1f}GB "
                      f"{el/60:.0f}min eta {(el/max(ok,1))*(len(m)-i)/60:.0f}min",
                      flush=True)
            time.sleep(args.delay)

    print(f"\ndone: ok={ok} skip={skip} fail={fail} unresolved={len(unresolved)}")
    print(f"log: {LOG}")


if __name__ == "__main__":
    main()
