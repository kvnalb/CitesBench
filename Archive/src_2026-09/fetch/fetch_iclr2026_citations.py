"""
Citation count for every ICLR 2026 accepted paper, then the distribution.

Population is the 5,357 accepted papers scraped by src/fetch/fetch_iclr2026_top_cited.py.
It is NOT all ~19.5k submissions: OpenReview's API is behind a bot challenge and
the rejected submissions are not publicly enumerable, so no rejected-side
distribution can be built. Any statement about "ICLR 2026 submissions" from this
file is a statement about the accepted half.

Counts come from Semantic Scholar's title-match endpoint, one call per title,
appended immediately so the run resumes after an interruption. OpenAlex is not
used: its 2025 CS preprint counts are 1-2 orders of magnitude low (see
src/fetch/fetch_iclr2026_top_cited.py).

Match quality is recorded, never assumed. The endpoint returns its best guess for
any query, so each row carries the returned title, a difflib ratio against the
query, and match_ok = ratio >= 0.92. Rows below that are kept but excluded from
the distribution, and the report says how many.

Outputs:
  outputs/iclr2026_citations.csv        one row per title, incremental, resumable
  outputs/iclr2026_citation_dist.md     quantiles, histogram, top 25, zero share
  outputs/iclr2026_citation_dist.png    log1p histogram + Lorenz curve

Run: python src/fetch/fetch_iclr2026_citations.py            # fetch (resumes) + report
     python src/fetch/fetch_iclr2026_citations.py --report-only
"""
import os
import re
import csv
import sys
import json
import time
import difflib
import argparse
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

TITLES = "outputs/iclr2026_accepted_titles.txt"
OUT_CSV = "outputs/iclr2026_citations.csv"
OUT_MD = "outputs/iclr2026_citation_dist.md"
OUT_PNG = "outputs/iclr2026_citation_dist.png"

MATCH = "https://api.semanticscholar.org/graph/v1/paper/search/match"
FIELDS = "title,citationCount,influentialCitationCount,year,publicationDate,externalIds,venue"
SLEEP = 1.1          # keyless pool is ~1 req/s shared; stay under it
MATCH_MIN = 0.92     # difflib ratio below this is not trusted

COLS = ["query_title", "matched_title", "title_ratio", "match_ok", "s2_paper_id",
        "arxiv_id", "doi", "citations", "influential_citations", "s2_year",
        "publication_date", "venue", "status"]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


def match_one(title):
    """Returns a row dict. status is ok / no_match / http_<code> / error."""
    url = f"{MATCH}?{urllib.parse.urlencode({'query': title, 'fields': FIELDS})}"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:                     # endpoint's way of saying no match
                return {"query_title": title, "status": "no_match"}
            wait = 5 * (attempt + 1) if e.code == 429 else 3 * (attempt + 1)
            if attempt == 5:
                return {"query_title": title, "status": f"http_{e.code}"}
            time.sleep(wait)
        except Exception as e:
            if attempt == 5:
                return {"query_title": title, "status": f"error_{type(e).__name__}"}
            time.sleep(3 * (attempt + 1))
    else:
        return {"query_title": title, "status": "exhausted"}

    hits = d.get("data") or []
    if not hits:
        return {"query_title": title, "status": "no_match"}
    h = hits[0]
    ext = h.get("externalIds") or {}
    ratio = difflib.SequenceMatcher(None, norm(title), norm(h.get("title"))).ratio()
    return {"query_title": title, "matched_title": h.get("title"),
            "title_ratio": round(ratio, 4), "match_ok": int(ratio >= MATCH_MIN),
            "s2_paper_id": h.get("paperId"), "arxiv_id": ext.get("ArXiv", ""),
            "doi": ext.get("DOI", ""), "citations": h.get("citationCount"),
            "influential_citations": h.get("influentialCitationCount"),
            "s2_year": h.get("year"), "publication_date": h.get("publicationDate"),
            "venue": h.get("venue", ""), "status": "ok"}


def fetch():
    if not os.path.exists(TITLES):
        raise SystemExit(f"{TITLES} missing — run src/fetch/fetch_iclr2026_top_cited.py first")
    titles = [l.rstrip("\n") for l in open(TITLES) if l.strip()]

    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        # retry transient failures on a later pass; keep real no_match results
        keep = prev[prev["status"].isin(["ok", "no_match"])]
        done = set(keep["query_title"])
        print(f"Resuming — {len(done):,} of {len(titles):,} already resolved")

    todo = [t for t in titles if t not in done]
    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    print(f"To fetch: {len(todo):,} titles at ~{SLEEP}s each "
          f"(~{len(todo) * SLEEP / 60:.0f} min)")

    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for i, t in enumerate(todo, 1):
            row = match_one(t)
            w.writerow(row)
            f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i:,}/{len(todo):,}  last: {row.get('status')} "
                      f"cites={row.get('citations')}  {t[:48]}", flush=True)
            time.sleep(SLEEP)


def report():
    d = pd.read_csv(OUT_CSV).drop_duplicates("query_title", keep="last")
    n_all = len(d)
    ok = d[(d["status"] == "ok") & (d["match_ok"] == 1) & d["citations"].notna()].copy()
    c = ok["citations"].astype(int)

    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quant = {f"p{int(q * 100)}": int(c.quantile(q)) for q in qs}
    top_share = c.nlargest(max(1, len(c) // 100)).sum() / max(c.sum(), 1)
    gini = (2 * np.sum((np.arange(1, len(c) + 1)) * np.sort(c))
            / (len(c) * c.sum()) - (len(c) + 1) / len(c)) if c.sum() else float("nan")

    bins = [0, 1, 5, 10, 25, 50, 100, 250, 500, 10**9]
    labels = ["0", "1-4", "5-9", "10-24", "25-49", "50-99", "100-249", "250-499", "500+"]
    hist = pd.cut(c, bins=bins, right=False, labels=labels).value_counts().reindex(labels)

    L = ["# ICLR 2026 accepted papers — citation distribution", "",
         f"Generated by `python src/fetch/fetch_iclr2026_citations.py --report-only`. "
         f"Citations from Semantic Scholar, fetched per title.", "",
         "## Population and coverage", "",
         f"- accepted papers scraped: **{n_all:,}**",
         f"- resolved with a trusted title match (ratio >= {MATCH_MIN}): "
         f"**{len(ok):,}** ({100 * len(ok) / n_all:.1f}%)",
         f"- weak match, excluded: {int(((d['status'] == 'ok') & (d['match_ok'] == 0)).sum()):,}",
         f"- no match in S2: {int((d['status'] == 'no_match').sum()):,}",
         f"- still failing after retries: "
         f"{int((~d['status'].isin(['ok', 'no_match'])).sum()):,}", "",
         "This is the **accepted** half only. ICLR 2026 rejected submissions are not "
         "publicly enumerable (OpenReview's API returns a bot challenge), so no "
         "accept-vs-reject comparison is possible from this file.", "",
         "## Distribution", "",
         f"| statistic | citations |", "|---|---|",
         f"| n | {len(c):,} |", f"| mean | {c.mean():.1f} |",
         f"| std | {c.std():.1f} |", f"| median | {int(c.median())} |"]
    L += [f"| {k} | {v:,} |" for k, v in quant.items()]
    L += [f"| max | {c.max():,} |",
          f"| zero citations | {int((c == 0).sum()):,} ({100 * (c == 0).mean():.1f}%) |",
          f"| <= 5 citations | {int((c <= 5).sum()):,} ({100 * (c <= 5).mean():.1f}%) |",
          f"| top 1% share of all citations | {100 * top_share:.1f}% |",
          f"| Gini | {gini:.3f} |", "",
          "## Histogram", "", "| citations | papers | share |", "|---|---|---|"]
    L += [f"| {k} | {int(v):,} | {100 * v / len(c):.1f}% |" for k, v in hist.items()]

    L += ["", "## Top 25", "", "| # | citations | paper |", "|---|---|---|"]
    for i, r in enumerate(ok.nlargest(25, "citations").itertuples(), 1):
        L.append(f"| {i} | {int(r.citations):,} | {str(r.query_title)[:88]} |")

    L += ["", "## Caveats", "",
          "- Counts are a snapshot; these papers are months old and still accruing.",
          "- A title-match endpoint returns its best guess for any query. Rows below "
          f"a {MATCH_MIN} difflib ratio are excluded rather than trusted, but a "
          "same-title different-paper collision is still possible.",
          "- Papers with no arXiv preprint are systematically less likely to resolve, "
          "which plausibly skews the resolved set toward more visible work."]
    open(OUT_MD, "w").write("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\nWrote {OUT_MD}")
    plot(c)

    print("\n=== selfcheck ===")
    bad = []
    if len(ok) > n_all:
        bad.append("resolved exceeds population")
    if c.min() < 0:
        bad.append("negative citation count")
    dup = d["query_title"].duplicated().sum()
    if dup:
        bad.append(f"{dup} duplicate query titles after dedup")
    for b in bad:
        print(f"  FAIL {b}")
    if not bad:
        print(f"  {len(ok):,} rows, no duplicates, counts non-negative; "
              f"{n_all - len(ok):,} papers excluded and accounted for above")


def plot(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.log1p(c), bins=40, color="#4C72B0", edgecolor="white")
    ax[0].set_xlabel("log(1 + citations)")
    ax[0].set_ylabel("papers")
    ax[0].set_title(f"ICLR 2026 accepted (n={len(c):,})\n"
                    f"median {int(c.median())}, mean {c.mean():.1f}")

    s = np.sort(c.values)
    cum = np.cumsum(s) / s.sum()
    x = np.arange(1, len(s) + 1) / len(s)
    ax[1].plot(x, cum, color="#C44E52")
    ax[1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[1].set_xlabel("papers, least-cited first")
    ax[1].set_ylabel("share of citations")
    ax[1].set_title("Lorenz curve")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        fetch()
    report()
