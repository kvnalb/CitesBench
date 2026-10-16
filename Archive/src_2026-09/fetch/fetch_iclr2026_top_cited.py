"""
Find the ICLR 2026 papers that blew up — highest-cited, as of the run date.

Why it works this way, since two obvious routes are dead:

  OpenReview   api2.openreview.net is behind a bot challenge (HTTP 403
               ChallengeRequiredError), so the submission list cannot be pulled
               programmatically. The accepted-paper list is scraped from
               papers.cool instead (5,357 titles, matching the reported 5,355
               acceptances).
  OpenAlex     has the records but not the citations. Its counts for 2025 CS
               preprints are 1-2 orders of magnitude low — "Dynamic Chunking"
               (H-Net) reads 0, Qwen3 Technical Report reads 98 against S2's
               6,957. Ranking on OpenAlex would be ranking on noise, the same
               reason it was dropped for the 2024 eval table. So Semantic
               Scholar supplies the ranking signal and OpenAlex is fetched
               alongside it as a recorded second opinion, not as the sort key.

Pipeline:
  1. scrape ICLR 2026 accepted titles (cached in outputs/)
  2. page S2 bulk search over CS papers in the ICLR 2026 submission window,
     sorted by citationCount desc
  3. keep those whose normalized title is in the ICLR 2026 set
  4. look each up in OpenAlex by title for a second count
  5. write the top --n

Outputs:
  data/iclr-2026-highest-cited.csv    the dataset (see CLAUDE.md note below)
  data/iclr-2026-highest-cited.json   same rows plus abstracts and match debris
  outputs/iclr2026_accepted_titles.txt   scraped title list, one per line
  outputs/iclr2026_papers_cool.html      raw scrape cache

CLAUDE.md says data/ is read-only and never written by scripts. Writing there is
an explicit instruction for this dataset — it is a fetched input other scripts
will read, not a generated result. Everything intermediate stays in outputs/.

Run: python src/fetch/fetch_iclr2026_top_cited.py [--n 10] [--pages 8]
"""
import os
import re
import csv
import json
import time
import argparse
import urllib.parse
import urllib.request
from datetime import datetime, timezone

os.makedirs("outputs", exist_ok=True)

VENUE_URL = "https://papers.cool/venue/ICLR.2026?show=6000"
HTML_CACHE = "outputs/iclr2026_papers_cool.html"
TITLES_OUT = "outputs/iclr2026_accepted_titles.txt"
OUT_CSV = "data/iclr-2026-highest-cited.csv"
OUT_JSON = "data/iclr-2026-highest-cited.json"

S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
OA_WORKS = "https://api.openalex.org/works"
EMAIL = open("OpenAlex.txt").read().strip() if os.path.exists("OpenAlex.txt") else ""

# ICLR 2026: abstracts due 2025-09-19, full papers 2025-09-24. A submission's
# preprint is normally posted within a couple of quarters before that.
WINDOW = ("2025-01-01", "2025-09-24")

UA = {"User-Agent": f"research/1.0 ({EMAIL})"}


def norm(t):
    """Match key: lowercase, alphanumerics and single spaces only."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


def get_json(url, tries=5):
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                       timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if a == tries - 1:
                raise
            print(f"    retry {a + 1}/{tries} ({type(e).__name__}) {e}")
            time.sleep(3 * (a + 1))


def accepted_titles():
    """Scrape (once, then cache) the ICLR 2026 accepted-paper titles."""
    if not os.path.exists(HTML_CACHE) or os.path.getsize(HTML_CACHE) < 1_000_000:
        print(f"fetching {VENUE_URL}")
        with urllib.request.urlopen(urllib.request.Request(VENUE_URL, headers=UA),
                                   timeout=300) as r:
            open(HTML_CACHE, "wb").write(r.read())
    html = open(HTML_CACHE, encoding="utf-8", errors="replace").read()
    raw = re.findall(r'<h2 class="title"[^>]*>(.*?)</h2>', html, re.S)
    titles = []
    for x in raw:
        t = re.sub(r"<[^>]+>", " ", x)
        t = re.sub(r"^\s*#\d+\s*", "", t)                     # strip the "#123" index
        # each entry ends in papers.cool's own UI affordances: "[PDF 236 ] [Copy]
        # [Kimi 334 ] [REL]" — drop every trailing bracketed group
        t = re.sub(r"(\s*\[[^\]]*\])+\s*$", "", t)
        t = re.sub(r"\s+", " ", t).replace("&amp;", "&").strip()
        if t:
            titles.append(t)
    with open(TITLES_OUT, "w") as f:
        f.write("\n".join(titles) + "\n")
    print(f"ICLR 2026 accepted titles: {len(titles):,} -> {TITLES_OUT}")
    if len(titles) < 4000:
        raise SystemExit(f"only {len(titles)} titles scraped — expected ~5,357; "
                         f"the page layout changed, inspect {HTML_CACHE}")
    return titles


def s2_top_cited(pages):
    """Page S2 bulk search, most-cited first, over CS papers in the window."""
    params = {
        # bulk search needs a query; this OR-set is broad enough to sweep ML titles
        "query": "learning|model|language|diffusion|reasoning|network|agent|transformer",
        "fields": "title,abstract,citationCount,publicationDate,year,externalIds,venue",
        "publicationDateOrYear": f"{WINDOW[0]}:{WINDOW[1]}",
        "fieldsOfStudy": "Computer Science",
        "sort": "citationCount:desc",
    }
    url = f"{S2_BULK}?{urllib.parse.urlencode(params)}"
    seen, out = set(), []
    for p in range(pages):
        d = get_json(url)
        batch = d.get("data") or []
        out += batch
        print(f"  s2 page {p + 1}: {len(batch)} papers "
              f"(cumulative {len(out):,} of {d.get('total', '?'):,} matching)")
        tok = d.get("token")
        if not tok or not batch or tok in seen:
            break
        seen.add(tok)
        url = f"{S2_BULK}?{urllib.parse.urlencode(params)}&token={tok}"
        time.sleep(1)          # keyless pool is rate-limited; be polite
    return out


def openalex_count(title):
    """Second opinion on citations. Returns (count, openalex_id) or (None, None)."""
    # a comma separates filters in OpenAlex and LaTeX ($\pi^3$) trips the parser,
    # so search on the alphanumeric skeleton only
    q = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]", " ", title)).strip()
    u = (f"{OA_WORKS}?filter=title.search:{urllib.parse.quote(q)}"
         f"&select=id,display_name,cited_by_count,publication_date&per_page=3"
         f"&mailto={EMAIL}")
    try:
        d = get_json(u, tries=3)
    except Exception as e:
        print(f"    openalex failed for {title[:40]!r}: {e}")
        return None, None
    for r in d.get("results", []):
        if norm(r["display_name"]) == norm(title):
            return r["cited_by_count"], r["id"]
    return None, None


def main(n, pages):
    titles = accepted_titles()
    by_norm = {norm(t): t for t in titles}

    print(f"\npaging S2 (window {WINDOW[0]} .. {WINDOW[1]}, sorted by citations)")
    cands = s2_top_cited(pages)

    hits, seen = [], set()
    for r in cands:
        k = norm(r.get("title"))
        if k in by_norm and k not in seen:
            seen.add(k)
            hits.append(r)
    hits.sort(key=lambda r: r.get("citationCount") or 0, reverse=True)
    print(f"\nICLR 2026 papers found among the {len(cands):,} most-cited: {len(hits)}")

    rows, fetched_at = [], datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, r in enumerate(hits[:n], 1):
        title = by_norm[norm(r["title"])]
        oa_cites, oa_id = openalex_count(title)
        ext = r.get("externalIds") or {}
        rows.append({
            "rank": i, "title": title, "venue": "ICLR 2026",
            "arxiv_id": ext.get("ArXiv", ""), "doi": ext.get("DOI", ""),
            "s2_paper_id": r.get("paperId", ""),
            "openalex_id": (oa_id or "").replace("https://openalex.org/", ""),
            "s2_citations": r.get("citationCount"),
            "openalex_citations": "" if oa_cites is None else oa_cites,
            "preprint_date": r.get("publicationDate") or "",
            "citations_fetched_at": fetched_at,
            "abstract": (r.get("abstract") or "").replace("\n", " "),
        })
        print(f"  {i:>2}. s2={r.get('citationCount'):>5}  oa={oa_cites if oa_cites is not None else '-':>5}"
              f"  {title[:62]}")
        time.sleep(0.3)

    if not rows:
        raise SystemExit("no ICLR 2026 papers matched — raise --pages")

    cols = [c for c in rows[0] if c != "abstract"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(OUT_JSON, "w"), indent=1)
    print(f"\nWrote {OUT_CSV} and {OUT_JSON} — {len(rows)} papers")
    print("Ranking is S2. openalex_citations is recorded for comparison only; it "
          "undercounts 2025 CS preprints badly (see this file's docstring).")

    # selfcheck: every stored title must be a real ICLR 2026 accepted title, and
    # ranks must be monotone in the sort key
    bad = [r["title"] for r in rows if norm(r["title"]) not in by_norm]
    cites = [r["s2_citations"] for r in rows]
    print("\n=== selfcheck ===")
    if bad:
        print(f"  FAIL {len(bad)} titles not in the accepted list: {bad[:3]}")
    if cites != sorted(cites, reverse=True):
        print("  FAIL rows are not in descending citation order")
    if not bad and cites == sorted(cites, reverse=True):
        print(f"  all {len(rows)} titles matched the scraped ICLR 2026 list; "
              f"order is descending by s2_citations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="how many papers to keep")
    ap.add_argument("--pages", type=int, default=8,
                    help="S2 bulk pages (1000 papers each) to sweep")
    a = ap.parse_args()
    main(a.n, a.pages)
