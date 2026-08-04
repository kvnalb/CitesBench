"""
Resolve every OpenReview submission to an arXiv ID, using one rule set for all papers.

Why this exists: the existing arXiv match (data/OpenAlex/openalex_rdd_arxiv_paper_level.csv)
was only ever run on the 1,579-paper RDD bandwidth subsample. The other 2,988 papers in
2018-2020 were never queried at all. Because the RDD subsample is selected on proximity to
the score cutoff, 63% of accepted papers have an arXiv ID versus 20% of rejected ones — and
papers without an ID fell through to unverifiable title matching in the S2 fetch. The
accept/reject provenance asymmetry in the citation ground truth starts here, not at S2.

This resolves ALL submissions in the DB (2018-2025, so the out-of-sample years come for
free in the same pass) against a local arXiv metadata dump, with identical rules for every
paper, so match rate becomes a measured quantity rather than an artifact of who got queried.

Matching, in order of precedence:
  exact     normalized title identical
  tokenset  same set of normalized tokens (word order / punctuation / case variants)
Both are then verified: SequenceMatcher ratio, token Jaccard, and author overlap where an
author list is known. Nothing is filtered here — thresholds are applied downstream so they
stay tunable without a re-scan.

Dump: HuggingFace `librarian-bots/arxiv-metadata-snapshot` (parquet, ~2.9 GB, no login).
  python -c "from huggingface_hub import snapshot_download; \
             snapshot_download(repo_id='librarian-bots/arxiv-metadata-snapshot', \
             repo_type='dataset', allow_patterns=['*.parquet'])"

Output: outputs/arxiv_resolution.csv       one row per submission
        outputs/arxiv_resolution_report.md match rate by year and by decision

Run: python src/resolve_arxiv_ids.py [--years 2018 2019 2020] [--report-only]

# ponytail: one scan of the dump serves every year at once; the title index is the only
# thing held in memory, and the dump streams shard by shard.
"""
import os
import re
import sys
import json
import glob
import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher

import pandas as pd
import pyarrow.parquet as pq

os.makedirs("outputs", exist_ok=True)
DB = "data/gen_review.db"
OUT = "outputs/arxiv_resolution.csv"
REPORT = "outputs/arxiv_resolution_report.md"
DUMP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--librarian-bots--arxiv-metadata-snapshot/"
    "snapshots/*/data/*.parquet")

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def normalize_title(t):
    """Lowercase, strip punctuation and collapse whitespace. Matches the rule used by
    Archive/CompletePipeline/design/fetch_arxiv_metadata.py so old and new matches align."""
    return _WS.sub(" ", _PUNCT.sub(" ", str(t).lower())).strip()


def tokenset_key(norm_title):
    return " ".join(sorted(set(norm_title.split())))


def title_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def token_jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def last_names(names):
    out = set()
    for n in names:
        parts = [p for p in re.split(r"[^A-Za-z\-]+", str(n)) if len(p) > 1]
        if parts:
            out.add(parts[-1].lower())
    return out


def load_submissions(years):
    con = sqlite3.connect(DB)
    q = ("SELECT id AS paper_id, title, when_submitted AS year, decision FROM SUBMISSION")
    if years:
        q += f" WHERE when_submitted IN ({','.join(str(int(y)) for y in years)})"
    df = pd.read_sql(q, con)
    con.close()
    df["title_norm"] = df["title"].map(normalize_title)
    df["tokenset"] = df["title_norm"].map(tokenset_key)
    return df


def known_authors():
    """Author last names from sources independent of arXiv, for verifying matches."""
    known = defaultdict(set)
    p = "outputs/paper_author_ids.csv"
    if os.path.exists(p):
        pa = pd.read_csv(p)
        for pid, g in pa.groupby("paper_id")["author_name"]:
            known[pid] |= last_names(g.dropna().tolist())
    return known


def first_version_year(versions, update_date):
    """arXiv `versions` is a list of dicts with RFC-822 `created` strings."""
    try:
        for v in (versions if isinstance(versions, (list, tuple)) else json.loads(versions or "[]")):
            created = (v or {}).get("created") if isinstance(v, dict) else None
            if created:
                m = re.search(r"\b(19|20)\d{2}\b", str(created))
                if m:
                    return int(m.group(0))
    except Exception:
        pass
    m = re.search(r"\b(19|20)\d{2}\b", str(update_date or ""))
    return int(m.group(0)) if m else None


def scan_dump(subs, files, batch_size=50_000):
    """One streaming pass over the dump. Keeps a candidate per submission per rule."""
    exact_idx = defaultdict(list)
    token_idx = defaultdict(list)
    for r in subs.itertuples():
        if r.title_norm:
            exact_idx[r.title_norm].append(r.paper_id)
            token_idx[r.tokenset].append(r.paper_id)

    hits = {}          # paper_id -> best candidate dict
    scanned = 0
    for path in files:
        pf = pq.ParquetFile(path)
        cols = [c for c in ["id", "title", "authors", "categories", "versions",
                            "update_date", "doi", "journal-ref"] if c in pf.schema.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d["id"])):
                scanned += 1
                tn = normalize_title(d["title"][i])
                if not tn:
                    continue
                pids = exact_idx.get(tn)
                rule = "exact"
                if not pids:
                    pids = token_idx.get(tokenset_key(tn))
                    rule = "tokenset"
                if not pids:
                    continue
                cand = {
                    "arxiv_id": d["id"][i], "arxiv_title": d["title"][i],
                    "arxiv_authors": d.get("authors", [None] * len(d["id"]))[i],
                    "arxiv_categories": d.get("categories", [None] * len(d["id"]))[i],
                    "arxiv_doi": d.get("doi", [None] * len(d["id"]))[i],
                    "arxiv_journal_ref": d.get("journal-ref", [None] * len(d["id"]))[i],
                    "arxiv_first_version_year": first_version_year(
                        d.get("versions", [None] * len(d["id"]))[i],
                        d.get("update_date", [None] * len(d["id"]))[i]),
                    "match_rule": rule, "arxiv_title_norm": tn,
                }
                for pid in pids:
                    prev = hits.get(pid)
                    # exact beats tokenset; then earlier first-version year (the submitted
                    # preprint, not a later re-post of the same title)
                    if prev is None or (
                        (rule == "exact") > (prev["match_rule"] == "exact")
                        or ((rule == "exact") == (prev["match_rule"] == "exact")
                            and (cand["arxiv_first_version_year"] or 9999)
                            < (prev["arxiv_first_version_year"] or 9999))
                    ):
                        hits[pid] = dict(cand)
                        hits[pid]["n_candidates"] = (prev or {}).get("n_candidates", 0) + 1
                    else:
                        prev["n_candidates"] = prev.get("n_candidates", 0) + 1
        print(f"  scanned {scanned:,} dump rows, {len(hits):,} submissions matched "
              f"({os.path.basename(path)})", flush=True)
    return hits, scanned


def resolve(years):
    files = sorted(glob.glob(DUMP_GLOB))
    if not files:
        sys.exit(f"No dump found at {DUMP_GLOB}\nDownload it first — see the module docstring.")
    subs = load_submissions(years)
    known = known_authors()
    print(f"{len(subs):,} submissions, {len(files)} dump shards")

    hits, scanned = scan_dump(subs, files)

    rows = []
    fetched_at = datetime.now(timezone.utc).date().isoformat()
    for r in subs.itertuples():
        h = hits.get(r.paper_id)
        base = {"paper_id": r.paper_id, "year": r.year, "decision": r.decision,
                "title": r.title, "resolved_at": fetched_at}
        if not h:
            rows.append({**base, "matched": 0})
            continue
        ka = known.get(r.paper_id, set())
        aa = last_names(re.split(r",| and ", str(h["arxiv_authors"] or "")))
        rows.append({
            **base, "matched": 1, "arxiv_id": h["arxiv_id"],
            "match_rule": h["match_rule"], "n_candidates": h.get("n_candidates", 1),
            "arxiv_title": h["arxiv_title"],
            "title_sim": round(title_similarity(r.title_norm, h["arxiv_title_norm"]), 4),
            "token_jaccard": round(token_jaccard(r.title_norm, h["arxiv_title_norm"]), 4),
            "author_overlap": len(ka & aa) if ka else "",
            "n_known_authors": len(ka),
            "arxiv_authors": h["arxiv_authors"], "arxiv_categories": h["arxiv_categories"],
            "arxiv_doi": h["arxiv_doi"], "arxiv_journal_ref": h["arxiv_journal_ref"],
            "arxiv_first_version_year": h["arxiv_first_version_year"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} — {int(out['matched'].sum()):,}/{len(out):,} matched "
          f"({out['matched'].mean():.1%})")
    write_report(out, scanned, len(files))


def write_report(out, scanned=None, n_shards=None):
    out = out.copy()
    out["accepted"] = out["decision"].fillna("").str.startswith("Accept")
    out["strong"] = (out["matched"].eq(1)
                     & (out.get("title_sim", pd.Series(0, index=out.index)).fillna(0) >= 0.95))

    L = ["# arXiv resolution report", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
         "`python src/resolve_arxiv_ids.py`.",
         "", f"Attempted for **all {len(out):,} submissions** under identical rules"
         + (f" against {n_shards} dump shards ({scanned:,} arXiv records scanned)." if scanned else "."),
         "", "## Match rate by year", "",
         "| year | submissions | matched | strong (sim>=0.95) |", "|---|---|---|---|"]
    for y, g in out.groupby("year"):
        L.append(f"| {int(y)} | {len(g):,} | {g['matched'].mean():.1%} | {g['strong'].mean():.1%} |")

    L += ["", "## Match rate by decision — the symmetry that matters", "",
          "| year | accepted | rejected | gap |", "|---|---|---|---|"]
    for y, g in out.groupby("year"):
        a = g[g["accepted"]]["matched"].mean() if g["accepted"].any() else float("nan")
        r = g[~g["accepted"]]["matched"].mean() if (~g["accepted"]).any() else float("nan")
        L.append(f"| {int(y)} | {a:.1%} | {r:.1%} | {a - r:+.1%} |")
    L += ["", "A gap here is a real property of arXiv posting behaviour (authors post accepted "
          "work more often), not a pipeline artifact — but it must be carried into the citation "
          "analysis as a known selection channel, since ID-matched papers get verifiable citation "
          "attribution and title-matched ones do not.", ""]

    if "match_rule" in out:
        L += ["## Match rules", "",
              "| rule | n |", "|---|---|"]
        for k, v in out["match_rule"].value_counts(dropna=False).items():
            L.append(f"| {k} | {v:,} |")
        amb = int((out.get("n_candidates", pd.Series(dtype=float)).fillna(1) > 1).sum())
        L += ["", f"- submissions with more than one dump candidate: **{amb:,}** "
                  "(kept the exact-rule match with the earliest first-version year)",
              f"- matched but title_sim < 0.95: "
              f"**{int((out['matched'].eq(1) & (out.get('title_sim', 0) < 0.95)).sum()):,}**", ""]

    open(REPORT, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int,
                    help="restrict to these submission years (default: all in the DB)")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        write_report(pd.read_csv(OUT))
    else:
        resolve(a.years)
