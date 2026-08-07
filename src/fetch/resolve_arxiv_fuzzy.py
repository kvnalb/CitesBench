"""
Second-pass arXiv resolution for papers exact-title matching missed, verified by abstract.

Why: resolve_arxiv_ids.py matches on exact normalized title and leaves 2,078 of the 4,567
2018-2020 submissions unmatched, with the match rate 40 points higher for accepted papers
than rejected ones. Two explanations are consistent with that gap:

  (a) genuine  — authors post accepted work to arXiv and abandon rejected work;
  (b) retitling — rejected papers get revised and reposted under a different title, so
      exact-title matching misses a preprint that does exist.

Only (b) is a measurement problem, and it is the one that would keep biasing citation
attribution. Abstracts survive retitling far better than titles do, so this pass searches
the dump by abstract similarity (TF-IDF cosine) and reports the top candidates with their
scores. Nothing is hard-filtered — the accept threshold is applied downstream so it stays
tunable.

Run on ALL unmatched papers, accepted and rejected alike. Recovering only the rejects would
replace one decision-correlated matching rule with its mirror image.

Output: outputs/arxiv_fuzzy_candidates.csv   top-3 candidates per unmatched paper
        outputs/arxiv_fuzzy_report.md        recovery by decision, and the revised gap

Run: python src/fetch/resolve_arxiv_fuzzy.py [--years 2018 2019 2020] [--min-cos 0.70]

# ponytail: TF-IDF + chunked sparse matmul, no embedding model. If the recovered set turns
# out to matter, upgrade to sentence embeddings then — not before.
"""
import os
import re
import glob
import argparse
import sqlite3
from datetime import datetime, timezone
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs("outputs", exist_ok=True)
DB = "data/gen_review.db"
RESOLUTION = "outputs/arxiv_resolution.csv"
OUT = "outputs/arxiv_fuzzy_candidates.csv"
REPORT = "outputs/arxiv_fuzzy_report.md"
DUMP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--librarian-bots--arxiv-metadata-snapshot/"
    "snapshots/*/data/*.parquet")

# Candidate pool: ML-adjacent categories, posted in a window around the submission years.
CATS = ("cs.", "stat.ML", "eess.")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def normalize(t):
    return _WS.sub(" ", _PUNCT.sub(" ", str(t).lower())).strip()


def year_of(created, update_date):
    m = re.search(r"\b(19|20)\d{2}\b", str(created or "")) or \
        re.search(r"\b(19|20)\d{2}\b", str(update_date or ""))
    return int(m.group(0)) if m else None


def load_unmatched(years):
    res = pd.read_csv(RESOLUTION, low_memory=False)
    res = res[res["year"].isin(years) & res["matched"].eq(0)]
    con = sqlite3.connect(DB)
    abs_df = pd.read_sql("SELECT id AS paper_id, abstract FROM SUBMISSION", con)
    con.close()
    df = res.merge(abs_df, on="paper_id", how="left")
    df["abstract"] = df["abstract"].fillna("")
    n_empty = int((df["abstract"].str.len() < 100).sum())
    if n_empty:
        print(f"  note: {n_empty} unmatched papers have <100 chars of abstract; "
              "they can only be scored on title")
    return df


def load_pool(years, pad=(-3, 2)):
    """Stream the dump once, keeping ML-adjacent preprints near the submission window."""
    lo, hi = min(years) + pad[0], max(years) + pad[1]
    ids, titles, abstracts, cats, yrs, auths = [], [], [], [], [], []
    for path in sorted(glob.glob(DUMP_GLOB)):
        pf = pq.ParquetFile(path)
        cols = ["id", "title", "abstract", "categories", "versions", "update_date", "authors"]
        cols = [c for c in cols if c in pf.schema.names]
        for batch in pf.iter_batches(batch_size=50_000, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d["id"])):
                c = str(d["categories"][i] or "")
                if not any(c.startswith(p) or f" {p}" in c for p in CATS):
                    continue
                v = d.get("versions", [None] * len(d["id"]))[i]
                created = (v[0].get("created") if isinstance(v, (list, tuple)) and v
                           and isinstance(v[0], dict) else None)
                y = year_of(created, d.get("update_date", [None] * len(d["id"]))[i])
                if y is None or not (lo <= y <= hi):
                    continue
                ids.append(d["id"][i]); titles.append(d["title"][i] or "")
                abstracts.append(d["abstract"][i] or ""); cats.append(c); yrs.append(y)
                auths.append(d.get("authors", [None] * len(d["id"]))[i] or "")
        print(f"  pool {len(ids):,} after {os.path.basename(path)}", flush=True)
    return pd.DataFrame({"arxiv_id": ids, "title": titles, "abstract": abstracts,
                         "categories": cats, "year": yrs, "authors": auths})


def topk_matches(queries, pool, k=3, chunk=200):
    vec = TfidfVectorizer(min_df=3, max_features=200_000, stop_words="english",
                          dtype=np.float32, sublinear_tf=True)
    D = vec.fit_transform(pool["abstract"].tolist())
    Q = vec.transform(queries["abstract"].tolist())
    print(f"  tfidf: {D.shape[0]:,} pool docs x {D.shape[1]:,} terms")
    out = []
    for s in range(0, Q.shape[0], chunk):
        sims = (Q[s:s + chunk] @ D.T).toarray()
        for r in range(sims.shape[0]):
            row = sims[r]
            idx = np.argpartition(-row, min(k, len(row) - 1))[:k]
            idx = idx[np.argsort(-row[idx])]
            q = queries.iloc[s + r]
            for rank, j in enumerate(idx, 1):
                c = pool.iloc[j]
                out.append({
                    "paper_id": q.paper_id, "year": q.year, "decision": q.decision,
                    "rank": rank, "abstract_cos": round(float(row[j]), 4),
                    "arxiv_id": c.arxiv_id, "arxiv_title": c.title,
                    "arxiv_year": c.year, "arxiv_categories": c.categories,
                    "arxiv_authors": c.authors,
                    "title_sim": round(SequenceMatcher(
                        None, normalize(q.title), normalize(c.title)).ratio(), 4),
                    "submission_title": q.title,
                })
        print(f"  scored {min(s + chunk, Q.shape[0]):,}/{Q.shape[0]:,}", flush=True)
    return pd.DataFrame(out)


def write_report(cand, years, min_cos):
    res = pd.read_csv(RESOLUTION, low_memory=False)
    res = res[res["year"].isin(years)].copy()
    res["accepted"] = res["decision"].fillna("").str.startswith("Accept")

    best = (cand[cand["rank"].eq(1)].copy())
    best["accepted"] = best["decision"].fillna("").str.startswith("Accept")
    best["recovered"] = best["abstract_cos"] >= min_cos
    rec_ids = set(best.loc[best["recovered"], "paper_id"])

    res["matched_after"] = res["matched"].eq(1) | res["paper_id"].isin(rec_ids)

    L = ["# arXiv fuzzy second pass (abstract-verified)", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
         f"Threshold: abstract cosine >= {min_cos}.", "",
         f"Run on **all {len(best):,} unmatched {min(years)}-{max(years)} submissions**, "
         "accepted and rejected alike.", "",
         "## Recovery among previously unmatched papers", "",
         "| group | unmatched before | recovered | rate |", "|---|---|---|---|"]
    for lab, g in [("accepted", best[best["accepted"]]), ("rejected", best[~best["accepted"]])]:
        L.append(f"| {lab} | {len(g):,} | {int(g['recovered'].sum()):,} | "
                 f"{g['recovered'].mean():.1%} |")

    L += ["", "## The decision gap, before and after", "",
          "| year | accepted before | rejected before | gap before | accepted after | "
          "rejected after | gap after |", "|---|---|---|---|---|---|---|"]
    for y, g in res.groupby("year"):
        a0, r0 = g[g["accepted"]]["matched"].mean(), g[~g["accepted"]]["matched"].mean()
        a1, r1 = g[g["accepted"]]["matched_after"].mean(), g[~g["accepted"]]["matched_after"].mean()
        L.append(f"| {int(y)} | {a0:.1%} | {r0:.1%} | {100*(a0 - r0):+.1f}pp | "
                 f"{a1:.1%} | {r1:.1%} | {100*(a1 - r1):+.1f}pp |")
    a0, r0 = res[res["accepted"]]["matched"].mean(), res[~res["accepted"]]["matched"].mean()
    a1, r1 = res[res["accepted"]]["matched_after"].mean(), res[~res["accepted"]]["matched_after"].mean()
    L += ["", f"**Gap {100*(a0-r0):.1f}pp → {100*(a1-r1):.1f}pp overall.** "
              "Whatever remains after an abstract-verified sweep is genuine arXiv posting "
              "behaviour, not a matching artifact, and has to be carried as a documented "
              "selection channel rather than fixed.", "",
          "## Retitling evidence", ""]
    rec = best[best["recovered"]]
    if len(rec):
        retitled = rec[rec["title_sim"] < 0.80]
        L += [f"- recovered papers whose title changed substantially (title_sim < 0.80): "
              f"**{len(retitled):,}** of {len(rec):,} ({len(retitled)/max(len(rec),1):.0%})",
              f"- median abstract cosine among recovered: **{rec['abstract_cos'].median():.3f}**",
              f"- retitled share among recovered, accepted: "
              f"{(rec[rec['accepted']]['title_sim'] < 0.80).mean():.0%}; "
              f"rejected: {(rec[~rec['accepted']]['title_sim'] < 0.80).mean():.0%}", "",
              "### Examples of recovered retitles", "",
              "| submission title | arXiv title | cos | title_sim |", "|---|---|---|---|"]
        for r in retitled.nlargest(8, "abstract_cos").itertuples():
            L.append(f"| {str(r.submission_title)[:60]} | {str(r.arxiv_title)[:60]} | "
                     f"{r.abstract_cos:.3f} | {r.title_sim:.2f} |")
    L += ["", "## Threshold sensitivity", "",
          "| min cosine | recovered (accepted) | recovered (rejected) |", "|---|---|---|"]
    for t in [0.60, 0.70, 0.75, 0.80, 0.90]:
        a = (best[best["accepted"]]["abstract_cos"] >= t).sum()
        r = (best[~best["accepted"]]["abstract_cos"] >= t).sum()
        L.append(f"| {t:.2f} | {a:,} | {r:,} |")
    L.append("")
    open(REPORT, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=[2018, 2019, 2020])
    ap.add_argument("--min-cos", type=float, default=0.70)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    if a.report_only:
        write_report(pd.read_csv(OUT), a.years, a.min_cos)
    else:
        q = load_unmatched(a.years)
        print(f"{len(q):,} unmatched submissions to search")
        pool = load_pool(a.years)
        print(f"candidate pool: {len(pool):,} arXiv preprints")
        cand = topk_matches(q, pool)
        cand.to_csv(OUT, index=False)
        print(f"Wrote {OUT}")
        write_report(cand, a.years, a.min_cos)
