"""
Quantify OpenAlex vs Semantic Scholar citation-count disagreement.

Motivation: DDSP (B1x1ma4tDr) has 78 citations in OpenAlex but 485 in S2 —
OpenAlex indexes 98.6% of our corpus as arXiv-preprint records (DOI
10.48550/*), and citation matching to arXiv-only records undercounts.
This script measures how big and how systematic that is.

Method: for every paper with a matched arXiv ID (n≈1,383 with OA citations),
fetch S2 citationCount + venue + DOI via the batch endpoint (500/request).
Compare, correlate, and check how many within-year top-decile labels flip.

Outputs:
  outputs/citation_source_comparison.csv    — per-paper OA vs S2 counts
  outputs/citation_source_comparison.md     — summary report

Run: python src/analysis/compare_citation_sources.py [--report-only]
"""
import os
import sys
import json
import time
import argparse

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OUT_CSV = "outputs/citation_source_comparison.csv"
OUT_MD = "outputs/citation_source_comparison.md"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "citationCount,title,year,venue,externalIds"
S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
HEADERS = {"x-api-key": S2_KEY} if S2_KEY else {}


def load_corpus():
    ev = pd.read_csv("outputs/eval_table.csv")
    ax = pd.read_csv("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv", low_memory=False)
    m = ev.merge(ax[["paper_id", "arxiv_id_canonical", "openalex_id", "openalex_doi",
                     "title_similarity"]], on="paper_id", how="left")
    m = m[m["arxiv_id_canonical"].notna() & m["openalex_citations"].notna()]
    return m.reset_index(drop=True)


def fetch(df):
    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV)["paper_id"])
    todo = df[~df["paper_id"].isin(done)].reset_index(drop=True)
    print(f"Papers: {len(df)}, done: {len(done)}, to fetch: {len(todo)}")

    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    with open(OUT_CSV, "a") as fout:
        if write_header:
            fout.write("paper_id,arxiv_id,year,oa_citations,s2_citations,"
                       "s2_venue,s2_has_pub_doi,s2_matched\n")
        for start in range(0, len(todo), 500):
            chunk = todo.iloc[start:start + 500]
            ids = ["ARXIV:" + str(a) for a in chunk["arxiv_id_canonical"]]
            for attempt in range(6):
                r = requests.post(BATCH_URL, params={"fields": FIELDS},
                                  json={"ids": ids}, headers=HEADERS, timeout=120)
                if r.status_code == 200:
                    break
                print(f"  batch {start//500}: HTTP {r.status_code}, retry in {20*(attempt+1)}s")
                time.sleep(20 * (attempt + 1))
            else:
                sys.exit(f"ERROR: batch {start//500} failed repeatedly — rerun to resume.")
            results = r.json()
            for row, res in zip(chunk.itertuples(), results):
                if res is None:
                    fout.write(f"{row.paper_id},{row.arxiv_id_canonical},{row.year},"
                               f"{row.openalex_citations:.0f},,,,False\n")
                    continue
                ext = res.get("externalIds") or {}
                # a DOI other than the arXiv 10.48550/* one = S2 linked a published version
                pub_doi = bool(ext.get("DOI") and not str(ext["DOI"]).startswith("10.48550"))
                venue = (res.get("venue") or "").replace(",", ";")
                fout.write(f"{row.paper_id},{row.arxiv_id_canonical},{row.year},"
                           f"{row.openalex_citations:.0f},{res.get('citationCount')},"
                           f"{venue},{pub_doi},True\n")
            fout.flush()
            print(f"  batch {start//500 + 1}/{(len(todo)+499)//500} written")
            time.sleep(3)


def report():
    from scipy import stats
    df = pd.read_csv(OUT_CSV)
    ev = pd.read_csv("outputs/eval_table.csv")[["paper_id", "decision"]]
    df = df.merge(ev, on="paper_id", how="left")
    df["accepted"] = df["decision"].str.startswith("Accept", na=False)
    matched = df[df["s2_matched"] & df["s2_citations"].notna()].copy()
    matched["ratio"] = matched["s2_citations"] / matched["oa_citations"].clip(lower=1)
    matched["undercount"] = matched["s2_citations"] - matched["oa_citations"]

    sp = stats.spearmanr(matched["oa_citations"], matched["s2_citations"])
    # decile flips under each source (within-year, within this matched subset)
    for src in ("oa_citations", "s2_citations"):
        matched[src + "_rank"] = matched.groupby("year")[src].rank(pct=True)
    flip = ((matched["oa_citations_rank"] >= 0.9) != (matched["s2_citations_rank"] >= 0.9)).mean()

    grp = matched.groupby("s2_has_pub_doi")["ratio"].agg(["count", "median", "mean"])
    acc = matched.groupby("accepted")["ratio"].median()

    lines = [
        f"# OpenAlex vs Semantic Scholar citation counts",
        f"",
        f"Corpus: papers with matched arXiv ID and OpenAlex citations "
        f"(n={len(df)}, S2 matched: {df['s2_matched'].mean():.1%}).",
        f"",
        f"| Statistic | Value |",
        f"|---|---|",
        f"| Median S2/OA ratio | {matched['ratio'].median():.2f} |",
        f"| Mean S2/OA ratio | {matched['ratio'].mean():.2f} |",
        f"| Share with S2 > 2x OA | {(matched['ratio'] > 2).mean():.1%} |",
        f"| Share with S2 > 5x OA | {(matched['ratio'] > 5).mean():.1%} |",
        f"| Spearman ρ (OA vs S2) | {sp.statistic:.3f} (p={sp.pvalue:.2g}) |",
        f"| Top-decile label flips (within-year) | {flip:.1%} |",
        f"| Median ratio, accepted / rejected | "
        f"{acc.get(True, float('nan')):.2f} / {acc.get(False, float('nan')):.2f} |",
        f"",
        f"## Ratio by whether S2 linked a published (non-arXiv) DOI",
        f"",
        grp.to_markdown(floatfmt='.2f'),
        f"",
        f"## Worst undercounts",
        f"",
        matched.nlargest(15, "undercount")[
            ["paper_id", "year", "oa_citations", "s2_citations", "ratio", "s2_venue"]
        ].to_markdown(index=False, floatfmt=".0f"),
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nReport: {OUT_MD}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        fetch(load_corpus())
    report()
