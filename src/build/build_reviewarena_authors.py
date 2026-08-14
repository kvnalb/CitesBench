"""
Author names per paper, from the ReviewArena parquet dump.

Why: fetch_citations_s2_v2 verifies a title match by requiring >=1 shared author
surname between our record and S2's. It sources known authors from
outputs/paper_author_ids.csv (OpenAlex, 3,264 papers, all 2018-2020) and from
arxiv_resolution.csv's arxiv_authors column. Neither covers a 2025 paper that has
no arXiv preprint — which is exactly the population that needs title matching.

Without this, every 2025 title match gets author_overlap="" and assign_tiers drops
it to tier C, i.e. the entire title-match pass would be excluded from the outcome
while the 2018-2020 equivalent is retained. That is a coverage asymmetry across the
comparison, the same flaw that disqualified the OpenAlex counts.

ReviewArena is the right source because it is the same dump the 2025 review markdown
came from, so authorship and text agree by construction.

Output schema matches outputs/paper_author_ids.csv so load_inputs treats them alike.

Run: python src/build/build_reviewarena_authors.py
"""
import glob
import os

import pandas as pd

PARQUET_GLOB = "data/ReviewArena/raw/data/*.parquet"
OUT_CSV = "outputs/paper_author_names_reviewarena.csv"


def build():
    os.makedirs("outputs", exist_ok=True)
    files = sorted(glob.glob(PARQUET_GLOB))
    if not files:
        raise SystemExit(f"no parquet under {PARQUET_GLOB}")

    df = pd.concat([pd.read_parquet(f, columns=["forum_id", "year", "authors"])
                    for f in files], ignore_index=True)

    rows = []
    for r in df.itertuples():
        for name in (r.authors if r.authors is not None else []):
            name = str(name).strip()
            if name:
                rows.append((r.forum_id, r.year, name))

    out = pd.DataFrame(rows, columns=["paper_id", "year", "author_name"])
    out = out.drop_duplicates()
    out.to_csv(OUT_CSV, index=False)
    print(f"{len(out):,} author rows for {out.paper_id.nunique():,} papers -> {OUT_CSV}")
    print(out.groupby("year").paper_id.nunique().to_string())
    return out


def demo():
    out = build()
    # the property that matters: a paper with authors yields >=1 surname-bearing row
    assert out.author_name.str.len().gt(1).all()
    assert out.paper_id.nunique() > 1000
    # no list-valued cells leaked through
    assert not out.author_name.str.startswith("[").any()
    print("ok")


if __name__ == "__main__":
    demo()
