"""
Build paper-level author covariates.

Team size and h-index come from Semantic Scholar. The institution flags come from
OpenAlex, because S2 resolves affiliations for only 34% of papers.

WHY S2 FOR TEAM SIZE AND H-INDEX. OpenAlex matches 89.0% of accepted papers but only
62.7% of rejected ones. Every column built on that match carries the 26 pp gap, so a
cut on it is partly a cut on the decision. S2 covers 96.7% of the pool with a 2.8 pp
gap, the same gap as the citation match itself. The OpenAlex values stay as oa_*
columns for comparison.

POST-TREATMENT. Both sources report h-index as of the fetch date (2026). For a
2018-2020 submission it is downstream of the decision. The same holds for the
institution flags, which use each author's last known institution, not the one at
submission.

Inputs:
  outputs/paper_author_ids.csv     — OpenAlex: paper_id, author_id, author_position
  outputs/author_stats.csv         — OpenAlex: author_id, h_index, institution, country
  outputs/s2_paper_metadata.csv    — S2 author counts and h-index, joined to paper_id
                                     by build_paper_master.s2_metadata_attrs

Output:
  outputs/paper_author_covariates.csv  — one row per paper_id

Run: python src/build/build_author_covariates.py
"""
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build.build_paper_master import s2_metadata_attrs  # noqa: E402

os.makedirs("outputs", exist_ok=True)

EVAL_TABLE = "outputs/eval_table.csv"
OUT_CSV = "outputs/paper_author_covariates.csv"

# S2 column -> the name downstream scripts already read
S2_COLS = {
    "n_s2_authors": "team_size",
    "s2_first_author_h_index": "first_author_h_index",
    "s2_max_h_index": "max_h_index",
    "s2_mean_h_index": "mean_h_index",
}

TOP_INSTITUTIONS = [
    "Massachusetts Institute of Technology", "MIT",
    "Stanford",
    "Carnegie Mellon",
    "Berkeley",
    "University of Oxford", "Oxford",
    "Cambridge",
    "ETH",
    "Princeton",
    "Cornell",
    "Columbia University",
    "University of Toronto", "Toronto",
    "Tsinghua",
    "Peking University",
    "EPFL",
    "Imperial College",
    "University of Washington",
    "University of Michigan",
    "New York University", "NYU",
    "Max Planck",
    "Google",
    "DeepMind",
    "Facebook AI", "FAIR", "Meta AI",
    "Microsoft Research",
    "OpenAI",
    "Anthropic",
    "NVIDIA",
    "Amazon",
    "IBM Research",
    "Samsung Research",
    "Adobe Research",
    "Baidu",
    "Huawei",
    "Allen Institute",
]

INDUSTRY_LABS = [
    "Google",
    "DeepMind",
    "Facebook AI", "FAIR", "Meta AI",
    "Microsoft Research",
    "OpenAI",
    "Anthropic",
    "NVIDIA",
    "Amazon",
    "Apple",
    "IBM Research",
    "Samsung Research",
    "Adobe Research",
    "Baidu",
    "Huawei",
    "Waymo",
    "Uber AI",
    "Two Sigma",
    "D.E. Shaw",
    "Scale AI",
    "Cohere",
    "Stability AI",
]


def _match_any(name, keywords):
    if not name or pd.isna(name):
        return False
    n = name.lower()
    return any(kw.lower() in n for kw in keywords)


def main():
    authors = pd.read_csv("outputs/author_stats.csv")
    paper_authors = pd.read_csv("outputs/paper_author_ids.csv")

    # Join author stats onto paper-author links
    df = paper_authors.merge(authors, on="author_id", how="left")

    # Per-author flags
    df["is_top_inst"] = df["last_institution_name"].apply(
        lambda x: _match_any(x, TOP_INSTITUTIONS)
    )
    df["is_industry"] = df["last_institution_name"].apply(
        lambda x: _match_any(x, INDUSTRY_LABS)
    )
    df["is_us"] = df["last_institution_country"].eq("US")
    df["h"] = pd.to_numeric(df["h_index"], errors="coerce")

    # Aggregate to paper level
    rows = []
    for paper_id, g in df.groupby("paper_id"):
        first = g[g["author_position"] == "first"]
        rows.append(
            {
                "paper_id": paper_id,
                "oa_team_size": len(g),
                "oa_first_author_h_index": first["h"].iloc[0] if len(first) else np.nan,
                "oa_max_h_index": g["h"].max(),
                "oa_mean_h_index": g["h"].mean(),
                "top_institution_flag": int(g["is_top_inst"].any()),
                "industry_flag": int(g["is_industry"].any()),
                "us_team_flag": int(g["is_us"].any()),
            }
        )
    oa = pd.DataFrame(rows)

    ids = set(pd.read_csv(EVAL_TABLE, usecols=["paper_id"]).paper_id)
    s2 = s2_metadata_attrs(ids)
    if s2 is None:
        sys.exit("outputs/s2_paper_metadata.csv is missing. Run src/fetch/fetch_s2_metadata.py.")
    s2 = s2[["paper_id", *S2_COLS]].rename(columns=S2_COLS)

    # Outer join: a paper only S2 found gets S2 columns and blank flags, and the reverse.
    out = s2.merge(oa, on="paper_id", how="outer", validate="1:1")
    out = out[out.paper_id.isin(ids)]
    out.to_csv(OUT_CSV, index=False)

    print(f"Papers: {len(out)} (S2 {out['team_size'].notna().sum()}, "
          f"OpenAlex {out['oa_team_size'].notna().sum()})")
    print(f"  top_institution_flag: {out['top_institution_flag'].mean():.1%}")
    print(f"  industry_flag:        {out['industry_flag'].mean():.1%}")
    print(f"  us_team_flag:         {out['us_team_flag'].mean():.1%}")
    print(f"  median max_h_index:   {out['max_h_index'].median():.0f} (S2), "
          f"{out['oa_max_h_index'].median():.0f} (OpenAlex)")
    print(f"  median team_size:     {out['team_size'].median():.0f}")
    print(f"Written: {OUT_CSV}")


if __name__ == "__main__":
    main()
