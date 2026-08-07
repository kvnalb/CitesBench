"""
Build paper-level author covariates from OpenAlex author stats.

Inputs:
  outputs/paper_author_ids.csv   — paper_id, author_id, author_position
  outputs/author_stats.csv       — author_id, h_index, works_count, institution, country

Output:
  outputs/paper_author_covariates.csv  — one row per paper_id

Run: python src/build/build_author_covariates.py
"""
import os
import pandas as pd
import numpy as np

os.makedirs("outputs", exist_ok=True)

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
                "team_size": len(g),
                "first_author_h_index": first["h"].iloc[0] if len(first) else np.nan,
                "max_h_index": g["h"].max(),
                "mean_h_index": g["h"].mean(),
                "top_institution_flag": int(g["is_top_inst"].any()),
                "industry_flag": int(g["is_industry"].any()),
                "us_team_flag": int(g["is_us"].any()),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv("outputs/paper_author_covariates.csv", index=False)

    print(f"Papers: {len(out)}")
    print(f"  top_institution_flag: {out['top_institution_flag'].mean():.1%}")
    print(f"  industry_flag:        {out['industry_flag'].mean():.1%}")
    print(f"  us_team_flag:         {out['us_team_flag'].mean():.1%}")
    print(f"  median max_h_index:   {out['max_h_index'].median():.0f}")
    print(f"  median team_size:     {out['team_size'].median():.0f}")
    print(f"Written: outputs/paper_author_covariates.csv")


if __name__ == "__main__":
    main()
