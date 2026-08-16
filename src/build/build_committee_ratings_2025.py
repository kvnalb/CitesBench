"""
Roll the 2025 per-paper run directories into one ratings table.

The run writes outputs/runs/<slug>/papers/<paper_id>/paper_result.json, one dir per
paper. Analysis wants a table. This is the only step between the run output and the
era comparison — nothing is recomputed here, every field is copied.

Run: python src/build/build_committee_ratings_2025.py [--run-dir ...]
"""
import argparse
import json
import os

import pandas as pd

RUN_DIR = "outputs/runs/iclr2025_gemma_full"
OUT_CSV = "outputs/committee_ratings_2025.csv"

FIELDS = ["paper_id", "decision", "primary_area", "markdown_chars", "model",
          "rating", "confidence", "soundness", "presentation", "contribution",
          "recommendation", "n_calls", "text_synthesis"]


def build(run_dir=RUN_DIR, out_csv=OUT_CSV):
    papers = os.path.join(run_dir, "papers")
    rows, bad = [], 0
    for pid in sorted(os.listdir(papers)):
        f = os.path.join(papers, pid, "paper_result.json")
        if not os.path.exists(f):
            bad += 1
            continue
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            bad += 1
            continue
        rows.append({k: r.get(k) for k in FIELDS})

    df = pd.DataFrame(rows)
    df["year"] = 2025
    df.to_csv(out_csv, index=False)
    print(f"{len(df):,} papers -> {out_csv}  ({bad} unreadable)")
    print(f"  rating: n={df.rating.notna().sum():,} distinct={df.rating.nunique()} "
          f"range {df.rating.min()}-{df.rating.max()} median {df.rating.median()}")
    print(f"  calls: {df.n_calls.value_counts().to_dict()}")
    return df


def demo():
    df = build()
    assert df.paper_id.is_unique
    assert df.rating.notna().mean() > 0.95, "ratings should be near-complete"
    assert df.year.eq(2025).all()
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()
    build(a.run_dir, a.out) if a.run_dir != RUN_DIR else demo()
