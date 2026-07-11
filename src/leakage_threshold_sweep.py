"""
Threshold-sensitivity sweep for the leakage exclusion re-run (5d).

leakage_exclusion_eval.py picks one cutoff (LAP/FAME >= 0.5) to define the
"memorized" pool. That cutoff is a necessary discretization (regime.select()
needs a fixed in/out pool to rank over) but 0.5 itself is arbitrary. This
script re-runs the same exclusion eval across a range of cutoffs and reports
whether the leakage-excluded lift is stable — if the LLM regimes' edge over
random shrinks and stays low across a wide threshold range, 0.5 wasn't doing
special work.

No new API calls — regime.select() is pure Python over already-collected
LAP/FAME scores, so this sweep is free.

Output: outputs/leakage_threshold_sweep.csv

Run: python src/leakage_threshold_sweep.py [--mode raw]
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from leakage_exclusion_eval import eval_pools

os.makedirs("outputs", exist_ok=True)
OUT_CSV = "outputs/leakage_threshold_sweep.csv"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="raw", choices=["raw", "normalized"])
    args = parser.parse_args()

    eval_table = pd.read_csv("outputs/eval_table.csv")
    lap = pd.read_csv("outputs/leakage_lap_v1.csv")
    lap = lap[lap["lap"].notna()]
    fame_path = "outputs/leakage_fame_v1.csv"
    fame = pd.read_csv(fame_path) if os.path.exists(fame_path) else None
    if fame is not None:
        fame = fame[fame["fame"].notna()]

    rows = []
    for threshold in THRESHOLDS:
        excluded = set(lap[lap["lap"] >= threshold]["paper_id"])
        if fame is not None:
            excluded |= set(fame[fame["fame"] >= threshold]["paper_id"])
        df = eval_pools(eval_table, excluded, args.mode)
        piv = df.pivot_table(index="regime", columns="pool", values="lift", aggfunc="mean")
        if {"full", "leakage_excluded"} <= set(piv.columns):
            for regime, r in piv.iterrows():
                rows.append({
                    "threshold": threshold, "regime": regime,
                    "full": r["full"], "leakage_excluded": r["leakage_excluded"],
                    "delta": r["leakage_excluded"] - r["full"],
                    "n_excluded": len(excluded),
                })
        print(f"threshold={threshold}: n_excluded={len(excluded)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
