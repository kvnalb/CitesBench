"""
Leakage-robust headline eval: re-run all regimes excluding memorized papers.

Papers with LAP >= threshold (decision recall, outputs/leakage_lap_v1.csv) —
and, if available, FAME >= threshold (outputs/leakage_fame_v1.csv) — are
dropped from the pool; N is rescaled proportionally. If the LLM regimes still
beat humans on papers the model demonstrably cannot recall, the thesis
survives on its own merits.

NOTE: only probed papers can be excluded. With a 300-paper probe sample, this
is directional; run leakage_lap_v1 --full for the defensible version. Probe
coverage is printed with the results.

Output: outputs/leakage_exclusion_eval.csv + printed comparison.

Run: python src/leakage_exclusion_eval.py [--threshold 0.5] [--mode raw]
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from metrics import compute_metrics
from baselines import random_baseline, ideal_baseline
from regimes.human_actual import HumanActual
from regimes.human_score import HumanScore
from regimes.human_disagree import HumanDisagree
from regimes.llm_committee import LLMCommittee
from regimes.llm_deepseek import LLMDeepSeek

# same regime set as the dashboard (not ALL_REGIMES, which has dead LLM1-3)
REGIMES = [HumanActual(), HumanScore(), HumanDisagree(), LLMCommittee(), LLMDeepSeek()]

os.makedirs("outputs", exist_ok=True)
OUT_CSV = "outputs/leakage_exclusion_eval.csv"


def eval_pools(eval_table, excluded_ids, mode):
    """Run all regimes on full vs leakage-excluded pool, return long df."""
    n_accepts = (
        eval_table[eval_table["decision"].str.startswith("Accept", na=False)]
        .groupby("year").size().to_dict()
    )
    rows = []
    for year in sorted(eval_table["year"].unique()):
        full_pool = eval_table[eval_table["year"] == year].copy()
        if mode == "normalized":
            full_pool = full_pool[full_pool["citation_pct_rank"].notna()].copy()
        n_full = n_accepts[year]

        for pool_name, pool in [("full", full_pool),
                                ("leakage_excluded", full_pool[~full_pool["paper_id"].isin(excluded_ids)])]:
            # N = accept count within the pool (dashboard convention), so
            # HumanActual stays well-defined after exclusion
            n = int(pool["decision"].str.startswith("Accept", na=False).sum())
            rand = random_baseline(pool, n, mode)
            ideal = ideal_baseline(pool, n, mode)
            for regime in REGIMES:
                try:
                    selected = regime.select(pool, n)
                    assert len(selected) == n
                except Exception as e:
                    print(f"  {year} {pool_name} {regime.name}: SKIP — {e}")
                    continue
                for metric, value in compute_metrics(selected, pool, mode).items():
                    rand_val = rand.get(metric, np.nan)
                    lift = (value - rand_val) / abs(rand_val) if rand_val else np.nan
                    rows.append({"regime": regime.name, "year": year, "pool": pool_name,
                                 "metric": metric, "value": value,
                                 "random_value": rand_val,
                                 "ideal_value": ideal.get(metric, np.nan),
                                 "lift": lift})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5, help="LAP/FAME exclusion cutoff")
    parser.add_argument("--mode", default="raw", choices=["raw", "normalized"])
    args = parser.parse_args()

    eval_table = pd.read_csv("outputs/eval_table.csv")

    excluded = set()
    probed = set()
    lap_path = "outputs/leakage_lap_v1.csv"
    if not os.path.exists(lap_path):
        sys.exit(f"ERROR: {lap_path} not found — run leakage_lap_v1.py first.")
    lap = pd.read_csv(lap_path)
    lap = lap[lap["lap"].notna()]
    probed |= set(lap["paper_id"])
    excluded |= set(lap[lap["lap"] >= args.threshold]["paper_id"])

    fame_path = "outputs/leakage_fame_v1.csv"
    if os.path.exists(fame_path):
        fame = pd.read_csv(fame_path)
        fame = fame[fame["fame"].notna()]
        probed |= set(fame["paper_id"])
        excluded |= set(fame[fame["fame"] >= args.threshold]["paper_id"])
    else:
        print("(fame probe CSV not found — excluding on decision-LAP only)")

    print(f"Probe coverage: {len(probed)}/{len(eval_table)} papers "
          f"({len(probed)/len(eval_table):.1%}) — excluded as memorized: {len(excluded)}")
    if len(probed) < len(eval_table) * 0.9:
        print("WARNING: coverage < 90% — unprobed papers may also be memorized; "
              "treat results as directional until leakage_lap_v1 --full is run.")

    df = eval_pools(eval_table, excluded, args.mode)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")

    # Headline comparison: mean lift across years per regime, full vs excluded
    pivot = df.pivot_table(index="regime", columns="pool", values="lift", aggfunc="mean")
    if {"full", "leakage_excluded"} <= set(pivot.columns):
        pivot["delta"] = pivot["leakage_excluded"] - pivot["full"]
        print("\nMean lift over random (all metrics × years):")
        print(pivot.round(3).to_string())
        print("\ndelta < 0 → the regime's edge shrinks once memorized papers are removed.")
