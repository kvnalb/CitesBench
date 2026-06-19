"""
Run all regimes × years, compute metrics + baselines, write eval_results.csv.

Usage: python src/run_eval.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from metrics import compute_metrics, METRIC_LABELS
from baselines import random_baseline, ideal_baseline
from regimes import ALL_REGIMES

os.makedirs("outputs", exist_ok=True)

eval_table = pd.read_csv("outputs/eval_table.csv")
YEARS = sorted(eval_table["year"].unique())
MODES = ["raw", "normalized"]

n_accepts = (
    eval_table[eval_table["decision"].str.startswith("Accept", na=False)]
    .groupby("year").size().to_dict()
)

rows = []

for mode in MODES:
    print(f"\n=== Mode: {mode} ===")
    for year in YEARS:
        pool = eval_table[eval_table["year"] == year].copy()
        n = n_accepts[year]

        if mode == "normalized":
            tagged = pool["citation_pct_rank"].notna().sum()
            if tagged == 0:
                print(f"  {year}: skipping normalized (no field tags)")
                continue
            if tagged < len(pool):
                print(f"  {year}: normalized mode — restricting pool to {tagged}/{len(pool)} tagged papers")
            # only evaluate on papers that have a rank; n scales proportionally
            pool = pool[pool["citation_pct_rank"].notna()].copy()
            n = int(round(n * len(pool) / (eval_table[eval_table["year"] == year].shape[0])))

        rand = random_baseline(pool, n, mode)
        ideal = ideal_baseline(pool, n, mode)

        for regime in ALL_REGIMES:
            try:
                selected = regime.select(pool, n)
                assert len(selected) == n, f"{regime.name}: got {len(selected)}, expected {n}"
            except Exception as e:
                print(f"  {year} {regime.name}: SKIP — {e}")
                continue

            metrics = compute_metrics(selected, pool, mode)

            for metric, value in metrics.items():
                rand_val = rand.get(metric, np.nan)
                ideal_val = ideal.get(metric, np.nan)
                lift = (value - rand_val) / abs(rand_val) if rand_val and rand_val != 0 else np.nan
                drawdown = (ideal_val - value) / abs(ideal_val) if ideal_val and ideal_val != 0 else np.nan
                rows.append({
                    "regime": regime.name,
                    "year": year,
                    "metric": metric,
                    "mode": mode,
                    "value": value,
                    "random_value": rand_val,
                    "ideal_value": ideal_val,
                    "lift": lift,
                    "drawdown": drawdown,
                })

        print(f"  {year}: done (n={n})")

df = pd.DataFrame(rows)
df.to_csv("outputs/eval_results.csv", index=False)
print(f"\nWrote {len(df):,} rows to outputs/eval_results.csv")

# Quick summary table
pivot = df[df["mode"] == "raw"].pivot_table(
    index="regime", columns="metric", values="value", aggfunc="mean"
)
print("\nMean metrics across years (raw mode):")
print(pivot[[c for c in METRIC_LABELS if c in pivot.columns]].round(3).to_string())
