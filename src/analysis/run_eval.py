"""
Run all regimes × years, compute metrics + baselines, write eval_results.csv.

Usage: python src/analysis/run_eval.py [--include-normalized]

FIELD NORMALIZATION IS OFF BY DEFAULT. The normalized arm ranks citations within
field × year, and the field labels do not support it:

  - `paper_fields.csv` covers 2,726 of 4,567 papers (59.7%)
  - coverage is not independent of the decision: 63.7% of accepted papers carry a
    label against 57.7% of rejected ones, a 6.0 pp differential — twice the 3.9 pp
    citation-coverage differential this project already reports as a limitation,
    and on the same axis
  - `citation_pct_rank` is therefore defined for 57.7% of the pool, with an 8.1 pp
    accept/reject gap of its own
  - 1,749 of the 2,726 labels are `theory_methods`, so most of the normalization
    happens inside one catch-all bucket
  - the arm rescales n proportionally to the labeled subset, which makes it a
    different selection problem on a different population, not a robustness check
    on the same one

Writing both arms into one `value` column was also the direct cause of a wrong
result: a `pivot_table` that forgot the mode filter averaged a median of 184.0
with a percentile of 0.75 and reported 92.4, which read as a real citation count.
Guarding every reader against that is weaker than not producing it. With the
default, `eval_results.csv` holds one scale in one column.

`--include-normalized` restores the second arm for anyone who wants to look at it.
Anything it produces is not a supported result.

Usage: python src/analysis/run_eval.py
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from metrics import compute_metrics, METRIC_LABELS
from baselines import random_baseline, ideal_baseline
from regimes import ALL_REGIMES

os.makedirs("outputs", exist_ok=True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--include-normalized", action="store_true",
                 help="also run the field×year normalized arm — see the module "
                      "docstring for why it is off, and do not report it")
_args = _ap.parse_args()

eval_table = pd.read_csv("outputs/eval_table.csv")
YEARS = sorted(eval_table["year"].unique())
MODES = ["raw", "normalized"] if _args.include_normalized else ["raw"]
if _args.include_normalized:
    print("WARNING: field-normalized arm enabled. Field labels cover 59.7% of the "
          "pool with a 6.0 pp accept/reject differential; results are not supported.")

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

# Quick summary table. The mode filter stays even though only one mode is written
# by default, because forgetting it is exactly how 92.4 happened.
pivot = df[df["mode"] == "raw"].pivot_table(
    index="regime", columns="metric", values="value", aggfunc="mean"
)
print("\nMean metrics across years (raw mode):")
print(pivot[[c for c in METRIC_LABELS if c in pivot.columns]].round(3).to_string())
