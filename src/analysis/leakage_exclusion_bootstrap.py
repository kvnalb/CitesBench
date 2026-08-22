"""
Bootstrap CIs for the leakage-exclusion headline comparison (P3).

Question: is the LLM Committee's lift over random — and its gap vs Human AC —
distinguishable from zero on the leakage-excluded pool, under each citation
ground truth?

Method: paired percentile bootstrap over papers, conditional on realized
selections. Each regime's selected set is computed once per (year, pool) on
the original data (deterministic, same as leakage_exclusion_eval). Replicates
resample pool rows with replacement (same draw for every regime -> paired,
so regime-gap CIs benefit from correlated noise cancelling). Random baselines
are computed analytically per replicate (pool median / mean log / n_sel over
pool size), so point estimates and replicates use one consistent convention —
they differ slightly (<1%) from the simulated baseline in leakage_exclusion_eval.

Headline statistic matches dashboard 5d: mean lift over random across all
5 metrics x 3 years, per regime x pool. Gap = LLM regime minus human regime,
same replicate.

Output: outputs/leakage_exclusion_bootstrap_{openalex|s2}.csv + printed summary.

Run: python src/analysis/leakage_exclusion_bootstrap.py [--citation-source s2] [--B 2000]
     (pure recompute — no API calls)
"""
import os
import sys
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from regimes.human_actual import HumanActual
from regimes.human_score import HumanScore
from regimes.human_disagree import HumanDisagree
from regimes.llm_committee import LLMCommittee
from regimes.llm_deepseek import LLMDeepSeek

REGIMES = [HumanActual(), HumanScore(), HumanDisagree(), LLMCommittee(), LLMDeepSeek()]
METRICS = ["median_citations", "mean_log_citations", "recall_at_1", "recall_at_5", "recall_at_10"]
TOP_K = [1, 5, 10]

os.makedirs("outputs", exist_ok=True)


def load_eval_table(source, venue_premium=0.0):
    et = pd.read_csv("outputs/eval_table.csv")
    if source == "s2":
        s2 = pd.read_csv("outputs/s2_citations_full.csv")
        ok = s2[s2["s2_citations"].notna() &
                ((s2["method"] == "arxiv_batch") | (s2["title_sim"].fillna(0) >= 0.9))]
        et = et.merge(ok[["paper_id", "s2_citations"]].drop_duplicates("paper_id"),
                      on="paper_id", how="left")
        et["s2_citations"] = et["s2_citations"]
        et = et.drop(columns=["s2_citations"])
    if venue_premium:
        # P1 add-back: log(1+c_cf) = log(1+c) + LATE for rejected papers
        rej = ~et["decision"].str.startswith("Accept", na=False)
        c = et.loc[rej, "s2_citations"]
        et.loc[rej, "s2_citations"] = (1 + c) * np.exp(venue_premium) - 1
    return et


def replicate_lifts(q, flags, n_sel, idx):
    """Metrics + analytic random baseline on one replicate; returns {regime: {metric: lift}}."""
    qb = q[idx]
    known = ~np.isnan(qb)
    qk = qb[known]
    order = np.argsort(-qk)  # descending among known
    med_r = float(np.median(qk))
    mlog_r = float(np.log1p(qk).mean())
    pool_size = len(qb)

    out = {}
    for name, flag in flags.items():
        fb = flag[idx]
        sel_q = qb[fb & known]
        lifts = {}
        med = float(np.median(sel_q)) if len(sel_q) else np.nan
        mlog = float(np.log1p(sel_q).mean()) if len(sel_q) else np.nan
        lifts["median_citations"] = (med - med_r) / abs(med_r) if med_r else np.nan
        lifts["mean_log_citations"] = (mlog - mlog_r) / abs(mlog_r) if mlog_r else np.nan
        fk = fb[known]
        for k in TOP_K:
            cutoff_n = math.ceil(k / 100 * len(qk))
            hits = int(fk[order[:cutoff_n]].sum())
            rec = hits / cutoff_n if cutoff_n else np.nan
            rec_r = n_sel / pool_size  # E[recall] under uniform random selection
            lifts[f"recall_at_{k}"] = (rec - rec_r) / abs(rec_r) if rec_r else np.nan
        out[name] = lifts
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--citation-source", default="openalex", choices=["openalex", "s2"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--B", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--venue-premium", type=float, default=0.0,
                        help="RDD LATE (log 1+cites units) added back to rejected papers")
    args = parser.parse_args()

    et = load_eval_table(args.citation_source, args.venue_premium)

    lap = pd.read_csv("outputs/leakage_lap_v1.csv")
    fame = pd.read_csv("outputs/leakage_fame_v1.csv")
    excluded_ids = (set(lap.loc[lap["lap"] >= args.threshold, "paper_id"]) |
                    set(fame.loc[fame["fame"] >= args.threshold, "paper_id"]))

    # Fixed selections + arrays per (year, pool)
    cells = []  # (year, pool_name, q, flags, n_sel)
    for year in sorted(et["year"].unique()):
        full_pool = et[et["year"] == year].copy()
        for pool_name, pool in [("full", full_pool),
                                ("leakage_excluded",
                                 full_pool[~full_pool["paper_id"].isin(excluded_ids)])]:
            n = int(pool["decision"].str.startswith("Accept", na=False).sum())
            flags = {}
            for regime in REGIMES:
                try:
                    sel = set(regime.select(pool, n))
                except Exception as e:
                    print(f"  {year} {pool_name} {regime.name}: SKIP — {e}")
                    continue
                flags[regime.name] = pool["paper_id"].isin(sel).values
            cells.append((year, pool_name, pool["s2_citations"].values.astype(float),
                          flags, n))

    regime_names = [r.name for r in REGIMES]
    rng = np.random.default_rng(args.seed)

    # replicate 0 = identity (point estimate), then B bootstrap draws
    agg = {pn: {rn: [] for rn in regime_names} for pn in ("full", "leakage_excluded")}
    for b in range(args.B + 1):
        sums = {pn: {rn: [] for rn in regime_names} for pn in ("full", "leakage_excluded")}
        for year, pool_name, q, flags, n_sel in cells:
            idx = np.arange(len(q)) if b == 0 else rng.integers(0, len(q), len(q))
            res = replicate_lifts(q, flags, n_sel, idx)
            for rn, lifts in res.items():
                sums[pool_name][rn].append(np.nanmean([lifts[m] for m in METRICS]))
        for pn in sums:
            for rn in regime_names:
                agg[pn][rn].append(float(np.mean(sums[pn][rn])) if sums[pn][rn] else np.nan)
        if b % 500 == 0:
            print(f"  replicate {b}/{args.B}")

    rows = []
    for pn in ("full", "leakage_excluded"):
        for rn in regime_names:
            v = np.array(agg[pn][rn])
            point, boot = v[0], v[1:]
            rows.append({"regime": rn, "pool": pn, "stat": "lift",
                         "point": point, "lo": np.nanpercentile(boot, 2.5),
                         "hi": np.nanpercentile(boot, 97.5)})
        # paired gaps vs each human regime
        for llm in ["LLM Committee (Gemma)", "LLM Decision Head"]:
            for hum in ["Human (AC decisions)", "Human (score top-N)"]:
                g = np.array(agg[pn][llm]) - np.array(agg[pn][hum])
                point, boot = g[0], g[1:]
                # two-sided bootstrap p-value for gap = 0
                p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
                rows.append({"regime": f"{llm} − {hum}", "pool": pn, "stat": "gap",
                             "point": point, "lo": np.nanpercentile(boot, 2.5),
                             "hi": np.nanpercentile(boot, 97.5), "p_boot": min(p, 1.0)})

    out = pd.DataFrame(rows)
    _vp = "_vp" if args.venue_premium else ""
    out_path = f"outputs/leakage_exclusion_bootstrap_{args.citation_source}{_vp}.csv"
    out.to_csv(out_path, index=False)

    print(f"\n=== {args.citation_source.upper()} — mean lift over random "
          f"(5 metrics × 3 years), B={args.B} ===")
    for pn in ("full", "leakage_excluded"):
        print(f"\n[{pn}]")
        sub = out[(out["pool"] == pn)]
        for r in sub.itertuples():
            pstr = f"  p={r.p_boot:.3f}" if r.stat == "gap" and not np.isnan(r.p_boot) else ""
            print(f"  {r.regime:<55} {r.point:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]{pstr}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
