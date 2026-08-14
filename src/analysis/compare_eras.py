"""
Does the LLM committee rank ACCEPTED papers by eventual citation impact, and does
that ability differ between ICLR 2018-2020 and ICLR 2025?

Accepted-only on both sides, deliberately. The 2018-2020 committee reviewed 1,526
accepted and 3,041 rejected papers; the 2025 committee reviewed accepted papers only.
"Separate accepts from rejects" and "rank within accepts" are different tasks, so the
2018-2020 side is restricted to its accepted papers to make one task on both sides.

Citations are within-year percentile ranks, not raw counts. 2018-2020 papers have had
6-8 years to accrue citations and 2025 papers ~18 months, so raw counts are not
comparable across eras; ranks are, under the assumption that 18-month rank proxies
long-run rank. That assumption is the main threat to the comparison and is untested
here — testing it needs citation timestamps from the S2 edge list (issue #11).

Ties matter. Committee ratings take 19-31 distinct values over thousands of papers, so
top-k selection is mostly a question of how ties break. Every top-k metric is averaged
over N_SHUFFLE random tie-breaks rather than relying on input order.

Run: python src/analysis/compare_eras.py [--tier-a-only]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT_MD = "outputs/era_comparison.md"
OUT_CSV = "outputs/era_comparison.csv"

EVAL_1820 = "outputs/eval_table.csv"
TIER_1820 = "outputs/s2_citations_v2_tiered.csv"
RATE_2025 = "outputs/committee_ratings_2025.csv"
TIER_2025 = "outputs/s2_citations_2025_tiered.csv"

N_SHUFFLE = 200
SEED = 0
KS = (0.10, 0.20)


def load_era(ratings, tiers, rating_col, tiers_ok):
    """Join committee ratings to tiered S2 citations. Unmatched papers are DROPPED,
    never imputed as zero — a paper we could not find is not a paper with no cites."""
    t = pd.read_csv(tiers, low_memory=False)
    t = t[t["tier"].isin(tiers_ok)][["paper_id", "s2_citations", "tier"]]
    d = ratings.merge(t, on="paper_id", how="inner")
    d = d[d[rating_col].notna() & d["s2_citations"].notna()].copy()
    d["rating"] = d[rating_col].astype(float)
    # within-year percentile rank of citations: the cross-era comparable outcome
    d["cite_pct"] = d.groupby("year")["s2_citations"].rank(pct=True)
    return d


def topk_recall(rating, cite_pct, frac, rng):
    """Share of the true top-`frac` captured by the committee's top-`frac`.
    Averaged over random tie-breaks in the rating."""
    n = len(rating)
    k = max(1, int(round(n * frac)))
    truth = set(np.argsort(-cite_pct, kind="stable")[:k])
    hits = []
    for _ in range(N_SHUFFLE):
        jitter = rng.random(n)
        order = np.lexsort((jitter, -rating))       # rating desc, ties broken at random
        hits.append(len(truth & set(order[:k])) / k)
    return float(np.mean(hits)), k


def era_metrics(d, label):
    rng = np.random.default_rng(SEED)
    rating = d["rating"].to_numpy()
    cite_pct = d["cite_pct"].to_numpy()
    cites = d["s2_citations"].to_numpy()
    n = len(d)

    rho, p = stats.spearmanr(rating, cite_pct)
    row = {"era": label, "n": n, "spearman": rho, "p": p,
           "distinct_ratings": int(d["rating"].nunique()),
           "median_cites": float(np.median(cites))}

    for frac in KS:
        rec, k = topk_recall(rating, cite_pct, frac, rng)
        row[f"recall@{int(frac*100)}%"] = rec
        row[f"lift@{int(frac*100)}%"] = rec / frac      # random baseline recall = frac
        # median citations of the committee's pick vs the corpus
        order = np.lexsort((rng.random(n), -rating))
        row[f"median_cites_top{int(frac*100)}%"] = float(np.median(cites[order[:k]]))
    return row


def main(tier_a_only=False):
    tiers_ok = ["A"] if tier_a_only else ["A", "B"]

    ev = pd.read_csv(EVAL_1820, low_memory=False)
    acc = ev[ev["decision"].str.startswith("Accept", na=False)]
    d1820 = load_era(acc[["paper_id", "year", "committee_rating"]],
                     TIER_1820, "committee_rating", tiers_ok)

    r25 = pd.read_csv(RATE_2025)
    d2025 = load_era(r25[["paper_id", "year", "rating"]],
                     TIER_2025, "rating", tiers_ok)

    res = pd.DataFrame([era_metrics(d1820, "ICLR 2018-2020"),
                        era_metrics(d2025, "ICLR 2025")])
    os.makedirs("outputs", exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    cov = {"ICLR 2018-2020": (len(d1820), len(acc)), "ICLR 2025": (len(d2025), len(r25))}
    L = ["# Committee ranking ability: ICLR 2018-2020 vs 2025", "",
         f"Tiers used: {'A only (ID-matched)' if tier_a_only else 'A+B'}. "
         "Accepted papers only on both sides. Outcome is the within-year citation "
         "percentile rank. Top-k metrics averaged over "
         f"{N_SHUFFLE} random tie-breaks.", "",
         "## Coverage", "", "| era | analysed | accepted | coverage |", "|---|---|---|---|"]
    for k, (a, b) in cov.items():
        L.append(f"| {k} | {a:,} | {b:,} | {a/b:.1%} |")
    L += ["", "## Ranking ability", "",
          "| era | n | Spearman rho | p | recall@10% | lift@10% | recall@20% | lift@20% |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res.to_dict("records"):
        L.append(f"| {r['era']} | {r['n']:,} | {r['spearman']:.3f} | {r['p']:.2e} | "
                 f"{r['recall@10%']:.3f} | {r['lift@10%']:.2f}x | "
                 f"{r['recall@20%']:.3f} | {r['lift@20%']:.2f}x |")
    L += ["", "Random selection scores recall = the fraction itself (lift 1.00x); a "
          "perfect ranker scores 1.000 (lift 10x at 10%).", ""]
    open(OUT_MD, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"Wrote {OUT_MD} and {OUT_CSV}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
