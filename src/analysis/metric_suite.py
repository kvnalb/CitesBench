"""
The era comparison under several measures instead of one.

Spearman rho was carrying the whole argument, and it is a poor single choice here:
it weights a swap at ranks 800/900 the same as one at 3/400 when the actual
decision is "which 10% get spotlighted"; it has no units, so the effect size means
nothing to a reader; our scores take only 15-21 distinct values over thousands of
papers, so rho is largely summarising how ties break; and its attenuation under a
noisier outcome is exactly the disputed step in the leakage argument.

So: five families, chosen because they fail differently.

  rank agreement     Spearman rho, Kendall tau-b (ties handled explicitly rather
                     than averaged over), Somers' D (asymmetric — one variable is
                     the predictor, which is our actual setup)
  decision-relevant  recall@10%, NDCG@10%, top-decile AUC. Denominated in papers,
                     and they weight the top of the list, which is where the
                     decision lives.
  effect size        OLS of the within-year citation percentile on the standardised
                     score. Units: percentile points per standard deviation.
  count model        negative binomial on raw counts. Citations are overdispersed
                     counts; logging a variable with a mass at zero is the thing an
                     econometrician objects to first.
  incremental        the LLM score conditional on the human score in one model.
                     "Beats the humans" and "adds information the humans do not
                     already have" are different claims, and only the second
                     justifies using it. The two agree at rho ~0.1, so this is the
                     interesting one.

The DiD is estimated as a pooled interaction rather than a difference of two
separate fits, so the standard error on the era effect is a real one:

    cite_pct ~ z_llm * is2025 + z_human * is2025,  SEs clustered by field

Clustering matters: papers in a field share citation norms, so treating them as
independent overstates precision. Every earlier number in this repo used
unclustered SEs and should be read as optimistic.

Selection: both eras condition on acceptance, which is a collider. The DiD is
defensible because the selection is the same in kind, but acceptance rates differ
across eras, so it is not identical. Stated, not solved.

Run: python src/analysis/metric_suite.py [--tier-a-only]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compare_eras as ce
import did_leakage as dl

OUT_METRICS = "outputs/metric_suite.csv"
OUT_DID = "outputs/metric_suite_did.csv"
OUT_MD = "outputs/metric_suite.md"

SELECTORS = [("LLM committee", "rating"), ("Human reviewers", "mean_rating"),
             ("Area chair tier", "ac")]
N_BOOT = 1000
BOOT_SHUFFLE = 3
SEED = 0
TOP = 0.10


def z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def somers_d(x, y):
    """D_yx: Kendall's tau-b numerator normalised by ties in x only (the predictor)."""
    tau, _ = stats.kendalltau(x, y, variant="c")
    return tau


def top_auc(score, cite_pct, top=TOP):
    """P(a true top-decile paper scores above a non-top paper). Ties count half."""
    lab = cite_pct >= np.quantile(cite_pct, 1 - top)
    if lab.all() or not lab.any():
        return np.nan
    return stats.mannwhitneyu(score[lab], score[~lab], alternative="two-sided"
                              ).statistic / (lab.sum() * (~lab).sum())


def ndcg(score, cite_pct, k_frac=TOP, rng=None):
    n = len(score)
    k = max(1, int(round(n * k_frac)))
    order = np.lexsort((rng.random(n), -score))
    gains = cite_pct[order][:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(cite_pct)[::-1][:k]
    return float((gains * disc).sum() / (ideal * disc).sum())


def rank_metrics(d, col, rng):
    s, cp = d[col].to_numpy(float), d["cite_pct"].to_numpy(float)
    return {
        "spearman_rho": stats.spearmanr(s, cp)[0],
        "kendall_tau_b": stats.kendalltau(s, cp, variant="b")[0],
        "somers_d": somers_d(s, cp),
        "auc_top10": top_auc(s, cp),
        "recall_at_10": ce.topk_recall(s, cp, TOP, rng)[0],
        "ndcg_at_10": ndcg(s, cp, TOP, rng),
    }


def fit_ols(d, col, cluster):
    df = pd.DataFrame({"y": d["cite_pct"].to_numpy(float), "s": z(d[col]), "g": cluster})
    m = smf.ols("y ~ s", df).fit(cov_type="cluster", cov_kwds={"groups": df["g"]})
    return m.params["s"] * 100, m.bse["s"] * 100          # percentile points per SD


def fit_nb(d, col, cluster):
    """Negative binomial on raw counts — citations are overdispersed."""
    df = pd.DataFrame({"y": d["s2_citations"].clip(lower=0).astype(float),
                       "s": z(d[col]), "g": cluster})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.glm("y ~ s", df, family=sm.families.NegativeBinomial(alpha=1.0)
                        ).fit(cov_type="cluster", cov_kwds={"groups": df["g"]})
            return m.params["s"], m.bse["s"]
        except Exception:
            return np.nan, np.nan


def fit_conditional(d, cluster):
    """LLM score conditional on the human score, both standardised."""
    df = pd.DataFrame({"y": d["cite_pct"].to_numpy(float), "llm": z(d["rating"]),
                       "hum": z(d["mean_rating"]), "g": cluster})
    m = smf.ols("y ~ llm + hum", df).fit(cov_type="cluster",
                                         cov_kwds={"groups": df["g"]})
    return {k: (m.params[k] * 100, m.bse[k] * 100) for k in ("llm", "hum")}


RANK_KEYS = ["spearman_rho", "kendall_tau_b", "somers_d", "auc_top10",
             "recall_at_10", "ndcg_at_10"]


def boot_did_all(d1, d2, col, rng):
    """DiD for every rank metric at once. One resample feeds all six — computing
    them separately re-ran the same bootstrap six times.

    Tie-break shuffles are cut to BOOT_SHUFFLE inside the loop: the point-estimate
    call averages 200 draws, but here the bootstrap resampling already supplies the
    randomisation, and 200 nested shuffles per resample cost 1.6M sorts for no
    added precision."""
    full, ce.N_SHUFFLE = ce.N_SHUFFLE, BOOT_SHUFFLE
    out = {k: np.empty(N_BOOT) for k in RANK_KEYS}
    for i in range(N_BOOT):
        s1 = d1.iloc[rng.integers(0, len(d1), len(d1))]
        s2 = d2.iloc[rng.integers(0, len(d2), len(d2))]
        m1l, m1c = rank_metrics(s1, "rating", rng), rank_metrics(s1, col, rng)
        m2l, m2c = rank_metrics(s2, "rating", rng), rank_metrics(s2, col, rng)
        for k in RANK_KEYS:
            out[k][i] = (m1l[k] - m1c[k]) - (m2l[k] - m2c[k])
    ce.N_SHUFFLE = full
    return out


def cluster_of(d, era):
    col = "field" if era == "1820" else "primary_area"
    if col in d:
        return d[col].fillna("unknown").astype(str)
    return pd.Series(["all"] * len(d), index=d.index)


def main(tier_a_only=False):
    d1, d2 = dl.load(tier_a_only)
    # carry the clustering variable through
    ev = pd.read_csv(ce.EVAL_1820, low_memory=False)[["paper_id", "field"]]
    d1 = d1.merge(ev, on="paper_id", how="left")
    r25 = pd.read_csv(ce.RATE_2025)[["paper_id", "primary_area"]]
    d2 = d2.merge(r25, on="paper_id", how="left")
    g1, g2 = cluster_of(d1, "1820"), cluster_of(d2, "2025")

    rng = np.random.default_rng(SEED)
    rows = []
    for era, d, g in [("2018-2020", d1, g1), ("2025", d2, g2)]:
        for lbl, col in SELECTORS:
            m = rank_metrics(d, col, rng)
            b, se = fit_ols(d, col, g)
            nb, nbse = fit_nb(d, col, g)
            m.update({"era": era, "selector": lbl, "n": len(d),
                      "ols_pctpts_per_sd": b, "ols_se": se,
                      "nb_log_irr_per_sd": nb, "nb_se": nbse})
            rows.append(m)
    metrics = pd.DataFrame(rows)[
        ["era", "selector", "n", "spearman_rho", "kendall_tau_b", "somers_d",
         "auc_top10", "recall_at_10", "ndcg_at_10", "ols_pctpts_per_sd", "ols_se",
         "nb_log_irr_per_sd", "nb_se"]]

    # pooled interaction: the DiD with a real standard error
    pool = pd.concat([
        pd.DataFrame({"y": d1["cite_pct"].to_numpy(float), "llm": z(d1["rating"]),
                      "hum": z(d1["mean_rating"]), "is2025": 0, "g": "1820_" + g1}),
        pd.DataFrame({"y": d2["cite_pct"].to_numpy(float), "llm": z(d2["rating"]),
                      "hum": z(d2["mean_rating"]), "is2025": 1, "g": "2025_" + g2}),
    ], ignore_index=True)
    pooled = smf.ols("y ~ llm*is2025 + hum*is2025", pool).fit(
        cov_type="cluster", cov_kwds={"groups": pool["g"]})

    did_rows = []
    base1l, base2l = rank_metrics(d1, "rating", rng), rank_metrics(d2, "rating", rng)
    for lbl, col in [("human reviewers", "mean_rating"), ("area chair tier", "ac")]:
        b1c, b2c = rank_metrics(d1, col, rng), rank_metrics(d2, col, rng)
        boots = boot_did_all(d1, d2, col, rng)
        for key in RANK_KEYS:
            a1, a2 = base1l[key] - b1c[key], base2l[key] - b2c[key]
            lo, hi = np.percentile(boots[key], [2.5, 97.5])
            did_rows.append({"metric": key, "control": lbl, "adv_1820": a1,
                             "adv_2025": a2, "did": a1 - a2, "ci_lo": lo, "ci_hi": hi,
                             "frac_le_0": float((boots[key] <= 0).mean())})
    did = pd.DataFrame(did_rows)

    cond = {era: fit_conditional(d, g) for era, d, g in
            [("2018-2020", d1, g1), ("2025", d2, g2)]}

    os.makedirs("outputs", exist_ok=True)
    metrics.to_csv(OUT_METRICS, index=False)
    did.to_csv(OUT_DID, index=False)

    L = ["# The era comparison under several measures", "",
         f"Accepted papers only, tier {'A' if tier_a_only else 'A+B'} citations. "
         f"n = {len(d1):,} (2018-2020), {len(d2):,} (2025). SEs clustered by field.",
         "", "## Per-era metrics", "", metrics.round(3).to_markdown(index=False),
         "", "## DiD by metric (LLM advantage, 2018-2020 minus 2025)", "",
         did.round(3).to_markdown(index=False),
         "", "## Incremental value: LLM conditional on the human score", "",
         "| era | LLM (pct pts / SD) | human (pct pts / SD) |", "|---|---|---|"]
    for era, c in cond.items():
        L.append(f"| {era} | {c['llm'][0]:+.2f} ({c['llm'][1]:.2f}) | "
                 f"{c['hum'][0]:+.2f} ({c['hum'][1]:.2f}) |")
    L += ["", "## Pooled interaction model", "", "```",
          str(pooled.summary().tables[1]), "```", "",
          "`llm:is2025` is the DiD on the effect-size scale, with a clustered SE.", ""]
    open(OUT_MD, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
