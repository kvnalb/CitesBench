"""
Covariate heterogeneity analysis.

Tests whether LLM committee performance (citation predictive accuracy and
regime recall) varies by paper- and author-level covariates from OpenAlex.

Inputs:
  outputs/eval_table.csv
  outputs/paper_author_covariates.csv

Output:
  outputs/hetero_analysis.md

Run: python src/analysis/hetero_analysis.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import compute_metrics
from regimes.llm_committee import LLMCommittee
from regimes.human_actual import HumanActual
from regimes.human_score import HumanScore

os.makedirs("outputs", exist_ok=True)

OUTCOME = "s2_citations"
SCORE = "committee_rating"


def ols_hc1(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with HC1 robust standard errors. Returns (beta, se)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    meat = (X * e[:, None]).T @ (X * e[:, None])
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def interaction_reg(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    For each group g in group_col, run:
      log1p(citations) ~ committee_rating + year_dummies
    then a pooled interaction:
      log1p(citations) ~ committee_rating * group + year_dummies

    Returns per-group Spearman rho and slope from separate regressions.
    """
    sub = df.dropna(subset=[SCORE, OUTCOME, group_col]).copy()
    sub["log_cites"] = np.log1p(sub[OUTCOME])

    rows = []
    for g, grp in sub.groupby(group_col):
        if len(grp) < 20:
            continue
        rho, pval = spearmanr(grp[SCORE], grp["log_cites"])
        # Simple slope in this subgroup (no FE, just bivariate)
        X = np.column_stack([np.ones(len(grp)), grp[SCORE].values])
        y = grp["log_cites"].values
        beta, se = ols_hc1(X, y)
        rows.append(
            {
                "group": group_col,
                "value": str(g),
                "n": len(grp),
                "spearman_rho": round(rho, 3),
                "spearman_p": round(pval, 4),
                "slope": round(beta[1], 4),
                "slope_se": round(se[1], 4),
            }
        )
    return pd.DataFrame(rows)


def regime_recall_by_group(eval_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """recall@10% for three regimes, separately for each value of group_col."""
    regimes = [LLMCommittee(), HumanActual(), HumanScore()]
    rows = []
    for g, grp in eval_df.groupby(group_col):
        for year, ypool in grp.groupby("year"):
            n_accept = int(ypool["decision"].str.startswith("Accept").sum())
            if n_accept < 5:
                continue
            for reg in regimes:
                try:
                    sel = reg.select(ypool, n_accept)
                    m = compute_metrics(sel, ypool, mode="raw")
                    rows.append(
                        {
                            "group": group_col,
                            "value": str(g),
                            "year": year,
                            "regime": reg.name,
                            "recall_at_10": m["recall_at_10"],
                        }
                    )
                except Exception:
                    pass
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(["group", "value", "regime"])["recall_at_10"]
        .mean()
        .reset_index()
        .rename(columns={"recall_at_10": "mean_recall_at_10"})
    )


def spearman_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ between committee_rating and citation_pct_rank, per field×year."""
    rows = []
    for (field, year), grp in df.groupby(["field", "year"]):
        sub = grp.dropna(subset=[SCORE, "citation_pct_rank"])
        if len(sub) < 10:
            continue
        rho, pval = spearmanr(sub[SCORE], sub["citation_pct_rank"])
        rows.append({"field": field, "year": year, "n": len(sub), "rho": round(rho, 3), "p": round(pval, 4)})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_\n"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return header + "\n" + sep + "\n" + rows + "\n"


def main():
    eval_df = pd.read_csv("outputs/eval_table.csv")
    cov_df = pd.read_csv("outputs/paper_author_covariates.csv")
    df = eval_df.merge(cov_df, on="paper_id", how="left")

    # Quartile bins for continuous covariates
    df["h_index_quartile"] = pd.qcut(
        df["max_h_index"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    df["team_size_group"] = pd.cut(
        df["team_size"], bins=[0, 2, 4, 999], labels=["1-2", "3-4", "5+"]
    )

    lines = ["# Covariate Heterogeneity Analysis\n"]

    # ── Spearman ρ grid: field × year ────────────────────────────────────────
    lines.append("## Spearman ρ: committee_rating vs citation_pct_rank by field × year\n")
    grid = spearman_grid(df)
    lines.append(md_table(grid))

    # ── Interaction regressions by subgroup ──────────────────────────────────
    lines.append("## Subgroup slopes: log(1+citations) ~ committee_rating\n")
    lines.append("*(Within-subgroup bivariate OLS slope + HC1 SE; Spearman ρ shown for reference.)*\n")
    for col in ["field", "year", "top_institution_flag", "industry_flag", "us_team_flag",
                "h_index_quartile", "team_size_group"]:
        sub = df.dropna(subset=[col])
        if col in ("top_institution_flag", "industry_flag", "us_team_flag"):
            sub = sub.copy()
            sub[col] = sub[col].astype(str)
        result = interaction_reg(sub, col)
        if not result.empty:
            lines.append(f"### {col}\n")
            lines.append(md_table(result))

    # ── Regime recall@10% by field ───────────────────────────────────────────
    lines.append("## Regime recall@10% by field\n")
    recall = regime_recall_by_group(df, "field")
    if not recall.empty:
        pivot = recall.pivot_table(
            index="value", columns="regime", values="mean_recall_at_10"
        ).reset_index()
        lines.append(md_table(pivot))

    lines.append("## Regime recall@10% by top_institution_flag\n")
    recall_inst = regime_recall_by_group(df, "top_institution_flag")
    if not recall_inst.empty:
        pivot2 = recall_inst.pivot_table(
            index="value", columns="regime", values="mean_recall_at_10"
        ).reset_index()
        lines.append(md_table(pivot2))

    report = "\n".join(lines)
    with open("outputs/hetero_analysis.md", "w") as f:
        f.write(report)
    print("Written: outputs/hetero_analysis.md")
    print(f"Papers in analysis (with committee_rating + citations): "
          f"{df.dropna(subset=[SCORE, OUTCOME]).shape[0]}")


if __name__ == "__main__":
    main()
