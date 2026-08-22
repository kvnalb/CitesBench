"""
Outlier analysis: rejected papers that became highly cited.
Defines outliers per year as rejected papers exceeding the 75th pct of accepted
citations from the same year. Outputs two CSVs:
  outputs/outlier_quantitative.csv  — 3-group (outlier-reject / reject / accept) stats
  outputs/outlier_reviews.csv       — one row per outlier with raw review text
"""
import os
import sqlite3
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
con = sqlite3.connect("data/gen_review.db")
submissions = pd.read_sql("SELECT id, title, decision, when_submitted AS year FROM SUBMISSION", con)
reviews = pd.read_sql(
    "SELECT paper_id, rating, confidence, soundness, contribution, "
    "technical_novelty_and_significance, empirical_novelty_and_significance, "
    "correctness, presentation, weaknesses, summary_of_the_review, main_review "
    "FROM REVIEW",
    con,
)
con.close()

cites = pd.read_csv("output/citations_2018_2020.csv")  # paper_id, year, s2_citations, status

# ── collapse decisions ────────────────────────────────────────────────────────
def collapse_decision(d):
    d = str(d).lower()
    if "accept" in d:
        return "accept"
    if "reject" in d or "desk" in d or "workshop" in d:
        return "reject"
    return None

submissions["group"] = submissions["decision"].map(collapse_decision)
submissions = submissions[submissions["group"].notna()]

# ── merge citations ───────────────────────────────────────────────────────────
found = cites[cites["status"] == "found"][["paper_id", "s2_citations"]]
df = submissions.merge(found, left_on="id", right_on="paper_id", how="left")

# report missingness by group before dropping
missing = df.groupby("group")["s2_citations"].apply(lambda s: s.isna().mean())
print("Citation missingness rate by group:")
print(missing.round(3).to_string())

df = df[df["s2_citations"].notna()].copy()

# ── define outliers: per year, reject > 75th pct of accepts that year ─────────
thresholds = (
    df[df["group"] == "accept"]
    .groupby("year")["s2_citations"]
    .quantile(0.75)
    .rename("accept_p75")
)
df = df.join(thresholds, on="year")
df["is_outlier"] = (df["group"] == "reject") & (df["s2_citations"] > df["accept_p75"])

print("\nOutliers per year:")
print(df[df["is_outlier"]].groupby("year").size().to_string())

# ── per-paper review aggregates ───────────────────────────────────────────────
# ratings are stored as "6: Marginally above acceptance threshold" — extract leading int
def parse_ordinal(s):
    try:
        return int(str(s).split(":")[0])
    except (ValueError, AttributeError):
        return np.nan

numeric_cols = ["rating", "confidence", "soundness", "contribution",
                "technical_novelty_and_significance", "empirical_novelty_and_significance",
                "correctness", "presentation"]
for c in numeric_cols:
    reviews[c] = reviews[c].map(parse_ordinal)

agg = reviews.groupby("paper_id").agg(
    rating_mean=("rating", "mean"),
    rating_std=("rating", "std"),
    confidence_mean=("confidence", "mean"),
    soundness_mean=("soundness", "mean"),
    contribution_mean=("contribution", "mean"),
    tech_novelty_mean=("technical_novelty_and_significance", "mean"),
    emp_novelty_mean=("empirical_novelty_and_significance", "mean"),
    correctness_mean=("correctness", "mean"),
    presentation_mean=("presentation", "mean"),
    n_reviews=("rating", "count"),
).reset_index()

df = df.merge(agg, left_on="id", right_on="paper_id", how="left")

# ── 3-group quantitative summary ──────────────────────────────────────────────
def cohens_d(a, b):
    """pooled-std Cohen's d"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan

df["analysis_group"] = df["group"].copy()
df.loc[df["is_outlier"], "analysis_group"] = "outlier_reject"

# soundness/contribution/novelty sub-axes are blank for pre-2022 ICLR schema
metric_cols = ["rating_mean", "rating_std", "confidence_mean", "s2_citations"]

quant_rows = []
for col in metric_cols:
    row = {"metric": col}
    for grp in ["outlier_reject", "reject", "accept"]:
        vals = df.loc[df["analysis_group"] == grp, col].dropna()
        row[f"{grp}_median"] = vals.median()
        row[f"{grp}_n"] = len(vals)
    outlier_vals = df.loc[df["analysis_group"] == "outlier_reject", col].dropna()
    reject_vals  = df.loc[df["analysis_group"] == "reject",         col].dropna()
    row["cohens_d_vs_reject"] = cohens_d(outlier_vals, reject_vals)
    quant_rows.append(row)

quant = pd.DataFrame(quant_rows)
quant.to_csv("outputs/outlier_quantitative.csv", index=False)
print("\nQuantitative summary (medians):")
print(quant[["metric", "outlier_reject_median", "reject_median", "accept_median",
             "cohens_d_vs_reject"]].to_string(index=False))

# ── free-text for outliers ─────────────────────────────────────────────────────
text_cols = ["weaknesses", "summary_of_the_review", "main_review"]
text = reviews.groupby("paper_id")[text_cols].apply(
    lambda g: pd.Series({c: "\n---\n".join(g[c].dropna().astype(str)) for c in text_cols})
).reset_index()

outliers = df[df["is_outlier"]].sort_values("s2_citations", ascending=False)
outliers = outliers.merge(text, left_on="id", right_on="paper_id", how="left")

out_cols = (
    ["title", "year", "s2_citations", "accept_p75",
     "rating_mean", "rating_std", "confidence_mean", "n_reviews"]
    + text_cols
)
outliers[out_cols].to_csv("outputs/outlier_reviews.csv", index=False)
print(f"\nWrote {len(outliers)} outlier rows → outputs/outlier_reviews.csv")
print(f"Wrote quantitative summary → outputs/outlier_quantitative.csv")
