"""
Pure metric functions. No I/O.

All metrics computed on the selected set vs. the full year pool.
mode='raw'        → quality signal = s2_citations
mode='normalized' → quality signal = citation_pct_rank
"""
import math
import numpy as np
import pandas as pd

TOP_K = [1, 5, 10]  # percentages


def _quality(pool_df: pd.DataFrame, mode: str) -> pd.Series:
    col = "s2_citations" if mode == "raw" else "citation_pct_rank"
    return pool_df.set_index("paper_id")[col]


def compute_metrics(selected_ids: list, pool_df: pd.DataFrame, mode: str = "raw") -> dict:
    quality = _quality(pool_df, mode)

    sel_quality = quality.reindex(selected_ids).dropna()

    metrics = {
        "median_citations":   float(sel_quality.median()) if len(sel_quality) else np.nan,
        "mean_log_citations": float(np.log1p(sel_quality).mean()) if len(sel_quality) else np.nan,
    }

    quality_known = quality.dropna()
    in_selected = set(selected_ids)

    for k in TOP_K:
        # fixed-count cutoff avoids inflating denominator on citation ties
        cutoff_n = math.ceil(k / 100 * len(quality_known))
        true_top = set(quality_known.nlargest(cutoff_n).index)
        hits = len(true_top & in_selected)
        metrics[f"recall_at_{k}"] = hits / len(true_top) if true_top else np.nan

    return metrics


METRIC_LABELS = {
    "median_citations":   "Median citations",
    "mean_log_citations": "Mean log(1+citations)",
    "recall_at_1":        "Recall @ top 1%",
    "recall_at_5":        "Recall @ top 5%",
    "recall_at_10":       "Recall @ top 10%",
}
