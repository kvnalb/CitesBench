"""
Pure metric functions. No I/O.

All metrics computed on the selected set vs. the full year pool.
mode='raw'        → quality signal = openalex_citations
mode='normalized' → quality signal = citation_pct_rank
"""
import numpy as np
import pandas as pd

TOP_K = [1, 5, 10]  # percentages


def _quality(pool_df: pd.DataFrame, mode: str) -> pd.Series:
    col = "openalex_citations" if mode == "raw" else "citation_pct_rank"
    return pool_df.set_index("paper_id")[col]


def compute_metrics(selected_ids: list, pool_df: pd.DataFrame, mode: str = "raw") -> dict:
    quality = _quality(pool_df, mode)
    n_pool = len(pool_df)

    sel_quality = quality.reindex(selected_ids).dropna()

    metrics = {
        "median_citations": float(sel_quality.median()) if len(sel_quality) else np.nan,
        "mean_log_citations": float(np.log1p(sel_quality).mean()) if len(sel_quality) else np.nan,
    }

    for k in TOP_K:
        threshold = quality.quantile(1 - k / 100)
        true_top = set(quality[quality >= threshold].index)
        in_selected = set(selected_ids)
        metrics[f"top{k}_count"] = len(true_top & in_selected)
        metrics[f"recall_at_{k}"] = (
            len(true_top & in_selected) / len(true_top) if true_top else np.nan
        )

    return metrics


METRIC_LABELS = {
    "median_citations": "Median citations",
    "mean_log_citations": "Mean log(1+citations)",
    "top1_count": "Count in true top 1%",
    "top5_count": "Count in true top 5%",
    "top10_count": "Count in true top 10%",
    "recall_at_1": "Recall @ top 1%",
    "recall_at_5": "Recall @ top 5%",
    "recall_at_10": "Recall @ top 10%",
}
