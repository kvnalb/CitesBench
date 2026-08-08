"""Random and ideal baselines."""
import numpy as np
import pandas as pd
from metrics import compute_metrics

N_RUNS = 1000
SEED = 42


def random_baseline(pool_df: pd.DataFrame, n: int, mode: str = "raw") -> dict:
    rng = np.random.default_rng(SEED)
    paper_ids = pool_df["paper_id"].tolist()
    runs = []
    for _ in range(N_RUNS):
        selected = rng.choice(paper_ids, size=n, replace=False).tolist()
        runs.append(compute_metrics(selected, pool_df, mode))
    keys = runs[0].keys()
    return {k: float(np.mean([r[k] for r in runs if not np.isnan(r[k])])) for k in keys}


def ideal_baseline(pool_df: pd.DataFrame, n: int, mode: str = "raw") -> dict:
    col = "openalex_citations" if mode == "raw" else "citation_pct_rank"
    top_n = pool_df.dropna(subset=[col]).nlargest(n, col)["paper_id"].tolist()
    return compute_metrics(top_n, pool_df, mode)
