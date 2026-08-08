from . import Regime
import pandas as pd


class LLMEnsemble(Regime):
    """LLM2: top-N by mean score across all three LLM personas."""
    name = "LLM2 (ensemble)"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        return (
            pool_df.dropna(subset=["llm_mean_rating"])
            .sort_values("llm_mean_rating", ascending=False)["paper_id"]
            .head(n).tolist()
        )
