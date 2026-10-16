from . import Regime
import pandas as pd


class LLMPositive(Regime):
    """LLM3: top-N by positive-advocate LLM review score."""
    name = "LLM3 (positive advocate)"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        return (
            pool_df.dropna(subset=["llm_positive_rating"])
            .sort_values("llm_positive_rating", ascending=False)["paper_id"]
            .head(n).tolist()
        )
