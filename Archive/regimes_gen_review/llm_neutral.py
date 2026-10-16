from . import Regime
import pandas as pd


class LLMNeutral(Regime):
    """LLM1: top-N by neutral-persona LLM review score."""
    name = "LLM1 (neutral)"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        return (
            pool_df.dropna(subset=["llm_neutral_rating"])
            .sort_values("llm_neutral_rating", ascending=False)["paper_id"]
            .head(n).tolist()
        )
