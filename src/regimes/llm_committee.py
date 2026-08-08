from . import Regime
import pandas as pd


class LLMCommittee(Regime):
    """LLM Committee: top-N by Gemma-4-31B committee rating."""
    name = "LLM Committee (Gemma)"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        return (
            pool_df.dropna(subset=["committee_rating"])
            .sort_values("committee_rating", ascending=False)["paper_id"]
            .head(n).tolist()
        )
