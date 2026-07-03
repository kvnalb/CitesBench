from . import Regime
import pandas as pd


class LLMDeepSeek(Regime):
    """Decision Head: top-N by DeepSeek V3.1 / GPT-oss-20b accept probability."""
    name = "LLM Decision Head"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        return (
            pool_df.dropna(subset=["deepseek_p_accept"])
            .sort_values("deepseek_p_accept", ascending=False)["paper_id"]
            .head(n).tolist()
        )
