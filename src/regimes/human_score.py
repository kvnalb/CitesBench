from . import Regime
import pandas as pd


class HumanScore(Regime):
    name = "Human (score top-N)"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        ranked = pool_df.dropna(subset=["mean_rating"]).sort_values(
            "mean_rating", ascending=False
        )
        return ranked["paper_id"].head(n).tolist()
