from . import Regime
import pandas as pd


class HumanDisagreeReward(Regime):
    """Score = mean_rating + lam * rating_std. Rewards contested papers."""

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    @property
    def name(self):
        return f"Human (reward disagreement, λ={self.lam})"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        df = pool_df.dropna(subset=["mean_rating", "rating_std"]).copy()
        df["score"] = df["mean_rating"] + self.lam * df["rating_std"]
        return df.sort_values("score", ascending=False)["paper_id"].head(n).tolist()
