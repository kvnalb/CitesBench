from . import Regime
import pandas as pd


class HumanDisagree(Regime):
    """Score = mean_rating + lam * rating_std.
    lam > 0: rewards disagreement (champion a contested paper)
    lam < 0: penalizes disagreement (prefer consensus)
    lam = 0: equivalent to score top-N
    """

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    @property
    def name(self):
        return f"Human (disagreement-adjusted, λ={self.lam:+.2g})"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        df = pool_df.dropna(subset=["mean_rating", "rating_std"]).copy()
        df["score"] = df["mean_rating"] + self.lam * df["rating_std"]
        return df.sort_values("score", ascending=False)["paper_id"].head(n).tolist()
