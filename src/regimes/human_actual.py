from . import Regime
import pandas as pd


class HumanActual(Regime):
    name = "Human (AC decisions)"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        selected = pool_df[pool_df["decision"].str.startswith("Accept", na=False)]["paper_id"].tolist()
        assert len(selected) == n, (
            f"HumanActual: expected {n} accepts, got {len(selected)}"
        )
        return selected
