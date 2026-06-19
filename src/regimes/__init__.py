from abc import ABC, abstractmethod
import pandas as pd


class Regime(ABC):
    name: str

    @abstractmethod
    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        """Return exactly n paper_ids from pool_df."""
        ...


from .human_actual import HumanActual
from .human_score import HumanScore
from .human_disagree import HumanDisagree

ALL_REGIMES = [
    HumanActual(),
    HumanScore(),
    HumanDisagree(),
]
