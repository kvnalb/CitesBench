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
from .llm_neutral import LLMNeutral
from .llm_ensemble import LLMEnsemble
from .llm_positive import LLMPositive

ALL_REGIMES = [
    HumanActual(),
    HumanScore(),
    HumanDisagree(),
    LLMNeutral(),
    LLMEnsemble(),
    LLMPositive(),
]
