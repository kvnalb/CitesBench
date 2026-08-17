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
from .llm_committee import LLMCommittee
from .llm_deepseek import LLMDeepSeek

# The council and the decision head were written but never listed here, so every
# committed eval compared humans against the naive single-call prompts only — the
# 8-call committee pipeline this project is built around had never been scored on
# the select-n-from-the-pool task, despite committee_rating sitting in eval_table
# for 4,497 papers since the archive run.
ALL_REGIMES = [
    HumanActual(),
    HumanScore(),
    HumanDisagree(),
    LLMCommittee(),
    LLMDeepSeek(),
    LLMNeutral(),
    LLMEnsemble(),
    LLMPositive(),
]
