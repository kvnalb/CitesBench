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
from .llm_committee import LLMCommittee
from .llm_deepseek import LLMDeepSeek

# The GENAI_REVIEW persona regimes (LLMNeutral, LLMEnsemble, LLMPositive) were
# removed: their scores came with data/gen_review.db, not from this project's
# pipeline, so nothing here can state what produced them. Classes are in
# Archive/regimes_gen_review/, which is append-only and never imported.
ALL_REGIMES = [
    HumanActual(),
    HumanScore(),
    HumanDisagree(),
    LLMCommittee(),
    LLMDeepSeek(),
]
