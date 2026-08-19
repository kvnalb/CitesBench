from . import Regime
import pandas as pd


class LLMSingleCall(Regime):
    """Single-call baseline: top-N by a one-shot review rating.

    The control for the 9-call council. Same model, same response schema, same
    temperature and token budget as a persona_review stage — one call instead of nine.
    Produced by src/probes/run_single_call_baseline.py; see that docstring for what is
    and is not held identical, and for the Liang et al. prompt provenance.

    NOT in ALL_REGIMES yet: `single_call_rating` is not joined into eval_table until a
    run has produced ratings, and a regime that silently returns fewer than n ids is
    worse than one that is absent.
    """
    name = "LLM Single Call (1 call)"

    def select(self, pool_df: pd.DataFrame, n: int) -> list:
        return (
            pool_df.dropna(subset=["single_call_rating"])
            .sort_values("single_call_rating", ascending=False)["paper_id"]
            .head(n).tolist()
        )
