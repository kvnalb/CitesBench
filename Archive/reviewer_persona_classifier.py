#!/usr/bin/env python3
"""
Classify human reviewers into persona types based on their review text.

This classifier uses the weaknesses section (primary) and questions section
(secondary, half weight) to infer what lens a reviewer is using. It avoids
the full review text to prevent generic empirical language in summaries and
strengths from dominating the classification.

Design choices:
  - Strong patterns (specific concern phrases) get 2x weight on weaknesses, 1x on questions
  - Weak patterns (generic domain words) get 0.5x on weaknesses, 0.25x on questions
  - A margin requirement prevents forcing a label when two personas are close
  - A minimum score prevents labeling reviews with no identifiable concern focus
  - "mixed" is a valid output when concerns are split
  - "generic" is a valid output when no patterns fire

Expected distribution on ICLR 2025 Foundation/LLMs reviews (5204 reviews):
  empiricist           ~30%  (confident assignments)
  generic              ~25%  (no dominant concern detected)
  mixed                ~16%  (two+ concerns close in score)
  novelty_gatekeeper   ~12%
  systems_pragmatist   ~11%
  theorist              ~6%

Usage:
  from reviewer_persona_classifier import classify_reviewer, classify_review_sections

  # From raw text
  persona, scores, confident = classify_reviewer(weakness_text, questions_text)

  # From a review dict (as stored in processed/HumanReview/*.json)
  persona, scores, confident = classify_review_dict(review_dict)
"""

from __future__ import annotations

import re
from typing import Any

PERSONA_PATTERNS: dict[str, dict[str, list[str]]] = {
    "empiricist": {
        "strong": [
            r"ablation",
            r"missing.*(?:baseline|comparison|experiment)",
            r"(?:lack|no|without).*(?:ablation|comparison|baseline)",
            r"insufficient.*(?:experiment|evaluation|empirical)",
            r"(?:weak|limited|inadequate).*(?:experiment|evaluation|baseline)",
            r"more.*(?:baseline|dataset|benchmark|experiment).*(?:need|should|required)",
            r"(?:unfair|incomplete).*comparison",
            r"cherry.pick",
            r"(?:not|no).*(?:state.of.the.art|SOTA|competitive)",
            r"(?:only|single|one).*(?:dataset|benchmark|task)",
            r"statistical significance",
            r"error bar",
            r"reproducib",
            r"hyperparameter.*(?:sensitivity|tuning|search)",
        ],
        "weak": [
            r"empirical(?:ly)?",
            r"benchmark",
            r"(?:additional|more).*experiment",
        ],
    },
    "theorist": {
        "strong": [
            r"(?:proof|theorem|lemma|corollary).*(?:error|gap|wrong|incomplete|missing|unclear|questionable)",
            r"(?:incorrect|flawed|invalid|wrong).*(?:proof|theorem|derivation|analysis)",
            r"(?:assumption|condition).*(?:strong|unrealistic|restrictive|violated|not.*hold)",
            r"convergence.*(?:guarantee|rate|proof|analysis)",
            r"(?:theoretical|formal).*(?:justification|analysis|foundation|grounding)",
            r"(?:lack|no|without|missing).*(?:theory|theoretical|formal|proof|guarantee)",
            r"mathematical.*(?:rigor|error|issue|problem)",
            r"bound.*(?:loose|tight|vacuous|trivial)",
            r"(?:hand.wav|informal|heuristic).*(?:argument|justification|reasoning)",
        ],
        "weak": [
            r"theoretical",
            r"formal",
            r"mathematical",
            r"assumption",
        ],
    },
    "systems_pragmatist": {
        "strong": [
            r"(?:memory|GPU|compute|FLOPs?).*(?:cost|overhead|requirement|footprint|consumption)",
            r"(?:latency|throughput|speed|runtime).*(?:high|slow|overhead|concern|cost)",
            r"(?:not|no|lack).*(?:practical|scalable|efficient|deployable)",
            r"(?:scaling|scale).*(?:concern|issue|problem|limit|poorly|not)",
            r"(?:inference|training).*(?:cost|time|overhead|expensive|slow)",
            r"(?:real.world|production|deployment|practical).*(?:concern|issue|applicab|limit)",
            r"hardware.*(?:requirement|assumption|specific|limit)",
            r"(?:wall.clock|actual|real).*time",
            r"computational.*(?:cost|overhead|burden|expensive|bottleneck)",
        ],
        "weak": [
            r"efficiency",
            r"scalab",
            r"compute",
            r"overhead",
        ],
    },
    "novelty_gatekeeper": {
        "strong": [
            r"(?:limited|lack|no|minimal|marginal|insufficient).*(?:novelty|contribution|innovation)",
            r"(?:incremental|marginal|minor|trivial).*(?:contribution|improvement|advance|extension)",
            r"(?:straightforward|trivial|simple).*(?:extension|combination|application|adaptation)",
            r"(?:already|previously).*(?:proposed|explored|known|studied|done)",
            r"(?:prior|existing|previous).*work.*(?:already|similar|same|overlap)",
            r"(?:not|no).*(?:novel|new|original|innovative|unique)",
            r"(?:direct|simple|naive).*(?:application|extension|adaptation).*(?:of|from).*(?:existing|prior|known)",
            r"(?:combination|concatenation|mashup).*(?:existing|known|prior)",
            r"distinguish.*from.*prior",
        ],
        "weak": [
            r"novelty",
            r"prior work",
            r"contribution",
            r"existing",
        ],
    },
}

ALL_PERSONAS = list(PERSONA_PATTERNS.keys())
VALID_LABELS = ALL_PERSONAS + ["mixed", "generic"]

# Default thresholds
DEFAULT_MARGIN_RATIO = 1.5
DEFAULT_MIN_SCORE = 2.0

# Weights
WEAKNESS_STRONG_WEIGHT = 2.0
WEAKNESS_WEAK_WEIGHT = 0.5
QUESTIONS_STRONG_WEIGHT = 1.0
QUESTIONS_WEAK_WEIGHT = 0.25


def _score_text(text: str, pattern_set: dict[str, list[str]], strong_weight: float, weak_weight: float) -> float:
    score = 0.0
    for pat in pattern_set.get("strong", []):
        score += len(re.findall(pat, text, re.IGNORECASE)) * strong_weight
    for pat in pattern_set.get("weak", []):
        score += len(re.findall(pat, text, re.IGNORECASE)) * weak_weight
    return score


def classify_reviewer(
    weakness_text: str,
    questions_text: str = "",
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    min_score: float = DEFAULT_MIN_SCORE,
) -> tuple[str, dict[str, float], bool]:
    """
    Classify a reviewer's persona from their weakness and questions text.

    Returns:
        (persona_label, scores_dict, is_confident)

        persona_label: one of "empiricist", "theorist", "systems_pragmatist",
                       "novelty_gatekeeper", "mixed", or "generic"
        scores_dict: raw scores per persona
        is_confident: True if a single persona was dominant with margin
    """
    scores: dict[str, float] = {}
    for persona, pattern_set in PERSONA_PATTERNS.items():
        s = _score_text(weakness_text, pattern_set, WEAKNESS_STRONG_WEIGHT, WEAKNESS_WEAK_WEIGHT)
        if questions_text:
            s += _score_text(questions_text, pattern_set, QUESTIONS_STRONG_WEIGHT, QUESTIONS_WEAK_WEIGHT)
        scores[persona] = round(s, 3)

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top_persona, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1]

    if top_score < min_score:
        return "generic", scores, False

    if second_score > 0 and top_score < margin_ratio * second_score:
        return "mixed", scores, False

    return top_persona, scores, True


def classify_review_dict(review: dict[str, Any]) -> tuple[str, dict[str, float], bool]:
    """
    Classify a reviewer from a review dict as stored in processed/HumanReview/*.json.

    Looks for review_sections.weaknesses first, falls back to review_text.
    """
    sections = review.get("review_sections", {})
    if isinstance(sections, dict) and sections.get("weaknesses"):
        weakness_text = sections["weaknesses"]
    else:
        weakness_text = review.get("review_text", "")

    if isinstance(sections, dict) and sections.get("questions"):
        questions_text = sections["questions"]
    else:
        questions_text = ""

    return classify_reviewer(weakness_text, questions_text)


def classify_paper_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Classify all reviewers for a paper and return summary.

    Returns dict with:
      reviewer_personas: list of {reviewer_id, persona, scores, confident}
      persona_distribution: Counter-like dict
      dominant_persona: most common persona across reviewers
      diversity: number of distinct personas
    """
    from collections import Counter

    results = []
    for rev in reviews:
        persona, scores, confident = classify_review_dict(rev)
        results.append({
            "reviewer_id": rev.get("reviewer_id", "unknown"),
            "persona": persona,
            "scores": scores,
            "confident": confident,
        })

    dist = Counter(r["persona"] for r in results)
    dominant = dist.most_common(1)[0][0] if dist else "generic"
    diversity = len(set(r["persona"] for r in results if r["persona"] not in ("generic", "mixed")))

    return {
        "reviewer_personas": results,
        "persona_distribution": dict(dist),
        "dominant_persona": dominant,
        "diversity": diversity,
        "n_reviews": len(results),
    }
