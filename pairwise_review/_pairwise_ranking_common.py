#!/usr/bin/env python3
"""
Shared utilities for pairwise ranking runs in the root code/ pipeline.
"""

from __future__ import annotations

import json
import random
import re
import sys
import statistics
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib import error, request

from _abstract_review_common import (
    ModelSpec,
    PRIMARY_AREA,
    TOGETHER_API_URL,
    now_utc,
    strip_model_scaffolding,
)
from _pairwise_prompt_library import (
    DEFAULT_PERSONA_SLUG,
    DEFAULT_PROMPT_ROOT,
    PairwisePromptBundle,
    build_pairwise_prompt_bundle,
    render_pairwise_user_prompt,
)
from _paper_content import resolve_paper_content

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.append(str(_CODE_ROOT))

from reviewer_persona_classifier import (  # noqa: E402
    ALL_PERSONAS,
    DEFAULT_MARGIN_RATIO,
    DEFAULT_MIN_SCORE,
    VALID_LABELS,
    classify_paper_reviews,
    classify_review_dict,
)

PAIRWISE_CATEGORY_KEYS = ("soundness", "presentation", "contribution")
FOCUS_PERSONA_SLUGS = tuple(ALL_PERSONAS)
HUMAN_FOCUS_LABELS = tuple(VALID_LABELS)

def summarize_prompt_bundles(prompt_bundles: list[PairwisePromptBundle]) -> dict[str, Any]:
    if not prompt_bundles:
        raise ValueError("At least one prompt bundle is required.")
    if len(prompt_bundles) == 1:
        bundle = prompt_bundles[0]
        return {
            "name": bundle.prompt_name,
            "source": bundle.prompt_source,
            "committee_mode": False,
            "personas": [
                {
                    "slug": bundle.persona.slug,
                    "label": bundle.persona.label,
                    "description": bundle.persona.description,
                    "path": str(bundle.persona.path),
                    "prompt_name": bundle.prompt_name,
                    "prompt_source": bundle.prompt_source,
                    "system_prompt_sha256": bundle.system_prompt_sha256,
                    "user_prompt_template_sha256": bundle.user_prompt_template_sha256,
                }
            ],
        }

    committee_name = "pairwise_committee_equal__" + "__".join(bundle.persona.slug for bundle in prompt_bundles)
    return {
        "name": committee_name,
        "source": " + ".join(bundle.prompt_source for bundle in prompt_bundles),
        "committee_mode": True,
        "personas": [
            {
                "slug": bundle.persona.slug,
                "label": bundle.persona.label,
                "description": bundle.persona.description,
                "path": str(bundle.persona.path),
                "prompt_name": bundle.prompt_name,
                "prompt_source": bundle.prompt_source,
                "system_prompt_sha256": bundle.system_prompt_sha256,
                "user_prompt_template_sha256": bundle.user_prompt_template_sha256,
            }
            for bundle in prompt_bundles
        ],
    }


@dataclass(frozen=True)
class PairwiseJudgeConfig:
    model: ModelSpec
    api_key: str
    prompt_root: str | None = None
    persona_slugs: tuple[str, ...] = (DEFAULT_PERSONA_SLUG,)
    content_mode: str = "abstract"
    fulltext_dir: str | None = None
    fulltext_selection: str = "core-sections"
    section_char_limit: int = 2500
    output_schema: str = "simple"
    prompt_strength: str = "standard"
    temperature: float = 0.0
    max_tokens: int = 400
    max_content_chars: int = 12000
    swap_order: bool = False
    timeout_seconds: float = 90.0
    max_retries: int = 3
    winner_threshold: float = 0.15
    sleep_seconds: float = 0.0


def total_unique_pairs(num_papers: int) -> int:
    return num_papers * (num_papers - 1) // 2


def make_pair_id(paper_a_id: str, paper_b_id: str) -> str:
    left, right = sorted((str(paper_a_id), str(paper_b_id)))
    return f"{left}__{right}"


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _human_review_record(paper: dict) -> dict | None:
    human_review = paper.get("human_review")
    if isinstance(human_review, dict) and human_review:
        return human_review
    if isinstance(paper.get("aggregated"), dict):
        return paper
    return None


def get_human_mean_rating(paper: dict, dimension: str = "rating") -> float | None:
    review = _human_review_record(paper)
    if not review:
        return None
    aggregated = review.get("aggregated", {}) or {}
    dim = aggregated.get(dimension, {}) or {}
    value = dim.get("mean")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_human_aggregated_stat(
    paper: dict,
    dimension: str = "rating",
    stat: str = "std",
) -> float | int | None:
    review = _human_review_record(paper)
    if not review:
        return None
    aggregated = review.get("aggregated", {}) or {}
    dim = aggregated.get(dimension, {}) or {}
    value = dim.get(stat)
    if value is None:
        return None
    try:
        if stat == "count":
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def human_pair_label(
    paper_a: dict,
    paper_b: dict,
    tie_delta: float = 0.25,
    dimension: str = "rating",
) -> str | None:
    score_a = get_human_mean_rating(paper_a, dimension=dimension)
    score_b = get_human_mean_rating(paper_b, dimension=dimension)
    if score_a is None or score_b is None:
        return None
    diff = score_a - score_b
    if abs(diff) <= tie_delta:
        return "Tie"
    return "A" if diff > 0 else "B"


def build_pair_schedule(
    papers: list[dict],
    strategy: str = "all",
    max_comparisons: int | None = None,
    seed: int = 42,
) -> list[dict]:
    indexed_pairs = []
    for paper_a, paper_b in combinations(sorted(papers, key=lambda paper: str(paper.get("paper_id"))), 2):
        indexed_pairs.append(
            {
                "pair_id": make_pair_id(paper_a["paper_id"], paper_b["paper_id"]),
                "paper_a_id": paper_a["paper_id"],
                "paper_b_id": paper_b["paper_id"],
            }
        )

    if strategy == "all":
        return indexed_pairs

    if strategy == "random":
        if max_comparisons is None or max_comparisons <= 0:
            raise ValueError("--max-comparisons must be set for strategy='random'")
        rng = random.Random(seed)
        rng.shuffle(indexed_pairs)
        return indexed_pairs[:max_comparisons]

    raise ValueError(f"Unsupported pair strategy: {strategy}")


def choose_anchor_ids(
    papers: list[dict],
    anchor_count: int,
    seed: int = 42,
    anchor_ids: list[str] | None = None,
) -> list[str]:
    paper_ids = sorted(str(paper["paper_id"]) for paper in papers)
    available = set(paper_ids)

    if anchor_ids:
        chosen = []
        seen = set()
        for paper_id in anchor_ids:
            if paper_id not in available or paper_id in seen:
                continue
            chosen.append(paper_id)
            seen.add(paper_id)
        if len(chosen) < 2:
            raise ValueError("At least two valid --anchor-paper-id values are required when explicit anchors are provided.")
        return chosen

    if anchor_count <= 0:
        raise ValueError("--anchor-count must be positive.")
    if anchor_count >= len(paper_ids):
        return paper_ids

    rng = random.Random(f"anchor:{seed}:{len(paper_ids)}")
    shuffled = list(paper_ids)
    rng.shuffle(shuffled)
    return sorted(shuffled[:anchor_count])


def build_anchor_schedule(
    papers: list[dict],
    anchor_count: int,
    seed: int = 42,
    anchor_ids: list[str] | None = None,
    max_anchor_comparisons_per_paper: int | None = None,
    include_anchor_pairs: bool = True,
) -> dict[str, Any]:
    paper_ids = sorted(str(paper["paper_id"]) for paper in papers)
    anchors = choose_anchor_ids(papers, anchor_count=anchor_count, seed=seed, anchor_ids=anchor_ids)
    anchor_set = set(anchors)
    pairs = []

    if include_anchor_pairs:
        for left_id, right_id in combinations(anchors, 2):
            pairs.append(
                {
                    "pair_id": make_pair_id(left_id, right_id),
                    "paper_a_id": left_id,
                    "paper_b_id": right_id,
                    "anchor_pair": True,
                    "board_index": len(pairs) + 1,
                }
            )

    for paper_id in paper_ids:
        if paper_id in anchor_set:
            continue
        selected_anchors = list(anchors)
        if max_anchor_comparisons_per_paper is not None and max_anchor_comparisons_per_paper < len(selected_anchors):
            rng = random.Random(f"anchor-paper:{seed}:{paper_id}")
            selected_anchors = sorted(rng.sample(selected_anchors, max_anchor_comparisons_per_paper))
        for anchor_id in selected_anchors:
            pairs.append(
                {
                    "pair_id": make_pair_id(anchor_id, paper_id),
                    "paper_a_id": anchor_id,
                    "paper_b_id": paper_id,
                    "anchor_pair": True,
                    "anchor_id": anchor_id,
                    "board_index": len(pairs) + 1,
                }
            )

    return {
        "anchor_ids": anchors,
        "pairs": pairs,
    }


def swiss_match_points(
    paper_ids: list[str],
    judgments: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    points = {paper_id: 0.0 for paper_id in paper_ids}
    games_played = {paper_id: 0 for paper_id in paper_ids}

    for judgment in judgments:
        paper_a_id = judgment["paper_a_id"]
        paper_b_id = judgment["paper_b_id"]
        if paper_a_id not in points or paper_b_id not in points:
            continue

        winner = judgment.get("final_winner")
        if winner == "A":
            points[paper_a_id] += 1.0
        elif winner == "B":
            points[paper_b_id] += 1.0
        else:
            points[paper_a_id] += 0.5
            points[paper_b_id] += 0.5

        games_played[paper_a_id] += 1
        games_played[paper_b_id] += 1

    return points, games_played


def build_swiss_round(
    paper_ids: list[str],
    judgments: list[dict[str, Any]],
    round_index: int,
    seed: int = 42,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    ordered_paper_ids = sorted(str(paper_id) for paper_id in paper_ids)
    if len(ordered_paper_ids) < 2:
        return {
            "round_index": round_index,
            "ordered_paper_ids": ordered_paper_ids,
            "pairs": [],
            "unpaired_paper_ids": ordered_paper_ids,
        }

    match_points, games_played = swiss_match_points(ordered_paper_ids, judgments)
    latent_scores = fit_bradley_terry(judgments, ordered_paper_ids) if judgments else {
        paper_id: 1.0 for paper_id in ordered_paper_ids
    }

    jitter_rng = random.Random(f"swiss:{seed}:{round_index}")
    jitter = {paper_id: jitter_rng.random() for paper_id in ordered_paper_ids}
    ranking_order = sorted(
        ordered_paper_ids,
        key=lambda paper_id: (
            -match_points[paper_id],
            games_played[paper_id],
            -latent_scores[paper_id],
            jitter[paper_id],
            paper_id,
        ),
    )

    used_pair_ids = {
        make_pair_id(judgment["paper_a_id"], judgment["paper_b_id"])
        for judgment in judgments
    }
    remaining = list(ranking_order)
    pairs = []
    unpaired_paper_ids = []

    while remaining:
        paper_a_id = remaining.pop(0)
        candidate_index = None
        for idx, paper_b_id in enumerate(remaining):
            if make_pair_id(paper_a_id, paper_b_id) not in used_pair_ids:
                candidate_index = idx
                break

        if candidate_index is None:
            unpaired_paper_ids.append(paper_a_id)
            continue

        paper_b_id = remaining.pop(candidate_index)
        pair = {
            "pair_id": make_pair_id(paper_a_id, paper_b_id),
            "paper_a_id": paper_a_id,
            "paper_b_id": paper_b_id,
            "round_index": round_index,
            "board_index": len(pairs) + 1,
        }
        pairs.append(pair)
        if max_pairs is not None and len(pairs) >= max_pairs:
            unpaired_paper_ids.extend(remaining)
            break

    return {
        "round_index": round_index,
        "ordered_paper_ids": ranking_order,
        "pairs": pairs,
        "unpaired_paper_ids": unpaired_paper_ids,
    }


def normalise_winner(value: object) -> str:
    if value is None:
        return "Tie"
    norm = str(value).strip().lower()
    if norm in {"a", "paper_a", "paper a", "first", "left"}:
        return "A"
    if norm in {"b", "paper_b", "paper b", "second", "right"}:
        return "B"
    if norm in {"tie", "tied", "equal", "draw", "neither"}:
        return "Tie"
    return "Tie"


def invert_winner(value: str) -> str:
    if value == "A":
        return "B"
    if value == "B":
        return "A"
    return "Tie"


def invert_category_winners(category_winners: dict[str, str]) -> dict[str, str]:
    return {
        key: invert_winner(normalise_winner(value))
        for key, value in category_winners.items()
    }


def parse_pairwise_response(content: str) -> dict[str, Any]:
    raw_text = strip_model_scaffolding((content or "").strip())
    parsed: dict[str, Any] | None = None

    try:
        maybe = json.loads(raw_text)
        if isinstance(maybe, dict):
            parsed = maybe
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                maybe = json.loads(raw_text[start : end + 1])
                if isinstance(maybe, dict):
                    parsed = maybe
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        winner_match = re.search(r'"?(?:overall_winner|winner)"?\s*:\s*"?(A|B|Tie|tie|a|b)"?', raw_text)
        confidence_match = re.search(r'"?confidence"?\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_text)
        rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"]+)"', raw_text)
        if winner_match:
            parsed = {
                "overall_winner": winner_match.group(1),
                "category_winners": {},
                "confidence": confidence_match.group(1) if confidence_match else 0.5,
                "rationale": rationale_match.group(1) if rationale_match else "",
                "rationale_bullets": [],
            }
        else:
            parsed = {}

    overall_winner = normalise_winner(parsed.get("overall_winner", parsed.get("winner")))
    try:
        confidence = _clamp_unit_interval(float(parsed.get("confidence", 0.5)))
    except (TypeError, ValueError):
        confidence = 0.5

    raw_category_winners = parsed.get("category_winners", {})
    if not isinstance(raw_category_winners, dict):
        raw_category_winners = {}
    category_winners = {}
    if raw_category_winners:
        category_winners = {
            key: normalise_winner(raw_category_winners.get(key, overall_winner))
            for key in PAIRWISE_CATEGORY_KEYS
        }

    bullets_value = parsed.get("rationale_bullets", [])
    if isinstance(bullets_value, str):
        rationale_bullets = [bullets_value.strip()] if bullets_value.strip() else []
    elif isinstance(bullets_value, list):
        rationale_bullets = [str(item).strip() for item in bullets_value if str(item).strip()]
    else:
        rationale_bullets = []
    if not rationale_bullets:
        rationale_text = str(parsed.get("rationale", "")).strip()
        if rationale_text:
            rationale_bullets = [rationale_text]

    return {
        "overall_winner": overall_winner,
        "category_winners": category_winners,
        "confidence": confidence,
        "rationale_bullets": rationale_bullets[:4],
        "raw_text": raw_text,
    }


def aggregate_winner_votes(
    calls: list[dict[str, Any]],
    winner_threshold: float = 0.15,
    field: str = "overall_winner",
) -> dict[str, Any]:
    valid_calls = [call for call in calls if not call.get("http_error")]
    if not valid_calls:
        return {
            "winner": None,
            "margin": None,
            "confidence": 0.0,
            "a_share": None,
            "b_share": None,
            "valid_calls": 0,
            "invalid_calls": len(calls),
        }

    signed_support = 0.0
    confidences = []
    for call in valid_calls:
        confidence = _clamp_unit_interval(float(call.get("confidence", 0.5)))
        confidences.append(confidence)
        winner = normalise_winner(call.get(field))
        if winner == "A":
            signed_support += confidence
        elif winner == "B":
            signed_support -= confidence

    final_margin = signed_support / len(valid_calls)
    if final_margin > winner_threshold:
        final_winner = "A"
    elif final_margin < -winner_threshold:
        final_winner = "B"
    else:
        final_winner = "Tie"

    a_share = _clamp_unit_interval((final_margin + 1.0) / 2.0)
    b_share = 1.0 - a_share
    return {
        "winner": final_winner,
        "margin": final_margin,
        "confidence": statistics.mean(confidences),
        "a_share": a_share,
        "b_share": b_share,
        "valid_calls": len(valid_calls),
        "invalid_calls": len(calls) - len(valid_calls),
    }


def aggregate_pair_calls(
    calls: list[dict[str, Any]],
    winner_threshold: float = 0.15,
) -> dict[str, Any]:
    overall = aggregate_winner_votes(
        calls,
        winner_threshold=winner_threshold,
        field="overall_winner",
    )
    available_category_keys = []
    for key in PAIRWISE_CATEGORY_KEYS:
        if any(isinstance(call.get("category_winners"), dict) and key in call["category_winners"] for call in calls):
            available_category_keys.append(key)

    by_category = {}
    for key in available_category_keys:
        category_calls = []
        for call in calls:
            category_calls.append(
                {
                    "confidence": call.get("confidence", 0.5),
                    "overall_winner": (call.get("category_winners") or {}).get(key, "Tie"),
                }
            )
        by_category[key] = aggregate_winner_votes(
            category_calls,
            winner_threshold=winner_threshold,
            field="overall_winner",
        )

    return {
        "final_winner": overall["winner"],
        "final_margin": overall["margin"],
        "final_confidence": overall["confidence"],
        "a_share": overall["a_share"],
        "b_share": overall["b_share"],
        "valid_call_count": overall["valid_calls"],
        "invalid_call_count": overall["invalid_calls"],
        "has_api_error": overall["invalid_calls"] > 0,
        "final_category_winners": {key: value["winner"] for key, value in by_category.items()},
        "final_category_margins": {key: value["margin"] for key, value in by_category.items()},
    }


def fit_bradley_terry(
    judgments: list[dict[str, Any]],
    paper_ids: list[str],
    max_iter: int = 200,
    tol: float = 1e-8,
    prior: float = 1e-6,
) -> dict[str, float]:
    weights = {paper_id: 1.0 for paper_id in paper_ids}
    pair_weights: dict[tuple[str, str], float] = {}
    wins: dict[str, float] = {paper_id: prior for paper_id in paper_ids}

    for judgment in judgments:
        if judgment.get("final_winner") not in {"A", "B", "Tie"}:
            continue
        paper_a_id = judgment["paper_a_id"]
        paper_b_id = judgment["paper_b_id"]
        a_share = float(judgment.get("a_share", 0.5))
        b_share = float(judgment.get("b_share", 0.5))

        wins[paper_a_id] += a_share
        wins[paper_b_id] += b_share

        key = tuple(sorted((paper_a_id, paper_b_id)))
        pair_weights[key] = pair_weights.get(key, 0.0) + a_share + b_share

    if not pair_weights:
        return weights

    for _ in range(max_iter):
        max_delta = 0.0
        new_weights: dict[str, float] = {}
        for paper_id in paper_ids:
            denominator = 0.0
            for other_id in paper_ids:
                if paper_id == other_id:
                    continue
                key = tuple(sorted((paper_id, other_id)))
                total_weight = pair_weights.get(key, 0.0)
                if total_weight == 0.0:
                    continue
                denominator += total_weight / (weights[paper_id] + weights[other_id])

            if denominator == 0.0:
                new_weight = weights[paper_id]
            else:
                new_weight = max(prior, wins[paper_id] / denominator)
            new_weights[paper_id] = new_weight
            max_delta = max(max_delta, abs(new_weight - weights[paper_id]))

        normalizer = statistics.mean(new_weights.values()) or 1.0
        for paper_id in paper_ids:
            new_weights[paper_id] /= normalizer

        weights = new_weights
        if max_delta < tol:
            break

    return weights


def build_ranking(scores: dict[str, float], papers_by_id: dict[str, dict]) -> list[dict[str, Any]]:
    ordered_ids = sorted(scores, key=lambda paper_id: (-scores[paper_id], paper_id))
    ranking = []
    for rank, paper_id in enumerate(ordered_ids, 1):
        paper = papers_by_id[paper_id]
        ranking.append(
            {
                "rank": rank,
                "paper_id": paper_id,
                "latent_score": scores[paper_id],
                "title": paper.get("title"),
                "primary_area": paper.get("primary_area", PRIMARY_AREA),
                "decision": paper.get("decision"),
                "human_mean_rating": get_human_mean_rating(paper),
            }
        )
    return ranking


def spearman_rank_correlation_from_orders(predicted_ids: list[str], human_ids: list[str]) -> float | None:
    common_ids = set(predicted_ids) & set(human_ids)
    n = len(common_ids)
    if n < 2:
        return None

    pred_ranks = {paper_id: rank for rank, paper_id in enumerate(predicted_ids, 1)}
    human_ranks = {paper_id: rank for rank, paper_id in enumerate(human_ids, 1)}

    d_squared = 0
    for paper_id in common_ids:
        d = pred_ranks[paper_id] - human_ranks[paper_id]
        d_squared += d * d
    return 1 - (6 * d_squared) / (n * (n * n - 1))


def build_human_order(papers: list[dict], dimension: str = "rating") -> list[str]:
    scored = []
    for paper in papers:
        score = get_human_mean_rating(paper, dimension=dimension)
        if score is None:
            continue
        scored.append((paper["paper_id"], score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [paper_id for paper_id, _score in scored]


def category_judgment_as_pairwise_row(judgment: dict[str, Any], category: str) -> dict[str, Any] | None:
    final_category_winners = judgment.get("final_category_winners") or {}
    winner = final_category_winners.get(category)
    if winner not in {"A", "B", "Tie"}:
        return None

    final_category_margins = judgment.get("final_category_margins") or {}
    margin_value = final_category_margins.get(category)
    if isinstance(margin_value, (int, float)):
        margin = max(-1.0, min(1.0, float(margin_value)))
        a_share = _clamp_unit_interval((margin + 1.0) / 2.0)
    elif winner == "A":
        a_share = 1.0
    elif winner == "B":
        a_share = 0.0
    else:
        a_share = 0.5

    return {
        "paper_a_id": judgment["paper_a_id"],
        "paper_b_id": judgment["paper_b_id"],
        "a_share": a_share,
        "b_share": 1.0 - a_share,
    }


def judgment_has_call_disagreement(judgment: dict[str, Any]) -> bool:
    calls = judgment.get("calls") or []
    winners = [normalise_winner(call.get("overall_winner")) for call in calls if isinstance(call, dict)]
    return len(set(winners)) > 1


def select_uncertain_judgments(
    judgments: list[dict[str, Any]],
    margin_threshold: float = 0.1,
    max_pairs: int | None = None,
    anchor_ids: set[str] | None = None,
    reask_anchor_upsets: bool = False,
    upset_margin_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    upset_ranked = []
    uncertain_ranked = []
    for judgment in judgments:
        final_winner = normalise_winner(judgment.get("final_winner"))
        final_margin = abs(float(judgment.get("final_margin", 0.0)))
        disagreement = judgment_has_call_disagreement(judgment)
        is_uncertain = final_winner == "Tie" or final_margin < margin_threshold or disagreement
        is_anchor_upset = False
        if reask_anchor_upsets and anchor_ids:
            paper_a_id = str(judgment.get("paper_a_id", ""))
            paper_b_id = str(judgment.get("paper_b_id", ""))
            anchor_a = paper_a_id in anchor_ids
            anchor_b = paper_b_id in anchor_ids
            if anchor_a ^ anchor_b:
                expected_anchor_winner = "A" if anchor_a else "B"
                if final_winner in {"A", "B"} and final_winner != expected_anchor_winner and final_margin >= upset_margin_threshold:
                    is_anchor_upset = True

        if is_anchor_upset:
            upset_ranked.append(
                (
                    -final_margin,
                    judgment.get("pair_id", ""),
                    judgment,
                )
            )
        if is_uncertain:
            uncertain_ranked.append(
                (
                    0 if disagreement else 1,
                    0 if final_winner == "Tie" else 1,
                    final_margin,
                    judgment.get("pair_id", ""),
                    judgment,
                )
            )

    upset_ranked.sort(key=lambda item: item[:2])
    uncertain_ranked.sort(key=lambda item: item[:4])

    selected = []
    seen_pair_ids = set()
    for _priority, _pair_id, judgment in upset_ranked:
        pair_id = judgment.get("pair_id", "")
        if pair_id in seen_pair_ids:
            continue
        selected.append(judgment)
        seen_pair_ids.add(pair_id)
    for _disagreement_rank, _tie_rank, _margin, _pair_id, judgment in uncertain_ranked:
        pair_id = judgment.get("pair_id", "")
        if pair_id in seen_pair_ids:
            continue
        selected.append(judgment)
        seen_pair_ids.add(pair_id)

    if max_pairs is not None:
        return selected[:max_pairs]
    return selected


def evaluate_results(
    judgments: list[dict],
    ranking: list[dict],
    papers_by_id: dict[str, dict],
    tie_delta: float,
) -> dict:
    evaluation = _evaluate_results_core(judgments, ranking, papers_by_id, tie_delta)
    persona_judgments_by_slug = build_persona_specific_judgments(judgments)
    evaluation["human_focus_overlap"] = build_human_focus_overlap_summary(
        papers_by_id,
        persona_judgments_by_slug=persona_judgments_by_slug,
    )
    if persona_judgments_by_slug:
        persona_evaluation = {}
        paper_ids = sorted(papers_by_id)
        for persona_slug, persona_judgments in sorted(persona_judgments_by_slug.items()):
            persona_scores = fit_bradley_terry(persona_judgments, paper_ids)
            persona_ranking = build_ranking(persona_scores, papers_by_id)
            persona_metrics = _evaluate_results_core(persona_judgments, persona_ranking, papers_by_id, tie_delta)
            persona_metrics["aligned_focus_proxy"] = build_persona_focus_proxy(persona_slug, persona_metrics)
            persona_metrics["num_pairs"] = len(persona_judgments)
            persona_evaluation[persona_slug] = persona_metrics
        evaluation["persona_evaluation"] = persona_evaluation

    return evaluation


def _evaluate_results_core(
    judgments: list[dict],
    ranking: list[dict],
    papers_by_id: dict[str, dict],
    tie_delta: float,
) -> dict:
    valid_judgments = [judgment for judgment in judgments if judgment.get("final_winner") in {"A", "B", "Tie"}]
    invalid_judgments = len(judgments) - len(valid_judgments)
    api_error_calls = sum(
        1
        for judgment in judgments
        for call in (judgment.get("calls") or [])
        if isinstance(call, dict) and call.get("http_error")
    )
    papers = list(papers_by_id.values())
    human_order = build_human_order(papers)
    predicted_order = [row["paper_id"] for row in ranking]
    rank_rho = spearman_rank_correlation_from_orders(predicted_order, human_order) if valid_judgments else None

    comparable_pairs = 0
    correct_pairs = 0
    decisive_pairs = 0
    decisive_correct = 0
    for judgment in valid_judgments:
        paper_a = papers_by_id[judgment["paper_a_id"]]
        paper_b = papers_by_id[judgment["paper_b_id"]]
        label = human_pair_label(paper_a, paper_b, tie_delta=tie_delta)
        if label is None:
            continue
        comparable_pairs += 1
        if label == judgment["final_winner"]:
            correct_pairs += 1
        if label != "Tie":
            decisive_pairs += 1
            if label == judgment["final_winner"]:
                decisive_correct += 1

    topk_overlap = {}
    for k in (10, 20, 30):
        if not human_order or not valid_judgments:
            continue
        k_eff = min(k, len(human_order), len(predicted_order))
        if k_eff <= 0:
            continue
        human_top = set(human_order[:k_eff])
        predicted_top = set(predicted_order[:k_eff])
        topk_overlap[str(k_eff)] = len(human_top & predicted_top) / k_eff

    category_evaluation = {}
    paper_ids = sorted(papers_by_id)
    for category in PAIRWISE_CATEGORY_KEYS:
        category_comparable_pairs = 0
        category_correct_pairs = 0
        category_decisive_pairs = 0
        category_decisive_correct = 0
        category_pair_rows = []

        for judgment in valid_judgments:
            predicted_label = (judgment.get("final_category_winners") or {}).get(category)
            if predicted_label not in {"A", "B", "Tie"}:
                continue

            paper_a = papers_by_id[judgment["paper_a_id"]]
            paper_b = papers_by_id[judgment["paper_b_id"]]
            human_label = human_pair_label(paper_a, paper_b, tie_delta=tie_delta, dimension=category)
            if human_label is not None:
                category_comparable_pairs += 1
                if human_label == predicted_label:
                    category_correct_pairs += 1
                if human_label != "Tie":
                    category_decisive_pairs += 1
                    if human_label == predicted_label:
                        category_decisive_correct += 1

            category_row = category_judgment_as_pairwise_row(judgment, category)
            if category_row is not None:
                category_pair_rows.append(category_row)

        human_category_order = build_human_order(papers, dimension=category)
        category_scores = fit_bradley_terry(category_pair_rows, paper_ids) if category_pair_rows else {paper_id: 1.0 for paper_id in paper_ids}
        predicted_category_order = sorted(category_scores, key=lambda paper_id: (-category_scores[paper_id], paper_id))

        category_evaluation[category] = {
            "rank_spearman_rho": spearman_rank_correlation_from_orders(predicted_category_order, human_category_order),
            "pairwise_accuracy": (category_correct_pairs / category_comparable_pairs) if category_comparable_pairs else None,
            "pairwise_accuracy_n": category_comparable_pairs,
            "decisive_pairwise_accuracy": (category_decisive_correct / category_decisive_pairs) if category_decisive_pairs else None,
            "decisive_pairwise_accuracy_n": category_decisive_pairs,
        }

    return {
        "rank_spearman_rho": rank_rho,
        "pairwise_accuracy": (correct_pairs / comparable_pairs) if comparable_pairs else None,
        "pairwise_accuracy_n": comparable_pairs,
        "decisive_pairwise_accuracy": (decisive_correct / decisive_pairs) if decisive_pairs else None,
        "decisive_pairwise_accuracy_n": decisive_pairs,
        "topk_overlap": topk_overlap,
        "category_evaluation": category_evaluation,
        "invalid_judgments": invalid_judgments,
        "api_error_calls": api_error_calls,
    }


def build_persona_specific_judgments(
    judgments: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_persona: dict[str, list[dict[str, Any]]] = {}
    for judgment in judgments:
        calls = judgment.get("calls") or []
        persona_calls: dict[str, list[dict[str, Any]]] = {}
        for call in calls:
            if not isinstance(call, dict):
                continue
            persona_slug = str(call.get("persona_slug") or "").strip()
            if not persona_slug:
                continue
            persona_calls.setdefault(persona_slug, []).append(call)

        for persona_slug, grouped_calls in persona_calls.items():
            aggregate = aggregate_pair_calls(grouped_calls)
            if aggregate.get("final_winner") not in {"A", "B", "Tie"}:
                continue
            by_persona.setdefault(persona_slug, []).append(
                {
                    "pair_id": judgment["pair_id"],
                    "paper_a_id": judgment["paper_a_id"],
                    "paper_b_id": judgment["paper_b_id"],
                    **aggregate,
                }
            )
    return by_persona


def infer_review_focus(review: dict[str, Any]) -> dict[str, Any]:
    review_sections = review.get("review_sections") or {}
    has_structured = any(
        str(review_sections.get(key) or "").strip()
        for key in ("summary", "strength", "weaknesses", "questions")
    )
    has_review_text = bool(str(review.get("review_text") or "").strip())
    if not has_structured and not has_review_text:
        return {
            "available": False,
            "dominant_persona": None,
            "persona_scores": {persona: 0.0 for persona in FOCUS_PERSONA_SLUGS},
            "confident": False,
        }

    dominant_persona, persona_scores, confident = classify_review_dict(review)

    return {
        "available": True,
        "dominant_persona": dominant_persona,
        "persona_scores": persona_scores,
        "confident": confident,
    }


def build_human_paper_focus_profiles(papers_by_id: dict[str, dict]) -> dict[str, dict[str, Any]]:
    profiles = {}
    for paper_id, paper in papers_by_id.items():
        human_review = _human_review_record(paper) or {}
        paper_reviews = human_review.get("reviews") or []
        paper_personas = classify_paper_reviews(paper_reviews)
        aggregate_scores = {persona: 0.0 for persona in FOCUS_PERSONA_SLUGS}
        review_counts = {persona: 0 for persona in HUMAN_FOCUS_LABELS}
        usable_reviews = 0
        confident_reviews = 0
        for review_result in paper_personas.get("reviewer_personas", []):
            usable_reviews += 1
            if review_result.get("confident"):
                confident_reviews += 1
            for persona, value in (review_result.get("scores") or {}).items():
                if persona in aggregate_scores:
                    aggregate_scores[persona] += float(value)
            dominant = review_result.get("persona")
            if dominant in review_counts:
                review_counts[dominant] += 1
        dominant_persona = paper_personas.get("dominant_persona")
        profiles[paper_id] = {
            "paper_id": paper_id,
            "dominant_persona": dominant_persona,
            "aggregate_scores": aggregate_scores,
            "review_counts": review_counts,
            "usable_reviews": usable_reviews,
            "confident_reviews": confident_reviews,
            "diversity": paper_personas.get("diversity"),
        }
    return profiles


def human_pair_focus_label(
    paper_a_id: str,
    paper_b_id: str,
    paper_focus_profiles: dict[str, dict[str, Any]],
) -> str | None:
    profile_a = paper_focus_profiles.get(paper_a_id) or {}
    profile_b = paper_focus_profiles.get(paper_b_id) or {}
    combined = {
        persona: float((profile_a.get("aggregate_scores") or {}).get(persona, 0.0))
        + float((profile_b.get("aggregate_scores") or {}).get(persona, 0.0))
        for persona in FOCUS_PERSONA_SLUGS
    }
    ranked = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] < DEFAULT_MIN_SCORE:
        return None
    if len(ranked) > 1 and ranked[1][1] > 0 and ranked[0][1] < DEFAULT_MARGIN_RATIO * ranked[1][1]:
        return "mixed"
    return ranked[0][0]


def _evaluate_pair_label_accuracy(
    judgments: list[dict[str, Any]],
    papers_by_id: dict[str, dict],
    tie_delta: float,
) -> dict[str, Any]:
    comparable_pairs = 0
    correct_pairs = 0
    decisive_pairs = 0
    decisive_correct = 0
    for judgment in judgments:
        if judgment.get("final_winner") not in {"A", "B", "Tie"}:
            continue
        paper_a = papers_by_id[judgment["paper_a_id"]]
        paper_b = papers_by_id[judgment["paper_b_id"]]
        label = human_pair_label(paper_a, paper_b, tie_delta=tie_delta)
        if label is None:
            continue
        comparable_pairs += 1
        if label == judgment["final_winner"]:
            correct_pairs += 1
        if label != "Tie":
            decisive_pairs += 1
            if label == judgment["final_winner"]:
                decisive_correct += 1

    return {
        "pairwise_accuracy": (correct_pairs / comparable_pairs) if comparable_pairs else None,
        "pairwise_accuracy_n": comparable_pairs,
        "decisive_pairwise_accuracy": (decisive_correct / decisive_pairs) if decisive_pairs else None,
        "decisive_pairwise_accuracy_n": decisive_pairs,
    }


def build_human_focus_overlap_summary(
    papers_by_id: dict[str, dict],
    persona_judgments_by_slug: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    total_reviews = 0
    nonempty_review_text = 0
    nonempty_structured_reviews = 0
    for paper in papers_by_id.values():
        human_review = _human_review_record(paper) or {}
        for review in human_review.get("reviews", []) or []:
            total_reviews += 1
            if str(review.get("review_text") or "").strip():
                nonempty_review_text += 1
            review_sections = review.get("review_sections") or {}
            if any(str(review_sections.get(key) or "").strip() for key in ("summary", "strength", "weaknesses", "questions")):
                nonempty_structured_reviews += 1

    if nonempty_review_text == 0 and nonempty_structured_reviews == 0:
        return {
            "available": False,
            "reason": (
                "Local human-review exports contain no non-empty review_text fields and no structured review "
                "sections, so human focus labels cannot be inferred yet."
            ),
            "total_reviews": total_reviews,
            "nonempty_review_text": nonempty_review_text,
            "nonempty_structured_reviews": nonempty_structured_reviews,
        }

    paper_focus_profiles = build_human_paper_focus_profiles(papers_by_id)
    review_focus_counts = {persona: 0 for persona in HUMAN_FOCUS_LABELS}
    paper_focus_counts = {persona: 0 for persona in HUMAN_FOCUS_LABELS}
    for paper in papers_by_id.values():
        human_review = _human_review_record(paper) or {}
        for review in human_review.get("reviews", []) or []:
            dominant = infer_review_focus(review).get("dominant_persona")
            if dominant in review_focus_counts:
                review_focus_counts[dominant] += 1
    for profile in paper_focus_profiles.values():
        dominant = profile.get("dominant_persona")
        if dominant in paper_focus_counts:
            paper_focus_counts[dominant] += 1

    result = {
        "available": True,
        "reason": (
            "Human focus is inferred with reviewer_persona_classifier from structured review text, "
            "primarily weaknesses and secondarily questions, with explicit generic/mixed labels."
        ),
        "total_reviews": total_reviews,
        "nonempty_review_text": nonempty_review_text,
        "nonempty_structured_reviews": nonempty_structured_reviews,
        "review_focus_counts": review_focus_counts,
        "paper_focus_counts": paper_focus_counts,
    }

    if persona_judgments_by_slug:
        persona_overlap = {}
        for persona_slug, persona_judgments in sorted(persona_judgments_by_slug.items()):
            if persona_slug not in FOCUS_PERSONA_SLUGS:
                continue
            matched = []
            unmatched = []
            unlabeled = []
            for judgment in persona_judgments:
                pair_focus = human_pair_focus_label(
                    judgment["paper_a_id"],
                    judgment["paper_b_id"],
                    paper_focus_profiles,
                )
                if pair_focus is None or pair_focus not in FOCUS_PERSONA_SLUGS:
                    unlabeled.append(judgment)
                elif pair_focus == persona_slug:
                    matched.append(judgment)
                else:
                    unmatched.append(judgment)
            persona_overlap[persona_slug] = {
                "pair_focus_definition": "dominant persona from aggregated structured human review text across both papers in the pair",
                "matched": _evaluate_pair_label_accuracy(matched, papers_by_id, tie_delta=0.25),
                "matched_pairs": len(matched),
                "unmatched": _evaluate_pair_label_accuracy(unmatched, papers_by_id, tie_delta=0.25),
                "unmatched_pairs": len(unmatched),
                "unlabeled_pairs": len(unlabeled),
            }
        result["persona_overlap"] = persona_overlap

    return result


def build_persona_focus_proxy(persona_slug: str, metrics: dict[str, Any]) -> dict[str, Any]:
    dimension_by_persona = {
        "theorist": "soundness",
        "empiricist": "soundness",
        "novelty_gatekeeper": "contribution",
        "systems_pragmatist": "contribution",
    }
    dimension = dimension_by_persona.get(persona_slug)
    if not dimension:
        return {
            "available": False,
            "reason": "No aligned human-score proxy dimension defined for this persona.",
        }

    category_metrics = (metrics.get("category_evaluation") or {}).get(dimension)
    if not category_metrics:
        return {
            "available": False,
            "dimension": dimension,
            "reason": "Category-level proxy metrics are unavailable for this run.",
        }

    return {
        "available": True,
        "dimension": dimension,
        "rank_spearman_rho": category_metrics.get("rank_spearman_rho"),
        "pairwise_accuracy": category_metrics.get("pairwise_accuracy"),
        "pairwise_accuracy_n": category_metrics.get("pairwise_accuracy_n"),
        "decisive_pairwise_accuracy": category_metrics.get("decisive_pairwise_accuracy"),
        "decisive_pairwise_accuracy_n": category_metrics.get("decisive_pairwise_accuracy_n"),
    }


def render_summary_markdown(
    config: dict,
    budget: dict,
    evaluation: dict,
    ranking: list[dict],
) -> str:
    content_cfg = config.get("content", {})
    pairing_cfg = config.get("pairing", {})
    lines = [
        "# Pairwise Ranking Run",
        "",
        "## Configuration",
        "",
        f"- Input: `{config['input']}`",
        f"- Model: `{config['model']}`",
        f"- Content mode: `{content_cfg.get('requested_mode')}`",
        f"- Fulltext selection: `{content_cfg.get('fulltext_selection')}`",
        f"- Pair strategy: `{pairing_cfg.get('pair_strategy')}`",
        f"- Swap order: `{pairing_cfg.get('swap_order')}`",
        f"- Papers ranked: {budget['num_papers']}",
        f"- Unique pairs evaluated: {budget['scheduled_pairs']}",
        f"- Together API calls: {budget['api_calls']}",
        "",
        "## Metrics",
        "",
    ]

    if evaluation["rank_spearman_rho"] is not None:
        lines.append(f"- Rank Spearman rho vs human mean rating: {evaluation['rank_spearman_rho']:.4f}")
    if budget.get("planned_rounds") is not None:
        lines.append(
            f"- Swiss rounds: {budget['rounds_scheduled']}/{budget['planned_rounds']} "
            f"(target pairs: {budget['planned_pairs']})"
        )
        lines.append(f"- Planned Together API calls: {budget['planned_api_calls']}")
    if evaluation["pairwise_accuracy"] is not None:
        lines.append(
            f"- Pairwise accuracy: {evaluation['pairwise_accuracy']:.4f} "
            f"({evaluation['pairwise_accuracy_n']} judged pairs)"
        )
    invalid_judgments = evaluation.get("invalid_judgments")
    if invalid_judgments:
        lines.append(f"- Invalid judgments excluded from scoring: {invalid_judgments}")
    api_error_calls = evaluation.get("api_error_calls")
    if api_error_calls:
        lines.append(f"- API-error calls excluded from scoring: {api_error_calls}")
    if evaluation["decisive_pairwise_accuracy"] is not None:
        lines.append(
            f"- Decisive-pair accuracy: {evaluation['decisive_pairwise_accuracy']:.4f} "
            f"({evaluation['decisive_pairwise_accuracy_n']} decisive human pairs)"
        )
    if evaluation["topk_overlap"]:
        for k, overlap in sorted(evaluation["topk_overlap"].items(), key=lambda item: int(item[0])):
            lines.append(f"- Top-{k} overlap: {overlap:.4f}")

    category_evaluation = evaluation.get("category_evaluation") or {}
    if category_evaluation:
        lines.extend(["", "## Category Diagnostics", ""])
        for category in PAIRWISE_CATEGORY_KEYS:
            metrics = category_evaluation.get(category)
            if not metrics:
                continue
            label = category.title()
            rho = metrics.get("rank_spearman_rho")
            if rho is not None:
                lines.append(f"- {label} rank Spearman rho: {rho:.4f}")
            if metrics.get("pairwise_accuracy") is not None:
                lines.append(
                    f"- {label} pairwise accuracy: {metrics['pairwise_accuracy']:.4f} "
                    f"({metrics['pairwise_accuracy_n']} judged pairs)"
                )
            if metrics.get("decisive_pairwise_accuracy") is not None:
                lines.append(
                    f"- {label} decisive-pair accuracy: {metrics['decisive_pairwise_accuracy']:.4f} "
                    f"({metrics['decisive_pairwise_accuracy_n']} decisive human pairs)"
                )

    human_focus_overlap = evaluation.get("human_focus_overlap") or {}
    if human_focus_overlap:
        lines.extend(["", "## Human Focus Overlap", ""])
        available = bool(human_focus_overlap.get("available"))
        lines.append(f"- Available: {available}")
        if human_focus_overlap.get("reason"):
            lines.append(f"- Note: {human_focus_overlap['reason']}")
        if human_focus_overlap.get("total_reviews") is not None:
            lines.append(
                f"- Human review text coverage: {human_focus_overlap.get('nonempty_review_text', 0)}/"
                f"{human_focus_overlap.get('total_reviews', 0)} non-empty reviews"
            )
        if human_focus_overlap.get("nonempty_structured_reviews") is not None:
            lines.append(
                f"- Structured review coverage: {human_focus_overlap.get('nonempty_structured_reviews', 0)}/"
                f"{human_focus_overlap.get('total_reviews', 0)} reviews with summary/strength/weaknesses/questions"
            )
        review_focus_counts = human_focus_overlap.get("review_focus_counts") or {}
        if review_focus_counts:
            lines.append(f"- Review-level focus counts: {review_focus_counts}")
        paper_focus_counts = human_focus_overlap.get("paper_focus_counts") or {}
        if paper_focus_counts:
            lines.append(f"- Paper-level dominant focus counts: {paper_focus_counts}")
        persona_overlap = human_focus_overlap.get("persona_overlap") or {}
        if persona_overlap:
            for persona_slug, overlap in persona_overlap.items():
                matched = overlap.get("matched") or {}
                unmatched = overlap.get("unmatched") or {}
                lines.append(
                    f"- {persona_slug} matched vs unmatched: "
                    f"matched_pairs={overlap.get('matched_pairs', 0)}, "
                    f"matched_pair_acc={matched.get('pairwise_accuracy') if matched.get('pairwise_accuracy') is not None else 'n/a'}, "
                    f"matched_decisive_acc={matched.get('decisive_pairwise_accuracy') if matched.get('decisive_pairwise_accuracy') is not None else 'n/a'}; "
                    f"unmatched_pairs={overlap.get('unmatched_pairs', 0)}, "
                    f"unmatched_pair_acc={unmatched.get('pairwise_accuracy') if unmatched.get('pairwise_accuracy') is not None else 'n/a'}, "
                    f"unmatched_decisive_acc={unmatched.get('decisive_pairwise_accuracy') if unmatched.get('decisive_pairwise_accuracy') is not None else 'n/a'}"
                )

    persona_evaluation = evaluation.get("persona_evaluation") or {}
    if persona_evaluation:
        lines.extend(["", "## Persona Diagnostics", ""])
        for persona_slug, metrics in persona_evaluation.items():
            lines.append(f"- {persona_slug}: rho={metrics.get('rank_spearman_rho') if metrics.get('rank_spearman_rho') is not None else 'n/a'}; "
                         f"pair_acc={metrics.get('pairwise_accuracy') if metrics.get('pairwise_accuracy') is not None else 'n/a'}; "
                         f"decisive_acc={metrics.get('decisive_pairwise_accuracy') if metrics.get('decisive_pairwise_accuracy') is not None else 'n/a'}; "
                         f"pairs={metrics.get('num_pairs')}")
            aligned = metrics.get("aligned_focus_proxy") or {}
            if aligned:
                if aligned.get("available"):
                    lines.append(
                        f"  aligned_proxy[{aligned['dimension']}]: "
                        f"rho={aligned.get('rank_spearman_rho') if aligned.get('rank_spearman_rho') is not None else 'n/a'}, "
                        f"pair_acc={aligned.get('pairwise_accuracy') if aligned.get('pairwise_accuracy') is not None else 'n/a'}, "
                        f"decisive_acc={aligned.get('decisive_pairwise_accuracy') if aligned.get('decisive_pairwise_accuracy') is not None else 'n/a'}"
                    )
                else:
                    reason = aligned.get("reason")
                    dimension = aligned.get("dimension")
                    if dimension:
                        lines.append(f"  aligned_proxy[{dimension}]: unavailable ({reason})")
                    else:
                        lines.append(f"  aligned_proxy: unavailable ({reason})")

    lines.extend(["", "## Top Ranked Papers", "", "| Rank | Paper ID | Latent Score | Human Mean | Title |", "|------|----------|--------------|------------|-------|"])
    for row in ranking[:20]:
        human_mean = f"{row['human_mean_rating']:.3f}" if row["human_mean_rating"] is not None else ""
        lines.append(
            f"| {row['rank']} | {row['paper_id']} | {row['latent_score']:.4f} | {human_mean} | {row['title']} |"
        )
    lines.append("")
    return "\n".join(lines)


class TogetherPairwiseJudge:
    def __init__(self, config: PairwiseJudgeConfig):
        if not config.api_key:
            raise ValueError("TOGETHER_API_KEY is required for pairwise Together judging.")
        self.config = config
        prompt_root = DEFAULT_PROMPT_ROOT if config.prompt_root is None else Path(config.prompt_root).resolve()
        persona_slugs = tuple(config.persona_slugs or (DEFAULT_PERSONA_SLUG,))
        self.prompt_bundles = [
            build_pairwise_prompt_bundle(
                content_mode=config.content_mode,
                output_schema=config.output_schema,
                prompt_strength=config.prompt_strength,
                primary_area=PRIMARY_AREA,
                persona_slug=persona_slug,
                prompt_root=prompt_root,
            )
            for persona_slug in persona_slugs
        ]
        self.prompt_summary = summarize_prompt_bundles(self.prompt_bundles)
        self._content_cache: dict[str, dict] = {}

    def _content_meta(self, paper: dict) -> dict:
        paper_id = str(paper["paper_id"])
        cached = self._content_cache.get(paper_id)
        if cached is not None:
            return cached
        meta = resolve_paper_content(
            paper=paper,
            content_mode=self.config.content_mode,
            fulltext_dir=None if self.config.fulltext_dir is None else Path(self.config.fulltext_dir),
            max_content_chars=self.config.max_content_chars,
            fulltext_selection=self.config.fulltext_selection,
            section_char_limit=self.config.section_char_limit,
        )
        self._content_cache[paper_id] = meta
        return meta

    def _build_user_message(
        self,
        paper_a: dict,
        paper_b: dict,
        meta_a: dict,
        meta_b: dict,
        prompt_bundle: PairwisePromptBundle,
    ) -> str:
        return render_pairwise_user_prompt(
            prompt_bundle.user_template,
            paper_a=paper_a,
            paper_b=paper_b,
            meta_a=meta_a,
            meta_b=meta_b,
            primary_area=PRIMARY_AREA,
        )

    def _resolved_sampling_params(self) -> tuple[float, float]:
        model_id = self.config.model.model_id.lower()
        if "deepseek-ai/deepseek-r1" in model_id:
            return (0.6 if self.config.temperature == 0.0 else self.config.temperature, 0.95)
        if "moonshotai/kimi-k2.5" in model_id:
            return (1.0 if self.config.temperature == 0.0 else self.config.temperature, 0.95)
        return (self.config.temperature, 1.0)

    def _build_messages(
        self,
        paper_a: dict,
        paper_b: dict,
        meta_a: dict,
        meta_b: dict,
        prompt_bundle: PairwisePromptBundle,
    ) -> list[dict[str, str]]:
        user_content = self._build_user_message(paper_a, paper_b, meta_a, meta_b, prompt_bundle)
        model_id = self.config.model.model_id.lower()
        if "deepseek-ai/deepseek-r1" in model_id:
            # R1 performs better when the full instruction lives in the user message.
            return [
                {
                    "role": "user",
                    "content": f"{prompt_bundle.system_prompt}\n\n{user_content}",
                }
            ]
        return [
            {"role": "system", "content": prompt_bundle.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _call_once(
        self,
        paper_a: dict,
        paper_b: dict,
        meta_a: dict,
        meta_b: dict,
        prompt_bundle: PairwisePromptBundle,
    ) -> dict[str, Any]:
        temperature, top_p = self._resolved_sampling_params()
        payload = {
            "model": self.config.model.model_id,
            "messages": self._build_messages(paper_a, paper_b, meta_a, meta_b, prompt_bundle),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.model.model_id.lower() == "moonshotai/kimi-k2.5":
            payload["reasoning"] = {"enabled": True}
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        data = json.dumps(payload).encode("utf-8")

        last_error = None
        for attempt in range(self.config.max_retries):
            started = time.time()
            req = request.Request(TOGETHER_API_URL, data=data, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                elapsed = time.time() - started
                response_payload = json.loads(body)
                content = response_payload["choices"][0]["message"]["content"]
                parsed = parse_pairwise_response(content)
                parsed["usage"] = response_payload.get("usage", {})
                parsed["elapsed_seconds"] = round(elapsed, 3)
                parsed["http_error"] = None
                return parsed
            except (error.HTTPError, error.URLError, json.JSONDecodeError, KeyError) as exc:
                last_error = str(exc)
                elapsed = time.time() - started
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {
                    "overall_winner": "Tie",
                    "category_winners": {key: "Tie" for key in PAIRWISE_CATEGORY_KEYS},
                    "confidence": 0.0,
                    "rationale_bullets": [],
                    "raw_text": f"[API error] {last_error}",
                    "usage": {},
                    "elapsed_seconds": round(elapsed, 3),
                    "http_error": last_error,
                }

        return {
            "overall_winner": "Tie",
            "category_winners": {key: "Tie" for key in PAIRWISE_CATEGORY_KEYS},
            "confidence": 0.0,
            "rationale_bullets": [],
            "raw_text": f"[API error] {last_error or 'unknown error'}",
            "usage": {},
            "elapsed_seconds": None,
            "http_error": last_error or "unknown error",
        }

    def judge_pair(self, pair: dict, papers_by_id: dict[str, dict]) -> dict[str, Any]:
        paper_a = papers_by_id[pair["paper_a_id"]]
        paper_b = papers_by_id[pair["paper_b_id"]]
        meta_a = self._content_meta(paper_a)
        meta_b = self._content_meta(paper_b)

        calls = []
        for prompt_bundle in self.prompt_bundles:
            forward = self._call_once(paper_a, paper_b, meta_a, meta_b, prompt_bundle)
            forward["persona_slug"] = prompt_bundle.persona.slug
            forward["persona_label"] = prompt_bundle.persona.label
            forward["prompt_name"] = prompt_bundle.prompt_name
            forward["prompt_source"] = prompt_bundle.prompt_source
            forward["system_prompt_sha256"] = prompt_bundle.system_prompt_sha256
            forward["user_prompt_template_sha256"] = prompt_bundle.user_prompt_template_sha256
            forward["prompt_order"] = "AB"
            forward["normalized_overall_winner"] = forward["overall_winner"]
            forward["normalized_category_winners"] = dict(forward["category_winners"])
            calls.append(forward)

            if self.config.swap_order:
                reverse = self._call_once(paper_b, paper_a, meta_b, meta_a, prompt_bundle)
                reverse["persona_slug"] = prompt_bundle.persona.slug
                reverse["persona_label"] = prompt_bundle.persona.label
                reverse["prompt_name"] = prompt_bundle.prompt_name
                reverse["prompt_source"] = prompt_bundle.prompt_source
                reverse["system_prompt_sha256"] = prompt_bundle.system_prompt_sha256
                reverse["user_prompt_template_sha256"] = prompt_bundle.user_prompt_template_sha256
                reverse["prompt_order"] = "BA"
                reverse["normalized_overall_winner"] = invert_winner(reverse["overall_winner"])
                reverse["normalized_category_winners"] = invert_category_winners(reverse["category_winners"])
                calls.append(reverse)

        normalized_calls = []
        total_elapsed = 0.0
        for call in calls:
            if call.get("elapsed_seconds") is not None:
                total_elapsed += float(call["elapsed_seconds"])
            normalized_calls.append(
                {
                    "prompt_order": call["prompt_order"],
                    "persona_slug": call.get("persona_slug"),
                    "persona_label": call.get("persona_label"),
                    "prompt_name": call.get("prompt_name"),
                    "prompt_source": call.get("prompt_source"),
                    "system_prompt_sha256": call.get("system_prompt_sha256"),
                    "user_prompt_template_sha256": call.get("user_prompt_template_sha256"),
                    "overall_winner": call["normalized_overall_winner"],
                    "raw_overall_winner": call["overall_winner"],
                    "category_winners": call["normalized_category_winners"],
                    "raw_category_winners": call["category_winners"],
                    "confidence": call["confidence"],
                    "rationale_bullets": call["rationale_bullets"],
                    "error": call.get("http_error"),
                    "raw_text": call["raw_text"][:2000],
                    "usage": call.get("usage", {}),
                    "elapsed_seconds": call.get("elapsed_seconds"),
                    "http_error": call.get("http_error"),
                }
            )

        aggregate = aggregate_pair_calls(normalized_calls, self.config.winner_threshold)
        if self.config.sleep_seconds > 0:
            time.sleep(self.config.sleep_seconds)

        return {
            "pair_id": pair["pair_id"],
            "paper_a_id": pair["paper_a_id"],
            "paper_b_id": pair["paper_b_id"],
            "model": {
                "id": self.config.model.model_id,
                "label": self.config.model.label,
                "provider": "together",
            },
            "prompt": self.prompt_summary,
            "requested_content_mode": self.config.content_mode,
            "fulltext_selection": self.config.fulltext_selection,
            "content_a": {
                "used_source": meta_a["used_source"],
                "path": meta_a["path"],
                "char_count_used": meta_a["char_count_used"],
                "word_count_used": meta_a["word_count_used"],
                "selected_sections": meta_a["selected_sections"],
            },
            "content_b": {
                "used_source": meta_b["used_source"],
                "path": meta_b["path"],
                "char_count_used": meta_b["char_count_used"],
                "word_count_used": meta_b["word_count_used"],
                "selected_sections": meta_b["selected_sections"],
            },
            "swap_order": self.config.swap_order,
            "calls": normalized_calls,
            "elapsed_seconds": round(total_elapsed, 3) if total_elapsed else None,
            "received_at_utc": now_utc(),
            **aggregate,
        }
