#!/usr/bin/env python3
"""
Shared utilities for the abstract-review generation and comparison scripts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from _review_prompt_library import DEFAULT_PERSONA_SLUG, build_review_prompt_bundle, render_review_user_prompt


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = ROOT / "rawdata" / "ICLR2025" / "foundation_or_frontier_models_including_LLMs" / "abstracts.jsonl"
DEFAULT_HUMAN_REVIEW_DIR = ROOT / "processed" / "ICLR2025_Foundation_LLMs" / "HumanReview"
DEFAULT_OUTPUT_ROOT = ROOT / "LLMOutput"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
PRIMARY_AREA = "foundation/frontier models including LLMs"

MODEL_ALIASES = {
    "gpt-oss": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "openai/gpt-oss-20b": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    "openai/gpt-oss-120b": ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    "deepseekv3": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-v3": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-ai/deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-r1": ("deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
    "deepseek-ai/deepseek-r1": ("deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
    "kimi-k2.5": ("moonshotai/Kimi-K2.5", "Kimi-K2.5"),
    "moonshotai/kimi-k2.5": ("moonshotai/Kimi-K2.5", "Kimi-K2.5"),
    "qwen3-235b-reasoning": ("Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "Qwen3-235B-Instruct-2507"),
    "qwen/qwen3-235b-a22b-instruct-2507-tput": ("Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "Qwen3-235B-Instruct-2507"),
    "qwen3-235b-thinking": ("Qwen/Qwen3-235B-A22B-Thinking-2507", "Qwen3-235B-Thinking-2507"),
    "qwen/qwen3-235b-a22b-thinking-2507": ("Qwen/Qwen3-235B-A22B-Thinking-2507", "Qwen3-235B-Thinking-2507"),
}

SCORE_SPECS = {
    "rating": {"min": 1.0, "max": 10.0, "accept_threshold": 6.0},
    "confidence": {"min": 1.0, "max": 5.0},
    "soundness": {"min": 1.0, "max": 4.0},
    "presentation": {"min": 1.0, "max": 4.0},
    "contribution": {"min": 1.0, "max": 4.0},
}
SCORE_KEYS = list(SCORE_SPECS)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_human_reviews(review_dir: Path) -> dict[str, dict]:
    reviews = {}
    for path in sorted(review_dir.glob("*.json")):
        reviews[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return reviews


def sample_papers(papers: list[dict], max_papers: int | None, seed: int) -> list[dict]:
    if max_papers is None or max_papers >= len(papers):
        return sorted(papers, key=lambda row: row["paper_id"])
    import random

    rng = random.Random(seed)
    sampled = list(papers)
    rng.shuffle(sampled)
    sampled = sampled[:max_papers]
    return sorted(sampled, key=lambda row: row["paper_id"])


def resolve_models(models_arg: str) -> list[ModelSpec]:
    models = []
    for raw_model in models_arg.split(","):
        token = raw_model.strip()
        if not token:
            continue
        alias = MODEL_ALIASES.get(token.lower())
        if alias is not None:
            models.append(ModelSpec(alias[0], alias[1]))
        else:
            models.append(ModelSpec(token, token))
    if not models:
        raise ValueError("At least one model must be specified.")
    return models


def system_prompt() -> str:
    bundle = build_review_prompt_bundle("abstract", DEFAULT_PERSONA_SLUG, PRIMARY_AREA)
    return bundle.system_prompt


def user_prompt(paper: dict) -> str:
    bundle = build_review_prompt_bundle("abstract", DEFAULT_PERSONA_SLUG, PRIMARY_AREA)
    return render_review_user_prompt(
        bundle.user_template,
        paper,
        {
            "evidence_description": "title, abstract, and author-provided keywords only",
            "content_label": "Abstract",
            "content": paper.get("abstract", "") or "",
        },
        PRIMARY_AREA,
    )


def strip_model_scaffolding(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        content = fence_match.group(1).strip()
    return content.strip()


def clamp_score(value: float, key: str) -> float:
    spec = SCORE_SPECS[key]
    return max(spec["min"], min(spec["max"], value))


def _maybe_json_object(content: str) -> dict | None:
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(content[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            return None
    return None


def parse_model_response(raw_content: str) -> dict[str, Any]:
    cleaned = strip_model_scaffolding(raw_content or "")
    payload = _maybe_json_object(cleaned)
    parser = "json"
    rationale = ""

    if payload is None:
        parser = "regex"
        payload = {}
        for key in SCORE_KEYS:
            match = re.search(rf'"?{re.escape(key)}"?\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
            if match:
                payload[key] = float(match.group(1))
        rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"]+)"', cleaned)
        if rationale_match:
            rationale = rationale_match.group(1).strip()
        else:
            rationale = cleaned[:1000].strip()
    else:
        rationale = str(payload.get("rationale", "")).strip()

    parsed_scores: dict[str, float | None] = {}
    parsed_ok = True
    for key in SCORE_KEYS:
        value = payload.get(key)
        if value is None:
            parsed_scores[key] = None
            parsed_ok = False
            continue
        try:
            parsed_scores[key] = round(clamp_score(float(value), key), 3)
        except (TypeError, ValueError):
            parsed_scores[key] = None
            parsed_ok = False

    return {
        "scores": parsed_scores,
        "rationale": rationale,
        "parsed_ok": parsed_ok,
        "parser": parser,
        "cleaned_content": cleaned,
    }


def model_decision_from_rating(rating: float | None) -> str | None:
    if rating is None:
        return None
    return "Accept" if rating >= SCORE_SPECS["rating"]["accept_threshold"] else "Reject"


def human_decision_bucket(decision: str | None) -> str | None:
    if decision is None:
        return None
    norm = str(decision).strip().lower()
    if not norm or norm == "withdrawn":
        return None
    return "Accept" if norm.startswith("accept") else "Reject"


def build_comparison(paper: dict, llm_scores: dict[str, float | None]) -> dict[str, Any]:
    human_review = paper.get("human_review")
    if not human_review:
        return {
            "available": False,
            "nmae": None,
            "decision_agree": None,
            "score_deltas": {},
            "absolute_errors": {},
            "normalized_errors": {},
        }

    aggregated = human_review.get("aggregated", {})
    score_deltas: dict[str, float | None] = {}
    abs_errors: dict[str, float | None] = {}
    norm_errors: dict[str, float | None] = {}
    comparable_norm_errors: list[float] = []

    for key in SCORE_KEYS:
        model_value = llm_scores.get(key)
        human_stats = aggregated.get(key)
        human_mean = None
        if isinstance(human_stats, dict):
            human_mean = human_stats.get("mean")

        if model_value is None or human_mean is None:
            score_deltas[key] = None
            abs_errors[key] = None
            norm_errors[key] = None
            continue

        delta = float(model_value) - float(human_mean)
        abs_err = abs(delta)
        scale_range = SCORE_SPECS[key]["max"] - SCORE_SPECS[key]["min"]
        norm_err = abs_err / scale_range if scale_range > 0 else None

        score_deltas[key] = round(delta, 4)
        abs_errors[key] = round(abs_err, 4)
        norm_errors[key] = round(norm_err, 4) if norm_err is not None else None
        if norm_err is not None:
            comparable_norm_errors.append(norm_err)

    human_bucket = human_decision_bucket(paper.get("decision"))
    model_bucket = model_decision_from_rating(llm_scores.get("rating"))
    return {
        "available": True,
        "nmae": round(statistics.mean(comparable_norm_errors), 4) if comparable_norm_errors else None,
        "decision_agree": (human_bucket == model_bucket) if human_bucket is not None and model_bucket is not None else None,
        "human_decision_bucket": human_bucket,
        "model_decision_bucket": model_bucket,
        "score_deltas": score_deltas,
        "absolute_errors": abs_errors,
        "normalized_errors": norm_errors,
    }


def average_ranks(pairs: list[tuple[str, float]]) -> dict[str, float]:
    sorted_pairs = sorted(pairs, key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    idx = 0
    while idx < len(sorted_pairs):
        j = idx + 1
        while j < len(sorted_pairs) and sorted_pairs[j][1] == sorted_pairs[idx][1]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            ranks[sorted_pairs[k][0]] = avg_rank
        idx = j
    return ranks


def spearman_rho(predicted: list[tuple[str, float]], human: list[tuple[str, float]]) -> float | None:
    pred_map = {paper_id: value for paper_id, value in predicted}
    human_map = {paper_id: value for paper_id, value in human}
    common_ids = sorted(set(pred_map) & set(human_map))
    if len(common_ids) < 2:
        return None

    pred_ranks = average_ranks([(paper_id, pred_map[paper_id]) for paper_id in common_ids])
    human_ranks = average_ranks([(paper_id, human_map[paper_id]) for paper_id in common_ids])

    pred_values = [pred_ranks[paper_id] for paper_id in common_ids]
    human_values = [human_ranks[paper_id] for paper_id in common_ids]
    mean_pred = statistics.mean(pred_values)
    mean_human = statistics.mean(human_values)

    cov = sum((p - mean_pred) * (h - mean_human) for p, h in zip(pred_values, human_values))
    var_pred = sum((p - mean_pred) ** 2 for p in pred_values)
    var_human = sum((h - mean_human) ** 2 for h in human_values)
    if var_pred == 0 or var_human == 0:
        return None
    return cov / math.sqrt(var_pred * var_human)


def summarise_model(results: list[dict], model: ModelSpec, output_file: Path) -> dict[str, Any]:
    parsed_count = sum(1 for row in results if row["llm_review"]["parsed_ok"])
    nmae_values = [row["comparison"]["nmae"] for row in results if row["comparison"]["nmae"] is not None]
    decision_values = [row["comparison"]["decision_agree"] for row in results if row["comparison"]["decision_agree"] is not None]

    dim_mae: dict[str, float | None] = {}
    dim_nmae: dict[str, float | None] = {}
    mean_model_scores: dict[str, float | None] = {}
    mean_human_scores: dict[str, float | None] = {}

    predicted_ratings = []
    human_ratings = []
    accepted_by_model = 0
    accepted_by_human = 0

    for key in SCORE_KEYS:
        dim_abs = [
            row["comparison"]["absolute_errors"][key]
            for row in results
            if row["comparison"]["absolute_errors"].get(key) is not None
        ]
        dim_norm = [
            row["comparison"]["normalized_errors"][key]
            for row in results
            if row["comparison"]["normalized_errors"].get(key) is not None
        ]
        model_values = [
            row["llm_review"]["scores"][key]
            for row in results
            if row["llm_review"]["scores"].get(key) is not None
        ]
        human_values = []
        for row in results:
            aggregated = row["human_review"].get("aggregated") if row["human_review"].get("available") else None
            if not aggregated:
                continue
            stats = aggregated.get(key)
            if isinstance(stats, dict) and stats.get("mean") is not None:
                human_values.append(stats["mean"])

        dim_mae[key] = round(statistics.mean(dim_abs), 4) if dim_abs else None
        dim_nmae[key] = round(statistics.mean(dim_norm), 4) if dim_norm else None
        mean_model_scores[key] = round(statistics.mean(model_values), 4) if model_values else None
        mean_human_scores[key] = round(statistics.mean(human_values), 4) if human_values else None

    for row in results:
        rating = row["llm_review"]["scores"].get("rating")
        if rating is not None:
            predicted_ratings.append((row["paper_id"], rating))
            if model_decision_from_rating(rating) == "Accept":
                accepted_by_model += 1

        human_review = row["human_review"]
        if human_review.get("available"):
            aggregated = human_review.get("aggregated", {})
            human_rating = aggregated.get("rating", {}).get("mean") if aggregated.get("rating") else None
            if human_rating is not None:
                human_ratings.append((row["paper_id"], float(human_rating)))
            if human_decision_bucket(row.get("decision")) == "Accept":
                accepted_by_human += 1

    rho = spearman_rho(predicted_ratings, human_ratings) if predicted_ratings and human_ratings else None
    return {
        "model_id": model.model_id,
        "model_label": model.label,
        "results_file": str(output_file),
        "papers_total": len(results),
        "parsed_count": parsed_count,
        "parsed_rate": round(parsed_count / len(results), 4) if results else None,
        "nmae_mean": round(statistics.mean(nmae_values), 4) if nmae_values else None,
        "decision_agreement_pct": round(100 * statistics.mean(1.0 if value else 0.0 for value in decision_values), 2) if decision_values else None,
        "decision_agreement_n": len(decision_values),
        "rating_spearman_rho": round(rho, 4) if rho is not None else None,
        "accepted_by_model": accepted_by_model,
        "accepted_by_human": accepted_by_human,
        "dim_mae": dim_mae,
        "dim_nmae": dim_nmae,
        "mean_model_scores": mean_model_scores,
        "mean_human_scores": mean_human_scores,
    }


def leaderboard_markdown(summaries: list[dict]) -> str:
    lines = [
        "# LLM Baseline Run",
        "",
        "| Model | Parsed | NMAE | Rating Spearman | Decision Agree | Model Accepts | Human Accepts |",
        "|-------|--------|------|-----------------|----------------|---------------|---------------|",
    ]
    for summary in summaries:
        parsed = f"{summary['parsed_count']}/{summary['papers_total']}"
        nmae = f"{summary['nmae_mean']:.4f}" if summary["nmae_mean"] is not None else "n/a"
        rho = f"{summary['rating_spearman_rho']:.4f}" if summary["rating_spearman_rho"] is not None else "n/a"
        agree = f"{summary['decision_agreement_pct']:.2f}%" if summary["decision_agreement_pct"] is not None else "n/a"
        lines.append(
            f"| {summary['model_label']} | {parsed} | {nmae} | {rho} | {agree} | "
            f"{summary['accepted_by_model']} | {summary['accepted_by_human']} |"
        )
    lines.append("")
    return "\n".join(lines)


def together_request(
    model: ModelSpec,
    paper: dict,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    system_message: str | None = None,
    user_message: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model.model_id,
        "messages": [
            {"role": "system", "content": system_message or system_prompt()},
            {"role": "user", "content": user_message or user_prompt(paper)},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    last_error = None
    for attempt in range(max_retries):
        started = time.time()
        req = request.Request(TOGETHER_API_URL, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
            elapsed = time.time() - started
            payload = json.loads(body)
            choice = payload["choices"][0]
            content = choice["message"].get("content") or ""
            usage = payload.get("usage", {})
            return {
                "raw_response": content,
                "usage": usage,
                "finish_reason": choice.get("finish_reason"),
                "elapsed_seconds": round(elapsed, 3),
                "http_error": None,
            }
        except (error.HTTPError, error.URLError, json.JSONDecodeError, KeyError, socket.timeout, TimeoutError) as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "raw_response": f"[API error] {last_error}",
                "usage": {},
                "finish_reason": None,
                "elapsed_seconds": round(time.time() - started, 3),
                "http_error": last_error,
            }

    return {
        "raw_response": f"[API error] {last_error or 'unknown error'}",
        "usage": {},
        "finish_reason": None,
        "elapsed_seconds": None,
        "http_error": last_error or "unknown error",
    }


def prepare_output_dir(output_root: Path, explicit_output_dir: Path | None, suffix: str) -> Path:
    if explicit_output_dir is not None:
        explicit_output_dir.mkdir(parents=True, exist_ok=True)
        return explicit_output_dir.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + suffix
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()
