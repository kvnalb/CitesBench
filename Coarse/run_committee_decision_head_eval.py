#!/usr/bin/env python3
"""
Evaluate a second-stage accept/reject decision head on top of an existing slim
committee run.

Inputs:
  - one completed slim committee run directory produced by run_direct_coarse_benchmark.py
  - local human review JSONs
  - local paper fulltext TXT files

Outputs:
  - per-paper decision packets
  - per-model predictions and raw responses
  - logistic leave-one-out baseline
  - leaderboard / summary metrics

This script is intentionally downstream-only: it does not regenerate committee
reviews. It consumes existing parsed committee outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PAPER_REVIEW_DIR = ROOT / "Code" / "paper_review"
if str(PAPER_REVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_REVIEW_DIR))

from _paper_content import build_core_section_excerpt  # noqa: E402


DEFAULT_COMMITTEE_RUN_DIR = (
    ROOT / "Output" / "Coarse" / "slim_benchmark_sample20_inventory_committee_parallel" / "deepseek_v3_1"
)
DEFAULT_HUMAN_REVIEW_DIR = ROOT / "processed" / "ICLR2025_Foundation_LLMs" / "HumanReview"
DEFAULT_KEY_FILE = ROOT / "key.txt"
DEFAULT_OUTPUT_ROOT = ROOT / "Output" / "Coarse"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
DEFAULT_HEAD_MODELS = "gpt-oss-20b,deepseek-v3.1"

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "gpt-oss": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "openai/gpt-oss-20b": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    "openai/gpt-oss-120b": ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    "deepseek-v3": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-ai/deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
}

HEAD_SYSTEM_PROMPT = """\
You are an ICLR decision head making a forced binary accept/reject decision.

You will receive a compact evidence packet containing:
- the paper abstract
- a core full-text excerpt
- a deterministic structural inventory
- a committee review and persona score table

Instructions:
- Output a forced binary decision: accept or reject.
- Use the committee as evidence, not as authority.
- Do not simply mirror the committee recommendation or mean score.
- Prefer concrete evidence and calibrated skepticism over generic praise.
- Do not output "borderline" or hedge the final decision.
- Keep the reasons short and evidence-based.

Return only one valid JSON object with this schema:
{
  "decision": "accept" | "reject",
  "p_accept": 0.0 to 1.0,
  "margin": -1.0 to 1.0,
  "top_accept_reasons": ["...", "..."],
  "top_reject_reasons": ["...", "..."],
  "evidence_used": ["...", "..."]
}
"""


@dataclass(frozen=True)
class HeadModel:
    model_id: str
    label: str


@dataclass(frozen=True)
class DecisionPacket:
    paper_id: str
    title: str
    keywords: str
    abstract: str
    fulltext_excerpt: str
    selected_sections: list[dict[str, Any]]
    committee_source_model: str
    committee_scores: dict[str, float]
    committee_recommendation: str
    committee_summary: str
    committee_strength: str
    committee_weaknesses: str
    committee_questions: str
    committee_rationale: str
    structural_inventory: dict[str, Any]
    persona_rows: list[dict[str, Any]]
    disagreement: dict[str, float]
    feature_vector: dict[str, float]
    true_label: int
    true_decision: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def resolve_models(raw: str) -> list[HeadModel]:
    models: list[HeadModel] = []
    for token in (piece.strip() for piece in raw.split(",")):
        if not token:
            continue
        alias = MODEL_ALIASES.get(token.lower())
        if alias is not None:
            models.append(HeadModel(alias[0], alias[1]))
        else:
            models.append(HeadModel(token, token))
    return models


def human_decision_bucket(decision: str | None) -> str | None:
    if decision is None:
        return None
    normalized = str(decision).strip().lower()
    if not normalized:
        return None
    if normalized.startswith("accept"):
        return "accept"
    if normalized.startswith("reject"):
        return "reject"
    return None


def recommendation_to_binary(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if "accept" in normalized and "reject" not in normalized:
        return "accept"
    if normalized in {"strong accept", "accept", "borderline accept"}:
        return "accept"
    return "reject"


def _strip_model_scaffolding(raw_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return cleaned


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = _strip_model_scaffolding(raw_text)
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _first_sentence(text: str, limit: int = 220) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "").strip())
    if not collapsed:
        return ""
    match = re.match(r"(.+?[.!?])(?:\s|$)", collapsed)
    sentence = match.group(1).strip() if match else collapsed
    if len(sentence) > limit:
        sentence = sentence[: limit - 3].rstrip() + "..."
    return sentence


def _shorten(text: str, limit: int = 350) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "").strip())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "std": 0.0}
    if len(values) == 1:
        return {
            "mean": values[0],
            "min": values[0],
            "max": values[0],
            "range": 0.0,
            "std": 0.0,
        }
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "std": statistics.pstdev(values),
    }


def _committee_model_slug(run_dir: Path) -> str:
    parsed_root = run_dir / "parsed_reviews" / "slim"
    subdirs = sorted(path for path in parsed_root.iterdir() if path.is_dir())
    if len(subdirs) != 1:
        raise ValueError(
            f"Expected exactly one parsed_reviews/slim/* subdir under {run_dir}, found {len(subdirs)}"
        )
    return subdirs[0].name


def _persona_rows_by_paper(run_dir: Path, model_slug: str) -> dict[str, list[dict[str, Any]]]:
    persona_dir = run_dir / "persona_parsed_reviews" / "slim" / model_slug
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(persona_dir.glob("*__*_review.json")):
        paper_id = path.name.split("__", 1)[0]
        grouped.setdefault(paper_id, []).append(read_json(path))
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row.get("persona_slug") or ""))
    return grouped


def _load_committee_rows(run_dir: Path, model_slug: str) -> dict[str, dict[str, Any]]:
    parsed_dir = run_dir / "parsed_reviews" / "slim" / model_slug
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(parsed_dir.glob("*_review.json")):
        paper_id = path.name.replace("_review.json", "")
        rows[paper_id] = read_json(path)
    return rows


def _load_sample_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    sample_path = run_dir / "sample_papers.jsonl"
    return {str(row["paper_id"]): row for row in read_jsonl(sample_path)}


def _build_fulltext_excerpt(
    *,
    abstract: str,
    fulltext_path: Path | None,
    max_content_chars: int,
    section_char_limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    if fulltext_path is None or not fulltext_path.exists():
        return abstract.strip(), [{"kind": "abstract", "heading": "ABSTRACT", "char_count_used": len(abstract)}]
    full_text = fulltext_path.read_text(encoding="utf-8")
    excerpt = build_core_section_excerpt(
        abstract=abstract,
        full_text=full_text,
        max_content_chars=max_content_chars,
        section_char_limit=section_char_limit,
    )
    if excerpt is None:
        return abstract.strip(), [{"kind": "abstract", "heading": "ABSTRACT", "char_count_used": len(abstract)}]
    return str(excerpt["content"]), list(excerpt["selected_sections"])


def _persona_disagreement(persona_rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for score_name in ("rating", "confidence", "soundness", "presentation", "contribution"):
        values = [_safe_float(row.get(score_name)) for row in persona_rows]
        stats = _score_stats(values)
        for stat_name, stat_value in stats.items():
            result[f"{score_name}_{stat_name}"] = round(float(stat_value), 6)
    accept_votes = sum(1 for row in persona_rows if _safe_float(row.get("rating")) >= 6.0)
    result["persona_accept_votes"] = float(accept_votes)
    result["persona_reject_votes"] = float(max(len(persona_rows) - accept_votes, 0))
    return result


def _feature_vector(
    *,
    sample_row: dict[str, Any],
    committee_row: dict[str, Any],
    persona_rows: list[dict[str, Any]],
    structural_inventory: dict[str, Any],
    selected_sections: list[dict[str, Any]],
    excerpt_text: str,
) -> dict[str, float]:
    disagreement = _persona_disagreement(persona_rows)
    features = {
        "committee_rating": _safe_float(committee_row.get("rating")),
        "committee_confidence": _safe_float(committee_row.get("confidence")),
        "committee_soundness": _safe_float(committee_row.get("soundness")),
        "committee_presentation": _safe_float(committee_row.get("presentation")),
        "committee_contribution": _safe_float(committee_row.get("contribution")),
        "committee_accept_flag": 1.0 if recommendation_to_binary(committee_row.get("recommendation")) == "accept" else 0.0,
        "table_count": _safe_float(structural_inventory.get("table_count")),
        "figure_count": _safe_float(structural_inventory.get("figure_count")),
        "appendix_present": 1.0 if structural_inventory.get("appendix_present") else 0.0,
        "ablation_evidence_count": float(len(structural_inventory.get("ablation_evidence") or [])),
        "task_evidence_count": float(len(structural_inventory.get("task_evidence") or [])),
        "evaluation_evidence_count": float(len(structural_inventory.get("evaluation_evidence") or [])),
        "section_heading_count": float(len(structural_inventory.get("section_headers") or [])),
        "subsection_heading_count": float(len(structural_inventory.get("subsection_headers") or [])),
        "abstract_word_count": _safe_float(sample_row.get("abstract_word_count")),
        "keyword_count": float(len([chunk for chunk in str(sample_row.get("keywords") or "").split(",") if chunk.strip()])),
        "selected_section_count": float(len(selected_sections)),
        "excerpt_chars": float(len(excerpt_text)),
        "summary_chars": float(len(str(committee_row.get("summary") or ""))),
        "strength_chars": float(len(str(committee_row.get("strength") or ""))),
        "weakness_chars": float(len(str(committee_row.get("weaknesses") or ""))),
        "questions_chars": float(len(str(committee_row.get("questions") or ""))),
        "rationale_chars": float(len(str(committee_row.get("rationale") or ""))),
    }
    features.update(disagreement)
    return {key: round(float(value), 6) for key, value in features.items()}


def build_packet(
    *,
    sample_row: dict[str, Any],
    committee_row: dict[str, Any],
    persona_rows: list[dict[str, Any]],
    committee_source_model: str,
    max_content_chars: int,
    section_char_limit: int,
) -> DecisionPacket:
    bucket = human_decision_bucket(sample_row.get("decision"))
    if bucket is None:
        raise ValueError(f"Unsupported human decision bucket for {sample_row.get('paper_id')}: {sample_row.get('decision')}")

    fulltext_path = Path(sample_row["fulltext_path"]) if sample_row.get("fulltext_path") else None
    excerpt_text, selected_sections = _build_fulltext_excerpt(
        abstract=str(sample_row.get("abstract") or ""),
        fulltext_path=fulltext_path,
        max_content_chars=max_content_chars,
        section_char_limit=section_char_limit,
    )
    structural_inventory = (
        committee_row.get("structural_inventory")
        or (committee_row.get("committee") or {}).get("structural_inventory")
        or {}
    )
    feature_vector = _feature_vector(
        sample_row=sample_row,
        committee_row=committee_row,
        persona_rows=persona_rows,
        structural_inventory=structural_inventory,
        selected_sections=selected_sections,
        excerpt_text=excerpt_text,
    )
    disagreement = _persona_disagreement(persona_rows)
    return DecisionPacket(
        paper_id=str(sample_row["paper_id"]),
        title=str(sample_row.get("title") or ""),
        keywords=str(sample_row.get("keywords") or ""),
        abstract=str(sample_row.get("abstract") or ""),
        fulltext_excerpt=excerpt_text,
        selected_sections=selected_sections,
        committee_source_model=committee_source_model,
        committee_scores={
            "rating": _safe_float(committee_row.get("rating")),
            "confidence": _safe_float(committee_row.get("confidence")),
            "soundness": _safe_float(committee_row.get("soundness")),
            "presentation": _safe_float(committee_row.get("presentation")),
            "contribution": _safe_float(committee_row.get("contribution")),
        },
        committee_recommendation=str(committee_row.get("recommendation") or ""),
        committee_summary=str(committee_row.get("summary") or ""),
        committee_strength=str(committee_row.get("strength") or ""),
        committee_weaknesses=str(committee_row.get("weaknesses") or ""),
        committee_questions=str(committee_row.get("questions") or ""),
        committee_rationale=str(committee_row.get("rationale") or ""),
        structural_inventory=structural_inventory,
        persona_rows=persona_rows,
        disagreement=disagreement,
        feature_vector=feature_vector,
        true_label=1 if bucket == "accept" else 0,
        true_decision=bucket,
    )


def render_packet_markdown(packet: DecisionPacket) -> str:
    persona_lines = []
    for row in packet.persona_rows:
        persona_lines.append(
            (
                f"- {row.get('persona_slug')}: rating={row.get('rating')}, confidence={row.get('confidence')}, "
                f"soundness={row.get('soundness')}, presentation={row.get('presentation')}, "
                f"contribution={row.get('contribution')}, recommendation={row.get('recommendation')}\n"
                f"  key concern: {_first_sentence(str(row.get('weaknesses') or row.get('rationale') or ''))}"
            )
        )

    inventory = packet.structural_inventory or {}
    return (
        f"# Decision Packet: {packet.title}\n\n"
        f"- Paper ID: {packet.paper_id}\n"
        f"- True decision: {packet.true_decision}\n"
        f"- Stage-1 committee source model: {packet.committee_source_model}\n"
        f"- Keywords: {packet.keywords or 'n/a'}\n\n"
        f"## Abstract\n\n{packet.abstract}\n\n"
        f"## Core Full-Text Excerpt\n\n{packet.fulltext_excerpt}\n\n"
        f"## Structural Inventory\n\n"
        f"- Sections: {', '.join(inventory.get('section_headers') or []) or 'none'}\n"
        f"- Subsections: {', '.join(inventory.get('subsection_headers') or []) or 'none'}\n"
        f"- Tables: {inventory.get('table_count', 0)}\n"
        f"- Figures: {inventory.get('figure_count', 0)}\n"
        f"- Appendix: {'yes' if inventory.get('appendix_present') else 'no'}\n"
        f"- Ablation evidence: {len(inventory.get('ablation_evidence') or [])}\n"
        f"- Task/setup evidence: {len(inventory.get('task_evidence') or [])}\n"
        f"- Evaluation evidence: {len(inventory.get('evaluation_evidence') or [])}\n\n"
        f"## Committee Scores\n\n"
        f"- Rating: {packet.committee_scores['rating']}\n"
        f"- Confidence: {packet.committee_scores['confidence']}\n"
        f"- Soundness: {packet.committee_scores['soundness']}\n"
        f"- Presentation: {packet.committee_scores['presentation']}\n"
        f"- Contribution: {packet.committee_scores['contribution']}\n"
        f"- Recommendation: {packet.committee_recommendation}\n\n"
        f"## Committee Review\n\n"
        f"Summary: {_shorten(packet.committee_summary, 900)}\n\n"
        f"Strengths: {_shorten(packet.committee_strength, 1200)}\n\n"
        f"Weaknesses: {_shorten(packet.committee_weaknesses, 1400)}\n\n"
        f"Questions: {_shorten(packet.committee_questions, 900)}\n\n"
        f"Rationale: {_shorten(packet.committee_rationale, 900)}\n\n"
        f"## Persona Table\n\n" + "\n".join(persona_lines) + "\n\n"
        f"## Disagreement Signals\n\n"
        f"- Rating range: {packet.disagreement.get('rating_range', 0.0)}\n"
        f"- Contribution range: {packet.disagreement.get('contribution_range', 0.0)}\n"
        f"- Soundness range: {packet.disagreement.get('soundness_range', 0.0)}\n"
        f"- Accept votes: {packet.disagreement.get('persona_accept_votes', 0.0)}\n"
        f"- Reject votes: {packet.disagreement.get('persona_reject_votes', 0.0)}\n"
    )


def build_user_prompt(packet: DecisionPacket) -> str:
    persona_lines = []
    for row in packet.persona_rows:
        persona_lines.append(
            f"- {row.get('persona_slug')}: rating={row.get('rating')}, confidence={row.get('confidence')}, "
            f"soundness={row.get('soundness')}, presentation={row.get('presentation')}, "
            f"contribution={row.get('contribution')}, recommendation={row.get('recommendation')}; "
            f"key concern={_first_sentence(str(row.get('weaknesses') or row.get('rationale') or ''))}"
        )

    inventory = packet.structural_inventory or {}
    return f"""\
Paper title: {packet.title}
Paper ID: {packet.paper_id}
Keywords: {packet.keywords or "n/a"}
Committee source model: {packet.committee_source_model}

Abstract:
{packet.abstract}

Core full-text excerpt:
{packet.fulltext_excerpt}

Deterministic structural inventory:
- Sections: {", ".join(inventory.get("section_headers") or []) or "none"}
- Subsections: {", ".join(inventory.get("subsection_headers") or []) or "none"}
- Tables detected: {inventory.get("table_count", 0)}
- Figures detected: {inventory.get("figure_count", 0)}
- Appendix present: {"yes" if inventory.get("appendix_present") else "no"}
- Ablation evidence count: {len(inventory.get("ablation_evidence") or [])}
- Task/setup evidence count: {len(inventory.get("task_evidence") or [])}
- Evaluation evidence count: {len(inventory.get("evaluation_evidence") or [])}

Committee aggregate scores:
- rating: {packet.committee_scores["rating"]}
- confidence: {packet.committee_scores["confidence"]}
- soundness: {packet.committee_scores["soundness"]}
- presentation: {packet.committee_scores["presentation"]}
- contribution: {packet.committee_scores["contribution"]}
- recommendation: {packet.committee_recommendation}

Committee review summary:
Summary: {_shorten(packet.committee_summary, 1200)}
Strengths: {_shorten(packet.committee_strength, 1400)}
Weaknesses: {_shorten(packet.committee_weaknesses, 1600)}
Questions: {_shorten(packet.committee_questions, 1000)}
Rationale: {_shorten(packet.committee_rationale, 1000)}

Persona score table:
{chr(10).join(persona_lines)}

Disagreement signals:
- rating_range: {packet.disagreement.get("rating_range", 0.0)}
- soundness_range: {packet.disagreement.get("soundness_range", 0.0)}
- contribution_range: {packet.disagreement.get("contribution_range", 0.0)}
- persona_accept_votes: {packet.disagreement.get("persona_accept_votes", 0.0)}
- persona_reject_votes: {packet.disagreement.get("persona_reject_votes", 0.0)}

Make the final program-committee style binary decision.
Return JSON only.
"""


def parse_llm_decision(raw_text: str) -> dict[str, Any]:
    payload = _extract_json_object(raw_text)
    if payload is None:
        cleaned = _strip_model_scaffolding(raw_text)
        decision_match = re.search(r'"?decision"?\s*:\s*"?(accept|reject)"?', cleaned, flags=re.IGNORECASE)
        p_match = re.search(r'"?p_accept"?\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
        margin_match = re.search(r'"?margin"?\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
        payload = {
            "decision": decision_match.group(1).lower() if decision_match else None,
            "p_accept": float(p_match.group(1)) if p_match else None,
            "margin": float(margin_match.group(1)) if margin_match else None,
            "top_accept_reasons": [],
            "top_reject_reasons": [],
            "evidence_used": [],
        }

    decision = str(payload.get("decision") or "").strip().lower()
    p_accept = payload.get("p_accept")
    try:
        p_accept = float(p_accept)
    except (TypeError, ValueError):
        p_accept = None
    if p_accept is not None:
        p_accept = max(0.0, min(1.0, p_accept))
    if decision not in {"accept", "reject"}:
        if p_accept is not None:
            decision = "accept" if p_accept >= 0.5 else "reject"
        else:
            decision = "reject"
    if p_accept is None:
        p_accept = 1.0 if decision == "accept" else 0.0

    margin = payload.get("margin")
    try:
        margin = float(margin)
    except (TypeError, ValueError):
        margin = (2.0 * p_accept) - 1.0
    margin = max(-1.0, min(1.0, margin))

    def normalize_list(value: Any, limit: int) -> list[str]:
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        elif value:
            items = [str(value).strip()]
        else:
            items = []
        return items[:limit]

    return {
        "decision": decision,
        "p_accept": round(p_accept, 6),
        "margin": round(margin, 6),
        "top_accept_reasons": normalize_list(payload.get("top_accept_reasons"), 3),
        "top_reject_reasons": normalize_list(payload.get("top_reject_reasons"), 3),
        "evidence_used": normalize_list(payload.get("evidence_used"), 5),
    }


def together_request(
    *,
    model: HeadModel,
    api_key: str,
    system_message: str,
    user_message: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model.model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
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
        "User-Agent": "LLMReview/1.0",
    }

    last_error: str | None = None
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
            return {
                "raw_response": content,
                "usage": payload.get("usage", {}),
                "finish_reason": choice.get("finish_reason"),
                "elapsed_seconds": round(elapsed, 3),
                "http_error": None,
            }
        except (error.HTTPError, error.URLError, json.JSONDecodeError, KeyError, socket.timeout, TimeoutError) as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
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


def logistic_loocv(packets: list[DecisionPacket]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_names = sorted(packets[0].feature_vector.keys())
    X = np.array([[packet.feature_vector[name] for name in feature_names] for packet in packets], dtype=float)
    y = np.array([packet.true_label for packet in packets], dtype=int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    predictions: list[dict[str, Any]] = []
    for idx, packet in enumerate(packets):
        train_mask = np.ones(len(packets), dtype=bool)
        train_mask[idx] = False
        y_train = y[train_mask]
        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            p_accept = float(unique_train[0])
        else:
            pipeline.fit(X[train_mask], y_train)
            p_accept = float(pipeline.predict_proba(X[[idx]])[0, 1])
        pred = "accept" if p_accept >= 0.5 else "reject"
        predictions.append(
            {
                "paper_id": packet.paper_id,
                "title": packet.title,
                "true_decision": packet.true_decision,
                "true_label": packet.true_label,
                "decision": pred,
                "p_accept": round(p_accept, 6),
                "margin": round((2.0 * p_accept) - 1.0, 6),
                "source": "logistic_loocv",
            }
        )

    feature_effects: list[dict[str, Any]] = []
    if len(np.unique(y)) >= 2:
        pipeline.fit(X, y)
        clf: LogisticRegression = pipeline.named_steps["clf"]
        scaler: StandardScaler = pipeline.named_steps["scaler"]
        standardized_coefs = clf.coef_[0]
        for name, coef, scale in zip(feature_names, standardized_coefs, scaler.scale_):
            feature_effects.append(
                {
                    "feature": name,
                    "standardized_coef": round(float(coef), 6),
                    "scale": round(float(scale), 6),
                }
            )
        feature_effects.sort(key=lambda row: abs(row["standardized_coef"]), reverse=True)
    return predictions, {"feature_names": feature_names, "top_coefficients": feature_effects[:15]}


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = np.array([int(row["true_label"]) for row in rows], dtype=int)
    y_pred = np.array([1 if row["decision"] == "accept" else 0 for row in rows], dtype=int)
    y_prob = np.array([float(row.get("p_accept", y_pred[idx])) for idx, row in enumerate(rows)], dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "n": int(len(rows)),
        "accepted_by_model": int(y_pred.sum()),
        "accepted_by_human": int(y_true.sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_tpr": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tnr": round(float(tn / (tn + fp)), 6) if (tn + fp) else None,
        "decision_agreement_pct": round(float(100.0 * accuracy_score(y_true, y_pred)), 2),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 6),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def leaderboard_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Committee Decision-Head Evaluation",
        "",
        "| Rank | System | N | Acc | Bal Acc | F1 | TPR | TNR | Brier | ROC AUC | Model Accepts | Human Accepts |",
        "|------|--------|---|-----|---------|----|-----|-----|-------|---------|---------------|---------------|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {name} | {n} | {acc} | {bal} | {f1} | {tpr} | {tnr} | {brier} | {auc} | {pred_acc} | {human_acc} |".format(
                rank=idx,
                name=row["system"],
                n=row["n"],
                acc=f"{row['accuracy']:.3f}",
                bal=f"{row['balanced_accuracy']:.3f}",
                f1=f"{row['f1']:.3f}",
                tpr=f"{row['recall_tpr']:.3f}",
                tnr=f"{row['tnr']:.3f}" if row["tnr"] is not None else "n/a",
                brier=f"{row['brier']:.3f}",
                auc=f"{row['roc_auc']:.3f}" if row["roc_auc"] is not None else "n/a",
                pred_acc=row["accepted_by_model"],
                human_acc=row["accepted_by_human"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a second-stage decision head on an existing committee run.")
    parser.add_argument("--committee-run-dir", type=Path, default=DEFAULT_COMMITTEE_RUN_DIR)
    parser.add_argument("--human-review-dir", type=Path, default=DEFAULT_HUMAN_REVIEW_DIR)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--head-models", default=DEFAULT_HEAD_MODELS)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--paper-id", dest="paper_ids", action="append", default=None)
    parser.add_argument("--max-content-chars", type=int, default=9000)
    parser.add_argument("--section-char-limit", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--skip-llm-heads", action="store_true")
    return parser.parse_args()


def prepare_output_dir(explicit_output_dir: Path | None, committee_run_dir: Path) -> Path:
    if explicit_output_dir is not None:
        explicit_output_dir.mkdir(parents=True, exist_ok=True)
        return explicit_output_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = DEFAULT_OUTPUT_ROOT / f"decision_head_eval__{committee_run_dir.name}__{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def main() -> None:
    args = parse_args()
    committee_run_dir = args.committee_run_dir.resolve()
    output_dir = prepare_output_dir(args.output_dir, committee_run_dir)

    head_models = [] if args.skip_llm_heads else resolve_models(args.head_models)
    requested_ids = set(args.paper_ids) if args.paper_ids else None

    run_manifest = read_json(committee_run_dir / "run_manifest.json")
    committee_model_slug = _committee_model_slug(committee_run_dir)
    committee_source_model = str((run_manifest.get("models") or [committee_model_slug])[0])

    sample_rows = _load_sample_rows(committee_run_dir)
    committee_rows = _load_committee_rows(committee_run_dir, committee_model_slug)
    persona_rows = _persona_rows_by_paper(committee_run_dir, committee_model_slug)

    packets: list[DecisionPacket] = []
    human_review_dir = args.human_review_dir.resolve()
    for paper_id in sorted(sample_rows):
        if requested_ids is not None and paper_id not in requested_ids:
            continue
        if args.max_papers is not None and len(packets) >= args.max_papers:
            break
        sample_row = sample_rows[paper_id]
        decision_bucket = human_decision_bucket(sample_row.get("decision"))
        if decision_bucket is None:
            continue
        if paper_id not in committee_rows:
            raise FileNotFoundError(f"Missing committee parsed review for {paper_id}")
        if paper_id not in persona_rows:
            raise FileNotFoundError(f"Missing persona parsed reviews for {paper_id}")
        # Ensure the human review exists; the packet uses the run's sample/decision labels,
        # but we validate local reference coverage here.
        human_path = human_review_dir / f"{paper_id}.json"
        if not human_path.exists():
            raise FileNotFoundError(f"Missing human review JSON for {paper_id}: {human_path}")
        packet = build_packet(
            sample_row=sample_row,
            committee_row=committee_rows[paper_id],
            persona_rows=persona_rows[paper_id],
            committee_source_model=committee_source_model,
            max_content_chars=args.max_content_chars,
            section_char_limit=args.section_char_limit,
        )
        packets.append(packet)

    if not packets:
        raise ValueError("No packets selected.")

    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at_utc": now_utc(),
            "committee_run_dir": str(committee_run_dir),
            "committee_source_model": committee_source_model,
            "committee_model_slug": committee_model_slug,
            "human_review_dir": str(human_review_dir),
            "head_models": [{"id": model.model_id, "label": model.label} for model in head_models],
            "requested_ids": sorted(requested_ids) if requested_ids else None,
            "n_packets": len(packets),
            "max_content_chars": args.max_content_chars,
            "section_char_limit": args.section_char_limit,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
        },
    )

    packet_rows_jsonl: list[dict[str, Any]] = []
    for packet in packets:
        packet_payload = {
            "paper_id": packet.paper_id,
            "title": packet.title,
            "true_decision": packet.true_decision,
            "committee_source_model": packet.committee_source_model,
            "committee_scores": packet.committee_scores,
            "committee_recommendation": packet.committee_recommendation,
            "selected_sections": packet.selected_sections,
            "structural_inventory": packet.structural_inventory,
            "disagreement": packet.disagreement,
            "feature_vector": packet.feature_vector,
        }
        write_json(output_dir / "packets" / f"{packet.paper_id}.json", packet_payload)
        (output_dir / "packets" / f"{packet.paper_id}.md").write_text(
            render_packet_markdown(packet),
            encoding="utf-8",
        )
        packet_rows_jsonl.append(packet_payload)
    write_jsonl(output_dir / "packets.jsonl", packet_rows_jsonl)

    system_rows: list[dict[str, Any]] = []
    prediction_outputs: dict[str, list[dict[str, Any]]] = {}

    baseline_rating_rows: list[dict[str, Any]] = []
    baseline_reco_rows: list[dict[str, Any]] = []
    for packet in packets:
        rating_prob = max(0.0, min(1.0, (packet.committee_scores["rating"] - 1.0) / 9.0))
        rating_decision = "accept" if packet.committee_scores["rating"] >= 6.0 else "reject"
        baseline_rating_rows.append(
            {
                "paper_id": packet.paper_id,
                "title": packet.title,
                "true_decision": packet.true_decision,
                "true_label": packet.true_label,
                "decision": rating_decision,
                "p_accept": round(rating_prob, 6),
                "margin": round((2.0 * rating_prob) - 1.0, 6),
                "source": "committee_rating_threshold",
            }
        )
        reco_decision = recommendation_to_binary(packet.committee_recommendation)
        reco_prob = 1.0 if reco_decision == "accept" else 0.0
        baseline_reco_rows.append(
            {
                "paper_id": packet.paper_id,
                "title": packet.title,
                "true_decision": packet.true_decision,
                "true_label": packet.true_label,
                "decision": reco_decision,
                "p_accept": reco_prob,
                "margin": 1.0 if reco_decision == "accept" else -1.0,
                "source": "committee_recommendation",
            }
        )

    prediction_outputs["committee_rating_threshold"] = baseline_rating_rows
    prediction_outputs["committee_recommendation"] = baseline_reco_rows

    for system_name, rows in prediction_outputs.items():
        metrics = compute_metrics(rows)
        metrics["system"] = system_name
        system_rows.append(metrics)
        write_jsonl(output_dir / "predictions" / f"{slugify(system_name)}.jsonl", rows)

    logistic_rows, logistic_meta = logistic_loocv(packets)
    prediction_outputs["logistic_loocv"] = logistic_rows
    write_jsonl(output_dir / "predictions" / "logistic_loocv.jsonl", logistic_rows)
    write_json(output_dir / "predictions" / "logistic_loocv_meta.json", logistic_meta)
    logistic_metrics = compute_metrics(logistic_rows)
    logistic_metrics["system"] = "logistic_loocv"
    system_rows.append(logistic_metrics)

    if head_models:
        api_key = load_api_key(args.key_file.resolve())
        for model in head_models:
            rows: list[dict[str, Any]] = []
            for idx, packet in enumerate(packets, start=1):
                user_message = build_user_prompt(packet)
                api_result = together_request(
                    model=model,
                    api_key=api_key,
                    system_message=HEAD_SYSTEM_PROMPT,
                    user_message=user_message,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                )
                parsed = parse_llm_decision(api_result["raw_response"])
                rows.append(
                    {
                        "paper_id": packet.paper_id,
                        "title": packet.title,
                        "true_decision": packet.true_decision,
                        "true_label": packet.true_label,
                        "decision": parsed["decision"],
                        "p_accept": parsed["p_accept"],
                        "margin": parsed["margin"],
                        "top_accept_reasons": parsed["top_accept_reasons"],
                        "top_reject_reasons": parsed["top_reject_reasons"],
                        "evidence_used": parsed["evidence_used"],
                        "raw_response": api_result["raw_response"],
                        "usage": api_result["usage"],
                        "finish_reason": api_result["finish_reason"],
                        "elapsed_seconds": api_result["elapsed_seconds"],
                        "http_error": api_result["http_error"],
                        "source": model.model_id,
                    }
                )
                req_seconds = api_result["elapsed_seconds"]
                req_display = f"{req_seconds:.1f}s" if req_seconds is not None else "n/a"
                finish_reason = api_result["finish_reason"] or "n/a"
                print(
                    f"[{idx}/{len(packets)}] {model.label} {packet.paper_id} "
                    f"decision={parsed['decision']} p={parsed['p_accept']:.3f} "
                    f"finish={finish_reason} req={req_display}"
                )
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            prediction_outputs[model.label] = rows
            write_jsonl(output_dir / "predictions" / f"{slugify(model.label)}.jsonl", rows)
            metrics = compute_metrics(rows)
            metrics["system"] = model.label
            system_rows.append(metrics)

    system_rows.sort(
        key=lambda row: (
            -(row["balanced_accuracy"] if row["balanced_accuracy"] is not None else float("-inf")),
            -(row["f1"] if row["f1"] is not None else float("-inf")),
            row["system"],
        )
    )
    write_json(output_dir / "summary.json", system_rows)
    write_csv(output_dir / "summary.csv", system_rows)
    (output_dir / "leaderboard.md").write_text(leaderboard_markdown(system_rows), encoding="utf-8")

    print(f"\nWrote outputs to: {output_dir}")
    print((output_dir / "leaderboard.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
