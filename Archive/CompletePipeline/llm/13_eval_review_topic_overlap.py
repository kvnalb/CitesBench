#!/usr/bin/env python3
"""Evaluate topical overlap between human OpenReview comments and generated reviews.

This implements an extract-then-match workflow:
1. Independently extract atomic topics from human reviews and generated reviews.
2. For each topic, ask whether it appears in the other review text.
3. Run the match prompt in two order variants and only count stable verdicts.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import quote


def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_OUTPUT_ROOTS = [
    ROOT / "OutputNew" / "Empirics" / "gemma_ready7_wave1_cached_v2",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave2_incremental",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave3_single_managed",
    ROOT / "OutputNew" / "Coarse",
]
DEFAULT_OPENREVIEW_CACHE = ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview" / "forum_notes"
DEFAULT_KEY_FILE = ROOT / "key.txt"
DEFAULT_REVIEW_DB = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
OPENREVIEW_DETAILS = "replyCount,writable,revisions,original,overwriting,invitation,tags"

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "deepseek-v3": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-ai/deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-r1": ("deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
    "deepseek-ai/deepseek-r1": ("deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
}

TOPIC_CATEGORIES = [
    "experiments_baselines",
    "theory_math",
    "novelty_prior_work",
    "systems_efficiency",
    "clarity_presentation",
    "reproducibility",
    "scope_claims",
    "application_impact",
    "other",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and match human-vs-generated review topics for RDD papers."
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Paper ID to evaluate.")
    parser.add_argument("--sample-jsonl", type=Path, default=None, help="Optional sample JSONL.")
    parser.add_argument("--max-papers", type=int, default=1, help="Maximum papers to evaluate.")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel papers to evaluate.")
    parser.add_argument(
        "--generated-output-roots",
        type=Path,
        nargs="*",
        default=DEFAULT_OUTPUT_ROOTS,
        help="Roots searched for generated coarse_review.json files.",
    )
    parser.add_argument(
        "--openreview-cache-dir",
        type=Path,
        default=DEFAULT_OPENREVIEW_CACHE,
        help="Directory for cached OpenReview forum thread JSON wrappers.",
    )
    parser.add_argument(
        "--fetch-openreview",
        choices=["cache_only", "playwright"],
        default="cache_only",
        help="How to handle missing OpenReview forum thread caches.",
    )
    parser.add_argument(
        "--human-review-source",
        choices=["sqlite", "openreview"],
        default="sqlite",
        help="Source for human reviews. SQLite is the covered RDD source; OpenReview is a fallback/debug path.",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=DEFAULT_REVIEW_DB,
        help="SQLite database containing SUBMISSION and REVIEW tables.",
    )
    parser.add_argument(
        "--generated-text-source",
        choices=["committee", "committee_and_personas"],
        default="committee",
        help="Generated review text used for topic overlap.",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=["per_topic", "document"],
        default="per_topic",
        help="per_topic uses swapped per-topic judgments; document uses one aggregate comparison call.",
    )
    parser.add_argument("--model", default="deepseek-v3.1", help="Together model for extraction/matching.")
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-topics", type=int, default=8, help="Max topics per side per paper.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--extract-max-tokens", type=int, default=1400)
    parser.add_argument("--match-max-tokens", type=int, default=650)
    parser.add_argument("--document-compare-max-tokens", type=int, default=2200)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Only extract topics. Useful for prompt/debug smoke tests.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def load_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def resolve_model(raw: str) -> ModelSpec:
    alias = MODEL_ALIASES.get(raw.lower())
    if alias is None:
        return ModelSpec(raw, raw)
    return ModelSpec(alias[0], alias[1])


def _strip_model_scaffolding(raw_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
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


def together_request(
    *,
    model: ModelSpec,
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
            response_payload = json.loads(body)
            choice = response_payload["choices"][0]
            return {
                "raw_response": choice["message"].get("content") or "",
                "usage": response_payload.get("usage", {}),
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
    raise RuntimeError("unreachable")


def cached_llm_call(
    *,
    cache_path: Path,
    model: ModelSpec,
    api_key: str,
    system_message: str,
    user_message: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    overwrite: bool,
) -> dict[str, Any]:
    if cache_path.exists() and not overwrite:
        return read_json(cache_path)
    result = together_request(
        model=model,
        api_key=api_key,
        system_message=system_message,
        user_message=user_message,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    payload = {
        "model_id": model.model_id,
        "model_label": model.label,
        "system_message": system_message,
        "user_message": user_message,
        "received_at_utc": now_utc(),
        **result,
    }
    write_json(cache_path, payload)
    return payload


def collect_requested_papers(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.sample_jsonl:
        rows.extend(read_jsonl(args.sample_jsonl))
    for paper_id in args.paper_id:
        rows.append({"paper_id": paper_id})

    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id and paper_id not in by_id:
            by_id[paper_id] = row
    selected = list(by_id.values())
    if not selected:
        selected = collect_generated_review_rows(args.generated_output_roots)
    return selected[: max(args.max_papers, 0)]


def collect_generated_review_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/coarse_review.json")):
            try:
                review = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            paper_id = str(review.get("paper_id") or path.parent.name)
            if paper_id in seen:
                continue
            seen.add(paper_id)
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": review.get("title"),
                    "year": review.get("year"),
                    "coarse_review_path": str(path),
                }
            )
    return rows


def find_generated_review_path(paper_id: str, roots: list[Path], row: dict[str, Any]) -> Path | None:
    for key in ("coarse_review_path", "generated_review_path"):
        raw = row.get(key)
        if raw:
            path = Path(str(raw))
            if path.exists():
                return path

    for root in roots:
        if not root.exists():
            continue
        candidates = list(root.glob(f"**/papers/{paper_id}/coarse_review.json"))
        candidates.extend(root.glob(f"**/{paper_id}/coarse_review.json"))
        candidates.extend(root.glob(f"**/{paper_id}_review.json"))
        candidates = [path for path in candidates if path.exists()]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return candidates[0]
    return None


def unwrap_content_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("value", "values", "value-radio", "value-dropdown"):
            if key in value:
                return value[key]
    return value


def normalize_text(value: Any) -> str:
    value = unwrap_content_value(value)
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None).strip()
    return str(value).strip()


def load_openreview_thread(
    *,
    paper_id: str,
    cache_dir: Path,
    fetch_mode: str,
) -> dict[str, Any] | None:
    cache_path = cache_dir / f"{paper_id}.json"
    if cache_path.exists():
        return read_json(cache_path)
    if fetch_mode != "playwright":
        return None
    return fetch_openreview_thread_with_playwright(paper_id, cache_path)


def fetch_openreview_thread_with_playwright(paper_id: str, cache_path: Path) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    quoted = quote(paper_id, safe="")
    url = (
        f"https://api.openreview.net/notes?forum={quoted}"
        f"&trash=true&details={OPENREVIEW_DETAILS}&limit=1000&offset=0"
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"https://openreview.net/forum?id={quoted}", wait_until="domcontentloaded", timeout=60000)
        result = page.evaluate(
            """
            async ({ url }) => {
              const resp = await fetch(url, { credentials: 'include' });
              const text = await resp.text();
              let body = null;
              try { body = JSON.parse(text); } catch (err) { body = null; }
              return { ok: resp.ok, status: resp.status, text, body };
            }
            """,
            {"url": url},
        )
        context.close()
        browser.close()

    wrapper = {
        "paper_id": paper_id,
        "endpoint": "forum_notes",
        "url": url,
        "fetched_at": now_utc(),
        "ok": result.get("ok"),
        "status": result.get("status"),
        "body": result.get("body"),
        "text": result.get("text"),
    }
    write_json(cache_path, wrapper)
    return wrapper


def extract_human_review_text(thread: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    body = thread.get("body") if isinstance(thread, dict) else None
    notes = body.get("notes") if isinstance(body, dict) else None
    if not isinstance(notes, list):
        return "", []

    review_notes: list[dict[str, Any]] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        invitation = str(note.get("invitation") or "")
        content = note.get("content") if isinstance(note.get("content"), dict) else {}
        content_keys = set(content.keys())
        looks_like_review = "Official_Review" in invitation or (
            "review" in content_keys and ("rating" in content_keys or "confidence" in content_keys)
        )
        if not looks_like_review:
            continue
        review_notes.append(note)

    review_notes.sort(key=lambda note: float(note.get("cdate") or note.get("tcdate") or 0.0))
    parts: list[str] = []
    for idx, note in enumerate(review_notes, start=1):
        content = note.get("content") if isinstance(note.get("content"), dict) else {}
        title = normalize_text(content.get("title"))
        rating = normalize_text(content.get("rating"))
        confidence = normalize_text(content.get("confidence"))
        review = normalize_text(content.get("review"))
        summary = normalize_text(content.get("summary"))
        strengths = normalize_text(content.get("strength") or content.get("strengths"))
        weaknesses = normalize_text(content.get("weaknesses") or content.get("weakness"))
        questions = normalize_text(content.get("questions"))
        section_lines = [f"Human reviewer {idx}"]
        if title:
            section_lines.append(f"Title: {title}")
        if rating:
            section_lines.append(f"Rating: {rating}")
        if confidence:
            section_lines.append(f"Confidence: {confidence}")
        if summary:
            section_lines.append(f"Summary:\n{summary}")
        if strengths:
            section_lines.append(f"Strengths:\n{strengths}")
        if weaknesses:
            section_lines.append(f"Weaknesses:\n{weaknesses}")
        if questions:
            section_lines.append(f"Questions:\n{questions}")
        if review and not any([summary, strengths, weaknesses, questions]):
            section_lines.append(f"Review:\n{review}")
        parts.append("\n".join(section_lines))
    return "\n\n---\n\n".join(parts).strip(), review_notes


def extract_human_review_text_from_db(db_path: Path, paper_id: str) -> tuple[str, list[dict[str, Any]]]:
    if not db_path.exists():
        return "", []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                r.paper_id,
                r.reviewer_id,
                s.title,
                s.decision,
                s.when_submitted,
                r.rating,
                r.confidence,
                r.soundness,
                r.presentation,
                r.contribution,
                r.summary,
                r.strength,
                r.weaknesses,
                r.questions,
                r.main_review,
                r.summary_of_the_review
            FROM REVIEW r
            JOIN SUBMISSION s ON r.paper_id = s.id
            WHERE r.paper_id = ?
            ORDER BY r.reviewer_id
            """,
            (paper_id,),
        ).fetchall()
    finally:
        conn.close()

    reviews: list[dict[str, Any]] = [dict(row) for row in rows]
    parts: list[str] = []
    for idx, review in enumerate(reviews, start=1):
        summary = normalize_text(review.get("summary"))
        strengths = normalize_text(review.get("strength"))
        weaknesses = normalize_text(review.get("weaknesses"))
        questions = normalize_text(review.get("questions"))
        legacy = normalize_text(review.get("main_review")) or normalize_text(review.get("summary_of_the_review"))
        section_lines = [f"Human reviewer {idx}"]
        if review.get("rating"):
            section_lines.append(f"Rating: {normalize_text(review.get('rating'))}")
        if review.get("confidence"):
            section_lines.append(f"Confidence: {normalize_text(review.get('confidence'))}")
        if summary:
            section_lines.append(f"Summary:\n{summary}")
        if strengths:
            section_lines.append(f"Strengths:\n{strengths}")
        if weaknesses:
            section_lines.append(f"Weaknesses:\n{weaknesses}")
        if questions:
            section_lines.append(f"Questions:\n{questions}")
        if legacy and not any([summary, strengths, weaknesses, questions]):
            section_lines.append(f"Review:\n{legacy}")
        parts.append("\n".join(section_lines))
    return "\n\n---\n\n".join(parts).strip(), reviews


def extract_generated_review_text(path: Path, source: str) -> tuple[str, dict[str, Any]]:
    review = read_json(path)
    parts = [review_sections_to_text("Generated committee review", review)]
    if source == "committee_and_personas":
        persona_dir = path.parent / "persona_reviews"
        if persona_dir.exists():
            for persona_path in sorted(persona_dir.glob("*.json")):
                persona = read_json(persona_path)
                label = f"Generated persona review: {persona.get('persona_slug') or persona_path.stem}"
                parts.append(review_sections_to_text(label, persona))
    return "\n\n---\n\n".join(part for part in parts if part.strip()).strip(), review


def review_sections_to_text(label: str, payload: dict[str, Any]) -> str:
    fields = [
        ("Summary", payload.get("summary")),
        ("Strengths", payload.get("strength") or payload.get("strengths")),
        ("Weaknesses", payload.get("weaknesses") or payload.get("weakness")),
        ("Questions", payload.get("questions")),
        ("Rationale", payload.get("rationale")),
    ]
    lines = [label]
    for field_label, value in fields:
        text = normalize_text(value)
        if text:
            lines.append(f"{field_label}:\n{text}")
    return "\n".join(lines)


def extraction_prompt(title: str, source_label: str, review_text: str, max_topics: int) -> tuple[str, str]:
    system = f"""You extract atomic peer-review topics for comparison.

Your entire response must be one JSON object. The first character must be {{ and the last character must be }}.
Do not write analysis, explanations, markdown fences, or any text outside JSON.

Return only one valid JSON object with this schema:
{{
  "topics": [
    {{
      "topic_id": "t1",
      "section": "summary|strengths|weaknesses|questions|review|rationale",
      "polarity": "strength|weakness|question|neutral",
      "category": one of {TOPIC_CATEGORIES},
      "description": "one specific concern, question, or strength",
      "quote_anchor": "short exact quote from the review text",
      "severity": 1
    }}
  ]
}}

Rules:
- Extract at most {max_topics} topics.
- Prioritize weaknesses and questions, then important strengths.
- Each topic must be atomic: one issue, one question, or one strength.
- Do not infer from the paper; only extract what appears in the review text.
- The quote_anchor must be copied from the review text and should be under 25 words.
- Use severity 3 for central decision-driving issues, 2 for moderate issues, 1 for minor issues."""
    user = f"""Paper: {title}
Review source: {source_label}

Review text:
{review_text}

Return the JSON object now. Do not explain your choices.
"""
    return system, user


def normalize_topics(raw_payload: dict[str, Any] | None, prefix: str, max_topics: int) -> list[dict[str, Any]]:
    if not raw_payload:
        return []
    topics = raw_payload.get("topics")
    if not isinstance(topics, list):
        return []
    normalized: list[dict[str, Any]] = []
    for idx, topic in enumerate(topics[:max_topics], start=1):
        if not isinstance(topic, dict):
            continue
        description = normalize_text(topic.get("description"))
        if not description:
            continue
        category = normalize_text(topic.get("category")).lower()
        if category not in TOPIC_CATEGORIES:
            category = "other"
        polarity = normalize_text(topic.get("polarity")).lower()
        if polarity not in {"strength", "weakness", "question", "neutral"}:
            polarity = "neutral"
        try:
            severity = int(float(topic.get("severity")))
        except (TypeError, ValueError):
            severity = 1
        normalized.append(
            {
                "topic_id": f"{prefix}{idx}",
                "section": normalize_text(topic.get("section")).lower() or "review",
                "polarity": polarity,
                "category": category,
                "description": description,
                "quote_anchor": normalize_text(topic.get("quote_anchor")),
                "severity": max(1, min(3, severity)),
            }
        )
    return normalized


def match_prompt(
    *,
    title: str,
    source_label: str,
    target_label: str,
    topic: dict[str, Any],
    target_review_text: str,
    variant: str,
) -> tuple[str, str]:
    system = """You judge whether one peer-review topic is addressed in another review.

Return only one valid JSON object:
{
  "verdict": "yes|partial|no",
  "confidence": 0.0,
  "target_quote": "short quote from target review, or empty string",
  "rationale": "brief reason"
}

Verdict rules:
- yes: the target review clearly discusses the same specific topic.
- partial: the target review discusses the same broad area but misses key specificity.
- no: the target review does not discuss the topic.
- Use only the target review text. Do not use outside paper knowledge.
- If verdict is no, target_quote must be empty."""
    topic_json = json.dumps(topic, ensure_ascii=False)
    if variant == "topic_first":
        user = f"""Paper: {title}
Source review: {source_label}
Target review: {target_label}

Source topic:
{topic_json}

Target review text:
{target_review_text}
"""
    else:
        user = f"""Paper: {title}
Target review: {target_label}

Target review text:
{target_review_text}

Source review: {source_label}
Source topic to check:
{topic_json}
"""
    return system, user


def normalize_match(raw_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_payload:
        return {"verdict": "parse_error", "confidence": 0.0, "target_quote": "", "rationale": ""}
    verdict = normalize_text(raw_payload.get("verdict")).lower()
    if verdict in {"y", "true", "match", "matched", "full"}:
        verdict = "yes"
    elif verdict in {"p", "partially", "partial_match"}:
        verdict = "partial"
    elif verdict in {"n", "false", "missing", "none"}:
        verdict = "no"
    if verdict not in {"yes", "partial", "no"}:
        verdict = "parse_error"
    try:
        confidence = float(raw_payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "verdict": verdict,
        "confidence": max(0.0, min(1.0, confidence)),
        "target_quote": normalize_text(raw_payload.get("target_quote")),
        "rationale": normalize_text(raw_payload.get("rationale")),
    }


def stable_verdict(a: str, b: str) -> str:
    if a == b and a in {"yes", "partial", "no"}:
        return a
    return "unstable"


def verdict_score(verdict: str) -> float:
    if verdict == "yes":
        return 1.0
    if verdict == "partial":
        return 0.5
    return 0.0


def document_compare_prompt(
    *,
    title: str,
    human_topics: list[dict[str, Any]],
    generated_topics: list[dict[str, Any]],
) -> tuple[str, str]:
    system = """You compare two sets of extracted peer-review topics.

Return only one valid JSON object:
{
  "human_to_generated": [
    {
      "human_topic_id": "h1",
      "verdict": "yes|partial|no",
      "matched_generated_topic_ids": ["g1"],
      "rationale": "brief reason"
    }
  ],
  "generated_to_human": [
    {
      "generated_topic_id": "g1",
      "verdict": "yes|partial|no",
      "matched_human_topic_ids": ["h1"],
      "rationale": "brief reason"
    }
  ],
  "summary": {
    "main_human_topics_missing_from_generated": ["..."],
    "main_generated_topics_not_in_human": ["..."]
  }
}

Verdict rules:
- yes: the same specific topic appears in the other topic set.
- partial: the same broad area appears, but the specificity differs.
- no: the topic is absent from the other topic set.
- Match by substantive meaning, not exact wording.
- Do not infer beyond the provided topic lists."""
    user = f"""Paper: {title}

Human review topics:
{json.dumps(human_topics, indent=2, ensure_ascii=False)}

Generated review topics:
{json.dumps(generated_topics, indent=2, ensure_ascii=False)}

Compare the two topic sets in both directions and return JSON only.
"""
    return system, user


def normalize_document_comparison(
    payload: dict[str, Any] | None,
    human_topics: list[dict[str, Any]],
    generated_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    if not payload:
        payload = {}

    def norm_verdict(value: Any) -> str:
        verdict = normalize_text(value).lower()
        if verdict in {"yes", "partial", "no"}:
            return verdict
        if verdict in {"match", "matched", "full"}:
            return "yes"
        if verdict in {"partially", "partial_match"}:
            return "partial"
        return "no"

    def normalize_h2g(rows: Any) -> list[dict[str, Any]]:
        raw_rows = rows if isinstance(rows, list) else []
        by_id = {str(row.get("human_topic_id") or row.get("topic_id") or ""): row for row in raw_rows if isinstance(row, dict)}
        normalized = []
        for topic in human_topics:
            raw = by_id.get(topic["topic_id"], {})
            normalized.append(
                {
                    "topic": topic,
                    "verdict": norm_verdict(raw.get("verdict")),
                    "matched_topic_ids": raw.get("matched_generated_topic_ids") if isinstance(raw.get("matched_generated_topic_ids"), list) else [],
                    "rationale": normalize_text(raw.get("rationale")),
                    "score": verdict_score(norm_verdict(raw.get("verdict"))),
                }
            )
        return normalized

    def normalize_g2h(rows: Any) -> list[dict[str, Any]]:
        raw_rows = rows if isinstance(rows, list) else []
        by_id = {str(row.get("generated_topic_id") or row.get("topic_id") or ""): row for row in raw_rows if isinstance(row, dict)}
        normalized = []
        for topic in generated_topics:
            raw = by_id.get(topic["topic_id"], {})
            normalized.append(
                {
                    "topic": topic,
                    "verdict": norm_verdict(raw.get("verdict")),
                    "matched_topic_ids": raw.get("matched_human_topic_ids") if isinstance(raw.get("matched_human_topic_ids"), list) else [],
                    "rationale": normalize_text(raw.get("rationale")),
                    "score": verdict_score(norm_verdict(raw.get("verdict"))),
                }
            )
        return normalized

    return {
        "human_to_generated": normalize_h2g(payload.get("human_to_generated")),
        "generated_to_human": normalize_g2h(payload.get("generated_to_human")),
        "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
    }


def summarize_document_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["verdict"] for row in rows)
    n_topics = len(rows)
    score_sum = sum(float(row["score"]) for row in rows)
    return {
        "n_topics": n_topics,
        "coverage_all_topics": round(score_sum / n_topics, 4) if n_topics else None,
        "verdict_counts": dict(counts),
    }


def evaluate_direction(
    *,
    paper_dir: Path,
    model: ModelSpec,
    api_key: str,
    title: str,
    source_label: str,
    target_label: str,
    source_topics: list[dict[str, Any]],
    target_review_text: str,
    args: argparse.Namespace,
    direction_slug: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topic in source_topics:
        variant_results: dict[str, dict[str, Any]] = {}
        for variant in ("topic_first", "target_first"):
            system, user = match_prompt(
                title=title,
                source_label=source_label,
                target_label=target_label,
                topic=topic,
                target_review_text=target_review_text,
                variant=variant,
            )
            cache_path = paper_dir / "calls" / f"match_{direction_slug}_{topic['topic_id']}_{variant}.json"
            call = cached_llm_call(
                cache_path=cache_path,
                model=model,
                api_key=api_key,
                system_message=system,
                user_message=user,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.match_max_tokens,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                overwrite=args.overwrite,
            )
            parsed = normalize_match(extract_json_object(call.get("raw_response", "")))
            variant_results[variant] = parsed
        stable = stable_verdict(
            variant_results["topic_first"]["verdict"],
            variant_results["target_first"]["verdict"],
        )
        rows.append(
            {
                "topic": topic,
                "topic_first": variant_results["topic_first"],
                "target_first": variant_results["target_first"],
                "stable_verdict": stable,
                "stable_score": verdict_score(stable),
            }
        )
    return rows


def summarize_direction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["stable_verdict"] for row in rows)
    n_topics = len(rows)
    score_sum = sum(float(row["stable_score"]) for row in rows)
    stable_n = sum(counts[key] for key in ("yes", "partial", "no"))
    stable_score_sum = sum(float(row["stable_score"]) for row in rows if row["stable_verdict"] != "unstable")
    return {
        "n_topics": n_topics,
        "stable_judgments": stable_n,
        "swap_stability_rate": round(stable_n / n_topics, 4) if n_topics else None,
        "coverage_all_topics": round(score_sum / n_topics, 4) if n_topics else None,
        "coverage_stable_only": round(stable_score_sum / stable_n, 4) if stable_n else None,
        "verdict_counts": dict(counts),
    }


def run_paper(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    model: ModelSpec,
    api_key: str,
    output_dir: Path,
) -> dict[str, Any]:
    paper_id = str(row.get("paper_id") or "").strip()
    if not paper_id:
        raise ValueError("Missing paper_id")
    generated_path = find_generated_review_path(paper_id, args.generated_output_roots, row)
    if generated_path is None:
        return {"paper_id": paper_id, "status": "missing_generated_review"}

    generated_text, generated_payload = extract_generated_review_text(generated_path, args.generated_text_source)
    title = str(row.get("title") or generated_payload.get("title") or paper_id)
    if args.human_review_source == "sqlite":
        human_text, review_notes = extract_human_review_text_from_db(args.review_db, paper_id)
        missing_status = "missing_sqlite_human_reviews"
    else:
        thread = load_openreview_thread(
            paper_id=paper_id,
            cache_dir=args.openreview_cache_dir,
            fetch_mode=args.fetch_openreview,
        )
        if thread is None:
            return {"paper_id": paper_id, "title": title, "status": "missing_openreview_thread_cache"}
        human_text, review_notes = extract_human_review_text(thread)
        missing_status = "missing_human_official_reviews"
    if not human_text:
        return {"paper_id": paper_id, "title": title, "status": missing_status}

    paper_dir = output_dir / "papers" / paper_id
    write_json(
        paper_dir / "source_texts.json",
        {
            "paper_id": paper_id,
            "title": title,
            "human_review_count": len(review_notes),
            "generated_review_path": str(generated_path),
            "human_review_text": human_text,
            "generated_review_text": generated_text,
        },
    )

    extracted: dict[str, list[dict[str, Any]]] = {}
    for side, source_label, text, prefix in (
        ("human", "human OpenReview official reviews", human_text, "h"),
        ("generated", "generated Gemma/Coarse review", generated_text, "g"),
    ):
        system, user = extraction_prompt(title, source_label, text, args.max_topics)
        call = cached_llm_call(
            cache_path=paper_dir / "calls" / f"extract_{side}.json",
            model=model,
            api_key=api_key,
            system_message=system,
            user_message=user,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.extract_max_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            overwrite=args.overwrite,
        )
        topics = normalize_topics(extract_json_object(call.get("raw_response", "")), prefix, args.max_topics)
        extracted[side] = topics
        write_json(paper_dir / f"{side}_topics.json", {"topics": topics})

    if args.skip_matching:
        result = {
            "paper_id": paper_id,
            "title": title,
            "status": "extracted_only",
            "human_topic_count": len(extracted["human"]),
            "generated_topic_count": len(extracted["generated"]),
            "generated_review_path": str(generated_path),
        }
        write_json(paper_dir / "paper_topic_overlap.json", result)
        return result

    if args.comparison_mode == "document":
        system, user = document_compare_prompt(
            title=title,
            human_topics=extracted["human"],
            generated_topics=extracted["generated"],
        )
        call = cached_llm_call(
            cache_path=paper_dir / "calls" / "document_compare.json",
            model=model,
            api_key=api_key,
            system_message=system,
            user_message=user,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.document_compare_max_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            overwrite=args.overwrite,
        )
        comparison = normalize_document_comparison(
            extract_json_object(call.get("raw_response", "")),
            extracted["human"],
            extracted["generated"],
        )
        write_json(paper_dir / "document_topic_comparison.json", comparison)
        result = {
            "paper_id": paper_id,
            "title": title,
            "status": "ok",
            "comparison_mode": "document",
            "human_review_count": len(review_notes),
            "human_topic_count": len(extracted["human"]),
            "generated_topic_count": len(extracted["generated"]),
            "human_to_generated": summarize_document_rows(comparison["human_to_generated"]),
            "generated_to_human": summarize_document_rows(comparison["generated_to_human"]),
            "generated_review_path": str(generated_path),
        }
        write_json(paper_dir / "paper_topic_overlap.json", result)
        return result

    human_to_generated = evaluate_direction(
        paper_dir=paper_dir,
        model=model,
        api_key=api_key,
        title=title,
        source_label="human reviews",
        target_label="generated review",
        source_topics=extracted["human"],
        target_review_text=generated_text,
        args=args,
        direction_slug="human_to_generated",
    )
    generated_to_human = evaluate_direction(
        paper_dir=paper_dir,
        model=model,
        api_key=api_key,
        title=title,
        source_label="generated review",
        target_label="human reviews",
        source_topics=extracted["generated"],
        target_review_text=human_text,
        args=args,
        direction_slug="generated_to_human",
    )
    write_jsonl(paper_dir / "human_to_generated_matches.jsonl", human_to_generated)
    write_jsonl(paper_dir / "generated_to_human_matches.jsonl", generated_to_human)

    result = {
        "paper_id": paper_id,
        "title": title,
        "status": "ok",
        "human_review_count": len(review_notes),
        "human_topic_count": len(extracted["human"]),
        "generated_topic_count": len(extracted["generated"]),
        "human_to_generated": summarize_direction(human_to_generated),
        "generated_to_human": summarize_direction(generated_to_human),
        "generated_review_path": str(generated_path),
    }
    write_json(paper_dir / "paper_topic_overlap.json", result)
    return result


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row.get("status") == "ok"]

    def avg(path: tuple[str, str]) -> float | None:
        values: list[float] = []
        for row in ok:
            cur: Any = row
            for key in path:
                cur = cur.get(key) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                values.append(float(cur))
        return round(sum(values) / len(values), 4) if values else None

    return {
        "created_at_utc": now_utc(),
        "n_papers": len(results),
        "n_ok": len(ok),
        "status_counts": dict(Counter(str(row.get("status")) for row in results)),
        "mean_human_to_generated_coverage": avg(("human_to_generated", "coverage_all_topics")),
        "mean_generated_to_human_coverage": avg(("generated_to_human", "coverage_all_topics")),
        "mean_human_to_generated_stability": avg(("human_to_generated", "swap_stability_rate")),
        "mean_generated_to_human_stability": avg(("generated_to_human", "swap_stability_rate")),
        "papers": results,
    }


def write_summary_markdown(path: Path, summary: dict[str, Any], args: argparse.Namespace, model: ModelSpec) -> None:
    lines = [
        "# Review Topic Overlap Evaluation",
        "",
        f"- Created: {summary['created_at_utc']}",
        f"- Model: {model.label} (`{model.model_id}`)",
        f"- Papers evaluated: {summary['n_papers']}",
        f"- OK papers: {summary['n_ok']}",
        f"- Human review source: {args.human_review_source}",
        f"- Generated text source: {args.generated_text_source}",
        f"- Comparison mode: {args.comparison_mode}",
        f"- Max topics per side: {args.max_topics}",
        f"- Human to generated coverage: {summary['mean_human_to_generated_coverage']}",
        f"- Generated to human coverage: {summary['mean_generated_to_human_coverage']}",
        f"- Human to generated swap stability: {summary['mean_human_to_generated_stability']}",
        f"- Generated to human swap stability: {summary['mean_generated_to_human_stability']}",
        "",
        "## Per Paper",
        "",
        "| Paper | Status | H->G coverage | G->H coverage | H topics | G topics |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["papers"]:
        h2g = row.get("human_to_generated", {}).get("coverage_all_topics") if isinstance(row.get("human_to_generated"), dict) else ""
        g2h = row.get("generated_to_human", {}).get("coverage_all_topics") if isinstance(row.get("generated_to_human"), dict) else ""
        lines.append(
            f"| {row.get('paper_id')} | {row.get('status')} | {h2g} | {g2h} | "
            f"{row.get('human_topic_count', '')} | {row.get('generated_topic_count', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ROOT / "OutputNew" / "Empirics" / f"review_topic_overlap_{stamp}"
    model = resolve_model(args.model)
    api_key = load_api_key(args.key_file)
    rows = collect_requested_papers(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_config.json",
        {
            "created_at_utc": now_utc(),
            "model_id": model.model_id,
            "model_label": model.label,
            "max_topics": args.max_topics,
            "fetch_openreview": args.fetch_openreview,
            "generated_text_source": args.generated_text_source,
            "comparison_mode": args.comparison_mode,
            "human_review_source": args.human_review_source,
            "review_db": str(args.review_db),
            "generated_output_roots": [str(path) for path in args.generated_output_roots],
            "openreview_cache_dir": str(args.openreview_cache_dir),
        },
    )
    if args.max_workers <= 1:
        results = [
            run_paper(row=row, args=args, model=model, api_key=api_key, output_dir=output_dir)
            for row in rows
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    run_paper,
                    row=row,
                    args=args,
                    model=model,
                    api_key=api_key,
                    output_dir=output_dir,
                )
                for row in rows
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: str(row.get("paper_id") or ""))
    summary = aggregate_results(results)
    write_json(output_dir / "summary.json", summary)
    write_summary_markdown(output_dir / "summary.md", summary, args, model)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
