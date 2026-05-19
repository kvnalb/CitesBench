#!/usr/bin/env python3
"""
Run the slim Coarse committee + DeepSeek decision-head pipeline on the
year-specific-bandwidth ICLR RDD sample.

Workflow:
1. Select 2018-2020 papers from the year-specific-bandwidth RDD sample.
2. Merge in OpenReview metadata (abstract, keywords, PDF URL).
3. Cache PDFs and extracted plain text under rawdata/Design/OpenReview.
4. Run the slim committee review pipeline on each paper's local text file.
5. Build a decision packet and run the DeepSeek decision head.
6. Persist per-paper artifacts plus run-level manifests and summaries.

The script is resumable. If a paper already has saved stage-1 / stage-2 outputs,
the corresponding step is skipped unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib import error, request

import fitz
import litellm

from coarse.config import CoarseConfig


def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
COARSE_DIR = Path(__file__).resolve().parent

if str(COARSE_DIR) not in sys.path:
    sys.path.insert(0, str(COARSE_DIR))

from run_committee_decision_head_eval import (  # noqa: E402
    HEAD_SYSTEM_PROMPT,
    HeadModel,
    build_packet,
    build_user_prompt,
    parse_llm_decision,
    render_packet_markdown,
    resolve_models,
    together_request,
)
from slim_coarse_pipeline import COMMITTEE_BIAS_PROMPTS, DEFAULT_PERSONA_ENSEMBLE, review_paper_slim  # noqa: E402


DEFAULT_RDD_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
DEFAULT_OPENREVIEW_CSV = ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview" / "openreview_yearly_submissions.csv"
DEFAULT_KEY_FILE = ROOT / "key.txt"

DEFAULT_COMMITTEE_MODEL = "together_ai/google/gemma-4-31B-it"
DEFAULT_DECISION_HEAD_MODELS = "deepseek-v3.1"
DEFAULT_YEARS = "2018,2019,2020"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_").lower()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def build_local_config(together_key: str) -> CoarseConfig:
    return CoarseConfig(
        extraction_qa=False,
        api_keys={
            "together": together_key,
            "together_ai": together_key,
        },
    )


def parse_years(raw: str) -> list[int]:
    years = sorted({int(token.strip()) for token in raw.split(",") if token.strip()})
    if not years:
        raise ValueError("At least one year must be provided.")
    return years


def parse_personas(raw: str) -> list[str]:
    personas = [token.strip() for token in raw.split(",") if token.strip()]
    if not personas:
        raise ValueError("At least one persona must be provided.")
    if len(personas) == 1 and personas[0].lower() in {"default-ensemble", "committee4"}:
        return list(DEFAULT_PERSONA_ENSEMBLE)
    return personas


def parse_weights(raw: str | None, personas: list[str]) -> dict[str, float] | None:
    if raw is None or not raw.strip():
        return None
    weights: dict[str, float] = {}
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid weight token '{token}'. Expected persona=weight.")
        persona, raw_value = token.split("=", 1)
        value = float(raw_value.strip())
        if value <= 0:
            raise ValueError("Weights must be positive.")
        weights[persona.strip()] = value
    missing = [persona for persona in personas if persona not in weights]
    if missing:
        raise ValueError(f"Missing weights for personas: {', '.join(missing)}")
    return weights


def infer_run_slug(
    years: list[int],
    committee_model: str,
    head_model: HeadModel,
    committee_bias: str,
) -> str:
    years_slug = f"{min(years)}_{max(years)}" if len(years) > 1 else str(years[0])
    bias_slug = "" if committee_bias == "plain" else f"__committee_{slugify(committee_bias)}"
    return (
        f"rdd_bandwidth_{years_slug}"
        f"__{slugify(committee_model)}"
        f"__{slugify(head_model.label)}"
        f"{bias_slug}"
    )


def clean_csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")


def parse_keywords(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return ",".join(str(item).strip() for item in payload if str(item).strip())
    except Exception:
        pass
    return text


def bool_from_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def abstract_word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def parse_retry_after_seconds(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max((parsed - datetime.now(timezone.utc)).total_seconds(), 0.0)


def load_rdd_sample(
    *,
    path: Path,
    years: set[int],
    max_papers: int | None,
    requested_ids: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            paper_id = str(row["paper_id"])
            year = int(row["year"])
            if year not in years:
                continue
            if requested_ids is not None and paper_id not in requested_ids:
                continue
            if "in_year_specific_rdd_sample" in row and not bool_from_text(row["in_year_specific_rdd_sample"]):
                continue
            rows.append(row)
    rows.sort(key=lambda row: (int(row["year"]), float(row["score_centered"]), str(row["paper_id"])))
    if max_papers is not None:
        rows = rows[:max_papers]
    return rows


def metadata_score(row: dict[str, Any], target_year: int) -> tuple[int, int, int]:
    return (
        1 if str(row.get("query_year") or "").strip() == str(target_year) else 0,
        1 if str(row.get("openreview_pdf_url") or row.get("openreview_reviewed_pdf_url") or "").strip() else 0,
        1 if str(row.get("openreview_abstract") or "").strip() else 0,
    )


def load_openreview_metadata(path: Path, target_years: dict[str, int]) -> dict[str, dict[str, Any]]:
    best_rows: dict[str, dict[str, Any]] = {}
    best_scores: dict[str, tuple[int, int, int]] = {}
    cleaned = clean_csv_text(path)
    reader = csv.DictReader(io.StringIO(cleaned))
    for row in reader:
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in target_years:
            continue
        score = metadata_score(row, target_years[paper_id])
        if paper_id not in best_rows or score > best_scores[paper_id]:
            best_rows[paper_id] = row
            best_scores[paper_id] = score
    return best_rows


def build_selected_papers(
    *,
    rdd_rows: list[dict[str, Any]],
    metadata_rows: dict[str, dict[str, Any]],
    fulltext_dir: Path,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rdd_rows:
        paper_id = str(row["paper_id"])
        metadata = metadata_rows.get(paper_id, {})
        title = str(metadata.get("openreview_title") or row.get("title") or "").strip()
        abstract = str(metadata.get("openreview_abstract") or "").strip()
        keywords = parse_keywords(str(metadata.get("openreview_keywords_json") or ""))
        pdf_url = str(
            metadata.get("openreview_pdf_url")
            or metadata.get("openreview_reviewed_pdf_url")
            or f"https://openreview.net/pdf?id={paper_id}"
        ).strip()
        forum_url = str(metadata.get("openreview_forum_url") or f"https://openreview.net/forum?id={paper_id}")
        fulltext_path = fulltext_dir / f"{paper_id}.txt"
        selected.append(
            {
                "paper_id": paper_id,
                "title": title,
                "year": int(row["year"]),
                "decision": str(row.get("decision") or ""),
                "accepted": float(row.get("accepted") or 0.0),
                "primary_area": str(row.get("primary_area") or ""),
                "mean_rating": float(row.get("mean_rating") or 0.0),
                "score_centered": float(row.get("score_centered") or 0.0),
                "cutoff": float(row.get("cutoff") or 0.0),
                "bandwidth": float(row.get("bandwidth") or 0.0),
                "keywords": keywords,
                "abstract": abstract,
                "abstract_char_count": len(abstract),
                "abstract_word_count": abstract_word_count(abstract),
                "pdf_url": pdf_url,
                "forum_url": forum_url,
                "openreview_venue": str(metadata.get("openreview_venue") or ""),
                "openreview_invitation": str(metadata.get("openreview_invitation") or ""),
                "fulltext_available": fulltext_path.exists(),
                "fulltext_path": str(fulltext_path),
            }
        )
    return selected


def download_pdf(
    *,
    pdf_url: str,
    paper_id: str,
    dest_path: Path,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_base_seconds: float = 2.0,
    retry_after_floor_seconds: float = 30.0,
    retry_backoff_cap_seconds: float = 300.0,
) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://openreview.net/forum?id={paper_id}",
        "Accept": "application/pdf,*/*",
    }
    last_error: str | None = None
    last_status_code: int | None = None
    for attempt in range(max_retries):
        started = time.time()
        req = request.Request(pdf_url, headers=headers, method="GET")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = response.read()
                status = getattr(response, "status", 200)
            if not body.startswith(b"%PDF-"):
                raise ValueError(f"Downloaded content is not a PDF for {paper_id}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(body)
            return {
                "status_code": int(status),
                "elapsed_seconds": round(time.time() - started, 3),
                "error": None,
                "bytes": len(body),
            }
        except (error.HTTPError, error.URLError, TimeoutError, ValueError) as exc:
            retry_after_seconds = None
            if isinstance(exc, error.HTTPError):
                last_status_code = int(exc.code)
                retry_after_seconds = parse_retry_after_seconds(exc.headers.get("Retry-After"))
            last_error = str(exc)
            if attempt == max_retries - 1:
                break
            sleep_seconds = min(retry_backoff_cap_seconds, retry_backoff_base_seconds * (2**attempt))
            if retry_after_seconds is not None:
                sleep_seconds = max(sleep_seconds, retry_after_seconds)
            if last_status_code == 429:
                sleep_seconds = max(sleep_seconds, retry_after_floor_seconds)
            time.sleep(max(sleep_seconds, 0.0))
    return {
        "status_code": last_status_code,
        "elapsed_seconds": None,
        "error": last_error or "unknown error",
        "bytes": 0,
    }


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    try:
        text = "".join(page.get_text() for page in doc)
    finally:
        doc.close()
    text = re.sub(r"^\d{3}\n", "", text, flags=re.MULTILINE)
    text = text.replace("\x00", "")
    if len(text.strip()) < 500:
        raise ValueError(f"Extracted text is unexpectedly short for {pdf_path.name}")
    return text


def ensure_local_fulltext(
    *,
    paper_row: dict[str, Any],
    pdf_dir: Path,
    fulltext_dir: Path,
    timeout_seconds: int,
    max_retries: int,
    overwrite: bool,
    retry_backoff_base_seconds: float = 2.0,
    retry_after_floor_seconds: float = 30.0,
    retry_backoff_cap_seconds: float = 300.0,
) -> dict[str, Any]:
    paper_id = str(paper_row["paper_id"])
    pdf_path = pdf_dir / f"{paper_id}.pdf"
    fulltext_path = fulltext_dir / f"{paper_id}.txt"
    meta_path = fulltext_dir / f"{paper_id}.download.json"

    if not overwrite and fulltext_path.exists() and fulltext_path.stat().st_size > 500:
        meta = read_json(meta_path) if meta_path.exists() else {}
        return {
            "paper_id": paper_id,
            "pdf_path": str(pdf_path) if pdf_path.exists() else None,
            "fulltext_path": str(fulltext_path),
            "status": "cached",
            "download_meta": meta,
        }

    if overwrite and fulltext_path.exists():
        fulltext_path.unlink()

    if overwrite and pdf_path.exists():
        pdf_path.unlink()

    if not pdf_path.exists():
        download_meta = download_pdf(
            pdf_url=str(paper_row["pdf_url"]),
            paper_id=paper_id,
            dest_path=pdf_path,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base_seconds=retry_backoff_base_seconds,
            retry_after_floor_seconds=retry_after_floor_seconds,
            retry_backoff_cap_seconds=retry_backoff_cap_seconds,
        )
        write_json(meta_path, download_meta)
        if download_meta.get("error"):
            raise RuntimeError(f"PDF download failed for {paper_id}: {download_meta['error']}")
    else:
        download_meta = read_json(meta_path) if meta_path.exists() else {"status_code": None, "error": None}

    text = extract_text_from_pdf(pdf_path)
    fulltext_path.parent.mkdir(parents=True, exist_ok=True)
    fulltext_path.write_text(text, encoding="utf-8")

    return {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
        "fulltext_path": str(fulltext_path),
        "status": "downloaded",
        "download_meta": download_meta,
    }


def save_coarse_outputs(
    *,
    paper_dir: Path,
    paper_row: dict[str, Any],
    slim_result: Any,
) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "coarse_review.md").write_text(slim_result.markdown, encoding="utf-8")
    write_json(
        paper_dir / "coarse_review.json",
        {
            "paper_id": paper_row["paper_id"],
            "title": slim_result.title,
            "source_title": paper_row.get("title"),
            "year": paper_row.get("year"),
            "decision": paper_row.get("decision"),
            "committee_model": paper_row.get("committee_model"),
            "committee_bias": (slim_result.committee or {}).get("bias_variant", paper_row.get("committee_bias", "plain")),
            "llm_calls": slim_result.llm_calls,
            "review_cost_usd": slim_result.cost_usd,
            "call_costs": slim_result.call_costs,
            "rating": slim_result.review.rating,
            "confidence": slim_result.review.confidence,
            "soundness": slim_result.review.soundness,
            "presentation": slim_result.review.presentation,
            "contribution": slim_result.review.contribution,
            "recommendation": slim_result.review.recommendation,
            "summary": slim_result.review.summary,
            "strength": slim_result.review.strength,
            "weaknesses": slim_result.review.weaknesses,
            "questions": slim_result.review.questions,
            "rationale": slim_result.review.rationale,
            "committee": slim_result.committee,
            "structural_inventory": slim_result.structural_inventory.as_dict(),
        },
    )
    write_json(paper_dir / "coarse_call_costs.json", {"call_costs": slim_result.call_costs})
    write_json(paper_dir / "coarse_call_traces.json", {"call_traces": slim_result.call_traces})

    persona_dir = paper_dir / "persona_reviews"
    persona_dir.mkdir(parents=True, exist_ok=True)
    for persona_slug, persona_markdown in slim_result.persona_markdowns.items():
        (persona_dir / f"{persona_slug}.md").write_text(persona_markdown, encoding="utf-8")
        persona = slim_result.persona_reviews[persona_slug]
        write_json(
            persona_dir / f"{persona_slug}.json",
            {
                "paper_id": paper_row["paper_id"],
                "title": slim_result.title,
                "persona_slug": persona_slug,
                "committee_bias": (slim_result.committee or {}).get("bias_variant", paper_row.get("committee_bias", "plain")),
                "rating": persona.rating,
                "confidence": persona.confidence,
                "soundness": persona.soundness,
                "presentation": persona.presentation,
                "contribution": persona.contribution,
                "recommendation": persona.recommendation,
                "summary": persona.summary,
                "strength": persona.strength,
                "weaknesses": persona.weaknesses,
                "questions": persona.questions,
                "rationale": persona.rationale,
            },
        )


def load_saved_coarse_outputs(paper_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    committee_row = read_json(paper_dir / "coarse_review.json")
    persona_rows: list[dict[str, Any]] = []
    for path in sorted((paper_dir / "persona_reviews").glob("*.json")):
        persona_rows.append(read_json(path))
    if not persona_rows:
        raise FileNotFoundError(f"No saved persona JSONs found in {paper_dir / 'persona_reviews'}")
    return committee_row, persona_rows


def cached_committee_bias(paper_dir: Path) -> str:
    committee_path = paper_dir / "coarse_review.json"
    if not committee_path.exists():
        return "plain"
    committee_row = read_json(committee_path)
    return str(committee_row.get("committee_bias") or (committee_row.get("committee") or {}).get("bias_variant") or "plain")


def ensure_cached_committee_bias(paper_dir: Path, requested_bias: str) -> None:
    cached_bias = cached_committee_bias(paper_dir)
    if cached_bias != requested_bias:
        raise FileExistsError(
            f"Cached committee output in {paper_dir} uses committee_bias={cached_bias!r}; "
            f"requested {requested_bias!r}. Use --overwrite or a new --output-dir."
        )


def estimate_together_cost(
    *,
    model_id: str,
    system_message: str,
    user_message: str,
    raw_response: str,
) -> float | None:
    try:
        cost = litellm.completion_cost(
            model=f"together_ai/{model_id}",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            completion=raw_response,
            call_type="completion",
        )
    except Exception:
        return None
    if cost is None:
        return None
    return round(float(cost), 6)


@dataclass
class ProcessResult:
    paper_id: str
    year: int
    status: str
    summary: dict[str, Any]


def summarize_prefetch_results(successes: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_year = Counter(int(row["year"]) for row in successes)
    status_counts = Counter(str(row.get("status") or "unknown") for row in successes)
    failure_counts = Counter(str(row.get("error") or "unknown") for row in failures)
    return {
        "n_success": len(successes),
        "n_failed": len(failures),
        "years_completed": dict(sorted(by_year.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def prefetch_local_fulltexts(
    *,
    selected_papers: list[dict[str, Any]],
    output_dir: Path,
    pdf_dir: Path,
    fulltext_dir: Path,
    timeout_seconds: float,
    max_retries: int,
    overwrite: bool,
    request_delay_seconds: float,
    retry_backoff_base_seconds: float,
    retry_after_floor_seconds: float,
    retry_backoff_cap_seconds: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fulltext_dir.mkdir(parents=True, exist_ok=True)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()

    for idx, paper_row in enumerate(selected_papers, start=1):
        paper_id = str(paper_row["paper_id"])
        try:
            fulltext_meta = ensure_local_fulltext(
                paper_row=paper_row,
                pdf_dir=pdf_dir,
                fulltext_dir=fulltext_dir,
                timeout_seconds=int(math.ceil(timeout_seconds)),
                max_retries=max_retries,
                overwrite=overwrite,
                retry_backoff_base_seconds=retry_backoff_base_seconds,
                retry_after_floor_seconds=retry_after_floor_seconds,
                retry_backoff_cap_seconds=retry_backoff_cap_seconds,
            )
            txt_path = Path(str(fulltext_meta["fulltext_path"]))
            successes.append(
                {
                    "paper_id": paper_id,
                    "year": int(paper_row["year"]),
                    "title": paper_row.get("title"),
                    "status": str(fulltext_meta.get("status") or "unknown"),
                    "pdf_path": fulltext_meta.get("pdf_path"),
                    "fulltext_path": fulltext_meta.get("fulltext_path"),
                    "fulltext_bytes": txt_path.stat().st_size if txt_path.exists() else 0,
                    "download_meta": fulltext_meta.get("download_meta") or {},
                    "updated_at_utc": now_utc(),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "paper_id": paper_id,
                    "year": int(paper_row["year"]),
                    "title": paper_row.get("title"),
                    "pdf_url": paper_row.get("pdf_url"),
                    "error": str(exc),
                    "updated_at_utc": now_utc(),
                }
            )

        if idx % 25 == 0 or idx == len(selected_papers):
            elapsed_minutes = round((time.time() - started) / 60.0, 2)
            print(
                f"[prefetch {idx}/{len(selected_papers)}] ok={len(successes)} "
                f"failed={len(failures)} elapsed={elapsed_minutes}m",
                flush=True,
            )

        if request_delay_seconds > 0 and idx < len(selected_papers):
            time.sleep(request_delay_seconds)

    successes.sort(key=lambda row: (int(row["year"]), str(row["paper_id"])))
    failures.sort(key=lambda row: (int(row["year"]), str(row["paper_id"])))
    write_jsonl(output_dir / "prefetch_successes.jsonl", successes)
    write_jsonl(output_dir / "prefetch_failures.jsonl", failures)

    summary = {
        "created_at_utc": now_utc(),
        "n_selected": len(selected_papers),
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
        "metrics": summarize_prefetch_results(successes, failures),
        "pdf_dir": str(pdf_dir),
        "fulltext_dir": str(fulltext_dir),
        "request_delay_seconds": request_delay_seconds,
        "retry_backoff_base_seconds": retry_backoff_base_seconds,
        "retry_after_floor_seconds": retry_after_floor_seconds,
        "retry_backoff_cap_seconds": retry_backoff_cap_seconds,
        "download_max_retries": max_retries,
    }
    write_json(output_dir / "prefetch_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def process_one_paper(
    *,
    paper_row: dict[str, Any],
    output_root: Path,
    pdf_dir: Path,
    fulltext_dir: Path,
    committee_model: str,
    committee_bias: str,
    decision_head: HeadModel,
    config: CoarseConfig,
    api_key: str,
    personas: list[str],
    persona_weights: dict[str, float] | None,
    timeout_seconds: float,
    max_retries: int,
    overwrite: bool,
    max_content_chars: int,
    section_char_limit: int,
    intro_max_chars: int,
    method_max_chars: int,
    conclusion_max_chars: int,
    head_temperature: float,
    head_top_p: float,
    head_max_tokens: int,
    stage: str,
) -> ProcessResult:
    paper_id = str(paper_row["paper_id"])
    year = int(paper_row["year"])
    paper_dir = output_root / "papers" / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paper {paper_id}] start year={year}", flush=True)

    input_payload = dict(paper_row)
    input_payload["committee_model"] = committee_model
    input_payload["committee_bias"] = committee_bias
    input_payload["decision_head_model"] = decision_head.model_id
    write_json(paper_dir / "input.json", input_payload)

    fulltext_meta = ensure_local_fulltext(
        paper_row=paper_row,
        pdf_dir=pdf_dir,
        fulltext_dir=fulltext_dir,
        timeout_seconds=int(timeout_seconds),
        max_retries=max_retries,
        overwrite=overwrite,
    )
    print(
        f"[paper {paper_id}] fulltext ready status={fulltext_meta.get('status')} "
        f"path={fulltext_meta.get('fulltext_path')}",
        flush=True,
    )
    sample_row = dict(paper_row)
    sample_row["fulltext_available"] = True
    sample_row["fulltext_path"] = fulltext_meta["fulltext_path"]
    sample_row["committee_model"] = committee_model
    sample_row["committee_bias"] = committee_bias
    write_json(paper_dir / "local_fulltext.json", fulltext_meta)

    committee_path = paper_dir / "coarse_review.json"
    deepseek_path = paper_dir / "deepseek_decision.json"
    run_committee = stage in {"both", "committee_only"}
    run_decision = stage in {"both", "decision_only"}

    if run_committee and run_decision and committee_path.exists() and deepseek_path.exists() and not overwrite:
        ensure_cached_committee_bias(paper_dir, committee_bias)
        final_payload = read_json(paper_dir / "paper_result.json")
        print(f"[paper {paper_id}] using cached committee + deepseek outputs", flush=True)
        return ProcessResult(
            paper_id=paper_id,
            year=year,
            status="cached",
            summary=final_payload,
        )

    if committee_path.exists() and not overwrite:
        ensure_cached_committee_bias(paper_dir, committee_bias)
        print(f"[paper {paper_id}] reusing cached committee outputs", flush=True)
        committee_row, persona_rows = load_saved_coarse_outputs(paper_dir)
    elif not run_committee:
        raise FileNotFoundError(
            f"Decision-only stage requested but no committee outputs exist for paper {paper_id}."
        )
    else:
        print(
            f"[paper {paper_id}] running slim committee model={committee_model} "
            f"committee_bias={committee_bias}",
            flush=True,
        )
        slim_result = review_paper_slim(
            pdf_path=sample_row["fulltext_path"],
            model=committee_model,
            config=config,
            title_hint=str(sample_row.get("title") or ""),
            personas=personas,
            persona_weights=persona_weights,
            committee_bias=committee_bias,
            timeout_seconds=int(timeout_seconds),
            intro_max_chars=intro_max_chars,
            method_max_chars=method_max_chars,
            conclusion_max_chars=conclusion_max_chars,
        )
        save_coarse_outputs(paper_dir=paper_dir, paper_row=sample_row, slim_result=slim_result)
        committee_row, persona_rows = load_saved_coarse_outputs(paper_dir)
        print(
            f"[paper {paper_id}] committee finished rating={committee_row.get('rating')} "
            f"reco={committee_row.get('recommendation')}",
            flush=True,
        )

    packet = build_packet(
        sample_row=sample_row,
        committee_row=committee_row,
        persona_rows=persona_rows,
        committee_source_model=committee_model,
        max_content_chars=max_content_chars,
        section_char_limit=section_char_limit,
    )

    packet_payload = {
        "paper_id": packet.paper_id,
        "title": packet.title,
        "year": year,
        "true_decision": packet.true_decision,
        "committee_source_model": packet.committee_source_model,
        "committee_scores": packet.committee_scores,
        "committee_recommendation": packet.committee_recommendation,
        "selected_sections": packet.selected_sections,
        "structural_inventory": packet.structural_inventory,
        "disagreement": packet.disagreement,
        "feature_vector": packet.feature_vector,
    }
    write_json(paper_dir / "decision_packet.json", packet_payload)
    (paper_dir / "decision_packet.md").write_text(render_packet_markdown(packet), encoding="utf-8")

    if not run_decision:
        deepseek_payload: dict[str, Any] = {}
        print(f"[paper {paper_id}] committee-only stage complete", flush=True)
    elif deepseek_path.exists() and not overwrite:
        deepseek_payload = read_json(deepseek_path)
        print(f"[paper {paper_id}] reusing cached deepseek output", flush=True)
    else:
        print(f"[paper {paper_id}] running decision head model={decision_head.label}", flush=True)
        user_prompt = build_user_prompt(packet)
        api_result = together_request(
            model=decision_head,
            api_key=api_key,
            system_message=HEAD_SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=head_temperature,
            top_p=head_top_p,
            max_tokens=head_max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        parsed = parse_llm_decision(api_result["raw_response"])
        deepseek_payload = {
            "paper_id": paper_id,
            "title": packet.title,
            "year": year,
            "true_decision": packet.true_decision,
            "decision_head_model": decision_head.model_id,
            "decision_head_label": decision_head.label,
            "system_message": HEAD_SYSTEM_PROMPT,
            "user_message": user_prompt,
            "parsed": parsed,
            "raw_response": api_result["raw_response"],
            "usage": api_result["usage"],
            "finish_reason": api_result["finish_reason"],
            "elapsed_seconds": api_result["elapsed_seconds"],
            "http_error": api_result["http_error"],
            "estimated_cost_usd": estimate_together_cost(
                model_id=decision_head.model_id,
                system_message=HEAD_SYSTEM_PROMPT,
                user_message=user_prompt,
                raw_response=api_result["raw_response"],
            ),
        }
        write_json(deepseek_path, deepseek_payload)
        print(
            f"[paper {paper_id}] decision head finished decision="
            f"{(deepseek_payload.get('parsed') or {}).get('decision')} "
            f"p={(deepseek_payload.get('parsed') or {}).get('p_accept')}",
            flush=True,
        )

    paper_result = {
        "paper_id": paper_id,
        "title": sample_row.get("title"),
        "year": year,
        "decision": sample_row.get("decision"),
        "accepted": sample_row.get("accepted"),
        "mean_rating": sample_row.get("mean_rating"),
        "score_centered": sample_row.get("score_centered"),
        "cutoff": sample_row.get("cutoff"),
        "bandwidth": sample_row.get("bandwidth"),
        "committee_model": committee_model,
        "committee_bias": committee_bias,
        "decision_head_model": decision_head.model_id,
        "committee_rating": committee_row.get("rating"),
        "committee_recommendation": committee_row.get("recommendation"),
        "committee_cost_usd": committee_row.get("review_cost_usd"),
        "committee_llm_calls": committee_row.get("llm_calls"),
        "deepseek_decision": (deepseek_payload.get("parsed") or {}).get("decision"),
        "deepseek_p_accept": (deepseek_payload.get("parsed") or {}).get("p_accept"),
        "deepseek_margin": (deepseek_payload.get("parsed") or {}).get("margin"),
        "deepseek_elapsed_seconds": deepseek_payload.get("elapsed_seconds"),
        "deepseek_http_error": deepseek_payload.get("http_error"),
        "deepseek_estimated_cost_usd": deepseek_payload.get("estimated_cost_usd"),
        "pdf_path": fulltext_meta.get("pdf_path"),
        "fulltext_path": fulltext_meta.get("fulltext_path"),
        "packet_path": str(paper_dir / "decision_packet.json"),
        "coarse_review_path": str(committee_path),
        "deepseek_path": str(deepseek_path) if run_decision or deepseek_path.exists() else None,
        "stage": stage,
        "updated_at_utc": now_utc(),
    }
    write_json(paper_dir / "paper_result.json", paper_result)
    return ProcessResult(paper_id=paper_id, year=year, status="completed", summary=paper_result)


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_year = Counter(int(row["year"]) for row in rows)
    deepseek_errors = sum(1 for row in rows if row.get("deepseek_http_error"))
    committee_cost = sum(float(row.get("committee_cost_usd") or 0.0) for row in rows)
    deepseek_cost = sum(float(row.get("deepseek_estimated_cost_usd") or 0.0) for row in rows)
    deepseek_accepts = sum(1 for row in rows if row.get("deepseek_decision") == "accept")
    return {
        "n_completed": len(rows),
        "years": dict(sorted(by_year.items())),
        "deepseek_accepts": deepseek_accepts,
        "deepseek_errors": deepseek_errors,
        "committee_cost_usd": round(committee_cost, 6),
        "deepseek_estimated_cost_usd": round(deepseek_cost, 6),
        "total_estimated_cost_usd": round(committee_cost + deepseek_cost, 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma committee + DeepSeek on the 2018-2020 RDD bandwidth sample.")
    parser.add_argument("--rdd-csv", type=Path, default=DEFAULT_RDD_CSV)
    parser.add_argument("--openreview-csv", type=Path, default=DEFAULT_OPENREVIEW_CSV)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--years", default=DEFAULT_YEARS)
    parser.add_argument("--committee-model", default=DEFAULT_COMMITTEE_MODEL)
    parser.add_argument("--committee-bias", choices=sorted(COMMITTEE_BIAS_PROMPTS), default="plain")
    parser.add_argument("--decision-head-models", default=DEFAULT_DECISION_HEAD_MODELS)
    parser.add_argument("--personas", default="default-ensemble")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--paper-id", dest="paper_ids", action="append", default=None)
    parser.add_argument("--max-parallel-papers", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=0,
        help="Abort with non-zero exit after this many consecutive paper-level failures. 0 disables the circuit breaker.",
    )
    parser.add_argument("--head-temperature", type=float, default=0.0)
    parser.add_argument("--head-top-p", type=float, default=0.9)
    parser.add_argument("--head-max-tokens", type=int, default=1200)
    parser.add_argument("--max-content-chars", type=int, default=9000)
    parser.add_argument("--section-char-limit", type=int, default=1800)
    parser.add_argument("--intro-max-chars", type=int, default=4000)
    parser.add_argument("--method-max-chars", type=int, default=8000)
    parser.add_argument("--conclusion-max-chars", type=int, default=2000)
    parser.add_argument("--stage", choices=("both", "committee_only", "decision_only"), default="both")
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--prefetch-delay-seconds", type=float, default=2.0)
    parser.add_argument("--download-retry-backoff-base-seconds", type=float, default=10.0)
    parser.add_argument("--download-retry-after-floor-seconds", type=float, default=30.0)
    parser.add_argument("--download-retry-backoff-cap-seconds", type=float, default=300.0)
    parser.add_argument("--download-max-retries", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selection-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--fulltext-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    requested_ids = set(args.paper_ids) if args.paper_ids else None
    personas = parse_personas(args.personas)
    persona_weights = parse_weights(args.weights, personas)
    head_models = resolve_models(args.decision_head_models)
    if len(head_models) != 1:
        raise ValueError("This runner expects exactly one decision head model.")
    decision_head = head_models[0]
    run_slug = infer_run_slug(years, args.committee_model, decision_head, args.committee_bias)

    selection_dir = (args.selection_dir or (ROOT / "OutputNew" / "LLMOutput" / run_slug)).resolve()
    output_dir = (args.output_dir or (ROOT / "OutputNew" / "Empirics" / run_slug)).resolve()
    pdf_dir = (args.pdf_dir or (ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview" / run_slug / "pdf")).resolve()
    fulltext_dir = (args.fulltext_dir or (ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview" / run_slug / "fulltext")).resolve()

    target_rows = load_rdd_sample(
        path=args.rdd_csv.resolve(),
        years=set(years),
        max_papers=args.max_papers,
        requested_ids=requested_ids,
    )
    if not target_rows:
        raise ValueError("No RDD sample papers matched the requested filter.")

    paper_year_map = {str(row["paper_id"]): int(row["year"]) for row in target_rows}
    metadata_rows = load_openreview_metadata(args.openreview_csv.resolve(), paper_year_map)
    selected_papers = build_selected_papers(
        rdd_rows=target_rows,
        metadata_rows=metadata_rows,
        fulltext_dir=fulltext_dir,
    )

    selection_manifest = {
        "created_at_utc": now_utc(),
        "run_slug": run_slug,
        "years": years,
        "rdd_csv": str(args.rdd_csv.resolve()),
        "openreview_csv": str(args.openreview_csv.resolve()),
        "committee_model": args.committee_model,
        "committee_bias": args.committee_bias,
        "decision_head_model": {"id": decision_head.model_id, "label": decision_head.label},
        "personas": personas,
        "persona_weights": persona_weights,
        "max_papers": args.max_papers,
        "requested_ids": sorted(requested_ids) if requested_ids else None,
        "n_selected": len(selected_papers),
        "selection_dir": str(selection_dir),
        "output_dir": str(output_dir),
        "pdf_dir": str(pdf_dir),
        "fulltext_dir": str(fulltext_dir),
    }
    write_json(selection_dir / "run_manifest.json", selection_manifest)
    write_jsonl(selection_dir / "selected_papers.jsonl", selected_papers)

    print(
        f"Selected {len(selected_papers)} papers for years {years}. "
        f"Selection manifest: {selection_dir / 'run_manifest.json'}"
    )

    if args.prepare_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = read_api_key(args.key_file.resolve())
    config = build_local_config(api_key)

    run_manifest = {
        **selection_manifest,
        "api_key_path": str(args.key_file.resolve()),
        "max_parallel_papers": args.max_parallel_papers,
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "head_temperature": args.head_temperature,
        "head_top_p": args.head_top_p,
        "head_max_tokens": args.head_max_tokens,
        "max_consecutive_failures": args.max_consecutive_failures,
        "max_content_chars": args.max_content_chars,
        "section_char_limit": args.section_char_limit,
        "intro_max_chars": args.intro_max_chars,
        "method_max_chars": args.method_max_chars,
        "conclusion_max_chars": args.conclusion_max_chars,
        "stage": args.stage,
        "prefetch_only": args.prefetch_only,
        "prefetch_delay_seconds": args.prefetch_delay_seconds,
        "download_retry_backoff_base_seconds": args.download_retry_backoff_base_seconds,
        "download_retry_after_floor_seconds": args.download_retry_after_floor_seconds,
        "download_retry_backoff_cap_seconds": args.download_retry_backoff_cap_seconds,
        "download_max_retries": args.download_max_retries,
        "prepare_only": False,
        "overwrite": args.overwrite,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_jsonl(output_dir / "sample_papers.jsonl", selected_papers)

    if args.prefetch_only:
        prefetch_local_fulltexts(
            selected_papers=selected_papers,
            output_dir=output_dir,
            pdf_dir=pdf_dir,
            fulltext_dir=fulltext_dir,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.download_max_retries,
            overwrite=args.overwrite,
            request_delay_seconds=args.prefetch_delay_seconds,
            retry_backoff_base_seconds=args.download_retry_backoff_base_seconds,
            retry_after_floor_seconds=args.download_retry_after_floor_seconds,
            retry_backoff_cap_seconds=args.download_retry_backoff_cap_seconds,
        )
        return

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    abort_reason: str | None = None
    consecutive_failures = 0
    completed_or_failed = 0

    started = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_papers)
    future_to_paper: dict[concurrent.futures.Future[ProcessResult], dict[str, Any]] = {}
    paper_iter = iter(selected_papers)
    submitted = 0

    def submit_next_papers() -> None:
        nonlocal submitted
        while len(future_to_paper) < args.max_parallel_papers:
            try:
                paper_row = next(paper_iter)
            except StopIteration:
                return
            future = executor.submit(
                process_one_paper,
                paper_row=paper_row,
                output_root=output_dir,
                pdf_dir=pdf_dir,
                fulltext_dir=fulltext_dir,
                committee_model=args.committee_model,
                committee_bias=args.committee_bias,
                decision_head=decision_head,
                config=config,
                api_key=api_key,
                personas=personas,
                persona_weights=persona_weights,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                overwrite=args.overwrite,
                max_content_chars=args.max_content_chars,
                section_char_limit=args.section_char_limit,
                intro_max_chars=args.intro_max_chars,
                method_max_chars=args.method_max_chars,
                conclusion_max_chars=args.conclusion_max_chars,
                head_temperature=args.head_temperature,
                head_top_p=args.head_top_p,
                head_max_tokens=args.head_max_tokens,
                stage=args.stage,
            )
            future_to_paper[future] = paper_row
            submitted += 1

    try:
        submit_next_papers()
        while future_to_paper:
            done, _ = concurrent.futures.wait(
                future_to_paper,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                paper_row = future_to_paper.pop(future)
                paper_id = str(paper_row["paper_id"])
                completed_or_failed += 1
                try:
                    result = future.result()
                    results.append(result.summary)
                    consecutive_failures = 0
                    elapsed = time.time() - started
                    print(
                        f"[{completed_or_failed}/{len(selected_papers)}] {paper_id} year={paper_row['year']} "
                        f"status={result.status} deepseek={result.summary.get('deepseek_decision')} "
                        f"elapsed={elapsed/60:.1f}m"
                    )
                except Exception as exc:
                    consecutive_failures += 1
                    failure = {
                        "paper_id": paper_id,
                        "year": int(paper_row["year"]),
                        "title": paper_row.get("title"),
                        "error": str(exc),
                        "consecutive_failures": consecutive_failures,
                        "updated_at_utc": now_utc(),
                    }
                    failures.append(failure)
                    write_json(output_dir / "papers" / paper_id / "failure.json", failure)
                    print(
                        f"[{completed_or_failed}/{len(selected_papers)}] {paper_id} FAILED: {exc} "
                        f"(consecutive_failures={consecutive_failures})",
                        flush=True,
                    )
                    if (
                        args.max_consecutive_failures > 0
                        and consecutive_failures >= args.max_consecutive_failures
                    ):
                        abort_reason = (
                            f"max_consecutive_failures reached: "
                            f"{consecutive_failures}/{args.max_consecutive_failures}"
                        )
                        break

            if abort_reason is not None:
                break
            submit_next_papers()
    finally:
        if abort_reason is not None:
            for future in future_to_paper:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)

    results.sort(key=lambda row: (int(row["year"]), float(row["score_centered"]), str(row["paper_id"])))
    failures.sort(key=lambda row: (int(row["year"]), str(row["paper_id"])))

    write_jsonl(output_dir / "paper_results.jsonl", results)
    write_jsonl(output_dir / "failures.jsonl", failures)
    summary = {
        "created_at_utc": now_utc(),
        "run_slug": run_slug,
        "years": years,
        "n_selected": len(selected_papers),
        "n_completed": len(results),
        "n_failed": len(failures),
        "abort_reason": abort_reason,
        "max_consecutive_failures": args.max_consecutive_failures,
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
        "metrics": summarize_results(results),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if abort_reason is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
