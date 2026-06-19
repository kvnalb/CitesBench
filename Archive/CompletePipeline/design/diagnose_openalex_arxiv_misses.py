#!/usr/bin/env python3
"""
Diagnose residual OpenAlex misses after arXiv DOI matching.

This script takes the paper-level OpenAlex linkage file produced by
fetch_openalex_citations_from_arxiv_matches.py, isolates the unmatched rows,
queries OpenAlex by title, and classifies likely failure modes:

- exact title candidate but under a different/non-arXiv record
- candidate exists but only near-title
- no convincing OpenAlex candidate

Outputs:
- rawdata/Design/OpenAlex/openalex_rdd_miss_title_search_diagnostics.csv
- rawdata/Design/OpenAlex/openalex_rdd_miss_title_search_candidates.csv
- Output/Design/iclr_local_rdd/openalex_rdd_miss_diagnostic_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_INPUT_CSV = ROOT / "OutputNew" / "rawdata" / "Design" / "OpenAlex" / "openalex_rdd_arxiv_paper_level.csv"
DEFAULT_RAW_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "OpenAlex"
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "Design" / "iclr_local_rdd"
DEFAULT_OPENALEX_KEY_PATH = ROOT / "OpenAlex.txt"
OPENALEX_API_URL = "https://api.openalex.org/works"
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class ApiResult:
    status_code: int | None
    ok: bool
    error: str | None
    elapsed_seconds: float
    response_json: dict[str, object] | None
    cache_path: Path
    from_cache: bool
    request_url: str
    response_headers: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose residual OpenAlex misses with title search.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Paper-level OpenAlex linkage CSV.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for raw diagnostics and title-search cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary JSON.",
    )
    parser.add_argument(
        "--openalex-key-path",
        type=Path,
        default=DEFAULT_OPENALEX_KEY_PATH,
        help="Path to the OpenAlex API key file.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=5,
        help="OpenAlex candidates per title query.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for each request.",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=5,
        help="Maximum attempts per live request.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Initial backoff after retryable failures.",
    )
    parser.add_argument(
        "--retry-backoff-factor",
        type=float,
        default=2.0,
        help="Multiplicative retry backoff factor.",
    )
    parser.add_argument(
        "--retry-max-sleep-seconds",
        type=float,
        default=120.0,
        help="Maximum backoff sleep.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Inter-request sleep for polite title queries.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap for testing.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached title-search responses and re-query OpenAlex.",
    )
    parser.add_argument(
        "--user-agent",
        default="LLMReview-openalex-miss-diagnostic/0.1",
        help="User-Agent header.",
    )
    return parser.parse_args()


def maybe_read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_title(title: object) -> str:
    if title is None:
        return ""
    text = str(title).strip().lower()
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\n": " ",
    }
    for raw, clean in replacements.items():
        text = text.replace(raw, clean)
    text = text.replace("&", " and ")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_doi(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = text.strip().lower()
    return text or None


def title_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def response_headers_dict(headers: object) -> dict[str, str]:
    if headers is None:
        return {}
    pairs: dict[str, str] = {}
    try:
        for key, value in headers.items():
            if key:
                pairs[str(key)] = str(value)
    except Exception:
        return {}
    return pairs


def choose_retry_sleep(payload: dict[str, object], default_sleep: float, max_sleep: float) -> float:
    headers = payload.get("response_headers")
    if isinstance(headers, dict):
        for key in ["Retry-After", "retry-after", "X-RateLimit-Reset", "x-ratelimit-reset"]:
            raw = headers.get(key)
            if raw is None:
                continue
            try:
                seconds = float(str(raw).strip())
            except Exception:
                continue
            if seconds > 0:
                return min(seconds, max_sleep)
    return min(default_sleep, max_sleep)


def perform_json_request(
    url: str,
    cache_path: Path,
    *,
    refresh: bool,
    timeout_seconds: float,
    retry_max_attempts: int,
    retry_backoff_seconds: float,
    retry_backoff_factor: float,
    retry_max_sleep_seconds: float,
    user_agent: str,
) -> ApiResult:
    if cache_path.exists() and not refresh:
        cached = maybe_read_json(cache_path)
        if cached is None:
            raise RuntimeError(f"Failed to read cache file: {cache_path}")
        return ApiResult(
            status_code=cached.get("status_code"),
            ok=bool(cached.get("ok")),
            error=cached.get("error"),
            elapsed_seconds=float(cached.get("elapsed_seconds") or 0.0),
            response_json=cached.get("response_json"),
            cache_path=cache_path.resolve(),
            from_cache=True,
            request_url=url,
            response_headers=response_headers_dict(cached.get("response_headers")),
        )

    backoff = retry_backoff_seconds
    for attempt_number in range(1, retry_max_attempts + 1):
        started = time.perf_counter()
        now = datetime.now(timezone.utc).isoformat()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body_text = response.read().decode("utf-8")
                elapsed = time.perf_counter() - started
                payload = {
                    "url": url,
                    "fetched_at": now,
                    "attempt_number": attempt_number,
                    "status_code": response.status,
                    "ok": True,
                    "error": None,
                    "elapsed_seconds": elapsed,
                    "response_headers": response_headers_dict(response.headers),
                    "response_json": json.loads(body_text),
                }
                write_json(cache_path, payload)
                return ApiResult(
                    status_code=response.status,
                    ok=True,
                    error=None,
                    elapsed_seconds=elapsed,
                    response_json=payload["response_json"],
                    cache_path=cache_path.resolve(),
                    from_cache=False,
                    request_url=url,
                    response_headers=response_headers_dict(response.headers),
                )
        except urllib.error.HTTPError as exc:
            elapsed = time.perf_counter() - started
            body_text = exc.read().decode("utf-8", errors="replace")
            response_json = None
            try:
                response_json = json.loads(body_text)
            except Exception:
                response_json = None
            payload = {
                "url": url,
                "fetched_at": now,
                "attempt_number": attempt_number,
                "status_code": exc.code,
                "ok": False,
                "error": f"HTTPError {exc.code}",
                "elapsed_seconds": elapsed,
                "response_headers": response_headers_dict(exc.headers),
                "response_json": response_json,
                "response_text": body_text[:4000],
            }
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            elapsed = time.perf_counter() - started
            payload = {
                "url": url,
                "fetched_at": now,
                "attempt_number": attempt_number,
                "status_code": None,
                "ok": False,
                "error": repr(exc),
                "elapsed_seconds": elapsed,
                "response_headers": {},
                "response_json": None,
            }

        status_code = payload.get("status_code")
        retryable = status_code in RETRYABLE_HTTP_STATUS_CODES or status_code is None
        if not retryable or attempt_number == retry_max_attempts:
            write_json(cache_path, payload)
            return ApiResult(
                status_code=payload.get("status_code"),
                ok=False,
                error=payload.get("error"),
                elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
                response_json=payload.get("response_json"),
                cache_path=cache_path.resolve(),
                from_cache=False,
                request_url=url,
                response_headers=response_headers_dict(payload.get("response_headers")),
            )
        time.sleep(choose_retry_sleep(payload, backoff, retry_max_sleep_seconds))
        backoff = min(backoff * retry_backoff_factor, retry_max_sleep_seconds)

    raise RuntimeError(f"OpenAlex request failed without returning a payload: {url}")


def safe_paper_id(paper_id: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(paper_id))


def build_search_url(title: str, api_key: str, per_page: int) -> str:
    params = {
        "api_key": api_key,
        "search": title,
        "per-page": str(per_page),
        "select": "id,doi,display_name,publication_year,cited_by_count,ids,type,primary_location,locations",
    }
    return f"{OPENALEX_API_URL}?{urllib.parse.urlencode(params)}"


def extract_candidate_rows(
    row: dict[str, object],
    response_json: dict[str, object] | None,
) -> list[dict[str, object]]:
    if not isinstance(response_json, dict):
        return []
    results = response_json.get("results")
    if not isinstance(results, list):
        return []

    query_title = normalize_title(row.get("title") or row.get("arxiv_title"))
    query_doi = normalize_doi(row.get("openalex_query_doi"))
    query_arxiv_id = str(row.get("arxiv_id_canonical") or "").strip().lower()
    input_year = row.get("year")
    if input_year is not None and not (isinstance(input_year, float) and math.isnan(input_year)):
        input_year = int(float(input_year))
    else:
        input_year = None

    records: list[dict[str, object]] = []
    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        candidate_title = item.get("display_name")
        candidate_title_norm = normalize_title(candidate_title)
        similarity = title_similarity(query_title, candidate_title_norm)
        jaccard = token_jaccard(query_title, candidate_title_norm)
        exact = bool(query_title) and query_title == candidate_title_norm
        near = similarity >= 0.97 and jaccard >= 0.90

        candidate_year = item.get("publication_year")
        if isinstance(candidate_year, int):
            year_diff = abs(candidate_year - input_year) if input_year is not None else None
        else:
            year_diff = None

        ids = item.get("ids") if isinstance(item.get("ids"), dict) else {}
        candidate_doi_norm = normalize_doi(item.get("doi") or ids.get("doi"))
        doi_matches_query = bool(query_doi and candidate_doi_norm and query_doi == candidate_doi_norm)

        locations = item.get("locations") if isinstance(item.get("locations"), list) else []
        location_urls: list[str] = []
        for location in locations:
            if not isinstance(location, dict):
                continue
            for key in ["landing_page_url", "pdf_url"]:
                value = location.get(key)
                if value:
                    location_urls.append(str(value))
        primary_location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else None
        if primary_location is not None:
            for key in ["landing_page_url", "pdf_url"]:
                value = primary_location.get(key)
                if value:
                    location_urls.append(str(value))
        has_arxiv_location = any("arxiv.org/" in url.lower() for url in location_urls)
        arxiv_id_in_location = any(query_arxiv_id and query_arxiv_id in url.lower() for url in location_urls)
        has_arxiv_doi = bool(candidate_doi_norm and candidate_doi_norm.startswith("10.48550/arxiv."))

        score = (
            (6.0 if exact else 0.0)
            + (2.0 if near else 0.0)
            + (1.5 if doi_matches_query else 0.0)
            + (1.0 if arxiv_id_in_location else 0.0)
            + (0.5 if has_arxiv_location else 0.0)
            + (0.5 if year_diff == 0 else 0.25 if year_diff == 1 else 0.0)
            + (math.log1p(float(item.get("cited_by_count") or 0)) / 10.0)
        )
        records.append(
            {
                "paper_id": row.get("paper_id"),
                "rank": rank,
                "candidate_score": score,
                "openalex_id": item.get("id"),
                "openalex_display_name": candidate_title,
                "openalex_doi": item.get("doi"),
                "openalex_publication_year": candidate_year,
                "openalex_cited_by_count": item.get("cited_by_count"),
                "openalex_type": item.get("type"),
                "title_similarity": similarity,
                "token_jaccard": jaccard,
                "exact_title_match": exact,
                "near_exact_title_match": near,
                "doi_matches_query": doi_matches_query,
                "has_arxiv_location": has_arxiv_location,
                "arxiv_id_in_location": arxiv_id_in_location,
                "has_arxiv_doi": has_arxiv_doi,
                "year_diff": year_diff,
                "location_urls_json": json.dumps(location_urls, ensure_ascii=False) if location_urls else None,
                "ids_json": json.dumps(ids, ensure_ascii=False) if ids else None,
            }
        )

    if records:
        frame = pd.DataFrame(records).sort_values(
            by=[
                "exact_title_match",
                "doi_matches_query",
                "arxiv_id_in_location",
                "candidate_score",
                "openalex_cited_by_count",
            ],
            ascending=[False, False, False, False, False],
        )
        return frame.to_dict(orient="records")
    return []


def classify_top_candidate(row: dict[str, object], candidates: list[dict[str, object]]) -> tuple[str, dict[str, object] | None]:
    if not candidates:
        return "no_candidates", None

    top = candidates[0]
    exact = bool(top.get("exact_title_match"))
    near = bool(top.get("near_exact_title_match"))
    doi_matches = bool(top.get("doi_matches_query"))
    has_arxiv_location = bool(top.get("has_arxiv_location"))
    arxiv_id_in_location = bool(top.get("arxiv_id_in_location"))
    candidate_year = top.get("openalex_publication_year")
    input_year = row.get("year")
    year_ok = True
    if isinstance(candidate_year, int) and input_year is not None and not (isinstance(input_year, float) and math.isnan(input_year)):
        year_ok = abs(candidate_year - int(float(input_year))) <= 2

    if doi_matches:
        return "same_doi_found_via_title_search", top
    if exact and year_ok and (arxiv_id_in_location or has_arxiv_location):
        return "exact_title_arxiv_record_without_doi_match", top
    if exact and year_ok:
        return "exact_title_other_record", top
    if near and year_ok:
        return "near_exact_candidate", top
    return "no_confident_match", top


def summarize_diagnostics(diag_df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_diagnostics": int(len(diag_df)),
        "classification_counts": diag_df["diagnostic_status"].value_counts(dropna=False).to_dict(),
        "by_year": (
            diag_df.groupby("year")["diagnostic_status"]
            .value_counts()
            .unstack(fill_value=0)
            .sort_index()
            .to_dict(orient="index")
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    openalex_key = args.openalex_key_path.read_text(encoding="utf-8").strip()
    if not openalex_key:
        raise ValueError(f"OpenAlex key file is empty: {args.openalex_key_path}")

    input_df = pd.read_csv(args.input_csv, low_memory=False)
    miss_df = input_df.loc[~input_df["openalex_matched"].fillna(False)].copy()
    if args.max_papers is not None:
        miss_df = miss_df.head(args.max_papers).copy()

    cache_dir = args.raw_dir / "openalex_miss_title_search_cache"
    diagnostic_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    total = len(miss_df)
    for index, row in enumerate(miss_df.to_dict(orient="records"), start=1):
        query_title = str(row.get("title") or row.get("arxiv_title") or "").strip()
        if not query_title:
            diagnostic_rows.append(
                {
                    "paper_id": row.get("paper_id"),
                    "year": row.get("year"),
                    "title": row.get("title"),
                    "arxiv_title": row.get("arxiv_title"),
                    "arxiv_id_canonical": row.get("arxiv_id_canonical"),
                    "openalex_query_doi": row.get("openalex_query_doi"),
                    "diagnostic_status": "missing_query_title",
                    "n_candidates": 0,
                }
            )
            continue

        cache_path = cache_dir / f"{safe_paper_id(row.get('paper_id'))}.json"
        url = build_search_url(query_title, api_key=openalex_key, per_page=args.per_page)
        result = perform_json_request(
            url,
            cache_path,
            refresh=args.refresh,
            timeout_seconds=args.timeout_seconds,
            retry_max_attempts=args.retry_max_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            retry_backoff_factor=args.retry_backoff_factor,
            retry_max_sleep_seconds=args.retry_max_sleep_seconds,
            user_agent=args.user_agent,
        )
        candidates = extract_candidate_rows(row, result.response_json)
        candidate_rows.extend(candidates)
        status, top = classify_top_candidate(row, candidates)
        diagnostic_rows.append(
            {
                "paper_id": row.get("paper_id"),
                "year": row.get("year"),
                "accepted": row.get("accepted"),
                "title": row.get("title"),
                "arxiv_title": row.get("arxiv_title"),
                "arxiv_id_canonical": row.get("arxiv_id_canonical"),
                "openalex_query_doi": row.get("openalex_query_doi"),
                "arxiv_first_version_year": row.get("arxiv_first_version_year"),
                "arxiv_update_date": row.get("arxiv_update_date"),
                "diagnostic_status": status,
                "n_candidates": len(candidates),
                "request_ok": result.ok,
                "request_status_code": result.status_code,
                "request_error": result.error,
                "from_cache": result.from_cache,
                "elapsed_seconds": result.elapsed_seconds,
                "top_openalex_id": top.get("openalex_id") if top else None,
                "top_openalex_display_name": top.get("openalex_display_name") if top else None,
                "top_openalex_doi": top.get("openalex_doi") if top else None,
                "top_openalex_publication_year": top.get("openalex_publication_year") if top else None,
                "top_openalex_cited_by_count": top.get("openalex_cited_by_count") if top else None,
                "top_title_similarity": top.get("title_similarity") if top else None,
                "top_token_jaccard": top.get("token_jaccard") if top else None,
                "top_exact_title_match": top.get("exact_title_match") if top else None,
                "top_doi_matches_query": top.get("doi_matches_query") if top else None,
                "top_has_arxiv_location": top.get("has_arxiv_location") if top else None,
                "top_arxiv_id_in_location": top.get("arxiv_id_in_location") if top else None,
                "top_year_diff": top.get("year_diff") if top else None,
            }
        )
        print(f"[{index:,}/{total:,}] {row.get('paper_id')} -> {status} ({len(candidates)} candidates)")
        if not result.from_cache and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    diagnostic_df = pd.DataFrame(diagnostic_rows)
    candidates_df = pd.DataFrame(candidate_rows)

    diagnostic_path = args.raw_dir / "openalex_rdd_miss_title_search_diagnostics.csv"
    candidates_path = args.raw_dir / "openalex_rdd_miss_title_search_candidates.csv"
    summary_path = args.output_dir / "openalex_rdd_miss_diagnostic_summary.json"

    diagnostic_df.to_csv(diagnostic_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    summary = summarize_diagnostics(diagnostic_df)
    summary.update(
        {
            "input_csv": str(args.input_csv.resolve()),
            "diagnostic_path": str(diagnostic_path.resolve()),
            "candidates_path": str(candidates_path.resolve()),
        }
    )
    write_json(summary_path, summary)

    print(f"Wrote diagnostics to {diagnostic_path}")
    print(f"Wrote candidates to {candidates_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
