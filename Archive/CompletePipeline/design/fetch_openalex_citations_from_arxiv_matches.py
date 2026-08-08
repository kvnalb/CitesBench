#!/usr/bin/env python3
"""
Fetch OpenAlex citations for the arXiv-matched ICLR RDD sample.

This script takes the year-specific-bandwidth RDD sample, links it to the
existing local arXiv exact-match results, synthesizes canonical arXiv DOIs when
needed, and queries OpenAlex in conservative batches using DOI OR-filters.

Outputs:
- rawdata/Design/OpenAlex/openalex_rdd_arxiv_query_input.csv
- rawdata/Design/OpenAlex/openalex_rdd_arxiv_batch_manifest.csv
- rawdata/Design/OpenAlex/openalex_rdd_arxiv_unique_works.csv
- rawdata/Design/OpenAlex/openalex_rdd_arxiv_paper_level.csv
- Output/Design/iclr_local_rdd/rdd_sample_year_specific_bandwidth_with_openalex_citations.csv
- Output/Design/iclr_local_rdd/openalex_rdd_arxiv_match_rate_by_year.csv
- Output/Design/iclr_local_rdd/openalex_rdd_arxiv_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
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
from pathlib import Path

import pandas as pd

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_RDD_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
DEFAULT_ARXIV_MATCH_CSV = ROOT / "OutputNew" / "rawdata" / "Design" / "arXiv" / "arxiv_dump_combined_best_matches.csv"
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
    parser = argparse.ArgumentParser(
        description="Fetch OpenAlex citations for the arXiv-matched ICLR RDD sample."
    )
    parser.add_argument(
        "--rdd-csv",
        type=Path,
        default=DEFAULT_RDD_CSV,
        help="RDD sample CSV with year-specific bandwidths.",
    )
    parser.add_argument(
        "--arxiv-match-csv",
        type=Path,
        default=DEFAULT_ARXIV_MATCH_CSV,
        help="Exact arXiv match results CSV.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for raw OpenAlex inputs and cache files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for merged outputs and summaries.",
    )
    parser.add_argument(
        "--openalex-key-path",
        type=Path,
        default=DEFAULT_OPENALEX_KEY_PATH,
        help="Path to the OpenAlex API key file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of DOI lookups per OpenAlex request. Conservative default matches official examples.",
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
        default=0.25,
        help="Inter-request sleep. OpenAlex allows much faster throughput, but this keeps runs polite.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap on arXiv-matched sample rows for testing.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of OpenAlex batches to run.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached batch responses and re-query OpenAlex.",
    )
    parser.add_argument(
        "--user-agent",
        default="LLMReview-openalex-arxiv/0.1",
        help="User-Agent header for OpenAlex requests.",
    )
    return parser.parse_args()


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def maybe_read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def doi_url(value: object) -> str | None:
    normalized = normalize_doi(value)
    return f"https://doi.org/{normalized}" if normalized else None


def canonical_arxiv_doi_from_id(arxiv_id: object) -> str | None:
    if arxiv_id is None:
        return None
    if isinstance(arxiv_id, float) and math.isnan(arxiv_id):
        return None
    text = str(arxiv_id).strip()
    if not text:
        return None
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        return None
    return f"https://doi.org/10.48550/arxiv.{text.lower()}"


def extract_arxiv_id_from_url(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text or None


def best_arxiv_id(row: pd.Series) -> str | None:
    for field in ["arxiv_abs_url", "arxiv_pdf_url", "arxiv_id"]:
        candidate = extract_arxiv_id_from_url(row.get(field))
        if candidate:
            return candidate
    return None


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
    last_payload = None
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

        last_payload = payload
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


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def batch_hash(values: list[str]) -> str:
    digest = hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()
    return digest[:12]


def build_openalex_url(doi_urls: list[str], api_key: str) -> str:
    params = {
        "api_key": api_key,
        "filter": f"doi:{'|'.join(doi_urls)}",
        "per-page": str(len(doi_urls)),
        "select": "id,doi,display_name,publication_year,cited_by_count,cited_by_api_url,ids,type,primary_location",
    }
    return f"{OPENALEX_API_URL}?{urllib.parse.urlencode(params)}"


def partition(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def extract_openalex_records(response_json: dict[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(response_json, dict):
        return []
    results = response_json.get("results")
    if not isinstance(results, list):
        return []
    records: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        ids = item.get("ids") if isinstance(item.get("ids"), dict) else {}
        primary_location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else None
        doi_norm = normalize_doi(item.get("doi") or ids.get("doi"))
        records.append(
            {
                "openalex_query_doi": doi_norm,
                "openalex_id": item.get("id"),
                "openalex_doi": doi_url(item.get("doi") or ids.get("doi")),
                "openalex_display_name": item.get("display_name"),
                "openalex_publication_year": item.get("publication_year"),
                "openalex_cited_by_count": item.get("cited_by_count"),
                "openalex_cited_by_api_url": item.get("cited_by_api_url"),
                "openalex_type": item.get("type"),
                "openalex_primary_location_json": json.dumps(primary_location, ensure_ascii=False)
                if primary_location is not None
                else None,
                "openalex_ids_json": json.dumps(ids, ensure_ascii=False) if ids else None,
            }
        )
    return records


def build_query_panel(rdd_df: pd.DataFrame, arxiv_df: pd.DataFrame, max_papers: int | None) -> pd.DataFrame:
    matched_df = arxiv_df.loc[arxiv_df["matched"].fillna(False).map(to_bool)].copy()
    matched_df = matched_df.loc[matched_df["arxiv_id"].notna()].copy()
    sort_cols = [col for col in ["candidate_score", "title_similarity", "token_jaccard"] if col in matched_df.columns]
    if sort_cols:
        matched_df = matched_df.sort_values(sort_cols, ascending=False)
    matched_df = matched_df.drop_duplicates(subset=["paper_id"], keep="first")

    keep_cols = [
        col
        for col in [
            "paper_id",
            "input_title",
            "input_year",
            "source_id",
            "match_source",
            "arxiv_id",
            "arxiv_title",
            "arxiv_authors",
            "arxiv_categories",
            "arxiv_doi",
            "arxiv_abs_url",
            "arxiv_pdf_url",
            "arxiv_update_date",
            "arxiv_first_version_year",
            "candidate_score",
            "title_similarity",
            "token_jaccard",
        ]
        if col in matched_df.columns
    ]
    matched_df = matched_df[keep_cols].copy()
    matched_df["arxiv_id_canonical"] = matched_df.apply(best_arxiv_id, axis=1)

    merged = rdd_df.merge(matched_df, on="paper_id", how="left", indicator=True)
    merged["has_arxiv_match"] = merged["_merge"] == "both"
    merged.drop(columns=["_merge"], inplace=True)
    arxiv_matched = merged.loc[merged["has_arxiv_match"]].copy()
    arxiv_matched["openalex_query_doi"] = arxiv_matched["arxiv_doi"].map(doi_url)
    missing_doi = arxiv_matched["openalex_query_doi"].isna()
    arxiv_matched.loc[missing_doi, "openalex_query_doi"] = arxiv_matched.loc[missing_doi, "arxiv_id_canonical"].map(
        canonical_arxiv_doi_from_id
    )
    arxiv_matched["openalex_query_doi_norm"] = arxiv_matched["openalex_query_doi"].map(normalize_doi)
    arxiv_matched["openalex_query_source"] = arxiv_matched["arxiv_doi"].notna().map(
        lambda flag: "arxiv_doi" if flag else "synthetic_10_48550"
    )
    arxiv_matched = arxiv_matched.loc[arxiv_matched["openalex_query_doi_norm"].notna()].copy()

    if max_papers is not None:
        arxiv_matched = arxiv_matched.head(max_papers).copy()

    return arxiv_matched


def summarize_years(full_df: pd.DataFrame, matched_df: pd.DataFrame) -> pd.DataFrame:
    full_counts = (
        full_df.groupby("year", dropna=False)
        .agg(
            n_rdd_rows=("paper_id", "size"),
            n_arxiv_matched=("has_arxiv_match", lambda s: int(s.fillna(False).sum())),
        )
        .reset_index()
    )
    matched_summary = (
        matched_df.groupby("year", dropna=False)
        .agg(
            n_openalex_matched=("openalex_matched", lambda s: int(s.fillna(False).sum())),
            mean_cited_by_count=("openalex_cited_by_count", "mean"),
            median_cited_by_count=("openalex_cited_by_count", "median"),
        )
        .reset_index()
    )
    year_df = full_counts.merge(matched_summary, on="year", how="left")
    year_df["n_openalex_matched"] = year_df["n_openalex_matched"].fillna(0).astype(int)
    year_df["openalex_match_rate_given_arxiv_match"] = year_df["n_openalex_matched"] / year_df["n_arxiv_matched"].replace(0, pd.NA)
    return year_df.sort_values("year").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    openalex_key = args.openalex_key_path.read_text(encoding="utf-8").strip()
    if not openalex_key:
        raise ValueError(f"OpenAlex key file is empty: {args.openalex_key_path}")

    rdd_df = pd.read_csv(args.rdd_csv, low_memory=False)
    arxiv_df = pd.read_csv(args.arxiv_match_csv, low_memory=False)

    query_df = build_query_panel(rdd_df, arxiv_df, args.max_papers)
    if query_df.empty:
        raise ValueError("No arXiv-matched rows with usable OpenAlex query DOIs were found.")

    raw_input_path = args.raw_dir / "openalex_rdd_arxiv_query_input.csv"
    query_df.to_csv(raw_input_path, index=False)

    unique_doi_df = (
        query_df[["openalex_query_doi_norm", "openalex_query_doi"]]
        .drop_duplicates()
        .sort_values("openalex_query_doi_norm")
        .reset_index(drop=True)
    )
    unique_doi_urls = unique_doi_df["openalex_query_doi"].tolist()
    batches = partition(unique_doi_urls, args.batch_size)
    if args.max_batches is not None:
        batches = batches[: args.max_batches]
    requested_doi_norms = {normalize_doi(value) for batch in batches for value in batch}
    unique_doi_df = unique_doi_df.loc[unique_doi_df["openalex_query_doi_norm"].isin(requested_doi_norms)].copy()
    query_df = query_df.loc[query_df["openalex_query_doi_norm"].isin(requested_doi_norms)].copy()

    cache_dir = args.raw_dir / "openalex_batch_cache"
    manifest_rows: list[dict[str, object]] = []
    unique_work_rows: list[dict[str, object]] = []

    total_batches = len(batches)
    for batch_index, doi_batch in enumerate(batches, start=1):
        url = build_openalex_url(doi_batch, api_key=openalex_key)
        digest = batch_hash(doi_batch)
        cache_path = cache_dir / f"batch_{batch_index:04d}_{safe_slug(digest)}.json"
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
        batch_records = extract_openalex_records(result.response_json)
        unique_work_rows.extend(batch_records)
        manifest_rows.append(
            {
                "batch_index": batch_index,
                "batch_hash": digest,
                "n_requested": len(doi_batch),
                "n_results": len(batch_records),
                "status_code": result.status_code,
                "ok": result.ok,
                "error": result.error,
                "elapsed_seconds": result.elapsed_seconds,
                "from_cache": result.from_cache,
                "cache_path": str(result.cache_path),
                "request_url": result.request_url,
                "x_ratelimit_limit": result.response_headers.get("X-RateLimit-Limit"),
                "x_ratelimit_remaining": result.response_headers.get("X-RateLimit-Remaining"),
                "x_ratelimit_credits_used": result.response_headers.get("X-RateLimit-Credits-Used"),
                "x_ratelimit_reset": result.response_headers.get("X-RateLimit-Reset"),
            }
        )
        print(
            f"[{batch_index:,}/{total_batches:,}] queried {len(doi_batch):,} DOIs; "
            f"received {len(batch_records):,} results."
        )
        if not result.from_cache and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = args.raw_dir / "openalex_rdd_arxiv_batch_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    unique_work_df = pd.DataFrame(unique_work_rows)
    if unique_work_df.empty:
        unique_work_df = pd.DataFrame(
            columns=[
                "openalex_query_doi",
                "openalex_id",
                "openalex_doi",
                "openalex_display_name",
                "openalex_publication_year",
                "openalex_cited_by_count",
                "openalex_cited_by_api_url",
                "openalex_type",
                "openalex_primary_location_json",
                "openalex_ids_json",
            ]
        )
    unique_work_df["openalex_query_doi_norm"] = unique_work_df["openalex_query_doi"].map(normalize_doi)
    unique_work_df = unique_work_df.loc[unique_work_df["openalex_query_doi_norm"].isin(requested_doi_norms)].copy()
    if not unique_work_df.empty:
        unique_work_df = (
            unique_work_df.sort_values(
                by=["openalex_query_doi_norm", "openalex_cited_by_count", "openalex_publication_year"],
                ascending=[True, False, False],
            )
            .drop_duplicates(subset=["openalex_query_doi_norm"], keep="first")
            .reset_index(drop=True)
        )
    unique_work_path = args.raw_dir / "openalex_rdd_arxiv_unique_works.csv"
    unique_work_df.to_csv(unique_work_path, index=False)

    paper_level_df = query_df.merge(
        unique_work_df,
        on="openalex_query_doi_norm",
        how="left",
        suffixes=("", "_openalex"),
    )
    paper_level_df["openalex_matched"] = paper_level_df["openalex_id"].notna()
    paper_level_path = args.raw_dir / "openalex_rdd_arxiv_paper_level.csv"
    paper_level_df.to_csv(paper_level_path, index=False)

    merged_full_df = rdd_df.merge(
        paper_level_df.drop_duplicates(subset=["paper_id"]),
        on="paper_id",
        how="left",
        suffixes=("", "_openalex"),
    )
    merged_full_df["has_arxiv_match"] = merged_full_df["has_arxiv_match"].map(to_bool)
    merged_full_path = args.output_dir / "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv"
    merged_full_df.to_csv(merged_full_path, index=False)

    year_summary_df = summarize_years(merged_full_df, paper_level_df)
    year_summary_path = args.output_dir / "openalex_rdd_arxiv_match_rate_by_year.csv"
    year_summary_df.to_csv(year_summary_path, index=False)

    summary = {
        "rdd_csv": str(args.rdd_csv.resolve()),
        "arxiv_match_csv": str(args.arxiv_match_csv.resolve()),
        "raw_query_input_path": str(raw_input_path.resolve()),
        "batch_manifest_path": str(manifest_path.resolve()),
        "unique_work_path": str(unique_work_path.resolve()),
        "paper_level_path": str(paper_level_path.resolve()),
        "merged_full_path": str(merged_full_path.resolve()),
        "year_summary_path": str(year_summary_path.resolve()),
        "n_rdd_rows": int(len(rdd_df)),
        "n_arxiv_matched_rows": int(len(query_df)),
        "n_unique_openalex_query_dois": int(len(unique_doi_df)),
        "n_batches_requested": int(total_batches),
        "n_openalex_unique_matches": int(unique_work_df["openalex_id"].notna().sum()) if not unique_work_df.empty else 0,
        "n_openalex_paper_level_matches": int(paper_level_df["openalex_matched"].fillna(False).sum()),
        "openalex_match_rate_given_arxiv_match": float(paper_level_df["openalex_matched"].fillna(False).mean()),
        "mean_cited_by_count_matched": float(paper_level_df.loc[paper_level_df["openalex_matched"], "openalex_cited_by_count"].mean())
        if paper_level_df["openalex_matched"].fillna(False).any()
        else None,
        "median_cited_by_count_matched": float(
            paper_level_df.loc[paper_level_df["openalex_matched"], "openalex_cited_by_count"].median()
        )
        if paper_level_df["openalex_matched"].fillna(False).any()
        else None,
    }
    summary_path = args.output_dir / "openalex_rdd_arxiv_summary.json"
    write_json(summary_path, summary)

    print(f"Wrote raw query input to {raw_input_path}")
    print(f"Wrote batch manifest to {manifest_path}")
    print(f"Wrote unique OpenAlex works to {unique_work_path}")
    print(f"Wrote paper-level OpenAlex merge to {paper_level_path}")
    print(f"Wrote merged RDD sample to {merged_full_path}")
    print(f"Wrote year summary to {year_summary_path}")
    print(f"Wrote summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
