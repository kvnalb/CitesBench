#!/usr/bin/env python3
"""
Query arXiv metadata for ICLR papers and cache raw responses locally.

The local ICLR analytic files do not contain author names, so matching here is
primarily title-based. The script therefore uses conservative auto-match rules:
an arXiv record is marked as an automatic match only when the normalized arXiv
title exactly matches the normalized local title. Near matches are retained in
the candidate output for later review or secondary matching.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_INPUT_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
DEFAULT_RAW_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "arXiv"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class QueryAttempt:
    label: str
    search_query: str
    xml_path: Path
    meta_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch arXiv metadata for local ICLR papers with on-disk caching."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input paper-level CSV. Defaults to the year-specific RDD sample.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where raw arXiv query data and derived outputs are written.",
    )
    parser.add_argument(
        "--paper-id-col",
        default="paper_id",
        help="Paper identifier column in the input CSV.",
    )
    parser.add_argument(
        "--title-col",
        default="title",
        help="Title column in the input CSV.",
    )
    parser.add_argument(
        "--year-col",
        default="year",
        help="Year column in the input CSV.",
    )
    parser.add_argument(
        "--source-id-col",
        default="source_id",
        help="Optional source-id column in the input CSV.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of arXiv results returned per query attempt.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.5,
        help=(
            "Minimum delay between live arXiv requests. arXiv's current limit is "
            "no more than one request every 3 seconds."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of input rows to process.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for each arXiv request.",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=5,
        help="Maximum live request attempts for a single arXiv query before giving up.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=15.0,
        help="Initial backoff after retryable failures.",
    )
    parser.add_argument(
        "--retry-backoff-factor",
        type=float,
        default=2.0,
        help="Multiplicative backoff factor after each retryable failure.",
    )
    parser.add_argument(
        "--retry-max-sleep-seconds",
        type=float,
        default=300.0,
        help="Maximum sleep for any one retry backoff.",
    )
    parser.add_argument(
        "--cooldown-after-consecutive-failures",
        type=int,
        default=3,
        help="Trigger a longer cooldown after this many consecutive query-level failures.",
    )
    parser.add_argument(
        "--cooldown-sleep-seconds",
        type=float,
        default=900.0,
        help="Cooldown sleep after a sustained run of failed queries.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a one-line progress update every N papers during live querying.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached responses and re-query arXiv.",
    )
    parser.add_argument(
        "--allow-near-exact-auto-match",
        action="store_true",
        help=(
            "Allow high-similarity near-exact title matches to count as automatic matches. "
            "By default only exact normalized-title matches are auto-matched."
        ),
    )
    parser.add_argument(
        "--user-agent",
        default="LLMReview-arxiv-match/0.1",
        help="User-Agent header sent to arXiv.",
    )
    return parser.parse_args()


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
    }
    for raw, clean in replacements.items():
        text = text.replace(raw, clean)
    text = text.replace("&", " and ")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def title_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def build_attempts(raw_dir: Path, paper_id: str, title: str) -> list[QueryAttempt]:
    query_dir = raw_dir / "query_cache"
    query_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", paper_id)
    exact_query = f'ti:"{title}"'
    fallback_query = f'all:"{title}"'
    return [
        QueryAttempt(
            label="title_exact",
            search_query=exact_query,
            xml_path=query_dir / f"{safe_id}__title_exact.xml",
            meta_path=query_dir / f"{safe_id}__title_exact.json",
        ),
        QueryAttempt(
            label="all_fields_phrase",
            search_query=fallback_query,
            xml_path=query_dir / f"{safe_id}__all_fields_phrase.xml",
            meta_path=query_dir / f"{safe_id}__all_fields_phrase.json",
        ),
    ]


def maybe_read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def wait_for_min_interval(last_live_request_at: float | None, min_delay_seconds: float) -> None:
    if last_live_request_at is None:
        return
    elapsed = time.monotonic() - last_live_request_at
    if elapsed < min_delay_seconds:
        time.sleep(min_delay_seconds - elapsed)


def parse_retry_after_seconds(headers: object) -> float | None:
    if headers is None:
        return None
    raw_value = None
    try:
        raw_value = headers.get("Retry-After")
    except AttributeError:
        raw_value = None
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    try:
        retry_dt = parsedate_to_datetime(text)
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def request_arxiv(
    attempt: QueryAttempt,
    max_results: int,
    timeout_seconds: float,
    user_agent: str,
) -> tuple[bytes, dict[str, object]]:
    params = urllib.parse.urlencode(
        {
            "search_query": attempt.search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    request_url = f"{ARXIV_API_URL}?{params}"
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": user_agent},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        xml_bytes = response.read()
        http_status = getattr(response, "status", 200)
    metadata = {
        "attempt": attempt.label,
        "request_url": request_url,
        "search_query": attempt.search_query,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": http_status,
    }
    return xml_bytes, metadata


def fetch_arxiv_with_retries(
    attempt: QueryAttempt,
    args: argparse.Namespace,
    last_live_request_at: float | None,
) -> tuple[bytes | None, dict[str, object], float]:
    request_errors: list[dict[str, object]] = []
    retry_wait_seconds_total = 0.0
    backoff_seconds = max(args.sleep_seconds, args.retry_backoff_seconds)

    for request_attempt in range(1, args.retry_max_attempts + 1):
        wait_for_min_interval(last_live_request_at, args.sleep_seconds)

        try:
            xml_bytes, metadata = request_arxiv(
                attempt=attempt,
                max_results=args.max_results,
                timeout_seconds=args.timeout_seconds,
                user_agent=args.user_agent,
            )
            metadata["request_attempts"] = request_attempt
            metadata["attempt_errors"] = request_errors
            metadata["retry_wait_seconds_total"] = retry_wait_seconds_total
            metadata["policy_min_delay_seconds"] = args.sleep_seconds
            return xml_bytes, metadata, time.monotonic()
        except urllib.error.HTTPError as exc:
            last_live_request_at = time.monotonic()
            retry_after_seconds = parse_retry_after_seconds(getattr(exc, "headers", None))
            retryable = exc.code in RETRYABLE_HTTP_STATUS_CODES
            error_row = {
                "request_attempt": request_attempt,
                "error_type": "HTTPError",
                "http_status": exc.code,
                "error": str(exc.reason),
                "request_url": getattr(exc, "url", None),
                "retryable": retryable,
                "retry_after_seconds": retry_after_seconds,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            request_errors.append(error_row)
            if not retryable or request_attempt >= args.retry_max_attempts:
                return (
                    None,
                    {
                        "attempt": attempt.label,
                        "search_query": attempt.search_query,
                        "request_url": getattr(exc, "url", None),
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "http_status": exc.code,
                        "error": f"HTTPError: {exc.reason}",
                        "request_attempts": request_attempt,
                        "attempt_errors": request_errors,
                        "retry_wait_seconds_total": retry_wait_seconds_total,
                        "policy_min_delay_seconds": args.sleep_seconds,
                    },
                    last_live_request_at,
                )

            sleep_seconds = max(
                args.sleep_seconds,
                min(
                    args.retry_max_sleep_seconds,
                    retry_after_seconds if retry_after_seconds is not None else backoff_seconds,
                ),
            )
            time.sleep(sleep_seconds)
            retry_wait_seconds_total += sleep_seconds
            backoff_seconds = min(
                args.retry_max_sleep_seconds,
                max(args.sleep_seconds, backoff_seconds * args.retry_backoff_factor),
            )
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            last_live_request_at = time.monotonic()
            reason = getattr(exc, "reason", exc)
            error_row = {
                "request_attempt": request_attempt,
                "error_type": type(exc).__name__,
                "error": str(reason),
                "retryable": True,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            request_errors.append(error_row)
            if request_attempt >= args.retry_max_attempts:
                return (
                    None,
                    {
                        "attempt": attempt.label,
                        "search_query": attempt.search_query,
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": f"{type(exc).__name__}: {reason}",
                        "request_attempts": request_attempt,
                        "attempt_errors": request_errors,
                        "retry_wait_seconds_total": retry_wait_seconds_total,
                        "policy_min_delay_seconds": args.sleep_seconds,
                    },
                    last_live_request_at,
                )

            sleep_seconds = min(args.retry_max_sleep_seconds, backoff_seconds)
            time.sleep(max(args.sleep_seconds, sleep_seconds))
            retry_wait_seconds_total += max(args.sleep_seconds, sleep_seconds)
            backoff_seconds = min(
                args.retry_max_sleep_seconds,
                max(args.sleep_seconds, backoff_seconds * args.retry_backoff_factor),
            )

    return (
        None,
        {
            "attempt": attempt.label,
            "search_query": attempt.search_query,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": "Retry loop exhausted unexpectedly.",
            "request_attempts": args.retry_max_attempts,
            "attempt_errors": request_errors,
            "retry_wait_seconds_total": retry_wait_seconds_total,
            "policy_min_delay_seconds": args.sleep_seconds,
        },
        time.monotonic(),
    )


def extract_arxiv_id(id_url: str | None) -> str | None:
    if not id_url:
        return None
    match = re.search(r"/abs/([^/?#]+)", id_url)
    if match:
        return match.group(1)
    return id_url.rstrip("/").rsplit("/", 1)[-1]


def extract_pdf_url(entry: ET.Element) -> str | None:
    for link in entry.findall("atom:link", ATOM_NS):
        title = link.attrib.get("title")
        href = link.attrib.get("href")
        if title == "pdf" and href:
            return href
    return None


def extract_affiliations(entry: ET.Element) -> list[str]:
    affiliations: list[str] = []
    for author in entry.findall("atom:author", ATOM_NS):
        affiliation = author.findtext("arxiv:affiliation", default="", namespaces=ATOM_NS).strip()
        if affiliation:
            affiliations.append(affiliation)
    return affiliations


def parse_feed(xml_text: str) -> list[dict[str, object]]:
    root = ET.fromstring(xml_text)
    entries: list[dict[str, object]] = []
    for rank, entry in enumerate(root.findall("atom:entry", ATOM_NS), start=1):
        id_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip() or None
        title = entry.findtext("atom:title", default="", namespaces=ATOM_NS).strip()
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip() or None
        updated = entry.findtext("atom:updated", default="", namespaces=ATOM_NS).strip() or None
        summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS).strip() or None
        doi = entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS).strip() or None
        journal_ref = entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM_NS).strip() or None
        primary_category_node = entry.find("arxiv:primary_category", ATOM_NS)
        primary_category = (
            primary_category_node.attrib.get("term")
            if primary_category_node is not None
            else None
        )
        categories = [
            node.attrib.get("term", "").strip()
            for node in entry.findall("atom:category", ATOM_NS)
            if node.attrib.get("term", "").strip()
        ]
        authors = [
            author.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
            for author in entry.findall("atom:author", ATOM_NS)
            if author.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
        ]
        affiliations = extract_affiliations(entry)
        entries.append(
            {
                "rank": rank,
                "id_url": id_url,
                "arxiv_id": extract_arxiv_id(id_url),
                "title": title,
                "published": published,
                "updated": updated,
                "summary": summary,
                "doi": doi,
                "journal_ref": journal_ref,
                "primary_category": primary_category,
                "categories": categories,
                "authors": authors,
                "affiliations": affiliations,
                "pdf_url": extract_pdf_url(entry),
            }
        )
    return entries


def published_year(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None
    return int(text[:4])


def score_candidate(input_title: str, input_year: int | None, candidate_title: str, candidate_published: str | None) -> dict[str, object]:
    normalized_input = normalize_title(input_title)
    normalized_candidate = normalize_title(candidate_title)
    similarity = title_similarity(normalized_input, normalized_candidate)
    jaccard = token_jaccard(normalized_input, normalized_candidate)
    exact = normalized_input == normalized_candidate and normalized_input != ""
    candidate_year = published_year(candidate_published)
    year_diff = None
    if input_year is not None and candidate_year is not None:
        year_diff = abs(input_year - candidate_year)
    penalty = 0.0 if year_diff is None else max(0, year_diff - 1) * 0.03
    score = (1.0 if exact else 0.0) + similarity + (0.2 * jaccard) - penalty
    near_exact = similarity >= 0.96 and jaccard >= 0.90
    return {
        "input_title_norm": normalized_input,
        "candidate_title_norm": normalized_candidate,
        "title_similarity": similarity,
        "token_jaccard": jaccard,
        "exact_title_match": exact,
        "near_exact_title_match": near_exact,
        "published_year": candidate_year,
        "year_diff": year_diff,
        "candidate_score": score,
    }


def load_input_rows(
    input_csv: Path,
    paper_id_col: str,
    title_col: str,
    year_col: str,
    source_id_col: str,
    limit: int | None,
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv)
    required_cols = {paper_id_col, title_col, year_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    keep_cols = [paper_id_col, title_col, year_col]
    if source_id_col in df.columns:
        keep_cols.append(source_id_col)

    out = df[keep_cols].copy()
    out = out.rename(
        columns={
            paper_id_col: "paper_id",
            title_col: "title",
            year_col: "year",
            source_id_col: "source_id" if source_id_col in out.columns else source_id_col,
        }
    )
    if "source_id" not in out.columns:
        out["source_id"] = pd.NA

    out["paper_id"] = out["paper_id"].astype(str).str.strip()
    out["title"] = out["title"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["source_id"] = out["source_id"].astype("string")
    out = out.drop_duplicates(subset=["paper_id"]).reset_index(drop=True)
    out = out.loc[out["paper_id"].ne("") & out["title"].ne("")].copy()
    if limit is not None:
        out = out.head(limit).copy()
    return out.reset_index(drop=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def collect_attempt_data(attempt: QueryAttempt) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    xml_text = maybe_read_text(attempt.xml_path)
    if xml_text is None:
        return None, []
    meta_payload: dict[str, object] | None = None
    meta_text = maybe_read_text(attempt.meta_path)
    if meta_text is not None:
        meta_payload = json.loads(meta_text)
    try:
        entries = parse_feed(xml_text)
    except ET.ParseError as exc:
        entries = []
        if meta_payload is None:
            meta_payload = {"attempt": attempt.label}
        meta_payload["parse_error"] = str(exc)
    return meta_payload, entries


def run_queries(args: argparse.Namespace, input_rows: pd.DataFrame) -> None:
    last_live_request_at: float | None = None
    consecutive_query_failures = 0
    total_rows = len(input_rows)

    for row_number, row in enumerate(input_rows.itertuples(index=False), start=1):
        attempts = build_attempts(args.raw_dir, row.paper_id, row.title)
        for attempt in attempts:
            if not args.refresh and attempt.xml_path.exists() and attempt.meta_path.exists():
                _, cached_entries = collect_attempt_data(attempt)
                if cached_entries:
                    break
                continue

            xml_bytes, metadata, last_live_request_at = fetch_arxiv_with_retries(
                attempt=attempt,
                args=args,
                last_live_request_at=last_live_request_at,
            )

            if xml_bytes is None:
                write_json(attempt.meta_path, metadata)
                consecutive_query_failures += 1
                print(
                    (
                        f"[{row_number}/{total_rows}] {row.paper_id} {attempt.label} failed "
                        f"after {metadata.get('request_attempts')} attempt(s): "
                        f"{metadata.get('error')}"
                    ),
                    flush=True,
                )
                if (
                    args.cooldown_after_consecutive_failures > 0
                    and consecutive_query_failures >= args.cooldown_after_consecutive_failures
                ):
                    print(
                        (
                            f"Cooling down for {args.cooldown_sleep_seconds:.0f}s after "
                            f"{consecutive_query_failures} consecutive failed queries."
                        ),
                        flush=True,
                    )
                    time.sleep(args.cooldown_sleep_seconds)
                    consecutive_query_failures = 0
                continue

            consecutive_query_failures = 0

            attempt.xml_path.parent.mkdir(parents=True, exist_ok=True)
            attempt.xml_path.write_bytes(xml_bytes)

            entries = parse_feed(xml_bytes.decode("utf-8"))
            metadata["n_entries"] = len(entries)
            metadata["xml_path"] = str(attempt.xml_path)
            write_json(attempt.meta_path, metadata)

            if args.progress_every > 0 and (
                row_number == 1 or row_number % args.progress_every == 0 or row_number == total_rows
            ):
                print(
                    (
                        f"[{row_number}/{total_rows}] {row.paper_id} {attempt.label} "
                        f"attempts={metadata.get('request_attempts')} entries={len(entries)}"
                    ),
                    flush=True,
                )

            if entries:
                break


def build_outputs(
    args: argparse.Namespace,
    input_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    best_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for row in input_rows.itertuples(index=False):
        attempts = build_attempts(args.raw_dir, row.paper_id, row.title)
        best_candidate: dict[str, object] | None = None
        best_attempt_label: str | None = None
        best_meta: dict[str, object] | None = None
        last_attempt_label: str | None = None
        last_meta: dict[str, object] | None = None
        total_entries = 0
        attempt_status = "no_cache"

        for attempt in attempts:
            meta_payload, entries = collect_attempt_data(attempt)
            if meta_payload is None and not attempt.xml_path.exists():
                continue

            attempt_status = "cached"
            last_attempt_label = attempt.label
            if meta_payload is not None:
                last_meta = meta_payload
            total_entries += len(entries)
            for candidate in entries:
                metrics = score_candidate(
                    input_title=row.title,
                    input_year=(int(row.year) if pd.notna(row.year) else None),
                    candidate_title=str(candidate["title"]),
                    candidate_published=candidate.get("published"),
                )
                candidate_row = {
                    "paper_id": row.paper_id,
                    "source_id": row.source_id,
                    "input_year": int(row.year) if pd.notna(row.year) else pd.NA,
                    "input_title": row.title,
                    "query_attempt": attempt.label,
                    "raw_xml_path": str(attempt.xml_path),
                    "query_meta_path": str(attempt.meta_path),
                    "arxiv_rank": candidate["rank"],
                    "arxiv_id": candidate["arxiv_id"],
                    "arxiv_abs_url": candidate["id_url"],
                    "arxiv_pdf_url": candidate["pdf_url"],
                    "arxiv_title": candidate["title"],
                    "arxiv_published": candidate["published"],
                    "arxiv_updated": candidate["updated"],
                    "arxiv_doi": candidate["doi"],
                    "arxiv_journal_ref": candidate["journal_ref"],
                    "arxiv_primary_category": candidate["primary_category"],
                    "arxiv_categories": "|".join(candidate["categories"]),
                    "arxiv_authors": " | ".join(candidate["authors"]),
                    "arxiv_affiliations": " | ".join(candidate["affiliations"]),
                    **metrics,
                }
                candidate_rows.append(candidate_row)
                if best_candidate is None or float(candidate_row["candidate_score"]) > float(best_candidate["candidate_score"]):
                    best_candidate = candidate_row
                    best_attempt_label = attempt.label
                    best_meta = meta_payload

            if entries:
                break

        best_row = {
            "paper_id": row.paper_id,
            "source_id": row.source_id,
            "input_year": int(row.year) if pd.notna(row.year) else pd.NA,
            "input_title": row.title,
            "input_title_norm": normalize_title(row.title),
            "attempt_status": attempt_status,
            "best_query_attempt": best_attempt_label or last_attempt_label,
            "total_entries_seen": total_entries,
            "matched": False,
            "match_status": "no_results" if best_candidate is None else "candidate_only",
            "http_status": (best_meta or last_meta or {}).get("http_status"),
            "request_url": (best_meta or last_meta or {}).get("request_url"),
        }

        if best_candidate is not None:
            best_row.update(best_candidate)
            exact_title_match = bool(best_candidate["exact_title_match"])
            near_exact_title_match = bool(best_candidate["near_exact_title_match"])
            auto_match = exact_title_match or (
                args.allow_near_exact_auto_match and near_exact_title_match
            )
            best_row["matched"] = auto_match
            if exact_title_match:
                best_row["match_status"] = "exact_title_match"
            elif auto_match:
                best_row["match_status"] = "near_exact_title_match"
            elif near_exact_title_match:
                best_row["match_status"] = "near_exact_review_needed"
            else:
                best_row["match_status"] = "candidate_only"
        best_rows.append(best_row)

    best_df = pd.DataFrame(best_rows)
    candidate_df = pd.DataFrame(candidate_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv),
        "raw_dir": str(args.raw_dir),
        "n_input_rows": int(len(input_rows)),
        "n_best_rows": int(len(best_df)),
        "n_candidate_rows": int(len(candidate_df)),
        "n_exact_title_matches": int((best_df.get("match_status") == "exact_title_match").sum())
        if not best_df.empty
        else 0,
        "n_near_exact_title_matches": int((best_df.get("match_status") == "near_exact_title_match").sum())
        if not best_df.empty
        else 0,
        "n_review_needed": int((best_df.get("match_status") == "near_exact_review_needed").sum())
        if not best_df.empty
        else 0,
        "n_no_results": int((best_df.get("match_status") == "no_results").sum())
        if not best_df.empty
        else 0,
    }
    return best_df, candidate_df, summary


def write_outputs(
    args: argparse.Namespace,
    input_rows: pd.DataFrame,
    best_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.raw_dir / "query_manifest.csv"
    best_path = args.raw_dir / "arxiv_best_matches.csv"
    candidates_path = args.raw_dir / "arxiv_candidate_matches.csv"
    enriched_path = args.raw_dir / f"{args.input_csv.stem}_with_arxiv_best_match.csv"
    summary_path = args.raw_dir / "arxiv_query_summary.json"

    input_rows.to_csv(manifest_path, index=False)
    best_df.to_csv(best_path, index=False)
    candidate_df.to_csv(candidates_path, index=False)

    enriched_df = input_rows.merge(
        best_df.drop(columns=["input_title", "input_year", "source_id"], errors="ignore"),
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    enriched_df.to_csv(enriched_path, index=False)
    write_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    input_rows = load_input_rows(
        input_csv=args.input_csv,
        paper_id_col=args.paper_id_col,
        title_col=args.title_col,
        year_col=args.year_col,
        source_id_col=args.source_id_col,
        limit=args.limit,
    )
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    input_rows.to_csv(args.raw_dir / "query_manifest.csv", index=False)
    run_queries(args, input_rows)
    best_df, candidate_df, summary = build_outputs(args, input_rows)
    write_outputs(args, input_rows, best_df, candidate_df, summary)


if __name__ == "__main__":
    main()
