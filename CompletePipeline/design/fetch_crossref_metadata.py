#!/usr/bin/env python3
"""
Query Crossref metadata for ICLR papers enriched with OpenReview metadata.

The default workflow targets papers that were still unmatched in the arXiv pass
but now have OpenReview title/author metadata. Queries are cached on disk and
the matcher is conservative: automatic matches require an exact normalized title
match and a nearby publication year.
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
DEFAULT_INPUT_CSV = (
    ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview" / "arxiv_dump_combined_best_matches_with_openreview_yearly_submissions.csv"
)
DEFAULT_RAW_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "Crossref"
CROSSREF_API_URL = "https://api.crossref.org/works"
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class QueryAttempt:
    label: str
    params: dict[str, str]
    cache_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Crossref metadata for OpenReview-enriched ICLR papers."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV. Defaults to the OpenReview-enriched arXiv working file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where raw Crossref query data and derived outputs are written.",
    )
    parser.add_argument(
        "--paper-id-col",
        default="paper_id",
        help="Paper identifier column in the input CSV.",
    )
    parser.add_argument(
        "--title-col",
        default=None,
        help="Fallback title column. Defaults to `openreview_title`, then `input_title`, then `title`.",
    )
    parser.add_argument(
        "--year-col",
        default=None,
        help="Fallback year column. Defaults to `input_year`, then `year`.",
    )
    parser.add_argument(
        "--author-col",
        default=None,
        help="Fallback author column. Defaults to `openreview_authors`, then `arxiv_authors`.",
    )
    parser.add_argument(
        "--source-id-col",
        default="source_id",
        help="Optional source-id column in the input CSV.",
    )
    parser.add_argument(
        "--matched-col",
        default="matched",
        help="Column used to identify previously matched arXiv rows.",
    )
    parser.add_argument(
        "--openreview-id-col",
        default="openreview_note_id",
        help="Column used to require OpenReview metadata for the default query mode.",
    )
    parser.add_argument(
        "--query-mode",
        choices=["arxiv_unmatched_openreview", "openreview_only", "all"],
        default="arxiv_unmatched_openreview",
        help=(
            "Default is to query only papers unmatched in arXiv but already enriched "
            "with OpenReview metadata."
        ),
    )
    parser.add_argument(
        "--rows-per-query",
        type=int,
        default=5,
        help="Number of Crossref candidates returned per query attempt.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help=(
            "Minimum delay between live Crossref requests. The public pool limit is "
            "5 requests per second, so 0.25 stays safely below that."
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
        help="HTTP timeout for each Crossref request.",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=5,
        help="Maximum live request attempts for a single Crossref query before giving up.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=10.0,
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
        default=5,
        help="Trigger a longer cooldown after this many consecutive query-level failures.",
    )
    parser.add_argument(
        "--cooldown-sleep-seconds",
        type=float,
        default=300.0,
        help="Cooldown sleep after a sustained run of failed queries.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print a one-line progress update every N papers during live querying.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached responses and re-query Crossref.",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        help=(
            "Optional email address to send in the `mailto` parameter for Crossref's "
            "polite pool."
        ),
    )
    parser.add_argument(
        "--user-agent",
        default="LLMReview-crossref-match/0.1",
        help="User-Agent header sent to Crossref.",
    )
    return parser.parse_args()


def choose_column(
    df: pd.DataFrame,
    explicit_name: str | None,
    fallbacks: list[str],
    label: str,
) -> str | None:
    if explicit_name:
        if explicit_name not in df.columns:
            raise ValueError(f"{label} column `{explicit_name}` not found in input CSV.")
        return explicit_name
    for name in fallbacks:
        if name in df.columns:
            return name
    return None


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


def safe_paper_id(paper_id: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(paper_id))


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def split_authors(authors_value: object) -> list[str]:
    if authors_value is None:
        return []
    if isinstance(authors_value, float) and math.isnan(authors_value):
        return []
    text = str(authors_value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    if " and " in text.lower():
        return [part.strip() for part in re.split(r"(?i)\s+and\s+", text) if part.strip()]
    return [text]


def author_family_tokens(authors_value: object) -> set[str]:
    families: set[str] = set()
    for author in split_authors(authors_value):
        tokens = re.findall(r"[A-Za-z0-9]+", author.lower())
        if tokens:
            families.add(tokens[-1])
    return families


def first_author_family(authors_value: object) -> str | None:
    authors = split_authors(authors_value)
    if not authors:
        return None
    tokens = re.findall(r"[A-Za-z0-9]+", authors[0].lower())
    return tokens[-1] if tokens else None


def extract_crossref_year(item: dict[str, object]) -> int | None:
    date_fields = [
        "published-print",
        "published-online",
        "published",
        "issued",
        "created",
    ]
    for field in date_fields:
        value = item.get(field)
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts")
        if not isinstance(parts, list) or not parts:
            continue
        first = parts[0]
        if not isinstance(first, list) or not first:
            continue
        year = first[0]
        if isinstance(year, int):
            return year
    return None


def maybe_read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_attempts(
    raw_dir: Path,
    paper_id: str,
    title: str,
    authors: str,
    year: object,
    rows_per_query: int,
    mailto: str | None,
) -> list[QueryAttempt]:
    query_dir = raw_dir / "query_cache"
    query_dir.mkdir(parents=True, exist_ok=True)
    safe_id = safe_paper_id(paper_id)

    author_list = split_authors(authors)
    lead_authors = author_list[:2]
    bibliographic_parts_full = [title]
    if lead_authors:
        bibliographic_parts_full.extend(lead_authors)
    if year is not None and not (isinstance(year, float) and math.isnan(year)):
        bibliographic_parts_full.append(str(int(year)) if isinstance(year, (int, float)) else str(year))
    bibliographic_full = ". ".join(str(part).strip() for part in bibliographic_parts_full if str(part).strip())

    bibliographic_title_only_parts = [title]
    if year is not None and not (isinstance(year, float) and math.isnan(year)):
        bibliographic_title_only_parts.append(str(int(year)) if isinstance(year, (int, float)) else str(year))
    bibliographic_title_only = ". ".join(
        str(part).strip() for part in bibliographic_title_only_parts if str(part).strip()
    )

    common_params = {
        "rows": str(rows_per_query),
    }
    if mailto:
        common_params["mailto"] = mailto

    return [
        QueryAttempt(
            label="bibliographic_full",
            params={**common_params, "query.bibliographic": bibliographic_full},
            cache_path=query_dir / f"{safe_id}__bibliographic_full.json",
        ),
        QueryAttempt(
            label="bibliographic_title_only",
            params={**common_params, "query.bibliographic": bibliographic_title_only},
            cache_path=query_dir / f"{safe_id}__bibliographic_title_only.json",
        ),
    ]


def fetch_attempt(
    attempt: QueryAttempt,
    *,
    refresh: bool,
    timeout_seconds: float,
    retry_max_attempts: int,
    retry_backoff_seconds: float,
    retry_backoff_factor: float,
    retry_max_sleep_seconds: float,
    user_agent: str,
) -> tuple[dict[str, object], bool]:
    if attempt.cache_path.exists() and not refresh:
        cached = maybe_read_json(attempt.cache_path)
        if cached is None:
            raise RuntimeError(f"Failed to read Crossref cache file: {attempt.cache_path}")
        return cached, True

    url = f"{CROSSREF_API_URL}?{urllib.parse.urlencode(attempt.params)}"
    backoff = retry_backoff_seconds
    last_payload: dict[str, object] | None = None
    for attempt_number in range(1, retry_max_attempts + 1):
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
                payload = {
                    "label": attempt.label,
                    "url": url,
                    "params": attempt.params,
                    "fetched_at": now,
                    "attempt_number": attempt_number,
                    "status_code": response.status,
                    "ok": True,
                    "error": None,
                    "response_json": json.loads(body_text),
                }
                write_json(attempt.cache_path, payload)
                return payload, False
        except urllib.error.HTTPError as exc:
            body_bytes = exc.read()
            body_text = body_bytes.decode("utf-8", errors="replace")
            response_json = None
            try:
                response_json = json.loads(body_text)
            except Exception:
                response_json = None
            payload = {
                "label": attempt.label,
                "url": url,
                "params": attempt.params,
                "fetched_at": now,
                "attempt_number": attempt_number,
                "status_code": exc.code,
                "ok": False,
                "error": f"HTTPError {exc.code}",
                "response_json": response_json,
                "response_text": body_text[:4000],
            }
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            payload = {
                "label": attempt.label,
                "url": url,
                "params": attempt.params,
                "fetched_at": now,
                "attempt_number": attempt_number,
                "status_code": None,
                "ok": False,
                "error": repr(exc),
                "response_json": None,
            }

        last_payload = payload
        status_code = payload.get("status_code")
        retryable = status_code in RETRYABLE_HTTP_STATUS_CODES or status_code is None
        if not retryable or attempt_number == retry_max_attempts:
            write_json(attempt.cache_path, payload)
            return payload, False

        time.sleep(backoff)
        backoff = min(backoff * retry_backoff_factor, retry_max_sleep_seconds)

    if last_payload is None:
        raise RuntimeError(f"Crossref fetch failed before any attempts for {attempt.label}.")
    write_json(attempt.cache_path, last_payload)
    return last_payload, False


def flatten_candidates(
    row: dict[str, object],
    query_payload: dict[str, object],
    query_label: str,
) -> list[dict[str, object]]:
    response_json = query_payload.get("response_json")
    if not isinstance(response_json, dict):
        return []
    message = response_json.get("message")
    if not isinstance(message, dict):
        return []
    items = message.get("items")
    if not isinstance(items, list):
        return []

    input_title = row.get("query_title")
    input_title_norm = normalize_title(input_title)
    input_year = row.get("input_year")
    input_author_families = author_family_tokens(row.get("query_authors"))

    candidates: list[dict[str, object]] = []
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        titles = item.get("title")
        crossref_title = None
        if isinstance(titles, list) and titles:
            crossref_title = titles[0]
        elif isinstance(titles, str):
            crossref_title = titles
        crossref_title_norm = normalize_title(crossref_title)
        similarity = title_similarity(input_title_norm, crossref_title_norm)
        jaccard = token_jaccard(input_title_norm, crossref_title_norm)
        exact_title_match = bool(input_title_norm) and input_title_norm == crossref_title_norm
        near_exact_title_match = similarity >= 0.97 and jaccard >= 0.90

        crossref_year = extract_crossref_year(item)
        year_diff = None
        if input_year is not None and crossref_year is not None:
            try:
                year_diff = abs(int(float(input_year)) - int(crossref_year))
            except Exception:
                year_diff = None

        authors = item.get("author")
        crossref_authors = []
        if isinstance(authors, list):
            for author in authors:
                if not isinstance(author, dict):
                    continue
                given = str(author.get("given") or "").strip()
                family = str(author.get("family") or "").strip()
                full_name = " ".join(part for part in [given, family] if part)
                if full_name:
                    crossref_authors.append(full_name)
        crossref_author_families = author_family_tokens(crossref_authors)
        shared_author_families = sorted(input_author_families & crossref_author_families)
        first_author_family_match = False
        if input_author_families and crossref_author_families:
            input_first = first_author_family(row.get("query_authors"))
            crossref_first = first_author_family(crossref_authors)
            first_author_family_match = bool(input_first and crossref_first and input_first == crossref_first)

        container_titles = item.get("container-title")
        container_title = None
        if isinstance(container_titles, list) and container_titles:
            container_title = container_titles[0]
        elif isinstance(container_titles, str):
            container_title = container_titles

        api_score = item.get("score")
        candidate_score = (
            (5.0 if exact_title_match else 0.0)
            + (1.5 if near_exact_title_match else 0.0)
            + (0.75 if first_author_family_match else 0.0)
            + (0.25 * len(shared_author_families))
            + (0.25 if year_diff == 0 else 0.10 if year_diff == 1 else 0.0)
            + (float(api_score) / 100.0 if isinstance(api_score, (int, float)) else 0.0)
        )

        candidates.append(
            {
                "paper_id": row.get("paper_id"),
                "input_title": row.get("input_title"),
                "input_year": row.get("input_year"),
                "source_id": row.get("source_id"),
                "query_title": row.get("query_title"),
                "query_authors": row.get("query_authors"),
                "query_label": query_label,
                "query_status_code": query_payload.get("status_code"),
                "crossref_rank": rank,
                "crossref_doi": item.get("DOI"),
                "crossref_title": crossref_title,
                "crossref_title_norm": crossref_title_norm,
                "crossref_authors": "; ".join(crossref_authors) if crossref_authors else None,
                "crossref_authors_json": json.dumps(crossref_authors, ensure_ascii=False)
                if crossref_authors
                else None,
                "crossref_type": item.get("type"),
                "crossref_container_title": container_title,
                "crossref_url": item.get("URL"),
                "crossref_is_referenced_by_count": item.get("is-referenced-by-count"),
                "crossref_api_score": api_score,
                "crossref_published_year": crossref_year,
                "year_diff": year_diff,
                "title_similarity": similarity,
                "token_jaccard": jaccard,
                "exact_title_match": exact_title_match,
                "near_exact_title_match": near_exact_title_match,
                "first_author_family_match": first_author_family_match,
                "shared_author_families_json": json.dumps(shared_author_families, ensure_ascii=False)
                if shared_author_families
                else None,
                "candidate_score": candidate_score,
                "raw_query_path": str(query_payload.get("cache_path") or ""),
            }
        )
    return candidates


def author_family_tokens_from_list(authors_list: list[object]) -> set[str]:
    families: set[str] = set()
    for author in authors_list:
        tokens = re.findall(r"[A-Za-z0-9]+", str(author).lower())
        if tokens:
            families.add(tokens[-1])
    return families


def select_best_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None

    exact = candidates.loc[
        candidates["exact_title_match"].fillna(False)
        & (
            candidates["year_diff"].isna()
            | (candidates["year_diff"].fillna(999) <= 2)
        )
    ].copy()
    if not exact.empty:
        exact = exact.sort_values(
            by=[
                "first_author_family_match",
                "candidate_score",
                "crossref_is_referenced_by_count",
                "crossref_api_score",
            ],
            ascending=[False, False, False, False],
        )
        best = exact.iloc[0].copy()
        best["match_status"] = "exact_title_match"
        best["matched"] = True
        best["best_query_attempt"] = best["query_label"]
        return best

    near = candidates.loc[
        candidates["near_exact_title_match"].fillna(False)
        & candidates["first_author_family_match"].fillna(False)
        & (
            candidates["year_diff"].isna()
            | (candidates["year_diff"].fillna(999) <= 2)
        )
    ].copy()
    if not near.empty:
        near = near.sort_values(
            by=["candidate_score", "crossref_is_referenced_by_count", "crossref_api_score"],
            ascending=[False, False, False],
        )
        best = near.iloc[0].copy()
        best["match_status"] = "near_exact_review_candidate"
        best["matched"] = False
        best["best_query_attempt"] = best["query_label"]
        return best

    fallback = candidates.sort_values(
        by=["candidate_score", "crossref_api_score"], ascending=[False, False]
    ).iloc[0].copy()
    fallback["match_status"] = "no_confident_match"
    fallback["matched"] = False
    fallback["best_query_attempt"] = fallback["query_label"]
    return fallback


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(args.input_csv, low_memory=False)
    paper_id_col = choose_column(input_df, args.paper_id_col, ["paper_id"], "paper_id")
    if paper_id_col is None:
        raise ValueError("No paper id column found in input CSV.")
    title_col = choose_column(
        input_df,
        args.title_col,
        ["openreview_title", "input_title", "title"],
        "title",
    )
    year_col = choose_column(input_df, args.year_col, ["input_year", "year"], "year")
    author_col = choose_column(
        input_df,
        args.author_col,
        ["openreview_authors", "arxiv_authors"],
        "author",
    )
    source_id_col = args.source_id_col if args.source_id_col in input_df.columns else None
    matched_col = args.matched_col if args.matched_col in input_df.columns else None
    openreview_id_col = args.openreview_id_col if args.openreview_id_col in input_df.columns else None

    selected_df = input_df.copy()
    if args.query_mode == "arxiv_unmatched_openreview":
        if matched_col is not None:
            selected_df = selected_df.loc[~selected_df[matched_col].fillna(False).map(to_bool)].copy()
        if openreview_id_col is not None:
            selected_df = selected_df.loc[selected_df[openreview_id_col].notna()].copy()
    elif args.query_mode == "openreview_only":
        if openreview_id_col is not None:
            selected_df = selected_df.loc[selected_df[openreview_id_col].notna()].copy()

    selected_df = selected_df.loc[selected_df[paper_id_col].notna()].copy()
    selected_df[paper_id_col] = selected_df[paper_id_col].astype(str).str.strip()
    selected_df = selected_df.loc[selected_df[paper_id_col] != ""].copy()

    if args.limit is not None:
        selected_df = selected_df.head(args.limit).copy()

    query_rows = []
    for row in selected_df.to_dict(orient="records"):
        query_rows.append(
            {
                "paper_id": row.get(paper_id_col),
                "input_title": row.get("input_title") if "input_title" in row else row.get(title_col),
                "input_year": row.get(year_col) if year_col else None,
                "source_id": row.get(source_id_col) if source_id_col else None,
                "query_title": row.get(title_col) if title_col else None,
                "query_authors": row.get(author_col) if author_col else None,
            }
        )
    query_df = pd.DataFrame(query_rows)
    if not query_df.empty:
        query_df["query_title"] = query_df["query_title"].fillna(query_df["input_title"])
        query_df["query_authors"] = query_df["query_authors"].fillna("")
        query_df["query_title_norm"] = query_df["query_title"].map(normalize_title)
        query_df = query_df.loc[query_df["query_title_norm"] != ""].copy()

    print(
        f"Selected {len(query_df):,} query rows from {len(input_df):,} input rows "
        f"(mode={args.query_mode})."
    )

    candidate_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    consecutive_failures = 0

    for idx, row in enumerate(query_df.to_dict(orient="records"), start=1):
        attempts = build_attempts(
            args.raw_dir,
            str(row["paper_id"]),
            str(row.get("query_title") or ""),
            str(row.get("query_authors") or ""),
            row.get("input_year"),
            args.rows_per_query,
            args.mailto,
        )

        row_candidates: list[dict[str, object]] = []
        row_manifest = {
            "paper_id": row.get("paper_id"),
            "input_title": row.get("input_title"),
            "input_year": row.get("input_year"),
            "source_id": row.get("source_id"),
            "query_title": row.get("query_title"),
            "query_authors": row.get("query_authors"),
        }
        row_had_success = False

        for attempt in attempts:
            payload, from_cache = fetch_attempt(
                attempt,
                refresh=args.refresh,
                timeout_seconds=args.timeout_seconds,
                retry_max_attempts=args.retry_max_attempts,
                retry_backoff_seconds=args.retry_backoff_seconds,
                retry_backoff_factor=args.retry_backoff_factor,
                retry_max_sleep_seconds=args.retry_max_sleep_seconds,
                user_agent=args.user_agent,
            )
            payload["cache_path"] = str(attempt.cache_path.resolve())
            row_manifest[f"{attempt.label}_status_code"] = payload.get("status_code")
            row_manifest[f"{attempt.label}_ok"] = payload.get("ok")
            row_manifest[f"{attempt.label}_from_cache"] = from_cache
            row_manifest[f"{attempt.label}_cache_path"] = str(attempt.cache_path.resolve())

            if payload.get("ok"):
                row_had_success = True
                row_candidates.extend(flatten_candidates(row, payload, attempt.label))

            if args.sleep_seconds > 0 and not from_cache:
                time.sleep(args.sleep_seconds)

        manifest_rows.append(row_manifest)
        candidate_rows.extend(row_candidates)

        if row_had_success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= args.cooldown_after_consecutive_failures:
                time.sleep(args.cooldown_sleep_seconds)
                consecutive_failures = 0

        row_candidates_df = pd.DataFrame(row_candidates)
        best = select_best_candidate(row_candidates_df)
        if best is None:
            best_rows.append(
                {
                    "paper_id": row.get("paper_id"),
                    "input_title": row.get("input_title"),
                    "input_year": row.get("input_year"),
                    "source_id": row.get("source_id"),
                    "query_title": row.get("query_title"),
                    "query_authors": row.get("query_authors"),
                    "matched": False,
                    "match_status": "no_candidates",
                    "best_query_attempt": None,
                }
            )
        else:
            best_rows.append(best.to_dict())

        if idx % args.progress_every == 0 or idx == len(query_df):
            matched_count = sum(1 for best_row in best_rows if best_row.get("matched"))
            print(f"[{idx:,}/{len(query_df):,}] Crossref automatic matches for {matched_count:,} papers.")

    candidate_df = pd.DataFrame(candidate_rows)
    best_df = pd.DataFrame(best_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    candidate_path = args.raw_dir / "crossref_candidate_matches.csv"
    best_path = args.raw_dir / "crossref_best_matches.csv"
    manifest_path = args.raw_dir / "query_manifest.csv"
    joined_path = args.raw_dir / f"{args.input_csv.stem}_with_crossref_best_match.csv"
    summary_path = args.raw_dir / "crossref_query_summary.json"

    candidate_df.to_csv(candidate_path, index=False)
    best_df.to_csv(best_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    merge_cols = [col for col in best_df.columns if col not in {"query_title", "query_authors"}]
    joined_df = input_df.merge(best_df[merge_cols], on=["paper_id"], how="left")
    joined_df.to_csv(joined_path, index=False)

    summary = {
        "input_csv": str(args.input_csv.resolve()),
        "raw_dir": str(args.raw_dir.resolve()),
        "query_mode": args.query_mode,
        "input_rows_total": int(len(input_df)),
        "rows_selected_for_query": int(len(query_df)),
        "query_rows_with_candidates": int(best_df["match_status"].ne("no_candidates").sum()) if len(best_df) else 0,
        "matched_exact_title": int(best_df["matched"].fillna(False).sum()) if len(best_df) else 0,
        "review_candidates": int((best_df["match_status"] == "near_exact_review_candidate").sum()) if len(best_df) else 0,
        "no_confident_match": int((best_df["match_status"] == "no_confident_match").sum()) if len(best_df) else 0,
        "no_candidates": int((best_df["match_status"] == "no_candidates").sum()) if len(best_df) else 0,
        "matched_with_doi": int(best_df.loc[best_df["matched"].fillna(False), "crossref_doi"].notna().sum())
        if len(best_df)
        else 0,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote candidate matches to {candidate_path}")
    print(f"Wrote best matches to {best_path}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote joined output to {joined_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
