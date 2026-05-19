#!/usr/bin/env python3
"""
Benchmark OpenAlex and Crossref on a stratified sample of ICLR papers.

The default benchmark focuses on the papers that were unmatched in the arXiv
pass but have OpenReview metadata. It samples rows across years, queries both
APIs with one request per paper, and records:

- raw query latency
- whether the top local re-ranked candidate is an automatic match
- whether the matched candidate carries a DOI
- whether OpenAlex returns citation counts
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
DEFAULT_RAW_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "CitationBenchmark"
CROSSREF_API_URL = "https://api.crossref.org/works"
OPENALEX_API_URL = "https://api.openalex.org/works"
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class ApiResult:
    api_name: str
    status_code: int | None
    ok: bool
    error: str | None
    elapsed_seconds: float
    response_json: dict[str, object] | None
    cache_path: Path
    from_cache: bool
    request_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAlex and Crossref on a stratified sample of ICLR papers."
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
        help="Directory where benchmark outputs and raw query caches are written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of sampled papers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--query-mode",
        choices=["arxiv_unmatched_openreview", "openreview_only", "all"],
        default="arxiv_unmatched_openreview",
        help="Default is to benchmark on the arXiv-unmatched OpenReview-covered set.",
    )
    parser.add_argument(
        "--crossref-rows",
        type=int,
        default=5,
        help="Crossref candidates to request.",
    )
    parser.add_argument(
        "--openalex-per-page",
        type=int,
        default=5,
        help="OpenAlex candidates to request.",
    )
    parser.add_argument(
        "--openalex-key-path",
        type=Path,
        default=ROOT / "OpenAlex.txt",
        help="Path to the OpenAlex API key file.",
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
        default=4,
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
        "--crossref-sleep-seconds",
        type=float,
        default=0.25,
        help="Inter-request sleep for Crossref.",
    )
    parser.add_argument(
        "--openalex-sleep-seconds",
        type=float,
        default=0.15,
        help="Inter-request sleep for OpenAlex.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached API responses and re-query.",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        help="Optional email to send to Crossref via `mailto`.",
    )
    parser.add_argument(
        "--user-agent",
        default="LLMReview-citation-benchmark/0.1",
        help="User-Agent header.",
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


def extract_crossref_year(item: dict[str, object]) -> int | None:
    date_fields = ["published-print", "published-online", "published", "issued", "created"]
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


def extract_openalex_year(item: dict[str, object]) -> int | None:
    year = item.get("publication_year")
    return int(year) if isinstance(year, int) else None


def maybe_read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def choose_column(df: pd.DataFrame, fallbacks: list[str], label: str) -> str:
    for name in fallbacks:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find {label} column in input among {fallbacks}.")


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()

    counts = df["input_year"].value_counts().sort_index()
    total = len(df)
    base_alloc = {year: 0 for year in counts.index.tolist()}
    remainders = []
    for year, count in counts.items():
        quota = sample_size * count / total
        take = min(int(math.floor(quota)), count)
        base_alloc[year] = take
        remainders.append((quota - take, year))
    allocated = sum(base_alloc.values())
    remaining = sample_size - allocated
    remainders.sort(reverse=True)
    for _, year in remainders:
        if remaining <= 0:
            break
        if base_alloc[year] < counts[year]:
            base_alloc[year] += 1
            remaining -= 1

    sampled_frames = []
    for offset, (year, take) in enumerate(sorted(base_alloc.items())):
        if take <= 0:
            continue
        year_df = df.loc[df["input_year"] == year].copy()
        sampled_frames.append(year_df.sample(n=take, random_state=seed + offset))
    sampled = pd.concat(sampled_frames, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def perform_json_request(
    api_name: str,
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
            api_name=api_name,
            status_code=cached.get("status_code"),
            ok=bool(cached.get("ok")),
            error=cached.get("error"),
            elapsed_seconds=float(cached.get("elapsed_seconds") or 0.0),
            response_json=cached.get("response_json"),
            cache_path=cache_path.resolve(),
            from_cache=True,
            request_url=url,
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
                    "api_name": api_name,
                    "url": url,
                    "fetched_at": now,
                    "attempt_number": attempt_number,
                    "status_code": response.status,
                    "ok": True,
                    "error": None,
                    "elapsed_seconds": elapsed,
                    "response_json": json.loads(body_text),
                }
                write_json(cache_path, payload)
                return ApiResult(
                    api_name=api_name,
                    status_code=response.status,
                    ok=True,
                    error=None,
                    elapsed_seconds=elapsed,
                    response_json=payload["response_json"],
                    cache_path=cache_path.resolve(),
                    from_cache=False,
                    request_url=url,
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
                "api_name": api_name,
                "url": url,
                "fetched_at": now,
                "attempt_number": attempt_number,
                "status_code": exc.code,
                "ok": False,
                "error": f"HTTPError {exc.code}",
                "elapsed_seconds": elapsed,
                "response_json": response_json,
                "response_text": body_text[:4000],
            }
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            elapsed = time.perf_counter() - started
            payload = {
                "api_name": api_name,
                "url": url,
                "fetched_at": now,
                "attempt_number": attempt_number,
                "status_code": None,
                "ok": False,
                "error": repr(exc),
                "elapsed_seconds": elapsed,
                "response_json": None,
            }

        last_payload = payload
        status_code = payload.get("status_code")
        retryable = status_code in RETRYABLE_HTTP_STATUS_CODES or status_code is None
        if not retryable or attempt_number == retry_max_attempts:
            write_json(cache_path, payload)
            return ApiResult(
                api_name=api_name,
                status_code=payload.get("status_code"),
                ok=False,
                error=payload.get("error"),
                elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
                response_json=payload.get("response_json"),
                cache_path=cache_path.resolve(),
                from_cache=False,
                request_url=url,
            )
        time.sleep(backoff)
        backoff = min(backoff * retry_backoff_factor, retry_max_sleep_seconds)

    raise RuntimeError(f"{api_name} request failed without returning a payload: {url}")


def build_crossref_url(row: dict[str, object], rows: int, mailto: str | None) -> str:
    params = {
        "query.title": str(row.get("query_title") or ""),
        "rows": str(rows),
    }
    lead_authors = split_authors(row.get("query_authors"))[:1]
    if lead_authors:
        params["query.author"] = lead_authors[0]
    if row.get("input_year") is not None and not (isinstance(row.get("input_year"), float) and math.isnan(row.get("input_year"))):
        year = int(float(row["input_year"]))
        params["filter"] = f"from-pub-date:{year-1}-01-01,until-pub-date:{year+1}-12-31"
    if mailto:
        params["mailto"] = mailto
    return f"{CROSSREF_API_URL}?{urllib.parse.urlencode(params)}"


def build_openalex_url(row: dict[str, object], per_page: int, api_key: str) -> str:
    year_value = row.get("input_year")
    year_filter = None
    if year_value is not None and not (isinstance(year_value, float) and math.isnan(year_value)):
        year = int(float(year_value))
        year_filter = f"{year-1}-{year+1}"
    filters = [f"title.search:{row.get('query_title') or ''}"]
    if year_filter:
        filters.append(f"publication_year:{year_filter}")
    params = {
        "api_key": api_key,
        "filter": ",".join(filters),
        "per-page": str(per_page),
        "select": "id,doi,display_name,publication_year,cited_by_count,ids,authorships,type,primary_location",
    }
    return f"{OPENALEX_API_URL}?{urllib.parse.urlencode(params)}"


def rank_crossref_candidates(row: dict[str, object], response_json: dict[str, object] | None) -> pd.DataFrame:
    if not isinstance(response_json, dict):
        return pd.DataFrame()
    message = response_json.get("message")
    if not isinstance(message, dict):
        return pd.DataFrame()
    items = message.get("items")
    if not isinstance(items, list):
        return pd.DataFrame()

    input_title_norm = normalize_title(row.get("query_title"))
    input_author_families = author_family_tokens(row.get("query_authors"))
    input_first_author = first_author_family(row.get("query_authors"))
    input_year = row.get("input_year")

    records = []
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        titles = item.get("title")
        candidate_title = titles[0] if isinstance(titles, list) and titles else titles if isinstance(titles, str) else None
        candidate_title_norm = normalize_title(candidate_title)
        similarity = title_similarity(input_title_norm, candidate_title_norm)
        jaccard = token_jaccard(input_title_norm, candidate_title_norm)
        exact = bool(input_title_norm) and input_title_norm == candidate_title_norm
        near = similarity >= 0.97 and jaccard >= 0.90

        authors = []
        for author in item.get("author") or []:
            if not isinstance(author, dict):
                continue
            name = " ".join(part for part in [str(author.get("given") or "").strip(), str(author.get("family") or "").strip()] if part)
            if name:
                authors.append(name)
        candidate_author_families = author_family_tokens(authors)
        candidate_first_author = first_author_family(authors)
        shared_families = sorted(input_author_families & candidate_author_families)
        first_author_match = bool(input_first_author and candidate_first_author and input_first_author == candidate_first_author)

        candidate_year = extract_crossref_year(item)
        year_diff = None
        if candidate_year is not None and input_year is not None and not (isinstance(input_year, float) and math.isnan(input_year)):
            year_diff = abs(candidate_year - int(float(input_year)))

        score = (
            (5.0 if exact else 0.0)
            + (1.5 if near else 0.0)
            + (0.75 if first_author_match else 0.0)
            + (0.25 * len(shared_families))
            + (0.25 if year_diff == 0 else 0.10 if year_diff == 1 else 0.0)
            + (float(item.get("score")) / 100.0 if isinstance(item.get("score"), (int, float)) else 0.0)
        )
        records.append(
            {
                "rank": rank,
                "doi": item.get("DOI"),
                "display_name": candidate_title,
                "publication_year": candidate_year,
                "cited_by_count": item.get("is-referenced-by-count"),
                "type": item.get("type"),
                "title_similarity": similarity,
                "token_jaccard": jaccard,
                "exact_title_match": exact,
                "near_exact_title_match": near,
                "first_author_family_match": first_author_match,
                "year_diff": year_diff,
                "shared_author_families_json": json.dumps(shared_families, ensure_ascii=False) if shared_families else None,
                "candidate_score": score,
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values(
            by=["exact_title_match", "first_author_family_match", "candidate_score", "cited_by_count"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    return frame


def rank_openalex_candidates(row: dict[str, object], response_json: dict[str, object] | None) -> pd.DataFrame:
    if not isinstance(response_json, dict):
        return pd.DataFrame()
    items = response_json.get("results")
    if not isinstance(items, list):
        return pd.DataFrame()

    input_title_norm = normalize_title(row.get("query_title"))
    input_author_families = author_family_tokens(row.get("query_authors"))
    input_first_author = first_author_family(row.get("query_authors"))
    input_year = row.get("input_year")

    records = []
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        candidate_title = item.get("display_name")
        candidate_title_norm = normalize_title(candidate_title)
        similarity = title_similarity(input_title_norm, candidate_title_norm)
        jaccard = token_jaccard(input_title_norm, candidate_title_norm)
        exact = bool(input_title_norm) and input_title_norm == candidate_title_norm
        near = similarity >= 0.97 and jaccard >= 0.90

        authors = []
        for authorship in item.get("authorships") or []:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author")
            if isinstance(author, dict):
                name = author.get("display_name")
                if name:
                    authors.append(str(name))
        candidate_author_families = author_family_tokens(authors)
        candidate_first_author = first_author_family(authors)
        shared_families = sorted(input_author_families & candidate_author_families)
        first_author_match = bool(input_first_author and candidate_first_author and input_first_author == candidate_first_author)

        candidate_year = extract_openalex_year(item)
        year_diff = None
        if candidate_year is not None and input_year is not None and not (isinstance(input_year, float) and math.isnan(input_year)):
            year_diff = abs(candidate_year - int(float(input_year)))

        score = (
            (5.0 if exact else 0.0)
            + (1.5 if near else 0.0)
            + (0.75 if first_author_match else 0.0)
            + (0.25 * len(shared_families))
            + (0.25 if year_diff == 0 else 0.10 if year_diff == 1 else 0.0)
            + (math.log1p(float(item.get("cited_by_count") or 0)) / 10.0)
        )
        records.append(
            {
                "rank": rank,
                "id": item.get("id"),
                "doi": item.get("doi"),
                "display_name": candidate_title,
                "publication_year": candidate_year,
                "cited_by_count": item.get("cited_by_count"),
                "type": item.get("type"),
                "title_similarity": similarity,
                "token_jaccard": jaccard,
                "exact_title_match": exact,
                "near_exact_title_match": near,
                "first_author_family_match": first_author_match,
                "year_diff": year_diff,
                "shared_author_families_json": json.dumps(shared_families, ensure_ascii=False) if shared_families else None,
                "candidate_score": score,
                "primary_location_json": json.dumps(item.get("primary_location"), ensure_ascii=False)
                if item.get("primary_location") is not None
                else None,
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values(
            by=["exact_title_match", "first_author_family_match", "candidate_score", "cited_by_count"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    return frame


def classify_top_candidate(frame: pd.DataFrame) -> tuple[bool, str, dict[str, object] | None]:
    if frame.empty:
        return False, "no_candidates", None
    top = frame.iloc[0].to_dict()
    year_diff = top.get("year_diff")
    exact = bool(top.get("exact_title_match"))
    near = bool(top.get("near_exact_title_match"))
    first_author = bool(top.get("first_author_family_match"))
    year_ok = year_diff is None or (isinstance(year_diff, (int, float)) and year_diff <= 2)
    if exact and year_ok:
        return True, "exact_title_match", top
    if near and first_author and year_ok:
        return False, "near_exact_review_candidate", top
    return False, "no_confident_match", top


def build_sample(input_df: pd.DataFrame, query_mode: str, sample_size: int, seed: int) -> pd.DataFrame:
    df = input_df.copy()
    if query_mode == "arxiv_unmatched_openreview":
        df = df.loc[~df["matched"].fillna(False).map(to_bool) & df["openreview_note_id"].notna()].copy()
    elif query_mode == "openreview_only":
        df = df.loc[df["openreview_note_id"].notna()].copy()
    df["input_year"] = df["input_year"].astype(int)
    df["query_title"] = df["openreview_title"].fillna(df["input_title"])
    df["query_authors"] = df["openreview_authors"].fillna("")
    df = df.loc[df["query_title"].notna() & (df["query_title"].astype(str).str.strip() != "")].copy()
    sample_cols = ["paper_id", "input_year", "input_title", "source_id", "query_title", "query_authors"]
    sample_df = stratified_sample(df[sample_cols], sample_size=sample_size, seed=seed)
    return sample_df.reset_index(drop=True)


def summarize_api(results_df: pd.DataFrame, api_name: str) -> dict[str, object]:
    api_df = results_df.loc[results_df["api_name"] == api_name].copy()
    if api_df.empty:
        return {"api_name": api_name, "n": 0}
    latency = api_df["elapsed_seconds"]
    return {
        "api_name": api_name,
        "n": int(len(api_df)),
        "status_200": int((api_df["status_code"] == 200).sum()),
        "ok": int(api_df["ok"].fillna(False).sum()),
        "matched": int(api_df["matched"].fillna(False).sum()),
        "matched_rate": float(api_df["matched"].fillna(False).mean()),
        "top_doi_nonnull": int(api_df["top_doi"].notna().sum()),
        "top_cited_by_nonnull": int(api_df["top_cited_by_count"].notna().sum()),
        "mean_elapsed_seconds": float(latency.mean()),
        "median_elapsed_seconds": float(latency.median()),
        "p90_elapsed_seconds": float(latency.quantile(0.9)),
        "from_cache_count": int(api_df["from_cache"].fillna(False).sum()),
    }


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    openalex_key = args.openalex_key_path.read_text(encoding="utf-8").strip()
    if not openalex_key:
        raise ValueError(f"OpenAlex key file is empty: {args.openalex_key_path}")

    input_df = pd.read_csv(args.input_csv, low_memory=False)
    sample_df = build_sample(input_df, args.query_mode, args.sample_size, args.seed)

    sample_path = args.raw_dir / "sampled_papers.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Sampled {len(sample_df):,} papers to {sample_path}")

    comparison_rows = []
    candidate_rows = []
    raw_dir_crossref = args.raw_dir / "crossref_cache"
    raw_dir_openalex = args.raw_dir / "openalex_cache"

    total_start = time.perf_counter()
    for idx, row in enumerate(sample_df.to_dict(orient="records"), start=1):
        paper_id = str(row["paper_id"])
        safe_id = safe_paper_id(paper_id)

        crossref_url = build_crossref_url(row, rows=args.crossref_rows, mailto=args.mailto)
        crossref_cache = raw_dir_crossref / f"{safe_id}.json"
        crossref_result = perform_json_request(
            "crossref",
            crossref_url,
            crossref_cache,
            refresh=args.refresh,
            timeout_seconds=args.timeout_seconds,
            retry_max_attempts=args.retry_max_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            retry_backoff_factor=args.retry_backoff_factor,
            retry_max_sleep_seconds=args.retry_max_sleep_seconds,
            user_agent=args.user_agent,
        )
        crossref_candidates = rank_crossref_candidates(row, crossref_result.response_json)
        crossref_matched, crossref_status, crossref_top = classify_top_candidate(crossref_candidates)
        if not crossref_result.from_cache and args.crossref_sleep_seconds > 0:
            time.sleep(args.crossref_sleep_seconds)

        openalex_url = build_openalex_url(row, per_page=args.openalex_per_page, api_key=openalex_key)
        openalex_cache = raw_dir_openalex / f"{safe_id}.json"
        openalex_result = perform_json_request(
            "openalex",
            openalex_url,
            openalex_cache,
            refresh=args.refresh,
            timeout_seconds=args.timeout_seconds,
            retry_max_attempts=args.retry_max_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            retry_backoff_factor=args.retry_backoff_factor,
            retry_max_sleep_seconds=args.retry_max_sleep_seconds,
            user_agent=args.user_agent,
        )
        openalex_candidates = rank_openalex_candidates(row, openalex_result.response_json)
        openalex_matched, openalex_status, openalex_top = classify_top_candidate(openalex_candidates)
        if not openalex_result.from_cache and args.openalex_sleep_seconds > 0:
            time.sleep(args.openalex_sleep_seconds)

        for api_name, result, status, matched, top, frame in [
            ("crossref", crossref_result, crossref_status, crossref_matched, crossref_top, crossref_candidates),
            ("openalex", openalex_result, openalex_status, openalex_matched, openalex_top, openalex_candidates),
        ]:
            comparison_rows.append(
                {
                    "paper_id": row["paper_id"],
                    "input_year": row["input_year"],
                    "input_title": row["input_title"],
                    "query_title": row["query_title"],
                    "query_authors": row["query_authors"],
                    "api_name": api_name,
                    "status_code": result.status_code,
                    "ok": result.ok,
                    "error": result.error,
                    "elapsed_seconds": result.elapsed_seconds,
                    "from_cache": result.from_cache,
                    "request_url": result.request_url,
                    "cache_path": str(result.cache_path),
                    "n_candidates": int(len(frame)),
                    "match_status": status,
                    "matched": matched,
                    "top_display_name": top.get("display_name") if top else None,
                    "top_doi": top.get("doi") if top else None,
                    "top_publication_year": top.get("publication_year") if top else None,
                    "top_cited_by_count": top.get("cited_by_count") if top else None,
                    "top_type": top.get("type") if top else None,
                    "top_title_similarity": top.get("title_similarity") if top else None,
                    "top_token_jaccard": top.get("token_jaccard") if top else None,
                    "top_first_author_family_match": top.get("first_author_family_match") if top else None,
                    "top_year_diff": top.get("year_diff") if top else None,
                    "top_candidate_score": top.get("candidate_score") if top else None,
                }
            )
            if not frame.empty:
                temp = frame.copy()
                temp.insert(0, "api_name", api_name)
                temp.insert(0, "paper_id", row["paper_id"])
                temp.insert(1, "input_year", row["input_year"])
                temp.insert(2, "input_title", row["input_title"])
                candidate_rows.extend(temp.to_dict(orient="records"))

        if idx % 10 == 0 or idx == len(sample_df):
            print(f"[{idx:,}/{len(sample_df):,}] completed both API queries.")

    total_elapsed = time.perf_counter() - total_start

    comparison_df = pd.DataFrame(comparison_rows)
    candidates_df = pd.DataFrame(candidate_rows)

    comparison_path = args.raw_dir / "benchmark_results.csv"
    candidates_path = args.raw_dir / "benchmark_candidates.csv"
    summary_path = args.raw_dir / "benchmark_summary.json"

    comparison_df.to_csv(comparison_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    summary = {
        "input_csv": str(args.input_csv.resolve()),
        "sample_path": str(sample_path.resolve()),
        "sample_size": int(len(sample_df)),
        "query_mode": args.query_mode,
        "seed": args.seed,
        "total_elapsed_seconds": total_elapsed,
        "crossref": summarize_api(comparison_df, "crossref"),
        "openalex": summarize_api(comparison_df, "openalex"),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote benchmark results to {comparison_path}")
    print(f"Wrote candidate details to {candidates_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
