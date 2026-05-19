#!/usr/bin/env python3
"""
Fetch public OpenReview metadata for ICLR papers and cache raw API responses.

In this environment, direct unauthenticated HTTP requests to OpenReview can
return 403 even for public notes. The same public JSON endpoints are available
from a real browser session, so this script uses Playwright to open OpenReview
once and then issues API fetches from page context.

The default input is the combined arXiv match file and, by default, the script
only queries rows still unmatched in the arXiv pass. That keeps the first Open-
Review pass focused on the papers where public metadata is most likely to help
with later DOI / citation matching.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_INPUT_CSV = ROOT / "OutputNew" / "rawdata" / "Design" / "arXiv" / "arxiv_dump_combined_best_matches.csv"
DEFAULT_RAW_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "OpenReview"
OPENREVIEW_BASE_URL = "https://openreview.net"
OPENREVIEW_NOTES_DETAILS = "replyCount,writable,revisions,original,overwriting,invitation,tags"
RETRYABLE_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504}

JS_FETCH_JSON = """
async ({ url, timeoutMs }) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      credentials: 'include',
      signal: controller.signal,
    });
    const text = await resp.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch (err) {
      body = null;
    }
    return {
      ok: resp.ok,
      status: resp.status,
      body,
      text,
      error: null,
    };
  } catch (err) {
    return {
      ok: false,
      status: null,
      body: null,
      text: null,
      error: String(err && err.message ? err.message : err),
    };
  } finally {
    clearTimeout(timer);
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch public OpenReview note metadata via browser-context API calls."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=(
            "Input CSV. Defaults to the combined arXiv match file so the OpenReview "
            "pass can focus on the arXiv-unmatched papers."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where raw OpenReview API responses and derived outputs are written.",
    )
    parser.add_argument(
        "--paper-id-col",
        default="paper_id",
        help="Paper identifier column in the input CSV.",
    )
    parser.add_argument(
        "--title-col",
        default=None,
        help="Title column. Defaults to `input_title`, falling back to `title`.",
    )
    parser.add_argument(
        "--year-col",
        default=None,
        help="Year column. Defaults to `input_year`, falling back to `year`.",
    )
    parser.add_argument(
        "--source-id-col",
        default="source_id",
        help="Optional source identifier column in the input CSV.",
    )
    parser.add_argument(
        "--matched-col",
        default="matched",
        help="Column used to identify previously matched arXiv rows.",
    )
    parser.add_argument(
        "--query-mode",
        choices=["arxiv_unmatched", "all"],
        default="arxiv_unmatched",
        help=(
            "Query only rows still unmatched in the arXiv pass, or query every row "
            "in the input."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of unique paper ids to process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.35,
        help="Minimum delay between live OpenReview API fetches.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request timeout for browser-context fetches.",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=5,
        help="Maximum attempts for a single OpenReview endpoint query.",
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
        help="Maximum sleep after a retryable failure.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print a progress update every N papers.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached OpenReview responses and re-query the API.",
    )
    parser.add_argument(
        "--fetch-thread",
        action="store_true",
        help=(
            "Also fetch the full public thread via `notes?forum=...`. This is larger "
            "and slower, so it is off by default."
        ),
    )
    parser.add_argument(
        "--fetch-invitations",
        action="store_true",
        help=(
            "Also fetch `invitations?replyForum=...` to see what public reply types "
            "exist for each paper."
        ),
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch Chromium with a visible window instead of headless mode.",
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


def json_compact(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def pick_first_present(content: dict[str, object], keys: list[str]) -> object:
    for key in keys:
        if key in content:
            return unwrap_content_value(content.get(key))
    return None


def unwrap_content_value(value: object) -> object:
    if isinstance(value, dict):
        for key in ("value", "values", "value-radio", "value-dropdown"):
            if key in value:
                return value[key]
    return value


def full_openreview_url(path_or_url: object) -> str | None:
    if path_or_url is None:
        return None
    text = str(path_or_url).strip()
    if not text:
        return None
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if text.startswith("/"):
        return f"{OPENREVIEW_BASE_URL}{text}"
    return f"{OPENREVIEW_BASE_URL}/{text}"


def millis_to_iso(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        timestamp = float(value) / 1000.0
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


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


def build_endpoint_url(endpoint: str, paper_id: str) -> str:
    quoted_paper_id = quote(paper_id, safe="")
    if endpoint == "notes":
        return (
            f"https://api.openreview.net/notes?id={quoted_paper_id}"
            f"&details={OPENREVIEW_NOTES_DETAILS}"
        )
    if endpoint == "forum_notes":
        return (
            f"https://api.openreview.net/notes?forum={quoted_paper_id}"
            f"&trash=true&details={OPENREVIEW_NOTES_DETAILS}&limit=1000&offset=0"
        )
    if endpoint == "invitations":
        return (
            f"https://api.openreview.net/invitations?replyForum={quoted_paper_id}"
            f"&details=repliedNotes&limit=1000"
        )
    raise ValueError(f"Unknown endpoint: {endpoint}")


def endpoint_cache_path(raw_dir: Path, endpoint: str, paper_id: str) -> Path:
    safe_id = safe_paper_id(paper_id)
    if endpoint == "notes":
        return raw_dir / "notes" / f"{safe_id}.json"
    if endpoint == "forum_notes":
        return raw_dir / "forum_notes" / f"{safe_id}.json"
    if endpoint == "invitations":
        return raw_dir / "invitations" / f"{safe_id}.json"
    raise ValueError(f"Unknown endpoint: {endpoint}")


def load_cached_wrapper(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_wrapper(path: Path, wrapper: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(wrapper, handle, ensure_ascii=False, indent=2)


class OpenReviewBrowserClient:
    def __init__(self, headed: bool, timeout_ms: int) -> None:
        self.headed = headed
        self.timeout_ms = timeout_ms
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def __enter__(self) -> "OpenReviewBrowserClient":
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=not self.headed)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        for obj in (self.page, self.context, self.browser):
            if obj is None:
                continue
            try:
                obj.close()
            except Exception:
                pass
        if self.playwright is not None:
            self.playwright.stop()

    def warm_up(self, seed_paper_id: str | None) -> None:
        targets = []
        if seed_paper_id:
            targets.append(f"{OPENREVIEW_BASE_URL}/forum?id={quote(seed_paper_id, safe='')}")
        targets.extend(
            [
                f"{OPENREVIEW_BASE_URL}/about",
                OPENREVIEW_BASE_URL,
            ]
        )
        last_error = None
        for url in targets:
            try:
                self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                return
            except (PlaywrightError, PlaywrightTimeoutError) as exc:
                last_error = exc
        raise RuntimeError(f"Failed to warm up OpenReview browser context: {last_error}")

    def recycle(self, seed_paper_id: str | None) -> None:
        try:
            if self.page is not None:
                self.page.close()
        except Exception:
            pass
        self.page = self.context.new_page()
        self.warm_up(seed_paper_id)

    def fetch_json(self, url: str) -> dict[str, object]:
        return self.page.evaluate(
            JS_FETCH_JSON,
            {"url": url, "timeoutMs": self.timeout_ms},
        )


def fetch_endpoint(
    client: OpenReviewBrowserClient | None,
    raw_dir: Path,
    endpoint: str,
    paper_id: str,
    *,
    refresh: bool,
    retry_max_attempts: int,
    retry_backoff_seconds: float,
    retry_backoff_factor: float,
    retry_max_sleep_seconds: float,
) -> dict[str, object]:
    cache_path = endpoint_cache_path(raw_dir, endpoint, paper_id)
    if cache_path.exists() and not refresh:
        wrapper = load_cached_wrapper(cache_path)
        if wrapper is None:
            raise RuntimeError(f"Failed to load cached OpenReview response: {cache_path}")
        wrapper["from_cache"] = True
        wrapper["cache_path"] = str(cache_path.resolve())
        return wrapper

    if client is None:
        raise RuntimeError(f"Browser client is required to fetch uncached endpoint `{endpoint}`.")

    url = build_endpoint_url(endpoint, paper_id)
    backoff = retry_backoff_seconds
    last_wrapper = None
    for attempt in range(1, retry_max_attempts + 1):
        now = datetime.now(timezone.utc).isoformat()
        try:
            result = client.fetch_json(url)
            body = result.get("body")
            wrapper = {
                "paper_id": paper_id,
                "endpoint": endpoint,
                "url": url,
                "fetched_at": now,
                "attempts": attempt,
                "ok": bool(result.get("ok")),
                "status": result.get("status"),
                "error": result.get("error"),
                "body": body,
                "text": result.get("text") if body is None else None,
            }
        except Exception as exc:
            wrapper = {
                "paper_id": paper_id,
                "endpoint": endpoint,
                "url": url,
                "fetched_at": now,
                "attempts": attempt,
                "ok": False,
                "status": None,
                "error": str(exc),
                "body": None,
                "text": None,
            }

        status = wrapper.get("status")
        error = wrapper.get("error")
        retryable = status in RETRYABLE_STATUS_CODES or bool(error)
        last_wrapper = wrapper
        if wrapper.get("ok") or not retryable or attempt == retry_max_attempts:
            save_wrapper(cache_path, wrapper)
            wrapper["from_cache"] = False
            wrapper["cache_path"] = str(cache_path.resolve())
            return wrapper

        time.sleep(backoff)
        backoff = min(backoff * retry_backoff_factor, retry_max_sleep_seconds)
        client.recycle(seed_paper_id=paper_id)

    if last_wrapper is None:
        raise RuntimeError(f"OpenReview fetch failed before any attempts for {paper_id}.")
    save_wrapper(cache_path, last_wrapper)
    last_wrapper["from_cache"] = False
    last_wrapper["cache_path"] = str(cache_path.resolve())
    return last_wrapper


def parse_notes_metadata(base_row: dict[str, object], wrapper: dict[str, object]) -> dict[str, object]:
    body = wrapper.get("body")
    notes = []
    if isinstance(body, dict):
        maybe_notes = body.get("notes")
        if isinstance(maybe_notes, list):
            notes = maybe_notes
    note = notes[0] if notes else None
    content = note.get("content") if isinstance(note, dict) else {}
    content = content if isinstance(content, dict) else {}

    title = unwrap_content_value(content.get("title"))
    authors = unwrap_content_value(content.get("authors"))
    authorids = unwrap_content_value(content.get("authorids"))
    keywords = unwrap_content_value(content.get("keywords"))
    abstract = unwrap_content_value(content.get("abstract"))
    summary = pick_first_present(content, ["one-sentence_summary", "TL;DR", "tldr"])
    pdf_url = full_openreview_url(pick_first_present(content, ["pdf"]))
    reviewed_pdf_url = full_openreview_url(
        pick_first_present(content, ["reviewed_version_(pdf)", "reviewed_version_pdf"])
    )
    supplementary_url = full_openreview_url(
        pick_first_present(content, ["supplementary_material", "supplementary"])
    )
    bibtex = pick_first_present(content, ["_bibtex", "bibtex"])
    paperhash = pick_first_present(content, ["paperhash"])

    title_norm = normalize_title(title)
    input_title_norm = normalize_title(base_row.get("input_title"))
    similarity = title_similarity(input_title_norm, title_norm)
    exact_title_match = bool(title_norm) and title_norm == input_title_norm

    author_list = authors if isinstance(authors, list) else ([authors] if authors else [])

    return {
        **base_row,
        "openreview_note_found": note is not None,
        "openreview_status_code": wrapper.get("status"),
        "openreview_error": wrapper.get("error"),
        "openreview_note_count": len(notes),
        "openreview_note_id": note.get("id") if isinstance(note, dict) else None,
        "openreview_forum_id": note.get("forum") if isinstance(note, dict) else None,
        "openreview_forum_url": (
            f"{OPENREVIEW_BASE_URL}/forum?id={note.get('forum')}"
            if isinstance(note, dict) and note.get("forum")
            else None
        ),
        "openreview_original_note_id": note.get("original") if isinstance(note, dict) else None,
        "openreview_invitation": note.get("invitation") if isinstance(note, dict) else None,
        "openreview_number": note.get("number") if isinstance(note, dict) else None,
        "openreview_title": title,
        "openreview_title_norm": title_norm,
        "openreview_title_similarity": similarity if note is not None else None,
        "openreview_exact_title_match": exact_title_match if note is not None else None,
        "openreview_authors": "; ".join(str(value) for value in author_list) if author_list else None,
        "openreview_authors_json": json_compact(authors),
        "openreview_authorids_json": json_compact(authorids),
        "openreview_keywords_json": json_compact(keywords),
        "openreview_abstract": abstract,
        "openreview_one_sentence_summary": summary,
        "openreview_pdf_url": pdf_url,
        "openreview_reviewed_pdf_url": reviewed_pdf_url,
        "openreview_supplementary_url": supplementary_url,
        "openreview_bibtex": bibtex,
        "openreview_paperhash": paperhash,
        "openreview_reply_count": (
            note.get("details", {}).get("replyCount")
            if isinstance(note, dict) and isinstance(note.get("details"), dict)
            else None
        ),
        "openreview_created_at": millis_to_iso(note.get("cdate")) if isinstance(note, dict) else None,
        "openreview_tcdate_at": millis_to_iso(note.get("tcdate")) if isinstance(note, dict) else None,
        "openreview_last_modified_at": millis_to_iso(note.get("tmdate")) if isinstance(note, dict) else None,
        "notes_fetch_attempts": wrapper.get("attempts"),
        "notes_raw_path": wrapper.get("cache_path"),
        "notes_from_cache": wrapper.get("from_cache", False),
    }


def parse_thread_summary(wrapper: dict[str, object]) -> dict[str, object]:
    body = wrapper.get("body")
    notes = []
    if isinstance(body, dict):
        maybe_notes = body.get("notes")
        if isinstance(maybe_notes, list):
            notes = maybe_notes

    decision_values = []
    official_review_count = 0
    meta_review_count = 0
    public_comment_count = 0
    author_response_count = 0

    for note in notes:
        if not isinstance(note, dict):
            continue
        invitation = str(note.get("invitation") or "").lower()
        content = note.get("content") if isinstance(note.get("content"), dict) else {}

        if "official_review" in invitation or "review" in invitation:
            official_review_count += 1
        if "meta_review" in invitation or "decision" in invitation or "recommendation" in invitation:
            meta_review_count += 1
        if "comment" in invitation:
            public_comment_count += 1
        if "author" in invitation and ("response" in invitation or "rebuttal" in invitation):
            author_response_count += 1

        decision_value = pick_first_present(content, ["decision", "recommendation"])
        if decision_value:
            decision_values.append(decision_value)

    return {
        "thread_status_code": wrapper.get("status"),
        "thread_error": wrapper.get("error"),
        "thread_note_count": len(notes),
        "thread_decision_values_json": json_compact(decision_values) if decision_values else None,
        "thread_decision_count": len(decision_values),
        "thread_official_review_count": official_review_count,
        "thread_meta_review_count": meta_review_count,
        "thread_public_comment_count": public_comment_count,
        "thread_author_response_count": author_response_count,
        "thread_fetch_attempts": wrapper.get("attempts"),
        "thread_raw_path": wrapper.get("cache_path"),
        "thread_from_cache": wrapper.get("from_cache", False),
    }


def parse_invitations_summary(wrapper: dict[str, object]) -> dict[str, object]:
    body = wrapper.get("body")
    invitations = []
    if isinstance(body, dict):
        maybe_invites = body.get("invitations")
        if isinstance(maybe_invites, list):
            invitations = maybe_invites

    invitation_ids = [invite.get("id") for invite in invitations if isinstance(invite, dict)]
    invitation_ids = [value for value in invitation_ids if value]
    invitation_ids_lower = [value.lower() for value in invitation_ids]

    return {
        "invitations_status_code": wrapper.get("status"),
        "invitations_error": wrapper.get("error"),
        "invitation_count": len(invitation_ids),
        "invitation_ids_json": json_compact(invitation_ids) if invitation_ids else None,
        "has_decision_invitation": any("decision" in value for value in invitation_ids_lower),
        "has_official_review_invitation": any("official_review" in value for value in invitation_ids_lower),
        "has_meta_review_invitation": any("meta_review" in value for value in invitation_ids_lower),
        "invitations_fetch_attempts": wrapper.get("attempts"),
        "invitations_raw_path": wrapper.get("cache_path"),
        "invitations_from_cache": wrapper.get("from_cache", False),
    }


def summarize_results(
    input_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "input_csv": str(args.input_csv.resolve()),
        "raw_dir": str(args.raw_dir.resolve()),
        "query_mode": args.query_mode,
        "input_rows_total": int(len(input_df)),
        "rows_selected_for_query": int(len(selected_df)),
        "unique_paper_ids_selected": int(selected_df["paper_id"].nunique()) if len(selected_df) else 0,
        "notes_status_200": int((metadata_df["openreview_status_code"] == 200).sum()) if len(metadata_df) else 0,
        "notes_status_404": int((metadata_df["openreview_status_code"] == 404).sum()) if len(metadata_df) else 0,
        "notes_status_403": int((metadata_df["openreview_status_code"] == 403).sum()) if len(metadata_df) else 0,
        "note_found_count": int(metadata_df["openreview_note_found"].fillna(False).sum()) if len(metadata_df) else 0,
        "exact_title_match_count": int(metadata_df["openreview_exact_title_match"].fillna(False).sum()) if len(metadata_df) else 0,
    }
    if len(metadata_df):
        similarity_values = metadata_df["openreview_title_similarity"].dropna()
        summary["mean_title_similarity"] = (
            float(similarity_values.mean()) if not similarity_values.empty else None
        )
        summary["median_title_similarity"] = (
            float(similarity_values.median()) if not similarity_values.empty else None
        )
        summary["notes_from_cache_count"] = int(metadata_df["notes_from_cache"].fillna(False).sum())
        by_year = (
            metadata_df.groupby("input_year", dropna=False)["openreview_note_found"]
            .agg(["count", "sum"])
            .reset_index()
            .to_dict(orient="records")
        )
        summary["note_found_by_year"] = by_year
    else:
        summary["mean_title_similarity"] = None
        summary["median_title_similarity"] = None
        summary["notes_from_cache_count"] = 0
        summary["note_found_by_year"] = []
    return summary


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(args.input_csv)
    paper_id_col = choose_column(input_df, args.paper_id_col, ["paper_id"], "paper_id")
    if paper_id_col is None:
        raise ValueError("No paper id column found in input CSV.")
    title_col = choose_column(input_df, args.title_col, ["input_title", "title"], "title")
    year_col = choose_column(input_df, args.year_col, ["input_year", "year"], "year")
    source_id_col = args.source_id_col if args.source_id_col in input_df.columns else None

    selected_df = input_df.copy()
    if args.query_mode == "arxiv_unmatched" and args.matched_col in selected_df.columns:
        selected_df = selected_df.loc[~selected_df[args.matched_col].fillna(False).map(to_bool)].copy()
    selected_df = selected_df.loc[selected_df[paper_id_col].notna()].copy()
    selected_df[paper_id_col] = selected_df[paper_id_col].astype(str).str.strip()
    selected_df = selected_df.loc[selected_df[paper_id_col] != ""].copy()

    dedupe_cols = [paper_id_col]
    selected_df = selected_df.drop_duplicates(subset=dedupe_cols, keep="first").copy()

    if args.limit is not None:
        selected_df = selected_df.head(args.limit).copy()

    normalized_rows = []
    for row in selected_df.to_dict(orient="records"):
        normalized_rows.append(
            {
                "paper_id": row.get(paper_id_col),
                "input_title": row.get(title_col) if title_col else None,
                "input_year": row.get(year_col) if year_col else None,
                "source_id": row.get(source_id_col) if source_id_col else None,
            }
        )
    selected_norm_df = pd.DataFrame(normalized_rows)

    print(
        f"Selected {len(selected_norm_df):,} unique paper ids from {len(input_df):,} input rows "
        f"(mode={args.query_mode})."
    )
    if selected_norm_df.empty:
        metadata_df = pd.DataFrame(
            columns=[
                "paper_id",
                "input_title",
                "input_year",
                "source_id",
                "openreview_note_found",
            ]
        )
        metadata_path = args.raw_dir / "openreview_note_metadata.csv"
        manifest_path = args.raw_dir / "query_manifest.csv"
        joined_path = args.raw_dir / f"{args.input_csv.stem}_with_openreview_metadata.csv"
        summary_path = args.raw_dir / "openreview_query_summary.json"
        metadata_df.to_csv(metadata_path, index=False)
        pd.DataFrame().to_csv(manifest_path, index=False)
        input_df.to_csv(joined_path, index=False)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                summarize_results(input_df, selected_norm_df, metadata_df, args),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return

    endpoints = ["notes"]
    if args.fetch_thread:
        endpoints.append("forum_notes")
    if args.fetch_invitations:
        endpoints.append("invitations")

    needs_browser = args.refresh
    if not needs_browser:
        for endpoint in endpoints:
            if any(not endpoint_cache_path(args.raw_dir, endpoint, paper_id).exists() for paper_id in selected_norm_df["paper_id"]):
                needs_browser = True
                break

    seed_paper_id = str(selected_norm_df.iloc[0]["paper_id"])
    metadata_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    client_cm = OpenReviewBrowserClient(headed=args.headed, timeout_ms=int(args.timeout_seconds * 1000))
    client = client_cm.__enter__() if needs_browser else None
    try:
        if client is not None:
            client.warm_up(seed_paper_id)

        for idx, base_row in enumerate(selected_norm_df.to_dict(orient="records"), start=1):
            paper_id = str(base_row["paper_id"])

            notes_wrapper = fetch_endpoint(
                client,
                args.raw_dir,
                "notes",
                paper_id,
                refresh=args.refresh,
                retry_max_attempts=args.retry_max_attempts,
                retry_backoff_seconds=args.retry_backoff_seconds,
                retry_backoff_factor=args.retry_backoff_factor,
                retry_max_sleep_seconds=args.retry_max_sleep_seconds,
            )
            notes_record = parse_notes_metadata(base_row, notes_wrapper)
            manifest_row = {
                "paper_id": paper_id,
                "input_title": base_row.get("input_title"),
                "input_year": base_row.get("input_year"),
                "source_id": base_row.get("source_id"),
                "notes_status_code": notes_wrapper.get("status"),
                "notes_ok": notes_wrapper.get("ok"),
                "notes_error": notes_wrapper.get("error"),
                "notes_attempts": notes_wrapper.get("attempts"),
                "notes_from_cache": notes_wrapper.get("from_cache", False),
                "notes_raw_path": notes_wrapper.get("cache_path"),
            }

            if args.fetch_thread:
                thread_wrapper = fetch_endpoint(
                    client,
                    args.raw_dir,
                    "forum_notes",
                    paper_id,
                    refresh=args.refresh,
                    retry_max_attempts=args.retry_max_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    retry_backoff_factor=args.retry_backoff_factor,
                    retry_max_sleep_seconds=args.retry_max_sleep_seconds,
                )
                notes_record.update(parse_thread_summary(thread_wrapper))
                manifest_row.update(
                    {
                        "thread_status_code": thread_wrapper.get("status"),
                        "thread_ok": thread_wrapper.get("ok"),
                        "thread_error": thread_wrapper.get("error"),
                        "thread_attempts": thread_wrapper.get("attempts"),
                        "thread_from_cache": thread_wrapper.get("from_cache", False),
                        "thread_raw_path": thread_wrapper.get("cache_path"),
                    }
                )

            if args.fetch_invitations:
                invitations_wrapper = fetch_endpoint(
                    client,
                    args.raw_dir,
                    "invitations",
                    paper_id,
                    refresh=args.refresh,
                    retry_max_attempts=args.retry_max_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    retry_backoff_factor=args.retry_backoff_factor,
                    retry_max_sleep_seconds=args.retry_max_sleep_seconds,
                )
                notes_record.update(parse_invitations_summary(invitations_wrapper))
                manifest_row.update(
                    {
                        "invitations_status_code": invitations_wrapper.get("status"),
                        "invitations_ok": invitations_wrapper.get("ok"),
                        "invitations_error": invitations_wrapper.get("error"),
                        "invitations_attempts": invitations_wrapper.get("attempts"),
                        "invitations_from_cache": invitations_wrapper.get("from_cache", False),
                        "invitations_raw_path": invitations_wrapper.get("cache_path"),
                    }
                )

            metadata_rows.append(notes_record)
            manifest_rows.append(manifest_row)

            if args.sleep_seconds > 0 and needs_browser:
                time.sleep(args.sleep_seconds)

            if idx % args.progress_every == 0 or idx == len(selected_norm_df):
                found_count = sum(1 for row in metadata_rows if row.get("openreview_note_found"))
                print(
                    f"[{idx:,}/{len(selected_norm_df):,}] "
                    f"OpenReview notes found for {found_count:,} papers."
                )
    finally:
        if client is not None:
            client_cm.__exit__(None, None, None)

    metadata_df = pd.DataFrame(metadata_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    metadata_path = args.raw_dir / "openreview_note_metadata.csv"
    manifest_path = args.raw_dir / "query_manifest.csv"
    joined_path = args.raw_dir / f"{args.input_csv.stem}_with_openreview_metadata.csv"
    summary_path = args.raw_dir / "openreview_query_summary.json"

    metadata_df.to_csv(metadata_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    merge_cols = [col for col in metadata_df.columns if col not in {"input_title", "input_year", "source_id"}]
    joined_df = input_df.merge(metadata_df[merge_cols], on="paper_id", how="left")
    joined_df.to_csv(joined_path, index=False)

    summary = summarize_results(input_df, selected_norm_df, metadata_df, args)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote joined output to {joined_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
