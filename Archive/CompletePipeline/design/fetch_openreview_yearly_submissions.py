#!/usr/bin/env python3
"""
Fetch public OpenReview submission metadata for ICLR by year and cache raw pages.

This uses browser-context fetches because direct HTTP requests from this
environment can be denied even for public OpenReview endpoints. The script
handles the older OpenReview v1 invitation pattern used in ICLR 2018-2023 and
the newer OpenReview v2 domain-scoped submission invitation used in ICLR
2024-2025.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class YearQueryConfig:
    year: int
    api_type: str
    invitation: str
    domain: str | None = None


YEAR_CONFIGS: dict[int, YearQueryConfig] = {
    year: YearQueryConfig(
        year=year,
        api_type="v1",
        invitation=f"ICLR.cc/{year}/Conference/-/Blind_Submission",
    )
    for year in range(2018, 2024)
}
YEAR_CONFIGS.update(
    {
        2024: YearQueryConfig(
            year=2024,
            api_type="v2",
            domain="ICLR.cc/2024/Conference",
            invitation="ICLR.cc/2024/Conference/-/Submission",
        ),
        2025: YearQueryConfig(
            year=2025,
            api_type="v2",
            domain="ICLR.cc/2025/Conference",
            invitation="ICLR.cc/2025/Conference/-/Submission",
        ),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ICLR submission metadata from OpenReview in year-level batches."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=(
            "Optional input CSV to merge against fetched OpenReview submissions. "
            "Defaults to the combined arXiv match file."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where raw OpenReview pages and derived outputs are written.",
    )
    parser.add_argument(
        "--years",
        default="2018,2019,2020,2021,2022,2023,2024,2025",
        help="Comma-separated ICLR years to query.",
    )
    parser.add_argument(
        "--limit-per-page",
        type=int,
        default=1000,
        help="OpenReview page size.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.35,
        help="Minimum delay between live OpenReview page fetches.",
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
        help="Maximum attempts for a single OpenReview page fetch.",
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
        "--refresh",
        action="store_true",
        help="Ignore cached page responses and re-query OpenReview.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch Chromium with a visible window instead of headless mode.",
    )
    return parser.parse_args()


def parse_years(years_arg: str) -> list[int]:
    years = []
    for part in years_arg.split(","):
        part = part.strip()
        if not part:
            continue
        year = int(part)
        if year not in YEAR_CONFIGS:
            raise ValueError(f"Unsupported year `{year}`. Supported years: {sorted(YEAR_CONFIGS)}")
        years.append(year)
    deduped = []
    for year in years:
        if year not in deduped:
            deduped.append(year)
    return deduped


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


def unwrap_content_value(value: object) -> object:
    if isinstance(value, dict):
        for key in ("value", "values", "value-radio", "value-dropdown"):
            if key in value:
                return value[key]
    return value


def pick_first_present(content: dict[str, object], keys: list[str]) -> object:
    for key in keys:
        if key in content:
            return unwrap_content_value(content.get(key))
    return None


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


def json_compact(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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


def year_page_cache_path(raw_dir: Path, year: int, offset: int) -> Path:
    return raw_dir / "year_pages" / str(year) / f"page_{offset:05d}.json"


def load_cached_wrapper(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_wrapper(path: Path, wrapper: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(wrapper, handle, ensure_ascii=False, indent=2)


def build_year_page_url(config: YearQueryConfig, limit: int, offset: int) -> str:
    if config.api_type == "v1":
        return (
            "https://api.openreview.net/notes"
            f"?invitation={quote(config.invitation, safe='')}"
            f"&limit={limit}&offset={offset}"
        )
    if config.api_type == "v2":
        return (
            "https://api2.openreview.net/notes"
            f"?domain={quote(config.domain or '', safe='')}"
            f"&invitation={quote(config.invitation, safe='')}"
            f"&limit={limit}&offset={offset}&count=true"
        )
    raise ValueError(f"Unsupported api_type: {config.api_type}")


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

    def warm_up(self) -> None:
        targets = [
            f"{OPENREVIEW_BASE_URL}/group?id=ICLR.cc/2025/Conference",
            f"{OPENREVIEW_BASE_URL}/forum?id=-YCAwPdyPKw",
            f"{OPENREVIEW_BASE_URL}/about",
            OPENREVIEW_BASE_URL,
        ]
        last_error = None
        for url in targets:
            try:
                self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                return
            except (PlaywrightError, PlaywrightTimeoutError) as exc:
                last_error = exc
        raise RuntimeError(f"Failed to warm up OpenReview browser context: {last_error}")

    def recycle(self) -> None:
        try:
            if self.page is not None:
                self.page.close()
        except Exception:
            pass
        self.page = self.context.new_page()
        self.warm_up()

    def fetch_json(self, url: str) -> dict[str, object]:
        return self.page.evaluate(JS_FETCH_JSON, {"url": url, "timeoutMs": self.timeout_ms})


def fetch_year_page(
    client: OpenReviewBrowserClient | None,
    raw_dir: Path,
    config: YearQueryConfig,
    *,
    limit: int,
    offset: int,
    refresh: bool,
    retry_max_attempts: int,
    retry_backoff_seconds: float,
    retry_backoff_factor: float,
    retry_max_sleep_seconds: float,
) -> dict[str, object]:
    cache_path = year_page_cache_path(raw_dir, config.year, offset)
    if cache_path.exists() and not refresh:
        wrapper = load_cached_wrapper(cache_path)
        if wrapper is None:
            raise RuntimeError(f"Failed to load cached page response: {cache_path}")
        wrapper["from_cache"] = True
        wrapper["cache_path"] = str(cache_path.resolve())
        return wrapper

    if client is None:
        raise RuntimeError(f"Browser client is required to fetch uncached year page for {config.year}.")

    url = build_year_page_url(config, limit=limit, offset=offset)
    backoff = retry_backoff_seconds
    last_wrapper = None
    for attempt in range(1, retry_max_attempts + 1):
        now = datetime.now(timezone.utc).isoformat()
        try:
            result = client.fetch_json(url)
            body = result.get("body")
            wrapper = {
                "year": config.year,
                "api_type": config.api_type,
                "domain": config.domain,
                "invitation": config.invitation,
                "url": url,
                "offset": offset,
                "limit": limit,
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
                "year": config.year,
                "api_type": config.api_type,
                "domain": config.domain,
                "invitation": config.invitation,
                "url": url,
                "offset": offset,
                "limit": limit,
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
        client.recycle()

    if last_wrapper is None:
        raise RuntimeError(f"OpenReview page fetch failed before any attempts for {config.year}.")
    save_wrapper(cache_path, last_wrapper)
    last_wrapper["from_cache"] = False
    last_wrapper["cache_path"] = str(cache_path.resolve())
    return last_wrapper


def extract_notes(wrapper: dict[str, object]) -> list[dict[str, object]]:
    body = wrapper.get("body")
    if not isinstance(body, dict):
        return []
    notes = body.get("notes")
    if not isinstance(notes, list):
        return []
    return [note for note in notes if isinstance(note, dict)]


def extract_count(wrapper: dict[str, object], notes: list[dict[str, object]]) -> int | None:
    body = wrapper.get("body")
    if isinstance(body, dict):
        count = body.get("count")
        if isinstance(count, int):
            return count
    return None


def flatten_note(year: int, config: YearQueryConfig, note: dict[str, object]) -> dict[str, object]:
    content = note.get("content")
    content = content if isinstance(content, dict) else {}

    title = pick_first_present(content, ["title"])
    authors = pick_first_present(content, ["authors"])
    authorids = pick_first_present(content, ["authorids"])
    keywords = pick_first_present(content, ["keywords"])
    abstract = pick_first_present(content, ["abstract"])
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
    venue = pick_first_present(content, ["venue"])
    venueid = pick_first_present(content, ["venueid"])

    authors_list = authors if isinstance(authors, list) else ([authors] if authors else [])
    invitation_value = note.get("invitation")
    if invitation_value is None and isinstance(note.get("invitations"), list):
        invitations = [value for value in note.get("invitations") if value]
        invitation_value = invitations[0] if invitations else None

    note_id = note.get("id")
    forum_id = note.get("forum") or note_id

    return {
        "query_year": year,
        "openreview_api_type": config.api_type,
        "openreview_domain": config.domain,
        "openreview_query_invitation": config.invitation,
        "paper_id": forum_id,
        "openreview_note_id": note_id,
        "openreview_forum_id": forum_id,
        "openreview_forum_url": (
            f"{OPENREVIEW_BASE_URL}/forum?id={forum_id}" if forum_id else None
        ),
        "openreview_original_note_id": note.get("original"),
        "openreview_number": note.get("number"),
        "openreview_invitation": invitation_value,
        "openreview_title": title,
        "openreview_title_norm": normalize_title(title),
        "openreview_authors": "; ".join(str(value) for value in authors_list) if authors_list else None,
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
        "openreview_venue": venue,
        "openreview_venueid": venueid,
        "openreview_created_at": millis_to_iso(note.get("cdate")),
        "openreview_tcdate_at": millis_to_iso(note.get("tcdate")),
        "openreview_last_modified_at": millis_to_iso(note.get("tmdate")),
        "openreview_pdate_at": millis_to_iso(note.get("pdate")),
        "openreview_odate_at": millis_to_iso(note.get("odate")),
        "openreview_signatures_json": json_compact(note.get("signatures")),
        "openreview_readers_json": json_compact(note.get("readers")),
        "openreview_writers_json": json_compact(note.get("writers")),
        "openreview_nonreaders_json": json_compact(note.get("nonreaders")),
    }


def build_title_match_frame(openreview_df: pd.DataFrame) -> pd.DataFrame:
    if openreview_df.empty:
        return pd.DataFrame(columns=["query_year", "openreview_title_norm"])
    title_counts = (
        openreview_df.groupby(["query_year", "openreview_title_norm"])
        .size()
        .reset_index(name="n_year_title_matches")
    )
    unique_titles = title_counts.loc[
        (title_counts["openreview_title_norm"] != "")
        & (title_counts["n_year_title_matches"] == 1)
    ][["query_year", "openreview_title_norm"]]
    return openreview_df.merge(
        unique_titles,
        on=["query_year", "openreview_title_norm"],
        how="inner",
    )


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    page_manifest_rows: list[dict[str, object]] = []
    note_rows: list[dict[str, object]] = []

    needs_browser = args.refresh
    if not needs_browser:
        for year in years:
            first_page = year_page_cache_path(args.raw_dir, year, 0)
            if not first_page.exists():
                needs_browser = True
                break

    client_cm = OpenReviewBrowserClient(headed=args.headed, timeout_ms=int(args.timeout_seconds * 1000))
    client = client_cm.__enter__() if needs_browser else None
    try:
        if client is not None:
            client.warm_up()

        for year in years:
            config = YEAR_CONFIGS[year]
            print(f"Fetching OpenReview submissions for ICLR {year} ({config.api_type})")

            first_wrapper = fetch_year_page(
                client,
                args.raw_dir,
                config,
                limit=args.limit_per_page,
                offset=0,
                refresh=args.refresh,
                retry_max_attempts=args.retry_max_attempts,
                retry_backoff_seconds=args.retry_backoff_seconds,
                retry_backoff_factor=args.retry_backoff_factor,
                retry_max_sleep_seconds=args.retry_max_sleep_seconds,
            )
            first_notes = extract_notes(first_wrapper)
            total_count = extract_count(first_wrapper, first_notes)
            if total_count is None:
                if len(first_notes) < args.limit_per_page:
                    total_count = len(first_notes)

            page_manifest_rows.append(
                {
                    "year": year,
                    "offset": 0,
                    "status_code": first_wrapper.get("status"),
                    "ok": first_wrapper.get("ok"),
                    "error": first_wrapper.get("error"),
                    "attempts": first_wrapper.get("attempts"),
                    "from_cache": first_wrapper.get("from_cache", False),
                    "cache_path": first_wrapper.get("cache_path"),
                    "n_notes": len(first_notes),
                    "reported_count": total_count,
                }
            )
            for note in first_notes:
                note_rows.append(flatten_note(year, config, note))

            fetched_offsets = {0}
            next_offset = args.limit_per_page
            while True:
                if total_count is not None and next_offset >= total_count:
                    break

                wrapper = fetch_year_page(
                    client,
                    args.raw_dir,
                    config,
                    limit=args.limit_per_page,
                    offset=next_offset,
                    refresh=args.refresh,
                    retry_max_attempts=args.retry_max_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    retry_backoff_factor=args.retry_backoff_factor,
                    retry_max_sleep_seconds=args.retry_max_sleep_seconds,
                )
                notes = extract_notes(wrapper)
                page_manifest_rows.append(
                    {
                        "year": year,
                        "offset": next_offset,
                        "status_code": wrapper.get("status"),
                        "ok": wrapper.get("ok"),
                        "error": wrapper.get("error"),
                        "attempts": wrapper.get("attempts"),
                        "from_cache": wrapper.get("from_cache", False),
                        "cache_path": wrapper.get("cache_path"),
                        "n_notes": len(notes),
                        "reported_count": wrapper.get("body", {}).get("count")
                        if isinstance(wrapper.get("body"), dict)
                        else None,
                    }
                )
                for note in notes:
                    note_rows.append(flatten_note(year, config, note))
                fetched_offsets.add(next_offset)
                next_offset += args.limit_per_page

                if args.sleep_seconds > 0 and needs_browser:
                    time.sleep(args.sleep_seconds)

                if not notes:
                    break
                if total_count is None and len(notes) < args.limit_per_page:
                    break

            print(
                f"  cached/fetched {len(fetched_offsets)} page(s) for {year}; "
                f"accumulated {sum(1 for row in note_rows if row['query_year'] == year):,} notes."
            )
    finally:
        if client is not None:
            client_cm.__exit__(None, None, None)

    openreview_df = pd.DataFrame(note_rows)
    if not openreview_df.empty:
        openreview_df = openreview_df.drop_duplicates(subset=["openreview_note_id"], keep="first").copy()

    page_manifest_df = pd.DataFrame(page_manifest_rows)

    all_notes_path = args.raw_dir / "openreview_yearly_submissions.csv"
    page_manifest_path = args.raw_dir / "year_page_manifest.csv"
    summary_path = args.raw_dir / "openreview_yearly_submissions_summary.json"

    openreview_df.to_csv(all_notes_path, index=False)
    page_manifest_df.to_csv(page_manifest_path, index=False)

    input_df = pd.read_csv(args.input_csv)
    paper_id_col = choose_column(input_df, args.paper_id_col, ["paper_id"], "paper_id")
    if paper_id_col is None:
        raise ValueError("No paper id column found in input CSV.")
    title_col = choose_column(input_df, args.title_col, ["input_title", "title"], "title")
    year_col = choose_column(input_df, args.year_col, ["input_year", "year"], "year")
    source_id_col = args.source_id_col if args.source_id_col in input_df.columns else None

    match_input = pd.DataFrame(
        {
            "paper_id": input_df[paper_id_col],
            "input_title": input_df[title_col] if title_col else None,
            "input_year": input_df[year_col] if year_col else None,
            "source_id": input_df[source_id_col] if source_id_col else None,
        }
    )
    match_input["input_title_norm"] = match_input["input_title"].map(normalize_title)

    id_match_cols = [col for col in openreview_df.columns if col != "query_year"] + ["query_year"]
    id_merged = match_input.merge(openreview_df[id_match_cols], on="paper_id", how="left")
    id_merged["openreview_match_source"] = id_merged["openreview_note_id"].notna().map(
        lambda matched: "paper_id" if matched else None
    )

    title_match_df = build_title_match_frame(openreview_df)
    title_match_cols = [
        "query_year",
        "openreview_title_norm",
        "paper_id",
        "openreview_note_id",
        "openreview_forum_id",
        "openreview_forum_url",
        "openreview_original_note_id",
        "openreview_number",
        "openreview_invitation",
        "openreview_title",
        "openreview_authors",
        "openreview_authors_json",
        "openreview_authorids_json",
        "openreview_keywords_json",
        "openreview_abstract",
        "openreview_one_sentence_summary",
        "openreview_pdf_url",
        "openreview_reviewed_pdf_url",
        "openreview_supplementary_url",
        "openreview_bibtex",
        "openreview_paperhash",
        "openreview_venue",
        "openreview_venueid",
        "openreview_created_at",
        "openreview_tcdate_at",
        "openreview_last_modified_at",
        "openreview_pdate_at",
        "openreview_odate_at",
        "openreview_api_type",
        "openreview_domain",
        "openreview_query_invitation",
    ]
    title_lookup = title_match_df[title_match_cols].rename(
        columns={
            "paper_id": "matched_paper_id_from_title",
            "query_year": "input_year",
            "openreview_title_norm": "input_title_norm",
        }
    )
    joined_df = id_merged.merge(title_lookup, on=["input_year", "input_title_norm"], how="left", suffixes=("", "_titlematch"))

    unmatched_mask = joined_df["openreview_note_id"].isna() & joined_df["openreview_note_id_titlematch"].notna()
    fill_map = {
        "openreview_note_id": "openreview_note_id_titlematch",
        "openreview_forum_id": "openreview_forum_id_titlematch",
        "openreview_forum_url": "openreview_forum_url_titlematch",
        "openreview_original_note_id": "openreview_original_note_id_titlematch",
        "openreview_number": "openreview_number_titlematch",
        "openreview_invitation": "openreview_invitation_titlematch",
        "openreview_title": "openreview_title_titlematch",
        "openreview_title_norm": "input_title_norm",
        "openreview_authors": "openreview_authors_titlematch",
        "openreview_authors_json": "openreview_authors_json_titlematch",
        "openreview_authorids_json": "openreview_authorids_json_titlematch",
        "openreview_keywords_json": "openreview_keywords_json_titlematch",
        "openreview_abstract": "openreview_abstract_titlematch",
        "openreview_one_sentence_summary": "openreview_one_sentence_summary_titlematch",
        "openreview_pdf_url": "openreview_pdf_url_titlematch",
        "openreview_reviewed_pdf_url": "openreview_reviewed_pdf_url_titlematch",
        "openreview_supplementary_url": "openreview_supplementary_url_titlematch",
        "openreview_bibtex": "openreview_bibtex_titlematch",
        "openreview_paperhash": "openreview_paperhash_titlematch",
        "openreview_venue": "openreview_venue_titlematch",
        "openreview_venueid": "openreview_venueid_titlematch",
        "openreview_created_at": "openreview_created_at_titlematch",
        "openreview_tcdate_at": "openreview_tcdate_at_titlematch",
        "openreview_last_modified_at": "openreview_last_modified_at_titlematch",
        "openreview_pdate_at": "openreview_pdate_at_titlematch",
        "openreview_odate_at": "openreview_odate_at_titlematch",
        "openreview_api_type": "openreview_api_type_titlematch",
        "openreview_domain": "openreview_domain_titlematch",
        "openreview_query_invitation": "openreview_query_invitation_titlematch",
    }
    for target_col, source_col in fill_map.items():
        if source_col not in joined_df.columns:
            continue
        joined_df.loc[unmatched_mask, target_col] = joined_df.loc[unmatched_mask, source_col]
    joined_df.loc[unmatched_mask, "paper_id"] = joined_df.loc[unmatched_mask, "paper_id"]
    joined_df.loc[unmatched_mask, "openreview_match_source"] = "year_title_exact"

    titlematch_drop_cols = [col for col in joined_df.columns if col.endswith("_titlematch")] + ["matched_paper_id_from_title"]
    joined_df = joined_df.drop(columns=[col for col in titlematch_drop_cols if col in joined_df.columns])

    joined_output_path = args.raw_dir / f"{args.input_csv.stem}_with_openreview_yearly_submissions.csv"
    merged_full_df = input_df.copy()
    merged_full_df["input_title_norm"] = match_input["input_title_norm"]
    join_cols = [col for col in joined_df.columns if col not in {"input_title", "input_year", "source_id"}]
    merged_full_df = merged_full_df.merge(joined_df[join_cols], on=["paper_id", "input_title_norm"], how="left")
    merged_full_df.to_csv(joined_output_path, index=False)

    summary = {
        "years_requested": years,
        "raw_dir": str(args.raw_dir.resolve()),
        "all_notes_path": str(all_notes_path.resolve()),
        "page_manifest_path": str(page_manifest_path.resolve()),
        "input_csv": str(args.input_csv.resolve()),
        "joined_output_path": str(joined_output_path.resolve()),
        "n_openreview_rows": int(len(openreview_df)),
        "n_input_rows": int(len(input_df)),
        "n_joined_by_paper_id_or_title": int(merged_full_df["openreview_note_id"].notna().sum()),
        "n_joined_by_paper_id": int((merged_full_df["openreview_match_source"] == "paper_id").sum()),
        "n_joined_by_year_title_exact": int((merged_full_df["openreview_match_source"] == "year_title_exact").sum()),
        "page_counts_by_year": (
            page_manifest_df.groupby("year")["offset"].count().reset_index(name="n_pages").to_dict(orient="records")
            if not page_manifest_df.empty
            else []
        ),
        "note_counts_by_year": (
            openreview_df.groupby("query_year")["openreview_note_id"].count().reset_index(name="n_notes").to_dict(orient="records")
            if not openreview_df.empty
            else []
        ),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote yearly OpenReview submissions to {all_notes_path}")
    print(f"Wrote page manifest to {page_manifest_path}")
    print(f"Wrote joined input output to {joined_output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
