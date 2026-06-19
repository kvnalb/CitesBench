#!/usr/bin/env python3
"""
Second-pass fuzzy matching against a local arXiv metadata dump.

This script starts from the exact-title dump match outputs and only processes
papers that remain unmatched. It uses conservative blocking keys derived from
the unmatched paper titles, scans the local arXiv dump once, and records:

- high-confidence fuzzy matches that are safe to auto-promote
- lower-confidence review candidates for manual inspection

The goal is to extend coverage without falling back to live APIs.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_arxiv_metadata import normalize_title, title_similarity, token_jaccard, write_json
from match_arxiv_metadata_dump import (
    list_dump_files,
    iter_jsonl_rows,
    iter_parquet_rows,
    normalize_dump_row,
    dump_file_type,
)

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_INPUT_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
DEFAULT_RAW_DIR = ROOT / "OutputNew" / "rawdata" / "Design" / "arXiv"
DEFAULT_DUMP_PATH = DEFAULT_RAW_DIR / "dump" / "hf_snapshot"
DEFAULT_EXACT_MATCH_PATH = DEFAULT_RAW_DIR / "arxiv_dump_best_matches.csv"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
    "without",
    "using",
    "toward",
    "towards",
    "through",
    "over",
    "under",
    "new",
    "simple",
    "efficient",
    "general",
    "robust",
    "data",
    "model",
    "models",
    "learning",
    "learn",
    "neural",
    "network",
    "networks",
    "language",
    "large",
    "based",
    "approach",
    "approaches",
    "study",
    "studies",
    "understanding",
    "foundation",
    "foundational",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a second-pass fuzzy/blocking match against the local arXiv dump."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input paper-level CSV. Defaults to the year-specific RDD sample.",
    )
    parser.add_argument(
        "--exact-match-path",
        type=Path,
        default=DEFAULT_EXACT_MATCH_PATH,
        help="Path to the exact-title dump match output.",
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=DEFAULT_DUMP_PATH,
        help="Path to the local arXiv dump directory or file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where fuzzy dump-match outputs are written.",
    )
    parser.add_argument(
        "--limit-unmatched",
        type=int,
        default=None,
        help="Optional limit on the number of unmatched rows to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Row batch size when scanning Parquet shards.",
    )
    parser.add_argument(
        "--progress-every-files",
        type=int,
        default=2,
        help="Print a progress line every N dump files scanned.",
    )
    parser.add_argument(
        "--anchors-per-title",
        type=int,
        default=3,
        help="Maximum anchor tokens retained per unmatched title.",
    )
    parser.add_argument(
        "--max-anchor-doc-freq",
        type=int,
        default=40,
        help="Ignore candidate anchor tokens appearing in more than this many unmatched titles.",
    )
    parser.add_argument(
        "--candidate-min-similarity",
        type=float,
        default=0.80,
        help="Minimum title similarity for a fuzzy candidate to be retained.",
    )
    parser.add_argument(
        "--candidate-min-containment",
        type=float,
        default=0.72,
        help="Minimum token containment for a fuzzy candidate to be retained.",
    )
    parser.add_argument(
        "--candidate-min-overlap",
        type=int,
        default=2,
        help="Minimum shared token count for a fuzzy candidate to be retained.",
    )
    parser.add_argument(
        "--review-min-similarity",
        type=float,
        default=0.88,
        help="Minimum similarity for a review-worthy fuzzy suggestion.",
    )
    parser.add_argument(
        "--review-min-containment",
        type=float,
        default=0.78,
        help="Minimum token containment for a review-worthy fuzzy suggestion.",
    )
    parser.add_argument(
        "--auto-min-similarity",
        type=float,
        default=0.94,
        help="Minimum similarity for an automatic fuzzy promotion.",
    )
    parser.add_argument(
        "--auto-min-containment",
        type=float,
        default=0.88,
        help="Minimum token containment for an automatic fuzzy promotion.",
    )
    parser.add_argument(
        "--auto-max-year-diff",
        type=int,
        default=4,
        help="Maximum year difference for an automatic fuzzy promotion when both years exist.",
    )
    parser.add_argument(
        "--auto-min-gap",
        type=float,
        default=0.03,
        help="Minimum score gap between the best and second-best fuzzy candidates for auto promotion.",
    )
    return parser.parse_args()


def canonicalize_token(token: str) -> str:
    text = token.strip().lower()
    if len(text) > 5 and text.endswith("ies"):
        return text[:-3] + "y"
    if len(text) > 4 and text.endswith("es"):
        return text[:-2]
    if len(text) > 3 and text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text


def normalized_title_tokens(title_norm: str) -> list[str]:
    return [canonicalize_token(token) for token in title_norm.split() if token]


def informative_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if len(token) >= 3 and token not in STOPWORDS]


def title_head_norm(title: object) -> str:
    text = "" if title is None else str(title)
    return normalize_title(text.split(":", 1)[0])


def lead_informative_token(title_norm: str) -> str | None:
    tokens = informative_tokens(normalized_title_tokens(title_norm))
    if tokens:
        return tokens[0]
    all_tokens = normalized_title_tokens(title_norm)
    if all_tokens:
        return all_tokens[0]
    return None


def load_unmatched_rows(path: Path, limit_unmatched: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Exact match path does not exist: {path}")
    df = pd.read_csv(path)
    unmatched = df.loc[~df["matched"].fillna(False)].copy()
    unmatched = unmatched.rename(
        columns={
            "input_title": "title",
            "input_year": "year",
        }
    )
    unmatched = unmatched[["paper_id", "source_id", "title", "year", "input_title_norm"]].copy()
    unmatched["year"] = pd.to_numeric(unmatched["year"], errors="coerce").astype("Int64")
    if limit_unmatched is not None:
        unmatched = unmatched.head(limit_unmatched).copy()
    return unmatched.reset_index(drop=True)


def build_blocking_index(
    unmatched_rows: pd.DataFrame,
    anchors_per_title: int,
    max_anchor_doc_freq: int,
) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[str]], dict[str, dict[str, object]], dict[str, object]]:
    token_df: Counter[str] = Counter()
    paper_payload: dict[str, dict[str, object]] = {}

    for row in unmatched_rows.itertuples(index=False):
        title_norm = row.input_title_norm if isinstance(row.input_title_norm, str) else normalize_title(row.title)
        tokens = normalized_title_tokens(title_norm)
        informative = informative_tokens(tokens)
        token_df.update(set(informative))
        paper_payload[row.paper_id] = {
            "paper_id": row.paper_id,
            "source_id": row.source_id,
            "input_title": row.title,
            "input_title_norm": title_norm,
            "input_year": int(row.year) if pd.notna(row.year) else None,
            "title_tokens": tokens,
            "title_token_set": set(tokens),
        }

    pair_blocking_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    single_blocking_index: dict[str, list[str]] = defaultdict(list)
    papers_without_good_anchors = 0

    for paper_id, payload in paper_payload.items():
        informative = [
            token
            for token in payload["title_tokens"]
            if token not in STOPWORDS and len(token) >= 5 and token_df.get(token, 0) <= max_anchor_doc_freq
        ]
        if not informative:
            informative = [
                token
                for token in payload["title_tokens"]
                if token not in STOPWORDS and len(token) >= 3
            ]
        if not informative:
            informative = payload["title_tokens"]

        anchors = sorted(
            set(informative),
            key=lambda token: (token_df.get(token, 10**9), -len(token), token),
        )[:anchors_per_title]
        if not anchors:
            papers_without_good_anchors += 1
        payload["anchor_tokens"] = anchors
        if len(anchors) >= 2:
            for left, right in itertools.combinations(sorted(set(anchors)), 2):
                pair_blocking_index[(left, right)].append(paper_id)
        else:
            for anchor in anchors:
                single_blocking_index[anchor].append(paper_id)

    summary = {
        "n_unmatched_rows": int(len(unmatched_rows)),
        "n_pair_blocking_keys": int(len(pair_blocking_index)),
        "n_single_blocking_keys": int(len(single_blocking_index)),
        "n_papers_without_good_anchors": int(papers_without_good_anchors),
        "max_anchor_doc_freq": int(max_anchor_doc_freq),
        "anchors_per_title": int(anchors_per_title),
    }
    return pair_blocking_index, single_blocking_index, paper_payload, summary


def year_penalty(year_diff: int | None) -> float:
    if year_diff is None:
        return 0.0
    return max(0, year_diff - 1) * 0.03


def build_candidate_row(
    payload: dict[str, object],
    normalized: dict[str, object],
    candidate_title_norm: str,
    candidate_tokens: list[str],
    shared_anchor_tokens: list[str],
    args: argparse.Namespace,
) -> dict[str, object] | None:
    input_token_set = payload["title_token_set"]
    candidate_token_set = set(candidate_tokens)
    overlap_tokens = sorted(input_token_set & candidate_token_set)
    overlap_count = len(overlap_tokens)
    min_token_count = min(len(input_token_set), len(candidate_token_set))
    token_containment = overlap_count / min_token_count if min_token_count else 0.0
    input_token_recall = overlap_count / len(input_token_set) if input_token_set else 0.0
    similarity = title_similarity(str(payload["input_title_norm"]), candidate_title_norm)
    jaccard = token_jaccard(str(payload["input_title_norm"]), candidate_title_norm)
    input_head = title_head_norm(payload["input_title"])
    candidate_head = title_head_norm(normalized["arxiv_title"])
    head_similarity = title_similarity(input_head, candidate_head)
    input_lead_token = lead_informative_token(str(payload["input_title_norm"]))
    candidate_lead_token = lead_informative_token(candidate_title_norm)

    candidate_year = normalized["arxiv_first_version_year"]
    input_year = payload["input_year"]
    year_diff = abs(input_year - candidate_year) if input_year is not None and candidate_year is not None else None

    min_overlap = 1 if min_token_count <= 2 else args.candidate_min_overlap
    if overlap_count < min_overlap:
        return None
    if similarity < args.candidate_min_similarity and token_containment < args.candidate_min_containment:
        return None

    score = (
        (1.35 * similarity)
        + (0.55 * token_containment)
        + (0.25 * input_token_recall)
        + (0.20 * jaccard)
        + (0.04 * min(overlap_count, 5))
        - year_penalty(year_diff)
    )

    return {
        "paper_id": payload["paper_id"],
        "source_id": payload["source_id"],
        "input_title": payload["input_title"],
        "input_title_norm": payload["input_title_norm"],
        "input_year": payload["input_year"],
        "anchor_tokens": " | ".join(payload["anchor_tokens"]),
        "shared_anchor_tokens": " | ".join(shared_anchor_tokens),
        "shared_tokens": " | ".join(overlap_tokens),
        "shared_token_count": overlap_count,
        "token_containment": token_containment,
        "input_token_recall": input_token_recall,
        "title_similarity": similarity,
        "input_head_norm": input_head,
        "candidate_head_norm": candidate_head,
        "head_similarity": head_similarity,
        "input_lead_token": input_lead_token,
        "candidate_lead_token": candidate_lead_token,
        "lead_token_match": input_lead_token == candidate_lead_token if input_lead_token and candidate_lead_token else pd.NA,
        "token_jaccard": jaccard,
        "year_diff": year_diff,
        "published_year": candidate_year,
        "candidate_score": score,
        "match_source": "dump_fuzzy_blocking",
        **normalized,
    }


def scan_dump_for_fuzzy_candidates(
    dump_files: list[Path],
    pair_blocking_index: dict[tuple[str, str], list[str]],
    single_blocking_index: dict[str, list[str]],
    paper_payload: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate_rows: list[dict[str, object]] = []
    scanned_rows = 0
    blocked_title_hits = 0
    pair_blocking_vocab = {
        token
        for pair in pair_blocking_index.keys()
        for token in pair
    }

    for file_number, dump_file in enumerate(dump_files, start=1):
        file_type = dump_file_type(dump_file)
        row_iter = (
            iter_parquet_rows(dump_file, batch_size=args.batch_size)
            if file_type == "parquet"
            else iter_jsonl_rows(dump_file)
        )
        for raw_row in row_iter:
            scanned_rows += 1
            normalized = normalize_dump_row(raw_row, dump_file=dump_file)
            if not normalized["arxiv_title"]:
                continue

            candidate_title_norm = normalize_title(normalized["arxiv_title"])
            candidate_tokens = informative_tokens(normalized_title_tokens(candidate_title_norm))
            if not candidate_tokens:
                candidate_tokens = normalized_title_tokens(candidate_title_norm)
            if not candidate_tokens:
                continue

            candidate_token_set = set(candidate_tokens)
            matched_papers: dict[str, list[str]] = defaultdict(list)

            relevant_pair_tokens = sorted(token for token in candidate_token_set if token in pair_blocking_vocab)
            for left, right in itertools.combinations(relevant_pair_tokens, 2):
                key = (left, right)
                if key not in pair_blocking_index:
                    continue
                for paper_id in pair_blocking_index[key]:
                    matched_papers[paper_id].extend([left, right])

            for token in candidate_token_set:
                for paper_id in single_blocking_index.get(token, []):
                    matched_papers[paper_id].append(token)

            if not matched_papers:
                continue

            blocked_title_hits += 1
            for paper_id, shared_anchor_tokens in matched_papers.items():
                shared_anchor_tokens = sorted(set(shared_anchor_tokens))
                required_anchor_overlap = 1 if len(paper_payload[paper_id]["anchor_tokens"]) <= 1 else 2
                if len(shared_anchor_tokens) < required_anchor_overlap:
                    continue
                payload = paper_payload[paper_id]
                candidate_row = build_candidate_row(
                    payload=payload,
                    normalized=normalized,
                    candidate_title_norm=candidate_title_norm,
                    candidate_tokens=candidate_tokens,
                    shared_anchor_tokens=shared_anchor_tokens,
                    args=args,
                )
                if candidate_row is not None:
                    candidate_rows.append(candidate_row)

        if args.progress_every_files > 0 and (
            file_number == 1 or file_number % args.progress_every_files == 0 or file_number == len(dump_files)
        ):
            print(
                (
                    f"[fuzzy-scan] files={file_number}/{len(dump_files)} "
                    f"scanned_rows={scanned_rows} blocked_hits={blocked_title_hits} "
                    f"candidate_rows={len(candidate_rows)}"
                ),
                flush=True,
            )

    summary = {
        "n_dump_files": int(len(dump_files)),
        "n_scanned_rows": int(scanned_rows),
        "n_blocked_title_hits": int(blocked_title_hits),
        "n_candidate_rows": int(len(candidate_rows)),
    }
    return pd.DataFrame(candidate_rows), summary


def classify_best_fuzzy_matches(candidate_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(columns=["paper_id"])

    ranked = candidate_df.sort_values(
        ["paper_id", "candidate_score", "title_similarity", "token_containment", "year_diff", "arxiv_id"],
        ascending=[True, False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["rank_within_paper"] = ranked.groupby("paper_id").cumcount() + 1

    best = ranked.loc[ranked["rank_within_paper"] == 1].copy()
    second = (
        ranked.loc[ranked["rank_within_paper"] == 2, ["paper_id", "candidate_score"]]
        .rename(columns={"candidate_score": "second_best_score"})
        .copy()
    )
    best = best.merge(second, on="paper_id", how="left")
    best["confidence_gap"] = best["candidate_score"] - best["second_best_score"].fillna(float("-inf"))
    best.loc[best["second_best_score"].isna(), "confidence_gap"] = pd.NA

    auto_condition = (
        (best["title_similarity"] >= args.auto_min_similarity)
        & (best["token_containment"] >= args.auto_min_containment)
        & (best["shared_token_count"] >= 2)
        & (best["head_similarity"] >= 0.85)
        & (best["lead_token_match"].fillna(False))
        & (
            best["year_diff"].isna()
            | (best["year_diff"] <= args.auto_max_year_diff)
        )
        & (
            best["second_best_score"].isna()
            | (best["candidate_score"] - best["second_best_score"] >= args.auto_min_gap)
        )
    )
    review_condition = (
        (best["title_similarity"] >= args.review_min_similarity)
        & (best["token_containment"] >= args.review_min_containment)
        & (best["shared_token_count"] >= 2)
    )

    best["matched"] = auto_condition
    best["match_status"] = "fuzzy_low_confidence"
    best.loc[review_condition, "match_status"] = "fuzzy_review_candidate"
    best.loc[auto_condition, "match_status"] = "fuzzy_high_confidence"
    best["best_query_attempt"] = "local_dump_fuzzy_blocking"
    best["attempt_status"] = "local_dump_fuzzy_scanned"
    best["total_entries_seen"] = 1
    return best


def build_combined_matches(exact_best: pd.DataFrame, fuzzy_best: pd.DataFrame) -> pd.DataFrame:
    combined = exact_best.copy()
    if fuzzy_best.empty:
        combined["matched"] = combined["matched"].fillna(False).astype(bool)
        return combined

    fuzzy_lookup = fuzzy_best.set_index("paper_id")
    common_columns = [column for column in combined.columns if column in fuzzy_lookup.columns]
    for column in common_columns:
        combined[column] = combined[column].astype(object)

    for row in combined.itertuples():
        if bool(getattr(row, "matched")):
            continue
        if row.paper_id not in fuzzy_lookup.index:
            continue
        fuzzy_row = fuzzy_lookup.loc[row.paper_id]
        for column in common_columns:
            combined.loc[combined["paper_id"] == row.paper_id, column] = fuzzy_row[column]

    combined["matched"] = combined["matched"].fillna(False).astype(bool)
    return combined


def write_outputs(
    args: argparse.Namespace,
    candidate_df: pd.DataFrame,
    fuzzy_best: pd.DataFrame,
    combined_best: pd.DataFrame,
    scan_summary: dict[str, object],
    blocking_summary: dict[str, object],
) -> None:
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.raw_dir / "arxiv_dump_fuzzy_candidate_matches.csv"
    best_path = args.raw_dir / "arxiv_dump_fuzzy_best_matches.csv"
    combined_path = args.raw_dir / "arxiv_dump_combined_best_matches.csv"
    enriched_path = args.raw_dir / f"{args.input_csv.stem}_with_arxiv_dump_combined_match.csv"
    summary_path = args.raw_dir / "arxiv_dump_fuzzy_match_summary.json"

    candidate_df.to_csv(candidate_path, index=False)
    fuzzy_best.to_csv(best_path, index=False)
    combined_best.to_csv(combined_path, index=False)
    combined_best.to_csv(enriched_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv),
        "exact_match_path": str(args.exact_match_path),
        "dump_path": str(args.dump_path),
        "raw_dir": str(args.raw_dir),
        "n_fuzzy_candidate_rows": int(len(candidate_df)),
        "n_fuzzy_best_rows": int(len(fuzzy_best)),
        "n_fuzzy_high_confidence": int((fuzzy_best["match_status"] == "fuzzy_high_confidence").sum())
        if not fuzzy_best.empty
        else 0,
        "n_fuzzy_review_candidates": int((fuzzy_best["match_status"] == "fuzzy_review_candidate").sum())
        if not fuzzy_best.empty
        else 0,
        "n_combined_matched_rows": int(combined_best["matched"].sum()) if not combined_best.empty else 0,
        "n_combined_unmatched_rows": int(len(combined_best) - combined_best["matched"].sum())
        if not combined_best.empty
        else 0,
        **blocking_summary,
        **scan_summary,
    }
    write_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    exact_best = pd.read_csv(args.exact_match_path)
    unmatched_rows = load_unmatched_rows(args.exact_match_path, limit_unmatched=args.limit_unmatched)
    dump_files = list_dump_files(args.dump_path)

    pair_blocking_index, single_blocking_index, paper_payload, blocking_summary = build_blocking_index(
        unmatched_rows=unmatched_rows,
        anchors_per_title=args.anchors_per_title,
        max_anchor_doc_freq=args.max_anchor_doc_freq,
    )
    candidate_df, scan_summary = scan_dump_for_fuzzy_candidates(
        dump_files=dump_files,
        pair_blocking_index=pair_blocking_index,
        single_blocking_index=single_blocking_index,
        paper_payload=paper_payload,
        args=args,
    )
    fuzzy_best = classify_best_fuzzy_matches(candidate_df=candidate_df, args=args)
    combined_best = build_combined_matches(exact_best=exact_best, fuzzy_best=fuzzy_best)
    write_outputs(
        args=args,
        candidate_df=candidate_df,
        fuzzy_best=fuzzy_best,
        combined_best=combined_best,
        scan_summary=scan_summary,
        blocking_summary=blocking_summary,
    )


if __name__ == "__main__":
    main()
