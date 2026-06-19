#!/usr/bin/env python3
"""Aggregate Human x Persona overlap by inferred human reviewer bucket.

This script is intentionally post-hoc: it does not recompute embeddings. It
joins the local embedding overlap rows from 15_eval_human_persona_embedding_overlap.py
to human-reviewer persona labels inferred by Code/reviewer_persona_classifier.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_INPUT = ROOT / "OutputNew" / "Empirics" / "human_persona_embedding_overlap_all_available_t050_20260421" / "human_persona_pairs.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "Empirics" / "human_bucket_persona_overlap_all_available_t050_20260421"
DEFAULT_REVIEW_DB = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
CLASSIFIER_PATH = ROOT / "Code" / "reviewer_persona_classifier.py"
FOCUS_BUCKETS = {"empiricist", "theorist", "systems_pragmatist", "novelty_gatekeeper"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket-level Human x Persona overlap aggregation.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-unscored", action="store_true", help="Write unscored joined rows too.")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_classifier() -> Any:
    spec = importlib.util.spec_from_file_location("reviewer_persona_classifier", CLASSIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load classifier: {CLASSIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


classifier = load_classifier()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_review_rows(db_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                paper_id,
                reviewer_id,
                summary,
                strength,
                weaknesses,
                questions,
                main_review,
                summary_of_the_review,
                rating,
                confidence
            FROM REVIEW
            """
        ).fetchall()
    finally:
        conn.close()
    return {(str(row["paper_id"]), str(row["reviewer_id"])): dict(row) for row in rows}


def classify_db_review(review: dict[str, Any] | None) -> dict[str, Any]:
    if not review:
        return {
            "human_bucket": "missing",
            "human_bucket_confident": False,
            "human_bucket_scores": {},
            "classification_text_source": "missing",
        }

    weaknesses = clean_text(review.get("weaknesses"))
    questions = clean_text(review.get("questions"))
    main_review = clean_text(review.get("main_review"))
    summary_review = clean_text(review.get("summary_of_the_review"))

    if weaknesses or questions:
        label, scores, confident = classifier.classify_reviewer(weaknesses, questions)
        source = "weaknesses_questions"
    elif main_review or summary_review:
        # Older ICLR exports often have a single free-form review field rather
        # than structured weaknesses/questions. This is less precise, but avoids
        # forcing all older reviews into generic.
        label, scores, confident = classifier.classify_reviewer(main_review or summary_review, "")
        source = "main_review"
    else:
        label, scores, confident = "no_text", {}, False
        source = "no_text"

    return {
        "human_bucket": label,
        "human_bucket_confident": bool(confident),
        "human_bucket_scores": scores,
        "classification_text_source": source,
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("balanced_overlap") is not None]
    return {
        "n_pairs": len(rows),
        "n_scored_pairs": len(scored),
        "mean_balanced_overlap": mean([float(row["balanced_overlap"]) for row in scored]),
        "mean_human_to_persona": mean([float(row["human_to_persona"]["rate"]) for row in scored if row["human_to_persona"]["rate"] is not None]),
        "mean_persona_to_human": mean([float(row["persona_to_human"]["rate"]) for row in scored if row["persona_to_human"]["rate"] is not None]),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("balanced_overlap") is not None]
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_persona: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    diagonal_rows: list[dict[str, Any]] = []
    off_diagonal_rows: list[dict[str, Any]] = []

    for row in rows:
        bucket = str(row.get("human_bucket") or "missing")
        persona = str(row.get("persona") or "missing")
        by_bucket[bucket].append(row)
        by_persona[persona].append(row)
        by_cell[(bucket, persona)].append(row)
        if bucket in FOCUS_BUCKETS and persona in FOCUS_BUCKETS:
            if bucket == persona:
                diagonal_rows.append(row)
            else:
                off_diagonal_rows.append(row)

    matrix: dict[str, dict[str, Any]] = {}
    for (bucket, persona), group_rows in sorted(by_cell.items()):
        matrix.setdefault(bucket, {})[persona] = summarize_group(group_rows)

    bucket_summary = {bucket: summarize_group(group_rows) for bucket, group_rows in sorted(by_bucket.items())}
    persona_summary = {persona: summarize_group(group_rows) for persona, group_rows in sorted(by_persona.items())}

    scored_sorted = sorted(
        scored,
        key=lambda row: (float(row["balanced_overlap"]), row["paper_id"], row["reviewer_id"], row["persona"]),
    )
    return {
        "created_at_utc": now_utc(),
        "n_pairs": len(rows),
        "n_scored_pairs": len(scored),
        "n_unique_papers": len({row["paper_id"] for row in rows}),
        "n_unique_human_reviews": len({(row["paper_id"], row["reviewer_id"]) for row in rows}),
        "human_bucket_distribution_pairs": dict(Counter(str(row.get("human_bucket") or "missing") for row in rows)),
        "human_bucket_distribution_reviews": dict(
            Counter(
                str(row.get("human_bucket") or "missing")
                for row in {
                    (item["paper_id"], item["reviewer_id"]): item
                    for item in rows
                }.values()
            )
        ),
        "classification_text_sources_pairs": dict(Counter(str(row.get("classification_text_source") or "missing") for row in rows)),
        "overall": summarize_group(rows),
        "diagonal_focus_buckets": summarize_group(diagonal_rows),
        "off_diagonal_focus_buckets": summarize_group(off_diagonal_rows),
        "bucket_summary": bucket_summary,
        "generated_persona_summary": persona_summary,
        "bucket_by_generated_persona": matrix,
        "lowest_pairs": scored_sorted[:10],
        "highest_pairs": list(reversed(scored_sorted[-10:])),
    }


def format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    personas = sorted(summary["generated_persona_summary"])
    buckets = sorted(summary["bucket_summary"])
    lines = [
        "# Human Bucket x Generated Persona Overlap",
        "",
        f"- Created: {summary['created_at_utc']}",
        f"- Pairs scored: {summary['n_scored_pairs']} / {summary['n_pairs']}",
        f"- Papers: {summary['n_unique_papers']}",
        f"- Human reviews: {summary['n_unique_human_reviews']}",
        f"- Overall balanced overlap: {format_float(summary['overall']['mean_balanced_overlap'])}",
        "",
        "## Diagonal Check",
        "",
        "| Group | Pairs | Scored | Balanced | Human->Persona | Persona->Human |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Same bucket", "diagonal_focus_buckets"),
        ("Different focus bucket", "off_diagonal_focus_buckets"),
    ):
        row = summary[key]
        lines.append(
            f"| {label} | {row['n_pairs']} | {row['n_scored_pairs']} | "
            f"{format_float(row['mean_balanced_overlap'])} | "
            f"{format_float(row['mean_human_to_persona'])} | "
            f"{format_float(row['mean_persona_to_human'])} |"
        )

    lines.extend(
        [
            "",
            "## Human Bucket Distribution",
            "",
            "| Human bucket | Reviews | Pairs | Balanced | Human->Persona | Persona->Human |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket in buckets:
        row = summary["bucket_summary"][bucket]
        review_count = summary["human_bucket_distribution_reviews"].get(bucket, 0)
        lines.append(
            f"| {bucket} | {review_count} | {row['n_pairs']} | "
            f"{format_float(row['mean_balanced_overlap'])} | "
            f"{format_float(row['mean_human_to_persona'])} | "
            f"{format_float(row['mean_persona_to_human'])} |"
        )

    lines.extend(
        [
            "",
            "## Bucket x Generated Persona Matrix",
            "",
            "| Human bucket | Generated persona | Pairs | Scored | Balanced | Human->Persona | Persona->Human |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket in buckets:
        for persona in personas:
            row = summary["bucket_by_generated_persona"].get(bucket, {}).get(persona)
            if not row:
                continue
            lines.append(
                f"| {bucket} | {persona} | {row['n_pairs']} | {row['n_scored_pairs']} | "
                f"{format_float(row['mean_balanced_overlap'])} | "
                f"{format_float(row['mean_human_to_persona'])} | "
                f"{format_float(row['mean_persona_to_human'])} |"
            )

    for label, key in (("Lowest Pairs", "lowest_pairs"), ("Highest Pairs", "highest_pairs")):
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "| Paper | Human bucket | Reviewer | Generated persona | Balanced | Human->Persona | Persona->Human |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in summary[key]:
            lines.append(
                f"| {row['paper_id']} | {row['human_bucket']} | {row['reviewer_id']} | {row['persona']} | "
                f"{format_float(row['balanced_overlap'])} | "
                f"{format_float(row['human_to_persona']['rate'])} | "
                f"{format_float(row['persona_to_human']['rate'])} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overlap_rows = read_jsonl(args.input_jsonl)
    review_rows = load_review_rows(args.review_db)
    label_cache: dict[tuple[str, str], dict[str, Any]] = {}
    joined_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for row in overlap_rows:
        key = (str(row["paper_id"]), str(row["reviewer_id"]))
        if key not in label_cache:
            label = classify_db_review(review_rows.get(key))
            label_cache[key] = label
            label_rows.append({"paper_id": key[0], "reviewer_id": key[1], **label})
        joined = {**row, **label_cache[key]}
        if args.include_unscored or joined.get("balanced_overlap") is not None:
            joined_rows.append(joined)

    summary = aggregate(joined_rows)
    write_json(
        args.output_dir / "run_config.json",
        {
            "created_at_utc": now_utc(),
            "input_jsonl": str(args.input_jsonl),
            "review_db": str(args.review_db),
            "classifier": str(CLASSIFIER_PATH),
            "include_unscored": args.include_unscored,
        },
    )
    write_jsonl(args.output_dir / "human_review_bucket_labels.jsonl", sorted(label_rows, key=lambda item: (item["paper_id"], item["reviewer_id"])))
    write_jsonl(args.output_dir / "bucket_persona_pairs.jsonl", joined_rows)
    write_json(args.output_dir / "summary.json", summary)
    write_summary_md(args.output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
