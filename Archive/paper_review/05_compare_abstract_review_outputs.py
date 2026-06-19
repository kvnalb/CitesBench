#!/usr/bin/env python3
"""
Compare previously generated abstract-review outputs against human reviews.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _abstract_review_common import (
    DEFAULT_HUMAN_REVIEW_DIR,
    ModelSpec,
    build_comparison,
    leaderboard_markdown,
    load_human_reviews,
    now_utc,
    read_jsonl,
    slugify,
    summarise_model,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a saved LLMOutput generation run against local human-review scores."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to a generation run directory under LLMOutput/.")
    parser.add_argument("--human-review-dir", type=Path, default=DEFAULT_HUMAN_REVIEW_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit comparison directory. Defaults to <run-dir>/comparison.",
    )
    parser.add_argument(
        "--require-human-review",
        action="store_true",
        help="Drop papers that do not have a local human-review JSON instead of marking them unavailable.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def infer_model_spec(rows: list[dict], fallback_path: Path) -> ModelSpec:
    if rows:
        variant_info = rows[0].get("run_variant", {})
        if variant_info.get("id"):
            return ModelSpec(
                model_id=str(variant_info.get("id", fallback_path.stem)),
                label=str(variant_info.get("label", fallback_path.stem)),
            )
        model_info = rows[0].get("model", {})
        return ModelSpec(
            model_id=str(model_info.get("id", fallback_path.stem)),
            label=str(model_info.get("label", fallback_path.stem)),
        )
    return ModelSpec(model_id=fallback_path.stem, label=fallback_path.stem)


def build_compared_rows(
    response_rows: list[dict],
    human_reviews: dict[str, dict],
    require_human_review: bool,
) -> list[dict]:
    compared_rows = []
    for row in response_rows:
        paper_id = str(row["paper_id"])
        human_review = human_reviews.get(paper_id)
        if require_human_review and human_review is None:
            continue

        compared_row = dict(row)
        compared_row["human_review"] = {
            "available": human_review is not None,
            **(human_review or {}),
        }
        compared_row["comparison"] = build_comparison(
            {
                "decision": row.get("decision"),
                "human_review": human_review,
            },
            row.get("llm_review", {}).get("scores", {}),
        )
        compared_rows.append(compared_row)
    return compared_rows


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    response_dir = run_dir / "responses"
    if not response_dir.exists():
        raise ValueError(f"Response directory not found: {response_dir}")

    manifest_path = run_dir / "run_manifest.json"
    run_manifest = read_json(manifest_path) if manifest_path.exists() else None
    human_reviews = load_human_reviews(args.human_review_dir)

    output_dir = args.output_dir.resolve() if args.output_dir is not None else (run_dir / "comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_paper_dir = output_dir / "per_paper"
    summaries_dir = output_dir / "summaries"

    comparison_manifest = {
        "comparison_created_at_utc": now_utc(),
        "comparison_type": "offline_postprocess",
        "run_dir": str(run_dir),
        "response_dir": str(response_dir),
        "human_review_dir": str(args.human_review_dir),
        "require_human_review": args.require_human_review,
        "source_run_manifest": str(manifest_path) if manifest_path.exists() else None,
        "source_run_type": run_manifest.get("run_type") if run_manifest else None,
    }
    write_json(output_dir / "comparison_manifest.json", comparison_manifest)

    leaderboard_rows = []
    response_files = sorted(response_dir.glob("*.jsonl"))
    if not response_files:
        raise ValueError(f"No response files found in {response_dir}")

    for response_file in response_files:
        response_rows = read_jsonl(response_file)
        model = infer_model_spec(response_rows, response_file)
        compared_rows = build_compared_rows(
            response_rows=response_rows,
            human_reviews=human_reviews,
            require_human_review=args.require_human_review,
        )
        compared_path = per_paper_dir / f"{slugify(model.model_id)}.jsonl"
        write_jsonl(compared_path, compared_rows)
        summary = summarise_model(compared_rows, model, compared_path)
        write_json(summaries_dir / f"{slugify(model.model_id)}.json", summary)
        leaderboard_rows.append(summary)

        print(f"Compared {len(compared_rows)} rows for {model.label}")

    leaderboard_rows.sort(
        key=lambda row: (
            row["nmae_mean"] if row["nmae_mean"] is not None else float("inf"),
            -(row["decision_agreement_pct"] if row["decision_agreement_pct"] is not None else -1.0),
        )
    )
    write_json(output_dir / "leaderboard.json", leaderboard_rows)
    (output_dir / "leaderboard.md").write_text(leaderboard_markdown(leaderboard_rows), encoding="utf-8")
    print(f"Comparison outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
