#!/usr/bin/env python3
"""
Run a direct coarse benchmark on a fixed paper sample and compare the generated
reviews against local human reviews using coarse's own quality evaluator.

This harness supports two review-generation paths:
    - full: upstream coarse.pipeline.review_paper
    - slim: local slim_coarse_pipeline.review_paper_slim

It does not call the local pointwise reviewer pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coarse import __version__ as coarse_version
from coarse.config import CoarseConfig
from coarse.llm import LLMClient
from coarse.pipeline import review_paper
from coarse.quality import evaluate_review, evaluate_review_panel, save_quality_report

from slim_coarse_pipeline import DEFAULT_PERSONA_ENSEMBLE, review_paper_slim


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_PAPERS = ROOT / "LLMOutput" / "coarse_ab_item2_persona_gen" / "selected_papers.jsonl"
DEFAULT_HUMAN_REVIEW_DIR = ROOT / "processed" / "ICLR2025_Foundation_LLMs" / "HumanReview"
DEFAULT_FULLTEXT_DIR = ROOT / "rawdata" / "ICLR2025" / "foundation_or_frontier_models_including_LLMs" / "fulltext"
DEFAULT_OUTPUT_ROOT = ROOT / "Output" / "Coarse" / "slim_benchmark_sample20"
DEFAULT_KEY_FILE = ROOT / "key.txt"

DEFAULT_MODELS = [
    "together_ai/deepseek-ai/DeepSeek-V3.1",
    "together_ai/openai/gpt-oss-20b",
    "together_ai/mistralai/Mistral-Small-24B-Instruct-2501",
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
]
DEFAULT_EVAL_MODEL = "together_ai/Qwen/Qwen3.5-397B-A17B"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_").lower()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def build_config(together_key: str) -> CoarseConfig:
    # coarse's key lookup recognizes "together" but litellm routes Together
    # under the "together_ai" provider prefix. Populate both keys to avoid the
    # mismatch without patching the installed package.
    return CoarseConfig(
        extraction_qa=False,
        api_keys={
            "together": together_key,
            "together_ai": together_key,
        },
    )


def parse_models(raw: str) -> list[str]:
    models = [token.strip() for token in raw.split(",") if token.strip()]
    if not models:
        raise ValueError("At least one model must be provided.")
    return models


def parse_personas(raw: str) -> list[str]:
    personas = [token.strip() for token in raw.split(",") if token.strip()]
    if not personas:
        raise ValueError("At least one persona must be provided.")
    if len(personas) == 1 and personas[0].lower() in {"default-ensemble", "committee4"}:
        return list(DEFAULT_PERSONA_ENSEMBLE)
    return personas


def parse_weights(raw: str | None, personas: list[str]) -> dict[str, float] | None:
    if raw is None or not raw.strip():
        return None
    weights: dict[str, float] = {}
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid weight token '{token}'. Expected persona=weight.")
        persona, raw_value = token.split("=", 1)
        value = float(raw_value.strip())
        if value <= 0:
            raise ValueError("Weights must be positive.")
        weights[persona.strip()] = value
    missing = [persona for persona in personas if persona not in weights]
    if missing:
        raise ValueError(f"Missing weights for personas: {', '.join(missing)}")
    return weights


def load_sample(
    path: Path,
    max_papers: int | None,
    requested_ids: set[str] | None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    selected = []
    for row in rows:
        paper_id = str(row["paper_id"])
        if requested_ids is not None and paper_id not in requested_ids:
            continue
        selected.append(row)
    selected.sort(key=lambda row: str(row["paper_id"]))
    if max_papers is not None:
        selected = selected[:max_papers]
    return selected


def load_human_review(review_dir: Path, paper_id: str) -> dict[str, Any]:
    path = review_dir / f"{paper_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Human review not found for {paper_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_score_block(scores: dict[str, Any]) -> str:
    parts = []
    for key in ("rating", "confidence", "soundness", "presentation", "contribution"):
        if key in scores:
            parts.append(f"- {key}: {scores[key]}")
    return "\n".join(parts)


def render_human_reference(review: dict[str, Any]) -> str:
    lines = [
        f"# Human Review Reference: {review.get('title', '')}",
        "",
        f"- Paper ID: {review.get('paper_id', '')}",
        f"- Decision: {review.get('decision', '')}",
        f"- Number of reviews: {review.get('n_reviews', '')}",
        "",
        "## Aggregate Scores",
        "",
    ]

    aggregated = review.get("aggregated", {}) or {}
    for key in ("rating", "confidence", "soundness", "presentation", "contribution"):
        stats = aggregated.get(key) or {}
        if not stats:
            continue
        values = stats.get("values")
        lines.extend(
            [
                f"### {key}",
                f"- mean: {stats.get('mean')}",
                f"- std: {stats.get('std')}",
                f"- min: {stats.get('min')}",
                f"- max: {stats.get('max')}",
                f"- count: {stats.get('count')}",
                f"- values: {values}",
                "",
            ]
        )

    lines.extend(["## Individual Reviewer Reports", ""])
    for idx, reviewer in enumerate(review.get("reviews", []) or [], start=1):
        lines.append(f"### Reviewer {idx}")
        lines.append("")
        lines.append(f"- reviewer_id: {reviewer.get('reviewer_id', '')}")
        scores_block = format_score_block(reviewer.get("scores", {}) or {})
        if scores_block:
            lines.extend(["#### Scores", scores_block, ""])

        review_text = (reviewer.get("review_text") or "").strip()
        if review_text:
            lines.extend(["#### Review Text", review_text, ""])
        else:
            review_sections = reviewer.get("review_sections", {}) or {}
            for section_name in ("summary", "strength", "weaknesses", "questions"):
                section_text = (review_sections.get(section_name) or "").strip()
                if section_text:
                    lines.extend([f"#### {section_name.title()}", section_text, ""])

    return "\n".join(lines).strip() + "\n"


def quality_dims(report: Any) -> dict[str, float]:
    return {dim.dimension: dim.score for dim in report.dimensions}


@dataclass
class RunResult:
    row: dict[str, Any]
    review_markdown_path: Path
    quality_markdown_path: Path


def run_one(
    *,
    paper: dict[str, Any],
    human_review: dict[str, Any],
    model: str,
    eval_model: str,
    output_root: Path,
    config: CoarseConfig,
    pipeline_mode: str,
    personas: list[str],
    persona_weights: dict[str, float] | None,
    panel_eval: bool,
    overwrite: bool,
    sleep_seconds: float,
) -> RunResult:
    paper_id = str(paper["paper_id"])
    model_slug = slugify(model)
    pipeline_slug = slugify(pipeline_mode)

    ref_dir = output_root / "references"
    review_dir = output_root / "reviews" / pipeline_slug / model_slug
    quality_dir = output_root / "quality" / pipeline_slug / model_slug
    parsed_dir = output_root / "parsed_reviews" / pipeline_slug / model_slug
    persona_review_dir = output_root / "persona_reviews" / pipeline_slug / model_slug
    persona_parsed_dir = output_root / "persona_parsed_reviews" / pipeline_slug / model_slug

    ref_path = ref_dir / f"{paper_id}_human_reference.md"
    review_path = review_dir / f"{paper_id}_review.md"
    quality_path = quality_dir / f"{paper_id}_quality.md"
    parsed_path = parsed_dir / f"{paper_id}_review.json"

    reference_text = render_human_reference(human_review)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(reference_text, encoding="utf-8")

    fulltext_path = Path(paper["fulltext_path"]) if paper.get("fulltext_path") else None
    if fulltext_path is None or not fulltext_path.exists():
        raise FileNotFoundError(f"Fulltext missing for {paper_id}: {fulltext_path}")

    started = time.time()
    recommendation = ""
    review_payload: dict[str, Any] | None = None
    review_markdown = None
    paper_text = None
    reused_review = False

    if overwrite or not review_path.exists():
        if pipeline_mode == "full":
            review_obj, review_markdown, paper_text = review_paper(
                pdf_path=fulltext_path,
                model=model,
                skip_cost_gate=True,
                config=config,
            )
            recommendation = str(review_obj.overall_feedback.recommendation or "")
            review_payload = {
                "title": review_obj.title,
                "recommendation": recommendation,
                "llm_calls_estimate": None,
            }
        elif pipeline_mode == "slim":
            slim_result = review_paper_slim(
                pdf_path=fulltext_path,
                model=model,
                config=config,
                title_hint=str(paper.get("title", "") or ""),
                personas=personas,
                persona_weights=persona_weights,
            )
            review_markdown = slim_result.markdown
            paper_text = slim_result.paper_text
            recommendation = str(slim_result.review.recommendation or "")
            review_payload = {
                "title": slim_result.title,
                "recommendation": recommendation,
                "llm_calls": slim_result.llm_calls,
                "review_cost_usd": slim_result.cost_usd,
                "call_costs": slim_result.call_costs,
                "rating": slim_result.review.rating,
                "confidence": slim_result.review.confidence,
                "soundness": slim_result.review.soundness,
                "presentation": slim_result.review.presentation,
                "contribution": slim_result.review.contribution,
                "summary": slim_result.review.summary,
                "strength": slim_result.review.strength,
                "weaknesses": slim_result.review.weaknesses,
                "questions": slim_result.review.questions,
                "rationale": slim_result.review.rationale,
                "committee": slim_result.committee,
                "structural_inventory": slim_result.structural_inventory.as_dict(),
            }
            for persona_slug, persona_markdown in slim_result.persona_markdowns.items():
                persona_review_path = persona_review_dir / f"{paper_id}__{persona_slug}_review.md"
                persona_review_path.parent.mkdir(parents=True, exist_ok=True)
                persona_review_path.write_text(persona_markdown, encoding="utf-8")
                persona_payload = {
                    "title": slim_result.title,
                    "persona_slug": persona_slug,
                    "weight": (
                        slim_result.committee.get("personas", [{}])
                        and next(
                            (
                                entry.get("weight")
                                for entry in slim_result.committee.get("personas", [])
                                if entry.get("slug") == persona_slug
                            ),
                            None,
                        )
                    ),
                    "rating": slim_result.persona_reviews[persona_slug].rating,
                    "confidence": slim_result.persona_reviews[persona_slug].confidence,
                    "soundness": slim_result.persona_reviews[persona_slug].soundness,
                    "presentation": slim_result.persona_reviews[persona_slug].presentation,
                    "contribution": slim_result.persona_reviews[persona_slug].contribution,
                    "recommendation": slim_result.persona_reviews[persona_slug].recommendation,
                    "summary": slim_result.persona_reviews[persona_slug].summary,
                    "strength": slim_result.persona_reviews[persona_slug].strength,
                    "weaknesses": slim_result.persona_reviews[persona_slug].weaknesses,
                    "questions": slim_result.persona_reviews[persona_slug].questions,
                    "rationale": slim_result.persona_reviews[persona_slug].rationale,
                }
                write_json(
                    persona_parsed_dir / f"{paper_id}__{persona_slug}_review.json",
                    persona_payload,
                )
        else:
            raise ValueError(f"Unknown pipeline_mode: {pipeline_mode}")

        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(review_markdown, encoding="utf-8")
        if review_payload is not None:
            write_json(parsed_path, review_payload)
    else:
        reused_review = True
        review_markdown = review_path.read_text(encoding="utf-8")
        # Still need paper_text for evaluation; rerun extraction/review is too costly,
        # so read the source fulltext directly for the judge.
        paper_text = type("PaperTextProxy", (), {"full_markdown": fulltext_path.read_text(encoding="utf-8")})()
        if parsed_path.exists():
            review_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
            recommendation = str(review_payload.get("recommendation") or "")
    review_elapsed = None if reused_review else (time.time() - started)

    eval_started = time.time()
    quality_client = LLMClient(model=eval_model, config=config)
    if panel_eval:
        report, _individual = evaluate_review_panel(
            review_markdown,
            reference_text,
            client=quality_client,
            paper_text=paper_text.full_markdown,
            paper_pdf=None,
            model=eval_model,
        )
    else:
        report = evaluate_review(
            review_markdown,
            reference_text,
            client=quality_client,
            paper_text=paper_text.full_markdown,
            paper_pdf=None,
            model=eval_model,
        )
    eval_elapsed = time.time() - eval_started
    eval_cost_usd = quality_client.cost_usd
    review_cost_usd = None if review_payload is None else review_payload.get("review_cost_usd")
    total_cost_usd = None
    if review_cost_usd is not None:
        total_cost_usd = round(float(review_cost_usd) + float(eval_cost_usd), 6)

    quality_path.parent.mkdir(parents=True, exist_ok=True)
    save_quality_report(
        report,
        quality_path,
        str(ref_path),
        model=eval_model,
        mode="panel" if panel_eval else "single",
    )

    dim_scores = quality_dims(report)
    row = {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "decision": paper.get("decision", ""),
        "model": model,
        "eval_model": eval_model,
        "pipeline_mode": pipeline_mode,
        "personas": ",".join(personas) if pipeline_mode == "slim" else None,
        "panel_eval": panel_eval,
        "overall_quality_score": round(report.overall_score, 4),
        "coverage_score": dim_scores.get("coverage"),
        "specificity_score": dim_scores.get("specificity"),
        "depth_score": dim_scores.get("depth"),
        "consistency_score": dim_scores.get("consistency"),
        "review_seconds": round(review_elapsed, 3) if review_elapsed is not None else None,
        "eval_seconds": round(eval_elapsed, 3),
        "reused_review": reused_review,
        "review_path": str(review_path),
        "quality_path": str(quality_path),
        "reference_path": str(ref_path),
        "recommendation": recommendation,
        "llm_calls": None if review_payload is None else review_payload.get("llm_calls"),
        "review_cost_usd": review_cost_usd,
        "eval_cost_usd": round(eval_cost_usd, 6),
        "total_cost_usd": total_cost_usd,
    }

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return RunResult(row=row, review_markdown_path=review_path, quality_markdown_path=quality_path)


def build_model_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    summary_rows = []
    for model, items in sorted(grouped.items(), key=lambda item: item[0]):
        def mean_of(key: str) -> float | None:
            vals = [float(row[key]) for row in items if row.get(key) is not None]
            return round(statistics.mean(vals), 4) if vals else None

        summary_rows.append(
            {
                "model": model,
                "n_papers": len(items),
                "overall_quality_score_mean": mean_of("overall_quality_score"),
                "coverage_score_mean": mean_of("coverage_score"),
                "specificity_score_mean": mean_of("specificity_score"),
                "depth_score_mean": mean_of("depth_score"),
                "consistency_score_mean": mean_of("consistency_score"),
                "review_seconds_mean": mean_of("review_seconds"),
                "eval_seconds_mean": mean_of("eval_seconds"),
                "review_cost_usd_mean": mean_of("review_cost_usd"),
                "eval_cost_usd_mean": mean_of("eval_cost_usd"),
                "total_cost_usd_mean": mean_of("total_cost_usd"),
                "total_cost_usd_sum": round(
                    sum(float(row["total_cost_usd"]) for row in items if row.get("total_cost_usd") is not None),
                    6,
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -(row["overall_quality_score_mean"] or float("-inf")),
            row["model"],
        )
    )
    return summary_rows


def leaderboard_markdown(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Direct Coarse Benchmark",
        "",
        "| Rank | Model | N | Overall | Coverage | Specificity | Depth | Consistency | Review sec | Eval sec | Mean total cost | Total cost |",
        "|------|-------|---|---------|----------|-------------|-------|-------------|------------|----------|-----------------|------------|",
    ]
    for idx, row in enumerate(summary_rows, start=1):
        lines.append(
            "| {rank} | {model} | {n} | {overall} | {coverage} | {specificity} | {depth} | {consistency} | {review_sec} | {eval_sec} | {mean_cost} | {total_cost} |".format(
                rank=idx,
                model=row["model"],
                n=row["n_papers"],
                overall=f"{row['overall_quality_score_mean']:.3f}" if row["overall_quality_score_mean"] is not None else "n/a",
                coverage=f"{row['coverage_score_mean']:.3f}" if row["coverage_score_mean"] is not None else "n/a",
                specificity=f"{row['specificity_score_mean']:.3f}" if row["specificity_score_mean"] is not None else "n/a",
                depth=f"{row['depth_score_mean']:.3f}" if row["depth_score_mean"] is not None else "n/a",
                consistency=f"{row['consistency_score_mean']:.3f}" if row["consistency_score_mean"] is not None else "n/a",
                review_sec=f"{row['review_seconds_mean']:.1f}" if row["review_seconds_mean"] is not None else "n/a",
                eval_sec=f"{row['eval_seconds_mean']:.1f}" if row["eval_seconds_mean"] is not None else "n/a",
                mean_cost=f"${row['total_cost_usd_mean']:.4f}" if row["total_cost_usd_mean"] is not None else "n/a",
                total_cost=f"${row['total_cost_usd_sum']:.4f}",
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a coarse sample benchmark.")
    parser.add_argument("--selected-papers", type=Path, default=DEFAULT_SELECTED_PAPERS)
    parser.add_argument("--human-review-dir", type=Path, default=DEFAULT_HUMAN_REVIEW_DIR)
    parser.add_argument("--fulltext-dir", type=Path, default=DEFAULT_FULLTEXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--pipeline", choices=("slim", "full"), default="slim")
    parser.add_argument(
        "--personas",
        default="default-ensemble",
        help=(
            "Comma-separated persona slugs for the slim pipeline. "
            f"Use 'default-ensemble' for {', '.join(DEFAULT_PERSONA_ENSEMBLE)}."
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional comma-separated persona=weight list for slim committee aggregation.",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--panel-eval", action="store_true")
    parser.add_argument("--max-papers", type=int, default=20)
    parser.add_argument("--paper-id", dest="paper_ids", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def enrich_sample_rows(sample_rows: list[dict[str, Any]], fulltext_dir: Path) -> list[dict[str, Any]]:
    enriched = []
    for row in sample_rows:
        paper_id = str(row["paper_id"])
        fulltext_path = fulltext_dir / f"{paper_id}.txt"
        enriched.append(
            {
                **row,
                "fulltext_available": fulltext_path.exists(),
                "fulltext_path": str(fulltext_path) if fulltext_path.exists() else None,
            }
        )
    return enriched


def main() -> None:
    args = parse_args()
    requested_ids = set(args.paper_ids) if args.paper_ids else None
    models = parse_models(args.models)
    personas = parse_personas(args.personas)
    persona_weights = parse_weights(args.weights, personas)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    together_key = read_api_key(args.key_file.resolve())
    config = build_config(together_key)

    sample_rows = load_sample(
        args.selected_papers.resolve(),
        max_papers=args.max_papers,
        requested_ids=requested_ids,
    )
    sample_rows = enrich_sample_rows(sample_rows, args.fulltext_dir.resolve())

    manifest = {
        "created_at_utc": now_utc(),
        "coarse_version": coarse_version,
        "selected_papers_path": str(args.selected_papers.resolve()),
        "human_review_dir": str(args.human_review_dir.resolve()),
        "fulltext_dir": str(args.fulltext_dir.resolve()),
        "output_dir": str(output_dir),
        "pipeline_mode": args.pipeline,
        "personas": personas if args.pipeline == "slim" else None,
        "persona_weights": persona_weights if args.pipeline == "slim" else None,
        "models": models,
        "eval_model": args.eval_model,
        "panel_eval": args.panel_eval,
        "max_papers": args.max_papers,
        "requested_ids": sorted(requested_ids) if requested_ids else None,
        "sample_size": len(sample_rows),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_jsonl(output_dir / "sample_papers.jsonl", sample_rows)

    paper_rows = []
    failures = []

    for paper in sample_rows:
        paper_id = str(paper["paper_id"])
        human_review = load_human_review(args.human_review_dir.resolve(), paper_id)
        for model in models:
            try:
                result = run_one(
                paper=paper,
                    human_review=human_review,
                    model=model,
                    eval_model=args.eval_model,
                    output_root=output_dir,
                    config=config,
                    pipeline_mode=args.pipeline,
                    personas=personas,
                    persona_weights=persona_weights,
                    panel_eval=args.panel_eval,
                    overwrite=args.overwrite,
                    sleep_seconds=args.sleep_seconds,
                )
                paper_rows.append(result.row)
                print(
                    f"[ok] {paper_id} | {model} | overall={result.row['overall_quality_score']} | total_cost=${(result.row['total_cost_usd'] or 0):.4f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "paper_id": paper_id,
                    "model": model,
                    "error": repr(exc),
                }
                failures.append(failure)
                print(f"[fail] {paper_id} | {model} | {exc}", flush=True)

    model_summary = build_model_summary(paper_rows)
    write_csv(output_dir / "per_paper_quality_scores.csv", paper_rows)
    write_json(output_dir / "per_paper_quality_scores.json", paper_rows)
    write_csv(output_dir / "model_summary.csv", model_summary)
    write_json(output_dir / "model_summary.json", model_summary)
    (output_dir / "leaderboard.md").write_text(leaderboard_markdown(model_summary), encoding="utf-8")
    write_json(output_dir / "failures.json", failures)

    final_summary = {
        **manifest,
        "completed_rows": len(paper_rows),
        "failure_count": len(failures),
        "review_cost_usd_total": round(
            sum(float(row["review_cost_usd"]) for row in paper_rows if row.get("review_cost_usd") is not None),
            6,
        ),
        "eval_cost_usd_total": round(
            sum(float(row["eval_cost_usd"]) for row in paper_rows if row.get("eval_cost_usd") is not None),
            6,
        ),
        "total_cost_usd_total": round(
            sum(float(row["total_cost_usd"]) for row in paper_rows if row.get("total_cost_usd") is not None),
            6,
        ),
    }
    write_json(output_dir / "run_summary.json", final_summary)


if __name__ == "__main__":
    main()
