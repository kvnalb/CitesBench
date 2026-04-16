#!/usr/bin/env python3
"""
Run Together pairwise paper comparisons within the root code/ pipeline.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import time
from pathlib import Path

from _abstract_review_common import (
    DEFAULT_HUMAN_REVIEW_DIR,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_ROOT,
    PRIMARY_AREA,
    append_jsonl,
    load_human_reviews,
    now_utc,
    prepare_output_dir,
    read_jsonl,
    resolve_models,
    sample_papers,
    slugify,
    write_json,
    write_jsonl,
)
from _pairwise_prompt_library import (
    DEFAULT_PERSONA_SLUG,
    DEFAULT_PROMPT_ROOT,
    build_pairwise_prompt_bundle,
    resolve_personas,
)
from _pairwise_ranking_common import (
    PairwiseJudgeConfig,
    TogetherPairwiseJudge,
    build_anchor_schedule,
    build_pair_schedule,
    build_ranking,
    build_swiss_round,
    choose_anchor_ids,
    evaluate_results,
    get_human_aggregated_stat,
    fit_bradley_terry,
    get_human_mean_rating,
    human_pair_label,
    make_pair_id,
    render_summary_markdown,
    select_uncertain_judgments,
    summarize_prompt_bundles,
    swiss_match_points,
    total_unique_pairs,
)
from _paper_content import get_fulltext_path, infer_fulltext_dir


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Together AI pairwise ranking on the ICLR 2025 Foundation/LLMs set."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--selected-papers-path",
        type=Path,
        default=None,
        help="Optional selected_papers.jsonl to reuse an existing paper set in the same order.",
    )
    parser.add_argument("--human-review-dir", type=Path, default=DEFAULT_HUMAN_REVIEW_DIR)
    parser.add_argument("--model", default="gpt-oss", help="Together model alias or full model ID.")
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=DEFAULT_PROMPT_ROOT,
        help="Directory containing pairwise prompt and persona markdown files.",
    )
    parser.add_argument(
        "--persona",
        default=DEFAULT_PERSONA_SLUG,
        help="Single persona slug to use for pairwise judging. Defaults to generic.",
    )
    parser.add_argument(
        "--committee-personas",
        default=None,
        help="Optional comma-separated persona slugs. When set, each pair is judged once per persona and aggregated equally.",
    )
    parser.add_argument(
        "--content-mode",
        choices=["abstract", "fulltext"],
        default="abstract",
        help="Whether to compare papers using abstracts only or full text when available.",
    )
    parser.add_argument(
        "--fulltext-dir",
        type=Path,
        default=None,
        help="Optional directory of <paper_id>.txt full texts. If omitted, a sibling fulltext/ is auto-detected.",
    )
    parser.add_argument(
        "--fulltext-selection",
        choices=["core-sections", "full"],
        default="core-sections",
        help="When using fulltext mode, send selected core sections or a raw truncated fulltext window.",
    )
    parser.add_argument("--section-char-limit", type=int, default=2500)
    parser.add_argument(
        "--output-schema",
        choices=["simple", "detailed"],
        default="simple",
        help="Simple asks only for overall winner/confidence/rationale. Detailed also asks for per-category winners.",
    )
    parser.add_argument("--max-content-chars", type=int, default=12000)
    parser.add_argument(
        "--pair-strategy",
        choices=["all", "random", "swiss", "anchor"],
        default="swiss",
        help="How to choose which paper pairs to compare.",
    )
    parser.add_argument(
        "--max-comparisons",
        type=int,
        default=None,
        help="For random or swiss, caps the number of unique pairs to evaluate.",
    )
    parser.add_argument(
        "--swiss-rounds",
        type=int,
        default=None,
        help="Number of Swiss rounds to run. Defaults to 6 when pair-strategy=swiss and no max-comparisons is set.",
    )
    parser.add_argument("--max-papers", type=int, default=None, help="Optional cap on papers from the selected set.")
    parser.add_argument("--anchor-count", type=int, default=4, help="Number of anchors to use for pair-strategy=anchor.")
    parser.add_argument(
        "--anchor-paper-id",
        dest="anchor_paper_ids",
        action="append",
        default=None,
        help="Explicit anchor paper_id. Repeatable. Overrides --anchor-count when provided.",
    )
    parser.add_argument(
        "--max-anchor-comparisons-per-paper",
        type=int,
        default=None,
        help="For pair-strategy=anchor, optionally compare each non-anchor paper to only this many anchors.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument(
        "--prompt-strength",
        choices=["standard", "strong", "anti-hype"],
        default="standard",
        help="Standard is lighter-weight. Strong is stricter and intended for harder comparisons.",
    )
    parser.add_argument("--swap-order", action="store_true", help="Judge both (A,B) and (B,A) for debiasing.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--winner-threshold", type=float, default=0.15)
    parser.add_argument("--tie-delta", type=float, default=0.25, help="Human-score difference treated as a tie for eval.")
    parser.add_argument("--paper-id", dest="paper_ids", action="append", default=None, help="Restrict to a specific paper_id. Repeatable.")
    parser.add_argument("--include-withdrawn", action="store_true")
    parser.add_argument("--require-human-review", action="store_true", help="Drop papers missing local human-review scores.")
    parser.add_argument(
        "--agreement-dimension",
        choices=["rating", "confidence", "soundness", "presentation", "contribution"],
        default="rating",
        help="Human-review dimension used for reviewer-agreement filtering.",
    )
    parser.add_argument(
        "--max-rating-std",
        type=float,
        default=None,
        help="Optional maximum human-review std for the chosen agreement dimension.",
    )
    parser.add_argument(
        "--min-review-count",
        type=int,
        default=None,
        help="Optional minimum reviewer count for the chosen agreement dimension.",
    )
    parser.add_argument(
        "--reask-uncertain",
        action="store_true",
        help="After the first pass, re-judge uncertain pairs with richer evidence and/or a stronger prompt.",
    )
    parser.add_argument(
        "--reask-margin-threshold",
        type=float,
        default=0.1,
        help="Pairs with |final_margin| below this threshold are eligible for re-asking.",
    )
    parser.add_argument(
        "--reask-max-pairs",
        type=int,
        default=None,
        help="Optional cap on how many uncertain pairs to re-ask.",
    )
    parser.add_argument(
        "--reask-content-mode",
        choices=["abstract", "fulltext"],
        default="fulltext",
        help="Evidence mode for the uncertain-pair re-ask stage.",
    )
    parser.add_argument(
        "--reask-fulltext-dir",
        type=Path,
        default=None,
        help="Optional directory of <paper_id>.txt full texts for re-asking uncertain pairs.",
    )
    parser.add_argument(
        "--reask-fulltext-selection",
        choices=["core-sections", "full"],
        default="core-sections",
        help="When re-asking with fulltext, use selected sections or a raw truncated window.",
    )
    parser.add_argument("--reask-section-char-limit", type=int, default=3000)
    parser.add_argument("--reask-max-content-chars", type=int, default=16000)
    parser.add_argument(
        "--reask-prompt-strength",
        choices=["standard", "strong", "anti-hype"],
        default="strong",
        help="Prompt strength to use when re-asking uncertain pairs.",
    )
    parser.add_argument("--reask-max-tokens", type=int, default=None)
    parser.add_argument(
        "--reask-anchor-upsets",
        action="store_true",
        help="Also re-ask high-confidence cases where a non-anchor beats an anchor.",
    )
    parser.add_argument(
        "--reask-upset-margin-threshold",
        type=float,
        default=0.55,
        help="Minimum |final_margin| for an anchor upset to be re-asked.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. If omitted, a timestamped directory is created under LLMOutput/.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the pair schedule and manifest only; do not call Together.")
    return parser.parse_args()


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    total = max(0, int(round(value)))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def load_single_model(model_arg: str):
    models = resolve_models(model_arg)
    if len(models) != 1:
        raise ValueError("--model must resolve to exactly one Together model.")
    return models[0]


def normalise_papers(
    paper_rows: list[dict],
    include_withdrawn: bool,
    requested_ids: set[str] | None,
) -> list[dict]:
    normalized = []
    for row in paper_rows:
        paper_id = str(row["paper_id"])
        if requested_ids is not None and paper_id not in requested_ids:
            continue
        decision = row.get("decision", "")
        if not include_withdrawn and str(decision).strip().lower() == "withdrawn":
            continue
        normalized.append(
            {
                "paper_id": paper_id,
                "title": row.get("title", ""),
                "abstract": row.get("abstract", "") or "",
                "decision": decision,
                "keywords": row.get("keywords", ""),
                "pdf_url": row.get("pdf_url", ""),
                "primary_area": row.get("primary_area", PRIMARY_AREA),
            }
        )
    return normalized


def load_selected_papers(
    input_rows: list[dict],
    selected_papers_path: Path | None,
    include_withdrawn: bool,
    requested_ids: set[str] | None,
    max_papers: int | None,
    seed: int,
) -> tuple[list[dict], str]:
    input_map = {str(row["paper_id"]): row for row in input_rows}
    if selected_papers_path is not None:
        selected_rows = read_jsonl(selected_papers_path)
        selected = []
        for row in selected_rows:
            paper_id = str(row["paper_id"])
            merged = input_map.get(paper_id, row)
            merged_paper = {
                "paper_id": paper_id,
                "title": merged.get("title", row.get("title", "")),
                "abstract": merged.get("abstract", row.get("abstract", "")) or "",
                "decision": merged.get("decision", row.get("decision", "")),
                "keywords": merged.get("keywords", row.get("keywords", "")),
                "pdf_url": merged.get("pdf_url", row.get("pdf_url", "")),
                "primary_area": merged.get("primary_area", row.get("primary_area", PRIMARY_AREA)),
            }
            if requested_ids is not None and paper_id not in requested_ids:
                continue
            if not include_withdrawn and str(merged_paper["decision"]).strip().lower() == "withdrawn":
                continue
            selected.append(merged_paper)
        if max_papers is not None and max_papers < len(selected):
            selected = sample_papers(selected, max_papers, seed)
        return selected, f"reused from {selected_papers_path}"

    selected = sample_papers(
        normalise_papers(input_rows, include_withdrawn=include_withdrawn, requested_ids=requested_ids),
        max_papers,
        seed,
    )
    return selected, "sampled from input"


def attach_human_review_records(
    papers: list[dict],
    human_reviews: dict[str, dict],
    require_human_review: bool,
) -> list[dict]:
    attached = []
    for paper in papers:
        human_review = human_reviews.get(str(paper["paper_id"]))
        if require_human_review and human_review is None:
            continue
        enriched = dict(paper)
        enriched["human_review"] = human_review
        attached.append(enriched)
    return attached


def filter_by_reviewer_agreement(
    papers: list[dict],
    dimension: str,
    max_std: float | None,
    min_review_count: int | None,
) -> list[dict]:
    if max_std is None and min_review_count is None:
        return papers

    filtered = []
    for paper in papers:
        if paper.get("human_review") is None:
            continue
        std_value = get_human_aggregated_stat(paper, dimension=dimension, stat="std")
        count_value = get_human_aggregated_stat(paper, dimension=dimension, stat="count")
        if max_std is not None:
            if not isinstance(std_value, (int, float)) or float(std_value) > max_std:
                continue
        if min_review_count is not None:
            if not isinstance(count_value, int) or count_value < min_review_count:
                continue
        filtered.append(paper)
    return filtered


def build_selected_paper_rows(papers: list[dict], fulltext_dir: Path | None) -> list[dict]:
    rows = []
    for paper in papers:
        fulltext_path = get_fulltext_path(fulltext_dir, str(paper["paper_id"]))
        rows.append(
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "primary_area": paper.get("primary_area", PRIMARY_AREA),
                "decision": paper["decision"],
                "keywords": paper["keywords"],
                "abstract": paper["abstract"],
                "pdf_url": paper["pdf_url"],
                "abstract_char_count": len(paper["abstract"]),
                "abstract_word_count": len(paper["abstract"].split()),
                "fulltext_available": fulltext_path is not None,
                "fulltext_path": str(fulltext_path) if fulltext_path is not None else None,
                "human_review_available": paper.get("human_review") is not None,
                "human_mean_rating": get_human_mean_rating(paper),
                "human_rating_std": get_human_aggregated_stat(paper, dimension="rating", stat="std"),
                "human_review_count": get_human_aggregated_stat(paper, dimension="rating", stat="count"),
            }
        )
    return rows


def summarize_fulltext_availability(papers: list[dict], fulltext_dir: Path | None) -> dict[str, int | str | None]:
    if fulltext_dir is None:
        return {"fulltext_dir": None, "available": 0, "missing": len(papers)}
    available = 0
    for paper in papers:
        if get_fulltext_path(fulltext_dir, str(paper["paper_id"])) is not None:
            available += 1
    return {
        "fulltext_dir": str(fulltext_dir),
        "available": available,
        "missing": len(papers) - available,
    }


def load_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def load_existing_judgments(paths: Path | list[Path]) -> dict[str, dict]:
    path_list = [paths] if isinstance(paths, Path) else list(paths)
    judgments = {}
    for path in path_list:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            judgments[row["pair_id"]] = row
    return judgments


def resolve_swiss_plan(
    num_papers: int,
    swiss_rounds: int | None,
    max_comparisons: int | None,
) -> tuple[int, int, int]:
    pairs_per_round = num_papers // 2
    if pairs_per_round <= 0:
        raise ValueError("Swiss scheduling requires at least two papers.")
    if swiss_rounds is None and max_comparisons is None:
        swiss_rounds = 6
    if swiss_rounds is None:
        swiss_rounds = max(1, math.ceil(max_comparisons / pairs_per_round))
    planned_pairs = swiss_rounds * pairs_per_round
    if max_comparisons is not None:
        planned_pairs = min(planned_pairs, max_comparisons)
    planned_pairs = min(planned_pairs, total_unique_pairs(num_papers))
    return swiss_rounds, planned_pairs, pairs_per_round


def build_budget(
    num_papers: int,
    total_pairs: int,
    scheduled_pairs: list[dict],
    swap_order: bool,
    planned_rounds: int | None = None,
    planned_pairs: int | None = None,
    pairs_per_round: int | None = None,
    reask_pairs: list[dict] | None = None,
) -> dict:
    budget = {
        "num_papers": num_papers,
        "total_unique_pairs": total_pairs,
        "scheduled_pairs": len(scheduled_pairs),
        "api_calls": len(scheduled_pairs) * (2 if swap_order else 1),
    }
    if planned_rounds is not None:
        budget["planned_rounds"] = planned_rounds
        budget["planned_pairs"] = planned_pairs
        budget["planned_api_calls"] = (planned_pairs or 0) * (2 if swap_order else 1)
        budget["pairs_per_round"] = pairs_per_round
        budget["rounds_scheduled"] = len({pair.get("round_index") for pair in scheduled_pairs if pair.get("round_index")})
    if reask_pairs is not None:
        budget["reask_pairs"] = len(reask_pairs)
        budget["reask_api_calls"] = len(reask_pairs) * (2 if swap_order else 1)
    return budget


def build_manifest(
    config_payload: dict,
    selected_path: Path,
    responses_path: Path,
    selection_source: str,
    budget: dict,
    prompt_payload: dict,
    reask_prompt_payload: dict | None = None,
) -> dict:
    return {
        "run_created_at_utc": now_utc(),
        "run_type": "pairwise_ranking",
        "input_path": config_payload["input"],
        "selected_papers_path": str(selected_path),
        "selected_papers_source": selection_source,
        "reused_selected_papers_path": config_payload["selected_papers_path"],
        "human_review_dir": config_payload["human_review_dir"],
        "judgments_path": str(responses_path),
        "model": {"model_id": config_payload["model"], "label": config_payload["model_label"]},
        "selection": config_payload["selection"],
        "content": config_payload["content"],
        "pairing": config_payload["pairing"],
        "runtime": config_payload["runtime"],
        "prompt": prompt_payload,
        "reask_prompt": reask_prompt_payload,
        "config": config_payload,
        "budget": budget,
    }


def write_prompt_snapshots(
    output_dir: Path,
    stage: str,
    prompt_bundles: list,
) -> None:
    stage_dir = output_dir / "prompts" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    for bundle in prompt_bundles:
        slug = bundle.persona.slug
        (stage_dir / f"{slug}.system.txt").write_text(bundle.system_prompt, encoding="utf-8")
        (stage_dir / f"{slug}.user_template.txt").write_text(bundle.user_template, encoding="utf-8")


def resolve_pairwise_personas(
    persona_arg: str,
    committee_personas_arg: str | None,
    prompt_root: Path,
) -> tuple[list, str]:
    if committee_personas_arg:
        personas = resolve_personas(committee_personas_arg, prompt_root=prompt_root)
        return personas, "committee"
    personas = resolve_personas(persona_arg, prompt_root=prompt_root)
    return personas, ("committee" if len(personas) > 1 else "single")


def judge_pairs(
    pairs: list[dict],
    judge: TogetherPairwiseJudge,
    judgments_path: Path,
    judgments_by_id: dict[str, dict],
    papers_by_id: dict[str, dict],
    tie_delta: float,
    target_total_pairs: int,
    progress_label: str | None = None,
    extra_fields: dict | None = None,
) -> None:
    pair_times: list[float] = []
    total = len(pairs)
    started = time.time()
    for index, pair in enumerate(pairs, 1):
        if pair["pair_id"] in judgments_by_id:
            continue

        judgment = judge.judge_pair(pair, papers_by_id)
        judgment["human_label"] = human_pair_label(
            papers_by_id[judgment["paper_a_id"]],
            papers_by_id[judgment["paper_b_id"]],
            tie_delta=tie_delta,
        )
        if extra_fields:
            judgment.update(extra_fields)
        for key in ("round_index", "board_index"):
            if key in pair:
                judgment[key] = pair[key]

        append_jsonl(judgments_path, judgment)
        judgments_by_id[judgment["pair_id"]] = judgment

        if judgment.get("elapsed_seconds") is not None:
            pair_times.append(float(judgment["elapsed_seconds"]))
        avg_pair = statistics.mean(pair_times) if pair_times else None
        completed = len(judgments_by_id)
        remaining = max(0, target_total_pairs - completed)
        eta = avg_pair * remaining if avg_pair is not None else None
        req_display = f"{judgment['elapsed_seconds']:.1f}s" if judgment.get("elapsed_seconds") is not None else "n/a"
        avg_display = f"{avg_pair:.1f}s" if avg_pair is not None else "n/a"
        round_prefix = f"round={pair['round_index']} " if "round_index" in pair else ""
        stage_prefix = f"{progress_label} " if progress_label else ""
        winner_display = judgment.get("final_winner") if judgment.get("final_winner") is not None else "INVALID"
        margin_value = judgment.get("final_margin")
        margin_display = f"{margin_value:.3f}" if isinstance(margin_value, (int, float)) else "n/a"
        invalid_suffix = ""
        if judgment.get("invalid_call_count"):
            invalid_suffix = f" invalid_calls={judgment['invalid_call_count']}"
        log(
            f"[{completed}/{target_total_pairs} {100 * completed / target_total_pairs:5.1f}%] "
            f"{stage_prefix}{round_prefix}{judgment['pair_id']} "
            f"winner={winner_display} margin={margin_display}{invalid_suffix} "
            f"req={req_display} avg={avg_display} eta={format_seconds(eta)} "
            f"batch_elapsed={format_seconds(time.time() - started)}"
        )


def main() -> None:
    args = parse_args()
    model = load_single_model(args.model)
    prompt_root = args.prompt_root.resolve()
    personas, persona_mode = resolve_pairwise_personas(args.persona, args.committee_personas, prompt_root)
    persona_slugs = tuple(persona.slug for persona in personas)
    if args.max_content_chars <= 0:
        raise ValueError("--max-content-chars must be positive.")
    if args.section_char_limit <= 0:
        raise ValueError("--section-char-limit must be positive.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.reask_max_tokens is not None and args.reask_max_tokens <= 0:
        raise ValueError("--reask-max-tokens must be positive.")
    if args.reask_upset_margin_threshold < 0.0 or args.reask_upset_margin_threshold > 1.0:
        raise ValueError("--reask-upset-margin-threshold must be between 0.0 and 1.0.")
    if args.max_papers is not None and args.max_papers <= 1:
        raise ValueError("--max-papers must be at least 2.")
    if args.pair_strategy == "anchor" and args.max_comparisons is not None:
        raise ValueError("pair-strategy=anchor does not use --max-comparisons; use --max-anchor-comparisons-per-paper instead.")

    input_rows = read_jsonl(args.input)
    requested_ids = {paper_id.strip() for paper_id in args.paper_ids} if args.paper_ids else None
    defer_max_papers_until_after_agreement = args.max_rating_std is not None or args.min_review_count is not None
    selected, selection_source = load_selected_papers(
        input_rows=input_rows,
        selected_papers_path=args.selected_papers_path,
        include_withdrawn=args.include_withdrawn,
        requested_ids=requested_ids,
        max_papers=None if defer_max_papers_until_after_agreement else args.max_papers,
        seed=args.seed,
    )
    human_reviews = load_human_reviews(args.human_review_dir)
    papers = attach_human_review_records(selected, human_reviews, require_human_review=args.require_human_review)
    papers = filter_by_reviewer_agreement(
        papers,
        dimension=args.agreement_dimension,
        max_std=args.max_rating_std,
        min_review_count=args.min_review_count,
    )
    if defer_max_papers_until_after_agreement and args.max_papers is not None and len(papers) > args.max_papers:
        papers = sample_papers(papers, args.max_papers, args.seed)
    if len(papers) < 2:
        raise ValueError("Need at least two papers after filtering to run pairwise ranking.")

    fulltext_dir = args.fulltext_dir
    if args.content_mode == "fulltext" and fulltext_dir is None:
        fulltext_dir = infer_fulltext_dir(args.input)
    if fulltext_dir is not None:
        fulltext_dir = fulltext_dir.resolve()
    fulltext_summary = summarize_fulltext_availability(papers, fulltext_dir)
    reask_fulltext_dir = args.reask_fulltext_dir
    if args.reask_uncertain and args.reask_content_mode == "fulltext" and reask_fulltext_dir is None:
        reask_fulltext_dir = infer_fulltext_dir(args.input)
    if reask_fulltext_dir is not None:
        reask_fulltext_dir = reask_fulltext_dir.resolve()
    reask_fulltext_summary = summarize_fulltext_availability(papers, reask_fulltext_dir)

    output_dir = prepare_output_dir(
        args.output_root,
        args.output_dir,
        f"_iclr2025_foundation_llms_pairwise_{slugify(model.model_id)}_{args.content_mode}",
    )
    selected_path = output_dir / "selected_papers.jsonl"
    write_jsonl(selected_path, build_selected_paper_rows(papers, fulltext_dir))
    prompt_bundles = [
        build_pairwise_prompt_bundle(
            content_mode=args.content_mode,
            output_schema=args.output_schema,
            prompt_strength=args.prompt_strength,
            primary_area=PRIMARY_AREA,
            persona_slug=persona.slug,
            prompt_root=prompt_root,
        )
        for persona in personas
    ]
    prompt_summary = summarize_prompt_bundles(prompt_bundles)
    write_prompt_snapshots(output_dir, "base", prompt_bundles)
    reask_prompt_bundles = None
    reask_prompt_summary = None
    if args.reask_uncertain:
        reask_prompt_bundles = [
            build_pairwise_prompt_bundle(
                content_mode=args.reask_content_mode,
                output_schema=args.output_schema,
                prompt_strength=args.reask_prompt_strength,
                primary_area=PRIMARY_AREA,
                persona_slug=persona.slug,
                prompt_root=prompt_root,
            )
            for persona in personas
        ]
        reask_prompt_summary = summarize_prompt_bundles(reask_prompt_bundles)
        write_prompt_snapshots(output_dir, "reask", reask_prompt_bundles)

    pair_schedule_path = output_dir / "pairs.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    reask_pairs_path = output_dir / "pairs_reask.jsonl"
    reask_judgments_path = output_dir / "judgments_reask.jsonl"
    ranking_path = output_dir / "ranking.jsonl"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    scheduled_pairs = load_jsonl_if_exists(pair_schedule_path)
    base_judgments_by_id = load_existing_judgments(judgments_path)
    reask_judgments_by_id = load_existing_judgments(reask_judgments_path)
    judgments_by_id = dict(base_judgments_by_id)
    judgments_by_id.update(reask_judgments_by_id)
    papers_by_id = {paper["paper_id"]: paper for paper in papers}

    total_pairs = total_unique_pairs(len(papers))
    swiss_rounds = None
    planned_pairs = None
    pairs_per_round = None
    anchor_ids = None
    if args.pair_strategy == "swiss":
        swiss_rounds, planned_pairs, pairs_per_round = resolve_swiss_plan(
            len(papers),
            args.swiss_rounds,
            args.max_comparisons,
        )
    elif not scheduled_pairs:
        if args.pair_strategy == "anchor":
            anchor_plan = build_anchor_schedule(
                papers,
                anchor_count=args.anchor_count,
                seed=args.seed,
                anchor_ids=args.anchor_paper_ids,
                max_anchor_comparisons_per_paper=args.max_anchor_comparisons_per_paper,
            )
            scheduled_pairs = anchor_plan["pairs"]
            anchor_ids = anchor_plan["anchor_ids"]
        else:
            scheduled_pairs = build_pair_schedule(
                papers,
                strategy=args.pair_strategy,
                max_comparisons=args.max_comparisons,
                seed=args.seed,
            )
        write_jsonl(pair_schedule_path, scheduled_pairs)
    elif args.pair_strategy == "anchor":
        anchor_ids = choose_anchor_ids(
            papers,
            anchor_count=args.anchor_count,
            seed=args.seed,
            anchor_ids=args.anchor_paper_ids,
        )

    if args.pair_strategy == "anchor" and anchor_ids is None:
        anchor_ids = choose_anchor_ids(
            papers,
            anchor_count=args.anchor_count,
            seed=args.seed,
            anchor_ids=args.anchor_paper_ids,
        )

    config_payload = {
        "input": str(args.input.resolve()),
        "selected_papers_path": str(args.selected_papers_path.resolve()) if args.selected_papers_path is not None else None,
        "human_review_dir": str(args.human_review_dir.resolve()),
        "model": model.model_id,
        "model_label": model.label,
        "selection": {
            "max_papers": args.max_papers,
            "seed": args.seed,
            "include_withdrawn": args.include_withdrawn,
            "require_human_review": args.require_human_review,
            "agreement_dimension": args.agreement_dimension,
            "max_rating_std": args.max_rating_std,
            "min_review_count": args.min_review_count,
            "requested_paper_ids": sorted(requested_ids) if requested_ids else None,
            "selected_count": len(papers),
        },
        "content": {
            "requested_mode": args.content_mode,
            "fulltext_selection": args.fulltext_selection,
            "section_char_limit": args.section_char_limit,
            "output_schema": args.output_schema,
            "max_content_chars": args.max_content_chars,
            "fulltext_dir": str(fulltext_dir) if fulltext_dir is not None else None,
            "fulltext_available": fulltext_summary["available"],
            "fulltext_missing": fulltext_summary["missing"],
        },
        "pairing": {
            "pair_strategy": args.pair_strategy,
            "max_comparisons": args.max_comparisons,
            "swiss_rounds": swiss_rounds,
            "anchor_count": args.anchor_count if args.pair_strategy == "anchor" else None,
            "anchor_ids": anchor_ids,
            "max_anchor_comparisons_per_paper": args.max_anchor_comparisons_per_paper,
            "swap_order": args.swap_order,
            "winner_threshold": args.winner_threshold,
            "tie_delta": args.tie_delta,
        },
        "prompting": {
            "prompt_root": str(prompt_root),
            "persona_mode": persona_mode,
            "personas": [
                {
                    "slug": persona.slug,
                    "label": persona.label,
                    "description": persona.description,
                    "path": str(persona.path),
                }
                for persona in personas
            ],
        },
        "runtime": {
            "provider": "together",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_strength": args.prompt_strength,
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
            "sleep_seconds": args.sleep_seconds,
            "dry_run": args.dry_run,
        },
        "reask": {
            "enabled": args.reask_uncertain,
            "margin_threshold": args.reask_margin_threshold,
            "max_pairs": args.reask_max_pairs,
            "requested_mode": args.reask_content_mode if args.reask_uncertain else None,
            "fulltext_selection": args.reask_fulltext_selection if args.reask_uncertain else None,
            "section_char_limit": args.reask_section_char_limit if args.reask_uncertain else None,
            "max_content_chars": args.reask_max_content_chars if args.reask_uncertain else None,
            "fulltext_dir": str(reask_fulltext_dir) if reask_fulltext_dir is not None else None,
            "fulltext_available": reask_fulltext_summary["available"] if args.reask_uncertain else None,
            "fulltext_missing": reask_fulltext_summary["missing"] if args.reask_uncertain else None,
            "prompt_strength": args.reask_prompt_strength if args.reask_uncertain else None,
            "max_tokens": (args.reask_max_tokens or args.max_tokens) if args.reask_uncertain else None,
            "anchor_upsets": args.reask_anchor_upsets if args.reask_uncertain else False,
            "upset_margin_threshold": args.reask_upset_margin_threshold if args.reask_uncertain else None,
        },
    }
    budget = build_budget(
        len(papers),
        total_pairs,
        scheduled_pairs,
        args.swap_order,
        planned_rounds=swiss_rounds,
        planned_pairs=planned_pairs,
        pairs_per_round=pairs_per_round,
        reask_pairs=load_jsonl_if_exists(reask_pairs_path) if reask_pairs_path.exists() else None,
    )
    manifest = build_manifest(
        config_payload,
        selected_path,
        judgments_path,
        selection_source,
        budget,
        prompt_summary,
        reask_prompt_summary,
    )
    write_json(output_dir / "run_manifest.json", manifest)

    log(f"Papers: {len(papers)}")
    log(f"Output dir: {output_dir}")
    log(f"Model: {model.label} ({model.model_id})")
    log(f"Content mode: {args.content_mode}")
    log(f"Output schema: {args.output_schema}")
    log(f"Prompt strength: {args.prompt_strength}")
    log(f"Persona mode: {persona_mode}")
    log(f"Personas: {', '.join(persona_slugs)}")
    if args.content_mode == "fulltext":
        log(f"Fulltext selection: {args.fulltext_selection} (per-section cap {args.section_char_limit} chars)")
        log(
            "Fulltext availability: "
            f"{fulltext_summary['available']} available, {fulltext_summary['missing']} fallback-to-abstract"
        )
    log(f"Pair strategy: {args.pair_strategy}")
    if args.pair_strategy == "anchor":
        log(f"Anchor IDs: {', '.join(anchor_ids or [])}")
    log(f"Selected papers source: {selection_source}")
    log(f"Human-review coverage: {sum(1 for paper in papers if paper.get('human_review') is not None)}/{len(papers)}")
    if args.max_rating_std is not None or args.min_review_count is not None:
        log(
            "Reviewer-agreement filter: "
            f"dimension={args.agreement_dimension}, "
            f"max_std={args.max_rating_std}, min_review_count={args.min_review_count}"
        )
    if args.pair_strategy == "swiss":
        log(f"Swiss rounds planned: {swiss_rounds}")
        log(f"Target unique pairs: {planned_pairs}")
        log(f"Target Together API calls: {budget['planned_api_calls']}")
    else:
        log(f"Scheduled unique pairs: {len(scheduled_pairs)}")
        log(f"Target Together API calls: {budget['api_calls']}")

    if args.dry_run:
        if args.pair_strategy == "swiss" and not scheduled_pairs:
            first_round = build_swiss_round(
                sorted(papers_by_id),
                list(judgments_by_id.values()),
                round_index=1,
                seed=args.seed,
                max_pairs=min(planned_pairs, pairs_per_round),
            )
            if first_round["pairs"]:
                write_jsonl(pair_schedule_path, first_round["pairs"])
                scheduled_pairs = first_round["pairs"]
                budget = build_budget(
                    len(papers),
                    total_pairs,
                    scheduled_pairs,
                    args.swap_order,
                    planned_rounds=swiss_rounds,
                    planned_pairs=planned_pairs,
                    pairs_per_round=pairs_per_round,
                )
                manifest = build_manifest(
                    config_payload,
                    selected_path,
                    judgments_path,
                    selection_source,
                    budget,
                    prompt_summary,
                    reask_prompt_summary,
                )
                write_json(output_dir / "run_manifest.json", manifest)
                log(f"Swiss dry-run wrote round 1 with {len(first_round['pairs'])} scheduled pairs.")
            log("Dry run only. Later Swiss rounds depend on judged results.")
        else:
            log("Dry run only. Pair schedule and manifest written; no Together calls made.")
        return

    api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is required unless --dry-run is used.")

    judge = TogetherPairwiseJudge(
        PairwiseJudgeConfig(
            model=model,
            api_key=api_key,
            prompt_root=str(prompt_root),
            persona_slugs=persona_slugs,
            content_mode=args.content_mode,
            fulltext_dir=str(fulltext_dir) if fulltext_dir is not None else None,
            fulltext_selection=args.fulltext_selection,
            section_char_limit=args.section_char_limit,
            output_schema=args.output_schema,
            prompt_strength=args.prompt_strength,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_content_chars=args.max_content_chars,
            swap_order=args.swap_order,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            winner_threshold=args.winner_threshold,
            sleep_seconds=args.sleep_seconds,
        )
    )

    if args.pair_strategy == "swiss":
        log(f"Existing judgments: {len(judgments_by_id)}")
        while True:
            pending_pairs = [pair for pair in scheduled_pairs if pair["pair_id"] not in judgments_by_id]
            if pending_pairs:
                active_round = min(pair.get("round_index", 0) for pair in pending_pairs)
                round_pending = [
                    pair
                    for pair in scheduled_pairs
                    if pair.get("round_index", 0) == active_round and pair["pair_id"] not in judgments_by_id
                ]
                log(f"Pending Swiss round {active_round}: {len(round_pending)} pairs")
                judge_pairs(
                    round_pending,
                    judge,
                    judgments_path,
                    judgments_by_id,
                    papers_by_id,
                    args.tie_delta,
                    target_total_pairs=planned_pairs,
                )
                continue

            if len(judgments_by_id) >= planned_pairs:
                break

            current_max_round = max((pair.get("round_index", 0) for pair in scheduled_pairs), default=0)
            next_round_index = current_max_round + 1
            if next_round_index > swiss_rounds:
                break

            round_plan = build_swiss_round(
                sorted(papers_by_id),
                list(judgments_by_id.values()),
                round_index=next_round_index,
                seed=args.seed,
                max_pairs=min(pairs_per_round, planned_pairs - len(judgments_by_id)),
            )
            existing_pair_ids = {pair["pair_id"] for pair in scheduled_pairs}
            round_pairs = [pair for pair in round_plan["pairs"] if pair["pair_id"] not in existing_pair_ids]
            if not round_pairs:
                log("Swiss scheduling exhausted novel pairs before reaching the requested budget.")
                break

            for pair in round_pairs:
                append_jsonl(pair_schedule_path, pair)
            scheduled_pairs.extend(round_pairs)
            budget = build_budget(
                len(papers),
                total_pairs,
                scheduled_pairs,
                args.swap_order,
                planned_rounds=swiss_rounds,
                planned_pairs=planned_pairs,
                pairs_per_round=pairs_per_round,
            )
            manifest = build_manifest(
                config_payload,
                selected_path,
                judgments_path,
                selection_source,
                budget,
                prompt_summary,
                reask_prompt_summary,
            )
            write_json(output_dir / "run_manifest.json", manifest)

            log(f"Scheduled Swiss round {next_round_index}: {len(round_pairs)} pairs")
            judge_pairs(
                round_pairs,
                judge,
                judgments_path,
                judgments_by_id,
                papers_by_id,
                args.tie_delta,
                target_total_pairs=planned_pairs,
            )
    else:
        pending_pairs = [pair for pair in scheduled_pairs if pair["pair_id"] not in judgments_by_id]
        log(f"Existing judgments: {len(judgments_by_id)}")
        log(f"Pending judgments: {len(pending_pairs)}")
        judge_pairs(
            pending_pairs,
            judge,
            judgments_path,
            judgments_by_id,
            papers_by_id,
            args.tie_delta,
            target_total_pairs=len(scheduled_pairs),
        )

    reask_pairs = load_jsonl_if_exists(reask_pairs_path)
    if args.reask_uncertain:
        if not reask_pairs:
            already_reasked = set(reask_judgments_by_id)
            uncertain_judgments = select_uncertain_judgments(
                [judgments_by_id[pair["pair_id"]] for pair in scheduled_pairs if pair["pair_id"] in judgments_by_id and pair["pair_id"] not in already_reasked],
                margin_threshold=args.reask_margin_threshold,
                max_pairs=args.reask_max_pairs,
                anchor_ids=set(anchor_ids or []),
                reask_anchor_upsets=args.reask_anchor_upsets,
                upset_margin_threshold=args.reask_upset_margin_threshold,
            )
            uncertain_pair_ids = {judgment["pair_id"] for judgment in uncertain_judgments}
            reask_pairs = [pair for pair in scheduled_pairs if pair["pair_id"] in uncertain_pair_ids]
            if reask_pairs:
                write_jsonl(reask_pairs_path, reask_pairs)
                log(f"Selected uncertain pairs for re-asking: {len(reask_pairs)}")
        if reask_pairs:
            reask_judge = TogetherPairwiseJudge(
                PairwiseJudgeConfig(
                    model=model,
                    api_key=api_key,
                    prompt_root=str(prompt_root),
                    persona_slugs=persona_slugs,
                    content_mode=args.reask_content_mode,
                    fulltext_dir=str(reask_fulltext_dir) if reask_fulltext_dir is not None else None,
                    fulltext_selection=args.reask_fulltext_selection,
                    section_char_limit=args.reask_section_char_limit,
                    output_schema=args.output_schema,
                    prompt_strength=args.reask_prompt_strength,
                    temperature=args.temperature,
                    max_tokens=args.reask_max_tokens or args.max_tokens,
                    max_content_chars=args.reask_max_content_chars,
                    swap_order=args.swap_order,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                    winner_threshold=args.winner_threshold,
                    sleep_seconds=args.sleep_seconds,
                )
            )
            pending_reasks = [pair for pair in reask_pairs if pair["pair_id"] not in reask_judgments_by_id]
            if pending_reasks:
                log(
                    f"Re-asking uncertain pairs: {len(pending_reasks)} pending "
                    f"(mode={args.reask_content_mode}, prompt={args.reask_prompt_strength})"
                )
                judge_pairs(
                    pending_reasks,
                    reask_judge,
                    reask_judgments_path,
                    reask_judgments_by_id,
                    papers_by_id,
                    args.tie_delta,
                    target_total_pairs=len(reask_pairs),
                    progress_label="reask",
                    extra_fields={
                        "reask_stage": True,
                        "reask_content_mode": args.reask_content_mode,
                        "reask_prompt_strength": args.reask_prompt_strength,
                    },
                )
            judgments_by_id.update(reask_judgments_by_id)

    budget = build_budget(
        len(papers),
        total_pairs,
        scheduled_pairs,
        args.swap_order,
        planned_rounds=swiss_rounds,
        planned_pairs=planned_pairs,
        pairs_per_round=pairs_per_round,
        reask_pairs=reask_pairs if reask_pairs else None,
    )
    manifest = build_manifest(
        config_payload,
        selected_path,
        judgments_path,
        selection_source,
        budget,
        prompt_summary,
        reask_prompt_summary,
    )
    write_json(output_dir / "run_manifest.json", manifest)

    judgments = [judgments_by_id[pair["pair_id"]] for pair in scheduled_pairs if pair["pair_id"] in judgments_by_id]
    scores = fit_bradley_terry(judgments, sorted(papers_by_id))
    ranking = build_ranking(scores, papers_by_id)
    write_jsonl(ranking_path, ranking)

    evaluation = evaluate_results(judgments, ranking, papers_by_id, tie_delta=args.tie_delta)
    write_json(
        summary_json_path,
        {
            "config": config_payload,
            "run_manifest": manifest,
            "evaluation": evaluation,
            "top_ranked": ranking[:20],
        },
    )
    summary_md_path.write_text(render_summary_markdown(config_payload, budget, evaluation, ranking), encoding="utf-8")

    log(f"Judgments written to: {judgments_path}")
    log(f"Ranking written to: {ranking_path}")
    log(f"Summary written to: {summary_md_path}")


if __name__ == "__main__":
    main()
