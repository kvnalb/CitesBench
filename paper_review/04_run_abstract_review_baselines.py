#!/usr/bin/env python3
"""
Generation-only runner for abstract- or fulltext-based LLM peer-review outputs.

This script reads the ICLR 2025 Foundation/LLMs set, applies the shared prompt,
calls Together synchronously with progress reporting, and writes raw + parsed
responses to LLMOutput/. Human-review comparison is handled separately by
code/05_compare_abstract_review_outputs.py.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

from _abstract_review_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_ROOT,
    MODEL_ALIASES,
    PRIMARY_AREA,
    append_jsonl,
    model_decision_from_rating,
    now_utc,
    parse_model_response,
    prepare_output_dir,
    read_jsonl,
    resolve_models,
    sample_papers,
    slugify,
    together_request,
    write_json,
    write_jsonl,
)
from _paper_content import get_fulltext_path, infer_fulltext_dir, resolve_paper_content
from _review_prompt_library import (
    DEFAULT_PROMPT_ROOT,
    ReviewPromptBundle,
    build_review_prompt_bundle,
    render_review_user_prompt,
    resolve_personas,
)


CALIBRATION_SUFFIXES = {
    "none": "",
    "neg-mild": (
        "\n\nIMPORTANT CALIBRATION NOTE:\n"
        "When you are uncertain about a paper's quality, lean toward rejection and lower scores.\n"
        "Most submissions to top venues are rejected. A skeptical default is more accurate than a generous one.\n"
        "Only give scores of 6+ when you can clearly identify strong, specific evidence of merit.\n"
    ),
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Together AI review outputs on the ICLR 2025 Foundation/LLMs set."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. If omitted, a timestamped directory is created under LLMOutput/.",
    )
    parser.add_argument(
        "--selected-papers-path",
        type=Path,
        default=None,
        help="Optional selected_papers.jsonl to reuse an existing paper set in the same order.",
    )
    parser.add_argument(
        "--models",
        default="gpt-oss,deepseek-v3",
        help=f"Comma-separated Together model aliases or full model IDs. Known aliases: {', '.join(sorted(MODEL_ALIASES))}",
    )
    parser.add_argument(
        "--personas",
        default="generic",
        help="Comma-separated persona slugs, 'all', or 'default-ensemble'. Persona files live under code/prompts/personas/.",
    )
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=DEFAULT_PROMPT_ROOT,
        help="Root directory containing review/ and personas/ markdown prompt files.",
    )
    parser.add_argument(
        "--content-mode",
        choices=["abstract", "fulltext"],
        default="abstract",
        help="Whether to send abstracts only or extracted full text when available.",
    )
    parser.add_argument(
        "--fulltext-dir",
        type=Path,
        default=None,
        help="Optional directory of <paper_id>.txt full texts. If omitted, a sibling 'fulltext/' next to the input is auto-detected.",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=40000,
        help="Maximum characters of abstract/fulltext sent to the model.",
    )
    parser.add_argument(
        "--fulltext-selection",
        choices=["core-sections", "full"],
        default="core-sections",
        help="When using --content-mode fulltext, send either selected core sections or a raw truncated fulltext window.",
    )
    parser.add_argument(
        "--section-char-limit",
        type=int,
        default=5000,
        help="Maximum characters kept from any single extracted fulltext section when --fulltext-selection=core-sections.",
    )
    parser.add_argument("--max-papers", type=int, default=20, help="Sample size when not reusing a prior selection.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument(
        "--bias-mode",
        choices=sorted(CALIBRATION_SUFFIXES),
        default="none",
        help="Optional prompt calibration suffix to append to the system prompt.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--paper-id", dest="paper_ids", action="append", default=None, help="Restrict to a specific paper_id. Repeatable.")
    parser.add_argument("--include-withdrawn", action="store_true", help="Include withdrawn submissions.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts/manifests/selected papers, but do not call Together.")
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


def normalise_generation_papers(
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
            }
            if requested_ids is not None and paper_id not in requested_ids:
                continue
            if not include_withdrawn and str(merged_paper["decision"]).strip().lower() == "withdrawn":
                continue
            selected.append(merged_paper)
        return selected, f"reused from {selected_papers_path}"

    normalized = normalise_generation_papers(
        input_rows,
        include_withdrawn=include_withdrawn,
        requested_ids=requested_ids,
    )
    selected = sample_papers(normalized, max_papers, seed)
    return selected, "sampled from input"


def build_variant_identity(model_id: str, model_label: str, persona_slug: str, persona_label: str) -> dict[str, str]:
    variant_id = f"{model_id}::{persona_slug}"
    variant_label = f"{model_label} / {persona_label}"
    variant_slug = f"{slugify(model_id)}__{persona_slug}"
    return {
        "id": variant_id,
        "label": variant_label,
        "slug": variant_slug,
    }


def apply_calibration_bias(bundle: ReviewPromptBundle, bias_mode: str) -> ReviewPromptBundle:
    suffix = CALIBRATION_SUFFIXES[bias_mode]
    if not suffix:
        return bundle
    system_prompt = bundle.system_prompt.rstrip() + suffix
    return replace(
        bundle,
        prompt_name=f"{bundle.prompt_name}__{bias_mode}",
        prompt_source=f"{bundle.prompt_source} + calibration:{bias_mode}",
        system_prompt=system_prompt,
        system_prompt_sha256=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
    )


def build_variant_identity_with_bias(
    model_id: str,
    model_label: str,
    persona_slug: str,
    persona_label: str,
    bias_mode: str,
) -> dict[str, str]:
    variant = build_variant_identity(model_id, model_label, persona_slug, persona_label)
    if bias_mode == "none":
        return variant
    return {
        "id": f"{variant['id']}::{bias_mode}",
        "label": f"{variant['label']} / {bias_mode}",
        "slug": f"{variant['slug']}__{slugify(bias_mode)}",
    }


def build_selected_paper_rows(papers: list[dict], fulltext_dir: Path | None) -> list[dict]:
    rows = []
    for paper in papers:
        fulltext_path = get_fulltext_path(fulltext_dir, str(paper["paper_id"]))
        rows.append(
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "primary_area": PRIMARY_AREA,
                "decision": paper["decision"],
                "keywords": paper["keywords"],
                "abstract": paper["abstract"],
                "pdf_url": paper["pdf_url"],
                "abstract_char_count": len(paper["abstract"]),
                "abstract_word_count": len(paper["abstract"].split()),
                "fulltext_available": fulltext_path is not None,
                "fulltext_path": str(fulltext_path) if fulltext_path is not None else None,
            }
        )
    return rows


def summarize_fulltext_availability(papers: list[dict], fulltext_dir: Path | None) -> dict[str, int | str | None]:
    if fulltext_dir is None:
        return {
            "fulltext_dir": None,
            "available": 0,
            "missing": len(papers),
        }
    available = 0
    for paper in papers:
        if get_fulltext_path(fulltext_dir, str(paper["paper_id"])) is not None:
            available += 1
    return {
        "fulltext_dir": str(fulltext_dir),
        "available": available,
        "missing": len(papers) - available,
    }


def main() -> None:
    args = parse_args()
    models = resolve_models(args.models)
    prompt_root = args.prompt_root.resolve()
    personas = resolve_personas(args.personas, prompt_root=prompt_root)

    if args.max_papers is not None and args.max_papers <= 0:
        raise ValueError("--max-papers must be positive.")
    if args.max_content_chars <= 0:
        raise ValueError("--max-content-chars must be positive.")
    if args.section_char_limit <= 0:
        raise ValueError("--section-char-limit must be positive.")
    if not (0.0 <= args.temperature <= 2.0):
        raise ValueError("--temperature must be between 0 and 2.")

    input_rows = read_jsonl(args.input)
    requested_ids = {paper_id.strip() for paper_id in args.paper_ids} if args.paper_ids else None
    selected, selection_source = load_selected_papers(
        input_rows=input_rows,
        selected_papers_path=args.selected_papers_path,
        include_withdrawn=args.include_withdrawn,
        requested_ids=requested_ids,
        max_papers=args.max_papers,
        seed=args.seed,
    )
    if not selected:
        raise ValueError("No papers selected after applying the current filters.")

    fulltext_dir = args.fulltext_dir
    if args.content_mode == "fulltext" and fulltext_dir is None:
        fulltext_dir = infer_fulltext_dir(args.input)
    if fulltext_dir is not None:
        fulltext_dir = fulltext_dir.resolve()
    fulltext_summary = summarize_fulltext_availability(selected, fulltext_dir)

    output_dir = prepare_output_dir(
        args.output_root,
        args.output_dir,
        f"_iclr2025_foundation_llms_{args.content_mode}_generation",
    )
    responses_dir = output_dir / "responses"
    prompt_snapshots_dir = output_dir / "prompts"

    prompt_bundles = {
        persona.slug: apply_calibration_bias(
            build_review_prompt_bundle(args.content_mode, persona.slug, PRIMARY_AREA, prompt_root=prompt_root),
            args.bias_mode,
        )
        for persona in personas
    }

    selected_path = output_dir / "selected_papers.jsonl"
    write_jsonl(selected_path, build_selected_paper_rows(selected, fulltext_dir))

    prompt_variants = []
    for persona in personas:
        bundle = prompt_bundles[persona.slug]
        persona_snapshot_dir = prompt_snapshots_dir / persona.slug
        persona_snapshot_dir.mkdir(parents=True, exist_ok=True)
        (persona_snapshot_dir / "system_prompt.txt").write_text(bundle.system_prompt, encoding="utf-8")
        (persona_snapshot_dir / "user_template.txt").write_text(bundle.user_template, encoding="utf-8")
        (persona_snapshot_dir / "persona.md").write_text(persona.path.read_text(encoding="utf-8"), encoding="utf-8")
        prompt_variants.append(
            {
                "persona_slug": persona.slug,
                "persona_label": persona.label,
                "persona_description": persona.description,
                "persona_path": str(persona.path),
                "bias_mode": args.bias_mode,
                "prompt_name": bundle.prompt_name,
                "prompt_source": bundle.prompt_source,
                "system_template_path": str(bundle.system_template_path),
                "user_template_path": str(bundle.user_template_path),
                "system_prompt_sha256": bundle.system_prompt_sha256,
                "user_prompt_template_sha256": bundle.user_prompt_template_sha256,
            }
        )

    manifest = {
        "run_created_at_utc": now_utc(),
        "run_type": "generation_only",
        "input_path": str(args.input.resolve()),
        "selected_papers_path": str(selected_path),
        "selected_papers_source": selection_source,
        "reused_selected_papers_path": str(args.selected_papers_path.resolve()) if args.selected_papers_path is not None else None,
        "response_dir": str(responses_dir),
        "models": [{"model_id": model.model_id, "label": model.label} for model in models],
        "personas": [
            {
                "slug": persona.slug,
                "label": persona.label,
                "description": persona.description,
                "path": str(persona.path),
            }
            for persona in personas
        ],
        "selection": {
            "max_papers": args.max_papers,
            "seed": args.seed,
            "include_withdrawn": args.include_withdrawn,
            "requested_paper_ids": sorted(requested_ids) if requested_ids else None,
            "selected_count": len(selected),
        },
        "content": {
            "requested_mode": args.content_mode,
            "max_content_chars": args.max_content_chars,
            "fulltext_selection": args.fulltext_selection,
            "section_char_limit": args.section_char_limit,
            "fulltext_dir": str(fulltext_dir) if fulltext_dir is not None else None,
            "fulltext_available": fulltext_summary["available"],
            "fulltext_missing": fulltext_summary["missing"],
        },
        "runtime": {
            "provider": "together",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "bias_mode": args.bias_mode,
            "sleep_seconds": args.sleep_seconds,
            "max_retries": args.max_retries,
            "timeout_seconds": args.timeout_seconds,
            "dry_run": args.dry_run,
        },
        "prompt_library": {
            "root": str(prompt_root),
            "primary_area": PRIMARY_AREA,
            "variants": prompt_variants,
        },
        "prior_model_context": {
            "deepseek_v3_1": "Best prior local result on NMAE and ranking among the tested serverless models.",
            "gpt_oss_20b": "Best prior local result on decision agreement among the tested serverless models.",
            "source": "KunalCode/docs/experiment_8model_baseline.md",
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)

    log(f"Selected papers: {len(selected)} ({selection_source})")
    log(f"Output dir: {output_dir}")
    log(f"Content mode: {args.content_mode}")
    log(f"Personas: {', '.join(persona.slug for persona in personas)}")
    if args.content_mode == "fulltext":
        log(f"Fulltext selection: {args.fulltext_selection} (per-section cap {args.section_char_limit} chars)")
    for persona in personas:
        bundle = prompt_bundles[persona.slug]
        log(f"Prompt[{persona.slug}]: {bundle.prompt_name} ({bundle.system_prompt_sha256[:12]})")
    if args.content_mode == "fulltext":
        log(
            "Fulltext availability: "
            f"{fulltext_summary['available']} available, {fulltext_summary['missing']} fallback-to-abstract"
        )
    log("Prior local model result split:")
    log("  - DeepSeek-V3.1: best NMAE / ranking")
    log("  - GPT-OSS-20B: best decision agreement")

    if args.dry_run:
        log("Dry run only. Prompt, manifest, and selected papers were written; no Together calls made.")
        return

    api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is required unless --dry-run is used.")

    overall_started = time.time()
    for model in models:
        for persona in personas:
            bundle = prompt_bundles[persona.slug]
            variant = build_variant_identity_with_bias(
                model.model_id,
                model.label,
                persona.slug,
                persona.label,
                args.bias_mode,
            )
            output_file = responses_dir / f"{variant['slug']}.jsonl"
            existing_ids = set()
            if output_file.exists():
                for row in read_jsonl(output_file):
                    existing_ids.add(row["paper_id"])

            total = len(selected)
            remaining_to_run = total - len(existing_ids)
            model_started = time.time()
            request_times: list[float] = []

            log(f"\nVariant: {variant['label']}")
            log(f"Response file: {output_file.name}")
            log(f"Existing rows: {len(existing_ids)}")
            log(f"Remaining rows: {remaining_to_run}")

            for paper in selected:
                if paper["paper_id"] in existing_ids:
                    continue

                content_meta = resolve_paper_content(
                    paper=paper,
                    content_mode=args.content_mode,
                    fulltext_dir=fulltext_dir,
                    max_content_chars=args.max_content_chars,
                    fulltext_selection=args.fulltext_selection,
                    section_char_limit=args.section_char_limit,
                )
                user_message = render_review_user_prompt(
                    bundle.user_template,
                    paper,
                    content_meta,
                    PRIMARY_AREA,
                )

                api_result = together_request(
                    model=model,
                    paper=paper,
                    api_key=api_key,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                    system_message=bundle.system_prompt,
                    user_message=user_message,
                )
                parsed = parse_model_response(api_result["raw_response"])

                row = {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "primary_area": PRIMARY_AREA,
                    "decision": paper["decision"],
                    "keywords": paper["keywords"],
                    "abstract": paper["abstract"],
                    "pdf_url": paper["pdf_url"],
                    "abstract_char_count": len(paper["abstract"]),
                    "abstract_word_count": len(paper["abstract"].split()),
                    "content": {
                        "requested_mode": args.content_mode,
                        "used_source": content_meta["used_source"],
                        "path": content_meta["path"],
                        "char_count_total": content_meta["char_count_total"],
                        "char_count_used": content_meta["char_count_used"],
                        "word_count_total": content_meta["word_count_total"],
                        "word_count_used": content_meta["word_count_used"],
                        "content_sha256": content_meta["content_sha256"],
                        "selected_sections": content_meta["selected_sections"],
                        "all_detected_sections": content_meta["all_detected_sections"],
                    },
                    "prompt": {
                        "name": bundle.prompt_name,
                        "source": bundle.prompt_source,
                        "system_template_path": str(bundle.system_template_path),
                        "user_template_path": str(bundle.user_template_path),
                        "persona_slug": persona.slug,
                        "persona_label": persona.label,
                        "persona_description": persona.description,
                        "persona_path": str(persona.path),
                        "system_prompt_sha256": bundle.system_prompt_sha256,
                        "user_prompt_template_sha256": bundle.user_prompt_template_sha256,
                    },
                    "run_variant": {
                        "id": variant["id"],
                        "label": variant["label"],
                        "slug": variant["slug"],
                        "bias_mode": args.bias_mode,
                    },
                    "model": {
                        "id": model.model_id,
                        "label": model.label,
                        "provider": "together",
                    },
                    "request": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "timeout_seconds": args.timeout_seconds,
                        "max_retries": args.max_retries,
                    },
                    "llm_review": {
                        "scores": parsed["scores"],
                        "rationale": parsed["rationale"],
                        "parsed_ok": parsed["parsed_ok"],
                        "parser": parsed["parser"],
                        "cleaned_content": parsed["cleaned_content"],
                        "model_decision_bucket": model_decision_from_rating(parsed["scores"].get("rating")),
                        "raw_response": api_result["raw_response"],
                        "usage": api_result["usage"],
                        "finish_reason": api_result["finish_reason"],
                        "http_error": api_result["http_error"],
                        "elapsed_seconds": api_result["elapsed_seconds"],
                        "received_at_utc": now_utc(),
                    },
                }
                append_jsonl(output_file, row)
                existing_ids.add(paper["paper_id"])

                if api_result["elapsed_seconds"] is not None:
                    request_times.append(api_result["elapsed_seconds"])
                avg_request = statistics.mean(request_times) if request_times else None
                done = len(existing_ids)
                remaining = total - done
                eta_seconds = avg_request * remaining if avg_request is not None else None
                parse_status = "parsed" if parsed["parsed_ok"] else "unparsed"
                model_elapsed = time.time() - model_started
                overall_elapsed = time.time() - overall_started
                req_seconds = api_result["elapsed_seconds"]
                req_display = f"{req_seconds:.1f}s" if req_seconds is not None else "n/a"
                avg_display = f"{avg_request:.1f}s" if avg_request is not None else "n/a"
                finish_reason = api_result["finish_reason"] or "n/a"
                log(
                    f"[{done}/{total} {100 * done / total:5.1f}%] "
                    f"{variant['slug']} {paper['paper_id']} {parse_status} "
                    f"source={content_meta['used_source']} "
                    f"finish={finish_reason} "
                    f"req={req_display} avg={avg_display} "
                    f"eta={format_seconds(eta_seconds)} "
                    f"model_elapsed={format_seconds(model_elapsed)} "
                    f"overall={format_seconds(overall_elapsed)}"
                )

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    log(f"\nResponse files written under: {responses_dir}")


if __name__ == "__main__":
    main()
