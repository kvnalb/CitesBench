#!/usr/bin/env python3
"""
Aggregate persona-specific review outputs into committee-style outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from _abstract_review_common import (
    DEFAULT_HUMAN_REVIEW_DIR,
    MODEL_ALIASES,
    ModelSpec,
    build_comparison,
    leaderboard_markdown,
    load_human_reviews,
    now_utc,
    read_jsonl,
    slugify,
    summarise_model,
    together_request,
    write_json,
    write_jsonl,
)
from _review_prompt_library import DEFAULT_PERSONA_ENSEMBLE


EDITORIAL_SYSTEM_PROMPT = (
    "You are the senior editor consolidating a committee of persona reviewers into a "
    "single coherent committee statement on one paper.\n"
    "Each persona has produced a short rationale. Your job is to:\n"
    "  1. Group near-duplicate critiques across personas into single themes.\n"
    "  2. Flag contradictions where personas disagree on the same point.\n"
    "  3. Produce a tight consolidated rationale that drops redundancy and contradictions.\n"
    "Do not invent new critiques. Do not soften strong critiques to reach consensus.\n"
    "Prefer the most critical persona's framing when they disagree on severity.\n\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    "{\n"
    '  "summary": "concise consolidated rationale, <= 800 chars",\n'
    '  "themes": [\n'
    '    {"title": "short label", "personas": ["slug", ...], "consolidated": "merged claim"}\n'
    "  ],\n"
    '  "contradictions": [\n'
    '    {"description": "what they disagree on", "personas_a": ["slug"], "personas_b": ["slug"]}\n'
    "  ]\n"
    "}\n"
    "Use [] for themes or contradictions if none apply."
)


def build_editorial_user_message(member_rows: list[dict]) -> str:
    blocks = []
    for row in member_rows:
        prompt_info = row.get("prompt", {}) or {}
        llm_review = row.get("llm_review", {}) or {}
        slug = str(prompt_info.get("persona_slug", "generic"))
        label = str(prompt_info.get("persona_label", slug))
        rationale = str(llm_review.get("rationale", "")).strip()
        if not rationale:
            continue
        blocks.append(f"Persona slug: {slug}\nPersona label: {label}\nRationale:\n{rationale}")
    joined = "\n\n---\n\n".join(blocks) if blocks else "(no persona rationales available)"
    return (
        "Persona rationales for one paper are below. Consolidate them per the system "
        "instructions and return JSON only.\n\n"
        f"{joined}"
    )


def parse_editorial_response(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    parsed: dict[str, Any] | None = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            parsed = candidate
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                candidate = json.loads(text[start : end + 1])
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        return {
            "summary": "",
            "themes": [],
            "contradictions": [],
            "parsed_ok": False,
        }

    summary = str(parsed.get("summary", "")).strip()
    themes_raw = parsed.get("themes") or []
    contradictions_raw = parsed.get("contradictions") or []

    themes = []
    if isinstance(themes_raw, list):
        for item in themes_raw:
            if not isinstance(item, dict):
                continue
            personas = item.get("personas") or []
            if not isinstance(personas, list):
                personas = [str(personas)]
            themes.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "personas": [str(p).strip() for p in personas if str(p).strip()],
                    "consolidated": str(item.get("consolidated", "")).strip(),
                }
            )

    contradictions = []
    if isinstance(contradictions_raw, list):
        for item in contradictions_raw:
            if not isinstance(item, dict):
                continue
            personas_a = item.get("personas_a") or []
            personas_b = item.get("personas_b") or []
            if not isinstance(personas_a, list):
                personas_a = [str(personas_a)]
            if not isinstance(personas_b, list):
                personas_b = [str(personas_b)]
            contradictions.append(
                {
                    "description": str(item.get("description", "")).strip(),
                    "personas_a": [str(p).strip() for p in personas_a if str(p).strip()],
                    "personas_b": [str(p).strip() for p in personas_b if str(p).strip()],
                }
            )

    return {
        "summary": summary,
        "themes": themes,
        "contradictions": contradictions,
        "parsed_ok": bool(summary or themes or contradictions),
    }


def run_editorial_pass(
    member_rows: list[dict],
    model: ModelSpec,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    user_message = build_editorial_user_message(member_rows)
    api_result = together_request(
        model=model,
        paper={},
        api_key=api_key,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        system_message=EDITORIAL_SYSTEM_PROMPT,
        user_message=user_message,
    )
    parsed = parse_editorial_response(api_result.get("raw_response", ""))
    return {
        "model_id": model.model_id,
        "model_label": model.label,
        "summary": parsed["summary"],
        "themes": parsed["themes"],
        "contradictions": parsed["contradictions"],
        "parsed_ok": parsed["parsed_ok"],
        "usage": api_result.get("usage", {}),
        "elapsed_seconds": api_result.get("elapsed_seconds"),
        "http_error": api_result.get("http_error"),
        "raw_response": api_result.get("raw_response"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate persona review outputs into committee-style model outputs."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Persona generation run directory under LLMOutput/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to a sibling <run-dir>__committee_equal.",
    )
    parser.add_argument(
        "--personas",
        default=",".join(DEFAULT_PERSONA_ENSEMBLE),
        help="Comma-separated persona slugs to aggregate.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional comma-separated persona=weight list. Defaults to equal weights.",
    )
    parser.add_argument("--human-review-dir", type=Path, default=DEFAULT_HUMAN_REVIEW_DIR)
    parser.add_argument(
        "--require-human-review",
        action="store_true",
        help="Drop papers without local human review during comparison.",
    )
    parser.add_argument(
        "--editorial-dedup",
        action="store_true",
        help="Run an LLM editorial pass that dedupes themes and flags contradictions across personas.",
    )
    parser.add_argument(
        "--editorial-model",
        default="gpt-oss-120b",
        help=f"Together alias or model ID for the editorial pass. Aliases: {', '.join(sorted(MODEL_ALIASES))}",
    )
    parser.add_argument("--editorial-temperature", type=float, default=0.0)
    parser.add_argument("--editorial-max-tokens", type=int, default=2000)
    parser.add_argument("--editorial-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--editorial-max-retries", type=int, default=3)
    return parser.parse_args()


def resolve_editorial_model(model_arg: str) -> ModelSpec:
    alias = MODEL_ALIASES.get(model_arg.lower())
    if alias is not None:
        return ModelSpec(alias[0], alias[1])
    return ModelSpec(model_arg, model_arg)


def parse_personas(personas_arg: str) -> list[str]:
    personas = [token.strip() for token in personas_arg.split(",") if token.strip()]
    if not personas:
        raise ValueError("At least one persona must be provided.")
    return personas


def parse_weights(weights_arg: str | None, personas: list[str]) -> dict[str, float]:
    if not weights_arg:
        return {persona: 1.0 for persona in personas}
    weights: dict[str, float] = {}
    for chunk in weights_arg.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid weight token '{token}'. Expected persona=weight.")
        persona, raw_value = token.split("=", 1)
        persona = persona.strip()
        value = float(raw_value.strip())
        if value <= 0:
            raise ValueError("Weights must be positive.")
        weights[persona] = value
    missing = [persona for persona in personas if persona not in weights]
    if missing:
        raise ValueError(f"Missing weights for personas: {', '.join(missing)}")
    return weights


def load_response_groups(response_dir: Path) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for response_file in sorted(response_dir.glob("*.jsonl")):
        rows = read_jsonl(response_file)
        if not rows:
            continue
        first = rows[0]
        model_info = first.get("model", {})
        prompt_info = first.get("prompt", {})
        model_id = str(model_info.get("id", response_file.stem))
        persona_slug = str(prompt_info.get("persona_slug", "generic"))
        grouped.setdefault(model_id, {})[persona_slug] = rows
    return grouped


def aggregate_scores(member_rows: list[dict], weights: dict[str, float]) -> dict[str, float | None]:
    score_keys = ("rating", "confidence", "soundness", "presentation", "contribution")
    aggregated: dict[str, float | None] = {}
    for key in score_keys:
        weighted_sum = 0.0
        total_weight = 0.0
        for row in member_rows:
            persona_slug = str(row.get("prompt", {}).get("persona_slug", "generic"))
            score = row.get("llm_review", {}).get("scores", {}).get(key)
            if score is None:
                continue
            weight = weights[persona_slug]
            weighted_sum += weight * float(score)
            total_weight += weight
        aggregated[key] = round(weighted_sum / total_weight, 3) if total_weight > 0 else None
    return aggregated


def aggregate_rationale(member_rows: list[dict]) -> str:
    parts = []
    for row in member_rows:
        persona_label = str(row.get("prompt", {}).get("persona_label", row.get("prompt", {}).get("persona_slug", "persona")))
        rationale = str(row.get("llm_review", {}).get("rationale", "")).strip()
        if rationale:
            parts.append(f"{persona_label}: {rationale}")
    return " | ".join(parts)[:4000]


def aggregate_usage(member_rows: list[dict]) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    for row in member_rows:
        usage = row.get("llm_review", {}).get("usage", {}) or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def build_committee_row(
    model_id: str,
    model_label: str,
    paper_id: str,
    member_rows: list[dict],
    weights: dict[str, float],
    editorial_config: dict | None = None,
) -> dict:
    base_row = dict(member_rows[0])
    scores = aggregate_scores(member_rows, weights)
    total_elapsed = sum(float(row.get("llm_review", {}).get("elapsed_seconds") or 0.0) for row in member_rows)
    usage = aggregate_usage(member_rows)
    member_summaries = []
    for row in member_rows:
        prompt_info = row.get("prompt", {})
        llm_review = row.get("llm_review", {})
        persona_slug = str(prompt_info.get("persona_slug", "generic"))
        member_summaries.append(
            {
                "persona_slug": persona_slug,
                "persona_label": str(prompt_info.get("persona_label", persona_slug)),
                "weight": weights[persona_slug],
                "scores": llm_review.get("scores", {}),
                "rationale": llm_review.get("rationale", ""),
                "parsed_ok": bool(llm_review.get("parsed_ok")),
            }
        )

    variant_slug = f"{slugify(model_id)}__committee_equal"
    raw_concatenated = aggregate_rationale(member_rows)

    editorial_result: dict | None = None
    if editorial_config is not None:
        editorial_result = run_editorial_pass(
            member_rows=member_rows,
            model=editorial_config["model"],
            api_key=editorial_config["api_key"],
            temperature=editorial_config["temperature"],
            max_tokens=editorial_config["max_tokens"],
            timeout_seconds=editorial_config["timeout_seconds"],
            max_retries=editorial_config["max_retries"],
        )
        if editorial_result.get("elapsed_seconds"):
            total_elapsed += float(editorial_result["elapsed_seconds"])
        editorial_usage = editorial_result.get("usage", {}) or {}
        usage["prompt_tokens"] += int(editorial_usage.get("prompt_tokens") or 0)
        usage["completion_tokens"] += int(editorial_usage.get("completion_tokens") or 0)

    final_rationale = raw_concatenated
    parser_label = "committee_weighted_average"
    if editorial_result and editorial_result["parsed_ok"] and editorial_result["summary"]:
        final_rationale = editorial_result["summary"][:4000]
        parser_label = "committee_weighted_average__editorial_dedup"

    base_row["prompt"] = {
        "name": "persona_committee_equal_v1",
        "source": "code/08_aggregate_persona_reviews.py",
        "persona_slug": "committee_equal",
        "persona_label": "Committee Equal Weight",
        "member_personas": [row["persona_slug"] for row in member_summaries],
        "member_weights": weights,
        "editorial_dedup": editorial_result is not None,
    }
    base_row["run_variant"] = {
        "id": f"{model_id}::committee_equal",
        "label": f"{model_label} / Committee Equal Weight",
        "slug": variant_slug,
    }
    base_row["llm_review"] = {
        "scores": scores,
        "rationale": final_rationale,
        "parsed_ok": all(bool(row.get("llm_review", {}).get("parsed_ok")) for row in member_rows),
        "parser": parser_label,
        "cleaned_content": None,
        "model_decision_bucket": "Accept" if (scores.get("rating") or 0.0) >= 6.0 else "Reject" if scores.get("rating") is not None else None,
        "raw_response": None,
        "usage": usage,
        "http_error": editorial_result.get("http_error") if editorial_result else None,
        "elapsed_seconds": round(total_elapsed, 3),
        "received_at_utc": now_utc(),
    }
    base_row["committee"] = {
        "paper_id": paper_id,
        "aggregation": "weighted_average",
        "weights": weights,
        "members": member_summaries,
        "raw_concatenated_rationale": raw_concatenated,
        "editorial": editorial_result,
    }
    return base_row


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
    personas = parse_personas(args.personas)
    weights = parse_weights(args.weights, personas)

    run_dir = args.run_dir.resolve()
    response_dir = run_dir / "responses"
    if not response_dir.exists():
        raise ValueError(f"Response directory not found: {response_dir}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir.parent / f"{run_dir.name}__committee_equal"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    committee_response_dir = output_dir / "responses"
    comparison_dir = output_dir / "comparison"
    per_paper_dir = comparison_dir / "per_paper"
    summaries_dir = comparison_dir / "summaries"

    grouped = load_response_groups(response_dir)
    human_reviews = load_human_reviews(args.human_review_dir)

    editorial_config: dict | None = None
    if args.editorial_dedup:
        api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("TOGETHER_API_KEY is required when --editorial-dedup is set.")
        editorial_model = resolve_editorial_model(args.editorial_model)
        editorial_config = {
            "model": editorial_model,
            "api_key": api_key,
            "temperature": args.editorial_temperature,
            "max_tokens": args.editorial_max_tokens,
            "timeout_seconds": args.editorial_timeout_seconds,
            "max_retries": args.editorial_max_retries,
        }

    manifest = {
        "run_created_at_utc": now_utc(),
        "run_type": "persona_committee_aggregate",
        "source_run_dir": str(run_dir),
        "source_response_dir": str(response_dir),
        "response_dir": str(committee_response_dir),
        "personas": personas,
        "weights": weights,
        "aggregation": "weighted_average",
        "editorial_dedup": {
            "enabled": editorial_config is not None,
            "model_id": editorial_config["model"].model_id if editorial_config else None,
            "model_label": editorial_config["model"].label if editorial_config else None,
            "temperature": args.editorial_temperature if editorial_config else None,
            "max_tokens": args.editorial_max_tokens if editorial_config else None,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)

    leaderboard_rows = []
    for model_id, persona_rows in grouped.items():
        if any(persona not in persona_rows for persona in personas):
            missing = [persona for persona in personas if persona not in persona_rows]
            print(f"Skipping {model_id}: missing personas {', '.join(missing)}")
            continue

        model_label = str(persona_rows[personas[0]][0].get("model", {}).get("label", model_id))
        per_persona_by_paper = {
            persona: {str(row["paper_id"]): row for row in persona_rows[persona]}
            for persona in personas
        }
        common_paper_ids = sorted(set.intersection(*(set(rows) for rows in per_persona_by_paper.values())))
        committee_rows = [
            build_committee_row(
                model_id=model_id,
                model_label=model_label,
                paper_id=paper_id,
                member_rows=[per_persona_by_paper[persona][paper_id] for persona in personas],
                weights=weights,
                editorial_config=editorial_config,
            )
            for paper_id in common_paper_ids
        ]

        variant_slug = f"{slugify(model_id)}__committee_equal"
        response_path = committee_response_dir / f"{variant_slug}.jsonl"
        write_jsonl(response_path, committee_rows)

        compared_rows = build_compared_rows(
            committee_rows,
            human_reviews=human_reviews,
            require_human_review=args.require_human_review,
        )
        compared_path = per_paper_dir / f"{variant_slug}.jsonl"
        write_jsonl(compared_path, compared_rows)
        summary = summarise_model(
            compared_rows,
            ModelSpec(model_id=f"{model_id}::committee_equal", label=f"{model_label} / Committee Equal Weight"),
            compared_path,
        )
        write_json(summaries_dir / f"{variant_slug}.json", summary)
        leaderboard_rows.append(summary)
        print(f"Aggregated {len(committee_rows)} papers for {model_label}")

    leaderboard_rows.sort(
        key=lambda row: (
            row["nmae_mean"] if row["nmae_mean"] is not None else float("inf"),
            -(row["decision_agreement_pct"] if row["decision_agreement_pct"] is not None else -1.0),
        )
    )
    write_json(comparison_dir / "leaderboard.json", leaderboard_rows)
    (comparison_dir / "leaderboard.md").write_text(leaderboard_markdown(leaderboard_rows), encoding="utf-8")
    print(f"Committee outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
