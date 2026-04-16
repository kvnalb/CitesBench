#!/usr/bin/env python3
"""
A/B harness for the committee editorial-dedup intervention.

Compares per-paper raw_concatenated_rationale vs editorial_summary using:
  1. A third-party side-by-side judge LLM (different from the editor)
     scoring three dimensions: clarity, redundancy_freedom,
     contradiction_freedom. Optional position swap to debias.
  2. Cheap proxies: token count and Flesch-Kincaid grade.

NMAE / Spearman / decision-agreement are intentionally not computed:
the editorial pass does not change the numeric scores, so those
metrics are flat by construction and uninformative here.

Inputs are two committee-aggregation run dirs produced by
paper_review/08_aggregate_persona_reviews.py:
  --baseline-dir : aggregation run WITHOUT --editorial-dedup
  --editorial-dir: aggregation run WITH    --editorial-dedup
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper_review"))

from _abstract_review_common import (
    MODEL_ALIASES,
    ModelSpec,
    now_utc,
    read_jsonl,
    together_request,
    write_json,
    write_jsonl,
)


JUDGE_SYSTEM_PROMPT = (
    "You are evaluating two consolidated committee statements about the same research paper.\n"
    "You are NOT scoring the paper itself; only the rationale text.\n"
    "Judge each of three dimensions independently:\n"
    "  - clarity: which is easier to read and understand?\n"
    "  - redundancy_freedom: which has fewer repeated or overlapping points?\n"
    "  - contradiction_freedom: which has fewer internal contradictions or inconsistencies?\n"
    "Prefer Tie only when the two are genuinely indistinguishable on the dimension.\n\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    "{\n"
    '  "clarity": "A" | "B" | "Tie",\n'
    '  "redundancy_freedom": "A" | "B" | "Tie",\n'
    '  "contradiction_freedom": "A" | "B" | "Tie",\n'
    '  "reasoning": "one short sentence"\n'
    "}\n"
)


DIMENSIONS = ("clarity", "redundancy_freedom", "contradiction_freedom")


_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD_SPLIT = re.compile(r"[A-Za-z']+")


def count_syllables(word: str) -> int:
    word = word.lower().strip("'")
    if not word:
        return 0
    groups = _VOWEL_GROUPS.findall(word)
    syllables = len(groups)
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def flesch_kincaid_grade(text: str) -> float | None:
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    words = _WORD_SPLIT.findall(text)
    if not sentences or not words:
        return None
    syllables = sum(count_syllables(w) for w in words)
    return round(
        0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59,
        2,
    )


def token_count(text: str) -> int:
    return len(text.split())


def normalise_winner(value: object) -> str:
    norm = str(value or "").strip().lower()
    if norm in {"a", "first", "left"}:
        return "A"
    if norm in {"b", "second", "right"}:
        return "B"
    return "Tie"


def invert_winner(value: str) -> str:
    if value == "A":
        return "B"
    if value == "B":
        return "A"
    return "Tie"


def parse_judge_response(raw_text: str) -> dict[str, str]:
    text = (raw_text or "").strip()
    parsed: dict | None = None
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

    result = {dim: "Tie" for dim in DIMENSIONS}
    result["reasoning"] = ""
    result["parsed_ok"] = False
    if parsed is None:
        return result
    for dim in DIMENSIONS:
        result[dim] = normalise_winner(parsed.get(dim))
    result["reasoning"] = str(parsed.get("reasoning", "")).strip()[:500]
    result["parsed_ok"] = True
    return result


def resolve_model(model_arg: str) -> ModelSpec:
    alias = MODEL_ALIASES.get(model_arg.lower())
    if alias is not None:
        return ModelSpec(alias[0], alias[1])
    return ModelSpec(model_arg, model_arg)


def load_run_index(run_dir: Path) -> dict[tuple[str, str], dict]:
    """Map (model_id, paper_id) -> committee row."""
    response_dir = run_dir / "responses"
    if not response_dir.exists():
        raise ValueError(f"Response directory not found: {response_dir}")
    index: dict[tuple[str, str], dict] = {}
    for path in sorted(response_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            model_id = str(row.get("model", {}).get("id", path.stem))
            paper_id = str(row.get("paper_id", ""))
            if not paper_id:
                continue
            index[(model_id, paper_id)] = row
    return index


def extract_texts(baseline_row: dict, editorial_row: dict) -> tuple[str | None, str | None, dict]:
    """Return (raw_concat, editorial_summary, debug_info). Either may be None if missing."""
    committee_b = (baseline_row.get("committee") or {})
    committee_e = (editorial_row.get("committee") or {})
    llm_b = (baseline_row.get("llm_review") or {})
    llm_e = (editorial_row.get("llm_review") or {})

    raw_concat = (
        committee_b.get("raw_concatenated_rationale")
        or committee_e.get("raw_concatenated_rationale")
        or llm_b.get("rationale")
    )

    editorial_obj = committee_e.get("editorial") or {}
    if editorial_obj.get("parsed_ok") and (editorial_obj.get("summary") or "").strip():
        editorial_summary = llm_e.get("rationale", "").strip() or editorial_obj["summary"]
    else:
        editorial_summary = None

    debug = {
        "baseline_parser": llm_b.get("parser"),
        "editorial_parser": llm_e.get("parser"),
        "editorial_parsed_ok": bool(editorial_obj.get("parsed_ok")),
    }
    return raw_concat, editorial_summary, debug


def build_judge_user_message(text_a: str, text_b: str) -> str:
    return (
        "Statement A:\n"
        f"{text_a.strip()}\n\n"
        "Statement B:\n"
        f"{text_b.strip()}\n\n"
        "Judge per the system instructions and return JSON only."
    )


def judge_pair(
    raw_text: str,
    editorial_text: str,
    judge_model: ModelSpec,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    swap_order: bool,
) -> dict[str, Any]:
    calls = []

    forward_msg = build_judge_user_message(raw_text, editorial_text)
    forward = together_request(
        model=judge_model,
        paper={},
        api_key=api_key,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        system_message=JUDGE_SYSTEM_PROMPT,
        user_message=forward_msg,
    )
    forward_parsed = parse_judge_response(forward.get("raw_response", ""))
    forward_parsed["prompt_order"] = "raw=A,editorial=B"
    forward_parsed["raw_response"] = forward.get("raw_response", "")
    forward_parsed["http_error"] = forward.get("http_error")
    calls.append(forward_parsed)

    if swap_order:
        reverse_msg = build_judge_user_message(editorial_text, raw_text)
        reverse = together_request(
            model=judge_model,
            paper={},
            api_key=api_key,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            system_message=JUDGE_SYSTEM_PROMPT,
            user_message=reverse_msg,
        )
        reverse_parsed = parse_judge_response(reverse.get("raw_response", ""))
        for dim in DIMENSIONS:
            reverse_parsed[dim] = invert_winner(reverse_parsed[dim])
        reverse_parsed["prompt_order"] = "raw=B,editorial=A"
        reverse_parsed["raw_response"] = reverse.get("raw_response", "")
        reverse_parsed["http_error"] = reverse.get("http_error")
        calls.append(reverse_parsed)

    aggregate = aggregate_calls(calls)
    return {
        "calls": [
            {
                "prompt_order": c["prompt_order"],
                "clarity": c["clarity"],
                "redundancy_freedom": c["redundancy_freedom"],
                "contradiction_freedom": c["contradiction_freedom"],
                "reasoning": c["reasoning"],
                "parsed_ok": c["parsed_ok"],
                "http_error": c.get("http_error"),
            }
            for c in calls
        ],
        **aggregate,
    }


def aggregate_calls(calls: list[dict]) -> dict[str, str]:
    """For each dimension, agree across calls -> winner; else Tie."""
    out: dict[str, str] = {}
    for dim in DIMENSIONS:
        votes = [c[dim] for c in calls if c.get("parsed_ok")]
        if not votes:
            out[dim] = "Tie"
            continue
        if all(v == "A" for v in votes):
            out[dim] = "A"
        elif all(v == "B" for v in votes):
            out[dim] = "B"
        else:
            out[dim] = "Tie"
    return out


def summarise(per_paper: list[dict]) -> dict[str, Any]:
    n = len(per_paper)
    winners = {dim: {"raw": 0, "editorial": 0, "tie": 0} for dim in DIMENSIONS}
    raw_tokens: list[int] = []
    ed_tokens: list[int] = []
    raw_grade: list[float] = []
    ed_grade: list[float] = []

    for row in per_paper:
        for dim in DIMENSIONS:
            winner = row["judgment"][dim]
            if winner == "A":
                winners[dim]["raw"] += 1
            elif winner == "B":
                winners[dim]["editorial"] += 1
            else:
                winners[dim]["tie"] += 1
        raw_tokens.append(row["proxies"]["raw"]["token_count"])
        ed_tokens.append(row["proxies"]["editorial"]["token_count"])
        if row["proxies"]["raw"]["fk_grade"] is not None:
            raw_grade.append(row["proxies"]["raw"]["fk_grade"])
        if row["proxies"]["editorial"]["fk_grade"] is not None:
            ed_grade.append(row["proxies"]["editorial"]["fk_grade"])

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": None, "median": None}
        return {
            "n": len(xs),
            "mean": round(statistics.mean(xs), 2),
            "median": round(statistics.median(xs), 2),
        }

    return {
        "n_papers": n,
        "win_rates": {
            dim: {
                "raw_pct": round(100 * winners[dim]["raw"] / n, 1) if n else None,
                "editorial_pct": round(100 * winners[dim]["editorial"] / n, 1) if n else None,
                "tie_pct": round(100 * winners[dim]["tie"] / n, 1) if n else None,
            }
            for dim in DIMENSIONS
        },
        "proxies": {
            "raw_tokens": _stats(raw_tokens),
            "editorial_tokens": _stats(ed_tokens),
            "raw_fk_grade": _stats(raw_grade),
            "editorial_fk_grade": _stats(ed_grade),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Committee aggregation run dir WITHOUT --editorial-dedup.")
    parser.add_argument("--editorial-dir", type=Path, required=True,
                        help="Committee aggregation run dir WITH --editorial-dedup.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write per-paper judgments + summary.")
    parser.add_argument("--judge-model", default="deepseek-r1",
                        help=f"Judge model alias (must differ from the editor). Aliases: {', '.join(sorted(MODEL_ALIASES))}")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--swap-order", action="store_true", default=True,
                        help="Run forward and reverse positions; aggregate only when they agree.")
    parser.add_argument("--no-swap-order", dest="swap_order", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls; write prompts and proxies only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_index = load_run_index(args.baseline_dir.resolve())
    editorial_index = load_run_index(args.editorial_dir.resolve())
    common_keys = sorted(set(baseline_index) & set(editorial_index))
    if args.max_papers is not None:
        common_keys = common_keys[: args.max_papers]
    if not common_keys:
        raise SystemExit("No (model_id, paper_id) keys overlap between baseline and editorial runs.")

    judge_model = resolve_model(args.judge_model)
    api_key = ""
    if not args.dry_run:
        api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("TOGETHER_API_KEY is required unless --dry-run is set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_paper: list[dict] = []
    skipped: list[dict] = []

    for model_id, paper_id in common_keys:
        baseline_row = baseline_index[(model_id, paper_id)]
        editorial_row = editorial_index[(model_id, paper_id)]
        raw_text, editorial_text, debug = extract_texts(baseline_row, editorial_row)
        if not raw_text or not editorial_text:
            skipped.append({"model_id": model_id, "paper_id": paper_id, **debug,
                            "reason": "missing_raw_or_editorial_text"})
            continue

        proxies = {
            "raw": {"token_count": token_count(raw_text), "fk_grade": flesch_kincaid_grade(raw_text)},
            "editorial": {"token_count": token_count(editorial_text), "fk_grade": flesch_kincaid_grade(editorial_text)},
        }

        if args.dry_run:
            judgment = {dim: "Tie" for dim in DIMENSIONS}
            judgment["calls"] = []
        else:
            judgment = judge_pair(
                raw_text=raw_text,
                editorial_text=editorial_text,
                judge_model=judge_model,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                swap_order=args.swap_order,
            )

        per_paper.append({
            "model_id": model_id,
            "paper_id": paper_id,
            "raw_text_excerpt": raw_text[:400],
            "editorial_text_excerpt": editorial_text[:400],
            "proxies": proxies,
            "judgment": judgment,
            **debug,
        })

    summary = summarise(per_paper)
    summary["judge_model_id"] = judge_model.model_id
    summary["judge_model_label"] = judge_model.label
    summary["swap_order"] = args.swap_order
    summary["dry_run"] = args.dry_run
    summary["created_at_utc"] = now_utc()
    summary["baseline_dir"] = str(args.baseline_dir)
    summary["editorial_dir"] = str(args.editorial_dir)
    summary["n_skipped"] = len(skipped)

    write_jsonl(args.output_dir / "per_paper_judgments.jsonl", per_paper)
    if skipped:
        write_jsonl(args.output_dir / "skipped.jsonl", skipped)
    write_json(args.output_dir / "summary.json", summary)

    print(f"A/B harness complete. n_judged={len(per_paper)}, skipped={len(skipped)}.")
    print(f"Summary written to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
