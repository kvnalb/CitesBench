#!/usr/bin/env python3
"""Run alternative decision heads on cached Gemma decision packets.

This is a head-only experiment: it reuses the cached prompt payload saved in
deepseek_decision.json and does not rerun the Gemma committee stage.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_OUTPUT_ROOTS = [
    ROOT / "OutputNew" / "Empirics" / "gemma_ready7_wave1_cached_v2",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave2_incremental",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave3_single_managed",
]
DEFAULT_KEY_FILE = ROOT / "key.txt"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

PROMPT_VARIANTS: dict[str, str] = {
    "plain": "",
    "positive_bias": """\

Calibration adjustment:
- Do not treat reject as the safe default. The conference accepts many papers with real weaknesses when the contribution is useful, evidence is mostly credible, and the paper clears the bar.
- Give positive weight to concrete strengths, clear empirical support, meaningful novelty, and strong reviewer-score evidence even when the weaknesses section is non-empty.
- For borderline papers, choose accept when the evidence suggests the paper would plausibly be accepted after normal reviewer disagreement rather than requiring near-perfect evidence.
- Keep the decision evidence-based; this is a calibration correction against excessive conservatism, not permission to accept weak papers.
""",
}

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "gpt-oss": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "openai/gpt-oss-20b": ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    "openai/gpt-oss-120b": ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    "deepseek-v3": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-ai/deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "DeepSeek-V3.1"),
    "deepseek-r1": ("deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
    "deepseek-ai/deepseek-r1": ("deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
    "mistral-small-24b": ("mistralai/Mistral-Small-24B-Instruct-2501", "Mistral-Small-24B"),
    "mistral-24b": ("mistralai/Mistral-Small-24B-Instruct-2501", "Mistral-Small-24B"),
    "mistralai/mistral-small-24b-instruct-2501": (
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "Mistral-Small-24B",
    ),
    "qwen3-235b-thinking": ("Qwen/Qwen3-235B-A22B-Thinking-2507", "Qwen3-235B-Thinking"),
    "qwen/qwen3-235b-a22b-thinking-2507": (
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen3-235B-Thinking",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def load_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def resolve_models(raw: str) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for token in (piece.strip() for piece in raw.split(",")):
        if not token:
            continue
        alias = MODEL_ALIASES.get(token.lower())
        if alias is None:
            models.append(ModelSpec(token, token))
        else:
            models.append(ModelSpec(alias[0], alias[1]))
    return models


def apply_prompt_variant(system_message: str, prompt_variant: str) -> str:
    suffix = PROMPT_VARIANTS[prompt_variant]
    if not suffix:
        return system_message
    return system_message.rstrip() + "\n" + suffix.strip() + "\n"


def _strip_model_scaffolding(raw_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return cleaned


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = _strip_model_scaffolding(raw_text)
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_llm_decision(raw_text: str) -> dict[str, Any]:
    payload = _extract_json_object(raw_text)
    if payload is None:
        cleaned = _strip_model_scaffolding(raw_text)
        decision_match = re.search(r'"?decision"?\s*:\s*"?(accept|reject)"?', cleaned, flags=re.IGNORECASE)
        p_match = re.search(r'"?p_accept"?\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
        margin_match = re.search(r'"?margin"?\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
        payload = {
            "decision": decision_match.group(1).lower() if decision_match else None,
            "p_accept": float(p_match.group(1)) if p_match else None,
            "margin": float(margin_match.group(1)) if margin_match else None,
            "top_accept_reasons": [],
            "top_reject_reasons": [],
            "evidence_used": [],
        }

    decision = str(payload.get("decision") or "").strip().lower()
    p_accept = payload.get("p_accept")
    try:
        p_accept = float(p_accept)
    except (TypeError, ValueError):
        p_accept = None
    if p_accept is not None:
        p_accept = max(0.0, min(1.0, p_accept))
    if decision not in {"accept", "reject"}:
        decision = "accept" if p_accept is not None and p_accept >= 0.5 else "reject"
    if p_accept is None:
        p_accept = 1.0 if decision == "accept" else 0.0

    margin = payload.get("margin")
    try:
        margin = float(margin)
    except (TypeError, ValueError):
        margin = (2.0 * p_accept) - 1.0
    margin = max(-1.0, min(1.0, margin))

    return {
        "decision": decision,
        "p_accept": round(p_accept, 6),
        "margin": round(margin, 6),
    }


def together_request(
    *,
    model: ModelSpec,
    api_key: str,
    system_message: str,
    user_message: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model.model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "LLMReview/1.0",
    }

    last_error: str | None = None
    for attempt in range(max_retries):
        started = time.time()
        req = request.Request(TOGETHER_API_URL, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
            elapsed = time.time() - started
            response_payload = json.loads(body)
            choice = response_payload["choices"][0]
            content = choice["message"].get("content") or ""
            return {
                "raw_response": content,
                "usage": response_payload.get("usage", {}),
                "finish_reason": choice.get("finish_reason"),
                "elapsed_seconds": round(elapsed, 3),
                "http_error": None,
            }
        except (error.HTTPError, error.URLError, json.JSONDecodeError, KeyError, socket.timeout, TimeoutError) as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return {
                "raw_response": f"[API error] {last_error}",
                "usage": {},
                "finish_reason": None,
                "elapsed_seconds": round(time.time() - started, 3),
                "http_error": last_error,
            }
    raise RuntimeError("unreachable")


def collect_completed_rows(output_roots: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for root in output_roots:
        if not root.exists():
            continue
        for paper_result_path in root.glob("**/papers/*/paper_result.json"):
            paper_dir = paper_result_path.parent
            deepseek_path = paper_dir / "deepseek_decision.json"
            packet_path = paper_dir / "decision_packet.json"
            if not deepseek_path.exists() or not packet_path.exists():
                continue
            paper_result = read_json(paper_result_path)
            cached_head = read_json(deepseek_path)
            pred = str(paper_result.get("deepseek_decision") or "").strip().lower()
            if pred not in {"accept", "reject"}:
                continue
            paper_id = str(paper_result.get("paper_id") or paper_dir.name)
            row = {
                "paper_id": paper_id,
                "title": paper_result.get("title"),
                "year": int(paper_result.get("year") or 0),
                "true_label": 1 if float(paper_result.get("accepted") or 0.0) >= 0.5 else 0,
                "true_decision": "accept" if float(paper_result.get("accepted") or 0.0) >= 0.5 else "reject",
                "openreview_decision": paper_result.get("decision"),
                "mean_rating": paper_result.get("mean_rating"),
                "committee_rating": paper_result.get("committee_rating"),
                "committee_recommendation": paper_result.get("committee_recommendation"),
                "cached_deepseek_decision": pred,
                "cached_deepseek_p_accept": paper_result.get("deepseek_p_accept"),
                "paper_result_path": str(paper_result_path),
                "deepseek_path": str(deepseek_path),
                "packet_path": str(packet_path),
                "system_message": cached_head.get("system_message"),
                "user_message": cached_head.get("user_message"),
                "_mtime": paper_result_path.stat().st_mtime,
            }
            old = by_id.get(paper_id)
            if old is None or float(row["_mtime"]) > float(old["_mtime"]):
                by_id[paper_id] = row
    rows = list(by_id.values())
    rows.sort(key=lambda row: (int(row["year"]), str(row["paper_id"])))
    return rows


def select_representative_sample(rows: list[dict[str, Any]], max_papers: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= max_papers:
        return rows
    rng = random.Random(seed)
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["year"]), int(row["true_label"])), []).append(row)

    allocations: dict[tuple[int, int], int] = {}
    remainders: list[tuple[float, tuple[int, int]]] = []
    for key, group_rows in groups.items():
        exact = max_papers * (len(group_rows) / len(rows))
        base = int(exact)
        allocations[key] = base
        remainders.append((exact - base, key))
    remaining = max_papers - sum(allocations.values())
    for _, key in sorted(remainders, reverse=True)[:remaining]:
        allocations[key] += 1

    sample: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        take = min(allocations.get(key, 0), len(group_rows))
        sample.extend(rng.sample(group_rows, take))
    sample.sort(key=lambda row: (int(row["year"]), str(row["paper_id"])))
    return sample


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cm = Counter((int(row["true_label"]), 1 if row["decision"] == "accept" else 0) for row in rows)
    tn = cm[(0, 0)]
    fp = cm[(0, 1)]
    fn = cm[(1, 0)]
    tp = cm[(1, 1)]
    n = tn + fp + fn + tp

    def div(num: float, den: float) -> float | None:
        return round(num / den, 6) if den else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    return {
        "n": n,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": div(tp + tn, n),
        "balanced_accuracy": round(((recall or 0.0) + (specificity or 0.0)) / 2.0, 6),
        "precision_accept": precision,
        "recall_accept": recall,
        "specificity_reject": specificity,
        "f1_accept": div(2 * tp, (2 * tp) + fp + fn),
        "actual_accept_rate": div(tp + fn, n),
        "pred_accept_rate": div(tp + fp, n),
    }


def build_cached_deepseek_predictions(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sample_rows:
        rows.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "year": row["year"],
                "true_label": row["true_label"],
                "true_decision": row["true_decision"],
                "decision": row["cached_deepseek_decision"],
                "p_accept": row["cached_deepseek_p_accept"],
                "source": "cached_deepseek_v3_1",
            }
        )
    return rows


def run_one_call(
    *,
    model: ModelSpec,
    row: dict[str, Any],
    output_dir: Path,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    overwrite: bool,
    prompt_variant: str,
) -> dict[str, Any]:
    call_path = output_dir / "calls" / slugify(model.label) / f"{row['paper_id']}.json"
    if call_path.exists() and not overwrite:
        return read_json(call_path)

    system_message = apply_prompt_variant(str(row["system_message"] or ""), prompt_variant)
    api_result = together_request(
        model=model,
        api_key=api_key,
        system_message=system_message,
        user_message=str(row["user_message"] or ""),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    parsed = parse_llm_decision(api_result["raw_response"])
    payload = {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "year": row["year"],
        "true_label": row["true_label"],
        "true_decision": row["true_decision"],
        "model_id": model.model_id,
        "model_label": model.label,
        "prompt_variant": prompt_variant,
        "system_message": system_message,
        "decision": parsed["decision"],
        "p_accept": parsed["p_accept"],
        "margin": parsed["margin"],
        "raw_response": api_result["raw_response"],
        "usage": api_result["usage"],
        "finish_reason": api_result["finish_reason"],
        "elapsed_seconds": api_result["elapsed_seconds"],
        "http_error": api_result["http_error"],
    }
    write_json(call_path, payload)
    return payload


def summary_markdown(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Cached Decision-Head Sweep",
        "",
        "| Model | N | Acc | Bal Acc | Accept Rate | Precision | Recall | F1 | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {n} | {acc:.3f} | {bal:.3f} | {ar:.3f} | {p:.3f} | {r:.3f} | {f1:.3f} | {tn} | {fp} | {fn} | {tp} |".format(
                model=row["model"],
                n=row["n"],
                acc=row["accuracy"] or 0.0,
                bal=row["balanced_accuracy"] or 0.0,
                ar=row["pred_accept_rate"] or 0.0,
                p=row["precision_accept"] or 0.0,
                r=row["recall_accept"] or 0.0,
                f1=row["f1_accept"] or 0.0,
                tn=row["tn"],
                fp=row["fp"],
                fn=row["fn"],
                tp=row["tp"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alternative decision heads on cached Gemma packets.")
    parser.add_argument("--output-root", type=Path, action="append", default=[])
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--sample-jsonl",
        type=Path,
        default=None,
        help="Optional previously saved sample.jsonl to reuse exactly.",
    )
    parser.add_argument(
        "--models",
        default="mistral-small-24b,gpt-oss-120b,gpt-oss-20b,deepseek-r1",
        help="Comma-separated Together model aliases or full model IDs.",
    )
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS), default="plain")
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_roots = [path.resolve() for path in (args.output_root or DEFAULT_OUTPUT_ROOTS)]
    models = resolve_models(args.models)
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ROOT / "OutputNew" / "Empirics" / f"decision_head_sweep_cached100__{stamp}"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_jsonl is not None:
        completed_rows = collect_completed_rows(output_roots)
        sample_rows = read_jsonl(args.sample_jsonl.resolve())
        if args.max_papers is not None:
            sample_rows = sample_rows[: args.max_papers]
    else:
        completed_rows = collect_completed_rows(output_roots)
        sample_rows = select_representative_sample(completed_rows, args.max_papers, args.seed)
    for row in sample_rows:
        row.pop("_mtime", None)
    write_jsonl(output_dir / "sample.jsonl", sample_rows)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at_utc": now_utc(),
            "output_roots": [str(path) for path in output_roots],
            "n_completed_available": len(completed_rows),
            "n_sampled": len(sample_rows),
            "sample_jsonl": str(args.sample_jsonl.resolve()) if args.sample_jsonl else None,
            "seed": args.seed,
            "models": [{"model_id": model.model_id, "label": model.label} for model in models],
            "prompt_variant": args.prompt_variant,
            "prompt_variant_suffix": PROMPT_VARIANTS[args.prompt_variant],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
            "max_workers": args.max_workers,
            "dry_run": args.dry_run,
        },
    )

    prediction_sets: dict[str, list[dict[str, Any]]] = {
        "DeepSeek-V3.1 cached": build_cached_deepseek_predictions(sample_rows)
    }
    write_jsonl(output_dir / "predictions" / "deepseek_v3_1_cached.jsonl", prediction_sets["DeepSeek-V3.1 cached"])

    if not args.dry_run:
        api_key = load_api_key(args.key_file.resolve())
        for model in models:
            print(f"[model] {model.label} ({model.model_id}) on {len(sample_rows)} papers", flush=True)
            model_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
                futures = [
                    executor.submit(
                        run_one_call,
                        model=model,
                        row=row,
                        output_dir=output_dir,
                        api_key=api_key,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_seconds=args.timeout_seconds,
                        max_retries=args.max_retries,
                        overwrite=args.overwrite,
                        prompt_variant=args.prompt_variant,
                    )
                    for row in sample_rows
                ]
                for idx, future in enumerate(as_completed(futures), start=1):
                    payload = future.result()
                    model_rows.append(payload)
                    if idx % 10 == 0 or idx == len(futures):
                        errors = sum(1 for row in model_rows if row.get("http_error"))
                        accepts = sum(1 for row in model_rows if row.get("decision") == "accept")
                        print(
                            f"  [{idx}/{len(futures)}] accepts={accepts} errors={errors}",
                            flush=True,
                        )
            model_rows.sort(key=lambda row: (int(row["year"]), str(row["paper_id"])))
            prediction_sets[model.label] = model_rows
            write_jsonl(output_dir / "predictions" / f"{slugify(model.label)}.jsonl", model_rows)

    summary_rows: list[dict[str, Any]] = []
    for model_label, rows in prediction_sets.items():
        valid_rows = [row for row in rows if not row.get("http_error") and row.get("decision") in {"accept", "reject"}]
        metrics = compute_metrics(valid_rows)
        metrics["model"] = model_label
        metrics["n_errors"] = len(rows) - len(valid_rows)
        summary_rows.append(metrics)
    summary_rows.sort(key=lambda row: (-(row["balanced_accuracy"] or 0.0), -(row["f1_accept"] or 0.0), row["model"]))
    write_json(output_dir / "summary.json", summary_rows)
    (output_dir / "summary.md").write_text(summary_markdown(summary_rows), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
