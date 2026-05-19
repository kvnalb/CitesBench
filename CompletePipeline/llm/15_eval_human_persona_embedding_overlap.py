#!/usr/bin/env python3
"""Pairwise human-reviewer x generated-persona topic overlap.

This is the persona-granular companion to 14_eval_embedding_topic_overlap.py.
For each paper it compares every human reviewer against every generated persona
using local sentence embeddings and a fixed cosine threshold.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
BASE_SCRIPT = ROOT / "Code" / "CompletePipeline" / "llm" / "14_eval_embedding_topic_overlap.py"
DEFAULT_OUTPUT_ROOTS = [
    ROOT / "OutputNew" / "Empirics" / "gemma_ready7_wave1_cached_v2",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave2_incremental",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave3_single_managed",
    ROOT / "OutputNew" / "Coarse",
]
DEFAULT_REVIEW_DB = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("embedding_topic_overlap_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human reviewer x generated persona embedding overlap.")
    parser.add_argument("--sample-jsonl", type=Path, default=None)
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--generated-output-roots", type=Path, nargs="*", default=DEFAULT_OUTPUT_ROOTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-topic-chars", type=int, default=25)
    parser.add_argument("--max-topic-chars", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sentence_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=True)


def cosine_summary(source_embeddings: np.ndarray, target_embeddings: np.ndarray, threshold: float) -> dict[str, Any]:
    if source_embeddings.size == 0:
        return {
            "n_topics": 0,
            "matched": 0,
            "rate": None,
            "mean_best_similarity": None,
            "median_best_similarity": None,
            "max_best_similarity": None,
        }
    if target_embeddings.size == 0:
        scores = np.zeros(source_embeddings.shape[0], dtype=float)
    else:
        scores = (source_embeddings @ target_embeddings.T).max(axis=1)
    matched = int((scores >= threshold).sum())
    return {
        "n_topics": int(source_embeddings.shape[0]),
        "matched": matched,
        "rate": round(matched / int(source_embeddings.shape[0]), 4),
        "mean_best_similarity": round(float(scores.mean()), 4),
        "median_best_similarity": round(float(np.median(scores)), 4),
        "max_best_similarity": round(float(scores.max()), 4),
    }


def best_match_rows(
    source_topics: list[dict[str, Any]],
    target_topics: list[dict[str, Any]],
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    if not source_topics:
        return []
    if not target_topics:
        return [
            {"topic": topic, "best_match": None, "best_similarity": 0.0}
            for topic in source_topics
        ]
    sim = source_embeddings @ target_embeddings.T
    rows = []
    for idx, topic in enumerate(source_topics):
        best_idx = int(np.argmax(sim[idx]))
        rows.append(
            {
                "topic": topic,
                "best_match": target_topics[best_idx],
                "best_similarity": round(float(sim[idx, best_idx]), 6),
            }
        )
    return rows


def balanced_score(h2p_rate: float | None, p2h_rate: float | None) -> float | None:
    if h2p_rate is None or p2h_rate is None:
        return None
    return round((h2p_rate + p2h_rate) / 2.0, 4)


def persona_payloads(persona_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    if not persona_dir.exists():
        return payloads
    for path in sorted(persona_dir.glob("*.json")):
        payload = base.read_json(path)
        slug = str(payload.get("persona_slug") or path.stem)
        payload["_persona_slug"] = slug
        payloads.append((slug, payload))
    return payloads


def run_paper(row: dict[str, Any], args: argparse.Namespace, model: Any, output_dir: Path) -> list[dict[str, Any]]:
    paper_id = str(row.get("paper_id") or "").strip()
    generated_path = base.find_generated_review_path(paper_id, args.generated_output_roots, row)
    if not generated_path:
        return []
    db_title, human_reviews = base.load_human_reviews(args.review_db, paper_id)
    if not human_reviews:
        return []
    personas = persona_payloads(generated_path.parent / "persona_reviews")
    if not personas:
        return []

    title = str(row.get("title") or db_title or paper_id)
    human_by_reviewer: list[tuple[str, list[dict[str, Any]]]] = []
    persona_by_slug: list[tuple[str, list[dict[str, Any]]]] = []

    for index, review in enumerate(human_reviews, start=1):
        reviewer_id = str(review.get("reviewer_id") or f"reviewer_{index}")
        topics = base.extract_human_topics(
            paper_id,
            [review],
            args.min_topic_chars,
            args.max_topic_chars,
        )
        human_by_reviewer.append((reviewer_id, topics))

    for slug, payload in personas:
        topics = base.extract_generated_topics(
            paper_id,
            [payload],
            args.min_topic_chars,
            args.max_topic_chars,
        )
        persona_by_slug.append((slug, topics))

    all_texts: list[str] = []
    for _, topics in human_by_reviewer:
        all_texts.extend(topic["text"] for topic in topics)
    for _, topics in persona_by_slug:
        all_texts.extend(topic["text"] for topic in topics)
    if not all_texts:
        return []

    embeddings = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=float)
    offset = 0
    human_embeddings: dict[str, np.ndarray] = {}
    persona_embeddings: dict[str, np.ndarray] = {}
    for reviewer_id, topics in human_by_reviewer:
        human_embeddings[reviewer_id] = embeddings[offset : offset + len(topics)]
        offset += len(topics)
    for slug, topics in persona_by_slug:
        persona_embeddings[slug] = embeddings[offset : offset + len(topics)]
        offset += len(topics)

    paper_dir = output_dir / "papers" / paper_id
    write_json(
        paper_dir / "topics_by_source.json",
        {
            "paper_id": paper_id,
            "title": title,
            "human": {reviewer_id: topics for reviewer_id, topics in human_by_reviewer},
            "personas": {slug: topics for slug, topics in persona_by_slug},
        },
    )

    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for reviewer_id, human_topics in human_by_reviewer:
        h_emb = human_embeddings[reviewer_id]
        for slug, persona_topics in persona_by_slug:
            p_emb = persona_embeddings[slug]
            h2p = cosine_summary(h_emb, p_emb, args.threshold)
            p2h = cosine_summary(p_emb, h_emb, args.threshold)
            row_out = {
                "paper_id": paper_id,
                "title": title,
                "reviewer_id": reviewer_id,
                "persona": slug,
                "threshold": args.threshold,
                "human_topic_count": len(human_topics),
                "persona_topic_count": len(persona_topics),
                "human_to_persona": h2p,
                "persona_to_human": p2h,
                "balanced_overlap": balanced_score(h2p["rate"], p2h["rate"]),
                "generated_review_path": str(generated_path),
            }
            rows.append(row_out)
            detail_rows.append(
                {
                    **row_out,
                    "human_to_persona_matches": best_match_rows(human_topics, persona_topics, h_emb, p_emb),
                    "persona_to_human_matches": best_match_rows(persona_topics, human_topics, p_emb, h_emb),
                }
            )
    write_jsonl(paper_dir / "human_persona_pairs.jsonl", rows)
    write_json(paper_dir / "human_persona_pair_details.json", {"pairs": detail_rows})
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_scores = [row["balanced_overlap"] for row in rows if row.get("balanced_overlap") is not None]
    by_persona: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_persona[str(row["persona"])].append(row)

    persona_summary = {}
    for persona, persona_rows in sorted(by_persona.items()):
        scores = [float(row["balanced_overlap"]) for row in persona_rows if row.get("balanced_overlap") is not None]
        h_rates = [float(row["human_to_persona"]["rate"]) for row in persona_rows if row["human_to_persona"]["rate"] is not None]
        p_rates = [float(row["persona_to_human"]["rate"]) for row in persona_rows if row["persona_to_human"]["rate"] is not None]
        persona_summary[persona] = {
            "n_pairs": len(persona_rows),
            "mean_balanced_overlap": round(float(np.mean(scores)), 4) if scores else None,
            "mean_human_to_persona": round(float(np.mean(h_rates)), 4) if h_rates else None,
            "mean_persona_to_human": round(float(np.mean(p_rates)), 4) if p_rates else None,
        }

    sorted_rows = sorted(
        [row for row in rows if row.get("balanced_overlap") is not None],
        key=lambda row: (float(row["balanced_overlap"]), row["paper_id"], row["reviewer_id"], row["persona"]),
    )
    return {
        "created_at_utc": now_utc(),
        "n_pairs": len(rows),
        "n_scored_pairs": len(ok_scores),
        "mean_balanced_overlap": round(float(np.mean(ok_scores)), 4) if ok_scores else None,
        "status_counts": dict(Counter("ok" for _ in rows)),
        "persona_summary": persona_summary,
        "lowest_pairs": sorted_rows[:10],
        "highest_pairs": list(reversed(sorted_rows[-10:])),
    }


def write_summary_md(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Human x Persona Embedding Overlap",
        "",
        f"- Created: {summary['created_at_utc']}",
        f"- Threshold: {args.threshold}",
        f"- Pairs scored: {summary['n_scored_pairs']}",
        f"- Mean balanced overlap: {summary['mean_balanced_overlap']}",
        f"- Model: `{args.model}`",
        "",
        "## Persona Means",
        "",
        "| Persona | Pairs | Balanced | H->Persona | Persona->H |",
        "|---|---:|---:|---:|---:|",
    ]
    for persona, row in summary["persona_summary"].items():
        lines.append(
            f"| {persona} | {row['n_pairs']} | {row['mean_balanced_overlap']} | "
            f"{row['mean_human_to_persona']} | {row['mean_persona_to_human']} |"
        )
    for label, key in (("Lowest Pairs", "lowest_pairs"), ("Highest Pairs", "highest_pairs")):
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "| Paper | Reviewer | Persona | Balanced | H->P | P->H | H topics | P topics |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary[key]:
            lines.append(
                f"| {row['paper_id']} | {row['reviewer_id']} | {row['persona']} | "
                f"{row['balanced_overlap']} | {row['human_to_persona']['rate']} | "
                f"{row['persona_to_human']['rate']} | {row['human_topic_count']} | {row['persona_topic_count']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ROOT / "OutputNew" / "Empirics" / f"human_persona_embedding_overlap_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = base.collect_requested_papers(args)
    model = load_sentence_model(args.model)
    all_rows: list[dict[str, Any]] = []
    for row in requested:
        all_rows.extend(run_paper(row, args, model, output_dir))
    summary = aggregate(all_rows)
    write_json(
        output_dir / "run_config.json",
        {
            "created_at_utc": now_utc(),
            "sample_jsonl": str(args.sample_jsonl) if args.sample_jsonl else None,
            "max_papers": args.max_papers,
            "review_db": str(args.review_db),
            "threshold": args.threshold,
            "model": args.model,
            "min_topic_chars": args.min_topic_chars,
            "max_topic_chars": args.max_topic_chars,
        },
    )
    write_jsonl(output_dir / "human_persona_pairs.jsonl", all_rows)
    write_json(output_dir / "summary.json", summary)
    write_summary_md(output_dir / "summary.md", summary, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
