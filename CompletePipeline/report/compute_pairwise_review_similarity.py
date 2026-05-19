#!/usr/bin/env python3
"""
Compute within-paper pairwise review similarity using sentence-transformer
embeddings and a cosine threshold of 0.5.

For each paper:
  * Extract atomic topics per reviewer (human) or per persona (LLM).
  * Embed all topics once with all-MiniLM-L6-v2 (normalized).
  * For every pair (i, j) of reviewers/personas within the paper, compute
    balanced overlap = 0.5 * (frac of i's topics with best match in j >= 0.5
                              + frac of j's topics with best match in i >= 0.5).

Saves:
  Output/Empirics/review_similarity_within_paper/human_pairs.jsonl
  Output/Empirics/review_similarity_within_paper/llm_pairs.jsonl
  Output/Empirics/review_similarity_within_paper/cross_pairs.jsonl  (human x LLM)
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
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
REVIEW_DB = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
OUTPUT_DIR = ROOT / "OutputNew" / "Empirics" / "review_similarity_within_paper"

EMPIRICS = ROOT / "OutputNew" / "Empirics"
RUNS = [
    "gemma_ready7_wave1_cached_v2",
    "gemma_ready8_wave2_incremental",
    "gemma_ready8_wave3_single_managed",
]

RDD_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD = 0.50
MIN_CHARS = 25
MAX_CHARS = 900


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("embedding_topic_overlap_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def load_sentence_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, local_files_only=True)


def embed(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=float)
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=float,
    )


def balanced_overlap(embA: np.ndarray, embB: np.ndarray, threshold: float) -> dict[str, Any]:
    if embA.shape[0] == 0 or embB.shape[0] == 0:
        return {"n_a": embA.shape[0], "n_b": embB.shape[0],
                "a2b_rate": None, "b2a_rate": None, "balanced": None}
    sim = embA @ embB.T
    best_a2b = sim.max(axis=1)  # for each topic in A, best match in B
    best_b2a = sim.max(axis=0)
    a2b = float((best_a2b >= threshold).mean())
    b2a = float((best_b2a >= threshold).mean())
    return {
        "n_a": int(embA.shape[0]),
        "n_b": int(embB.shape[0]),
        "a2b_rate": round(a2b, 4),
        "b2a_rate": round(b2a, 4),
        "balanced": round(0.5 * (a2b + b2a), 4),
    }


def rdd_paper_ids() -> list[str]:
    import csv as _csv
    ids = []
    with open(RDD_CSV) as f:
        for row in _csv.DictReader(f):
            if int(row["year"]) in (2018, 2019, 2020):
                ids.append(row["paper_id"])
    return ids


def load_all_human_reviews(paper_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Batch-load reviews for all target papers in a single SQL query."""
    conn = sqlite3.connect(str(REVIEW_DB))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join(["?"] * len(paper_ids))
    rows = conn.execute(
        f"""
        SELECT r.paper_id, r.reviewer_id, s.title, s.decision, s.when_submitted,
               r.rating, r.confidence, r.summary, r.strength, r.weaknesses,
               r.questions, r.main_review, r.summary_of_the_review
        FROM REVIEW r JOIN SUBMISSION s ON r.paper_id = s.id
        WHERE r.paper_id IN ({placeholders})
        ORDER BY r.paper_id, r.reviewer_id
        """,
        paper_ids,
    ).fetchall()
    conn.close()
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r["paper_id"]].append(dict(r))
    return out


def find_generated_review_path(pid: str) -> Path | None:
    for run in RUNS:
        run_dir = EMPIRICS / run
        for search_root in [run_dir] + sorted(run_dir.glob("shard_*")):
            persona_dir = search_root / "papers" / pid / "persona_reviews"
            if persona_dir.is_dir():
                return persona_dir
    return None


def load_persona_payloads(persona_dir: Path) -> list[dict[str, Any]]:
    payloads = []
    for f in sorted(persona_dir.glob("*.json")):
        data = json.loads(f.read_text())
        data["_persona_slug"] = f.stem
        payloads.append(data)
    return payloads


def group_by_source(topics: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for t in topics:
        out[str(t["source_id"])].append(t["text"])
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model {MODEL_NAME}...", flush=True)
    model = load_sentence_model(MODEL_NAME)

    paper_ids = rdd_paper_ids()
    print(f"RDD sample papers: {len(paper_ids)}", flush=True)

    print("Batch-loading all human reviews...", flush=True)
    reviews_by_pid = load_all_human_reviews(paper_ids)
    print(f"Loaded reviews for {len(reviews_by_pid)} papers "
          f"({sum(len(v) for v in reviews_by_pid.values()):,} reviews total)",
          flush=True)

    human_rows: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    n_processed = 0
    for pid in paper_ids:
        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  processed {n_processed}/{len(paper_ids)} "
                  f"(human={len(human_rows)}, llm={len(llm_rows)}, cross={len(cross_rows)})",
                  flush=True)

        # ── Human within-paper ──
        reviews = reviews_by_pid.get(pid, [])

        human_topics = base.extract_human_topics(pid, reviews or [], MIN_CHARS, MAX_CHARS)
        human_groups = group_by_source(human_topics)
        human_groups = {k: v for k, v in human_groups.items() if len(v) >= 1}

        human_emb_map: dict[str, np.ndarray] = {}
        if human_groups:
            h_src_ids = sorted(human_groups.keys())
            human_emb_map = {sid: embed(model, human_groups[sid]) for sid in h_src_ids}
            if len(h_src_ids) >= 2:
                for a, b in combinations(h_src_ids, 2):
                    row = balanced_overlap(human_emb_map[a], human_emb_map[b], THRESHOLD)
                    row.update({"paper_id": pid, "src_a": a, "src_b": b})
                    human_rows.append(row)

        # ── LLM personas within-paper ──
        persona_dir = find_generated_review_path(pid)
        llm_emb_map: dict[str, np.ndarray] = {}
        if persona_dir is not None:
            payloads = load_persona_payloads(persona_dir)
            if payloads:
                llm_topics = base.extract_generated_topics(pid, payloads, MIN_CHARS, MAX_CHARS)
                llm_groups = group_by_source(llm_topics)
                llm_groups = {k: v for k, v in llm_groups.items() if len(v) >= 1}
                if llm_groups:
                    l_src_ids = sorted(llm_groups.keys())
                    llm_emb_map = {sid: embed(model, llm_groups[sid]) for sid in l_src_ids}
                    if len(l_src_ids) >= 2:
                        for a, b in combinations(l_src_ids, 2):
                            row = balanced_overlap(llm_emb_map[a], llm_emb_map[b], THRESHOLD)
                            row.update({"paper_id": pid, "src_a": a, "src_b": b})
                            llm_rows.append(row)

        # ── Cross: every human × every LLM persona ──
        if human_emb_map and llm_emb_map:
            for h_sid, h_emb in human_emb_map.items():
                for l_sid, l_emb in llm_emb_map.items():
                    row = balanced_overlap(h_emb, l_emb, THRESHOLD)
                    row.update({"paper_id": pid, "human_src": h_sid, "llm_src": l_sid})
                    cross_rows.append(row)

    # ── write outputs ──
    with open(OUTPUT_DIR / "human_pairs.jsonl", "w") as f:
        for r in human_rows:
            f.write(json.dumps(r) + "\n")
    with open(OUTPUT_DIR / "llm_pairs.jsonl", "w") as f:
        for r in llm_rows:
            f.write(json.dumps(r) + "\n")
    with open(OUTPUT_DIR / "cross_pairs.jsonl", "w") as f:
        for r in cross_rows:
            f.write(json.dumps(r) + "\n")

    n_h = sum(1 for r in human_rows if r.get("balanced") is not None)
    n_l = sum(1 for r in llm_rows if r.get("balanced") is not None)
    n_c = sum(1 for r in cross_rows if r.get("balanced") is not None)
    print(f"\nHuman pairs: {len(human_rows)} total, {n_h} scored")
    print(f"LLM pairs:   {len(llm_rows)} total, {n_l} scored")
    print(f"Cross pairs: {len(cross_rows)} total, {n_c} scored")
    print(f"Wrote to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
