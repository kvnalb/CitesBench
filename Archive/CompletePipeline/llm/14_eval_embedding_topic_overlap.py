#!/usr/bin/env python3
"""Deterministic embedding-based topic overlap for human vs generated reviews.

The script uses rule-based topic extraction and local sentence embeddings:
- human reviews come from gen_review.db
- generated reviews come from coarse_review.json plus optional persona reviews
- each extracted topic is embedded once
- overlap is max cosine similarity in both directions
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
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
DEFAULT_REVIEW_DB = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_ROOTS = [
    ROOT / "OutputNew" / "Empirics" / "gemma_ready7_wave1_cached_v2",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave2_incremental",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave3_single_managed",
    ROOT / "OutputNew" / "Coarse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding-based review-topic overlap.")
    parser.add_argument("--sample-jsonl", type=Path, default=None)
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument(
        "--generated-output-roots",
        type=Path,
        nargs="*",
        default=DEFAULT_OUTPUT_ROOTS,
    )
    parser.add_argument(
        "--generated-text-source",
        choices=["committee", "committee_and_personas"],
        default="committee_and_personas",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence_transformers", "tfidf"],
        default="sentence_transformers",
        help="Use sentence-transformers if available; tfidf is a deterministic fallback.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.45,0.55,0.65,0.75",
        help="Comma-separated cosine thresholds for reporting binary overlap.",
    )
    parser.add_argument("--min-topic-chars", type=int, default=25)
    parser.add_argument("--max-topic-chars", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(piece.strip()) for piece in raw.split(",") if piece.strip()]
    return sorted(set(thresholds))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "\n".join(str(item) for item in value if item is not None)
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collect_requested_papers(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.sample_jsonl:
        rows.extend(read_jsonl(args.sample_jsonl))
    for paper_id in args.paper_id:
        rows.append({"paper_id": paper_id})
    if not rows:
        rows = collect_generated_review_rows(args.generated_output_roots)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id and paper_id not in by_id:
            by_id[paper_id] = row
    return list(by_id.values())[: max(args.max_papers, 0)]


def collect_generated_review_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/coarse_review.json")):
            try:
                review = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            paper_id = str(review.get("paper_id") or path.parent.name)
            if paper_id in seen:
                continue
            seen.add(paper_id)
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": review.get("title"),
                    "year": review.get("year"),
                    "coarse_review_path": str(path),
                }
            )
    return rows


def find_generated_review_path(paper_id: str, roots: list[Path], row: dict[str, Any]) -> Path | None:
    for key in ("coarse_review_path", "generated_review_path"):
        raw = row.get(key)
        if raw:
            path = Path(str(raw))
            if path.exists():
                return path
    for root in roots:
        if not root.exists():
            continue
        candidates = list(root.glob(f"**/papers/{paper_id}/coarse_review.json"))
        candidates.extend(root.glob(f"**/{paper_id}/coarse_review.json"))
        candidates = [path for path in candidates if path.exists()]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return candidates[0]
    return None


def load_human_reviews(db_path: Path, paper_id: str) -> tuple[str | None, list[dict[str, Any]]]:
    if not db_path.exists():
        return None, []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                r.paper_id,
                r.reviewer_id,
                s.title,
                s.decision,
                s.when_submitted,
                r.rating,
                r.confidence,
                r.summary,
                r.strength,
                r.weaknesses,
                r.questions,
                r.main_review,
                r.summary_of_the_review
            FROM REVIEW r
            JOIN SUBMISSION s ON r.paper_id = s.id
            WHERE r.paper_id = ?
            ORDER BY r.reviewer_id
            """,
            (paper_id,),
        ).fetchall()
    finally:
        conn.close()
    reviews = [dict(row) for row in rows]
    title = str(reviews[0]["title"]) if reviews else None
    return title, reviews


def review_payload_to_sections(payload: dict[str, Any], fallback_label: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for label, keys in (
        ("summary", ("summary",)),
        ("strength", ("strength", "strengths")),
        ("weaknesses", ("weaknesses", "weakness")),
        ("questions", ("questions",)),
        ("rationale", ("rationale",)),
        ("review", ("main_review", "summary_of_the_review", "review_text")),
    ):
        text = ""
        for key in keys:
            text = clean_text(payload.get(key))
            if text:
                break
        if text:
            sections.append((label, text))
    if not sections:
        text = clean_text(payload)
        if text:
            sections.append((fallback_label, text))
    return sections


def load_generated_review(path: Path, source: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    committee = read_json(path)
    payloads = [committee]
    if source == "committee_and_personas":
        persona_dir = path.parent / "persona_reviews"
        if persona_dir.exists():
            for persona_path in sorted(persona_dir.glob("*.json")):
                payload = read_json(persona_path)
                payload["_persona_slug"] = payload.get("persona_slug") or persona_path.stem
                payloads.append(payload)
    return str(committee.get("title") or path.parent.name), committee, payloads


def split_numbered_or_bulleted(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    # Convert common inline numbering into line starts before splitting.
    text = re.sub(r"(?<!\w)(\d{1,2})[\.\)]\s+", r"\n\1. ", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "\n- ", text)
    chunks = re.split(r"(?m)(?:^\s*\d{1,2}[\.\)]\s+|^\s*[-*•]\s+)", text)
    chunks = [clean_text(chunk) for chunk in chunks if clean_text(chunk)]
    if len(chunks) > 1:
        return chunks

    paragraphs = [clean_text(chunk) for chunk in re.split(r"\n\s*\n+", text) if clean_text(chunk)]
    if len(paragraphs) > 1:
        return paragraphs

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    if len(sentences) > 1 and len(text) > 450:
        return [clean_text(sentence) for sentence in sentences if clean_text(sentence)]
    return [text]


def split_legacy_review(text: str) -> list[tuple[str, str]]:
    text = clean_text(text)
    if not text:
        return []
    markers = [
        "Summary",
        "Review",
        "Pros",
        "Cons",
        "Strengths",
        "Weaknesses",
        "Questions",
        "After Author",
        "After author's response",
    ]
    pattern = r"(?im)^\s*(%+\s*)?(" + "|".join(re.escape(marker) for marker in markers) + r")\s*:?\s*$"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return [("review", text)]
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        sections.append(("review", text[: matches[0].start()].strip()))
    for idx, match in enumerate(matches):
        label = re.sub(r"\W+", "_", match.group(2).lower()).strip("_") or "review"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((label, body))
    return sections


def make_topic(
    *,
    side: str,
    paper_id: str,
    source_id: str,
    section: str,
    text: str,
    idx: int,
) -> dict[str, Any] | None:
    text = clean_text(text)
    text = re.sub(r"^\(?[a-zA-Z0-9]{1,3}\)?[\.\)]\s+", "", text).strip()
    text = re.sub(r"^(Pros?|Cons?|Strengths?|Weaknesses?|Questions?)\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) < 25:
        return None
    if len(text) > 900:
        text = text[:900].rsplit(" ", 1)[0].strip()
    return {
        "topic_id": f"{side[0]}{idx}",
        "side": side,
        "paper_id": paper_id,
        "source_id": source_id,
        "section": section,
        "text": text,
    }


def extract_human_topics(paper_id: str, reviews: list[dict[str, Any]], min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    idx = 1
    for review_i, review in enumerate(reviews, start=1):
        reviewer_id = str(review.get("reviewer_id") or f"reviewer_{review_i}")
        structured_present = any(clean_text(review.get(key)) for key in ("summary", "strength", "weaknesses", "questions"))
        if structured_present:
            sections = review_payload_to_sections(review, "review")
        else:
            sections = split_legacy_review(clean_text(review.get("main_review")) or clean_text(review.get("summary_of_the_review")))
        for section, body in sections:
            if section == "summary":
                continue
            for chunk in split_numbered_or_bulleted(body):
                topic = make_topic(
                    side="human",
                    paper_id=paper_id,
                    source_id=reviewer_id,
                    section=section,
                    text=chunk,
                    idx=idx,
                )
                if topic and min_chars <= len(topic["text"]) <= max_chars:
                    topics.append(topic)
                    idx += 1
    return dedupe_topics(topics)


def extract_generated_topics(
    paper_id: str,
    payloads: list[dict[str, Any]],
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    idx = 1
    for payload_i, payload in enumerate(payloads):
        source_id = str(payload.get("_persona_slug") or ("committee" if payload_i == 0 else f"generated_{payload_i}"))
        for section, body in review_payload_to_sections(payload, "review"):
            if section == "summary":
                continue
            for chunk in split_numbered_or_bulleted(body):
                topic = make_topic(
                    side="generated",
                    paper_id=paper_id,
                    source_id=source_id,
                    section=section,
                    text=chunk,
                    idx=idx,
                )
                if topic and min_chars <= len(topic["text"]) <= max_chars:
                    topics.append(topic)
                    idx += 1
    return dedupe_topics(topics)


def dedupe_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for topic in topics:
        key = re.sub(r"\W+", " ", topic["text"].lower()).strip()
        key = key[:220]
        if key in seen:
            continue
        seen.add(key)
        topic = dict(topic)
        topic["topic_id"] = f"{topic['side'][0]}{len(deduped) + 1}"
        deduped.append(topic)
    return deduped


class Embedder:
    def __init__(self, backend: str, model_name: str) -> None:
        self.backend = backend
        self.model_name = model_name
        self.model = None

    def encode_pair(self, human_texts: list[str], generated_texts: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
        if self.backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name, local_files_only=True)
                human = self.model.encode(human_texts, normalize_embeddings=True, show_progress_bar=False)
                generated = self.model.encode(generated_texts, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(human, dtype=float), np.asarray(generated, dtype=float), self.model_name
            except Exception as exc:
                print(f"sentence-transformers unavailable for `{self.model_name}`; falling back to TF-IDF: {exc}")
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        matrix = vectorizer.fit_transform(human_texts + generated_texts).astype(float)
        human = matrix[: len(human_texts)].toarray()
        generated = matrix[len(human_texts) :].toarray()
        human = normalize_rows(human)
        generated = normalize_rows(generated)
        return human, generated, "tfidf_unigram_bigram"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_matrix(human_embeddings: np.ndarray, generated_embeddings: np.ndarray) -> np.ndarray:
    if human_embeddings.size == 0 or generated_embeddings.size == 0:
        return np.zeros((human_embeddings.shape[0], generated_embeddings.shape[0]))
    return human_embeddings @ generated_embeddings.T


def best_matches(
    source_topics: list[dict[str, Any]],
    target_topics: list[dict[str, Any]],
    sim: np.ndarray,
    direction: str,
) -> list[dict[str, Any]]:
    if not source_topics:
        return []
    rows: list[dict[str, Any]] = []
    for i, topic in enumerate(source_topics):
        if not target_topics:
            best_idx = None
            best_score = 0.0
            target = None
        else:
            if direction == "h2g":
                scores = sim[i, :]
            else:
                scores = sim[:, i]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            target = target_topics[best_idx]
        rows.append(
            {
                "topic": topic,
                "best_match": target,
                "best_similarity": round(best_score, 6),
            }
        )
    return rows


def summarize_matches(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    scores = [float(row["best_similarity"]) for row in rows]
    summary: dict[str, Any] = {
        "n_topics": len(rows),
        "mean_best_similarity": round(float(np.mean(scores)), 4) if scores else None,
        "median_best_similarity": round(float(np.median(scores)), 4) if scores else None,
        "p75_best_similarity": round(float(np.percentile(scores, 75)), 4) if scores else None,
        "max_best_similarity": round(max(scores), 4) if scores else None,
        "thresholds": {},
    }
    for threshold in thresholds:
        matched = sum(1 for score in scores if score >= threshold)
        summary["thresholds"][str(threshold)] = {
            "matched": matched,
            "rate": round(matched / len(scores), 4) if scores else None,
        }
    return summary


def run_paper(
    row: dict[str, Any],
    args: argparse.Namespace,
    embedder: Embedder,
    thresholds: list[float],
    output_dir: Path,
) -> dict[str, Any]:
    paper_id = str(row.get("paper_id") or "").strip()
    generated_path = find_generated_review_path(paper_id, args.generated_output_roots, row)
    if not generated_path:
        return {"paper_id": paper_id, "status": "missing_generated_review"}

    db_title, human_reviews = load_human_reviews(args.review_db, paper_id)
    if not human_reviews:
        return {"paper_id": paper_id, "status": "missing_human_reviews"}

    generated_title, generated_payload, generated_payloads = load_generated_review(generated_path, args.generated_text_source)
    title = str(row.get("title") or db_title or generated_title or paper_id)
    human_topics = extract_human_topics(paper_id, human_reviews, args.min_topic_chars, args.max_topic_chars)
    generated_topics = extract_generated_topics(paper_id, generated_payloads, args.min_topic_chars, args.max_topic_chars)

    paper_dir = output_dir / "papers" / paper_id
    write_json(paper_dir / "human_topics.json", {"topics": human_topics})
    write_json(paper_dir / "generated_topics.json", {"topics": generated_topics})

    if not human_topics or not generated_topics:
        result = {
            "paper_id": paper_id,
            "title": title,
            "status": "missing_topics",
            "human_topic_count": len(human_topics),
            "generated_topic_count": len(generated_topics),
            "generated_review_path": str(generated_path),
        }
        write_json(paper_dir / "paper_embedding_overlap.json", result)
        return result

    human_embeddings, generated_embeddings, embedding_model_used = embedder.encode_pair(
        [topic["text"] for topic in human_topics],
        [topic["text"] for topic in generated_topics],
    )
    sim = cosine_matrix(human_embeddings, generated_embeddings)
    h2g = best_matches(human_topics, generated_topics, sim, "h2g")
    g2h = best_matches(generated_topics, human_topics, sim, "g2h")
    write_json(paper_dir / "similarity_matrix.json", {"matrix": sim.round(6).tolist()})
    write_jsonl(paper_dir / "human_to_generated_matches.jsonl", h2g)
    write_jsonl(paper_dir / "generated_to_human_matches.jsonl", g2h)

    result = {
        "paper_id": paper_id,
        "title": title,
        "status": "ok",
        "human_review_count": len(human_reviews),
        "human_topic_count": len(human_topics),
        "generated_topic_count": len(generated_topics),
        "embedding_model": embedding_model_used,
        "human_to_generated": summarize_matches(h2g, thresholds),
        "generated_to_human": summarize_matches(g2h, thresholds),
        "generated_review_path": str(generated_path),
    }
    write_json(paper_dir / "paper_embedding_overlap.json", result)
    return result


def aggregate(results: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    ok = [row for row in results if row.get("status") == "ok"]

    def avg(path: tuple[str, ...]) -> float | None:
        values: list[float] = []
        for row in ok:
            cur: Any = row
            for key in path:
                cur = cur.get(key) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                values.append(float(cur))
        return round(sum(values) / len(values), 4) if values else None

    threshold_summary = {}
    for threshold in thresholds:
        key = str(threshold)
        threshold_summary[key] = {
            "human_to_generated_rate": avg(("human_to_generated", "thresholds", key, "rate")),
            "generated_to_human_rate": avg(("generated_to_human", "thresholds", key, "rate")),
        }

    return {
        "created_at_utc": now_utc(),
        "n_papers": len(results),
        "n_ok": len(ok),
        "status_counts": dict(Counter(str(row.get("status")) for row in results)),
        "mean_human_topics": round(float(np.mean([row["human_topic_count"] for row in ok])), 2) if ok else None,
        "mean_generated_topics": round(float(np.mean([row["generated_topic_count"] for row in ok])), 2) if ok else None,
        "mean_human_to_generated_best_similarity": avg(("human_to_generated", "mean_best_similarity")),
        "mean_generated_to_human_best_similarity": avg(("generated_to_human", "mean_best_similarity")),
        "threshold_summary": threshold_summary,
        "papers": results,
    }


def write_summary_md(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Embedding Topic Overlap",
        "",
        f"- Created: {summary['created_at_utc']}",
        f"- Papers: {summary['n_ok']}/{summary['n_papers']}",
        f"- Backend: {args.embedding_backend}",
        f"- Model: `{args.model}`",
        f"- Generated text source: {args.generated_text_source}",
        f"- Mean human topics: {summary['mean_human_topics']}",
        f"- Mean generated topics: {summary['mean_generated_topics']}",
        f"- Mean H->G best similarity: {summary['mean_human_to_generated_best_similarity']}",
        f"- Mean G->H best similarity: {summary['mean_generated_to_human_best_similarity']}",
        "",
        "## Threshold Rates",
        "",
        "| Threshold | H->G recall | G->H precision |",
        "|---:|---:|---:|",
    ]
    for threshold, row in summary["threshold_summary"].items():
        lines.append(f"| {threshold} | {row['human_to_generated_rate']} | {row['generated_to_human_rate']} |")
    lines.extend(
        [
            "",
            "## Per Paper",
            "",
            "| Paper | H topics | G topics | H->G mean max | G->H mean max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["papers"]:
        h2g = row.get("human_to_generated", {}).get("mean_best_similarity") if isinstance(row.get("human_to_generated"), dict) else ""
        g2h = row.get("generated_to_human", {}).get("mean_best_similarity") if isinstance(row.get("generated_to_human"), dict) else ""
        lines.append(
            f"| {row.get('paper_id')} | {row.get('human_topic_count', '')} | "
            f"{row.get('generated_topic_count', '')} | {h2g} | {g2h} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ROOT / "OutputNew" / "Empirics" / f"embedding_topic_overlap_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_requested_papers(args)
    embedder = Embedder(args.embedding_backend, args.model)
    results = [run_paper(row, args, embedder, thresholds, output_dir) for row in rows]
    summary = aggregate(results, thresholds)
    write_json(
        output_dir / "run_config.json",
        {
            "created_at_utc": now_utc(),
            "sample_jsonl": str(args.sample_jsonl) if args.sample_jsonl else None,
            "review_db": str(args.review_db),
            "generated_text_source": args.generated_text_source,
            "embedding_backend": args.embedding_backend,
            "model": args.model,
            "thresholds": thresholds,
            "min_topic_chars": args.min_topic_chars,
            "max_topic_chars": args.max_topic_chars,
        },
    )
    write_json(output_dir / "summary.json", summary)
    write_summary_md(output_dir / "summary.md", summary, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
