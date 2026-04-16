"""
03_export_human_scores.py

Export human review scores for selected papers from the SQLite database.

Produces two files:
  1. reviews_individual.jsonl  — one row per (paper, reviewer) with raw scores
  2. reviews_aggregated.jsonl  — one row per paper with mean/std/min/max across reviewers

Source: data/LLM-Reviewer-03042026/data/gen_review.db
Output: rawdata/ICLR2025/foundation_or_frontier_models_including_LLMs/
"""

import sqlite3
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# --- Config ---
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
YEAR = 2025
PRIMARY_AREA = "foundation or frontier models, including LLMs"
OUT_DIR = ROOT / "rawdata" / "ICLR2025" / "foundation_or_frontier_models_including_LLMs"
PROCESSED_HUMAN_REVIEW_DIR = ROOT / "processed" / "ICLR2025_Foundation_LLMs" / "HumanReview"

# ICLR 2025 review dimensions (different from earlier years)
# rating, confidence — always present
# soundness, presentation, contribution — ICLR 2025 form
# correctness, technical_novelty, empirical_novelty — NOT populated for 2025
SCORE_COLUMNS = [
    "rating",
    "confidence",
    "soundness",
    "presentation",
    "contribution",
]


def parse_score(raw):
    """Parse a numeric score from OpenReview format.

    Examples:
        '3: reject, not good enough' -> 3.0
        '4'                          -> 4.0
        ''                           -> None
    """
    if not raw or raw.strip() in ("", "Not applicable", "N/A"):
        return None
    raw = raw.strip()
    match = re.match(r"^(\d+(?:\.\d+)?)", raw)
    if match:
        return float(match.group(1))
    return None


def normalise_text(raw):
    return (raw or "").strip()


def compose_review_text(sections):
    ordered_keys = ("summary", "strength", "weaknesses", "questions")
    parts = []
    for key in ordered_keys:
        value = normalise_text(sections.get(key))
        if not value:
            continue
        label = key.replace("_", " ").title()
        parts.append(f"{label}:\n{value}")
    return "\n\n".join(parts)


def main():
    conn = sqlite3.connect(str(DB_PATH))

    # Get all reviews for our selected papers
    rows = conn.execute(
        """
        SELECT
            r.paper_id,
            r.reviewer_id,
            s.title,
            s.decision,
            r.rating,
            r.confidence,
            r.soundness,
            r.presentation,
            r.contribution,
            r.summary,
            r.strength,
            r.weaknesses,
            r.questions,
            r.main_review,
            r.summary_of_the_review
        FROM REVIEW r
        JOIN SUBMISSION s ON r.paper_id = s.id
        WHERE s.when_submitted = ?
          AND s.primary_area = ?
        ORDER BY r.paper_id, r.reviewer_id
        """,
        (YEAR, PRIMARY_AREA),
    ).fetchall()
    conn.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_HUMAN_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # --- Individual reviews ---
    individual_path = OUT_DIR / "reviews_individual.jsonl"
    paper_scores = defaultdict(list)
    paper_reviews = defaultdict(list)

    with open(individual_path, "w", encoding="utf-8") as f:
        for row in rows:
            paper_id, reviewer_id, title, decision = row[0], row[1], row[2], row[3]
            raw_scores = {
                "rating": row[4],
                "confidence": row[5],
                "soundness": row[6],
                "presentation": row[7],
                "contribution": row[8],
            }
            review_sections = {
                "summary": normalise_text(row[9]),
                "strength": normalise_text(row[10]),
                "weaknesses": normalise_text(row[11]),
                "questions": normalise_text(row[12]),
            }
            legacy_review_text = normalise_text(row[13]) or normalise_text(row[14])
            review_text = compose_review_text(review_sections) or legacy_review_text

            parsed = {}
            for col in SCORE_COLUMNS:
                parsed[col] = parse_score(raw_scores[col])

            record = {
                "paper_id": paper_id,
                "reviewer_id": reviewer_id,
                "title": title,
                "decision": decision,
                "scores": parsed,
                "review_text": review_text,
                "review_sections": review_sections,
            }
            f.write(json.dumps(record) + "\n")
            paper_scores[paper_id].append((title, decision, parsed))
            paper_reviews[paper_id].append(
                {
                    "reviewer_id": reviewer_id,
                    "scores": parsed,
                    "review_text": review_text,
                    "review_sections": review_sections,
                }
            )

    print(f"Individual reviews: {len(rows)} rows -> {individual_path}")

    # --- Aggregated per paper ---
    aggregated_path = OUT_DIR / "reviews_aggregated.jsonl"

    import numpy as np

    with open(aggregated_path, "w", encoding="utf-8") as f:
        for paper_id, entries in sorted(paper_scores.items()):
            title = entries[0][0]
            decision = entries[0][1]
            n_reviews = len(entries)

            agg = {}
            for col in SCORE_COLUMNS:
                values = [e[2][col] for e in entries if e[2][col] is not None]
                if values:
                    agg[col] = {
                        "mean": round(float(np.mean(values)), 3),
                        "std": round(float(np.std(values, ddof=1)), 3) if len(values) > 1 else 0.0,
                        "min": float(min(values)),
                        "max": float(max(values)),
                        "values": values,
                        "count": len(values),
                    }
                else:
                    agg[col] = None

            record = {
                "paper_id": paper_id,
                "title": title,
                "decision": decision,
                "n_reviews": n_reviews,
                "scores": agg,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Aggregated reviews: {len(paper_scores)} papers -> {aggregated_path}")

    # --- Processed per-paper JSONs used by the main pipelines ---
    for paper_id, entries in sorted(paper_scores.items()):
        title = entries[0][0]
        decision = entries[0][1]
        n_reviews = len(entries)

        agg = {}
        for col in SCORE_COLUMNS:
            values = [e[2][col] for e in entries if e[2][col] is not None]
            if values:
                import numpy as np

                agg[col] = {
                    "mean": round(float(np.mean(values)), 3),
                    "std": round(float(np.std(values, ddof=1)), 3) if len(values) > 1 else 0.0,
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "values": values,
                    "count": len(values),
                }
            else:
                agg[col] = None

        payload = {
            "paper_id": paper_id,
            "title": title,
            "decision": decision,
            "aggregated": agg,
            "n_reviews": n_reviews,
            "reviews": paper_reviews[paper_id],
        }
        out_path = PROCESSED_HUMAN_REVIEW_DIR / f"{paper_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Processed HumanReview JSONs refreshed: {len(paper_scores)} papers -> {PROCESSED_HUMAN_REVIEW_DIR}")

    # --- Summary stats ---
    print(f"\n--- Summary ---")
    print(f"Papers with reviews: {len(paper_scores)}")
    print(f"Total individual reviews: {len(rows)}")

    all_ratings = []
    for entries in paper_scores.values():
        for _, _, scores in entries:
            if scores["rating"] is not None:
                all_ratings.append(scores["rating"])

    if all_ratings:
        print(f"\nRating distribution (individual reviewers):")
        from collections import Counter
        dist = Counter(int(r) for r in all_ratings)
        for score in sorted(dist):
            pct = dist[score] / len(all_ratings) * 100
            print(f"  {score:>2}: {dist[score]:>5} ({pct:>5.1f}%) {'█' * int(pct)}")
        print(f"\n  Mean: {np.mean(all_ratings):.2f}, Median: {np.median(all_ratings):.1f}")

    # Dimension coverage
    print(f"\nDimension coverage:")
    for col in SCORE_COLUMNS:
        filled = sum(1 for row in rows if parse_score(row[4 + SCORE_COLUMNS.index(col)]) is not None)
        print(f"  {col:<30} {filled}/{len(rows)} ({filled / len(rows) * 100:.0f}%)")

    section_offsets = {
        "summary": 9,
        "strength": 10,
        "weaknesses": 11,
        "questions": 12,
    }
    print(f"\nStructured text coverage:")
    for name, offset in section_offsets.items():
        filled = sum(1 for row in rows if normalise_text(row[offset]))
        print(f"  {name:<30} {filled}/{len(rows)} ({filled / len(rows) * 100:.0f}%)")


if __name__ == "__main__":
    main()
