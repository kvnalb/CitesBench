"""
01_select_papers.py

Select ICLR 2025 papers from the Foundation/Frontier Models (LLMs) track
and export metadata + abstracts to JSONL.

Source: data/LLM-Reviewer-03042026/data/gen_review.db
Output: rawdata/ICLR2025/foundation_or_frontier_models_including_LLMs/abstracts.jsonl
"""

import sqlite3
import json
import os
from collections import Counter

# --- Config ---
DB_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "LLM-Reviewer-03042026", "data", "gen_review.db",
)
YEAR = 2025
PRIMARY_AREA = "foundation or frontier models, including LLMs"
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "rawdata", "ICLR2025", "foundation_or_frontier_models_including_LLMs",
)
OUT_FILE = os.path.join(OUT_DIR, "abstracts.jsonl")


def main():
    conn = sqlite3.connect(DB_PATH)

    # --- Selection query ---
    # We select ALL papers in this track for the given year.
    # No filtering on decision — we want rejects + accepts for calibration work.
    papers = conn.execute(
        """
        SELECT id, title, abstract, pdf, decision, keywords
        FROM SUBMISSION
        WHERE when_submitted = ?
          AND primary_area = ?
        ORDER BY id
        """,
        (YEAR, PRIMARY_AREA),
    ).fetchall()
    conn.close()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(OUT_FILE, "w") as f:
        for row in papers:
            paper = {
                "paper_id": row[0],
                "title": row[1],
                "abstract": row[2],
                "pdf_url": row[3],
                "decision": row[4],
                "keywords": row[5],
            }
            f.write(json.dumps(paper) + "\n")

    # --- Summary ---
    decisions = Counter(r[4] for r in papers)
    print(f"Exported {len(papers)} papers to {OUT_FILE}")
    print(f"\nDecision breakdown:")
    for dec, count in decisions.most_common():
        print(f"  {dec:<30} {count:>5} ({count / len(papers) * 100:.1f}%)")


if __name__ == "__main__":
    main()
