#!/usr/bin/env python3
"""
Table of 17 papers with low-confidence outlier reviewers.
Shows all human reviewer scores (marking the outlier), human mean, LLM committee score,
and whether removing the outlier would flip the paper's position relative to cutoff.
"""

from __future__ import annotations
import sqlite3, re, csv, json
from pathlib import Path
import pandas as pd

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DB = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
RDD_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
EMPIRICS = ROOT / "OutputNew" / "Empirics"
TABLE_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    "gemma_ready7_wave1_cached_v2",
    "gemma_ready8_wave2_incremental",
    "gemma_ready8_wave3_single_managed",
]


def parse_num(text):
    if not text:
        return None
    m = re.match(r"(\d+)", str(text))
    return int(m.group(1)) if m else None


# Load RDD metadata
rdd_meta = {}
with open(RDD_CSV) as f:
    for row in csv.DictReader(f):
        if int(row["year"]) in (2018, 2019, 2020):
            rdd_meta[row["paper_id"]] = {
                "accepted": float(row["accepted"]),
                "mean_rating": float(row["mean_rating"]),
                "cutoff": float(row["cutoff"]),
                "year": int(row["year"]),
            }

# Load LLM scores
llm_scores = {}
for run in RUNS:
    run_dir = EMPIRICS / run
    for search_root in [run_dir] + sorted(run_dir.glob("shard_*")):
        papers = search_root / "papers"
        if not papers.is_dir():
            continue
        for p in papers.iterdir():
            cr = p / "coarse_review.json"
            if not cr.exists() or p.name in llm_scores:
                continue
            c = json.loads(cr.read_text())
            llm_scores[p.name] = c.get("rating")

# Load reviews from DB
conn = sqlite3.connect(str(DB))
rdd_ids = list(rdd_meta.keys())
placeholders = ",".join(["?"] * len(rdd_ids))
rows = conn.execute(
    f"""
    SELECT r.paper_id, r.reviewer_id, r.rating, r.confidence
    FROM REVIEW r JOIN SUBMISSION s ON r.paper_id = s.id
    WHERE r.paper_id IN ({placeholders})
    ORDER BY r.paper_id, r.reviewer_id
""",
    rdd_ids,
).fetchall()
conn.close()

# Build per-paper reviews
paper_reviews = {}
for pid, rid, rating, conf in rows:
    r = parse_num(rating)
    c = parse_num(conf)
    if r is None:
        continue
    paper_reviews.setdefault(pid, []).append({"rid": rid, "rating": r, "confidence": c})

# Find the 17 low-conf + outlier cases
table_rows = []
for pid, revs in paper_reviews.items():
    if len(revs) < 2:
        continue
    ratings = [r["rating"] for r in revs]
    mean_r = sum(ratings) / len(ratings)

    for rev in revs:
        deviation = abs(rev["rating"] - mean_r)
        if rev["confidence"] is not None and rev["confidence"] <= 2 and deviation >= 2.0:
            meta = rdd_meta.get(pid, {})
            other_ratings = [r["rating"] for r in revs if r["rid"] != rev["rid"]]
            mean_without = sum(other_ratings) / len(other_ratings)
            cutoff = meta.get("cutoff", 5.67)
            orig_side = "accept" if mean_r >= cutoff else "reject"
            cf_side = "accept" if mean_without >= cutoff else "reject"

            # Format reviewer scores: bold the outlier
            rev_strs = []
            for r in sorted(revs, key=lambda x: x["rating"]):
                score = f"{r['rating']} (c={r['confidence']})"
                if r["rid"] == rev["rid"]:
                    score += " *"
                rev_strs.append(score)

            table_rows.append({
                "Paper ID": pid,
                "Year": meta.get("year", ""),
                "Decision": "Accept" if meta.get("accepted") == 1.0 else "Reject",
                "Human Scores (c=confidence)": ", ".join(rev_strs),
                "Human Mean": round(mean_r, 2),
                "Mean w/o Outlier": round(mean_without, 2),
                "LLM Score": llm_scores.get(pid, ""),
                "Cutoff": cutoff,
                "Flips Cutoff": "Yes" if orig_side != cf_side else "",
            })
            break  # one outlier per paper

# Sort: flips first, then by year
table_rows.sort(key=lambda r: (r["Flips Cutoff"] != "Yes", r["Year"], r["Paper ID"]))

df = pd.DataFrame(table_rows)
df.to_csv(TABLE_DIR / "low_conf_outlier_reviews.csv", index=False)
df.to_latex(
    TABLE_DIR / "low_conf_outlier_reviews.tex",
    index=False,
    escape=True,
    caption="Papers with low-confidence (1--2) outlier reviewers (deviation $\\geq$ 2 points). "
            "Asterisk marks the outlier. LLM Score is the Gemma-4-31B committee weighted-average rating.",
    label="tab:low_conf_outliers",
)

print(df.to_string(index=False))
print(f"\nSaved to {TABLE_DIR / 'low_conf_outlier_reviews.csv'}")
count = 0
for r in table_rows:
    llm = float(r["LLM Score"])
    majority = r["Mean w/o Outlier"]
    outlier_str = [s for s in r["Human Scores (c=confidence)"].split(", ") if "*" in s][0]
    outlier_val = float(outlier_str.split(" ")[0])
    if abs(llm - majority) < abs(llm - outlier_val):
        count += 1
print(f"LLM closer to majority than outlier: {count} / {len(table_rows)}")
