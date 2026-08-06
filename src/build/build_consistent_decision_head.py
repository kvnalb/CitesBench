"""Merge all_paper_results.csv with the GPT-OSS-20b old-RDD decision-head
rerun so every paper (old_rdd + new_remaining) uses openai/gpt-oss-20b as the
decision head, on the same cached committee reviews.
"""
import csv
import os

BASE_CSV = "data/archive/all_paper_results.csv"
RERUN_CSV = "data/fresh_dropbox_download/old_rdd_gptoss20b_decision_head_results.csv"
OUT_CSV = "outputs/all_paper_results_consistent_gptoss20b.csv"

DECISION_HEAD_FIELDS = [
    "decision",
    "accepted",
    "decision_head_model",
    "deepseek_decision",
    "deepseek_p_accept",
    "deepseek_margin",
    "deepseek_elapsed_seconds",
    "deepseek_http_error",
    "deepseek_estimated_cost_usd",
]


def load_rerun(path):
    with open(path, newline="") as f:
        return {row["paper_id"]: row for row in csv.DictReader(f)}


def main():
    os.makedirs("outputs", exist_ok=True)
    rerun = load_rerun(RERUN_CSV)

    with open(BASE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_replaced = 0
    for row in rows:
        replacement = rerun.get(row["paper_id"])
        if replacement is None:
            continue
        for field in DECISION_HEAD_FIELDS:
            row[field] = replacement[field]
        n_replaced += 1

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    heads = {row["decision_head_model"] for row in rows}
    print(f"rows: {len(rows)}, old_rdd rows replaced: {n_replaced}")
    print(f"distinct decision_head_model values: {heads}")
    assert heads == {"openai/gpt-oss-20b"}, "decision head is not consistent"
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
