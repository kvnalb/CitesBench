"""
Re-elicit thinking traces for a fame-stratified sample of the fame-recall probe.

Sample: 10 papers per stratum (answered high / answered low / unknown) from the
completed fame sweep. Greedy decode → traces closely reproduce the original
runs' hidden reasoning. Forensic question: does the model's thinking show
*memory* ("I remember this paper") or *judgment* ("this title sounds solid")?

Output: outputs/leakage_fame_traces_sample30.jsonl

Run: python src/leakage_fame_trace_sample.py [--n-per-stratum 10]
"""
import os
import sys
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from leakage_lap_v1 import probe_one
from leakage_fame_v1 import fame_prompt, HIGH_TOKENS, LOW_TOKENS, UNKNOWN_TOKENS

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OUT_JSONL = "outputs/leakage_fame_traces_sample30.jsonl"


def _answer_correct(answer, pct_rank):
    """TP/FP/FN/TN of the fame answer vs actual top-decile status; None for unknown/missing."""
    if answer == "unknown" or pd.isna(pct_rank):
        return None
    top = pct_rank >= 0.9
    return {("high", True): "TP", ("high", False): "FP",
            ("low", True): "FN", ("low", False): "TN"}[(answer, top)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-stratum", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        sys.exit("ERROR: TOGETHER_API_KEY not set in .env")

    fame = pd.read_csv("outputs/leakage_fame_v1.csv")
    fame = fame[fame["fame"].notna()]
    ev = pd.read_csv("outputs/eval_table.csv")
    # raw within-year citation percentile — matches the prompt's "top 10% of its
    # year" and covers papers the field-normalized citation_pct_rank misses
    ev["cite_year_pct_rank"] = ev.groupby("year")["openalex_citations"].rank(pct=True)
    df = fame.merge(ev[["paper_id", "title", "openalex_citations", "cite_year_pct_rank"]],
                    on="paper_id")

    rng = np.random.default_rng(42)
    strata = {"high": df[df["answer"] == "high"],
              "low": df[df["answer"] == "low"],
              "unknown": df[df["answer"] == "unknown"]}
    sample = pd.concat(
        g.iloc[rng.choice(len(g), size=min(args.n_per_stratum, len(g)), replace=False)]
        for g in strata.values()).reset_index(drop=True)

    done = set()
    if os.path.exists(OUT_JSONL):
        done = {json.loads(l)["paper_id"] for l in open(OUT_JSONL)}
    todo = sample[~sample["paper_id"].isin(done)]
    print(f"Sample: {len(sample)} ({args.n_per_stratum}/stratum), done: {len(done)}, to fetch: {len(todo)}")

    from openai import OpenAI
    lock = threading.Lock()
    counter = [0]

    with open(OUT_JSONL, "a") as fout:
        def fetch_one(row):
            client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
            try:
                answer, p_hi, p_lo, p_unk, ntok, trace = probe_one(
                    client, fame_prompt(row.title, row.year),
                    pos_set=HIGH_TOKENS, neg_set=LOW_TOKENS, unk_set=UNKNOWN_TOKENS,
                    labels=("high", "low", "unknown"))
            except Exception as e:
                print(f"  SKIP {row.paper_id}: {e}")
                return
            with lock:
                fout.write(json.dumps({
                    "paper_id": row.paper_id,
                    "title": row.title,
                    "year": int(row.year),
                    "citations": None if pd.isna(row.openalex_citations) else int(row.openalex_citations),
                    "citation_pct_rank": None if pd.isna(row.citation_pct_rank) else round(float(row.citation_pct_rank), 4),
                    "cite_year_pct_rank": None if pd.isna(row.cite_year_pct_rank) else round(float(row.cite_year_pct_rank), 4),
                    "is_top_decile": None if pd.isna(row.cite_year_pct_rank) else bool(row.cite_year_pct_rank >= 0.9),
                    "original_answer": row.answer,
                    "answer_correct": _answer_correct(row.answer, row.cite_year_pct_rank),
                    "reelicited_answer": answer,
                    "trace": trace,
                }) + "\n")
                fout.flush()
                counter[0] += 1
                print(f"  {counter[0]}/{len(todo)}  {row.paper_id}  orig={row.answer}"
                      f"  now={answer}  trace_chars={len(trace or '')}")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, row) for row in todo.itertuples()]
            for f in as_completed(futures):
                f.result()

    print(f"\nTraces: {OUT_JSONL}")


if __name__ == "__main__":
    main()
