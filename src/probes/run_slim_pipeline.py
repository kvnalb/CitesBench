"""
Run the slim 9-call review pipeline over ICLR 2025 accepted papers.

Population is the frozen list from src/build/build_slim_2025_papers.py — 3,703 papers,
accepts only, because ReviewArena's 2025 slice has no rejects and the 5,019 rejected
submissions in gen_review.db have no full text anywhere local. This run can therefore
measure score spread and citation correlation, but NOT accept-vs-reject separation.

Nine calls per paper: contribution_extraction, intro_notes, method_notes (skipped when
no methodology section is found), contribution_notes, four persona reviews, committee
synthesis. Papers are processed in `run_order`, a seeded shuffle, so --n 5 is a
reproducible smoke test rather than a decision-correlated slice.

Everything is traced. One JSONL line per LLM call, appended the moment it returns,
holding the verbatim messages and the untouched response. At ~33k calls a crash at
paper 2,800 must cost nothing, so both outputs are append-only and a restart skips
paper_ids already present in the CSV.

Two fields worth watching in the per-paper CSV, because both fail quietly:
  n_calls          8 rather than 9 means no methodology section was detected
  text_synthesis   'fallback' means the committee synthesis call threw and scores were
                   aggregated arithmetically — the row looks normal either way

Outputs:
  outputs/slim_2025_{model}.csv          one row per paper, incremental
  outputs/slim_2025_{model}_traces.jsonl one line per LLM call, incremental

Run: python src/probes/run_slim_pipeline.py --model gemma --n 5          # smoke test
     python src/probes/run_slim_pipeline.py --model gemma --workers 8    # full 3,703
"""
import os
import sys
import csv
import json
import time
import argparse
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import MODELS
from build.build_slim_2025_papers import load_year
from build.normalize_paper_markdown import normalize
from probes.slim_pipeline import review_paper_slim

load_dotenv()
os.makedirs("outputs", exist_ok=True)

SAMPLE_CSV = "outputs/samples/slim_2025_papers.csv"
PERSONAS = ["empiricist", "theorist", "systems_pragmatist", "novelty_gatekeeper"]

FIELDS = ["paper_id", "decision", "primary_area", "markdown_chars", "run_order",
          "model", "rating", "confidence", "soundness", "presentation", "contribution",
          "recommendation", "n_calls", "text_synthesis", "prompt_tokens",
          "completion_tokens", "retried_calls", "elapsed_s", "error", "ts"]

_lock = threading.Lock()


def paths(model_key):
    return (f"outputs/slim_2025_{model_key}.csv",
            f"outputs/slim_2025_{model_key}_traces.jsonl")


def done_ids(csv_path):
    """paper_ids already written, so a restart resumes instead of re-billing."""
    if not os.path.exists(csv_path):
        return set()
    try:
        return set(pd.read_csv(csv_path).paper_id.astype(str))
    except Exception:
        return set()


def one_paper(row, markdown, model_key, fcsv, writer, ftr):
    t0 = time.time()
    rec = {k: row.get(k) for k in ("paper_id", "decision", "primary_area",
                                   "markdown_chars", "run_order")}
    rec["model"] = model_key
    try:
        result, traces = review_paper_slim(
            paper_id=row["paper_id"],
            markdown=normalize(markdown),
            model_key=model_key,
            personas=PERSONAS,
        )
        rec.update({
            "rating": result.get("rating"),
            "confidence": result.get("confidence"),
            "soundness": result.get("soundness"),
            "presentation": result.get("presentation"),
            "contribution": result.get("contribution"),
            "recommendation": result.get("recommendation"),
            "n_calls": len(traces),
            "text_synthesis": result.get("text_synthesis"),
            "prompt_tokens": sum(t.get("prompt_tokens") or 0 for t in traces),
            "completion_tokens": sum(t.get("completion_tokens") or 0 for t in traces),
            # a call that needed a repair turn got a different conversation than a
            # clean one; worth seeing per paper rather than only in the traces
            "retried_calls": sum(1 for t in traces if (t.get("attempts") or 1) > 1),
            "error": "",
        })
    except Exception as e:
        # a dead paper must not kill a 3,703-paper run; record it and move on
        traces = []
        rec.update({"n_calls": 0, "error": f"{type(e).__name__}: {e}"[:300]})

    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _lock:
        writer.writerow({k: rec.get(k) for k in FIELDS})
        fcsv.flush()
        for t in traces:
            t["paper_id"] = row["paper_id"]
            ftr.write(json.dumps(t) + "\n")
        ftr.flush()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=0, help="first N by run_order; 0 = all")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(SAMPLE_CSV):
        sys.exit(f"ERROR: {SAMPLE_CSV} missing — run src/build/build_slim_2025_papers.py")

    papers = pd.read_csv(SAMPLE_CSV).sort_values("run_order")
    if args.n:
        papers = papers.head(args.n)

    out_csv, out_traces = paths(args.model)
    seen = done_ids(out_csv)
    todo = papers[~papers.paper_id.astype(str).isin(seen)]
    if seen:
        print(f"resuming: {len(seen)} papers already done, {len(todo)} to go")
    if todo.empty:
        sys.exit("nothing to do")

    # full text stays in the parquet until needed — 3,703 x 79k chars is ~290MB
    text = load_year().set_index("forum_id").markdown

    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fcsv, open(out_traces, "a") as ftr:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one_paper, r._asdict(), text[r.paper_id],
                              args.model, fcsv, writer, ftr): r.paper_id
                    for r in todo.itertuples(index=False)}
            for i, f in enumerate(as_completed(futs), 1):
                rec = f.result()
                flag = ""
                if rec.get("error"):
                    flag = f"  ERROR {rec['error'][:60]}"
                elif rec.get("n_calls") != 9:
                    flag = f"  only {rec['n_calls']} calls"
                print(f"[{i}/{len(futs)}] {rec['paper_id']} "
                      f"rating={rec.get('rating')} {rec['elapsed_s']}s "
                      f"{rec.get('prompt_tokens')}in/{rec.get('completion_tokens')}out"
                      f"{flag}", flush=True)

    d = pd.read_csv(out_csv)
    ok = d[d.error.isna() | (d.error == "")]
    print(f"\n{len(ok)}/{len(d)} papers scored -> {out_csv}")
    if len(ok):
        print(f"rating: mean {ok.rating.mean():.2f} sd {ok.rating.std():.2f} "
              f"range {ok.rating.min()}-{ok.rating.max()}")
        print(f"calls: {ok.n_calls.value_counts().to_dict()}   "
              f"synthesis: {ok.text_synthesis.value_counts().to_dict()}   "
              f"papers with a retried call: {(ok.retried_calls > 0).sum()}")
        print(f"tokens/paper: {ok.prompt_tokens.median():,.0f} in, "
              f"{ok.completion_tokens.median():,.0f} out   "
              f"median {ok.elapsed_s.median():.0f}s/paper")
        print(f"extrapolated to 3,703 papers: "
              f"{ok.prompt_tokens.median() * 3703 / 1e6:,.0f}M in, "
              f"{ok.completion_tokens.median() * 3703 / 1e6:,.0f}M out")


if __name__ == "__main__":
    main()
