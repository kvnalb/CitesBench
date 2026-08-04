"""
Fame-recall probe — the outcome-side twin of the LAP decision probe.

Our outcome variable is citations, so the direct leakage threat is memorized
*fame*, not memorized accept/reject decisions. Title-only recall query: is
this paper widely cited? FAME = P(high) + P(low) (commitment), fame U-D =
P(high) - P(low) (directional recall of prominence).

Same model, sample, and logprob machinery as leakage_lap_v1 (imported).

Outputs:
  outputs/leakage_fame_v1.csv    — incremental, one row per API call
  outputs/leakage_fame_report.md — recall accuracy vs actual citation rank

Run: python src/leakage_fame_v1.py [--smoke] [--n 300] [--report-only]
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
from leakage_lap_v1 import probe_one, build_sample, MODEL
from prompts import load

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OUT_CSV = "outputs/leakage_fame_v1.csv"
OUT_REPORT = "outputs/leakage_fame_report.md"

HIGH_TOKENS = {"high", "highly", "widely"}
LOW_TOKENS = {"low", "rarely", "not"}
UNKNOWN_TOKENS = {"unknown"}


def fame_prompt(title, year):
    return load("recall/fame_oneword", title=title, year=year)


def run_probes(df, smoke=False, workers=10):
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        sys.exit("ERROR: TOGETHER_API_KEY not set in .env")

    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV)["paper_id"].unique())
    todo = df[~df["paper_id"].isin(done)]
    if smoke:
        todo = todo.head(5)

    print(f"Model: {MODEL}")
    print(f"Already done: {len(done)}, to fetch: {len(todo)}, workers: {workers}")

    from openai import OpenAI
    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    lock = threading.Lock()
    counter = [0]

    with open(OUT_CSV, "a") as fout, open("outputs/leakage_fame_traces.jsonl", "a") as ftrace:
        if write_header:
            fout.write("paper_id,year,citation_pct_rank,answer,p_high,p_low,p_unknown,fame,fame_ud\n")

        def fetch_one(row):
            client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
            try:
                answer, p_hi, p_lo, p_unk, ntok, trace = probe_one(
                    client, fame_prompt(row.title, row.year),
                    pos_set=HIGH_TOKENS, neg_set=LOW_TOKENS, unk_set=UNKNOWN_TOKENS,
                    labels=("high", "low", "unknown"))
            except Exception as e:
                with lock:
                    fout.write(f"{row.paper_id},{row.year},{row.citation_pct_rank},ERROR,,,,,\n")
                    fout.flush()
                print(f"  SKIP {row.paper_id}: {e}")
                return
            fame = p_hi + p_lo
            fud = p_hi - p_lo
            with lock:
                fout.write(f"{row.paper_id},{row.year},{row.citation_pct_rank},{answer},"
                           f"{p_hi:.6f},{p_lo:.6f},{p_unk:.6f},{fame:.6f},{fud:.6f}\n")
                fout.flush()
                if trace:
                    ftrace.write(json.dumps({"paper_id": row.paper_id, "probe": "fame",
                                             "answer": answer, "trace": trace}) + "\n")
                    ftrace.flush()
                counter[0] += 1
                print(f"  {counter[0]}/{len(todo)}  {row.paper_id}  answer={answer}"
                      f"  fame={fame:.3f} ud={fud:+.3f}  (tokens={ntok})")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_one, row) for row in todo.itertuples()]
            for f in as_completed(futures):
                f.result()


def run_report():
    from scipy import stats

    res = pd.read_csv(OUT_CSV)
    res = res[res["fame"].notna() & res["citation_pct_rank"].notna()]
    n = len(res)
    res["is_top_decile"] = res["citation_pct_rank"] >= 0.9

    fame_mean = res["fame"].mean()
    fame_hi = (res["fame"] >= 0.5).mean()
    # Recall accuracy: among committed answers, does 'high' line up with actual top decile?
    committed = res[res["fame"] >= 0.5]
    acc = ((committed["fame_ud"] > 0) == committed["is_top_decile"]).mean() if len(committed) else float("nan")
    # Directional recall vs actual rank
    sp = stats.spearmanr(res["fame_ud"], res["citation_pct_rank"])
    # Fame recall by actual citation decile
    res["decile"] = (res["citation_pct_rank"] * 10).clip(0, 9).astype(int)
    by_dec = res.groupby("decile").agg(n=("fame", "size"), mean_fame=("fame", "mean"),
                                       frac_said_high=("fame_ud", lambda s: (s > 0).mean()))

    report = f"""# Fame-Recall Probe — {MODEL}

Title-only recall of citation prominence (the outcome-side twin of the LAP
decision probe). N = {n}.

| Statistic | Value |
|---|---|
| Mean FAME (commitment) | {fame_mean:.3f} |
| Fraction FAME ≥ 0.5 | {fame_hi:.1%} |
| Recall accuracy on committed answers | {acc:.1%} |
| Spearman ρ (fame U-D vs actual citation rank) | {sp.statistic:.3f} (p={sp.pvalue:.3g}) |

## Fame recall by actual citation decile

{by_dec.to_markdown(floatfmt=".3f")}

A positive Spearman ρ means the model can identify highly-cited papers from
the title alone — direct evidence that fame is memorized and available to
contaminate any citation-adjacent judgment. Use `fame` alongside `lap` as an
exclusion criterion in leakage_exclusion_eval.
"""
    with open(OUT_REPORT, "w") as f:
        f.write(report)
    print(f"\nFame commitment: mean={fame_mean:.3f}, ≥0.5: {fame_hi:.1%}, "
          f"accuracy={acc:.1%}, Spearman ρ={sp.statistic:.3f} (p={sp.pvalue:.3g})")
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="5 papers only")
    parser.add_argument("--full", action="store_true", help="all eligible papers")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv("outputs/eval_table.csv")
    df = df[df["committee_rating"].notna() & df["title"].notna()].reset_index(drop=True)

    if args.smoke:
        sample = df.head(5)
    elif args.full:
        sample = df
    else:
        # same stratified sample as the LAP probe → papers get both probes
        sample = build_sample(df, args.n)
        print(f"Stratified sample: {len(sample)} papers")

    if not args.report_only:
        run_probes(sample, smoke=args.smoke, workers=args.workers)

    if args.smoke:
        print("\nSmoke done — inspect outputs/leakage_fame_v1.csv")
    elif os.path.exists(OUT_CSV):
        run_report()
