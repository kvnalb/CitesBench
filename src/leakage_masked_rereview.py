"""
Masked re-review — identity-ablation experiment (the mechanism test).

Within-paper design. Each paper is scored twice by the same Gemma rubric:
  original — title + verbatim abstract (identity recoverable)
  masked   — no title, abstract paraphrased by a different model with proper
             names of methods/datasets replaced by generic descriptors
             (identity ablated, content preserved)

If the committee's citation-predictive power survives masking, memorization
can't be driving it. If scores on memorized (high-LAP) papers collapse under
masking, it was identity recall.

Sample: stratified by LAP from outputs/leakage_lap_v1.csv (run that first).

Outputs:
  outputs/leakage_masked_rereview.csv    — one row per paper, incremental
  outputs/leakage_masked_report.md       — paired analysis (skipped in smoke)

Run: python src/leakage_masked_rereview.py [--smoke] [--n 120]
"""
import os
import re
import sys
import json
import time
import sqlite3
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

SCORE_MODEL = "google/gemma-4-31B-it"                      # same as committee
PARAPHRASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # different family — won't share Gemma's recall triggers
OUT_CSV = "outputs/leakage_masked_rereview.csv"
OUT_REPORT = "outputs/leakage_masked_report.md"
LAP_CSV = "outputs/leakage_lap_v1.csv"

SCORE_SYSTEM = (
    "You are a reviewer for ICLR (a top machine-learning conference). "
    "Based on the material below, rate the submission's overall quality and "
    "likely scientific impact on the standard ICLR 1-10 scale "
    "(1=trivial/wrong, 5=borderline, 8=top 50% of accepted papers, 10=seminal). "
    'Respond with JSON only: {"score": <number 1-10>}'
)

PARAPHRASE_PROMPT = (
    "Rewrite the following machine-learning paper abstract entirely in your own words. "
    "Preserve every technical claim, method detail, and result, but change all phrasing "
    "and sentence structure. Replace any proper names of proposed methods, model names, "
    "or dataset names with generic descriptors (e.g. 'the proposed architecture', "
    "'a large image-classification benchmark'). Output ONLY the rewritten abstract, "
    "nothing else.\n\nAbstract:\n{abstract}"
)


def _chat(client, model, messages, max_tokens=512, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0)
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5)


def _score(client, body):
    # Gemma-4-31B-it is a thinking model: budget for ~1750 thinking tokens + JSON
    raw = _chat(client, SCORE_MODEL,
                [{"role": "system", "content": SCORE_SYSTEM},
                 {"role": "user", "content": body}],
                max_tokens=3000)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in score response: {raw[:200]!r}")
    return float(json.loads(m.group())["score"])


def build_lap_sample(n, smoke=False):
    """Stratified by LAP: half memorized (lap>=0.5), half not, from probed papers."""
    lap = pd.read_csv(LAP_CSV)
    lap = lap[lap["lap"].notna()][["paper_id", "lap"]]
    ev = pd.read_csv("outputs/eval_table.csv")
    con = sqlite3.connect("data/gen_review.db")
    abstracts = pd.read_sql("SELECT id AS paper_id, abstract FROM SUBMISSION", con)
    con.close()
    df = lap.merge(ev, on="paper_id").merge(abstracts, on="paper_id")
    df = df[df["abstract"].notna() & df["title"].notna()]
    if smoke:
        return df.head(5)
    rng = np.random.default_rng(42)
    hi = df[df["lap"] >= 0.5]
    lo = df[df["lap"] < 0.5]
    take = lambda g, k: g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)]
    return pd.concat([take(hi, n // 2), take(lo, n // 2)]).reset_index(drop=True)


def run(sample, workers=10):
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        sys.exit("ERROR: TOGETHER_API_KEY not set in .env")

    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV)["paper_id"].unique())
    todo = sample[~sample["paper_id"].isin(done)]
    print(f"Score model: {SCORE_MODEL}   Paraphrase model: {PARAPHRASE_MODEL}")
    print(f"Already done: {len(done)}, to run: {len(todo)}, workers: {workers} "
          f"(3 API calls per paper)")

    from openai import OpenAI
    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    lock = threading.Lock()
    counter = [0]

    with open(OUT_CSV, "a") as fout:
        if write_header:
            fout.write("paper_id,year,lap,openalex_citations,score_original,score_masked,masked_abstract_chars\n")

        def run_one(row):
            client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
            try:
                masked_abs = _chat(client, PARAPHRASE_MODEL,
                                   [{"role": "user",
                                     "content": PARAPHRASE_PROMPT.format(abstract=row.abstract)}],
                                   max_tokens=1024)
                if len(masked_abs) < 200:
                    raise ValueError(f"paraphrase suspiciously short ({len(masked_abs)} chars)")
                s_orig = _score(client, f"Title: {row.title}\n\nAbstract: {row.abstract}")
                s_mask = _score(client, f"Abstract: {masked_abs}")
            except Exception as e:
                print(f"  SKIP {row.paper_id}: {e}")
                return
            with lock:
                fout.write(f"{row.paper_id},{row.year},{row.lap:.6f},"
                           f"{row.openalex_citations},{s_orig},{s_mask},{len(masked_abs)}\n")
                fout.flush()
                counter[0] += 1
                print(f"  {counter[0]}/{len(todo)}  {row.paper_id}  lap={row.lap:.2f}  "
                      f"orig={s_orig:.1f}  masked={s_mask:.1f}  Δ={s_orig - s_mask:+.1f}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, row) for row in todo.itertuples()]
            for f in as_completed(futures):
                f.result()


def run_report():
    from scipy import stats

    df = pd.read_csv(OUT_CSV)
    df = df[df["openalex_citations"].notna()]
    df["log_cites"] = np.log1p(df["openalex_citations"])
    df["hi_lap"] = df["lap"] >= 0.5
    df["delta"] = df["score_original"] - df["score_masked"]
    n = len(df)

    def corr(sub, col):
        if len(sub) < 5 or sub[col].nunique() < 2:
            return float("nan"), float("nan")
        r = stats.spearmanr(sub[col], sub["log_cites"])
        return r.statistic, r.pvalue

    rows = []
    for label, sub in [("all", df), ("high-LAP (memorized)", df[df["hi_lap"]]),
                       ("low-LAP (not memorized)", df[~df["hi_lap"]])]:
        ro, po = corr(sub, "score_original")
        rm, pm = corr(sub, "score_masked")
        rows.append(f"| {label} | {len(sub)} | {ro:.3f} (p={po:.3g}) | {rm:.3f} (p={pm:.3g}) |")

    # Does masking deflate scores more for memorized papers?
    hi, lo = df[df["hi_lap"]], df[~df["hi_lap"]]
    if len(hi) >= 3 and len(lo) >= 3:
        tt = stats.mannwhitneyu(hi["delta"], lo["delta"], alternative="greater")
        delta_line = (f"Score drop under masking: high-LAP mean Δ = {hi['delta'].mean():+.2f}, "
                      f"low-LAP mean Δ = {lo['delta'].mean():+.2f} "
                      f"(Mann-Whitney one-sided p = {tt.pvalue:.3g})")
    else:
        delta_line = "Insufficient N per LAP stratum for the Δ comparison."

    report = f"""# Masked Re-Review — identity ablation ({SCORE_MODEL})

Within-paper design, N = {n}. Each paper scored twice on the same 1-10 rubric:
**original** (title + verbatim abstract) vs **masked** (no title, abstract
paraphrased by {PARAPHRASE_MODEL} with proper names genericized).

## Citation-predictive power by arm (Spearman ρ of score vs log citations)

| Stratum | N | original | masked |
|---|---|---|---|
{chr(10).join(rows)}

## Masking-induced score deflation

{delta_line}

## Reading

- Predictive power **survives masking** (masked ρ ≈ original ρ, incl. on
  low-LAP papers) → judgment is content-driven; memorization is not the story.
- Predictive power **collapses under masking**, or high-LAP papers lose
  disproportionately more → the original scores rode on identity recall.

Caveat: masking removes the title's semantic content along with its identity
value, so a small ρ drop is expected even with zero leakage; the LAP-stratified
contrast (does high-LAP drop MORE?) is the identifying comparison.
"""
    with open(OUT_REPORT, "w") as f:
        f.write(report)
    print(f"\n{delta_line}")
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="5 papers only")
    parser.add_argument("--n", type=int, default=120, help="total papers (half high-LAP, half low)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(LAP_CSV):
        sys.exit(f"ERROR: {LAP_CSV} not found — run leakage_lap_v1.py first.")

    if not args.report_only:
        sample = build_lap_sample(args.n, smoke=args.smoke)
        print(f"Sample: {len(sample)} papers "
              f"(high-LAP: {(sample['lap'] >= 0.5).sum()}, low-LAP: {(sample['lap'] < 0.5).sum()})")
        run(sample, workers=args.workers)

    if args.smoke:
        print("\nSmoke done — inspect outputs/leakage_masked_rereview.csv")
    elif os.path.exists(OUT_CSV):
        run_report()
