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

Rubric (--rubric, prompts live in prompts/review/):
  calibrated (default)  5 float dimensions on 0-5, anti-bias warnings, 5 few-shot
                        examples calibrated to normalized ICLR ground truth;
                        `rating` is the overall score that gets stored
  simple                the original single overall score on ICLR 1-10

Outputs (suffixed per rubric so the two scales never share a file):
  outputs/leakage_masked_rereview[_calibrated].csv  — one row per paper, every rubric
                        dimension for both arms, incremental
  outputs/leakage_masked_traces[_calibrated].jsonl  — one line per API call: system and
                        user prompt verbatim + sha1, params, raw completion, parsed JSON
                        (including `rationale`). Written for failed papers too.
  outputs/leakage_masked_report[_calibrated].md     — paired analysis (skipped in smoke)

Run: python src/probes/leakage_masked_rereview.py [--smoke] [--n 120] [--rubric simple]
"""
import os
import re
import sys
import json
import time
import sqlite3
import hashlib
import argparse
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import load

load_dotenv()
os.makedirs("outputs", exist_ok=True)

SCORE_MODEL = "google/gemma-4-31B-it"                      # same as committee
PARAPHRASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # different family — won't share Gemma's recall triggers
LAP_CSV = "outputs/leakage_lap_v1.csv"

# Two rubrics with incompatible scales, so each writes its own CSV — resuming into a
# file scored on the other scale would silently mix 1-10 and 0-5 values.
RUBRICS = {
    # original: single overall score, ICLR 1-10, no few-shot examples
    "simple": ("review/score_system", "score", "", ["score"]),
    # calibrated: 5 float dimensions on 0-5 with anti-bias warnings and 5 few-shot
    # examples fitted to normalized ICLR ground truth; `rating` is the overall score
    "calibrated": ("review/iclr_review_calibrated", "rating", "_calibrated",
                   ["rating", "confidence", "correctness",
                    "technical_novelty_and_significance",
                    "empirical_novelty_and_significance"]),
}
RUBRIC = "calibrated"                     # set by --rubric
OUT_CSV = OUT_REPORT = OUT_TRACES = None  # set by set_rubric()


def set_rubric(name):
    global RUBRIC, OUT_CSV, OUT_REPORT, OUT_TRACES
    RUBRIC = name
    suffix = RUBRICS[name][2]
    OUT_CSV = f"outputs/leakage_masked_rereview{suffix}.csv"
    OUT_REPORT = f"outputs/leakage_masked_report{suffix}.md"
    OUT_TRACES = f"outputs/leakage_masked_traces{suffix}.jsonl"


def csv_header():
    """paper metadata, then every rubric field for each arm, then the paraphrase size."""
    fields = RUBRICS[RUBRIC][3]
    cols = ["paper_id", "year", "lap", "openalex_citations",
            "score_original", "score_masked"]           # primary score, kept for the report
    cols += [f"{arm}_{f}" for arm in ("orig", "mask") for f in fields]
    return ",".join(cols + ["masked_abstract_chars"]) + "\n"


set_rubric(RUBRIC)


def _sha(s):
    return hashlib.sha1(str(s).encode()).hexdigest()[:12]


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


def _score(client, body, year):
    """Score one body. Returns (parsed_json, trace) — the trace holds everything sent
    and everything returned, so no field the model produced is discarded."""
    prompt_name, key, _, fields = RUBRICS[RUBRIC]
    system = load(prompt_name, year=year)
    # Gemma-4-31B-it is a thinking model: budget for ~1750 thinking tokens + JSON
    raw = _chat(client, SCORE_MODEL,
                [{"role": "system", "content": system},
                 {"role": "user", "content": body}],
                max_tokens=3000)
    # the calibrated rubric's few-shot examples contain JSON objects, so match the LAST
    # brace-delimited block rather than the first
    m = re.findall(r"\{[^{}]*\}", raw or "", re.DOTALL)
    parsed = json.loads(m[-1]) if m else {}
    trace = {"model": SCORE_MODEL, "rubric": RUBRIC, "prompt_name": prompt_name,
             "system_prompt": system, "system_prompt_sha1": _sha(system),
             "user_prompt": body, "user_prompt_sha1": _sha(body),
             "max_tokens": 3000, "temperature": 0,
             "completion": raw, "completion_chars": len(raw or ""),
             "parsed": parsed, "json_blocks_found": len(m)}
    if not m:
        raise ValueError(f"no JSON in score response: {(raw or '')[:200]!r}")
    if key not in parsed:
        raise ValueError(f"score response missing {key!r}: {parsed}")
    return parsed, trace


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
    primary, fields = RUBRICS[RUBRIC][1], RUBRICS[RUBRIC][3]
    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    lock = threading.Lock()
    counter = [0]

    with open(OUT_CSV, "a") as fout, open(OUT_TRACES, "a") as ftrace:
        if write_header:
            fout.write(csv_header())

        def emit(paper_id, arm, trace):
            """One JSONL line per API call — every prompt and every completion kept
            whole. The CSV is the numeric summary; this file is the evidence."""
            ftrace.write(json.dumps(
                {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "paper_id": paper_id, "arm": arm, **trace}, ensure_ascii=False) + "\n")
            ftrace.flush()

        def run_one(row):
            client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
            traces = []
            try:
                para_prompt = load("review/paraphrase_abstract", abstract=row.abstract)
                masked_abs = _chat(client, PARAPHRASE_MODEL,
                                   [{"role": "user", "content": para_prompt}],
                                   max_tokens=1024)
                traces.append(("paraphrase", {
                    "model": PARAPHRASE_MODEL, "rubric": RUBRIC,
                    "prompt_name": "review/paraphrase_abstract",
                    "system_prompt": None, "system_prompt_sha1": None,
                    "user_prompt": para_prompt, "user_prompt_sha1": _sha(para_prompt),
                    "max_tokens": 1024, "temperature": 0,
                    "completion": masked_abs, "completion_chars": len(masked_abs or ""),
                    "parsed": {}, "json_blocks_found": 0,
                    "title": row.title, "original_abstract": row.abstract}))
                if len(masked_abs) < 200:
                    raise ValueError(f"paraphrase suspiciously short ({len(masked_abs)} chars)")
                p_orig, t_orig = _score(client, load("review/title_abstract_body",
                                                    title=row.title,
                                                    abstract=row.abstract), row.year)
                traces.append(("original", t_orig))
                p_mask, t_mask = _score(client, load("review/abstract_only_body",
                                                    abstract=masked_abs), row.year)
                traces.append(("masked", t_mask))
            except Exception as e:
                # a failed paper still gets its traces written — a refusal or an
                # unparseable reply is data, not noise
                with lock:
                    for arm, t in traces:
                        emit(row.paper_id, arm, t)
                    emit(row.paper_id, "error", {"error": f"{type(e).__name__}: {e}"})
                print(f"  SKIP {row.paper_id}: {e}")
                return
            s_orig, s_mask = float(p_orig[primary]), float(p_mask[primary])
            vals = [str(p.get(f, "")) for p in (p_orig, p_mask) for f in fields]
            with lock:
                for arm, t in traces:
                    emit(row.paper_id, arm, t)
                fout.write(f"{row.paper_id},{row.year},{row.lap:.6f},"
                           f"{row.openalex_citations},{s_orig},{s_mask},"
                           + ",".join(vals) + f",{len(masked_abs)}\n")
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

Within-paper design, N = {n}. Rubric: `{RUBRIC}` (`prompts/{RUBRICS[RUBRIC][0]}.txt`).
Each paper scored twice on that same rubric:
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
    parser.add_argument("--rubric", choices=sorted(RUBRICS), default="calibrated",
                        help="calibrated = 5-dimension 0-5 few-shot ICLR rubric (default); "
                             "simple = original single 1-10 score")
    args = parser.parse_args()
    set_rubric(args.rubric)
    print(f"Rubric: {args.rubric} ({RUBRICS[args.rubric][0]})\n"
          f"  scores -> {OUT_CSV}\n  traces -> {OUT_TRACES}")

    if not os.path.exists(LAP_CSV):
        sys.exit(f"ERROR: {LAP_CSV} not found — run leakage_lap_v1.py first.")

    if not args.report_only:
        sample = build_lap_sample(args.n, smoke=args.smoke)
        print(f"Sample: {len(sample)} papers "
              f"(high-LAP: {(sample['lap'] >= 0.5).sum()}, low-LAP: {(sample['lap'] < 0.5).sum()})")
        run(sample, workers=args.workers)

    if args.smoke:
        print(f"\nSmoke done — inspect {OUT_CSV}")
    elif os.path.exists(OUT_CSV):
        run_report()
