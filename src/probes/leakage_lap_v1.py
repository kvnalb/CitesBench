"""
LAP (Lookahead Propensity) leakage test — adapted from Gao, Jiang, Yan (2026).

For each paper: send a title-only recall query (no abstract) to the same Gemma
model used in our committee. Gemma-4-31B-it is a thinking model; we let it
think, take the final one-word answer, and read soft probabilities from the
logprobs at the answer position (last position where a target token appears
in the top-5 — first occurrence is unreliable, the thinking chain echoes
'accepted'/'rejected' from the prompt).

LAP = P_accept + P_reject  (probability model commits to a direction)
U-D = P_accept - P_reject  (directional signal)

Sampling: 300 papers stratified by year × citation quartile × decision.

Regressions after all LAPs computed:
  1. Validation: log(1+citations) ~ (U-D)
  2. Detection:  log(1+citations) ~ committee_rating + LAP + LAP*committee_rating
     β₃ > 0 → memorization flows into forecast
  3. Decomposition: residualize committee_rating on human mean_rating, then
     log(1+citations) ~ mean_rating + resid + LAP + resid*LAP
     resid coefficient = genuine foresight on non-memorized (LAP=0) papers;
     resid*LAP = contaminated share.

Outputs:
  outputs/leakage_lap_v1.csv    — incremental, one row per API call
  outputs/leakage_lap_report.md — regression results

Run: python src/probes/leakage_lap_v1.py [--smoke] [--full] [--report-only]
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import load

load_dotenv()
os.makedirs("outputs", exist_ok=True)

MODEL = "google/gemma-4-31B-it"        # overridden by --model
EVAL_TABLE = "outputs/eval_table.csv"  # overridden by --eval-table
OUT_CSV = "outputs/leakage_lap_v1.csv"
OUT_REPORT = "outputs/leakage_lap_report.md"
OUT_TRACES = "outputs/leakage_lap_traces.jsonl"


def _slug(m):
    return m.split("/")[-1].replace(".", "-")
SAMPLE_N = 300       # default; --full uses all eligible papers
MAX_TOKENS = 2000    # enough for thinking (~1750) + answer (1)

ACCEPT_TOKENS = {"accepted", "accept"}
REJECT_TOKENS = {"rejected", "reject"}
UNKNOWN_TOKENS = {"unknown"}


def recall_prompt(title, year):
    return load("recall/lap_oneword", title=title, year=year)


def parse_answer(text, pos_set=ACCEPT_TOKENS, neg_set=REJECT_TOKENS,
                 labels=("accepted", "rejected", "unknown")):
    """Hard-parse the final one-word answer. Fallback when logprobs unavailable."""
    t = text.strip().lower().rstrip(".,!?")
    if t in pos_set:
        return labels[0], 1.0, 0.0, 0.0
    if t in neg_set:
        return labels[1], 0.0, 1.0, 0.0
    return labels[2], 0.0, 0.0, 1.0


def extract_answer_logprobs(lp_content, pos_set=ACCEPT_TOKENS,
                            neg_set=REJECT_TOKENS, unk_set=UNKNOWN_TOKENS):
    """
    Soft probabilities at the answer position: the LAST position where a
    target token appears in the top-5 (the final one-word answer).
    """
    if not lp_content:
        return None
    all_targets = pos_set | neg_set | unk_set
    for entry in reversed(lp_content):
        top5 = [t.token.strip().lower().rstrip(".,!?") for t in entry.top_logprobs[:5]]
        if any(t in all_targets for t in top5):
            p = [0.0, 0.0, 0.0]
            for top in entry.top_logprobs:
                t = top.token.strip().lower().rstrip(".,!?")
                prob = math.exp(top.logprob)
                if t in pos_set:
                    p[0] += prob
                elif t in neg_set:
                    p[1] += prob
                elif t in unk_set:
                    p[2] += prob
            return tuple(p)
    return None


def probe_one(client, prompt, pos_set=ACCEPT_TOKENS, neg_set=REJECT_TOKENS,
              unk_set=UNKNOWN_TOKENS, labels=("accepted", "rejected", "unknown")):
    """
    One recall probe against MODEL: greedy decode, logprob read at answer
    position, text-parse fallback. Retries 3x then raises.
    Returns (label, p_pos, p_neg, p_unk, completion_tokens, trace).
    trace = full completion reconstructed from logprob tokens (includes the
    thinking channel, which Together strips from message.content) — callers
    persist it to a JSONL sidecar for forensic audit of recall vs judgment.
    """
    resp = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
            )
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)

    content = (resp.choices[0].message.content or "").strip()
    label, p_pos, p_neg, p_unk = parse_answer(content, pos_set, neg_set, labels)
    lp = getattr(resp.choices[0], "logprobs", None)
    probs = extract_answer_logprobs(lp.content if lp else None, pos_set, neg_set, unk_set)
    if probs is not None:
        p_pos, p_neg, p_unk = probs
    trace = "".join(e.token for e in lp.content) if (lp and lp.content) else None
    return label, p_pos, p_neg, p_unk, (resp.usage.completion_tokens if resp.usage else None), trace


def build_sample(df, n, seed=42):
    """Stratified sample: year × citation_quartile (4 bins) × decision (accept/reject)."""
    rng = np.random.default_rng(seed)
    cite_col = "openalex_citations" if "openalex_citations" in df else "citations"
    sub = df[df[cite_col].notna() & df["citation_pct_rank"].notna()].copy()
    sub["cit_q"] = pd.qcut(sub["citation_pct_rank"], q=4, labels=False)
    sub["is_accept"] = sub["decision"].fillna("").str.startswith("Accept").astype(int)
    cells = sub.groupby(["year", "cit_q", "is_accept"])
    per_cell = max(1, n // len(cells))
    rows = []
    for _, g in cells:
        idx = rng.choice(len(g), size=min(per_cell, len(g)), replace=False)
        rows.append(g.iloc[idx])
    result = pd.concat(rows).drop_duplicates("paper_id")
    # Top up to n if short
    remainder = sub[~sub["paper_id"].isin(result["paper_id"])]
    if len(result) < n and len(remainder):
        extra_idx = rng.choice(len(remainder), size=min(n - len(result), len(remainder)), replace=False)
        result = pd.concat([result, remainder.iloc[extra_idx]])
    return result.head(n).reset_index(drop=True)


def run_laps(df, smoke=False, workers=10):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
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

    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    lock = threading.Lock()
    counter = [0]

    with open(OUT_CSV, "a") as fout, open(OUT_TRACES, "a") as ftrace:
        if write_header:
            fout.write("paper_id,year,decision,answer,p_accept,p_reject,p_unknown,lap,ud\n")

        def fetch_one(row):
            # ponytail: each thread gets its own client (openai client is not thread-safe)
            client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
            try:
                answer, p_acc, p_rej, p_unk, ntok, trace = probe_one(
                    client, recall_prompt(row.title, row.year))
            except Exception as e:
                with lock:
                    fout.write(f"{row.paper_id},{row.year},{row.decision},ERROR,,,,,\n")
                    fout.flush()
                print(f"  SKIP {row.paper_id}: {e}")
                return

            lap = p_acc + p_rej
            ud = p_acc - p_rej
            dec_safe = str(row.decision).replace(",", " ")
            with lock:
                fout.write(f"{row.paper_id},{row.year},{dec_safe},{answer},"
                           f"{p_acc:.6f},{p_rej:.6f},{p_unk:.6f},{lap:.6f},{ud:.6f}\n")
                fout.flush()
                if trace:
                    ftrace.write(json.dumps({"paper_id": row.paper_id, "probe": "lap",
                                             "answer": answer, "trace": trace}) + "\n")
                    ftrace.flush()
                counter[0] += 1
                print(f"  {counter[0]}/{len(todo)}  {row.paper_id}  answer={answer}"
                      f"  lap={lap:.3f} ud={ud:+.3f}  (tokens={ntok})")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_one, row) for row in todo.itertuples()]
            for f in as_completed(futures):
                f.result()  # re-raises exceptions


def _ols(X, y):
    """OLS with classical SEs. Returns (beta, se, p_values)."""
    from scipy import stats
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = np.dot(resid, resid) / (n - k)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sigma2)
    p = [2 * (1 - stats.t.cdf(abs(b / s), df=n - k)) for b, s in zip(beta, se)]
    return beta, se, p


def run_report():
    from scipy import stats

    lap_df = pd.read_csv(OUT_CSV)
    lap_df = lap_df[lap_df["lap"].notna()]

    eval_df = pd.read_csv("outputs/eval_table.csv")
    merged = eval_df.merge(lap_df[["paper_id", "p_accept", "p_reject", "p_unknown", "lap", "ud"]],
                           on="paper_id", how="inner")
    merged = merged[merged["openalex_citations"].notna() & merged["committee_rating"].notna()]

    merged["log_cites"] = np.log1p(merged["openalex_citations"])
    n = len(merged)

    # ── Validation regression: log_cites ~ U-D ─────────────────────────────────
    x_ud = merged["ud"].values
    y = merged["log_cites"].values
    slope_ud, intercept_ud, r_ud, p_ud, _ = stats.linregress(x_ud, y)

    # ── Detection regression: log_cites ~ committee + LAP + LAP*committee ───────
    cr = merged["committee_rating"].values
    lap = merged["lap"].values
    X = np.column_stack([np.ones(n), cr, lap, lap * cr])
    try:
        beta, se, pv = _ols(X, y)
        beta0, beta1, beta2, beta3 = beta
        se0, se1, se2, se3 = se
        p0, p1, p2, p3 = pv
    except Exception:
        beta0 = beta1 = beta2 = beta3 = float("nan")
        se0 = se1 = se2 = se3 = float("nan")
        p0 = p1 = p2 = p3 = float("nan")

    # ── Decomposition: residualized foresight, LAP-stratified ──────────────────
    dec = merged[merged["mean_rating"].notna()].copy()
    nd = len(dec)
    try:
        # residualize committee_rating on human mean_rating
        rb = np.polyfit(dec["mean_rating"], dec["committee_rating"], 1)
        dec["resid"] = dec["committee_rating"] - np.polyval(rb, dec["mean_rating"])
        Xd = np.column_stack([np.ones(nd), dec["mean_rating"], dec["resid"],
                              dec["lap"], dec["resid"] * dec["lap"]])
        yd = dec["log_cites"].values
        dbeta, dse, dp = _ols(Xd, yd)
    except Exception:
        dbeta = dse = [float("nan")] * 5
        dp = [float("nan")] * 5

    # ── LAP distribution summary ────────────────────────────────────────────────
    lap_mean = merged["lap"].mean()
    lap_hi = (merged["lap"] >= 0.5).mean()
    lap_sat = (merged["lap"] >= 0.95).mean()
    recall_acc = (merged["lap"] > 0.05).mean()

    # recall accuracy among committed answers
    committed = merged[merged["lap"] >= 0.5].copy()
    if len(committed):
        committed["true_accept"] = committed["decision"].str.startswith("Accept")
        committed["said_accept"] = committed["ud"] > 0
        acc = (committed["true_accept"] == committed["said_accept"]).mean()
    else:
        acc = float("nan")

    verdict = "CONTAMINATION DETECTED" if (beta3 > 0 and p3 < 0.05) else (
        "INCONCLUSIVE" if p3 >= 0.05 else "NO CONTAMINATION SIGNAL"
    )

    report = f"""# LAP Leakage Test — {MODEL}

## Design (Gao, Jiang, Yan 2026 — adapted)

Decision-only recall query: title + year, NO abstract. Model answers: accepted/rejected/unknown.
LAP = P(accepted) + P(rejected). High LAP = model memorized the outcome.

**N = {n}** papers with both committee_rating and openalex_citations.

---

## LAP Distribution

| Statistic | Value |
|---|---|
| Mean LAP | {lap_mean:.3f} |
| Fraction LAP ≥ 0.50 | {lap_hi:.1%} |
| Fraction LAP ≥ 0.95 (saturated) | {lap_sat:.1%} |
| Fraction with any recall (LAP > 0.05) | {recall_acc:.1%} |
| Recall accuracy on committed answers (LAP ≥ 0.5) | {acc:.1%} |

---

## Regression 1 — Validation (existence of recall)

**Y = log(1+citations) ~ (U-D)**

Does the directional recall signal (title-only, no content) predict actual citations?
Any predictive power here can only come from training-time memorization.

| Coefficient | Estimate | p-value |
|---|---|---|
| Slope on (U-D) | {slope_ud:.4f} | {p_ud:.4g} |

{"✅ Recall channel is directionally informative (p < 0.05) — model has absorbed outcome-relevant content." if p_ud < 0.05 else "◻ Directional recall not significantly predictive (p ≥ 0.05)."}

---

## Regression 2 — Detection (does memorization flow into forecast?)

**Y = log(1+citations) ~ β₁·committee_rating + β₂·LAP + β₃·(LAP × committee_rating)**

β₃ > 0 with p < 0.05 = contamination confirmed: committee_rating is more accurate
precisely when the model has memorized the paper's outcome.

| Term | β | SE | p-value |
|---|---|---|---|
| Intercept | {beta0:.4f} | {se0:.4f} | {p0:.4g} |
| committee_rating (β₁) | {beta1:.4f} | {se1:.4f} | {p1:.4g} |
| LAP (β₂) | {beta2:.4f} | {se2:.4f} | {p2:.4g} |
| LAP × committee_rating (β₃) | {beta3:.4f} | {se3:.4f} | {p3:.4g} |

---

## Regression 3 — Decomposition (genuine foresight vs contaminated share)

Residualize committee_rating on human mean_rating (what the LLM adds beyond
reviewers), then:

**Y = log(1+citations) ~ mean_rating + resid + LAP + (resid × LAP)**   (N = {nd})

The `resid` coefficient is the LLM's excess predictive power on papers with
NO decision recall (LAP=0) — the defensible "genuine foresight" estimate.
The interaction is the share concentrated on memorized papers — contamination.

| Term | β | SE | p-value |
|---|---|---|---|
| mean_rating | {dbeta[1]:.4f} | {dse[1]:.4f} | {dp[1]:.4g} |
| resid (foresight at LAP=0) | {dbeta[2]:.4f} | {dse[2]:.4f} | {dp[2]:.4g} |
| LAP | {dbeta[3]:.4f} | {dse[3]:.4f} | {dp[3]:.4g} |
| resid × LAP (contamination) | {dbeta[4]:.4f} | {dse[4]:.4f} | {dp[4]:.4g} |

---

## Verdict: **{verdict}**

- β₃ = {beta3:.4f} (p={p3:.4g})
- Foresight at LAP=0: {dbeta[2]:.4f} (p={dp[2]:.4g}); contaminated share: {dbeta[4]:.4f} (p={dp[4]:.4g})

Caveat: LAP measures recall of the accept/reject *decision*, not of *fame*.
See leakage_fame_v1 for the citation-prominence recall probe, and
leakage_controls for probe validity (placebo) checks.
"""

    with open(OUT_REPORT, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"LAP mean={lap_mean:.3f}  hi-LAP={lap_hi:.1%}  saturated={lap_sat:.1%}  recall-acc={acc:.1%}")
    print(f"Validation slope (U-D): {slope_ud:.4f}  p={p_ud:.4g}")
    print(f"Detection β₃ (interaction): {beta3:.4f}  p={p3:.4g}")
    print(f"Foresight at LAP=0: {dbeta[2]:.4f} p={dp[2]:.4g}   contamination: {dbeta[4]:.4f} p={dp[4]:.4g}")
    print(f"\nReport: {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="5 papers only")
    parser.add_argument("--model", default=MODEL,
                        help="Together model id. Use a pre-cutoff model for the "
                             "out-of-sample arm, e.g. meta-llama/Meta-Llama-3-70B-Instruct-Turbo "
                             "(knowledge cutoff Dec 2023)")
    parser.add_argument("--eval-table", default=EVAL_TABLE,
                        help="eval table to probe, e.g. outputs/eval_table_2025.csv")
    parser.add_argument("--tag", default="", help="suffix for the output files")
    parser.add_argument("--full", action="store_true", help="all eligible papers (~4500)")
    parser.add_argument("--report-only", action="store_true", help="skip API calls, run report")
    parser.add_argument("--n", type=int, default=SAMPLE_N, help=f"sample size (default {SAMPLE_N})")
    parser.add_argument("--workers", type=int, default=10, help="parallel API workers (default 10)")
    args = parser.parse_args()

    MODEL = args.model
    if args.tag:
        OUT_CSV = f"outputs/leakage_lap_{args.tag}.csv"
        OUT_REPORT = f"outputs/leakage_lap_report_{args.tag}.md"
        OUT_TRACES = f"outputs/leakage_lap_traces_{args.tag}.jsonl"

    df = pd.read_csv(args.eval_table, low_memory=False)
    # The out-of-sample tables have no committee scores yet; only require them when present.
    if "committee_rating" in df:
        df = df[df["committee_rating"].notna()]
    df = df[df["title"].notna()].reset_index(drop=True)
    print(f"Eligible papers: {len(df)}  ({args.eval_table})")

    if args.full:
        sample = df
    elif args.smoke:
        sample = df.head(5)
    else:
        sample = build_sample(df, args.n)
        print(f"Stratified sample: {len(sample)} papers "
              f"({sample.groupby('year').size().to_dict()})")

    if not args.report_only:
        run_laps(sample, smoke=args.smoke, workers=args.workers)

    if os.path.exists(OUT_CSV) and not args.smoke:
        run_report()
    elif args.smoke:
        print("\nSmoke done — inspect outputs/leakage_lap_v1.csv")
