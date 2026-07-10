"""
LAP (Lookahead Propensity) leakage test — adapted from Gao, Jiang, Yan (2026).

For each paper: send a title-only recall query (no abstract) to the same Gemma
model used in our committee. Gemma-4-31B-it is a thinking model, so we let it
think (~1750 tokens) and then read the final one-word answer plus the logprobs
at that answer position.

LAP = P_accept + P_reject  (probability model commits to a direction)
U-D = P_accept - P_reject  (directional signal)

Sampling: 300 papers stratified by year × citation quartile × decision.
Full-corpus run is ~15h; 300-paper sample takes ~90 min.

Two regressions after all LAPs computed:
  1. Validation: log(1+citations) ~ (U-D)
  2. Detection:  log(1+citations) ~ committee_rating + LAP + LAP*committee_rating
     β₃ > 0 → memorization flows into forecast

Outputs:
  outputs/leakage_lap_v1.csv    — incremental, one row per API call
  outputs/leakage_lap_report.md — regression results

Run: python src/leakage_lap_v1.py [--smoke] [--full] [--report-only]
"""
import os
import re
import sys
import math
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

MODEL = "google/gemma-4-31B-it"
OUT_CSV = "outputs/leakage_lap_v1.csv"
OUT_REPORT = "outputs/leakage_lap_report.md"
SAMPLE_N = 300       # default; --full uses all eligible papers
MAX_TOKENS = 2000    # enough for thinking (~1750) + answer (1)

ACCEPT_TOKENS = {"accepted", "Accepted", "ACCEPTED", "accept", "Accept"}
REJECT_TOKENS = {"rejected", "Rejected", "REJECTED", "reject", "Reject"}
UNKNOWN_TOKENS = {"unknown", "Unknown", "UNKNOWN"}

def recall_prompt(title, year):
    return (
        f'Title: "{title}"\n'
        f"Year: ICLR {year}\n\n"
        f"Based only on your recall — not on your assessment of quality — "
        f"was this paper accepted or rejected at ICLR {year}? "
        f"Respond with exactly one word: accepted, rejected, or unknown."
    )


def parse_answer(text):
    """Return ('accepted'|'rejected'|'unknown', p_acc, p_rej, p_unk) from model output."""
    t = text.strip().lower().rstrip(".,!?")
    if t in {s.lower() for s in ACCEPT_TOKENS}:
        return "accepted", 1.0, 0.0, 0.0
    if t in {s.lower() for s in REJECT_TOKENS}:
        return "rejected", 0.0, 1.0, 0.0
    return "unknown", 0.0, 0.0, 1.0


def extract_answer_logprobs(lp_content):
    """
    Find the pre-commitment answer position in the thinking chain.

    The thinking model reasons toward an answer. We find the FIRST position
    where target tokens appear in the TOP-5 logprobs — this is where the
    model's probability mass reflects raw recall before it commits.
    This mirrors the Gao et al. position-0 read for non-thinking models.
    """
    if not lp_content:
        return None
    ALL_TARGETS = {s.lower() for s in ACCEPT_TOKENS | REJECT_TOKENS | UNKNOWN_TOKENS}
    for entry in lp_content:
        top5 = [t.token.strip().lower().rstrip(".,!?") for t in entry.top_logprobs[:5]]
        if any(t in ALL_TARGETS for t in top5):
            p = {"accept": 0.0, "reject": 0.0, "unknown": 0.0}
            for top in entry.top_logprobs:
                t = top.token.strip().lower().rstrip(".,!?")
                prob = math.exp(top.logprob)
                if t in {s.lower() for s in ACCEPT_TOKENS}:
                    p["accept"] += prob
                elif t in {s.lower() for s in REJECT_TOKENS}:
                    p["reject"] += prob
                elif t in {s.lower() for s in UNKNOWN_TOKENS}:
                    p["unknown"] += prob
            return p["accept"], p["reject"], p["unknown"]
    return None


def build_sample(df, n, seed=42):
    """Stratified sample: year × citation_quartile (4 bins) × decision (accept/reject)."""
    rng = np.random.default_rng(seed)
    sub = df[df["openalex_citations"].notna() & df["citation_pct_rank"].notna()].copy()
    sub["cit_q"] = pd.qcut(sub["citation_pct_rank"], q=4, labels=False)
    sub["is_accept"] = sub["decision"].str.startswith("Accept").astype(int)
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
    est_min = len(todo) * 20 / 60 / workers
    print(f"Estimated time: ~{est_min:.0f} min")

    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    lock = threading.Lock()
    counter = [0]

    with open(OUT_CSV, "a") as fout:
        if write_header:
            fout.write("paper_id,year,decision,answer,p_accept,p_reject,p_unknown,lap,ud\n")

        def fetch_one(row):
            # ponytail: each thread gets its own client (openai client is not thread-safe)
            client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
            prompt = recall_prompt(row.title, row.year)
            resp = None
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=MAX_TOKENS,
                        logprobs=True,
                        top_logprobs=20,
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        with lock:
                            fout.write(f"{row.paper_id},{row.year},{row.decision},ERROR,,,,,\n")
                            fout.flush()
                        print(f"  SKIP {row.paper_id}: {e}")
                        return
                    import time; time.sleep(5)

            if resp is None:
                return

            content = (resp.choices[0].message.content or "").strip()
            answer, p_acc, p_rej, p_unk = parse_answer(content)
            lap = p_acc + p_rej
            ud = p_acc - p_rej
            dec_safe = str(row.decision).replace(",", " ")

            with lock:
                fout.write(f"{row.paper_id},{row.year},{dec_safe},{answer},"
                           f"{p_acc:.6f},{p_rej:.6f},{p_unk:.6f},{lap:.6f},{ud:.6f}\n")
                fout.flush()
                counter[0] += 1
                print(f"  {counter[0]}/{len(todo)}  {row.paper_id}  answer={answer}"
                      f"  lap={lap:.3f} ud={ud:+.3f}"
                      f"  (tokens={resp.usage.completion_tokens if resp.usage else '?'})")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_one, row) for row in todo.itertuples()]
            for f in as_completed(futures):
                f.result()  # re-raises exceptions


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
    # OLS via numpy (no scipy for multivariate)
    cr = merged["committee_rating"].values
    lap = merged["lap"].values
    X = np.column_stack([np.ones(n), cr, lap, lap * cr])
    try:
        beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        # std errors via residuals
        y_hat = X @ beta
        resid = y - y_hat
        sigma2 = np.dot(resid, resid) / (n - 4)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        t_stats = beta / se
        p_vals = [2 * (1 - stats.t.cdf(abs(t), df=n-4)) for t in t_stats]
        beta0, beta1, beta2, beta3 = beta
        se0, se1, se2, se3 = se
        p0, p1, p2, p3 = p_vals
    except Exception as e:
        beta0 = beta1 = beta2 = beta3 = float("nan")
        se0 = se1 = se2 = se3 = float("nan")
        p0 = p1 = p2 = p3 = float("nan")

    # ── LAP distribution summary ────────────────────────────────────────────────
    lap_mean = merged["lap"].mean()
    lap_hi = (merged["lap"] >= 0.5).mean()
    lap_sat = (merged["lap"] >= 0.95).mean()
    recall_acc = (merged["lap"] > 0.05).mean()

    verdict = "CONTAMINATION DETECTED" if (beta3 > 0 and p3 < 0.05) else (
        "INCONCLUSIVE" if p3 >= 0.05 else "NO CONTAMINATION SIGNAL"
    )

    report = f"""# LAP Leakage Test — {MODEL}

## Design (Gao, Jiang, Yan 2026 — adapted)

Date-only recall query: title + year, NO abstract. Model answers: accepted/rejected/unknown.
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

## Verdict: **{verdict}**

{"- β₃ = " + f"{beta3:.4f}" + " (p=" + f"{p3:.4g}" + ") — the interaction is positive and significant. committee_rating predicts citations better on papers the model has memorized. Genuine quality signal is contaminated." if (beta3 > 0 and p3 < 0.05) else "- β₃ = " + f"{beta3:.4f}" + " (p=" + f"{p3:.4g}" + ") — no significant amplification of forecast accuracy on high-LAP papers."}
"""

    with open(OUT_REPORT, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"LAP mean={lap_mean:.3f}  hi-LAP={lap_hi:.1%}  saturated={lap_sat:.1%}")
    print(f"Validation slope (U-D): {slope_ud:.4f}  p={p_ud:.4g}")
    print(f"Detection β₃ (interaction): {beta3:.4f}  p={p3:.4g}")
    print(f"\nReport: {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="5 papers only")
    parser.add_argument("--full", action="store_true", help="all eligible papers (~4500, ~15h)")
    parser.add_argument("--report-only", action="store_true", help="skip API calls, run report")
    parser.add_argument("--n", type=int, default=SAMPLE_N, help=f"sample size (default {SAMPLE_N})")
    parser.add_argument("--workers", type=int, default=10, help="parallel API workers (default 10)")
    args = parser.parse_args()

    df = pd.read_csv("outputs/eval_table.csv")
    df = df[df["committee_rating"].notna() & df["title"].notna()].reset_index(drop=True)
    print(f"Eligible papers: {len(df)}")

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
