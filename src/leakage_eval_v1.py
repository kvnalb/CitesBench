"""
Leakage evaluation v1: test whether the LLM shows anomalously good citation
prediction for famous (top-cited) papers — a signal of memorized outcomes.

Sample design (per year, 2018/2019):
  - top_decile: citation_pct_rank >= 0.9   (n=25 per year)
  - mid_decile:  0.4 <= citation_pct_rank <= 0.6 (n=25 per year)

For each paper the LLM receives only the title + abstract and must predict
expected citation count. We then test whether predictions are anomalously
high for top-decile papers relative to actual quality.

Outputs:
  outputs/leakage_eval_v1.csv  — incremental, one row per LLM call
  outputs/leakage_eval_report.md — statistical analysis

Model selection (first match):
  TOGETHER_API_KEY  →  meta-llama/Llama-3.3-70B-Instruct-Turbo  (Together AI)
  ANTHROPIC_API_KEY →  claude-haiku-4-5                          (Anthropic)
"""
import os
import re
import sys
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_PER_CELL = 25
YEARS = [2018, 2019]
OUT_CSV = "outputs/leakage_eval_v1.csv"
OUT_REPORT = "outputs/leakage_eval_report.md"

TOGETHER_MODEL = "google/gemma-4-31B-it"  # same base model as our LLM committee
ANTHROPIC_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are a machine-learning researcher evaluating papers from ICLR 2018-2020. "
    "Given a paper's title and abstract, estimate how many times this paper will be "
    "cited in the five years following publication. Consider: novelty, clarity, "
    "practical impact, and relevance to active research areas. "
    "Respond with JSON only: {\"predicted_citations\": <integer>, \"rationale\": <one sentence>}"
)

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="Process 2 papers only (no writes)")
parser.add_argument("--dry-run", action="store_true", help="Show sampling only; no LLM calls")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build LLM client (lazy — deferred to first call so dry-run/sampling works)
# ---------------------------------------------------------------------------
_client = None
_client_type = None
_model_id = None

def _get_client():
    global _client, _client_type, _model_id
    if _client is not None:
        return _client, _client_type, _model_id

    TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    if TOGETHER_KEY:
        from openai import OpenAI
        _client_type = "together"
        _client = OpenAI(api_key=TOGETHER_KEY, base_url="https://api.together.xyz/v1")
        _model_id = TOGETHER_MODEL
        print(f"Using Together AI: {_model_id}")
    elif ANTHROPIC_KEY:
        import anthropic as _anthropic
        _client_type = "anthropic"
        _client = _anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        _model_id = ANTHROPIC_MODEL
        print(f"Using Anthropic: {_model_id}")
    else:
        # Try bare Anthropic client (OAuth / ant auth login)
        try:
            import anthropic as _anthropic
            _client_type = "anthropic"
            _client = _anthropic.Anthropic()
            _model_id = ANTHROPIC_MODEL
            _client.models.list()  # quick probe
            print(f"Using Anthropic (OAuth): {_model_id}")
        except Exception as e:
            sys.exit(
                "ERROR: No API key found. Set TOGETHER_API_KEY or ANTHROPIC_API_KEY in .env\n"
                f"(Tried bare Anthropic OAuth: {e})"
            )
    return _client, _client_type, _model_id

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
eval_df = pd.read_csv("outputs/eval_table.csv")
con = sqlite3.connect("data/gen_review.db")
abstracts = pd.read_sql("SELECT id, abstract FROM SUBMISSION", con)
con.close()

df = eval_df.merge(abstracts, left_on="paper_id", right_on="id", how="inner")
df = df[df["citation_pct_rank"].notna() & df["openalex_citations"].notna()]

# ---------------------------------------------------------------------------
# Build sample
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

def _sample(year, group):
    sub = df[df["year"] == year].copy()
    if group == "top_decile":
        sub = sub[sub["citation_pct_rank"] >= 0.9]
    else:  # mid_decile
        sub = sub[(sub["citation_pct_rank"] >= 0.4) & (sub["citation_pct_rank"] <= 0.6)]
    idx = rng.choice(len(sub), size=min(N_PER_CELL, len(sub)), replace=False)
    return sub.iloc[idx].copy().assign(group=group)

sample = pd.concat([
    _sample(yr, grp)
    for yr in YEARS
    for grp in ["top_decile", "mid_decile"]
], ignore_index=True)

print(f"Sample: {len(sample)} papers  ({sample.groupby(['year','group']).size().to_dict()})")

if args.smoke:
    sample = sample.head(2)
    print("Smoke mode: processing 2 papers only (no writes)")

# ---------------------------------------------------------------------------
# Resumability: skip already-done paper_ids
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
done_ids: set = set()
if not args.smoke and os.path.exists(OUT_CSV):
    done = pd.read_csv(OUT_CSV)
    done_ids = set(done["paper_id"])
    remaining = len(sample) - len(sample[sample["paper_id"].isin(done_ids)])
    print(f"Resuming — {len(done_ids)} done, {remaining} remaining")

# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------
def _call_llm(title: str, abstract: str) -> dict:
    c, c_type, m_id = _get_client()
    user_msg = f"Title: {title}\n\nAbstract: {abstract}"
    for attempt in range(3):
        try:
            if c_type == "together":
                resp = c.chat.completions.create(
                    model=m_id,
                    max_tokens=256,
                    timeout=30,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
            else:  # anthropic
                resp = c.messages.create(
                    model=m_id,
                    max_tokens=256,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text.strip()

            # Extract JSON (tolerate markdown fences)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON in response: {raw[:200]!r}")
            parsed = json.loads(m.group())
            pred = int(parsed["predicted_citations"])
            rationale = str(parsed.get("rationale", ""))
            return {"predicted_citations": pred, "rationale": rationale, "raw": raw}
        except Exception as e:
            if attempt == 2:
                raise
            import time; time.sleep(3 * (attempt + 1))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
CSV_COLS = [
    "paper_id", "year", "group", "actual_citations", "citation_pct_rank",
    "predicted_citations", "rationale", "model_id",
]

if args.dry_run:
    print("\nDry run — sample preview:")
    print(sample[["paper_id", "year", "group", "openalex_citations", "citation_pct_rank", "title"]].to_string())
    sys.exit(0)

for i, row in sample.iterrows():
    pid = row["paper_id"]
    if pid in done_ids:
        continue

    result = _call_llm(row["title"], row["abstract"])
    record = {
        "paper_id": pid,
        "year": row["year"],
        "group": row["group"],
        "actual_citations": row["openalex_citations"],
        "citation_pct_rank": row["citation_pct_rank"],
        "predicted_citations": result["predicted_citations"],
        "rationale": result["rationale"],
        "model_id": _model_id,
    }

    if args.smoke:
        print(record)
    else:
        write_header = not os.path.exists(OUT_CSV)
        pd.DataFrame([record]).to_csv(
            OUT_CSV, mode="a", header=write_header, index=False
        )
        done_ids.add(pid)
        print(f"  {len(done_ids)}/{len(sample)}  paper={pid}  pred={result['predicted_citations']}  actual={int(row['openalex_citations'])}")

if args.smoke:
    print("Smoke done.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
from scipy import stats

results = pd.read_csv(OUT_CSV)
results = results[results["paper_id"].isin(sample["paper_id"])]

top = results[results["group"] == "top_decile"]
mid = results[results["group"] == "mid_decile"]

# Spearman correlation: predicted vs actual (full sample + per group)
sp_all = stats.spearmanr(results["actual_citations"], results["predicted_citations"])
sp_top = stats.spearmanr(top["actual_citations"], top["predicted_citations"])
sp_mid = stats.spearmanr(mid["actual_citations"], mid["predicted_citations"])

# Mann-Whitney U: are LLM predictions higher for top_decile?
mw = stats.mannwhitneyu(top["predicted_citations"], mid["predicted_citations"], alternative="greater")

# Prediction bias: log ratio of predicted means vs actual means
actual_ratio = top["actual_citations"].mean() / mid["actual_citations"].mean()
pred_ratio   = top["predicted_citations"].mean() / mid["predicted_citations"].mean()
bias_factor  = pred_ratio / actual_ratio  # >1 means LLM over-inflates top_decile

# Per-year breakdown
year_rows = []
for yr in YEARS:
    yr_df = results[results["year"] == yr]
    t = yr_df[yr_df["group"] == "top_decile"]
    m_ = yr_df[yr_df["group"] == "mid_decile"]
    if len(t) >= 3 and len(m_) >= 3:
        mw_yr = stats.mannwhitneyu(t["predicted_citations"], m_["predicted_citations"], alternative="greater")
        year_rows.append({
            "year": yr,
            "top_pred_mean": t["predicted_citations"].mean(),
            "mid_pred_mean": m_["predicted_citations"].mean(),
            "top_actual_mean": t["actual_citations"].mean(),
            "mid_actual_mean": m_["actual_citations"].mean(),
            "mw_p": mw_yr.pvalue,
        })
year_df = pd.DataFrame(year_rows)

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
verdict = "POSSIBLE LEAKAGE" if bias_factor > 1.5 and mw.pvalue < 0.05 else "NO STRONG LEAKAGE SIGNAL"

report = f"""# Leakage Evaluation v1 — {_model_id}

## Design

- Years: {YEARS}
- top_decile: citation_pct_rank ≥ 0.90  (N = {len(top)})
- mid_decile:  0.40 ≤ citation_pct_rank ≤ 0.60  (N = {len(mid)})
- Prompt: estimate 5-year citation count from title + abstract only

## Spearman ρ (predicted vs actual citations)

| Group | ρ | p-value |
|---|---|---|
| All | {sp_all.statistic:.3f} | {sp_all.pvalue:.3g} |
| top_decile | {sp_top.statistic:.3f} | {sp_top.pvalue:.3g} |
| mid_decile | {sp_mid.statistic:.3f} | {sp_mid.pvalue:.3g} |

Higher ρ in top_decile than mid_decile is consistent with memorization.

## Mann-Whitney U (predicted citations: top_decile > mid_decile?)

- U = {mw.statistic:.1f}, p = {mw.pvalue:.4g}
- Interpretation: {"significant (p<0.05)" if mw.pvalue < 0.05 else "not significant"}

## Prediction Bias Factor

| | top_decile mean | mid_decile mean | ratio |
|---|---|---|---|
| Actual citations | {top["actual_citations"].mean():.1f} | {mid["actual_citations"].mean():.1f} | {actual_ratio:.2f}× |
| LLM predicted | {top["predicted_citations"].mean():.1f} | {mid["predicted_citations"].mean():.1f} | {pred_ratio:.2f}× |

Bias factor (pred_ratio / actual_ratio) = **{bias_factor:.2f}×**
A bias factor > 1.0 means the LLM over-inflates top-decile predictions
relative to what actual citation quality alone would predict.

## Per-Year Breakdown

{year_df.to_markdown(index=False, floatfmt=".1f") if len(year_df) else "N/A"}

## Verdict

**{verdict}**

{"- bias_factor > 1.5 suggests the model predicts disproportionately high citations for top-decile papers" if bias_factor > 1.5 else "- bias_factor near 1.0 suggests model predictions scale proportionally across citation strata"}
{"- Mann-Whitney p < 0.05 confirms top_decile predictions are statistically higher" if mw.pvalue < 0.05 else "- Mann-Whitney p ≥ 0.05; no statistically significant elevation of top-decile predictions"}
{"- Spearman ρ notably higher for top_decile than mid_decile, consistent with memorization of high-citation outcomes" if sp_top.statistic > sp_mid.statistic + 0.1 else "- Spearman ρ similar between groups; no differential accuracy signal"}
"""

with open(OUT_REPORT, "w") as f:
    f.write(report)

print(f"\n{'='*60}")
print(f"VERDICT: {verdict}")
print(f"Bias factor: {bias_factor:.2f}×  (1.0 = no bias)")
print(f"Mann-Whitney p: {mw.pvalue:.4g}")
print(f"Spearman ρ — all: {sp_all.statistic:.3f}, top: {sp_top.statistic:.3f}, mid: {sp_mid.statistic:.3f}")
print(f"\nReport written to {OUT_REPORT}")
