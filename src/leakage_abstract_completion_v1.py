"""
Abstract-completion probe — verbatim text memorization test (training-data
extraction, Carlini et al. lineage).

LAP/fame probes show the model remembers a paper's *outcomes*; this tests
whether the paper's *text* is in the weights. Prompt: title + year + first
sentence of the abstract → model writes the rest (greedy decode: memorized
text sits at the mode). Score the continuation against the true remainder on
surface-overlap metrics (embeddings deliberately excluded — they measure
topicality, the exact confound to kill):

  - ROUGE-L F1 (token-level longest common subsequence)
  - normalized longest common substring (character-level)
  - 8-gram hit rate: fraction of generated 8-grams appearing verbatim in the
    true remainder. An 8-token verbatim collision is ~impossible by topical
    coincidence; nonzero rate = direct memorization evidence.

Null distribution: each continuation is also scored against K=5 decoy
abstracts from the same field × year. Genre boilerplate ("We propose...
state-of-the-art...") inflates target and decoy scores equally and cancels
in the margin. A paper is "extractable" when its target ROUGE-L beats every
decoy AND it has verbatim 8-gram hits.

Sample: stratified by citation decile (default 30/decile = 300).

Limitation (state in any writeup): Gemma-4-31B-it is instruction-tuned, which
suppresses verbatim regurgitation vs the base model. The test is one-sided —
positive results are strong evidence of memorization; a null does NOT prove
the text isn't in the weights.

Outputs:
  outputs/leakage_abstract_completion_v1.csv     — scores, incremental/resumable
  outputs/leakage_abstract_completion_texts.jsonl — generated continuations (audit)
  outputs/leakage_abstract_completion_report.md  — gradient, convergence, exhibits

Run: python src/leakage_abstract_completion_v1.py [--smoke] [--n-per-decile 30] [--workers 10]
"""
import os
import re
import sys
import json
import time
import sqlite3
import argparse
import threading
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

MODEL = "google/gemma-4-31B-it"   # same model as the committee
OUT_CSV = "outputs/leakage_abstract_completion_v1.csv"
OUT_JSONL = "outputs/leakage_abstract_completion_texts.jsonl"
OUT_REPORT = "outputs/leakage_abstract_completion_report.md"
MAX_TOKENS = 6000                 # writing tasks trigger long thinking; retry escalates to 10k
N_DECOYS = 5
MIN_REMAINDER_CHARS = 200         # skip papers whose abstract remainder is too short to score

PROMPT = (
    "Below are the title, year, and the first sentence of the abstract of a paper "
    "submitted to ICLR {year}. Continue the abstract from exactly where the first "
    "sentence ends. Write only the continuation text — do not repeat the title or "
    "the first sentence, and do not add commentary.\n\n"
    'Title: "{title}"\n'
    "Year: ICLR {year}\n"
    "Abstract (first sentence): {first_sentence}"
)


# ── Text utilities ────────────────────────────────────────────────────────────

def split_first_sentence(abstract):
    """First sentence + remainder. Returns (None, None) if unusable."""
    text = re.sub(r"\s+", " ", str(abstract)).strip()
    # sentence boundary: period/!/? followed by space and uppercase/digit
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    if not parts:
        return None, None
    first = parts[0]
    # too-short first "sentence" is usually an abbreviation artifact — extend
    i = 1
    while len(first) < 40 and i < len(parts):
        first = f"{first} {parts[i]}"
        i += 1
    remainder = " ".join(parts[i:]).strip()
    if len(remainder) < MIN_REMAINDER_CHARS:
        return None, None
    return first, remainder


def _norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l_f1(gen, ref):
    """Token-level LCS F1."""
    a, b = _tokens(gen), _tokens(ref)
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[len(b)]
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def lcs_substring_norm(gen, ref):
    """Longest common substring (chars) / len(gen) — share of generation copied verbatim."""
    a, b = _norm(gen), _norm(ref)
    if not a or not b:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size / len(a)


def eight_gram_hit_rate(gen, ref, n=8):
    a, b = _tokens(gen), _tokens(ref)
    if len(a) < n or len(b) < n:
        return 0.0
    ref_grams = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    gen_grams = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
    return sum(g in ref_grams for g in gen_grams) / len(gen_grams)


# ── Data assembly ─────────────────────────────────────────────────────────────

def load_frame():
    ev = pd.read_csv("outputs/eval_table.csv")
    con = sqlite3.connect("data/gen_review.db")
    abstracts = pd.read_sql("SELECT id AS paper_id, abstract FROM SUBMISSION", con)
    con.close()
    df = ev.merge(abstracts, on="paper_id")
    df = df[df["citation_pct_rank"].notna() & df["title"].notna() & df["abstract"].notna()].copy()
    splits = df["abstract"].map(split_first_sentence)
    df["first_sentence"] = splits.map(lambda t: t[0])
    df["remainder"] = splits.map(lambda t: t[1])
    df = df[df["first_sentence"].notna()].copy()
    df["decile"] = (df["citation_pct_rank"] * 10).clip(0, 9).astype(int)
    return df


def build_sample(df, n_per_decile, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _, g in df.groupby("decile"):
        idx = rng.choice(len(g), size=min(n_per_decile, len(g)), replace=False)
        rows.append(g.iloc[idx])
    return pd.concat(rows).reset_index(drop=True)


def pick_decoys(df, row, k=N_DECOYS, seed=42):
    """K decoy remainders, same field × year (relax to field-only if thin)."""
    pool = df[(df["paper_id"] != row.paper_id) & (df["field"] == row.field)
              & (df["year"] == row.year)]
    if len(pool) < k:
        pool = df[(df["paper_id"] != row.paper_id) & (df["field"] == row.field)]
    if len(pool) < k:
        pool = df[df["paper_id"] != row.paper_id]
    rng = np.random.default_rng(abs(hash(row.paper_id)) % (2**32))
    idx = rng.choice(len(pool), size=k, replace=False)
    return pool.iloc[idx]["remainder"].tolist()


# ── Probe ─────────────────────────────────────────────────────────────────────

def run_probes(df, sample, smoke=False, workers=10):
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        sys.exit("ERROR: TOGETHER_API_KEY not set in .env")

    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV)["paper_id"].unique())
    todo = sample[~sample["paper_id"].isin(done)]
    if smoke:
        todo = todo.head(5)

    print(f"Model: {MODEL}")
    print(f"Already done: {len(done)}, to fetch: {len(todo)}, workers: {workers}")

    from openai import OpenAI
    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    lock = threading.Lock()
    counter = [0]

    cols = ("paper_id,year,decile,citation_pct_rank,gen_chars,ref_chars,"
            "rougeL_target,rougeL_decoy_mean,rougeL_decoy_max,rougeL_margin,"
            "lcs_target,lcs_decoy_mean,"
            "eight_target,eight_decoy_mean,eight_decoy_max,extractable\n")

    with open(OUT_CSV, "a") as fout, open(OUT_JSONL, "a") as ftext:
        if write_header:
            fout.write(cols)

        def fetch_one(row):
            client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
            prompt = PROMPT.format(title=row.title, year=row.year,
                                   first_sentence=row.first_sentence)
            gen = None
            # empty content with finish_reason=length means thinking ate the whole
            # budget — escalate max_tokens rather than retrying the same call
            for attempt, budget in enumerate([MAX_TOKENS, 10000, 10000]):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=budget,
                        temperature=0,
                    )
                    gen = (resp.choices[0].message.content or "").strip()
                    if gen and len(gen) >= 100:
                        break
                except Exception as e:
                    if attempt == 2:
                        print(f"  SKIP {row.paper_id}: {e}")
                        return
                    time.sleep(5)
            if not gen or len(gen) < 100:
                print(f"  SKIP {row.paper_id}: empty/short continuation ({len(gen or '')} chars)")
                return

            ref = row.remainder
            decoys = pick_decoys(df, row)
            rl_t = rouge_l_f1(gen, ref)
            rl_d = [rouge_l_f1(gen, d) for d in decoys]
            lcs_t = lcs_substring_norm(gen, ref)
            lcs_d = [lcs_substring_norm(gen, d) for d in decoys]
            e_t = eight_gram_hit_rate(gen, ref)
            e_d = [eight_gram_hit_rate(gen, d) for d in decoys]
            extractable = int(rl_t > max(rl_d) and e_t > 0)

            with lock:
                fout.write(f"{row.paper_id},{row.year},{row.decile},{row.citation_pct_rank:.4f},"
                           f"{len(gen)},{len(ref)},"
                           f"{rl_t:.4f},{np.mean(rl_d):.4f},{max(rl_d):.4f},{rl_t - np.mean(rl_d):.4f},"
                           f"{lcs_t:.4f},{np.mean(lcs_d):.4f},"
                           f"{e_t:.4f},{np.mean(e_d):.4f},{max(e_d):.4f},{extractable}\n")
                fout.flush()
                ftext.write(json.dumps({"paper_id": row.paper_id, "gen": gen}) + "\n")
                ftext.flush()
                counter[0] += 1
                print(f"  {counter[0]}/{len(todo)}  {row.paper_id}  decile={row.decile}"
                      f"  rougeL={rl_t:.3f} (decoy {np.mean(rl_d):.3f})"
                      f"  8gram={e_t:.3f}  extractable={extractable}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_one, row) for row in todo.itertuples()]
            for f in as_completed(futures):
                f.result()


# ── Report ────────────────────────────────────────────────────────────────────

def run_report():
    from scipy import stats

    res = pd.read_csv(OUT_CSV)
    n = len(res)

    by_dec = res.groupby("decile").agg(
        n=("rougeL_margin", "size"),
        rougeL_margin=("rougeL_margin", "mean"),
        eight_target=("eight_target", "mean"),
        pct_extractable=("extractable", "mean"),
    )

    sp_grad = stats.spearmanr(res["citation_pct_rank"], res["rougeL_margin"])
    sp_8g = stats.spearmanr(res["citation_pct_rank"], res["eight_target"])

    # convergent validity vs LAP / FAME
    conv_lines = []
    for name, path, col in [("LAP (decision recall)", "outputs/leakage_lap_v1.csv", "lap"),
                            ("FAME (fame recall)", "outputs/leakage_fame_v1.csv", "fame")]:
        if os.path.exists(path):
            other = pd.read_csv(path)[["paper_id", col]].dropna()
            m = res.merge(other, on="paper_id")
            if len(m) > 10:
                sp = stats.spearmanr(m["rougeL_margin"], m[col])
                conv_lines.append(f"| {name} | {len(m)} | {sp.statistic:+.3f} | {sp.pvalue:.3g} |")

    ev = pd.read_csv("outputs/eval_table.csv")[["paper_id", "title", "openalex_citations"]]
    top = (res.sort_values(["extractable", "eight_target", "rougeL_margin"],
                           ascending=False).head(15).merge(ev, on="paper_id"))
    exhibit = "\n".join(
        f"| {r.title[:70]} | {int(r.openalex_citations) if pd.notna(r.openalex_citations) else '—'} "
        f"| {r.rougeL_margin:+.3f} | {r.eight_target:.3f} |"
        for r in top.itertuples())

    report = f"""# Abstract-Completion Probe — verbatim memorization ({MODEL})

Title + year + first abstract sentence → model writes the rest (greedy decode).
Continuation scored against the true remainder vs {N_DECOYS} same-field×year decoys.
N = {n} (stratified by citation decile).

## Memorization gradient by citation decile

| Decile | N | ROUGE-L margin (target − decoy) | 8-gram hit rate | % extractable |
|---|---|---|---|---|
{chr(10).join(f"| {d} | {r.n:.0f} | {r.rougeL_margin:+.4f} | {r.eight_target:.4f} | {r.pct_extractable:.1%} |" for d, r in by_dec.iterrows())}

Spearman ρ (citation rank vs ROUGE-L margin): **{sp_grad.statistic:+.3f}** (p={sp_grad.pvalue:.3g})
Spearman ρ (citation rank vs 8-gram hit rate): **{sp_8g.statistic:+.3f}** (p={sp_8g.pvalue:.3g})

## Convergent validity — does verbatim memorization track outcome recall?

| Probe | N overlap | Spearman ρ vs ROUGE-L margin | p |
|---|---|---|---|
{chr(10).join(conv_lines) if conv_lines else "| (LAP/FAME CSVs not found) | | | |"}

## Exhibit — most-extractable papers

| Title | Citations | ROUGE-L margin | 8-gram rate |
|---|---|---|---|
{exhibit}

## Reading

- "Extractable" = target ROUGE-L beats ALL {N_DECOYS} decoys AND ≥1 verbatim 8-gram
  hit. Overall extractable rate: **{res['extractable'].mean():.1%}**.
- Positive citation-rank gradient = famous papers' text is preferentially in the
  weights — verbatim-level confirmation of the fame-recall finding.
- **One-sided test**: {MODEL} is instruction-tuned, which suppresses regurgitation.
  Positive results are strong evidence of memorization; a null does NOT prove the
  text is absent from the weights.
"""
    with open(OUT_REPORT, "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"Extractable: {res['extractable'].mean():.1%} overall")
    print(f"Gradient (cit rank vs ROUGE-L margin): ρ={sp_grad.statistic:+.3f} p={sp_grad.pvalue:.3g}")
    print(f"Gradient (cit rank vs 8-gram rate):    ρ={sp_8g.statistic:+.3f} p={sp_8g.pvalue:.3g}")
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="5 papers only")
    parser.add_argument("--n-per-decile", type=int, default=30)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    df = load_frame()
    print(f"Eligible papers (abstract + citation rank, remainder ≥ {MIN_REMAINDER_CHARS} chars): {len(df)}")

    if not args.report_only:
        sample = build_sample(df, args.n_per_decile)
        print(f"Sample: {len(sample)} papers ({args.n_per_decile}/decile)")
        run_probes(df, sample, smoke=args.smoke, workers=args.workers)

    if args.smoke:
        print("\nSmoke done — inspect outputs/leakage_abstract_completion_v1.csv")
    elif os.path.exists(OUT_CSV):
        run_report()
