"""
Score a year's submissions with the calibrated ICLR rubric and compare against both
ground truths we have: human reviewer scores and eventual citations.

ICLR 2025 is the useful holdout. It has what 2026 lacks — rejected submissions and
human ratings — and it postdates Llama-3.3's Dec-2023 cutoff, so the model is scoring
papers it cannot have memorised the outcomes of (0 of 138 accept/reject commitments in
the leakage probes, see outputs/leakage_ledger.csv).

Design:
  population   papers with an abstract, a human mean_rating, and a verified S2 citation
               count (title_sim >= 0.95 or an ID match)
  sample       stratified by decision_class x citation quartile, seeded, written once
               and reused so a rerun scores the same papers
  treatment    prompts/review/iclr_review_calibrated.txt, title + abstract only,
               temperature 0 — identical to what src/probes/qualitative_check.py sends
  outcomes     Spearman rho of model rating vs (a) log1p citations, (b) human
               mean_rating; recall@top-decile; accept-vs-reject separation

Everything is logged: the traces file holds the verbatim system and user prompt, the
untouched completion, token usage, and the parse mode for every call.

Coverage caveat recorded in the report: S2 resolves 73% of 2025 accepts but only 37%
of rejects, so the eligible pool is biased toward rejects visible enough to have a
preprint. That understates the gap between accepted and rejected papers.

Outputs:
  outputs/samples/review_eval_{year}_sample.csv   frozen sample, written once
  outputs/review_eval_{year}_{model}.csv          one row per paper, incremental
  outputs/review_eval_{year}_{model}_traces.jsonl full prompt + reply per call
  outputs/review_eval_{year}_report.md            correlations and comparisons

Run: python src/probes/run_review_eval.py --year 2025 --n 60 --models gemma   # pilot
     python src/probes/run_review_eval.py --year 2025 --n 1500 --workers 12
     python src/probes/run_review_eval.py --year 2025 --report-only
"""
import os
import sys
import csv
import json
import argparse
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import load
from llm import MODELS, RUBRIC_FIELDS, call, parse_rubric, sha

load_dotenv()
os.makedirs("outputs/samples", exist_ok=True)

RUBRIC = "review/iclr_review_calibrated"
SEED = 42


def paths(year, model_key):
    return (f"outputs/review_eval_{year}_{model_key}.csv",
            f"outputs/review_eval_{year}_{model_key}_traces.jsonl")


def sample_path(year):
    return f"outputs/samples/review_eval_{year}_sample.csv"


def build_sample(year, n):
    """Frozen, stratified by decision_class x citation quartile. Written once."""
    p = sample_path(year)
    if os.path.exists(p):
        s = pd.read_csv(p)
        print(f"Reusing frozen sample {p} — {len(s)} papers "
              f"(delete it to draw a new one)")
        return s

    ev = pd.read_csv(f"outputs/eval_table_{year}.csv", low_memory=False)
    s2 = pd.read_csv(f"outputs/s2_citations_{year}.csv")[
        ["paper_id", "s2_citations", "title_sim"]]
    ev = ev.drop(columns=[c for c in ("s2_citations", "title_sim") if c in ev.columns])
    d = ev.merge(s2, on="paper_id", how="left")
    d = d[d["s2_citations"].notna()
          & ((d["title_sim"] >= 0.95) | d["title_sim"].isna())
          & d["abstract"].notna() & d["mean_rating"].notna()]
    d = d[d["decision_class"].isin(["accept", "reject"])]      # withdrawn excluded
    print(f"eligible population: {len(d):,} "
          f"({dict(d.decision_class.value_counts())})")

    d = d.copy()
    d["cite_q"] = pd.qcut(d["s2_citations"].rank(method="first"), 4, labels=False)
    rng = np.random.default_rng(SEED)
    cells = list(d.groupby(["decision_class", "cite_q"]))
    per = max(4, n // len(cells))
    parts = []
    for _, g in cells:
        k = min(per, len(g))
        parts.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    s = pd.concat(parts).reset_index(drop=True)
    keep = ["paper_id", "title", "abstract", "decision_class", "mean_rating",
            "rating_std", "n_reviews", "s2_citations", "cite_q"]
    s = s[[c for c in keep if c in s.columns]]
    s["year"] = year
    s.to_csv(p, index=False)
    print(f"Wrote {p} — {len(s)} papers across {len(cells)} cells "
          f"({dict(s.decision_class.value_counts())})")
    return s


def run(sample, year, model_key, workers):
    model, max_tokens = MODELS[model_key]
    out_csv, out_traces = paths(year, model_key)

    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv)["paper_id"])
    todo = sample[~sample["paper_id"].isin(done)]
    print(f"\n{model_key} ({model}): {len(done)} done, {len(todo)} to run, "
          f"{workers} workers")
    if todo.empty:
        return

    cols = (["paper_id", "year", "decision_class", "mean_rating", "s2_citations",
             "model_key", "model"] + RUBRIC_FIELDS
            + ["rationale_chars", "completion_chars", "parse_mode", "http_status",
               "prompt_tokens", "completion_tokens", "system_prompt_sha1",
               "user_prompt_sha1"])
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    lock, counter = threading.Lock(), [0]

    with open(out_csv, "a", newline="") as fcsv, open(out_traces, "a") as ftr:
        w = csv.DictWriter(fcsv, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()

        def one(r):
            system = load(RUBRIC, year=year)
            user = load("review/title_abstract_body", title=r.title, abstract=r.abstract)
            text, status, raw = call(model, max_tokens, system, user)
            parsed, nblocks, mode = parse_rubric(text)
            usage = raw.get("usage") or {}
            with lock:
                ftr.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "paper_id": r.paper_id, "year": int(year),
                    "decision_class": r.decision_class, "title": r.title,
                    "mean_rating": r.mean_rating, "s2_citations": r.s2_citations,
                    "model_key": model_key, "model": model, "temperature": 0,
                    "max_tokens": max_tokens, "prompt_name": RUBRIC,
                    "system_prompt": system, "system_prompt_sha1": sha(system),
                    "user_prompt": user, "user_prompt_sha1": sha(user),
                    "completion": text, "completion_chars": len(text),
                    "parsed": parsed, "json_blocks_found": nblocks,
                    "parse_mode": mode, "http_status": status, "usage": usage,
                    "finish_reason": ((raw.get("choices") or [{}])[0] or {}).get("finish_reason"),
                }, ensure_ascii=False) + "\n")
                ftr.flush()
                row = {"paper_id": r.paper_id, "year": year,
                       "decision_class": r.decision_class, "mean_rating": r.mean_rating,
                       "s2_citations": r.s2_citations, "model_key": model_key,
                       "model": model, "parse_mode": mode, "http_status": status,
                       "completion_chars": len(text),
                       "rationale_chars": len(str(parsed.get("rationale", ""))),
                       "prompt_tokens": usage.get("prompt_tokens"),
                       "completion_tokens": usage.get("completion_tokens"),
                       "system_prompt_sha1": sha(system),
                       "user_prompt_sha1": sha(user)}
                row.update({f: parsed.get(f, "") for f in RUBRIC_FIELDS})
                w.writerow(row)
                fcsv.flush()
                counter[0] += 1
                if counter[0] % 25 == 0 or counter[0] == len(todo):
                    print(f"  {counter[0]}/{len(todo)}  rating={parsed.get('rating','?')} "
                          f"({mode})  {str(r.title)[:44]}", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(one, r) for r in todo.itertuples()]):
                f.result()


def report(year, model_keys):
    from scipy import stats
    L = [f"# Review-rubric holdout — ICLR {year}", "",
         f"Generated by `python src/probes/run_review_eval.py --year {year} "
         f"--report-only`. Prompt: `prompts/{RUBRIC}.txt`, title + abstract only, "
         f"temperature 0.", ""]

    for mk in model_keys:
        out_csv, _ = paths(year, mk)
        if not os.path.exists(out_csv):
            continue
        d = pd.read_csv(out_csv).drop_duplicates("paper_id", keep="last")
        d = d[pd.to_numeric(d["rating"], errors="coerce").notna()].copy()
        d["rating"] = d["rating"].astype(float)
        d["log_cites"] = np.log1p(d["s2_citations"])
        acc = d[d.decision_class == "accept"]
        rej = d[d.decision_class == "reject"]

        def rho(x, y):
            if len(x) < 8 or x.nunique() < 2:
                return float("nan"), float("nan")
            r = stats.spearmanr(x, y)
            return r.statistic, r.pvalue

        r_c, p_c = rho(d["rating"], d["log_cites"])
        r_h, p_h = rho(d["rating"], d["mean_rating"])
        rh_c, ph_c = rho(d["mean_rating"], d["log_cites"])
        # recall@top-decile of citations, picking the same count by model rating
        k = max(1, len(d) // 10)
        true_top = set(d.nlargest(k, "s2_citations").paper_id)
        model_top = set(d.nlargest(k, "rating").paper_id)
        human_top = set(d.nlargest(k, "mean_rating").paper_id)

        L += [f"## {mk} — `{MODELS[mk][0]}`", "",
              f"N = {len(d):,} scored ({len(acc):,} accepted, {len(rej):,} rejected). "
              f"Parse modes: {d.parse_mode.value_counts().to_dict()}", "",
              "| relationship | Spearman rho | p |", "|---|---|---|",
              f"| model rating vs log citations | {r_c:.3f} | {p_c:.2g} |",
              f"| **human** mean_rating vs log citations | {rh_c:.3f} | {ph_c:.2g} |",
              f"| model rating vs human mean_rating | {r_h:.3f} | {p_h:.2g} |", "",
              f"Recall@top-{k} most-cited: model {len(true_top & model_top)}/{k} "
              f"({100*len(true_top & model_top)/k:.0f}%), "
              f"human {len(true_top & human_top)}/{k} "
              f"({100*len(true_top & human_top)/k:.0f}%).", "",
              "| group | n | model rating | human rating | median citations |",
              "|---|---|---|---|---|"]
        for name, g in (("accepted", acc), ("rejected", rej)):
            if len(g):
                L.append(f"| {name} | {len(g):,} | {g.rating.mean():.2f} | "
                         f"{g.mean_rating.mean():.2f} | {g.s2_citations.median():.0f} |")
        if len(acc) > 5 and len(rej) > 5:
            t = stats.mannwhitneyu(acc.rating, rej.rating, alternative="greater")
            L += ["", f"Accept-vs-reject separation on the model's rating: "
                      f"Mann-Whitney one-sided p = {t.pvalue:.3g}."]
        L.append("")

    L += ["## Caveats", "",
          "- Eligibility requires a verified S2 match, which resolves ~73% of accepts "
          "but only ~37% of rejects. Rejected papers in this sample are the visible "
          "ones (they have preprints), so the accept-reject contrast is understated.",
          "- Citations are a few months old; ranks are noisier than the 2018-20 tables "
          "and reward early preprint visibility.",
          "- Title + abstract only. The 2018-20 committee results came from a 9-call "
          "pipeline over full paper text, so those numbers are NOT directly comparable "
          "until the same abstract-only rubric is run on 2018-20.",
          "- Human `mean_rating` is the pre-rebuttal-through-final average recorded in "
          "the DB, not a reviewer's private score."]

    out = f"outputs/review_eval_{year}_report.md"
    open(out, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--n", type=int, default=1500, help="target sample size")
    ap.add_argument("--models", default="gemma", help="comma-separated: gemma,llama")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    keys = [k.strip() for k in a.models.split(",") if k.strip()]
    if not a.report_only:
        s = build_sample(a.year, a.n)
        for k in keys:
            run(s, a.year, k, a.workers)
    report(a.year, keys)
