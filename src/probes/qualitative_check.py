"""
Qualitative check — run the calibrated ICLR reviewer rubric over a handful of papers
with two models and keep every log for eyeballing.

Not an experiment. 9 papers x 2 models = 18 calls, chosen so the contamination
gradient is visible by inspection:

  2018   3 highest-cited ICLR 2018 accepts (GAT, Madry, ELMo) — deep inside both
         models' training data, and famous enough to be recalled by name
  2024   3 highest-cited ICLR 2024 accepts — inside Gemma's window, at or past
         Llama-3.3's Dec-2023 cutoff
  2026   top 3 from data/iclr-2026-highest-cited.csv — after both cutoffs, so any
         apparent familiarity is inference from the abstract, not recall

Both models get the identical system prompt (prompts/review/iclr_review_calibrated.txt,
`{year}` filled with the paper's own ICLR year) and the identical user message
(title + abstract). temperature=0.

Everything is logged. outputs/qualitative_check_logs.md is the file to read: it
contains, for all 18 calls, the exact system prompt, the exact user message, and
the model's untouched reply including whatever thinking it emitted.

Outputs:
  outputs/qualitative_check.csv          parsed scores, one row per (paper, model)
  outputs/qualitative_check_traces.jsonl full request/response per call, incremental
  outputs/qualitative_check_logs.md      the same calls rendered for reading
  outputs/qualitative_check_sample.csv   the 9 papers and where each came from

Run: python src/probes/qualitative_check.py            # fetch + report (resumes)
     python src/probes/qualitative_check.py --logs-only # re-render the markdown only
"""
import os
import re
import sys
import csv
import json
import time
import sqlite3
import hashlib
import argparse
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import load
from llm import MODELS, call, parse_rubric, sha

load_dotenv()
os.makedirs("outputs", exist_ok=True)

RUBRIC = "review/iclr_review_calibrated"
FIELDS = ["rating", "confidence", "correctness",
          "technical_novelty_and_significance", "empirical_novelty_and_significance"]


OUT_CSV = "outputs/qualitative_check.csv"
OUT_TRACES = "outputs/qualitative_check_traces.jsonl"
OUT_LOGS = "outputs/qualitative_check_logs.md"
OUT_SAMPLE = "outputs/qualitative_check_sample.csv"
URL = "https://api.together.xyz/v1/chat/completions"



def build_sample():
    """9 papers, deterministic: top 3 by S2 citations in each of 2018, 2024, 2026."""
    rows = []

    # 2018 — eval_table has no abstracts, so pull them from the source DB
    ev = pd.read_csv("outputs/eval_table.csv")
    s2 = pd.read_csv("outputs/s2_citations_v2.csv")
    con = sqlite3.connect("data/gen_review.db")
    abs18 = pd.read_sql("SELECT id AS paper_id, abstract FROM SUBMISSION", con)
    con.close()
    d = (ev[ev["year"] == 2018]
         .merge(s2[["paper_id", "s2_citations", "title_sim"]], on="paper_id")
         .merge(abs18, on="paper_id"))
    d = d[(d["title_sim"] >= 0.95) | d["title_sim"].isna()]
    for r in d.nlargest(3, "s2_citations").itertuples():
        rows.append({"paper_id": r.paper_id, "year": 2018, "arm": "2018_famous",
                     "title": r.title, "abstract": r.abstract,
                     "s2_citations": int(r.s2_citations),
                     "source": "outputs/eval_table.csv + s2_citations_v2.csv"})

    # 2024 — the per-year eval table already carries abstracts
    e24 = pd.read_csv("outputs/eval_table_2024.csv")
    s24 = pd.read_csv("outputs/s2_citations_2024.csv")
    m = e24.merge(s24[["paper_id", "s2_citations"]], on="paper_id")
    m = m[m["decision_class"].astype(str).str.contains("accept", case=False)]
    for r in m.nlargest(3, "s2_citations").itertuples():
        rows.append({"paper_id": r.paper_id, "year": 2024, "arm": "2024_boundary",
                     "title": r.title, "abstract": r.abstract,
                     "s2_citations": int(r.s2_citations),
                     "source": "outputs/eval_table_2024.csv + s2_citations_2024.csv"})

    # 2026 — abstracts live in the JSON companion of the CSV
    j = json.load(open("data/iclr-2026-highest-cited.json"))
    for r in sorted(j, key=lambda x: x["rank"])[:3]:
        rows.append({"paper_id": f"arxiv:{r['arxiv_id']}", "year": 2026,
                     "arm": "2026_after_cutoff", "title": r["title"],
                     "abstract": r["abstract"], "s2_citations": r["s2_citations"],
                     "source": "data/iclr-2026-highest-cited.json"})

    df = pd.DataFrame(rows)
    short = df["abstract"].fillna("").str.len() < 200
    if short.any():
        raise SystemExit(f"abstract too short for: {list(df.loc[short, 'paper_id'])}")
    df.to_csv(OUT_SAMPLE, index=False)
    print(f"Sample -> {OUT_SAMPLE}")
    for r in df.itertuples():
        print(f"  [{r.arm}] {r.s2_citations:>6} cites  {r.title[:64]}")
    return df






def run(df):
    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev["paper_id"], prev["model_key"]))
        print(f"Resuming — {len(done)} of {len(df) * len(MODELS)} calls already done")

    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    cols = (["paper_id", "arm", "year", "s2_citations", "model_key", "model", "title"]
            + FIELDS + ["rationale_chars", "completion_chars", "json_blocks",
                        "http_status", "system_prompt_sha1", "user_prompt_sha1"])

    with open(OUT_CSV, "a", newline="") as fcsv, open(OUT_TRACES, "a") as ftr:
        w = csv.DictWriter(fcsv, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in df.itertuples():
            system = load(RUBRIC, year=r.year)
            user = load("review/title_abstract_body", title=r.title, abstract=r.abstract)
            for mk, (model, mt) in MODELS.items():
                if (r.paper_id, mk) in done:
                    continue
                text, status, raw = call(model, mt, system, user)
                parsed, nblocks, mode = parse_rubric(text)
                ftr.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "paper_id": r.paper_id, "arm": r.arm, "year": int(r.year),
                    "title": r.title, "s2_citations": int(r.s2_citations),
                    "model_key": mk, "model": model,
                    "endpoint": URL, "temperature": 0, "max_tokens": mt,
                    "prompt_name": RUBRIC,
                    "system_prompt": system, "system_prompt_sha1": sha(system),
                    "user_prompt": user, "user_prompt_sha1": sha(user),
                    "completion": text, "completion_chars": len(text),
                    "parsed": parsed, "json_blocks_found": nblocks, "parse_mode": mode,
                    "http_status": status,
                    "usage": raw.get("usage"),
                    "finish_reason": ((raw.get("choices") or [{}])[0] or {}).get("finish_reason"),
                }, ensure_ascii=False) + "\n")
                ftr.flush()
                row = {"paper_id": r.paper_id, "arm": r.arm, "year": r.year,
                       "s2_citations": r.s2_citations, "model_key": mk,
                       "model": model, "title": r.title,
                       "rationale_chars": len(str(parsed.get("rationale", ""))),
                       "completion_chars": len(text), "json_blocks": nblocks,
                       "http_status": status, "system_prompt_sha1": sha(system),
                       "user_prompt_sha1": sha(user)}
                row.update({f: parsed.get(f, "") for f in FIELDS})
                w.writerow(row)
                fcsv.flush()
                print(f"  {mk:5} {r.arm:18} rating={parsed.get('rating', '?'):>5} "
                      f"conf={parsed.get('confidence', '?'):>5} "
                      f"({len(text)} chars) {r.title[:40]}")


def load_traces():
    """The JSONL is the source of truth; `parsed` is re-derived from the stored
    completion on every read, so fixing the parser fixes past runs too."""
    if not os.path.exists(OUT_TRACES):
        raise SystemExit(f"{OUT_TRACES} missing — run without --logs-only first")
    recs = [json.loads(l) for l in open(OUT_TRACES) if l.strip()]
    latest = {(r["paper_id"], r["model_key"]): r for r in recs}  # last write wins
    recs = sorted(latest.values(), key=lambda r: (r["year"], -r["s2_citations"],
                                                 r["model_key"]))
    for r in recs:
        r["parsed"], r["json_blocks_found"], r["parse_mode"] = parse_rubric(r["completion"])
    return recs


def rewrite_csv(recs):
    cols = (["paper_id", "arm", "year", "s2_citations", "model_key", "model", "title"]
            + FIELDS + ["rationale_chars", "completion_chars", "json_blocks",
                        "parse_mode", "http_status", "system_prompt_sha1",
                        "user_prompt_sha1"])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            p = r["parsed"]
            row = {k: r.get(k) for k in ("paper_id", "arm", "year", "s2_citations",
                                         "model_key", "model", "title",
                                         "parse_mode", "http_status",
                                         "system_prompt_sha1", "user_prompt_sha1",
                                         "completion_chars")}
            row["json_blocks"] = r["json_blocks_found"]
            row["rationale_chars"] = len(str(p.get("rationale", "")))
            row.update({fl: p.get(fl, "") for fl in FIELDS})
            w.writerow(row)
    print(f"Wrote {OUT_CSV} ({len(recs)} rows, re-derived from traces)")


def render_logs():
    """Every call, in full, in one readable file."""
    recs = load_traces()
    rewrite_csv(recs)
    systems = {r["system_prompt_sha1"]: r["system_prompt"] for r in recs}

    L = ["# Qualitative check — calibrated ICLR rubric, 9 papers x 2 models", "",
         f"Generated by `python src/probes/qualitative_check.py --logs-only` from "
         f"`{OUT_TRACES}` ({len(recs)} calls). Every prompt and reply below is "
         f"verbatim.", "",
         "| # | arm | paper | model | rating | confidence | chars | parse |",
         "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(recs, 1):
        p = r.get("parsed") or {}
        L.append(f"| {i} | {r['arm']} | {r['title'][:44]} | {r['model_key']} | "
                 f"{p.get('rating', '—')} | {p.get('confidence', '—')} | "
                 f"{r['completion_chars']} | {r['parse_mode']} |")

    L += ["", "## System prompt", "",
          f"`prompts/{RUBRIC}.txt` with `{{year}}` filled per paper — "
          f"{len(systems)} distinct strings across the run "
          f"(one per ICLR year), sha1 {', '.join(sorted(systems))}.", ""]
    for h, s in sorted(systems.items()):
        L += [f"<details><summary>sha1 <code>{h}</code> — "
              f"{len(s)} chars</summary>", "", "```text", s, "```", "", "</details>", ""]

    L += ["## Calls", ""]
    for i, r in enumerate(recs, 1):
        p = r.get("parsed") or {}
        L += [f"### {i}. [{r['arm']}] {r['title']}", "",
              f"- model: `{r['model']}` ({r['model_key']}), temperature "
              f"{r['temperature']}, max_tokens {r['max_tokens']}",
              f"- ICLR year sent: {r['year']} | S2 citations: {r['s2_citations']:,} "
              f"| paper_id: `{r['paper_id']}`",
              f"- system_prompt_sha1 `{r['system_prompt_sha1']}` | "
              f"user_prompt_sha1 `{r['user_prompt_sha1']}`",
              f"- HTTP {r['http_status']} | finish_reason `{r.get('finish_reason')}` | "
              f"usage {r.get('usage')} | JSON blocks found {r['json_blocks_found']} | "
              f"parse `{r['parse_mode']}`", "",
              "**User message sent:**", "", "```text", r["user_prompt"].rstrip(), "```", "",
              "**Raw reply (untouched, including any thinking):**", "",
              "```text", (r["completion"] or "(empty)").rstrip(), "```", "",
              "**Parsed:**", "", "```json", json.dumps(p, indent=2), "```", ""]

    open(OUT_LOGS, "w").write("\n".join(L))
    print(f"Wrote {OUT_LOGS} ({len(recs)} calls, "
          f"{os.path.getsize(OUT_LOGS) / 1024:.0f} KB)")

    # selfcheck: every call must have kept its prompts, and the rubric string must
    # be the one on disk right now
    bad = []
    for r in recs:
        if sha(load(RUBRIC, year=r["year"])) != r["system_prompt_sha1"]:
            bad.append(f"{r['paper_id']}/{r['model_key']}: system prompt has changed "
                       f"on disk since the run")
        if not r["user_prompt"] or not r["system_prompt"]:
            bad.append(f"{r['paper_id']}/{r['model_key']}: prompt not stored")
        if r["parse_mode"] == "none":
            bad.append(f"{r['paper_id']}/{r['model_key']}: no rubric fields could be "
                       f"read from the reply — inspect the raw completion")
    repaired = [r for r in recs if r["parse_mode"] != "strict"]
    if repaired:
        print(f"  NOTE {len(repaired)} replies needed a non-strict parse: "
              + ", ".join(f"{r['model_key']}/{r['paper_id']}({r['parse_mode']})"
                          for r in repaired))
    print("=== selfcheck ===")
    for b in bad:
        print(f"  FAIL {b}")
    if not bad:
        print(f"  all {len(recs)} calls stored both prompts; every system prompt "
              f"still matches prompts/{RUBRIC}.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-only", action="store_true",
                    help="re-render the markdown from existing traces, no API calls")
    a = ap.parse_args()
    if not a.logs_only:
        run(build_sample())
    render_logs()
