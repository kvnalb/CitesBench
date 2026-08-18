"""
Single-call review baseline: the same review, in one LLM call instead of nine.

Why this exists. The repo had no honest single-call control. The three naive LLM
regimes (llm_neutral / llm_positive / llm_ensemble) read title + abstract and score on
a different scale, so "the 9-call council beats a single call" was not something any
committed number could support — the input and the output schema both differed, and a
reviewer would read the gap as "less text" rather than "less scaffolding".

What is held identical, by importing from src/probes/slim_pipeline.py rather than
reimplementing, so the two cannot drift apart:

    model            the same Together-served model
    response model   SlimConferenceReview      (imported)
    completion       _complete_structured      (imported — same JSON spec, same
                                                <think>/fence stripping, same two
                                                repair retries)
    normalizer       to_archive_text(markdown) (imported)
    temperature      0.25                      matches stage persona_review
    max_tokens       3072                      matches stage persona_review, whose
                                                budget(3072) returns 3072 verbatim

If you change a parity number here, change it there.

WHAT IS DIFFERENT, AND WHY THAT IS THE RIGHT CHOICE. The input is not identical, and
it cannot be, because the council never shows any single call the whole paper:

    call 1        abstract + intro (8k chars) + conclusion (3k)
    calls 2-4     one budgeted excerpt each — intro 3.5k, method 6k, contribution ~2k
    calls 5-8     title, abstract, section titles, the regex structural inventory, and
                  the DISTILLED NOTES from calls 2-4 — no paper body text at all
    call 9        the four persona reviews

So the council reads the body in stages and hands its personas a summary. There is no
single "council input" to copy. This baseline is therefore given the whole normalized
paper in one user message, which is the deliberately conservative choice: the single
call sees strictly more raw paper text than any individual council call, so if the
council wins it cannot be because the baseline was starved of text. The comparison is
at the system level — same paper, same model, same schema, nine calls versus one.

The alternative arm, if a reviewer asks for it, is to feed this the persona's literal
view (abstract + structure, no notes). That handicaps it — it would never see the body
the council's calls 2-4 did read — so it belongs as a robustness row, not the headline.

"One call" is not the same as "cheap": the whole paper is ~42k chars here against ~45k
summed across the council's nine prompts, so prompt-token cost is comparable rather
than 9x lower. The traces record prompt_tokens per call, so use those, not intuition.

PROMPT PROVENANCE. prompts/review/single_call_liang_et_al{,_body}.txt is adapted from
Liang et al. 2023 (arXiv:2310.01783) appendix D.1. Their prompt emits NO SCORES — it
ends "Write Outlines only", because that paper measures overlap between GPT-4 comments
and human reviewer comments, not score agreement. The four-section review outline is
theirs and is preserved. The numeric scales, the JSON schema and the anti-compression
warnings are ours, added so the output lands in the council's schema. Do not describe
this as "the Liang et al. prompt" in the paper; it is adapted from it.

POPULATION. ReviewArena is the only local source of full paper text, which bounds this
hard (measured, not assumed):

    2018, 2019   no local full text at all — absent from ReviewArena, and the archive
                 run holding local_fulltext.json lives on a collaborator's Dropbox
                 (exactly one paper of it is on disk here)
    2020         2,211 of 2,213 papers (99.9%), INCLUDING 1,524 of 1,526 rejects
    2025         3,703 papers, accepts only

So 2020 is the only year where this baseline faces a complete accept-and-reject
selection task, and it is the largest of the three primary years. For 2025 the frozen
council population is reused verbatim when present, so both regimes score the same
papers rather than two overlapping sets.

Everything is traced: one JSONL line per call with the verbatim messages, the untouched
reply, token counts and latency. Both outputs are append-only and a restart skips
paper_ids that already SUCCEEDED, so a crash costs nothing already paid for.

Outputs:
  outputs/samples/single_call_{year}_papers.csv   frozen population, written once
  outputs/single_call_{year}_{model}.csv          one row per paper, incremental
  outputs/single_call_{year}_{model}_traces.jsonl one line per call, incremental

Run: python src/probes/run_single_call_baseline.py --dry-run          # no API calls
     python src/probes/run_single_call_baseline.py --year 2020 --n 2  # smoke, 2 calls
     python src/probes/run_single_call_baseline.py --year 2020 --workers 8
"""
import os
import re
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
from prompts import load as load_prompt
from llm import MODELS
from build.build_slim_2025_papers import load_year, MIN_CHARS, SEED
from build.normalize_paper_markdown import to_archive_text
from probes.slim_pipeline import (SlimConferenceReview, _complete_structured,
                                 resolve_base_model)

load_dotenv()
os.makedirs("outputs/samples", exist_ok=True)

SYSTEM_TEMPLATE = "review/single_call_liang_et_al"
BODY_TEMPLATE = "review/single_call_liang_et_al_body"

# Parity with stage persona_review in slim_pipeline.review_paper_slim. Not tunable
# knobs — they are what makes this a controlled comparison.
TEMPERATURE = 0.25
MAX_TOKENS = 3072
STAGE = "single_call_review"

# The council's frozen 2025 list. Reused when scoring 2025 so both regimes see the
# same papers; irrelevant for 2020, which the council never ran locally.
COUNCIL_2025_SAMPLE = "outputs/samples/slim_2025_papers.csv"

# Same column list as build_committee_ratings_2025.py, so anything already reading
# committee output reads this unchanged.
FIELDS = ["paper_id", "decision", "primary_area", "markdown_chars", "run_order",
          "model", "rating", "confidence", "soundness", "presentation", "contribution",
          "recommendation", "n_calls", "text_synthesis", "prompt_tokens",
          "completion_tokens", "cost_usd", "retried_calls", "elapsed_s", "error", "ts"]

_lock = threading.Lock()


def _slug(model_key):
    """Filesystem-safe tag: a dedicated endpoint id carries slashes and dots, which
    would otherwise scatter outputs across invented directories."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model_key)


def sample_path(year):
    return f"outputs/samples/single_call_{year}_papers.csv"


def paths(year, model_key):
    tag = _slug(model_key)
    return (f"outputs/single_call_{year}_{tag}.csv",
            f"outputs/single_call_{year}_{tag}_traces.jsonl")


def build_population(year):
    """Frozen population for a year, written once and reused.

    Rejects are KEPT. That is the point of choosing 2020 — a pool of accepts only can
    measure ranking within accepts but says nothing about accept-vs-reject separation,
    which is the selection task the benchmark is about.
    """
    p = sample_path(year)
    if os.path.exists(p):
        s = pd.read_csv(p)
        print(f"Reusing frozen population {p} — {len(s):,} papers "
              f"(delete it to draw a new one)")
        return s

    if year == 2025 and os.path.exists(COUNCIL_2025_SAMPLE):
        # score exactly the papers the council scored, not an overlapping set
        keep = pd.read_csv(COUNCIL_2025_SAMPLE)
        print(f"Using the council's frozen 2025 list ({len(keep):,} papers) so both "
              f"regimes score the same papers")
    else:
        d = load_year(year).rename(columns={"forum_id": "paper_id"})
        keep = d[d.markdown_chars >= MIN_CHARS].copy()
        # same seed as the council population, so run_order < N is a reproducible
        # smoke test rather than whatever order the parquet happened to be in
        keep = keep.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        keep["run_order"] = range(len(keep))
        keep = keep[["paper_id", "title", "decision", "markdown_chars",
                     "primary_area", "num_reviews", "run_order"]]

    keep.to_csv(p, index=False)
    acc = keep.decision.astype(str).str.startswith("Accept")
    print(f"wrote {p}: {len(keep):,} papers "
          f"({int(acc.sum()):,} accept / {int((~acc).sum()):,} reject)")
    return keep


def done_ids(csv_path):
    """paper_ids that SUCCEEDED, so a restart does not re-bill them.

    Failed rows are excluded on purpose. A failure is usually about the moment rather
    than the paper — an endpoint mid-redeploy 503s everything in flight — so treating
    "has a row" as "is done" would permanently skip papers that only needed a retry,
    and the run would report complete while the corpus was quietly short.
    """
    if not os.path.exists(csv_path):
        return set()
    try:
        d = pd.read_csv(csv_path)
        if "error" in d.columns:
            d = d[d.error.isna() | (d.error.astype(str).str.strip() == "")]
        return set(d.paper_id.astype(str))
    except Exception:
        return set()


def build_messages(title, markdown, year):
    """The two-message conversation for one paper.

    to_archive_text is the same normalizer the council uses: the ReviewArena markdown
    column is OCR'd PDF with no `#` headings, and without this the section parser
    returns one untyped blob. Passing raw markdown here would quietly make this a
    different measurement from the council.
    """
    return [
        {"role": "system", "content": load_prompt(SYSTEM_TEMPLATE, year=year)},
        {"role": "user", "content": load_prompt(BODY_TEMPLATE, title=title or "",
                                                paper_text=to_archive_text(markdown))},
    ]


def one_paper(row, markdown, model_key, year, fcsv, writer, ftr, base_model=None):
    t0 = time.time()
    rec = {k: row.get(k) for k in ("paper_id", "decision", "primary_area",
                                   "markdown_chars", "run_order")}
    rec["model"] = model_key
    rec["text_synthesis"] = "single_call"
    traces = []
    model = MODELS[model_key][0] if model_key in MODELS else model_key
    try:
        review = _complete_structured(
            model=model,
            messages=build_messages(row.get("title"), markdown, year),
            response_model=SlimConferenceReview,
            call_traces=traces,
            stage=STAGE,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=600,
            cost_model=resolve_base_model(model, base_model),
        )
        rec.update({
            "rating": review.rating,
            "confidence": review.confidence,
            "soundness": review.soundness,
            "presentation": review.presentation,
            "contribution": review.contribution,
            "recommendation": review.recommendation,
            "n_calls": 1,
            "prompt_tokens": sum(t.get("prompt_tokens") or 0 for t in traces),
            "completion_tokens": sum(t.get("completion_tokens") or 0 for t in traces),
            "cost_usd": round(sum(t.get("cost_usd") or 0.0 for t in traces), 6) or None,
            "retried_calls": sum(1 for t in traces if (t.get("attempts") or 1) > 1),
            "error": "",
        })
    except Exception as e:
        # one dead paper must not kill the run; record it and move on
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
    ap.add_argument("--year", type=int, default=2020, choices=[2020, 2025],
                    help="2020 is the only year with full text for rejects too")
    ap.add_argument("--model", default="gemma", help="registry key or a Together id")
    ap.add_argument("--n", type=int, default=0, help="first N by run_order; 0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-model", default=None,
                    help="what a dedicated endpoint actually serves, for cost accounting")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble and print one prompt, make no API calls")
    a = ap.parse_args()

    pop = build_population(a.year)
    # full text stays in the parquet until needed — thousands x ~79k chars is ~GBs
    text = load_year(a.year).set_index("forum_id").markdown

    if a.dry_run:
        row = pop.sort_values("run_order").iloc[0]
        msgs = build_messages(row.get("title"), text.get(row.paper_id, ""), a.year)
        print(f"\n=== DRY RUN — {row.paper_id} ({row.decision}) — no API call ===")
        for m in msgs:
            print(f"\n--- {m['role']} ({len(m['content']):,} chars) ---")
            print(m["content"][:1500])
            if len(m["content"]) > 1500:
                print(f"... [{len(m['content'])-1500:,} more chars]")
        return

    out_csv, out_traces = paths(a.year, a.model)
    done = done_ids(out_csv)
    todo = pop.sort_values("run_order")
    if a.n:
        todo = todo.head(a.n)
    todo = todo[~todo.paper_id.astype(str).isin(done)]
    print(f"{len(done):,} already done, {len(todo):,} to go "
          f"-> {out_csv}  (1 call each, model={a.model})")
    if todo.empty:
        return

    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fcsv, open(out_traces, "a") as ftr:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(one_paper, r._asdict(), text.get(r.paper_id, ""),
                              a.model, a.year, fcsv, writer, ftr, a.base_model)
                    for r in todo.itertuples(index=False)]
            ok = bad = 0
            for i, f in enumerate(as_completed(futs), 1):
                rec = f.result()
                ok, bad = ok + (rec["error"] == ""), bad + (rec["error"] != "")
                if i % 25 == 0 or i == len(futs):
                    print(f"  {i:,}/{len(futs):,}  ok={ok:,} failed={bad:,}", flush=True)
    print(f"Done. {ok:,} scored, {bad:,} failed -> {out_csv}")


def demo():
    """Offline self-check: prompt assembly and schema parity, no API calls.

    Deliberately does not hit the network. The expensive failure mode for this file is
    not a crash, it is a run that bills thousands of calls and returns scores on the
    wrong scale, so what is worth asserting cheaply is the schema and the parity
    constants.
    """
    msgs = build_messages("A Test Paper", "# Intro\nWe propose a thing.\n", 2020)
    assert len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "ICLR 2020" in msgs[0]["content"], "year placeholder never filled"
    assert "{" + "year}" not in msgs[0]["content"], "unfilled placeholder"
    assert "A Test Paper" in msgs[1]["content"] and "======" in msgs[1]["content"]
    assert "propose a thing" in msgs[1]["content"], "paper text missing from body"

    # the schema this must land in is the council's, imported not redeclared
    r = SlimConferenceReview(rating=6.0, confidence=3.0, soundness=3.0, presentation=3.0,
                             contribution=3.0, recommendation="borderline accept",
                             rationale="r", summary="s", strength="st",
                             weaknesses="w", questions="q")
    assert 1.0 <= r.rating <= 10.0
    for k in ("rating", "confidence", "soundness", "presentation", "contribution",
              "recommendation"):
        assert k in FIELDS, f"{k} missing from the CSV columns"

    # parity with stage persona_review — if these drift, the comparison is not controlled
    assert (TEMPERATURE, MAX_TOKENS) == (0.25, 3072)
    print("ok — prompts assemble, schema is the council's, parity constants intact")


if __name__ == "__main__":
    demo() if len(sys.argv) == 1 else main()
