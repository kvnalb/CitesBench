"""
Run the slim 9-call review pipeline over ICLR 2025 accepted papers.

Population is the frozen list from src/build/build_slim_2025_papers.py — 3,703 papers,
accepts only, because ReviewArena's 2025 slice has no rejects and the 5,019 rejected
submissions in gen_review.db have no full text anywhere local. This run can therefore
measure score spread and citation correlation, but NOT accept-vs-reject separation.

Nine calls per paper: contribution_extraction, intro_notes, method_notes (skipped when
no methodology section is found), contribution_notes, four persona reviews, committee
synthesis. Papers are processed in `run_order`, a seeded shuffle, so --n 5 is a
reproducible smoke test rather than a decision-correlated slice.

Everything is traced. One JSONL line per LLM call, appended the moment it returns,
holding the verbatim messages and the untouched response. At ~33k calls a crash at
paper 2,800 must cost nothing, so both outputs are append-only and a restart skips
paper_ids already present in the CSV.

Two fields worth watching in the per-paper CSV, because both fail quietly:
  n_calls          one short of the expected count (8 for gemma, 9 otherwise) means
                   no methodology section was detected
  text_synthesis   'fallback' means the committee synthesis call threw and scores were
                   aggregated arithmetically — the row looks normal either way

Writes the archive's per-paper directory layout, not just flat tables. That is what the
2018-2020 runs produced, what the Committee Pipeline X-Ray page reads, and what any
comparison between the two eras has to line up. Retrofitting it after a 3,703-paper run
would mean running twice.

One deviation, stated rather than papered over: the archive recorded per-call cost in
dollars via litellm, which the port dropped. `call_costs` here carries prompt and
completion token counts instead. Tokens are the durable quantity anyway — prices move.

Outputs:
  outputs/runs/{run_slug}/run_manifest.json     config, written before any paper
  outputs/runs/{run_slug}/summary.json          totals, written at the end
  outputs/runs/{run_slug}/papers/{id}/          input.json, coarse_review.{json,md},
                                                coarse_call_traces.json,
                                                coarse_call_costs.json,
                                                persona_reviews/{slug}.{json,md},
                                                paper_result.json
  outputs/runs/{run_slug}/paper_results.jsonl    one row per paper, incremental
  outputs/slim_2025_{model}.csv                 one row per paper, incremental
  outputs/slim_2025_{model}_traces.jsonl        one line per LLM call, incremental

Run: python src/probes/run_slim_pipeline.py --model gemma --n 5          # smoke test
     python src/probes/run_slim_pipeline.py --model gemma --workers 8    # full 3,703
"""
import os
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
from llm import MODELS
from build.build_slim_2025_papers import load_year
from build.normalize_paper_markdown import to_archive_text
from probes.slim_pipeline import review_paper_slim, _skips_contribution_extraction

load_dotenv()
os.makedirs("outputs", exist_ok=True)

SAMPLE_CSV = "outputs/samples/slim_2025_papers.csv"
PERSONAS = ["empiricist", "theorist", "systems_pragmatist", "novelty_gatekeeper"]

FIELDS = ["paper_id", "decision", "primary_area", "markdown_chars", "run_order",
          "model", "rating", "confidence", "soundness", "presentation", "contribution",
          "recommendation", "n_calls", "skipped_stages", "text_synthesis", "prompt_tokens",
          "completion_tokens", "cost_usd", "retried_calls", "elapsed_s", "error", "ts"]

_lock = threading.Lock()


RUNS_DIR = "outputs/runs"


def paths(model_key):
    return (f"outputs/slim_2025_{model_key}.csv",
            f"outputs/slim_2025_{model_key}_traces.jsonl")


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def write_paper_dir(run_dir, row, result, traces, model_key):
    """The archive's per-paper layout, so 2025 runs are inspectable the same way the
    2018-2020 ones are — same filenames, same shapes, same X-Ray page.

    Cost is computed the archive's way (litellm.completion_cost on the same arguments),
    so coarse_call_costs.json is comparable across eras; token counts are recorded
    alongside it because prices move and tokens do not.
    """
    pdir = os.path.join(run_dir, "papers", str(row["paper_id"]))
    os.makedirs(os.path.join(pdir, "persona_reviews"), exist_ok=True)

    _write_json(os.path.join(pdir, "input.json"), {
        "paper_id": row["paper_id"], "title": row.get("title"),
        "year": 2025, "decision": row.get("decision"),
        "accepted": 1.0,                       # ReviewArena 2025 is accepts-only
        "primary_area": row.get("primary_area"),
        "num_reviews": row.get("num_reviews"),
        "markdown_chars": row.get("markdown_chars"),
        "run_order": row.get("run_order"),
        "text_source": "ReviewArena parquet (markdown column), heading markers stripped",
        "committee_model": MODELS[model_key][0], "committee_bias": "plain",
    })

    call_costs = [{"stage": t.get("stage"), "model": t.get("model"),
                   "prompt_chars": t.get("prompt_chars"),
                   "response_chars": t.get("response_chars"),
                   "prompt_tokens": t.get("prompt_tokens"),
                   "completion_tokens": t.get("completion_tokens"),
                   "cost_usd": t.get("cost_usd")}
                  for t in traces]
    total_cost = sum(c["cost_usd"] or 0.0 for c in call_costs) or None

    _write_json(os.path.join(pdir, "coarse_review.json"), {
        "paper_id": row["paper_id"], "title": result.title,
        "source_title": row.get("title"), "year": 2025,
        "decision": row.get("decision"),
        "committee_model": result.model, "committee_bias": "plain",
        "llm_calls": result.llm_calls,
        "review_cost_usd": round(total_cost, 6) if total_cost else None,
        "call_costs": call_costs,
        **{f: getattr(result.review, f) for f in
           ("rating", "confidence", "soundness", "presentation", "contribution",
            "recommendation", "summary", "strength", "weaknesses", "questions",
            "rationale")},
        "committee": result.committee,
        "structural_inventory": result.structural_inventory.as_dict(),
    })
    with open(os.path.join(pdir, "coarse_review.md"), "w") as f:
        f.write(result.markdown)
    _write_json(os.path.join(pdir, "coarse_call_costs.json"), {"call_costs": call_costs})
    _write_json(os.path.join(pdir, "coarse_call_traces.json"), {"call_traces": traces})

    for slug, review in result.persona_reviews.items():
        _write_json(os.path.join(pdir, "persona_reviews", f"{slug}.json"), {
            "paper_id": row["paper_id"], "title": result.title,
            "persona_slug": slug, "committee_bias": "plain",
            **{f: getattr(review, f) for f in
               ("rating", "confidence", "soundness", "presentation", "contribution",
                "recommendation", "summary", "strength", "weaknesses", "questions",
                "rationale")},
        })
        md = result.persona_markdowns.get(slug)
        if md:
            with open(os.path.join(pdir, "persona_reviews", f"{slug}.md"), "w") as f:
                f.write(md)
    return pdir


def done_ids(csv_path):
    """paper_ids already written, so a restart resumes instead of re-billing."""
    if not os.path.exists(csv_path):
        return set()
    try:
        return set(pd.read_csv(csv_path).paper_id.astype(str))
    except Exception:
        return set()


def one_paper(row, markdown, model_key, fcsv, writer, ftr, run_dir=None):
    t0 = time.time()
    rec = {k: row.get(k) for k in ("paper_id", "decision", "primary_area",
                                   "markdown_chars", "run_order")}
    rec["model"] = model_key
    try:
        result, traces = review_paper_slim(
            paper_id=row["paper_id"],
            markdown=to_archive_text(markdown),
            model_key=model_key,
            personas=PERSONAS,
        )
        rec.update({
            "rating": result.get("rating"),
            "confidence": result.get("confidence"),
            "soundness": result.get("soundness"),
            "presentation": result.get("presentation"),
            "contribution": result.get("contribution"),
            "recommendation": result.get("recommendation"),
            # llm_calls, not len(traces): a deliberately skipped stage still occupies
            # a trace slot (contribution_extraction on gemma), so traces overcount by one
            "n_calls": result.get("llm_calls"),
            "skipped_stages": sum(1 for t in traces if t.get("skipped")),
            "text_synthesis": result.get("text_synthesis"),
            "prompt_tokens": sum(t.get("prompt_tokens") or 0 for t in traces),
            "completion_tokens": sum(t.get("completion_tokens") or 0 for t in traces),
            # a call that needed a repair turn got a different conversation than a
            # clean one; worth seeing per paper rather than only in the traces
            "cost_usd": round(sum(t.get("cost_usd") or 0.0 for t in traces), 6) or None,
            "retried_calls": sum(1 for t in traces if (t.get("attempts") or 1) > 1),
            "error": "",
        })
    except Exception as e:
        # a dead paper must not kill a 3,703-paper run; record it and move on
        traces = []
        rec.update({"n_calls": 0, "error": f"{type(e).__name__}: {e}"[:300]})

    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if run_dir and rec.get("error") == "":
        pdir = write_paper_dir(run_dir, row, result, traces, model_key)
        _write_json(os.path.join(pdir, "paper_result.json"), rec)

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
    ap.add_argument("--model", default="gemma", choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=0, help="first N by run_order; 0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-slug", default=None,
                    help="run directory name under outputs/runs/ (default: derived)")
    args = ap.parse_args()

    if not os.path.exists(SAMPLE_CSV):
        sys.exit(f"ERROR: {SAMPLE_CSV} missing — run src/build/build_slim_2025_papers.py")

    papers = pd.read_csv(SAMPLE_CSV).sort_values("run_order")
    if args.n:
        papers = papers.head(args.n)

    out_csv, out_traces = paths(args.model)
    seen = done_ids(out_csv)
    todo = papers[~papers.paper_id.astype(str).isin(seen)]
    if seen:
        print(f"resuming: {len(seen)} papers already done, {len(todo)} to go")
    if todo.empty:
        sys.exit("nothing to do")

    # full text stays in the parquet until needed — 3,703 x 79k chars is ~290MB
    text = load_year().set_index("forum_id").markdown

    # Run config is written BEFORE any paper, so a run killed halfway is still
    # self-describing — the archive did the same and it is why we could reconstruct
    # what its runs did years later.
    slug = args.run_slug or f"iclr2025_{args.model}_slim"
    run_dir = os.path.join(RUNS_DIR, slug)
    started = datetime.now(timezone.utc)
    _write_json(os.path.join(run_dir, "run_manifest.json"), {
        "created_at_utc": started.isoformat(timespec="seconds"),
        "run_slug": slug,
        "years": [2025],
        "committee_model": MODELS[args.model][0],
        "committee_bias": "plain",
        "personas": PERSONAS,
        "persona_weights": None,
        "decision_head_model": None,     # not ported yet
        "n_selected": int(len(todo)),
        "max_parallel_papers": args.workers,
        "sample_csv": SAMPLE_CSV,
        "text_source": "data/ReviewArena/raw/data/*.parquet (markdown column)",
        "command": " ".join(sys.argv),
    })
    # gemma and mistral skip contribution_extraction (the archive's gate), everything
    # else runs it — so the expected call count is model-dependent, not a constant
    expected_calls = 8 if _skips_contribution_extraction(MODELS[args.model][0]) else 9
    print(f"run dir: {run_dir}  (expecting {expected_calls} calls/paper)", flush=True)

    # A run killed before its first row leaves a 0-byte file behind: DictWriter buffers
    # the header until a row is written or the file is closed. Treating "exists" as
    # "has a header" then silently produces a headerless CSV.
    new_file = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    with open(out_csv, "a", newline="") as fcsv, open(out_traces, "a") as ftr:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one_paper, r._asdict(), text[r.paper_id],
                              args.model, fcsv, writer, ftr, run_dir): r.paper_id
                    for r in todo.itertuples(index=False)}
            for i, f in enumerate(as_completed(futs), 1):
                rec = f.result()
                flag = ""
                if rec.get("error"):
                    flag = f"  ERROR {rec['error'][:60]}"
                elif rec.get("n_calls") != expected_calls:
                    flag = f"  {rec['n_calls']} calls, expected {expected_calls}"
                print(f"[{i}/{len(futs)}] {rec['paper_id']} "
                      f"rating={rec.get('rating')} {rec['elapsed_s']}s "
                      f"{rec.get('prompt_tokens')}in/{rec.get('completion_tokens')}out"
                      f"{flag}", flush=True)

    d = pd.read_csv(out_csv)
    ok = d[d.error.isna() | (d.error == "")]
    _write_json(os.path.join(run_dir, "summary.json"), {
        "created_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_slug": slug, "years": [2025],
        "n_selected": int(len(d)), "n_completed": int(len(ok)),
        "n_failed": int(len(d) - len(ok)),
        "elapsed_minutes": round((datetime.now(timezone.utc) - started).total_seconds() / 60, 2),
        "metrics": {
            "rating_mean": float(ok.rating.mean()) if len(ok) else None,
            "rating_sd": float(ok.rating.std()) if len(ok) else None,
            "calls_per_paper": {str(k): int(v) for k, v in ok.n_calls.value_counts().items()},
            "total_cost_usd": float(ok.cost_usd.sum()) if len(ok) else None,
            "median_elapsed_s": float(ok.elapsed_s.median()) if len(ok) else None,
        },
    })
    print(f"\n{len(ok)}/{len(d)} papers scored -> {out_csv}")
    if len(ok):
        print(f"rating: mean {ok.rating.mean():.2f} sd {ok.rating.std():.2f} "
              f"range {ok.rating.min()}-{ok.rating.max()}")
        print(f"calls: {ok.n_calls.value_counts().to_dict()}   "
              f"synthesis: {ok.text_synthesis.value_counts().to_dict()}   "
              f"papers with a retried call: {(ok.retried_calls > 0).sum()}")
        print(f"tokens/paper: {ok.prompt_tokens.median():,.0f} in, "
              f"{ok.completion_tokens.median():,.0f} out   "
              f"median {ok.elapsed_s.median():.0f}s/paper")
        print(f"extrapolated to 3,703 papers: "
              f"{ok.prompt_tokens.median() * 3703 / 1e6:,.0f}M in, "
              f"{ok.completion_tokens.median() * 3703 / 1e6:,.0f}M out")


if __name__ == "__main__":
    main()
