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

PAPER SET AND TEXT SOURCE ARE ARGUMENTS, not baked in. An earlier version took
`--year {2020,2025}` and read the ReviewArena parquet, because I had concluded that
2018 and 2019 had no local full text. That was WRONG: it is in data/OpenReview/, in
two run directories laid out as fulltext/<paper_id>.txt, covering 4,497 of the 4,567
papers (98.5%), rejects included, median 49,123 chars — the same text the 9-call
pipeline consumed. I reached the wrong conclusion by globbing for local_fulltext.json
under */papers/*/, the layout pipeline_xray documents, and never opening the directory
that had it. A hardcoded `choices=[2020, 2025]` is how that mistake became structural,
so both inputs are now arguments:

    --papers PATH        CSV of papers to score; needs a paper_id column, other
                         columns pass through. outputs/eval_table.csv works as-is.
    --text-dir DIR       repeatable; directories of <paper_id>.txt
    --text-parquet-year  the ReviewArena backend, for 2025

Exactly one text source. The resolved population is written out, so which papers a
run actually covered is recoverable from disk rather than re-derived.

NORMALIZATION. to_archive_text() strips markdown heading markers so ReviewArena text
matches the archive's shape. The archive .txt files carry zero such markers, so it is
a no-op on them — one code path, and both regimes read the same bytes.

Everything is traced: one JSONL line per call with the verbatim messages, the untouched
reply, token counts and latency. Both outputs are append-only and a restart skips
paper_ids that already SUCCEEDED, so a crash costs nothing already paid for.

Outputs (run = --run-name, default the papers-file stem):
  outputs/samples/single_call_{run}_papers.csv    resolved population, written once
  outputs/single_call_{run}_{model}.csv           one row per paper, incremental
  outputs/single_call_{run}_{model}_traces.jsonl  one line per call, incremental

Run: ARCHIVE=data/OpenReview
     python src/probes/run_single_call_baseline.py --dry-run \
       --papers outputs/eval_table.csv \
       --text-dir $ARCHIVE/rdd_bandwidth_2018_2020__gemma4_dedicated_stage1/fulltext \
       --text-dir $ARCHIVE/full_2018_2020_remaining/fulltext
     # ... then --n 2 for a smoke test, then --workers 8 for the full set
"""
import glob
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
from build.build_slim_2025_papers import MIN_CHARS, SEED
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

# Same column list as build_committee_ratings_2025.py, so anything already reading
# committee output reads this unchanged.
FIELDS = ["paper_id", "decision", "primary_area", "markdown_chars", "run_order",
          "model", "rating", "confidence", "soundness", "presentation", "contribution",
          "recommendation", "n_calls", "text_synthesis", "prompt_tokens",
          "completion_tokens", "cost_usd", "retried_calls", "elapsed_s", "error", "ts"]

_lock = threading.Lock()


def _slug(s):
    """Filesystem-safe tag: an endpoint id carries slashes and dots, which would
    otherwise scatter outputs across invented directories."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def sample_path(run):
    return f"outputs/samples/single_call_{_slug(run)}_papers.csv"


def paths(run, model_key):
    return (f"outputs/single_call_{_slug(run)}_{_slug(model_key)}.csv",
            f"outputs/single_call_{_slug(run)}_{_slug(model_key)}_traces.jsonl")


class DirText:
    """Text as <dir>/<paper_id>.txt, the archive's layout. Several dirs may be given;
    a paper present in more than one resolves to the largest file, on the assumption
    that a truncated extraction is the defective copy."""

    def __init__(self, dirs):
        self.index = {}
        for d in dirs:
            for path in glob.glob(os.path.join(d, "*.txt")):
                pid = os.path.basename(path)[:-4]
                size = os.path.getsize(path)
                if size > self.index.get(pid, (None, -1))[1]:
                    self.index[pid] = (path, size)
        self.describe = f"{len(dirs)} fulltext dir(s): " + ", ".join(dirs)

    def sizes(self):
        return {k: v[1] for k, v in self.index.items()}

    def get(self, pid):
        e = self.index.get(pid)
        if not e:
            return ""
        with open(e[0], errors="replace") as f:
            return f.read()


class ParquetText:
    """The ReviewArena backend, keyed by forum_id, for years it covers."""

    def __init__(self, year):
        from build.build_slim_2025_papers import load_year
        d = load_year(year).set_index("forum_id")
        self.md = d.markdown
        self.describe = f"ReviewArena parquet, year {year}"

    def sizes(self):
        return self.md.fillna("").str.len().to_dict()

    def get(self, pid):
        v = self.md.get(pid)
        return "" if v is None or not isinstance(v, str) else v


def build_population(papers_csv, text, run, seed=SEED, min_chars=MIN_CHARS):
    """Papers to score, resolved against the text we actually have.

    Written out once and reused. Drops are counted and printed rather than silently
    absorbed: a run that covers 4,100 of 4,567 papers is a different measurement from
    one that covers all of them, and the difference should not have to be rediscovered
    later from the output row count.
    """
    out = sample_path(run)
    if os.path.exists(out):
        s = pd.read_csv(out)
        print(f"Reusing resolved population {out} — {len(s):,} papers "
              f"(delete it to rebuild)")
        return s

    df = pd.read_csv(papers_csv, low_memory=False)
    if "paper_id" not in df.columns:
        sys.exit(f"{papers_csv} has no paper_id column")
    n_in = len(df)
    sizes = text.sizes()
    df["markdown_chars"] = df.paper_id.map(sizes)
    missing = int(df.markdown_chars.isna().sum())
    df = df[df.markdown_chars.notna()]
    short = int((df.markdown_chars < min_chars).sum())
    df = df[df.markdown_chars >= min_chars].copy()

    if "run_order" not in df.columns:
        # seeded, so --n 5 is a reproducible smoke test rather than whatever order
        # the input file happened to be in (which is decision-correlated)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df["run_order"] = range(len(df))
    keep = [c for c in ("paper_id", "title", "year", "decision", "primary_area",
                        "markdown_chars", "run_order") if c in df.columns]
    df = df[keep].sort_values("run_order")

    os.makedirs("outputs/samples", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{papers_csv}: {n_in:,} papers in -> {len(df):,} scoreable "
          f"({missing:,} no text, {short:,} under {min_chars:,} chars)")
    if "decision" in df.columns:
        acc = df.decision.astype(str).str.startswith("Accept")
        print(f"  {int(acc.sum()):,} accept / {int((~acc).sum()):,} reject")
    if "year" in df.columns:
        print("  by year: " + "  ".join(f"{int(y)}: {n:,}"
                                        for y, n in df.year.value_counts().sort_index().items()))
    print(f"  text source: {text.describe}\n  -> {out}")
    return df


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


def one_paper(row, markdown, model_key, year_default, fcsv, writer, ftr, base_model=None):
    # the run spans several years, so the prompt's {year} comes from the paper, not
    # from a flag. Telling the model a 2018 submission is a 2020 one changes what it
    # is being asked to judge, and would do so silently.
    year = int(row.get("year") or year_default)
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


def make_text(a):
    if a.text_dir and a.text_parquet_year:
        sys.exit("give --text-dir or --text-parquet-year, not both")
    if a.text_dir:
        missing = [d for d in a.text_dir if not os.path.isdir(d)]
        if missing:
            sys.exit("no such text dir(s): " + ", ".join(missing))
        return DirText(a.text_dir)
    if a.text_parquet_year:
        return ParquetText(a.text_parquet_year)
    sys.exit("a text source is required: --text-dir DIR (repeatable) "
             "or --text-parquet-year N")


def main(a):
    text = make_text(a)
    run = a.run_name or os.path.splitext(os.path.basename(a.papers))[0]
    pop = build_population(a.papers, text, run)

    if a.dry_run:
        row = pop.sort_values("run_order").iloc[0]
        msgs = build_messages(row.get("title"), text.get(row.paper_id),
                              int(row.get("year") or a.year_label))
        print(f"\n=== DRY RUN — {row.paper_id} "
              f"({row.get('decision', 'n/a')}) — no API call ===")
        for m in msgs:
            print(f"\n--- {m['role']} ({len(m['content']):,} chars) ---")
            print(m["content"][:1200])
            if len(m["content"]) > 1200:
                print(f"... [{len(m['content'])-1200:,} more chars]")
        return

    out_csv, out_traces = paths(run, a.model)
    done = done_ids(out_csv)
    todo = pop.sort_values("run_order")
    if a.n:
        todo = todo.head(a.n)
    todo = todo[~todo.paper_id.astype(str).isin(done)]
    print(f"{len(done):,} already done, {len(todo):,} to go "
          f"-> {out_csv}  (1 call each, model={a.model})")
    if todo.empty:
        return

    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fcsv, open(out_traces, "a") as ftr:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(one_paper, r._asdict(), text.get(r.paper_id),
                              a.model, a.year_label, fcsv, writer, ftr, a.base_model)
                    for r in todo.itertuples(index=False)]
            ok = bad = 0
            t0 = time.time()
            for i, f in enumerate(as_completed(futs), 1):
                rec = f.result()
                ok, bad = ok + (rec["error"] == ""), bad + (rec["error"] != "")
                if i % 25 == 0 or i == len(futs):
                    rate = i / max(time.time() - t0, 1e-9) * 60
                    left = (len(futs) - i) / max(rate, 1e-9)
                    print(f"  {i:,}/{len(futs):,}  ok={ok:,} failed={bad:,}  "
                          f"{rate:.1f}/min  ~{left:.0f} min left", flush=True)
    print(f"Done. {ok:,} scored, {bad:,} failed -> {out_csv}")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="outputs/eval_table.csv",
                    help="CSV of papers to score; needs a paper_id column")
    ap.add_argument("--text-dir", action="append", metavar="DIR",
                    help="directory of <paper_id>.txt; repeatable")
    ap.add_argument("--text-parquet-year", type=int, metavar="YEAR",
                    help="ReviewArena parquet backend instead of --text-dir")
    ap.add_argument("--run-name", default=None,
                    help="tag for output filenames; default is the papers-file stem")
    ap.add_argument("--year-label", type=int, default=2020,
                    help="fallback {year} for the prompt when a paper has no year column")
    ap.add_argument("--model", default="gemma", help="registry key or a Together id")
    ap.add_argument("--n", type=int, default=0, help="first N by run_order; 0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-model", default=None,
                    help="what a dedicated endpoint serves, for cost accounting")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble and print one prompt, make no API calls")
    return ap.parse_args()


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

    # the text backends are the part that just changed, so exercise them on disk
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "aaa.txt"), "w").write("x" * 5000)
        open(os.path.join(d, "bbb.txt"), "w").write("y" * 10)
        t = DirText([d])
        assert t.sizes() == {"aaa": 5000, "bbb": 10}
        assert t.get("aaa").startswith("x") and t.get("nope") == ""
        pop_csv = os.path.join(d, "papers.csv")
        pd.DataFrame({"paper_id": ["aaa", "bbb", "ccc"],
                      "decision": ["Accept", "Reject", "Reject"]}).to_csv(pop_csv, index=False)
        run = "unit_test_tmp"
        try:
            pop = build_population(pop_csv, t, run)
            # bbb is under the char floor, ccc has no text at all: both must drop,
            # and the drop must be visible rather than inferred from a short output
            assert list(pop.paper_id) == ["aaa"], list(pop.paper_id)
            assert "run_order" in pop.columns
        finally:
            if os.path.exists(sample_path(run)):
                os.remove(sample_path(run))

    for yr in (2018, 2019, 2020):
        m = build_messages("T", "body", yr)
        assert f"ICLR {yr}" in m[0]["content"], f"prompt year not {yr}"

    print("ok — prompts assemble per-paper year, schema is the council's, "
          "parity constants intact, text backend resolves and drops are explicit")


if __name__ == "__main__":
    a = cli()
    demo() if (len(sys.argv) == 1) else main(a)
