"""
Committee pipeline X-ray: one paper's run, end to end, at full granularity.

Strictly an observation tool. Every value shown is read from disk and displayed as-is —
nothing here computes, summarises, or infers. If a number looks wrong, the pipeline
produced it, not this page.

That constraint is the point. Reading these artifacts by hand already corrected two
things we believed about our own pipeline and could not see in any aggregate output:

  - production runs make 8 LLM calls, not 9, because `contribution_extraction` is
    deliberately skipped for Together-served gemma
  - production sections were a single "Full Document" blob, not parsed sections — the
    review stages never saw a sectioned paper at all

Both were invisible in eval_table.csv. This page is how they stay visible.

Works on any run directory laid out like the archive's:

    <run_dir>/run_manifest.json, summary.json
    <run_dir>/papers/<paper_id>/{input,coarse_review,coarse_call_traces,
                                 decision_packet,deepseek_decision,paper_result,
                                 local_fulltext}.json
                               /persona_reviews/<slug>.json

so it serves the 2018-2020 Dropbox logs and future 2025 runs, not just the smoke test.

Read-only: this module never writes.
"""
import os
import glob
import json

import pandas as pd
import streamlit as st

# Any directory matching the archive layout, wherever it lives: data/ holds runs that
# arrived from elsewhere (the 2018-2020 Dropbox logs), outputs/runs/ holds ours. Globbed
# rather than hardcoded so a new run appears without a code change.
#
# Anchored to the repo, not the cwd. As bare relative globs these matched nothing
# whenever Streamlit was launched from anywhere but the repo root, and the page then
# reported "no run directories found" with four of them sitting on disk.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_abs = lambda rel: os.path.join(REPO, rel)
_rel = lambda p: os.path.relpath(p, REPO)      # for display: paths stay readable

RUN_GLOBS = [_abs("data/*/run_manifest.json"), _abs("outputs/runs/*/run_manifest.json")]

# This page is for reading ONE paper closely, not for browsing a corpus. The 2025 run
# holds 3,632 papers and a dropdown of that length is slow and useless. So the default
# is a short curated list — one worked example per era and sample — with a checkbox to
# fall back to the full list when someone genuinely needs a specific paper.
#
# Entries are (label, run_dir, paper_id). Missing ones are skipped rather than erroring,
# so this list can name papers we do not have yet (the RDD-sample run is still on the
# collaborator's Dropbox).
EXAMPLES = [
    ("2018-2020 · RDD bandwidth sample",
     _abs("data/rdd_bandwidth_2018_2020_gemma4_gptoss20b"), None),
    ("2018-2020 · remaining (non-RDD) sample",
     _abs("data/full_2018_2020_remaining_gemma4_gptoss20b_smoke"), "r1AMITFaW"),
    # These two are the SAME paper on two models, so the pair is a direct comparison:
    # gemma skips contribution_extraction (8 calls), gpt-oss-120b does not (9).
    ("2025 · ICLR accepted — gemma (8 calls)",
     _abs("outputs/runs/iclr2025_gemma_pilot"), "b0WpXBABdu"),
    ("2025 · same paper — gpt-oss-120b (9 calls)",
     _abs("outputs/runs/smoke_oss120_1"), "b0WpXBABdu"),
    ("2025 · full run (3,632 papers) — first paper",
     _abs("outputs/runs/iclr2025_gemma_full"), None),
]

SCORE_FIELDS = ["rating", "confidence", "soundness", "presentation", "contribution"]


def _s(v):
    """Render any JSON value as text, so mixed-type columns stay displayable."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)


def _load(path, default=None):
    """Read a JSON artifact, or return `default`. Missing files are normal — a failed
    paper has no decision packet — and must not blank the whole page."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _find_runs():
    return sorted({os.path.dirname(p) for g in RUN_GLOBS for p in glob.glob(g)})


def _kv(d, keys):
    """Two-column table of selected keys, preserving the order asked for."""
    # values are deliberately stringified: these tables mix floats, lists and strings
    # in one column, which Arrow cannot type, and the point here is to display exactly
    # what is on disk rather than to compute on it
    rows = [{"field": k, "value": _s(d.get(k))} for k in keys if k in d]
    return pd.DataFrame(rows)


def _render_call(trace):
    """One call slot. A skipped stage still occupies a slot, and says so."""
    idx = trace.get("call_index")
    stage = trace.get("stage", "?")
    if trace.get("skipped"):
        st.markdown(f"**{idx}. `{stage}` — SKIPPED**")
        st.caption(trace.get("reason", "no reason recorded"))
        st.caption("Occupies a call slot so indices line up; no model call was made.")
        return

    head = (f"{idx}. `{stage}` — {trace.get('response_schema','?')} · "
            f"max_tokens={trace.get('max_tokens')} · temp={trace.get('temperature')} · "
            f"{trace.get('prompt_chars','?')}→{trace.get('response_chars','?')} chars")
    if trace.get("cost_usd") is not None:
        head += f" · ${trace['cost_usd']:.5f}"
    if trace.get("error"):
        head += "  ⚠ ERROR"

    with st.expander(head):
        if trace.get("error"):
            st.error(trace["error"])
        st.caption(f"model: `{trace.get('model','?')}`")

        msgs = trace.get("messages") or []
        for m in msgs:
            st.markdown(f"**{m.get('role','?')} prompt** "
                        f"({len(str(m.get('content','')))} chars)")
            st.code(str(m.get("content", "")), language="markdown")
        if not msgs:
            st.caption("no messages recorded")

        # raw_content is what the model actually emitted, before parsing — for a
        # thinking model that is most of the evidence. Older runs may not have it.
        if trace.get("raw_content"):
            st.markdown("**raw completion (pre-parse)**")
            st.code(trace["raw_content"], language="text")

        st.markdown("**parsed response**")
        st.json(trace.get("response") or {"(none)": None}, expanded=False)


def render():
    st.title("Committee Pipeline X-Ray")
    st.caption("One paper, every step. Everything below is read from disk verbatim — "
               "this page computes nothing.")

    runs = _find_runs()
    if not runs:
        st.warning("No run directories found matching " +
                   " or ".join(f"`{_rel(g)}`" for g in RUN_GLOBS) +
                   ". A run folder needs a `run_manifest.json` at its root.")
        return

    # curated examples first; only resolve the ones that actually exist on disk
    available = []
    for label, rdir, pid in EXAMPLES:
        if pid is None:
            found = sorted(glob.glob(os.path.join(rdir, "papers", "*")))
            pid = os.path.basename(found[0]) if found else None
        if pid and os.path.isdir(os.path.join(rdir, "papers", pid)):
            available.append((label, rdir, pid))

    browse = st.checkbox("Browse all papers instead", value=not available,
                         help="The curated list covers one worked example per era. "
                              "Turn this on to pick any paper from any run.")

    if not browse and available:
        labels = [a[0] for a in available]
        choice = st.radio("Example", labels, horizontal=False)
        _, run_dir, paper_id = available[labels.index(choice)]
        missing = [e[0] for e in EXAMPLES if e[0] not in labels]
        if missing:
            st.caption("Not on disk yet: " + "; ".join(missing))
    else:
        run_dir = st.selectbox(
            "Run", runs,
            format_func=lambda p: f"{os.path.basename(p)}  ({_rel(p).split(os.sep)[0]}/)")
        papers = sorted(os.path.basename(p) for p in
                        glob.glob(os.path.join(run_dir, "papers", "*")) if os.path.isdir(p))
        if not papers:
            st.warning(f"No paper directories under `{run_dir}/papers/`.")
            return
        paper_id = st.selectbox(f"Paper ({len(papers)} in this run)", papers)

    st.caption(f"`{_rel(run_dir)}/papers/{paper_id}`")
    pdir = os.path.join(run_dir, "papers", paper_id)

    manifest = _load(os.path.join(run_dir, "run_manifest.json"), {})
    summary = _load(os.path.join(run_dir, "summary.json"), {})
    inp = _load(os.path.join(pdir, "input.json"), {})
    review = _load(os.path.join(pdir, "coarse_review.json"), {})
    traces = (_load(os.path.join(pdir, "coarse_call_traces.json"), {}) or {}).get("call_traces", [])
    packet = _load(os.path.join(pdir, "decision_packet.json"), {})
    head = _load(os.path.join(pdir, "deepseek_decision.json"), {})
    result = _load(os.path.join(pdir, "paper_result.json"), {})
    fulltext = _load(os.path.join(pdir, "local_fulltext.json"), {})

    # headline numbers, so the reader knows what they are looking at before scrolling
    c = st.columns(5)
    c[0].metric("LLM calls", review.get("llm_calls", "—"))
    c[1].metric("Committee rating", review.get("rating", "—"))
    c[2].metric("Head decision", (head.get("parsed") or {}).get("decision", "—"))
    c[3].metric("True decision", inp.get("decision", "—"))
    c[4].metric("Human mean rating", inp.get("mean_rating", "—"))

    # ---- 1. configuration ------------------------------------------------
    st.header("1 · Run configuration")
    st.caption("What was requested, before any paper was touched.")
    a, b = st.columns(2)
    with a:
        st.dataframe(_kv(manifest, [
            "run_slug", "created_at_utc", "years", "committee_model", "committee_bias",
            "personas", "persona_weights", "timeout_seconds", "max_retries",
            "max_parallel_papers", "head_temperature", "head_top_p", "head_max_tokens",
        ]), hide_index=True, use_container_width=True)
    with b:
        st.markdown("**decision head**")
        st.json(manifest.get("decision_head_model") or {}, expanded=True)
        st.markdown("**run summary**")
        st.json({k: v for k, v in summary.items() if k != "metrics"}, expanded=False)
        if summary.get("metrics"):
            st.json(summary["metrics"], expanded=False)

    # ---- 2. input --------------------------------------------------------
    st.header("2 · Input")
    a, b = st.columns([2, 1])
    with a:
        st.dataframe(_kv(inp, [
            "paper_id", "title", "year", "decision", "accepted", "mean_rating",
            "score_centered", "cutoff", "bandwidth", "primary_area", "keywords",
            "abstract_char_count", "abstract_word_count", "fulltext_available",
        ]), hide_index=True, use_container_width=True)
    with b:
        st.markdown("**full-text acquisition**")
        st.json({"status": fulltext.get("status"),
                 **(fulltext.get("download_meta") or {})}, expanded=True)
        st.caption(f"pdf: `{fulltext.get('pdf_path','—')}`")
    with st.expander("Abstract as supplied"):
        st.code(inp.get("abstract", ""), language="markdown")

    # ---- 3. what the pipeline saw ---------------------------------------
    st.header("3 · What the pipeline saw")
    st.caption("The deterministic structures the prompts were built from. The section "
               "list is the pipeline's own parse — if it reads `Full Document`, the "
               "review stages received the whole paper as one untyped blob.")
    inv = review.get("structural_inventory") or packet.get("structural_inventory") or {}
    if inv:
        st.json(inv, expanded=False)
    else:
        st.caption("no structural inventory recorded")

    # ---- 4. the call slots ----------------------------------------------
    st.header(f"4 · Call slots ({len(traces)} recorded, "
              f"{sum(1 for t in traces if t.get('skipped'))} skipped)")
    st.caption("Verbatim prompts and responses, in the order the pipeline issued them.")
    if traces:
        st.dataframe(pd.DataFrame([{
            "#": t.get("call_index"), "stage": t.get("stage"),
            "schema": t.get("response_schema"), "skipped": bool(t.get("skipped")),
            "max_tokens": t.get("max_tokens"), "temp": t.get("temperature"),
            "prompt_chars": t.get("prompt_chars"), "resp_chars": t.get("response_chars"),
            "cost_usd": t.get("cost_usd"),
        } for t in traces]).astype(object).where(lambda d: d.notna(), None).astype(str),
            hide_index=True, use_container_width=True)
    for t in traces:
        _render_call(t)

    # ---- 5. personas -----------------------------------------------------
    st.header("5 · Persona reviewers")
    st.caption("Four independent reviews of the same paper. Spread across them is the "
               "signal the single committee score collapses.")
    prows, ptexts = [], {}
    for p in sorted(glob.glob(os.path.join(pdir, "persona_reviews", "*.json"))):
        j = _load(p, {})
        if not j:
            continue
        slug = j.get("persona_slug", os.path.basename(p).replace(".json", ""))
        prows.append({"persona": slug,
                      **{f: j.get(f) for f in SCORE_FIELDS},
                      "recommendation": j.get("recommendation")})
        ptexts[slug] = j
    if prows:
        pdf_ = pd.DataFrame(prows).set_index("persona")
        num = pdf_[SCORE_FIELDS].apply(pd.to_numeric, errors="coerce")
        st.dataframe(pdf_.assign(**{f: num[f] for f in SCORE_FIELDS}),
                     use_container_width=True)
        st.caption("range per dimension (max − min): " +
                   ", ".join(f"{f} {num[f].max() - num[f].min():g}" for f in SCORE_FIELDS))
        for slug, j in ptexts.items():
            with st.expander(f"{slug} — full review text"):
                for f in ("summary", "strength", "weaknesses", "questions", "rationale"):
                    if j.get(f):
                        st.markdown(f"**{f}**")
                        st.write(j[f])
    else:
        st.caption("no persona reviews on disk")

    if (review.get("committee") or {}).get("aggregate_scores"):
        st.markdown("**aggregate scores (committee)**")
        st.json(review["committee"]["aggregate_scores"], expanded=False)
    if packet.get("disagreement"):
        with st.expander("disagreement statistics (as handed to the decision head)"):
            st.json(packet["disagreement"], expanded=False)

    # ---- 6. committee synthesis -----------------------------------------
    st.header("6 · Committee synthesis")
    text_synth = (review.get("committee") or {}).get("text_synthesis")
    if text_synth:
        st.caption(f"text_synthesis mode: **{text_synth}** — `fallback` means the "
                   "synthesis call failed and scores were aggregated arithmetically.")
    st.dataframe(_kv(review, SCORE_FIELDS + ["recommendation", "llm_calls",
                                             "review_cost_usd"]),
                 hide_index=True, use_container_width=True)
    for f in ("summary", "strength", "weaknesses", "questions", "rationale"):
        if review.get(f):
            with st.expander(f"synthesised {f}"):
                st.write(review[f])
    md_path = os.path.join(pdir, "coarse_review.md")
    if os.path.exists(md_path):
        with st.expander("coarse_review.md (rendered review as written to disk)"):
            st.markdown(open(md_path).read())

    # ---- 7. decision packet ---------------------------------------------
    st.header("7 · Decision packet")
    st.caption("What the decision head is given — note this is a separate section "
               "selection from the review stages'.")
    if packet.get("selected_sections"):
        st.dataframe(pd.DataFrame(packet["selected_sections"]).astype(str), hide_index=True,
                     use_container_width=True)
    if packet.get("feature_vector"):
        fv = packet["feature_vector"]
        st.markdown(f"**feature vector** ({len(fv)} features)")
        st.dataframe(pd.DataFrame([{"feature": k, "value": _s(v)} for k, v in fv.items()]),
                     hide_index=True, use_container_width=True, height=300)

    # ---- 8. decision head ------------------------------------------------
    st.header("8 · Decision head")
    if head:
        a, b = st.columns(2)
        with a:
            st.dataframe(_kv(head, ["decision_head_model", "decision_head_label",
                                    "finish_reason", "elapsed_seconds",
                                    "estimated_cost_usd", "http_error"]),
                         hide_index=True, use_container_width=True)
        with b:
            st.json(head.get("usage") or {}, expanded=True)
        st.markdown("**parsed decision**")
        st.json(head.get("parsed") or {}, expanded=True)
        with st.expander("system message"):
            st.code(head.get("system_message", ""), language="markdown")
        with st.expander(f"user message ({len(head.get('user_message',''))} chars)"):
            st.code(head.get("user_message", ""), language="markdown")
        with st.expander("raw response (pre-parse)"):
            st.code(head.get("raw_response", ""), language="text")
    else:
        st.caption("no decision-head artifact for this paper")

    # ---- 9. final row ----------------------------------------------------
    st.header("9 · Final result row")
    st.caption("The row that reaches the analysis tables. Everything above collapses "
               "into this.")
    if result:
        st.dataframe(pd.DataFrame([{"field": k, "value": _s(v)} for k, v in result.items()]),
                     hide_index=True, use_container_width=True, height=400)
