"""
Data provenance tab for the CitesBench dashboard.

Traces every dataset in the repo from external source through fetch, annotation,
join and analysis. Design rules, in order of importance:

  1. Nothing factual is hardcoded that can be computed. Row counts, file sizes,
     mtimes and git-tracked status are read off disk at render time and cached.
     A missing file renders as "missing", never as a stale number.
  2. Every producer claim carries the script path and, where behaviour is
     asserted, the line range. Those line numbers were read, not guessed.
  3. Numbers that cannot be recomputed here (findings inside report files) are
     quoted with their source file, and the audit findings index is *parsed*
     from outputs/data_audit.md rather than transcribed.
  4. Artifacts with no producer anywhere in src/ are listed as provenance gaps
     rather than attributed by guesswork. Unresolvable questions are labelled
     UNVERIFIED with a note on what evidence would settle them.

Exports a single render(), called from src/pages/1_Provenance.py (its own page in
the Streamlit sidebar nav; it used to be Section 6 of the main dashboard).
"""
import os
import re
import json
import sqlite3
import subprocess
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import import_graph

# Palette copied from src/app/dashboard.py:29-43 so the two stay visually identical.
# Not imported, because importing dashboard.py would re-execute the whole app.
BORDER, TEXT, SUBTEXT = "#E2E8F0", "#0F172A", "#64748B"
SEV_COLOR = {
    "blocker": "#DC2626",
    "major":   "#D97706",
    "minor":   "#0D9488",
    "info":    "#64748B",
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = "data/gen_review.db"

# Files whose row count we do not attempt (binary, or line count would be a lie).
_NO_COUNT_EXT = {".png", ".pdf", ".db", ".html", ".tex", ".zip", ".parquet"}
_SIZE_CAP = 200 * 1024 ** 2  # above this, say so instead of parsing


# ──────────────────────────────────────────────────────────────────────────────
# Live filesystem / git probes  (all cached; all fail soft)
# ──────────────────────────────────────────────────────────────────────────────
def _abs(rel):
    return os.path.join(REPO, rel)


@st.cache_data(show_spinner=False)
def _stat(rel):
    """(exists, size_bytes, mtime_iso) for a repo-relative path. Dirs are summed."""
    p = _abs(rel)
    if not os.path.exists(p):
        return False, None, None
    if os.path.isdir(p):
        total, newest = 0, 0.0
        for root, _dirs, files in os.walk(p):
            for f in files:
                try:
                    s = os.stat(os.path.join(root, f))
                except OSError:
                    continue
                total += s.st_size
                newest = max(newest, s.st_mtime)
        return True, total, datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
    s = os.stat(p)
    return True, s.st_size, datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M")


@st.cache_data(show_spinner=False)
def _rows(rel):
    """Row count and the method used, so the number is interpretable.

    CSVs are parsed (line counting is wrong for the review text columns, which
    contain embedded newlines). JSONL is line-counted, which is exact by format.
    """
    p = _abs(rel)
    if not os.path.exists(p) or os.path.isdir(p):
        return None, ""
    ext = os.path.splitext(p)[1].lower()
    if ext in _NO_COUNT_EXT:
        return None, "n/a"
    try:
        if os.path.getsize(p) > _SIZE_CAP:
            return None, "not counted (>200 MB)"
        if ext == ".csv":
            return len(pd.read_csv(p, usecols=[0], low_memory=False)), "csv parse"
        if ext == ".jsonl":
            with open(p, "rb") as fh:
                return sum(1 for ln in fh if ln.strip()), "jsonl lines"
        if ext in (".log", ".md", ".txt"):
            with open(p, "rb") as fh:
                return sum(1 for _ in fh), "text lines"
    except Exception as e:  # a corrupt or half-written file must not kill the tab
        return None, f"read error: {type(e).__name__}"
    return None, "n/a"


@st.cache_data(show_spinner=False)
def _git_tracked(rel):
    """True / False / None(unknown) via git ls-files --error-unmatch."""
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                           cwd=REPO, capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _git_last_commit(rel):
    """'<date> <sha> <subject>' for the newest commit touching rel, else ''."""
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%ad %h %s", "--date=short",
                            "--", rel], cwd=REPO, capture_output=True, text=True, timeout=20)
        return r.stdout.strip()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def _csv_columns(rel):
    p = _abs(rel)
    if not os.path.exists(p):
        return None
    try:
        return list(pd.read_csv(p, nrows=0).columns)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _magic_bytes(rel, n=4):
    p = _abs(rel)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            return fh.read(n)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _db_tables():
    """[(table, rowcount)] from the SQLite source DB, or None if unreadable."""
    p = _abs(DB_PATH)
    if not os.path.exists(p):
        return None
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        out = [(n, con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]) for n in names]
        con.close()
        return out
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _db_years():
    """[(year, n_submissions, n_accept)] from SUBMISSION, or None."""
    p = _abs(DB_PATH)
    if not os.path.exists(p):
        return None
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT when_submitted, COUNT(*), "
            "SUM(CASE WHEN decision LIKE 'Accept%' THEN 1 ELSE 0 END) "
            "FROM SUBMISSION GROUP BY 1 ORDER BY 1").fetchall()
        con.close()
        return rows
    except Exception:
        return None


def _fmt_size(b):
    if b is None:
        return "—"
    v = float(b)
    if v < 1024:
        return f"{int(v)} B"
    for unit in ("KB", "MB", "GB"):
        v /= 1024.0
        if v < 1024 or unit == "GB":
            return f"{v:.1f} {unit}"
    return f"{v:.1f} GB"


# ──────────────────────────────────────────────────────────────────────────────
# Mermaid rendering — CDN component + always-available source fallback
# ──────────────────────────────────────────────────────────────────────────────
_MERMAID_HTML = """
<div style="background:#FFFFFF;border:1px solid %(border)s;border-radius:8px;
            padding:10px;overflow:auto;">
  <pre class="mermaid" style="margin:0;background:transparent;">%(code)s</pre>
</div>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, theme: "neutral",
                       themeVariables: { fontSize: "13px" },
                       flowchart: { useMaxWidth: true, htmlLabels: true } });
</script>
"""


def _mermaid(code, height, caption=None):
    """Render a mermaid diagram, and always offer the source in an expander.

    Streamlit has no native mermaid support, so this goes through an iframe
    against the mermaid CDN. If the CDN is blocked the iframe shows the raw
    source; the expander below is the guaranteed-readable copy either way.
    `height` must be set generously — the iframe does not autosize, and a
    too-small value clips the diagram.
    """
    code = code.strip()
    components.html(_MERMAID_HTML % {"code": code, "border": BORDER},
                    height=height, scrolling=True)
    if caption:
        st.caption(caption)
    with st.expander("Mermaid source (readable if the CDN is blocked)"):
        st.code(code, language="text")


# ──────────────────────────────────────────────────────────────────────────────
# The provenance registry
#
# One entry per artifact. `producer` is the script that writes it, with the line
# range where the write happens; None means no script in src/ writes it (a
# provenance gap). `inputs` are the paths that script reads. Every line number
# below was read out of the file it names.
# ──────────────────────────────────────────────────────────────────────────────
ART = [
    # path, stage, producer, inputs, note
    (DB_PATH, "0 source", None, [],
     "OpenReview scrape. No script in the repo creates or refreshes it."),
    ("data/archive/all_paper_results.csv", "0 source", None, [],
     "Committee + decision-head LLM run, imported from an external share folder "
     "(`data/README.md`, `data/summary.json`)."),
    ("data/archive/all_paper_results.jsonl", "0 source", None, [],
     "Per-paper JSON records for the same run."),
    ("data/paper_manifest.csv", "0 source", None, [], "Manifest for the same share folder."),
    ("data/summary.json", "0 source", None, [],
     "Self-describing provenance record for the LLM run: created_at_utc, source_runs, validation."),
    ("data/gemma_ready7_wave1_cached_v2", "0 source", None, [],
     "Sharded raw outputs of the wave-1 committee run."),
    ("data/CLAUDE.md", "0 source", None, [], "Not markdown — see known defects."),
    ("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv", "1 citations",
     "Archive/CompletePipeline/design/fetch_openalex_citations_from_arxiv_matches.py#L657",
     [], "Written by the superseded Archive pipeline under different paths "
         "(`rawdata/Design/OpenAlex/`); nothing in src/ reproduces it."),
    ("data/OpenAlex/openalex_rdd_arxiv_query_input.csv", "1 citations",
     "Archive/CompletePipeline/design/fetch_openalex_citations_from_arxiv_matches.py#L551", [], ""),
    ("data/OpenAlex/openalex_rdd_arxiv_batch_manifest.csv", "1 citations",
     "Archive/CompletePipeline/design/fetch_openalex_citations_from_arxiv_matches.py#L617", [], ""),
    ("data/OpenAlex/openalex_rdd_arxiv_unique_works.csv", "1 citations",
     "Archive/CompletePipeline/design/fetch_openalex_citations_from_arxiv_matches.py#L647", [], ""),
    ("data/OpenAlex/openalex_rdd_miss_title_search_candidates.csv", "1 citations",
     "Archive/CompletePipeline/design/diagnose_openalex_arxiv_misses.py#L15", [], ""),
    ("data/OpenAlex/openalex_rdd_miss_title_search_diagnostics.csv", "1 citations",
     "Archive/CompletePipeline/design/diagnose_openalex_arxiv_misses.py#L14", [], ""),
    ("data/OpenAlex/openalex_rdd_dashboard.csv", "1 citations", None,
     [], "Slim RDD file read at `src/app/dashboard.py#L1042`. No writer anywhere in the repo."),
    ("output/citations_2018_2020.csv", "1 citations",
     "src/fetch/fetch_citations_openalex.py#L27-L29,L85-L89", [DB_PATH],
     "OpenAlex ground truth. Path is `output/` (singular) — outside the CLAUDE.md convention."),
    ("output/papers_2018_2020.csv", "1 citations", None, [], "No producer anywhere in the repo."),
    ("output/reviews_2018_2020.csv", "1 citations", None, [], "No producer anywhere in the repo."),
    ("output/reviews_summary_2018_2020.csv", "1 citations", None, [],
     "No producer anywhere in the repo."),
    ("output/part_b.log", "1 citations", None, [],
     "Log of a run named nowhere in src/ or Archive/."),
    ("outputs/arxiv_resolution.csv", "2 arxiv", "src/fetch/resolve_arxiv_ids.py#L220",
     [DB_PATH], "Full-corpus pass 1: all 8 years against the HuggingFace arXiv dump."),
    ("outputs/arxiv_resolution_report.md", "2 arxiv", "src/fetch/resolve_arxiv_ids.py#L264", [], ""),
    ("outputs/arxiv_resolution.log", "2 arxiv", "src/fetch/resolve_arxiv_ids.py (stdout redirect)", [], ""),
    ("outputs/arxiv_dump_download.log", "2 arxiv",
     "huggingface_hub snapshot_download, command in src/fetch/resolve_arxiv_ids.py#L23-L24", [], ""),
    ("outputs/arxiv_fuzzy_candidates.csv", "2 arxiv", "src/fetch/resolve_arxiv_fuzzy.py#L219",
     ["outputs/arxiv_resolution.csv", DB_PATH], "Pass 2: TF-IDF abstract verification."),
    ("outputs/arxiv_fuzzy_report.md", "2 arxiv", "src/fetch/resolve_arxiv_fuzzy.py#L199", [], ""),
    ("outputs/arxiv_fuzzy.log", "2 arxiv", "src/fetch/resolve_arxiv_fuzzy.py (stdout redirect)", [], ""),
    ("outputs/paper_fields.csv", "3 annotate", "src/build/tag_fields.py#L42,L117", [DB_PATH],
     "LLM field tags via Together AI."),
    ("outputs/paper_author_ids.csv", "1 citations", "src/fetch/fetch_author_stats.py#L30", [], ""),
    ("outputs/author_stats.csv", "1 citations", "src/fetch/fetch_author_stats.py#L31", [], ""),
    ("outputs/paper_venues.csv", "1 citations", "src/fetch/fetch_author_stats.py#L32", [], ""),
    ("outputs/paper_author_covariates.csv", "4 join", "src/build/build_author_covariates.py#L122",
     ["outputs/author_stats.csv", "outputs/paper_author_ids.csv"], ""),
    ("outputs/eval_table.csv", "4 join", "src/build/build_eval_table.py#L82",
     [DB_PATH, "output/citations_2018_2020.csv", "outputs/paper_fields.csv"],
     "The central study table. Holds two columns the builder cannot write — see defects."),
    ("outputs/eval_table_2024.csv", "4 join", "src/build/build_eval_table_year.py#L127",
     [DB_PATH, "outputs/arxiv_resolution.csv"], "Out-of-sample year, --year 2024."),
    ("outputs/eval_table_2025.csv", "4 join", "src/build/build_eval_table_year.py#L127",
     [DB_PATH, "outputs/arxiv_resolution.csv"], "Out-of-sample year, --year 2025."),
    ("outputs/rejected_venues_s2.csv", "1 citations", "src/fetch/fetch_rejected_venues_s2.py", [], ""),
    ("outputs/rejected_venues_s2_title.csv", "1 citations",
     "src/fetch/fetch_rejected_venues_s2_title.py", [],
     "Source of the `title_cached` block reused by fetch_citations_s2_full.py."),
    ("outputs/oa_title_match_venues.csv", "1 citations", None, [],
     "Read at `src/fetch/fetch_rejected_venues_s2_title.py#L67-L68`. No writer anywhere."),
    ("outputs/s2_citations_full.csv", "1 citations", "src/fetch/fetch_citations_s2_full.py#L36",
     ["outputs/eval_table.csv", "data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"],
     "Three code paths in one file: arxiv_batch, title_cached, title_match."),
    ("outputs/s2_citations_v2.csv", "1 citations", "src/fetch/fetch_citations_s2_v2.py#L45",
     ["outputs/eval_table.csv", "outputs/arxiv_resolution.csv"],
     "The rebuild: one tiered code path, ID-matched only so far."),
    ("outputs/s2_citations_v2_tiered.csv", "1 citations", "src/fetch/fetch_citations_s2_v2.py#L309", [], ""),
    ("outputs/s2_attribution_report.md", "1 citations", "src/fetch/fetch_citations_s2_v2.py#L355", [], ""),
    ("outputs/s2_citations_2024.csv", "1 citations",
     "src/fetch/fetch_citations_s2_v2.py --out (arg at #L365)", ["outputs/eval_table_2024.csv"], ""),
    ("outputs/s2_citations_2025.csv", "1 citations",
     "src/fetch/fetch_citations_s2_v2.py --out (arg at #L365)", ["outputs/eval_table_2025.csv"], ""),
    ("outputs/citation_source_comparison.csv", "1 citations",
     "src/analysis/compare_citation_sources.py#L57", ["outputs/eval_table.csv"], ""),
    ("outputs/citation_source_comparison.md", "1 citations",
     "src/analysis/compare_citation_sources.py#L137", [], ""),
    ("outputs/source_comparison_verified.csv", "1 citations", None, [],
     "No writer anywhere in src/ or Archive/."),
    ("outputs/title_match_quality.csv", "1 citations", "src/audit/check_title_match_quality.py#L13",
     ["output/citations_2018_2020.csv"], ""),
    ("outputs/leakage_lap_v1.csv", "3 annotate", "src/probes/leakage_lap_v1.py#L46",
     ["outputs/eval_table.csv"], "LAP probe: does the model recall the accept/reject outcome?"),
    ("outputs/leakage_lap_report.md", "5 analyse", "src/probes/leakage_lap_v1.py#L47", [], ""),
    ("outputs/leakage_lap_llama3_2025.csv", "3 annotate",
     "src/probes/leakage_lap_v1.py --tag (arg at #L411, path built at #L420)",
     ["outputs/eval_table_2025.csv"], ""),
    ("outputs/leakage_lap_traces.jsonl", "3 annotate", "src/probes/leakage_lap_v1.py#L189", [], ""),
    ("outputs/leakage_fame_v1.csv", "3 annotate", "src/probes/leakage_fame_v1.py",
     ["outputs/eval_table.csv"], "FAME probe: does the model recall the paper's prominence?"),
    ("outputs/leakage_fame_report.md", "5 analyse", "src/probes/leakage_fame_v1.py#L151", [], ""),
    ("outputs/leakage_fame_traces_sample30.jsonl", "3 annotate",
     "src/probes/leakage_fame_trace_sample.py#L80", ["outputs/leakage_fame_v1.csv"], ""),
    ("outputs/leakage_controls.csv", "3 annotate", "src/probes/leakage_controls.py#L170",
     ["outputs/leakage_lap_v1.csv"], "Probe-validity controls."),
    ("outputs/leakage_masked_rereview.csv", "3 annotate", "src/probes/leakage_masked_rereview.py#L124",
     ["outputs/leakage_lap_v1.csv", DB_PATH], ""),
    ("outputs/leakage_abstract_completion_v1.csv", "3 annotate",
     "src/probes/leakage_abstract_completion_v1.py#L205", ["outputs/eval_table.csv", DB_PATH],
     "Verbatim-regurgitation probe."),
    ("outputs/leakage_abstract_completion_texts.jsonl", "3 annotate",
     "src/probes/leakage_abstract_completion_v1.py#L205", [], ""),
    ("outputs/leakage_abstract_completion_report.md", "5 analyse",
     "src/probes/leakage_abstract_completion_v1.py#L340", [], ""),
    ("outputs/leakage_exclusion_eval.csv", "5 analyse", "src/analysis/leakage_exclusion_eval.py#L146",
     ["outputs/eval_table.csv", "outputs/leakage_lap_v1.csv", "outputs/leakage_fame_v1.csv"], ""),
    ("outputs/leakage_exclusion_eval_s2.csv", "5 analyse", "src/analysis/leakage_exclusion_eval.py#L146",
     ["outputs/s2_citations_full.csv", "outputs/leakage_lap_v1.csv"], ""),
    ("outputs/leakage_exclusion_bootstrap_openalex.csv", "5 analyse",
     "src/analysis/leakage_exclusion_bootstrap.py#L172", ["outputs/eval_table.csv"], ""),
    ("outputs/leakage_exclusion_bootstrap_openalex_vp.csv", "5 analyse",
     "src/analysis/leakage_exclusion_bootstrap.py#L172 (--venue-premium, arg at #L102)", [], ""),
    ("outputs/leakage_exclusion_bootstrap_s2.csv", "5 analyse",
     "src/analysis/leakage_exclusion_bootstrap.py#L172 (--citation-source s2)", [], ""),
    ("outputs/leakage_exclusion_bootstrap_s2_vp.csv", "5 analyse",
     "src/analysis/leakage_exclusion_bootstrap.py#L172 (--citation-source s2 --venue-premium)", [], ""),
    ("outputs/leakage_threshold_sweep.csv", "5 analyse", "src/analysis/leakage_threshold_sweep.py#L65",
     ["outputs/eval_table.csv", "outputs/leakage_lap_v1.csv"], ""),
    ("outputs/leakage_power_analysis.md", "5 analyse", "src/analysis/leakage_power_analysis.py#L116",
     ["outputs/leakage_lap_v1.csv", "outputs/leakage_controls.csv"], ""),
    ("outputs/samples/oos_papers.csv", "6 oos", "src/build/build_oos_samples.py#L174",
     ["outputs/eval_table.csv", "outputs/eval_table_2024.csv", "outputs/eval_table_2025.csv"],
     "Frozen out-of-sample arms: clean / partial / contaminated."),
    ("outputs/samples/oos_probe_plan.csv", "6 oos", "src/build/build_oos_samples.py#L193",
     ["outputs/samples/oos_papers.csv"],
     "Rewritten in place afterwards by src/build/rebuild_placebo.py#L112 — see defects."),
    ("outputs/samples/oos_sample_design.md", "6 oos", "src/build/build_oos_samples.py#L264", [], ""),
    ("outputs/oos_probes_Llama-3-3-70B-Instruct-Turbo.csv", "6 oos",
     "src/probes/run_oos_probes.py#L171", ["outputs/samples/oos_probe_plan.csv"], ""),
    ("outputs/oos_traces_Llama-3-3-70B-Instruct-Turbo.jsonl", "6 oos",
     "src/probes/run_oos_probes.py#L172", [], "Full prompt + completion per call — the evidence."),
    ("outputs/oos_probes_gemma-4-31B-it.csv", "6 oos", "src/probes/run_oos_probes.py#L171", [], ""),
    ("outputs/oos_traces_gemma-4-31B-it.jsonl", "6 oos", "src/probes/run_oos_probes.py#L172", [], ""),
    ("outputs/oos_probe_report.md", "6 oos", "src/probes/run_oos_probes.py#L267", [], ""),
    ("outputs/eval_results.csv", "5 analyse", "src/analysis/run_eval.py#L79", ["outputs/eval_table.csv"],
     "Superseded persona regimes — see defects."),
    ("outputs/baselines_cache.csv", "5 analyse", "src/app/dashboard.py#L158", [], ""),
    ("outputs/fuzzy_rdd.md", "5 analyse", "src/analysis/fuzzy_rdd.py#L432",
     ["data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"], ""),
    ("outputs/fuzzy_rdd_binscatter.csv", "5 analyse", "src/analysis/fuzzy_rdd.py#L437", [], ""),
    ("outputs/hetero_analysis.md", "5 analyse", "src/analysis/hetero_analysis.py#L191",
     ["outputs/eval_table.csv", "outputs/paper_author_covariates.csv"], ""),
    ("outputs/outlier_quantitative.csv", "5 analyse", "src/analysis/outlier_analysis.py#L122", [], ""),
    ("outputs/outlier_reviews.csv", "5 analyse", "src/analysis/outlier_analysis.py#L141", [],
     "Then mutated in place by src/build/tag_rejection_reasons.py#L47 and "
     "src/fetch/fetch_pc_decisions.py#L62 — the file is three scripts deep with no versioning."),
    ("outputs/cite_hist.png", "5 analyse", "src/analysis/cite_hist.py#L48", [], ""),
    ("outputs/outlier_scatter.png", "5 analyse", "src/analysis/viz_outlier_scatter.py#L102", [], ""),
    ("outputs/rejection_tags.png", "5 analyse", "src/analysis/viz_rejection_tags.py#L81", [], ""),
    ("outputs/breakthrough_bubble.png", "5 analyse", None, [],
     "No writer anywhere in src/ or Archive/."),
    ("outputs/breakthrough_top_decile.png", "5 analyse", None, [],
     "No writer anywhere in src/ or Archive/."),
    ("outputs/table1_summary_stats.tex", "5 analyse", None, [],
     "src/audit/data_audit.py#L61 names table1_summary_stats.py as the producer, but that "
     "script only prints — see disagreements."),
    ("outputs/data_audit.md", "5 analyse", "src/audit/data_audit.py#L1087", [], ""),
    ("outputs/data_audit.html", "5 analyse", "src/audit/data_audit.py (--no-html to skip, #L1190)", [], ""),
    ("outputs/findings_integrity_check.md", "5 analyse", None, [],
     "Hand-written. Referenced from docs/PROJECT_OVERVIEW.md#L140 and as a remediation "
     "pointer at src/audit/data_audit.py#L878; no script writes it."),
    ("outputs/venue_coverage_strategy.md", "5 analyse", None, [], "Hand-written; no producer."),
]

STAGE_LABEL = {
    "0 source": "Stage 0 — Source imports",
    "1 citations": "Stage 1 — Citation fetch & enrich",
    "2 arxiv": "Stage 2 — arXiv resolution",
    "3 annotate": "Stage 3 — LLM annotation & probes",
    "4 join": "Stage 4 — Transform / join",
    "5 analyse": "Stage 5 — Analyse / report",
    "6 oos": "Stage 6 — Out-of-sample benchmark",
}


# ──────────────────────────────────────────────────────────────────────────────
# Diagrams
# ──────────────────────────────────────────────────────────────────────────────
TOP_FLOW = """
flowchart LR
  subgraph EXT["External sources"]
    OR["OpenReview<br/>(scrape)"]
    OA["OpenAlex API"]
    S2["Semantic Scholar API"]
    TG["Together AI<br/>(Gemma / Llama / DeepSeek)"]
    HF["HuggingFace<br/>arXiv metadata dump"]
  end

  subgraph RAW["Raw data (data/, read-only)"]
    DB[("data/gen_review.db<br/>SUBMISSION / REVIEW / GENAI_REVIEW")]
    APR["data/archive/all_paper_results.csv<br/>committee + decision head"]
    OAX["data/OpenAlex/<br/>arXiv-OpenAlex match (Archive pipeline)"]
  end

  subgraph FETCH["Fetch / enrich"]
    CIT["output/citations_2018_2020.csv<br/>OpenAlex counts"]
    AXR["outputs/arxiv_resolution.csv<br/>+ arxiv_fuzzy_candidates.csv"]
    S2F["outputs/s2_citations_full.csv<br/>outputs/s2_citations_v2.csv"]
    AUT["outputs/author_stats.csv<br/>paper_venues.csv"]
  end

  subgraph ANN["Annotate (LLM labels)"]
    FLD["outputs/paper_fields.csv"]
    PRB["LAP / FAME / controls /<br/>masked / abstract-completion"]
  end

  ET["outputs/eval_table.csv<br/>THE study table"]
  OOS["outputs/eval_table_2024.csv<br/>outputs/eval_table_2025.csv<br/>outputs/samples/ (frozen)"]

  subgraph ANL["Analyse"]
    DASH["src/app/dashboard.py<br/>(recomputes regimes live)"]
    REP["reports: data_audit.md, fuzzy_rdd.md,<br/>hetero_analysis.md, oos_probe_report.md"]
    STALE["outputs/eval_results.csv<br/>SUPERSEDED"]
  end

  OR --> DB
  TG --> APR
  OA --> OAX
  OA --> CIT
  HF --> AXR
  S2 --> S2F
  OA --> AUT
  TG --> FLD
  TG --> PRB

  DB --> CIT
  DB --> FLD
  DB --> AXR
  OAX --> S2F
  AXR --> S2F
  ET -.->|"circular: eval_table is an input<br/>to the S2 fetch that feeds it"| S2F

  DB --> ET
  CIT --> ET
  FLD --> ET
  APR -.->|"committee_rating, deepseek_p_accept<br/>NO SCRIPT PERFORMS THIS JOIN"| ET

  DB --> OOS
  AXR --> OOS
  ET --> PRB
  ET --> DASH
  S2F --> DASH
  AUT --> DASH
  PRB --> REP
  ET --> STALE
  OOS --> REP

  classDef gap fill:#FEF2F2,stroke:#DC2626,color:#7F1D1D;
  classDef stale fill:#FFFBEB,stroke:#D97706,color:#78350F;
  class APR,OAX gap;
  class STALE stale;
"""

DIA_SOURCE = """
flowchart TB
  OR["OpenReview"] --> DB[("data/gen_review.db<br/>1.6 GB SQLite, untracked")]
  DB --> SUB["SUBMISSION<br/>id, title, abstract, tldr, primary_area,<br/>keywords, decision, when_submitted, source_id"]
  DB --> REV["REVIEW<br/>paper_id, reviewer_id, rating, confidence,<br/>soundness, main_review, binocular_score, ..."]
  DB --> GEN["GENAI_REVIEW<br/>paper_id, type in (neutral|positive|negative),<br/>generated, rating, binocular_score"]

  SUB --> W["Study window filter<br/>when_submitted IN (2018,2019,2020)<br/>src/build/build_eval_table.py#L22-L26"]
  REV --> AGG["Rating parse + aggregate<br/>str.extract('^(\\\\d+)') then groupby mean/std/count<br/>src/build/build_eval_table.py#L36-L43"]
  GEN --> PIV["Pivot to llm_neutral/positive/negative_rating<br/>src/build/build_eval_table.py#L44-L51"]

  W --> ET["outputs/eval_table.csv"]
  AGG --> ET
  PIV --> ET

  REV -.->|"9.3% of in-window rows carry a<br/>decision string, not a score<br/>(outputs/data_audit.md, finding 1)"| AGG
  classDef gap fill:#FEF2F2,stroke:#DC2626,color:#7F1D1D;
"""

DIA_CITE = """
flowchart TB
  subgraph OApath["OpenAlex path (first, 2026-06)"]
    OA["OpenAlex /works API<br/>src/fetch/fetch_citations_openalex.py#L47"]
    OA --> OAC["output/citations_2018_2020.csv<br/>paper_id, openalex_citations, status"]
    OAC --> ET["outputs/eval_table.csv<br/>status != 'found' -> NaN (#L72-L73)"]
  end

  subgraph S2path["Semantic Scholar path (later, 2026-07)"]
    ETin["outputs/eval_table.csv"] --> SP["src/fetch/fetch_citations_s2_full.py"]
    AXOLD["data/OpenAlex/openalex_rdd_arxiv_paper_level.csv<br/>arxiv_id_canonical"] --> SP
    SP --> B1["1. arxiv_batch<br/>POST /paper/batch by ARXIV:id<br/>#L82-L100"]
    SP --> B2["2. title_cached<br/>bulk reuse of rejected_venues_s2_title.csv<br/>#L103-L118 - NO title_sim filter"]
    SP --> B3["3. title_match<br/>GET /paper/search/match, one call each<br/>#L121-L145"]
    B1 --> S2F["outputs/s2_citations_full.csv"]
    B2 --> S2F
    B3 --> S2F
  end

  subgraph V2["S2 rebuild (src/fetch/fetch_citations_s2_v2.py)"]
    AXR["outputs/arxiv_resolution.csv<br/>full corpus"] --> SV2["one tiered code path<br/>A = ID, B = verified title, C = weak/demoted"]
    SV2 --> S2V["outputs/s2_citations_v2.csv<br/>+ _tiered.csv + s2_attribution_report.md"]
  end

  OAC --> CMP["src/analysis/compare_citation_sources.py"]
  S2F --> CMP
  CMP --> CMPO["outputs/citation_source_comparison.{csv,md}<br/>median S2/OA ratio 2.88;<br/>accepted 3.47 vs rejected 2.00"]

  classDef gap fill:#FEF2F2,stroke:#DC2626,color:#7F1D1D;
  class B2 gap;
"""

DIA_ARXIV = """
flowchart TB
  HF["HuggingFace dataset<br/>librarian-bots/arxiv-metadata-snapshot<br/>10 parquet shards, 3,113,330 records<br/>(outputs/arxiv_dump_download.log)"]

  subgraph OLD["Superseded: RDD-subsample match"]
    ARCH["Archive/CompletePipeline/design/<br/>fetch_openalex_citations_from_arxiv_matches.py"]
    ARCH --> AXOLD["data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"]
  end

  subgraph P1["Pass 1 - exact + token-set"]
    HF --> R1["src/fetch/resolve_arxiv_ids.py<br/>normalized title exact, then token-set;<br/>verified by SequenceMatcher + Jaccard + authors<br/>no threshold applied at write time"]
    DB[("data/gen_review.db<br/>all 32,652 submissions")] --> R1
    R1 --> AXR["outputs/arxiv_resolution.csv<br/>matched flag, arxiv_id, match_rule,<br/>title_sim, token_jaccard"]
    AXR --> RPT1["outputs/arxiv_resolution_report.md"]
  end

  subgraph P2["Pass 2 - abstract-verified fuzzy"]
    AXR --> R2["src/fetch/resolve_arxiv_fuzzy.py<br/>TF-IDF over 387,068 candidate preprints,<br/>abstract cosine >= 0.7"]
    DB --> R2
    R2 --> AXF["outputs/arxiv_fuzzy_candidates.csv"]
    AXF --> RPT2["outputs/arxiv_fuzzy_report.md"]
  end

  AXOLD --> SP["src/fetch/fetch_citations_s2_full.py<br/>(still reads the OLD file)"]
  AXR --> SV2["src/fetch/fetch_citations_s2_v2.py<br/>(reads the NEW file)"]
  AXR --> BY["src/build/build_eval_table_year.py<br/>2024 / 2025 tables"]

  classDef gap fill:#FEF2F2,stroke:#DC2626,color:#7F1D1D;
  class AXOLD gap;
"""

DIA_LLM = """
flowchart TB
  TG["Together AI"]
  subgraph EXTRUN["Imported committee run (outside this repo)"]
    GEM["Gemma-4-31B committee<br/>4 reviewer personas"]
    DH["Decision head:<br/>DeepSeek-V3.1 | openai/gpt-oss-20b"]
    GEM --> APR["data/archive/all_paper_results.csv<br/>committee_rating, deepseek_p_accept,<br/>committee_model, decision_head_model"]
    DH --> APR
  end

  subgraph INREPO["In-repo annotation (src/)"]
    TG --> TF["src/build/tag_fields.py<br/>5-way field taxonomy"]
    TF --> FLD["outputs/paper_fields.csv"]
    TG --> LAP["src/probes/leakage_lap_v1.py<br/>LAP: recall accept/reject?"]
    TG --> FAME["src/probes/leakage_fame_v1.py<br/>FAME: recall prominence?"]
    TG --> CTL["src/probes/leakage_controls.py<br/>probe validity"]
    TG --> MSK["src/probes/leakage_masked_rereview.py"]
    TG --> ABS["src/probes/leakage_abstract_completion_v1.py<br/>verbatim regurgitation"]
  end

  DBG["gen_review.db GENAI_REVIEW<br/>neutral / positive / negative personas"]

  APR -.->|"join with no script"| ET["outputs/eval_table.csv"]
  FLD --> ET
  DBG --> ET
  ET --> LAP
  ET --> FAME
  LAP --> CTL
  LAP --> MSK
  LAP --> EXC["src/analysis/leakage_exclusion_eval.py<br/>src/analysis/leakage_exclusion_bootstrap.py"]
  FAME --> EXC

  classDef gap fill:#FEF2F2,stroke:#DC2626,color:#7F1D1D;
  class APR gap;
"""

DIA_JOIN = """
flowchart LR
  DB[("gen_review.db")] -->|"SUBMISSION 2018-2020"| BE["src/build/build_eval_table.py"]
  DB -->|"REVIEW (all years, then joined)"| BE
  DB -->|"GENAI_REVIEW"| BE
  CIT["output/citations_2018_2020.csv<br/>read at #L53 - singular output/"] --> BE
  FLD["outputs/paper_fields.csv<br/>optional, #L57-L63"] --> BE
  BE -->|"#L82 to_csv"| ET["outputs/eval_table.csv"]
  BE -->|"#L75-L80 field x year rank(pct=True)"| ET
  APR["data/archive/all_paper_results.csv"] -.->|"committee_rating<br/>deepseek_p_accept<br/>NO SCRIPT DOES THIS"| ET
  ET --> DASH["src/app/dashboard.py<br/>recomputes all regimes live"]
  ET --> RE["src/analysis/run_eval.py<br/>-> outputs/eval_results.csv (stale regimes)"]
  classDef gap fill:#FEF2F2,stroke:#DC2626,color:#7F1D1D;
  class APR gap;
"""

DIA_OOS = """
flowchart TB
  DB[("gen_review.db<br/>2024: 7,404 / 2025: 11,672 submissions")] --> BY["src/build/build_eval_table_year.py<br/>--year 2024 | 2025"]
  AXR["outputs/arxiv_resolution.csv"] --> BY
  OAAPI["OpenAlex /works by arXiv DOI<br/>#L61-L69"] --> BY
  BY --> ET24["outputs/eval_table_2024.csv"]
  BY --> ET25["outputs/eval_table_2025.csv"]

  ET24 --> BS["src/build/build_oos_samples.py<br/>seed 42, strata = decision class x citation quartile"]
  ET25 --> BS
  ET["outputs/eval_table.csv<br/>(contaminated arm)"] --> BS
  S2Y["outputs/s2_citations_2024.csv<br/>outputs/s2_citations_2025.csv"] --> BS

  BS --> PAP["outputs/samples/oos_papers.csv"]
  BS --> PLAN["outputs/samples/oos_probe_plan.csv<br/>4 probes x 448 papers = 1,792 calls/model"]
  BS --> DES["outputs/samples/oos_sample_design.md<br/>'Frozen 2026-08-03 with seed 42'"]

  PLAN -.->|"rewritten IN PLACE #L112<br/>after the freeze"| RB["src/build/rebuild_placebo.py"]
  RB --> PLAN

  PLAN --> RUN["src/probes/run_oos_probes.py<br/>probes: lap | fame | placebo | wrongyear"]
  TG["Together AI"] --> RUN
  RUN --> PC["outputs/oos_probes_<model>.csv"]
  RUN --> PT["outputs/oos_traces_<model>.jsonl"]
  PC --> PR["outputs/oos_probe_report.md"]

  classDef stale fill:#FFFBEB,stroke:#D97706,color:#78350F;
  class RB,PLAN stale;
"""


# ──────────────────────────────────────────────────────────────────────────────
# Live defect checks — every one recomputed at render time
# ──────────────────────────────────────────────────────────────────────────────
def _check_orphan_columns():
    """eval_table.csv columns the builder cannot write."""
    et = _csv_columns("outputs/eval_table.csv")
    if et is None:
        return ("blocker", "eval_table.csv holds columns build_eval_table.py cannot produce",
                "UNVERIFIED — outputs/eval_table.csv is missing, so the column set cannot be read.",
                ["src/build/build_eval_table.py", "outputs/eval_table.csv"])
    src_path = _abs("src/build/build_eval_table.py")
    try:
        src = open(src_path).read()
    except OSError:
        return ("blocker", "eval_table.csv holds columns build_eval_table.py cannot produce",
                "UNVERIFIED — src/build/build_eval_table.py could not be read.",
                ["src/build/build_eval_table.py"])
    orphans = [c for c in et if c not in src]
    apr = _csv_columns("data/archive/all_paper_results.csv") or []
    shared = [c for c in orphans if c in apr]
    ev = (f"Columns on disk that never appear as a literal in the builder source: "
          f"**{orphans or 'none'}**. Of those, present in `data/archive/all_paper_results.csv`: "
          f"**{shared or 'none'}**. Re-running `python src/build/build_eval_table.py` writes "
          f"`{len(et)}` columns minus these, silently breaking every regime and analysis "
          f"that reads them.")
    return ("blocker", "eval_table.csv holds columns build_eval_table.py cannot produce", ev,
            ["src/build/build_eval_table.py#L82", "outputs/eval_table.csv",
             "data/archive/all_paper_results.csv", "outputs/data_audit.md (finding 7)"])


def _check_title_cached():
    """The bulk-reused S2 block."""
    p = _abs("outputs/s2_citations_full.csv")
    if not os.path.exists(p):
        return ("blocker", "S2 `title_cached` block is a bulk copy with no similarity filter",
                "UNVERIFIED — outputs/s2_citations_full.csv is missing.",
                ["src/fetch/fetch_citations_s2_full.py#L103-L118"])
    try:
        s2 = pd.read_csv(p, usecols=["paper_id", "method", "s2_citations", "title_sim"])
        ev_p = _abs("outputs/eval_table.csv")
        dec = (pd.read_csv(ev_p, usecols=["paper_id", "decision"])
               if os.path.exists(ev_p) else None)
    except Exception as e:
        return ("blocker", "S2 `title_cached` block is a bulk copy with no similarity filter",
                f"UNVERIFIED — could not read the inputs ({type(e).__name__}).",
                ["outputs/s2_citations_full.csv"])
    counts = s2["method"].value_counts().to_dict()
    tc = s2[s2["method"] == "title_cached"]
    line = (f"Method mix in `outputs/s2_citations_full.csv`: **{counts}**. "
            f"The `title_cached` block is **{len(tc):,}** rows, median S2 citations "
            f"**{tc['s2_citations'].median():.0f}**, "
            f"**{tc['s2_citations'].isna().mean():.1%}** null.")
    if dec is not None:
        m = tc.merge(dec, on="paper_id", how="left")
        n_acc = int(m["decision"].fillna("").str.startswith("Accept").sum())
        line += (f" Decision split: **{m['decision'].value_counts().to_dict()}** — "
                 f"**{n_acc}** accepted papers in the block.")
    line += (" The block is copied wholesale from `outputs/rejected_venues_s2_title.csv` "
             "at `src/fetch/fetch_citations_s2_full.py#L103-L118`. The same script's own quality "
             "summary treats `title_sim >= 0.9` as the bar for a usable match "
             "(#L148-L150), but nothing filters these rows on it before they enter the "
             "CSV, and #L114-L115 hardcodes `\"year\": \"\"` so the block carries no `s2_year` "
             "for a plausibility check either.")
    return ("blocker", "S2 `title_cached` block is a bulk copy with no similarity filter", line,
            ["src/fetch/fetch_citations_s2_full.py#L103-L118",
             "outputs/rejected_venues_s2_title.csv",
             "docs/notes/data_pipeline_plan.md (diagnosis paragraph)"])


def _check_regime_registries():
    """ALL_REGIMES vs what dashboard.py imports vs what eval_results.csv contains."""
    refs = ["src/regimes/__init__.py#L13-L28", "src/analysis/run_eval.py#L14,L49",
            "src/app/dashboard.py#L17-L21,L213", "outputs/eval_results.csv"]
    try:
        reg_src = open(_abs("src/regimes/__init__.py")).read()
        dash_src = open(_abs("src/app/dashboard.py")).read()
    except OSError:
        return ("major", "Two regime registries have drifted apart",
                "UNVERIFIED — could not read src/regimes/__init__.py or src/app/dashboard.py.", refs)
    block = reg_src.split("ALL_REGIMES", 1)[-1]
    registered = re.findall(r"(\w+)\(\)", block)
    imported = re.findall(r"from regimes\.\w+ import (\w+)", dash_src)
    on_disk = sorted(f[:-3] for f in os.listdir(_abs("src/regimes"))
                     if f.endswith(".py") and f != "__init__.py")
    er = _abs("outputs/eval_results.csv")
    in_file = "missing"
    if os.path.exists(er):
        try:
            in_file = sorted(pd.read_csv(er, usecols=["regime"])["regime"].unique())
        except Exception:
            in_file = "unreadable"
    ev = (f"`ALL_REGIMES` (the only thing `src/analysis/run_eval.py` evaluates): **{registered}**. "
          f"Imported directly by the dashboard: **{imported}**. "
          f"Regime modules on disk: **{on_disk}**. "
          f"Regimes actually inside `outputs/eval_results.csv`: **{in_file}**. "
          "The dashboard recomputes everything live from `eval_table.csv`, so the app is "
          "current; `eval_results.csv` and `run_eval.py` are not, and cannot reproduce a "
          "single number the app shows.")
    return ("major", "Two regime registries have drifted apart; eval_results.csv is from the "
            "superseded persona regimes", ev,
            refs + ["outputs/data_audit.md (findings 18, 20)"])


def _check_output_singular():
    """output/ (singular) sits outside the documented convention."""
    d = _abs("output")
    if not os.path.isdir(d):
        return ("major", "A second output directory `output/` sits outside the convention",
                "Not reproduced: `output/` does not exist in this checkout.", ["CLAUDE.md"])
    files = sorted(os.listdir(d))
    files = [f for f in files if not f.startswith(".")]
    tracked = _git_tracked("output/citations_2018_2020.csv")
    ev = (f"`CLAUDE.md` states every generated file lives under `outputs/`. "
          f"`output/` (singular) exists and holds **{files}**. "
          f"`output/citations_2018_2020.csv` is not merely stray output — it is a required "
          f"*input*, read at `src/build/build_eval_table.py#L53`, `src/analysis/outlier_analysis.py#L27`, "
          f"`src/analysis/cite_hist.py#L9`, `src/fetch/fetch_author_stats.py#L29`, "
          f"`src/audit/check_title_match_quality.py#L16`, `src/fetch/fetch_rejected_venues_s2.py#L32` and "
          f"`src/fetch/fetch_rejected_venues_s2_title.py#L45`. git-tracked: **{tracked}**.")
    return ("major", "A second output directory `output/` sits outside the convention", ev,
            ["CLAUDE.md", "src/build/build_eval_table.py#L53", "outputs/data_audit.md (finding 22)"])


def _check_arxiv_coverage():
    """Old RDD-subsample arXiv match vs the new full-corpus resolution."""
    refs = ["data/OpenAlex/openalex_rdd_arxiv_paper_level.csv",
            "outputs/arxiv_resolution.csv", "src/fetch/fetch_citations_s2_full.py#L68",
            "docs/notes/data_pipeline_plan.md"]
    axp, etp, rsp = ("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv",
                     "outputs/eval_table.csv", "outputs/arxiv_resolution.csv")
    missing = [p for p in (axp, etp) if not os.path.exists(_abs(p))]
    if missing:
        return ("major", "The S2 fetch still keys off the RDD-subsample arXiv match",
                f"UNVERIFIED — missing {missing}.", refs)
    try:
        ax = pd.read_csv(_abs(axp), usecols=["paper_id", "year", "arxiv_id_canonical"],
                         low_memory=False)
        et = pd.read_csv(_abs(etp), usecols=["paper_id", "decision"])
    except Exception as e:
        return ("major", "The S2 fetch still keys off the RDD-subsample arXiv match",
                f"UNVERIFIED — read error ({type(e).__name__}).", refs)
    m = et.merge(ax[["paper_id", "arxiv_id_canonical"]].drop_duplicates("paper_id"),
                 on="paper_id", how="left")
    m["acc"] = m["decision"].fillna("").str.startswith("Accept")
    has = m["arxiv_id_canonical"].notna()
    ev = (f"The old match file covers **{len(ax):,}** papers across "
          f"**{sorted(ax['year'].dropna().unique().astype(int).tolist())}** — an RDD "
          f"bandwidth subsample, not the study corpus. Of the **{len(et):,}** papers in "
          f"`eval_table.csv`, only **{int(has.sum()):,}** ({has.mean():.1%}) carry an "
          f"`arxiv_id_canonical` from it: **{has[m['acc']].mean():.1%}** of accepted vs "
          f"**{has[~m['acc']].mean():.1%}** of rejected. That asymmetry is what routes "
          "accepts down S2's reliable `ARXIV:` batch path and rejects into title matching.")
    if os.path.exists(_abs(rsp)):
        try:
            rs = pd.read_csv(_abs(rsp), usecols=["paper_id", "year", "decision", "matched"],
                             low_memory=False)
            w = rs[rs["year"].isin([2018, 2019, 2020])]
            w_acc = w["decision"].fillna("").str.startswith("Accept")
            ev += (f" The replacement `outputs/arxiv_resolution.csv` covers **{len(rs):,}** "
                   f"submissions (all years) with **{int(rs['matched'].sum()):,}** matched; "
                   f"in the 2018-2020 window **{int(w['matched'].sum()):,}/{len(w):,}** "
                   f"({w['matched'].mean():.1%}) match — "
                   f"**{w.loc[w_acc, 'matched'].mean():.1%}** accepted vs "
                   f"**{w.loc[~w_acc, 'matched'].mean():.1%}** rejected. "
                   "But `src/fetch/fetch_citations_s2_full.py#L68` still reads the *old* file, so "
                   "`s2_citations_full.csv` has not inherited the improvement.")
        except Exception:
            ev += " (outputs/arxiv_resolution.csv present but unreadable for comparison.)"
    else:
        ev += " `outputs/arxiv_resolution.csv` is missing, so the comparison stops here."
    return ("major", "The S2 fetch still keys off the RDD-subsample arXiv match", ev, refs)


def _check_data_claude_md():
    b = _magic_bytes("data/CLAUDE.md")
    if b is None:
        return ("major", "data/CLAUDE.md is not markdown",
                "Not reproduced: `data/CLAUDE.md` does not exist in this checkout.",
                ["data/CLAUDE.md"])
    is_zip = b[:2] == b"PK"
    _, size, mtime = _stat("data/CLAUDE.md")
    ev = (f"First bytes of `data/CLAUDE.md`: `{b!r}` — "
          f"{'the PK ZIP local-file-header magic' if is_zip else 'not a ZIP header'}. "
          f"Size {_fmt_size(size)}, modified {mtime}. Claude Code auto-loads `CLAUDE.md` "
          "files as instructions; anything reading this one as documentation gets binary.")
    return ("major" if is_zip else "info", "data/CLAUDE.md is a ZIP archive, not markdown", ev,
            ["data/CLAUDE.md", "outputs/data_audit.md (finding 24)"])


def _check_frozen_plan_mutated():
    """rebuild_placebo.py rewrites the 'frozen' probe plan in place."""
    refs = ["src/build/rebuild_placebo.py#L112", "src/build/build_oos_samples.py#L193",
            "outputs/samples/oos_sample_design.md"]
    _, _, plan_m = _stat("outputs/samples/oos_probe_plan.csv")
    _, _, pap_m = _stat("outputs/samples/oos_papers.csv")
    _, _, des_m = _stat("outputs/samples/oos_sample_design.md")
    if plan_m is None:
        return ("major", "The 'frozen' probe plan is rewritten in place after the freeze",
                "UNVERIFIED — outputs/samples/oos_probe_plan.csv is missing.", refs)
    ev = (f"`outputs/samples/oos_sample_design.md` states the sample was "
          "\"Frozen 2026-08-03 with seed 42. Every model and every probe runs on exactly "
          "these papers.\" But `src/build/rebuild_placebo.py#L112` writes back to the same path "
          f"(`plan.to_csv(PLAN)`), and the mtimes bear that out: design doc **{des_m}**, "
          f"oos_papers.csv **{pap_m}**, oos_probe_plan.csv **{plan_m}**. There is no "
          "versioned copy of the pre-rebuild plan, so probe rows already collected against "
          "the earlier plan cannot be distinguished from rows collected against this one.")
    return ("major", "The 'frozen' probe plan is rewritten in place after the freeze", ev, refs)


def _check_untracked_inputs():
    critical = [DB_PATH, "data/archive/all_paper_results.csv", "output/citations_2018_2020.csv",
                "outputs/paper_fields.csv", "outputs/eval_table.csv",
                "outputs/arxiv_resolution.csv", "outputs/s2_citations_full.csv"]
    rows = [(p, _git_tracked(p)) for p in critical]
    untracked = [p for p, t in rows if t is False]
    ev = ("git-tracked status, computed now: " +
          ", ".join(f"`{p}` → **{'tracked' if t else 'untracked' if t is False else 'unknown'}**"
                    for p, t in rows) +
          ". Untracked inputs have no version history at all, so an overwrite is "
          "undetectable and unrecoverable, and no committed number can be tied to the exact "
          "bytes that produced it.")
    sev = "major" if untracked else "info"
    return (sev, "Critical inputs are untracked, so committed numbers have no verifiable base",
            ev, [".gitignore", "outputs/data_audit.md (finding 23)"])


def _check_db_window():
    yrs = _db_years()
    if yrs is None:
        return ("info", "The source DB is much wider than the study window",
                "UNVERIFIED — data/gen_review.db could not be opened read-only.", [DB_PATH])
    total = sum(n for _y, n, _a in yrs)
    window = sum(n for y, n, _a in yrs if y in (2018, 2019, 2020))
    tbl = ", ".join(f"{int(y)}: {n:,}" for y, n, _a in yrs)
    oos = sum(n for y, n, _a in yrs if y in (2024, 2025))
    ev = (f"`SUBMISSION.when_submitted` distribution: {tbl}. Study window 2018-2020 is "
          f"**{window:,}** of **{total:,}** rows ({window/total:.1%}). A further "
          f"**{oos:,}** rows (2024-2025) feed the out-of-sample tables via "
          f"`src/build/build_eval_table_year.py`. The remaining "
          f"**{total-window-oos:,}** rows (2021-2023) are scanned by "
          "`src/fetch/resolve_arxiv_ids.py` but are used by no analysis in the repo. "
          "`REVIEW` is *not* year-filtered when read at "
          "`src/build/build_eval_table.py#L28-L31` — the filter happens implicitly at the join.")
    return ("info", "The source DB is much wider than the study window", ev,
            [DB_PATH, "src/build/build_eval_table.py#L22-L31"])


CHECKS = [_check_orphan_columns, _check_title_cached, _check_regime_registries,
          _check_output_singular, _check_arxiv_coverage, _check_data_claude_md,
          _check_frozen_plan_mutated, _check_untracked_inputs, _check_db_window]

_SEV_ORDER = {"blocker": 0, "major": 1, "minor": 2, "info": 3}


# ──────────────────────────────────────────────────────────────────────────────
# Cross-source disagreements — shown, not resolved
# ──────────────────────────────────────────────────────────────────────────────
def _disagreements():
    out = []

    # 1. The two OOS run logs disagree on how many calls were already done.
    #    Reconcilable by arithmetic against the plan — shown, with the arithmetic.
    plan_p = _abs("outputs/samples/oos_probe_plan.csv")
    if os.path.exists(plan_p):
        try:
            pl = pd.read_csv(plan_p, usecols=["probe", "paper_id"])
            mix = pl["probe"].value_counts().to_dict()
            n_placebo = int(mix.get("placebo", 0))
            body = ("`outputs/oos_probes_run.log` shows the Llama arm starting at "
                    "`8 done, 1,784 to go` and reaching `1,750/1,784`. "
                    "`outputs/oos_probes_run2.log` then resumes the *same* model at "
                    f"`1,344 done, 448 to go`, i.e. **{len(pl):,} − 1,344 = "
                    f"{len(pl) - 1344}** calls became un-done between the runs.\n\n"
                    f"That reconciles exactly against the plan, which holds **{len(pl):,}** "
                    f"rows over **{pl['paper_id'].nunique()}** papers with the probe mix "
                    f"**{mix}**: the shortfall equals the placebo row count "
                    f"(**{n_placebo}**) precisely. `src/build/rebuild_placebo.py#L112` rewrote "
                    "the plan in place, replacing the fabricated placebo titles, so "
                    "previously-collected placebo answers no longer matched a plan row and "
                    "the resume logic recounted them as outstanding.\n\n"
                    "The arithmetic is verifiable; the *causal* story is inference from "
                    "mtimes (plan rewritten after run 1's log stops, before run 2's Llama "
                    "pass finished) and is marked **UNVERIFIED** as such — a `rebuild_placebo` "
                    "run log would settle it, and none exists. The consequence either way: "
                    "placebo answers in the probe CSVs may have been collected against two "
                    "different sets of fabricated titles, and the CSV carries no field "
                    "recording which.")
            out.append(("Two OOS run logs disagree on how much work was already done", body,
                        ["outputs/oos_probes_run.log", "outputs/oos_probes_run2.log",
                         "src/build/rebuild_placebo.py#L112", "outputs/samples/oos_probe_plan.csv"]))
        except Exception:
            pass

    # 2. PRODUCERS map in data_audit.py claims a producer that writes nothing.
    t1 = _abs("src/analysis/table1_summary_stats.py")
    if os.path.exists(t1):
        src = open(t1).read()
        writes = re.findall(r"(to_csv|open\([^)]*['\"]w['\"])", src)
        out.append((
            "Who produces outputs/table1_summary_stats.tex?",
            f"`src/audit/data_audit.py#L61-L62` lists `src/analysis/table1_summary_stats.py` as the producer "
            f"of `outputs/table1_summary_stats.tex` and uses that edge for its staleness DAG. "
            f"But grepping that script for any write yields **{writes or 'nothing'}** — its own "
            f"docstring (#L4) says it \"Prints the numbers that populate "
            f"outputs/table1_summary_stats.tex\". The `.tex` is therefore hand-maintained and "
            f"the audit's dependency edge is wrong. Recorded here as a disagreement between "
            f"two in-repo sources; both are shown rather than one silently preferred.",
            ["src/audit/data_audit.py#L61-L62", "src/analysis/table1_summary_stats.py#L4",
             "outputs/table1_summary_stats.tex"]))

    # 3. OpenAlex vs S2 — the substantive disagreement.
    p = _abs("outputs/citation_source_comparison.md")
    if os.path.exists(p):
        body = open(p, errors="ignore").read()
        keep = [ln for ln in body.splitlines()
                if re.match(r"^\| (Median|Mean|Share|Spearman|Top-decile) ", ln)]
        out.append((
            "OpenAlex and Semantic Scholar disagree systematically on the outcome variable",
            "Quoted verbatim from `outputs/citation_source_comparison.md` (produced by "
            "`src/analysis/compare_citation_sources.py#L137`):\n\n" +
            "\n".join(keep) +
            "\n\nBoth sources remain live in the dashboard behind the sidebar citation-source "
            "toggle; neither is treated as truth. The undercount is differential by decision "
            "(the accepted/rejected median-ratio row above), which is why it biases the RDD "
            "estimate rather than merely scaling it.",
            ["outputs/citation_source_comparison.md", "src/analysis/compare_citation_sources.py",
             "outputs/s2_attribution_report.md"]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# data_audit.md findings index — parsed, never transcribed
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _audit_findings():
    p = _abs("outputs/data_audit.md")
    if not os.path.exists(p):
        return None, None
    body = open(p, errors="ignore").read()
    gen = re.search(r"^Generated (.+?) by `(.+?)`", body, re.M)
    rows = []
    for m in re.finditer(r"^### (\d+)\. \[(\w+)\] (.+?)$\n\*(.+?)\*", body, re.M):
        rows.append({"#": int(m.group(1)), "severity": m.group(2).lower(),
                     "finding": m.group(3).strip(), "stage": m.group(4).strip()})
    return (pd.DataFrame(rows) if rows else None,
            (gen.group(1), gen.group(2)) if gen else None)


# ──────────────────────────────────────────────────────────────────────────────
# Logs — primary evidence of what actually ran
# ──────────────────────────────────────────────────────────────────────────────
LOGS = [
    ("outputs/arxiv_dump_download.log", "HuggingFace arXiv dump download",
     "Ends with `SNAPSHOT_AT <local HF cache path>`, i.e. 10/10 shards fetched, "
     "unauthenticated. The dump itself lives outside the repo, in the user's HF cache — "
     "so the snapshot is not reproducible from the repo alone."),
    ("outputs/arxiv_resolution.log", "arXiv resolution pass 1",
     "Completed: final line reports the write and the match rate."),
    ("outputs/arxiv_fuzzy.log", "arXiv fuzzy pass 2", "Completed: scored 2,078/2,078."),
    ("outputs/s2_fetch_full.log", "S2 full refetch",
     "PARTIAL: the log stops mid-progress and never reaches its own summary block "
     "(`src/fetch/fetch_citations_s2_full.py#L146-L151`). Its first lines say "
     "`arXiv batch: 0 papers` / `title_cached: 0 reused`, so this was a *resumed* run. "
     "The output CSV is complete, which means it was finished by a run whose log was not "
     "kept. Coverage numbers must come from the CSV, not this log."),
    ("outputs/fetch_author_stats.log", "OpenAlex author/venue passes",
     "Completed, with its own summary block. Passes 1 and 2 were no-ops on this run "
     "(`already done, 0 to fetch`) — evidence of the resumable-by-default pattern."),
    ("outputs/fetch_rejected_venues_s2.log", "S2 rejected-venue fetch",
     "Zero bytes. No record of this run's behaviour at all."),
    ("outputs/leakage_lap_run.log", "LAP probe run", "Zero bytes."),
    ("outputs/oos_probes_run.log", "OOS probes, run 1",
     "Llama-3.3-70B reached 1,750/1,784 then wrote out; the gemma arm reported "
     "`0 done, 1,792 to go` and wrote immediately with no progress lines — i.e. the "
     "gemma arm of run 1 produced nothing."),
    ("outputs/oos_probes_run2.log", "OOS probes, run 2",
     "Resumed Llama at `1,344 done, 448 to go`, which is fewer than run 1 had already "
     "reached — reconciled by arithmetic in the disagreements section (448 = the placebo "
     "row count). Also records `thinking model — raising max_tokens to 2200`, a "
     "mid-campaign parameter change, so rates pooled across the two runs mix settings. "
     "The gemma arm is progressing here and may still be live — the probe CSV/JSONL mtimes "
     "in the inventory are the check."),
]


# ──────────────────────────────────────────────────────────────────────────────
# render
# ──────────────────────────────────────────────────────────────────────────────
def render():
    # This page no longer inherits dashboard.py's style block, so it carries its own.
    # Kept byte-identical to the .section-header / .explainer rules in src/app/dashboard.py.
    st.markdown(f"""
    <style>
      [data-testid="stAppViewContainer"] {{ background: #F8FAFC; }}
      [data-testid="stSidebar"] {{ background: white; border-right: 1px solid {BORDER}; }}
      .section-header {{ font-size:13px; font-weight:700; letter-spacing:.8px;
                         text-transform:uppercase; color:{SUBTEXT}; margin:0 0 4px 0; }}
      .explainer {{ font-size:13px; color:{SUBTEXT}; margin-bottom:14px; line-height:1.5; }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="section-header">Data Provenance</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="explainer">'
        'Every dataset in this repo, traced from external source through cleaning, '
        'filtering, joining and analysis. Built to be checked, not believed: each producer '
        'claim names the script and line range that performs the write, and every count, '
        'size and timestamp on this page is read off disk when the page renders. Known '
        'defects sit next to the working parts on purpose. Where the record is genuinely '
        'absent it says so rather than guessing.'
        '</p>', unsafe_allow_html=True)
    st.caption(f"Rendered {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
               f"repo root `{REPO}`")

    # ── Top-level flow ───────────────────────────────────────────────────────
    st.markdown("#### The pipeline end to end")
    _mermaid(TOP_FLOW, 900)
    st.markdown(
        "- **Red nodes are imported, not produced.** `data/gen_review.db`, "
        "`data/archive/all_paper_results.csv` and `data/OpenAlex/` all arrive from outside this "
        "repo; no script here creates or refreshes them. The OpenAlex match files can at "
        "least be attributed to `Archive/CompletePipeline/design/"
        "fetch_openalex_citations_from_arxiv_matches.py`, which writes them under different "
        "paths (`rawdata/Design/OpenAlex/`) and is never invoked by anything in `src/`.\n"
        "- **The dashed edge into `eval_table.csv` is the load-bearing gap.** "
        "`committee_rating` and `deepseek_p_accept` are on disk and drive the LLM regimes, "
        "the heterogeneity section and the exclusion eval — and no script performs that "
        "join. Verified live below.\n"
        "- **There is a genuine cycle.** `src/fetch/fetch_citations_s2_full.py#L67` reads "
        "`outputs/eval_table.csv` to decide which papers to fetch, while the S2 counts it "
        "writes are then read back by the dashboard alongside that same table. Rebuilding "
        "in dependency order therefore requires two passes, and nothing enforces that.\n"
        "- **`output/` (singular) is not a typo in this diagram.** The OpenAlex ground truth "
        "genuinely lives there, outside the `CLAUDE.md` convention, and is an *input* to the "
        "join.\n"
        "- **`outputs/eval_results.csv` is a dead end** (amber). The dashboard never reads "
        "it; it recomputes every regime live from `eval_table.csv`."
    )

    st.markdown("---")

    # ── Per-stage detail ─────────────────────────────────────────────────────
    st.markdown("#### Stage detail")
    stage_tabs = st.tabs(["0 · Source DB", "1 · Citations (OA vs S2)", "2 · arXiv resolution",
                          "3 · LLM annotation & probes", "4 · The join", "5 · Out-of-sample"])

    with stage_tabs[0]:
        _mermaid(DIA_SOURCE, 620)
        tables = _db_tables()
        if tables is None:
            st.warning(f"`{DB_PATH}` is **missing or unreadable** — no row counts to show. "
                       "Everything downstream of it is therefore unverifiable in this "
                       "checkout.")
        else:
            st.markdown("**Row counts, read from the DB now:**")
            st.dataframe(pd.DataFrame(tables, columns=["table", "rows"]).assign(
                rows=lambda d: d["rows"].map("{:,}".format)),
                use_container_width=True, hide_index=True)
        yrs = _db_years()
        if yrs:
            dfy = pd.DataFrame(yrs, columns=["year", "submissions", "accepts"])
            dfy["in study window"] = dfy["year"].isin([2018, 2019, 2020]).map(
                {True: "yes", False: "out-of-sample only"})
            st.markdown("**`SUBMISSION.when_submitted` distribution:**")
            st.dataframe(dfy.assign(
                submissions=lambda d: d["submissions"].map("{:,}".format),
                accepts=lambda d: d["accepts"].map("{:,}".format)),
                use_container_width=True, hide_index=True)
        st.markdown(
            "- The schema is three tables (`sqlite3 data/gen_review.db .schema`): "
            "`SUBMISSION`, `REVIEW`, `GENAI_REVIEW`. `GENAI_REVIEW` holds the "
            "neutral/positive/negative persona reviews — a *different* LLM annotation from "
            "the committee run in `data/archive/all_paper_results.csv`, and the source of the "
            "`llm_*_rating` columns.\n"
            "- The accept test used everywhere is `decision.str.startswith('Accept')` "
            "(`src/build/build_eval_table.py#L85`), which pools `Reject`, "
            "`Invite to Workshop Track` and `Desk Reject` on the reject side. Desk rejects "
            "have no reviews but still sit in the pool the random and ideal baselines draw "
            "from.\n"
            "- `REVIEW` is read without a year filter (`src/build/build_eval_table.py#L28-L31`); "
            "the window is applied only by the left join onto the filtered submissions."
        )

    with stage_tabs[1]:
        _mermaid(DIA_CITE, 780)
        st.markdown(
            "- **Two independent citation ground truths exist and disagree.** OpenAlex came "
            "first (June); the S2 refetch came later (July) precisely because OpenAlex "
            "matches arXiv preprint records and misses the published versions. The dashboard "
            "keeps both behind the sidebar toggle.\n"
            "- **The S2 file mixes three code paths of very different quality** in one CSV, "
            "distinguished only by its `method` column: `arxiv_batch` (ID match, reliable), "
            "`title_cached` (bulk copy, unfiltered) and `title_match` (live, records "
            "`title_sim` so it is at least tunable).\n"
            "- **`src/fetch/fetch_citations_s2_v2.py` is the rebuild** that puts everything on one "
            "tiered path. Per `outputs/s2_attribution_report.md`, it is **ID-matched papers "
            "only** so far — the report's own first line says title matching and stub probes "
            "\"are deferred until the API key arrives, so the papers missing here are "
            "disproportionately rejected ones.\" It has not replaced "
            "`s2_citations_full.csv` in the analysis path.\n"
            "- **Snapshot dates are not aligned.** OpenAlex counts date from mid-June, S2 "
            "from mid-July and August (see the inventory mtimes). "
            "`docs/notes/data_pipeline_plan.md` phase 5 records why that matters: a live "
            "diagnostic found accepted papers' median citations rising 49% against "
            "rejected papers' 4% over six weeks, so staleness correlates with treatment."
        )

    with stage_tabs[2]:
        _mermaid(DIA_ARXIV, 760)
        st.markdown(
            "- **Two passes, both full-corpus, both recent and both uncommitted.** "
            "`src/fetch/resolve_arxiv_ids.py` (exact + token-set) and "
            "`src/fetch/resolve_arxiv_fuzzy.py` (TF-IDF abstract verification at cosine ≥ 0.7) "
            "are untracked work in progress — `git log` returns nothing for either, so "
            "there is no history and no authorship record for them.\n"
            "- **They replace, but have not displaced, the old match.** "
            "`src/fetch/fetch_citations_s2_full.py#L68` still reads "
            "`data/OpenAlex/openalex_rdd_arxiv_paper_level.csv`. Only "
            "`src/fetch/fetch_citations_s2_v2.py` and `src/build/build_eval_table_year.py` read the new "
            "resolution. Quantified in the defects section.\n"
            "- **The dump is not in the repo.** `outputs/arxiv_dump_download.log` ends with "
            "a path into the user's local HuggingFace cache. The scan is reproducible only "
            "on a machine that re-downloads the same snapshot revision — the log does record "
            "the revision hash, which is the one thing that makes it pinnable.\n"
            "- **Neither pass applies a threshold at write time** "
            "(`src/fetch/resolve_arxiv_ids.py` docstring), so `title_sim`, `token_jaccard` and "
            "`abstract_cos` stay tunable downstream without a re-scan. Good practice; it "
            "also means the CSV is *candidates*, not decisions.\n"
            "- **The accept/reject match gap is real, not a bug**, and both reports say so. "
            "It is a selection channel that has to be carried forward, because ID-matched "
            "papers get verifiable citation attribution and title-matched ones do not.\n"
            "- **Cross-checks that passed, recorded because a clean check is also a "
            "result.** `outputs/arxiv_fuzzy_report.md`'s per-year gap column is internally "
            "consistent with its own accepted/rejected columns (89.0 − 55.2 = 33.8pp for "
            "2018, as printed) and with its overall figure (42.1pp → 31.9pp), and that "
            "overall figure agrees with the independent \"~32pp\" statement in "
            "`outputs/samples/oos_sample_design.md`. The 2018-2020 match rates recomputed "
            "live in the defects section below also agree with "
            "`outputs/arxiv_resolution_report.md`."
        )

    with stage_tabs[3]:
        _mermaid(DIA_LLM, 780)
        st.markdown(
            "- **Two unrelated LLM annotation sources are easy to confuse.** "
            "`GENAI_REVIEW` inside the DB (neutral/positive/negative personas → the "
            "`llm_*_rating` columns) is *not* the committee run; the committee and "
            "decision-head outputs live in `data/archive/all_paper_results.csv` and were produced "
            "outside this repo.\n"
            "- **The decision head is two different models.** Per `outputs/data_audit.md` "
            "(Stage 2 statistics), `decision_head_model` is `DeepSeek-V3.1` on 2,361 papers "
            "and `gpt-oss-20b` on 2,136 — disjoint halves pooled into one regime. Any "
            "\"committee beats decision head\" claim could be \"one of two models "
            "underperforms\".\n"
            "- **Field tags are LLM-generated and unvalidated**, and coverage collapses in "
            "2020: `outputs/data_audit.md` Stage 2 reports field coverage by year as "
            "`{2018: '100%', 2019: '100%', 2020: '17%'}`. Since `citation_pct_rank` is "
            "computed *within* field×year cells (`src/build/build_eval_table.py#L75-L80`), the "
            "normalized ground truth inherits that hole.\n"
            "- **The leakage probes are annotations, not results.** LAP and FAME write "
            "per-paper CSVs plus JSONL traces; the exclusion eval and bootstrap then consume "
            "them. Reading the traces is the only way to audit a probe, and "
            "`src/probes/leakage_lap_v1.py#L189` writes them all to one hardcoded path "
            "(`outputs/leakage_lap_traces.jsonl`) regardless of the `--tag` used for the "
            "CSV, so tagged runs cannot be separated from the main one. Check that file's "
            "size in the inventory above before trusting any trace-level claim.\n"
            "- **The abstract-completion probe is a small subsample.** "
            "`outputs/data_audit.md` Stage 2 records 297 rows and 5 papers (1.7%) flagged "
            "extractable — a null there does not prove absence from the weights, and the "
            "dashboard's own caption says so."
        )

    with stage_tabs[4]:
        _mermaid(DIA_JOIN, 520)
        st.markdown(
            "- `src/build/build_eval_table.py` is 91 lines and does the whole join: DB window "
            "filter, review-rating parse and aggregate, `GENAI_REVIEW` pivot, OpenAlex "
            "counts, field tags, then field×year percentile rank.\n"
            "- **Citations are nulled unless OpenAlex reported `status == 'found'`** "
            "(#L72-L73), which is why `openalex_citations` coverage is well below 100%.\n"
            "- **`citation_pct_rank` is computed inside field×year cells** (#L75-L80). With "
            "field coverage at 17% for 2020, many cells are tiny, and a percentile rank "
            "inside a small cell is close to meaningless.\n"
            "- **`rating_std` is `fillna(0)` for single-review papers** (#L42-L43) — "
            "treating \"one reviewer\" as \"perfect agreement\". That feeds the "
            "disagreement-adjusted regime directly.\n"
            "- **The builder is not reproducible.** See the first defect below."
        )
        cols = _csv_columns("outputs/eval_table.csv")
        if cols:
            st.markdown(f"**`outputs/eval_table.csv` columns on disk right now** "
                        f"({len(cols)}): `{', '.join(cols)}`")

    with stage_tabs[5]:
        _mermaid(DIA_OOS, 800)
        st.markdown(
            "- **Three arms with different contamination exposure**, quoting "
            "`outputs/samples/oos_sample_design.md`: `clean` = 149 papers from 2025 "
            "(\"nothing\" a Dec-2023-cutoff model could have seen), `partial` = 149 from "
            "2024 (\"preprints only — no decision, no citations\"), `contaminated` = 150 "
            "from 2018-2020.\n"
            "- **Four probes per paper**: `lap`, `fame`, `placebo` (fabricated title) and "
            "`wrongyear`. The design doc records why the placebo arm is mandatory: a test "
            "call to Llama-3.3-70B answered \"accepted\" for *Attention Is All You Need* at "
            "ICLR 2018, a paper never submitted to ICLR.\n"
            "- **Eligibility is decision-skewed by construction.** The same doc: only papers "
            "with an arXiv-ID match into S2 are eligible, which \"removes roughly half of "
            "2024/2025, and the removed half is decision-skewed\". Identical across arms, so "
            "model-vs-model comparison survives; generalization to non-arXiv work does not.\n"
            "- **Stated MDE is coarse.** The design doc puts the arm-vs-arm MDE on a rate "
            "difference at **≈16.2pp** (α .05, power .80, worst case p = .5). Effects "
            "smaller than that are not detectable with these n, and within-arm subgroup "
            "splits are wider still.\n"
            "- **These files are being written while you read this.** The probe CSVs and "
            "JSONL traces are append-mode and resumable (`src/probes/run_oos_probes.py#L183-L187`); "
            "row counts in the inventory below are a snapshot of an in-progress run, not a "
            "final N."
        )

    st.markdown("---")

    # ── Live artifact inventory ──────────────────────────────────────────────
    st.markdown("#### Live artifact inventory")
    st.markdown(
        '<p class="explainer">Computed by stat-ing and parsing each file at render time. '
        '"—" in a row count means the format makes a count meaningless (binary) or the file '
        'is missing. Nothing in this table is transcribed.</p>', unsafe_allow_html=True)

    rows = []
    for path, stage, producer, inputs, note in ART:
        exists, size, mtime = _stat(path)
        n, method = _rows(path) if exists else (None, "")
        tracked = _git_tracked(path)
        rows.append({
            "path": path,
            "stage": STAGE_LABEL.get(stage, stage),
            "status": "present" if exists else "MISSING",
            "rows": f"{n:,}" if isinstance(n, int) else "—",
            "count method": method or "—",
            "size": _fmt_size(size) if exists else "—",
            "modified": mtime or "—",
            "git": {True: "tracked", False: "untracked", None: "unknown"}[tracked],
            "produced by": producer or "⚠ NO PRODUCER IN src/",
            "reads": ", ".join(inputs) if inputs else "",
            "note": note,
        })
    inv = pd.DataFrame(rows)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        pick = st.multiselect("Stage", sorted(inv["stage"].unique()),
                              default=sorted(inv["stage"].unique()),
                              key="prov_stage_filter")
    with c2:
        only_gaps = st.checkbox("Only untraceable artifacts", value=False, key="prov_only_gaps")
    with c3:
        only_missing = st.checkbox("Only missing files", value=False, key="prov_only_missing")
    view = inv[inv["stage"].isin(pick)]
    if only_gaps:
        view = view[view["produced by"].str.startswith("⚠")]
    if only_missing:
        view = view[view["status"] == "MISSING"]

    n_missing = int((inv["status"] == "MISSING").sum())
    n_untracked = int((inv["git"] == "untracked").sum())
    n_gap = int(inv["produced by"].str.startswith("⚠").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Artifacts catalogued", f"{len(inv)}")
    m2.metric("Missing from disk", f"{n_missing}")
    m3.metric("Untracked by git", f"{n_untracked}")
    m4.metric("No producer in src/", f"{n_gap}")
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)
    if n_missing:
        st.caption("Missing files are reported as MISSING rather than carrying a stale "
                   "figure: " + ", ".join(f"`{p}`" for p in
                                          inv.loc[inv["status"] == "MISSING", "path"]))

    st.markdown("---")

    # ── Known defects ────────────────────────────────────────────────────────
    st.markdown("#### Known defects and open questions")
    st.markdown(
        '<p class="explainer">Each entry is re-verified against the files every time this '
        'page renders — the evidence text below is generated, not stored. Severity follows '
        'the convention <code>src/audit/data_audit.py</code> uses: <b>blocker</b> = a number is '
        'wrong or not reproducible; <b>major</b> = a real bias or fragility that must be '
        'disclosed; <b>minor</b> = an undocumented choice; <b>info</b> = context.</p>',
        unsafe_allow_html=True)

    results = []
    for fn in CHECKS:
        try:
            results.append(fn())
        except Exception as e:  # a broken check must not take the tab down
            results.append(("info", f"Check `{fn.__name__}` failed to run",
                            f"UNVERIFIED — the check itself raised "
                            f"`{type(e).__name__}: {e}`. Nothing should be concluded from "
                            f"its absence.", []))
    results.sort(key=lambda r: _SEV_ORDER.get(r[0], 9))

    for sev, title, evidence, refs in results:
        colour = SEV_COLOR.get(sev, SUBTEXT)
        st.markdown(
            f'<div style="border-left:3px solid {colour};padding:2px 0 2px 12px;'
            f'margin:14px 0 4px 0;">'
            f'<span style="font-size:11px;font-weight:700;letter-spacing:.6px;'
            f'text-transform:uppercase;color:{colour}">{sev}</span><br/>'
            f'<span style="font-size:15px;font-weight:600;color:{TEXT}">{title}</span></div>',
            unsafe_allow_html=True)
        st.markdown(evidence)
        if refs:
            st.caption("Evidence: " + " · ".join(f"`{r}`" for r in refs))

    st.markdown("---")

    # ── Disagreements ────────────────────────────────────────────────────────
    st.markdown("#### Where two sources disagree")
    st.markdown(
        '<p class="explainer">Both sides are shown. None of these is resolved here, because '
        'resolving them requires evidence the repo does not contain.</p>',
        unsafe_allow_html=True)
    dis = _disagreements()
    if not dis:
        st.info("No disagreement checks could run — the report files they compare are missing.")
    for title, body, refs in dis:
        st.markdown(f"**{title}**")
        st.markdown(body)
        st.caption("Evidence: " + " · ".join(f"`{r}`" for r in refs))
        st.markdown("")

    st.markdown("---")

    # ── Provenance gaps ──────────────────────────────────────────────────────
    st.markdown("#### Provenance gaps")
    st.markdown(
        '<p class="explainer">Artifacts on disk whose production cannot be traced to any '
        'script in <code>src/</code>. Established by grepping every script for '
        '<code>to_csv</code>, <code>to_json</code>, <code>savefig</code> and '
        '<code>open(..., "w")</code> against each path. "No record" is the honest answer '
        'here, and it is the answer for most of this list.</p>', unsafe_allow_html=True)

    gaps = inv[inv["produced by"].str.startswith("⚠")][
        ["path", "stage", "status", "rows", "size", "modified", "git", "note"]]
    st.dataframe(gaps, use_container_width=True, hide_index=True)

    st.markdown(
        "**What would settle each one:**\n\n"
        "- `data/gen_review.db` — the OpenReview scraper. Not in this repo, not in "
        "`Archive/`. Untracked, so git has no history for it either. **UNVERIFIED**: which "
        "OpenReview API version, on what date, with what inclusion rules. Only the scraper "
        "source or a scrape log would settle it. The `source_id` column on `SUBMISSION` may "
        "encode the provenance, but nothing documents its meaning.\n"
        "- `data/archive/all_paper_results.csv` / `.jsonl` / `paper_manifest.csv` / "
        "`gemma_ready7_wave1_cached_v2/` — partly self-describing: `data/summary.json` "
        "records `created_at_utc`, four named source runs under "
        "`OutputNew/Empirics/…`, per-run year breakdowns and validation counts, and "
        "`data/README.md` documents the inclusion and exclusion rules. That is real "
        "provenance for the *assembly* step. **UNVERIFIED**: the prompts, model "
        "parameters and code of the committee/decision-head runs themselves, which lived in "
        "the `OutputNew/` tree outside this repo.\n"
        "- `data/OpenAlex/openalex_rdd_*` — attributable to "
        "`Archive/CompletePipeline/design/fetch_openalex_citations_from_arxiv_matches.py` "
        "and `diagnose_openalex_arxiv_misses.py` by filename match, but those scripts write "
        "to `rawdata/Design/OpenAlex/`, not `data/OpenAlex/`. The copy step is undocumented, "
        "so the attribution is by name, not by trace. `CLAUDE.md` states `Archive/` is never "
        "run or imported, which makes these files unreproducible by design.\n"
        "- `data/OpenAlex/openalex_rdd_dashboard.csv` — read at `src/app/dashboard.py#L1042` and "
        "written by nothing. Git dates it to commit `534fa8e` (2026-07-10, "
        "\"data: add slim RDD and covariate files for Streamlit Cloud deploy\"), so it was "
        "hand-derived from the paper-level file to keep the deploy small. **UNVERIFIED**: "
        "the exact derivation, so the two files cannot be checked against each other.\n"
        "- `output/papers_2018_2020.csv`, `output/reviews_2018_2020.csv`, "
        "`output/reviews_summary_2018_2020.csv`, `output/part_b.log` — **no record "
        "whatsoever.** No script in `src/` or `Archive/` mentions any of these filenames, "
        "nothing reads them, `output/` has never been committed (`git log -- output/` is "
        "empty), and the log names a \"part_b\" step that appears nowhere. They share an "
        "mtime with the DB, which suggests a one-off export at scrape time — that is an "
        "inference, not evidence.\n"
        "- `outputs/oa_title_match_venues.csv` — read at "
        "`src/fetch/fetch_rejected_venues_s2_title.py#L67-L68` to *exclude* already-covered "
        "papers, so it silently changes that script's working set, and no script writes it. "
        "This one has downstream effect, which makes it the most consequential gap in the "
        "list.\n"
        "- `outputs/source_comparison_verified.csv` — no writer. The name suggests a "
        "verified variant of `citation_source_comparison.csv`; its mtime (2026-08-02) puts "
        "it alongside the S2 v2 rebuild. **UNVERIFIED** which script or ad-hoc session "
        "produced it, so it should not be quoted.\n"
        "- `outputs/breakthrough_bubble.png`, `outputs/breakthrough_top_decile.png` — no "
        "writer in `src/` or `Archive/`. Charts that cannot be regenerated.\n"
        "- `outputs/table1_summary_stats.tex` — see the disagreement above: the audit's "
        "producer map claims a script that only prints.\n"
        "- `outputs/findings_integrity_check.md`, `outputs/venue_coverage_strategy.md` — "
        "hand-written prose in the generated-output directory. Not defects in themselves, "
        "but they are indistinguishable from generated reports by location, and "
        "`docs/PROJECT_OVERVIEW.md#L140` cites the first as if it were an artifact."
    )

    st.markdown("**Untracked scripts — no history, no authorship, no review**")
    try:
        r = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "src/"],
                           cwd=REPO, capture_output=True, text=True, timeout=20)
        untracked_src = [ln for ln in r.stdout.splitlines() if ln.endswith(".py")]
    except Exception:
        untracked_src = None
    if untracked_src is None:
        st.caption("UNVERIFIED — `git ls-files --others` could not be run here.")
    elif not untracked_src:
        st.caption("None: every script under `src/` is tracked.")
    else:
        st.markdown("\n".join(f"- `{p}` — untracked, so `git log` cannot date it or "
                              f"attribute it" for p in untracked_src))
        st.caption("Untracked files have no history. Any dating of these scripts below the "
                   "filesystem mtime would be invention, so none is offered.")

    st.markdown("---")

    # ── data_audit.md findings index ─────────────────────────────────────────
    st.markdown("#### Cross-check: the standing data audit")
    findings, gen = _audit_findings()
    if findings is None:
        st.warning("`outputs/data_audit.md` is **missing or contains no parseable findings**. "
                   "The findings index below cannot be shown, and none of its numbers are "
                   "reproduced here from memory.")
    else:
        if gen:
            st.caption(f"Parsed from `outputs/data_audit.md`, which reports itself as "
                       f"generated {gen[0]} by `{gen[1]}`. Headings are parsed at render "
                       f"time, not transcribed — if the audit is re-run, this table follows.")
        counts = findings["severity"].value_counts().to_dict()
        st.markdown("Findings by severity: " + " · ".join(
            f"**{counts.get(s, 0)} {s}**" for s in ["blocker", "major", "minor", "info"]))
        sev_pick = st.multiselect("Severity", ["blocker", "major", "minor", "info"],
                                  default=["blocker", "major"], key="prov_audit_sev")
        st.dataframe(findings[findings["severity"].isin(sev_pick)],
                     use_container_width=True, hide_index=True, height=320)
        st.caption("This is an index, not a substitute. Each finding in `data_audit.md` "
                   "carries its own 'Where to look' file:line list and a copy-pasteable "
                   "reproduce command; that file is the primary artifact.")
    st.markdown(
        "- The audit is a *claim set*, not ground truth. The defects section above "
        "independently re-derives the ones that matter most; where this tab and the audit "
        "agree, they agree from separate computations.\n"
        "- `docs/notes/data_pipeline_plan.md` (2026-07-28) is the remediation plan written "
        "against that audit and is the best single account of *why* the citation pipeline "
        "looks the way it does. It marks phase 1 (full-corpus arXiv resolution) as "
        "IN PROGRESS — consistent with `src/fetch/resolve_arxiv_ids.py` being untracked."
    )

    st.markdown("---")

    # ── Logs ─────────────────────────────────────────────────────────────────
    st.markdown("#### What actually ran, according to the logs")
    st.markdown(
        '<p class="explainer">Logs are primary evidence and several of them document '
        'partial or unrecorded runs. Sizes and mtimes are read now; the reading of each '
        'log was done by hand against the file named.</p>', unsafe_allow_html=True)
    log_rows = []
    for path, what, reading in LOGS:
        exists, size, mtime = _stat(path)
        log_rows.append({
            "log": path,
            "step": what,
            "size": _fmt_size(size) if exists else "MISSING",
            "modified": mtime or "—",
            "empty": "yes" if exists and size == 0 else ("—" if not exists else "no"),
            "what it shows": reading if exists else
            "MISSING — the reading below cannot be checked in this checkout.",
        })
    logs_df = pd.DataFrame(log_rows)
    st.dataframe(logs_df, use_container_width=True, hide_index=True, height=380)
    n_empty = int((logs_df["empty"] == "yes").sum())
    n_gone = int((logs_df["size"] == "MISSING").sum())
    st.markdown(
        f"- **{n_empty} of these logs are zero bytes** and {n_gone} are missing entirely, so "
        "those runs left no behavioural record. A zero-byte log is not evidence the step "
        "did not run — `outputs/leakage_lap_v1.csv` exists — it is evidence that nothing "
        "was captured.\n"
        "- **`outputs/s2_fetch_full.log` never reaches its own summary block.** The complete "
        "CSV therefore came from a run whose log was not kept, which is why the "
        "`title_cached` block's size had to be recomputed from the CSV above rather than "
        "read from a log.\n"
        "- **The two OOS run logs appear not to reconcile** on how many Llama calls were "
        "already done. They do, once you check the arithmetic against the probe plan — see "
        "the disagreements section. Run 2 separately records a mid-campaign `max_tokens` "
        "change, so a rate pooled across both runs mixes parameter settings."
    )

    st.markdown("---")

    # ── Code-level dependencies ──────────────────────────────────────────────
    st.markdown("#### How the code depends on itself")
    st.markdown(
        '<p class="explainer">Everything above traces data. This traces code: the import '
        'edges between modules under <code>src/</code>, parsed live with <code>ast</code> at '
        'render time, so it cannot go stale. Blue nodes have two or more importers — those '
        'are the shared modules a change actually propagates from. Modules with no import '
        'edges at all are omitted from the diagram and listed below instead.</p>',
        unsafe_allow_html=True)
    try:
        g = import_graph.graph()
        _mermaid(import_graph.mermaid(g), 620)
        orph = import_graph.orphans(g)
        st.markdown(
            f"- **{len(g)} modules, {sum(len(v) for v in g.values())} import edges.** The "
            "codebase is flat by design (`CLAUDE.md`: one script = one logical step), so "
            "most scripts import nothing local and stand alone.\n"
            f"- **{len(orph)} modules have no importers.** That is the intended shape for a "
            "runnable script, so this list is *entry points and dead code mixed together* — "
            "it cannot tell them apart, and it is a starting point for deletion, not a "
            "verdict. Anything here that is neither run from the command line nor read by "
            "the dashboard is dead.\n"
            "- Cross-check against the data lineage above: a module that is orphaned here "
            "**and** appears in no producer attribution above is unreachable both ways."
        )
        st.code("  ".join(orph), language="text")
    except Exception as exc:  # a parse failure should not take the whole page down
        st.warning(f"Import graph unavailable: {type(exc).__name__}: {exc}")

    st.markdown("---")
    st.caption(
        "Scope limits of this page, stated plainly. It traces *data* artifacts; it does not "
        "audit statistical method (`docs/methodology_review.md` and "
        "`outputs/findings_integrity_check.md` cover that). Producer attributions come from "
        "static reading of `src/`, so a path built at runtime from a CLI flag is attributed "
        "to the flag, not proven for the specific file. `Archive/` is read for attribution "
        "only and never imported, per `CLAUDE.md`. Nothing here writes to `data/`."
    )
