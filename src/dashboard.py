"""
Reviewer regime comparison dashboard.
Run: streamlit run src/dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, spearmanr

from metrics import METRIC_LABELS, compute_metrics
from baselines import random_baseline, ideal_baseline
from regimes.human_actual import HumanActual
from regimes.human_score import HumanScore
from regimes.human_disagree import HumanDisagree
from regimes.llm_committee import LLMCommittee
from regimes.llm_deepseek import LLMDeepSeek

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CitesBench — Reviewer Regime Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "Human (AC decisions)":                  "#2563EB",
    "Human (score top-N)":                   "#D97706",
    "Human (disagreement-adjusted)":         "#0D9488",
    "LLM Committee (Gemma)":                 "#DC2626",
    "LLM Decision Head":                     "#7C3AED",
}
QUAD_COLORS = {
    "regime ∩ ideal":  "#2563EB",
    "ideal only":      "#F59E0B",
    "regime only":     "#10B981",
    "neither":         "#CBD5E1",
}
RANDOM_COLOR = "#94A3B8"
IDEAL_COLOR  = "#1E293B"
BG, BORDER, TEXT, SUBTEXT = "#F8FAFC", "#E2E8F0", "#0F172A", "#64748B"

METRIC_SHORT = {
    "median_citations":   "Median cites",
    "mean_log_citations": "Mean log(1+c)",
    "recall_at_1":        "Recall @1%",
    "recall_at_5":        "Recall @5%",
    "recall_at_10":       "Recall @10%",
}

st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: {BG}; }}
  [data-testid="stSidebar"] {{ background: white; border-right: 1px solid {BORDER}; }}
  .section-header {{ font-size:13px; font-weight:700; letter-spacing:.8px;
                     text-transform:uppercase; color:{SUBTEXT}; margin:0 0 4px 0; }}
  .explainer {{ font-size:13px; color:{SUBTEXT}; margin-bottom:14px; line-height:1.5; }}
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_results(): return pd.read_csv("outputs/eval_results.csv")

S2_CSV = "outputs/s2_citations_full.csv"

@st.cache_data
def load_eval_table(source="OpenAlex", sz=0, _v=3):  # sz = S2 file size, busts cache as fetch grows
    et = pd.read_csv("outputs/eval_table.csv")
    rej_path = "outputs/outlier_reviews.csv"
    if os.path.exists(rej_path):
        rej = pd.read_csv(rej_path)[["title", "rejection_tags"]].drop_duplicates("title")
        et = et.merge(rej, on="title", how="left")
    else:
        et["rejection_tags"] = pd.NA
    if source == "Semantic Scholar" and os.path.exists(S2_CSV):
        s2 = pd.read_csv(S2_CSV)
        # arXiv-ID matches are exact; title matches need the similarity gate
        ok = s2[s2["s2_citations"].notna() &
                ((s2["method"] == "arxiv_batch") | (s2["title_sim"].fillna(0) >= 0.9))]
        et = et.merge(ok[["paper_id", "s2_citations"]].drop_duplicates("paper_id"),
                      on="paper_id", how="left")
        # keep the original column name so every downstream consumer works unchanged
        et["openalex_citations"] = et["s2_citations"]
        et = et.drop(columns=["s2_citations"])
        et["citation_pct_rank"] = et.groupby(["field", "year"])["openalex_citations"] \
                                    .rank(pct=True)
    return et

def _et(src):
    sz = os.path.getsize(S2_CSV) if (src == "Semantic Scholar" and os.path.exists(S2_CSV)) else 0
    return load_eval_table(src, sz)

st.sidebar.markdown("## Controls")
_s2_ready = os.path.exists(S2_CSV)
citation_source = st.sidebar.radio(
    "Citation source (ground truth)",
    ["OpenAlex", "Semantic Scholar"] if _s2_ready else ["OpenAlex"],
    help="OpenAlex matched ~99% of the corpus to arXiv-preprint records and misses citations "
         "to the published versions — median 2.9× undercount, and differential by acceptance "
         "(3.5× accepted vs 2.0× rejected). Semantic Scholar merges preprint + published "
         "versions and also indexes OpenReview-only submissions. "
         "See outputs/citation_source_comparison.md.")

try:
    df_static = load_results()
    eval_table = _et(citation_source)
except FileNotFoundError:
    st.error("Run `python src/run_eval.py` first.")
    st.stop()

BASELINE_CACHE = "outputs/baselines_cache.csv"

@st.cache_data
def prepare_pool(year, mode, impute_zeros, exclude_top_decile=False, src="OpenAlex"):
    et = _et(src)
    pool = et[et["year"] == year].copy()
    if impute_zeros:
        pool["openalex_citations"] = pool["openalex_citations"].fillna(0)
        for field, grp in pool.groupby("field"):
            mask = pool["field"].eq(field) & pool["openalex_citations"].notna()
            pool.loc[mask, "citation_pct_rank"] = pool.loc[mask, "openalex_citations"].rank(pct=True)
    if exclude_top_decile:
        known = pool.dropna(subset=["openalex_citations"])
        if not known.empty:
            cut = known["openalex_citations"].quantile(0.90)
            pool = pool[pool["openalex_citations"].isna() | (pool["openalex_citations"] <= cut)].copy()
    return pool

@st.cache_data
def get_baselines(year, mode, impute_zeros, exclude_top_decile=False, src="OpenAlex"):
    key = f"{year}_{mode}_{int(impute_zeros)}_{int(exclude_top_decile)}"
    if src != "OpenAlex":
        key += "_s2"
    if os.path.exists(BASELINE_CACHE):
        cached = pd.read_csv(BASELINE_CACHE)
        hit = cached[cached["key"] == key]
        if not hit.empty:
            rand  = dict(zip(hit[hit["which"]=="random"]["metric"], hit[hit["which"]=="random"]["value"]))
            ideal = dict(zip(hit[hit["which"]=="ideal"]["metric"],  hit[hit["which"]=="ideal"]["value"]))
            n     = int(hit["n"].values[0])
            return rand, ideal, n
    pool = prepare_pool(year, mode, impute_zeros, exclude_top_decile, src)
    n = pool[pool["decision"].str.startswith("Accept", na=False)].shape[0]
    rand  = random_baseline(pool, n, mode)
    ideal = ideal_baseline(pool, n, mode)
    rows = []
    for which, d in [("random", rand), ("ideal", ideal)]:
        for metric, value in d.items():
            rows.append({"key": key, "year": year, "mode": mode,
                         "impute_zeros": impute_zeros, "which": which,
                         "metric": metric, "value": value, "n": n})
    new_df = pd.DataFrame(rows)
    if os.path.exists(BASELINE_CACHE):
        new_df = pd.concat([pd.read_csv(BASELINE_CACHE), new_df], ignore_index=True)
    new_df.to_csv(BASELINE_CACHE, index=False)
    return rand, ideal, n

def compute_live(regime, year, mode, impute_zeros, exclude_top_decile=False, src="OpenAlex"):
    pool = prepare_pool(year, mode, impute_zeros, exclude_top_decile, src)
    rand, ideal_vals, n = get_baselines(year, mode, impute_zeros, exclude_top_decile, src)
    selected = regime.select(pool, n)
    metrics = compute_metrics(selected, pool, mode)
    rows = []
    for metric, value in metrics.items():
        rv, iv = rand.get(metric, np.nan), ideal_vals.get(metric, np.nan)
        rows.append({"regime": regime.name, "year": year, "metric": metric, "mode": mode,
                     "value": value, "random_value": rv, "ideal_value": iv,
                     "lift":     (value - rv) / abs(rv) if rv and rv != 0 else np.nan,
                     "drawdown": (iv - value) / abs(iv) if iv and iv != 0 else np.nan})
    return rows

# ── Sidebar (continued — Controls header + citation source live above, pre-load) ─
mode = "raw"
year_opts = ["All years"] + sorted(eval_table["year"].dropna().unique().astype(int).tolist())
selected_year = st.sidebar.selectbox("Year", year_opts)

st.sidebar.markdown("---")
lam = st.sidebar.slider("λ (disagreement weight)", -3.0, 3.0, 1.0, 0.25,
    help="score = mean_rating + λ × rating_std  |  λ > 0: boost contested  |  λ < 0: prefer consensus")
impute_zeros = st.sidebar.checkbox("Impute 0 citations for unmatched papers", value=False,
    help="~37% of papers have no OpenAlex match. Off: exclude from metrics. On: count as 0.")
show_drawdown = st.sidebar.checkbox("Show drawdown from ideal", value=False,
    help="Off: % of gap closed (higher = better). On: % left on table vs ideal (lower = better).")
exclude_top_decile = st.sidebar.checkbox("Exclude top-10% by citations (leakage-robust)", value=False,
    help="Removes the top citation decile per year and adjusts N. "
         "LLMs may recognise famous papers from training data — stripping them tests whether "
         "the signal survives without memorisation of high-impact work.")

st.sidebar.markdown("---")
if citation_source == "OpenAlex":
    st.sidebar.caption("⚠️ Median/mean citations computed over OpenAlex-matched papers only "
                       "(accepts ~89%, rejects ~63%). Recall metrics unaffected. "
                       "Note: OpenAlex undercounts (median 2.9× vs S2) and undercounts accepted "
                       "papers more — see the citation-source toggle above.")
else:
    _s2n = int(eval_table["openalex_citations"].notna().sum())
    st.sidebar.caption(f"⚠️ Semantic Scholar counts active: {_s2n:,}/{len(eval_table):,} papers "
                       f"matched ({_s2n/len(eval_table):.0%}). Counts merge preprint + published "
                       "versions. Field-normalized percentile ranks recomputed from S2 counts.")
st.sidebar.markdown("---")
st.sidebar.caption(
    "ℹ️ **LLM regimes** use a two-stage pipeline: Gemma-4-31B committee (4 reviewer "
    "personas) → decision head. Borderline papers (n=2,361, ratings 4–7) used a "
    "fine-tuned Gemma + DeepSeek V3.1; remaining papers used base Gemma + GPT-oss-20b. "
    "Base model scores are less calibrated (r=0.10 vs r=0.26 with human ratings). "
    "Results reflect a mixed-model pipeline and should be interpreted accordingly."
)

# ── Regime list ───────────────────────────────────────────────────────────────
all_regimes = [HumanActual(), HumanScore(), HumanDisagree(lam),
               LLMCommittee(), LLMDeepSeek()]
all_years = sorted(eval_table["year"].unique().astype(int).tolist())
years = all_years if selected_year == "All years" else [int(selected_year)]

# Build dff: run all regimes live
live_rows = []
for year in years:
    for regime in all_regimes:
        try:
            live_rows.extend(compute_live(regime, year, mode, impute_zeros, exclude_top_decile, citation_source))
        except Exception:
            pass

dff = pd.DataFrame(live_rows)
if not dff.empty and selected_year == "All years":
    dff = dff.groupby(["regime","metric","mode"])[
        ["value","random_value","ideal_value","lift","drawdown"]
    ].mean().reset_index()

if dff.empty:
    st.warning("No data — re-run run_eval.py"); st.stop()

regimes   = dff["regime"].unique().tolist()
color_map = {r: COLORS.get(r, RANDOM_COLOR) for r in regimes}
metrics   = [m for m in METRIC_LABELS if m in dff["metric"].unique()]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# CitesBench — Reviewer Regime Comparison")
_mode_label = "Top-10% excluded (leakage-robust)" if exclude_top_decile else "Full citation pool"
st.markdown(f"<span style='color:{SUBTEXT};font-size:15px'>ICLR 2018–2020  ·  "
            f"{selected_year}  ·  {_mode_label}</span>", unsafe_allow_html=True)
st.markdown("")

st.markdown('<p class="section-header">Section 1 — Performance by Metric</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">Select a metric to compare regimes. '
            'Bars show % of ideal gap closed (higher = better, 100% = ideal); '
            'raw values shown in parentheses. '
            'The dashed line is random selection; the dotted line is the theoretical ceiling '
            '(top-N papers by citations). Full per-metric table below.</p>',
            unsafe_allow_html=True)

sel_metric = st.selectbox("Select metric", metrics, format_func=lambda m: METRIC_LABELS[m])

sub_rows = []
for r in regimes:
    sub = dff[(dff["regime"] == r) & (dff["metric"] == sel_metric)]
    if sub.empty: continue
    v, rand, ideal = sub["value"].values[0], sub["random_value"].values[0], sub["ideal_value"].values[0]
    gap = ideal - rand
    pct = (ideal - v) / gap * 100 if (show_drawdown and gap and not np.isnan(gap)) \
          else (v - rand) / gap * 100 if gap and not np.isnan(gap) else 0
    sub_rows.append({"regime": r,
                     "label": r.replace("Human (","").rstrip(")"),
                     "score": pct, "value": v, "rand": rand, "ideal": ideal})

sub_df   = pd.DataFrame(sub_rows).sort_values("score", ascending=show_drawdown)
regime_order = sub_df["regime"].tolist()
rand_val = sub_df["rand"].mean(); ideal_val = sub_df["ideal"].mean()

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=sub_df["score"], y=sub_df["label"], orientation="h",
    marker_color=[color_map.get(r,"#94A3B8") for r in sub_df["regime"]],
    marker_line_width=0,
    text=[f"{p:.0f}%  ({v:.3f})" for p, v in zip(sub_df["score"], sub_df["value"])],
    textposition="outside", textfont=dict(size=11, color=TEXT),
    customdata=sub_df["value"],
    hovertemplate="%{y}<br>%{x:.1f}%  (raw: %{customdata:.3f})<extra></extra>",
))
fig2.add_vline(x=0 if not show_drawdown else 100, line_dash="dash",
               line_color=RANDOM_COLOR, line_width=1.5,
               annotation_text=f"Random ({rand_val:.3f})",
               annotation_position="top right",
               annotation_font=dict(size=10, color=RANDOM_COLOR))
fig2.add_vline(x=100 if not show_drawdown else 0, line_dash="dot",
               line_color=IDEAL_COLOR, line_width=1.5,
               annotation_text=f"Ideal ({ideal_val:.3f})",
               annotation_position="top left",
               annotation_font=dict(size=10, color=IDEAL_COLOR))
fig2.update_layout(
    height=40 * len(sub_df) + 80, margin=dict(l=0, r=160, t=8, b=32),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    xaxis=dict(range=[-5, 125], autorange="reversed" if show_drawdown else True,
               showgrid=True, gridcolor=BORDER, zeroline=False,
               ticksuffix="%", tickfont=dict(color=SUBTEXT, size=11)),
    yaxis=dict(showgrid=False, tickfont=dict(color=TEXT, size=12)),
)
st.plotly_chart(fig2, use_container_width=True)

# Summary table
pivot = dff[dff["regime"].isin(regimes)].pivot_table(
    index="regime", columns="metric", values="value")
display_cols = [m for m in METRIC_LABELS if m in pivot.columns]
pivot = pivot.loc[[r for r in regime_order if r in pivot.index], display_cols].round(3)
pivot.rename(columns=METRIC_SHORT, inplace=True)
pivot.index = pivot.index.str.replace("Human (", "").str.rstrip(")")

rand_row  = {METRIC_SHORT[m]: dff[dff["metric"]==m]["random_value"].mean()
             for m in display_cols if m in METRIC_SHORT}
ideal_row = {METRIC_SHORT[m]: dff[dff["metric"]==m]["ideal_value"].mean()
             for m in display_cols if m in METRIC_SHORT}
ref = pd.DataFrame([rand_row, ideal_row],
                   index=["— Random baseline", "— Ideal ceiling"]).round(3)
st.dataframe(pd.concat([pivot, ref]), use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REGIME DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 2 — Regime Deep Dive</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">Select a regime to understand <i>how</i> it differs from '
            'both the citation ideal and human AC decisions. Papers are split into four '
            'quadrants based on whether the regime selected them and whether they appear in '
            'the top-N by citations (the ideal set).</p>', unsafe_allow_html=True)

dive_regime_name = st.selectbox("Select regime to inspect", regimes,
    format_func=lambda r: r.replace("Human (","").rstrip(")"))
dive_year_opt = st.selectbox("Year", ["All years"] + all_years, key="dive_year")
dive_years = all_years if dive_year_opt == "All years" else [dive_year_opt]

# Build regime object
def get_regime(name):
    for r in [HumanActual(), HumanScore(), HumanDisagree(lam),
              LLMCommittee(), LLMDeepSeek()]:
        if r.name == name:
            return r
dive_regime = get_regime(dive_regime_name)

@st.cache_data
def compute_quadrants(regime_name, lam, years_tuple, mode, impute_zeros, exclude_top_decile=False, src="OpenAlex"):
    """Returns pool_df with 'quadrant' column and sets of IDs."""
    pools, regime_ids, ideal_ids, ac_ids = [], set(), set(), set()
    for year in years_tuple:
        pool = prepare_pool(year, mode, impute_zeros, exclude_top_decile, src)
        _, _, n = get_baselines(year, mode, impute_zeros, exclude_top_decile, src)
        regime = get_regime(regime_name)
        try:
            sel = set(regime.select(pool, n))
        except Exception:
            sel = set()
        known = pool.dropna(subset=["openalex_citations"])
        cutoff_n = math.ceil(1 * n / 100 * 10)  # top-N = same N as acceptance
        cutoff_n = n  # ideal = top-N by citations = same count as selected
        ideal = set(known.nlargest(n, "openalex_citations")["paper_id"])
        ac    = set(pool[pool["decision"].str.startswith("Accept", na=False)]["paper_id"])
        regime_ids |= sel
        ideal_ids  |= ideal
        ac_ids     |= ac
        pools.append(pool)

    full_pool = pd.concat(pools, ignore_index=True).drop_duplicates("paper_id")

    def quad(pid):
        in_r = pid in regime_ids
        in_i = pid in ideal_ids
        if in_r and in_i:    return "regime ∩ ideal"
        elif in_i:           return "ideal only"
        elif in_r:           return "regime only"
        else:                return "neither"

    full_pool["quadrant"] = full_pool["paper_id"].map(quad)
    return full_pool, regime_ids, ideal_ids, ac_ids

with st.spinner("Computing quadrants..."):
    pool_df, regime_ids, ideal_ids, ac_ids = compute_quadrants(
        dive_regime_name, lam, tuple(dive_years), mode, impute_zeros, exclude_top_decile, citation_source)

quad_order = ["regime ∩ ideal", "ideal only", "regime only", "neither"]

# ── 3a. CONFUSION MATRICES ────────────────────────────────────────────────────
st.markdown("#### 2a. Confusion Matrices")
st.markdown('<p class="explainer">'
            '<b>Left (vs Citation Ideal)</b>: treats the top-N papers by citations as '
            'ground truth positives. A high-performing regime should have high TP and low FN — '
            'it finds the papers that actually turned out to be impactful. '
            '<b>Right (vs AC Decisions)</b>: treats human AC decisions as ground truth. '
            'Compare the two: a regime that agrees with humans but not with citations is '
            'replicating human bias; one that agrees with citations but not humans is finding '
            'diamonds AC missed.</p>', unsafe_allow_html=True)

def cm_stats(selected, ground_truth, all_ids):
    tp = len(selected & ground_truth)
    fp = len(selected - ground_truth)
    fn = len(ground_truth - selected)
    tn = len(all_ids - selected - ground_truth)
    n  = tp + fp + fn + tn
    prec   = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, recall=recall, f1=f1, n=n)

def cm_figure(stats, title, gt_label):
    """Draws a 2×2 confusion matrix using shapes for full color control."""
    # Semantic cell colors
    cell_cfg = {
        (0, 0): ("#166534", "#DCFCE7", f"TP\n{stats['tp']:,}",  "Regime ∩ ground truth"),
        (0, 1): ("#9A3412", "#FEE2E2", f"FP\n{stats['fp']:,}",  "Regime − ground truth"),
        (1, 0): ("#92400E", "#FEF3C7", f"FN\n{stats['fn']:,}",  "Missed by regime"),
        (1, 1): ("#374151", "#F1F5F9", f"TN\n{stats['tn']:,}",  "Correctly excluded"),
    }
    col_labels = [f"In {gt_label}", f"Not in {gt_label}"]
    row_labels = ["Selected", "Not selected"]

    shapes, annotations = [], []
    for (row, col), (fg, bg, label, hover) in cell_cfg.items():
        x0, x1 = col, col + 1
        y0, y1 = row, row + 1
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           fillcolor=bg, line=dict(color="white", width=2)))
        annotations.append(dict(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            text=label.replace("\n", "<br>"),
            font=dict(size=13, color=fg, family="monospace"),
            showarrow=False, align="center",
        ))

    # Column headers (x-axis labels)
    for i, lbl in enumerate(col_labels):
        annotations.append(dict(x=i + 0.5, y=2.15, text=f"<b>{lbl}</b>",
                                font=dict(size=11, color=SUBTEXT), showarrow=False))
    # Row headers (y-axis labels)
    for i, lbl in enumerate(row_labels):
        annotations.append(dict(x=-0.22, y=i + 0.5, text=lbl,
                                font=dict(size=11, color=TEXT), showarrow=False,
                                xanchor="right"))
    # Stats footer
    annotations.append(dict(
        x=1, y=-0.12, xref="paper", yref="paper", showarrow=False,
        text=f"Precision {stats['prec']:.2f}  ·  Recall {stats['recall']:.2f}  ·  F1 {stats['f1']:.2f}",
        font=dict(size=11, color=SUBTEXT), xanchor="center",
    ))

    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=TEXT), x=0),
        shapes=shapes, annotations=annotations,
        height=280, margin=dict(l=80, r=10, t=36, b=48),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 2], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 2.4], showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
    )
    return fig

all_ids = set(pool_df["paper_id"])
cm_ideal = cm_stats(regime_ids, ideal_ids, all_ids)
cm_ac    = cm_stats(regime_ids, ac_ids,    all_ids)

col1, col2 = st.columns(2)
col1.plotly_chart(cm_figure(cm_ideal, "vs Citation Ideal (top-N by citations)", "citation ideal"),
                  use_container_width=True)
col2.plotly_chart(cm_figure(cm_ac, "vs AC Decisions (human ground truth)", "AC decisions"),
                  use_container_width=True)

st.markdown("---")

# ── 3b. FLIPPED PAPERS: CITATION RESIDUALS vs AC ─────────────────────────────
st.markdown("#### 2b. Flipped Papers vs Human AC — Citation Residuals")
st.markdown('<p class="explainer">'
            'Papers where this regime disagrees with human AC decisions, split by direction. '
            '<b>FP</b> = regime accepts, AC rejects. <b>FN</b> = AC accepts, regime rejects. '
            'Y-axis = citation residual after regressing out mean reviewer score per year — '
            'positive means more impactful than the score predicts, negative means less. '
            'If FP residuals are above zero, the regime finds underscored high-impact papers AC missed. '
            'If FN residuals are above zero, the regime is dropping impactful papers.</p>',
            unsafe_allow_html=True)

_resid_df = pool_df.dropna(subset=["mean_rating", "openalex_citations"]).copy()
_resid_df["log_cites"] = np.log1p(_resid_df["openalex_citations"])

from scipy.stats import linregress
_parts = []
for _yr, _grp in _resid_df.groupby("year"):
    _grp = _grp.copy()
    if len(_grp) >= 5:
        _sl, _ic, *_ = linregress(_grp["mean_rating"], _grp["log_cites"])
        _grp["cite_resid"] = _grp["log_cites"] - (_ic + _sl * _grp["mean_rating"])
    else:
        _grp["cite_resid"] = _grp["log_cites"] - _grp["log_cites"].mean()
    _parts.append(_grp)
_resid_df = pd.concat(_parts, ignore_index=True)

_AC_LABELS = {
    "FP — regime only": ("FP — regime only<br>(regime accepts, AC rejects)", "#10B981"),
    "FN — AC only":     ("FN — AC only<br>(AC accepts, regime rejects)",     "#F59E0B"),
    "TP — both accept": ("TP — both accept",                                  "#2563EB"),
    "TN — both reject": ("TN — both reject",                                  "#CBD5E1"),
}

def _vs_ac(pid):
    in_r = pid in regime_ids
    in_a = pid in ac_ids
    if in_r and in_a: return "TP — both accept"
    elif in_r:        return "FP — regime only"
    elif in_a:        return "FN — AC only"
    else:             return "TN — both reject"

_resid_df["vs_ac"] = _resid_df["paper_id"].map(_vs_ac)

if dive_regime_name == "Human (AC decisions)":
    st.info("FP/FN are both zero for Human (AC decisions) — this regime IS the AC baseline.")
else:
    fig_resid = go.Figure()
    for key, (label, color) in _AC_LABELS.items():
        sub = _resid_df[_resid_df["vs_ac"] == key]
        if sub.empty: continue
        fig_resid.add_trace(go.Box(
            y=sub["cite_resid"],
            name=label,
            marker_color=color,
            line_color=color,
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.25)",
            boxmean=True,
            whiskerwidth=0.6,
            hovertemplate="%{y:.3f}<extra>" + key + "</extra>",
        ))
    fig_resid.add_hline(y=0, line_dash="dash", line_color=SUBTEXT, line_width=1)
    fig_resid.update_layout(
        height=340, margin=dict(l=0, r=0, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
        yaxis=dict(title="Citation residual (log scale)", zeroline=False,
                   showgrid=True, gridcolor=BORDER, tickfont=dict(size=11, color=SUBTEXT)),
        xaxis=dict(showticklabels=False, showgrid=False),
    )
    st.plotly_chart(fig_resid, use_container_width=True)
st.markdown("---")

# ── 3c. SCATTER: HUMAN SCORE vs LOG CITATIONS ────────────────────────────────
st.markdown("#### 2c. Human Reviewer Score vs Citations")
st.markdown('<p class="explainer">'
            'Each dot is a paper. X-axis = mean human reviewer rating (1–10); '
            'Y-axis = log(1 + citations). Color = quadrant. '
            'This reveals systematic biases: if the regime misses papers in the upper-left '
            '(low rating, high citations) it is over-relying on reviewer scores. If its '
            'unique picks cluster in the lower-right (high rating, low citations) it is '
            'selecting papers reviewers liked but that had little impact.</p>',
            unsafe_allow_html=True)

scatter_df = pool_df.dropna(subset=["mean_rating","openalex_citations"]).copy()
scatter_df["log_cites"] = np.log1p(scatter_df["openalex_citations"])

fig_sc = go.Figure()
for quad in quad_order:
    sub = scatter_df[scatter_df["quadrant"] == quad]
    fig_sc.add_trace(go.Scatter(
        x=sub["mean_rating"], y=sub["log_cites"], mode="markers",
        name=f"{quad} (n={len(sub)})",
        marker=dict(color=QUAD_COLORS[quad], size=5, opacity=0.6,
                    line=dict(width=0)),
        hovertemplate="Rating: %{x:.1f}<br>log cites: %{y:.2f}<extra>" + quad + "</extra>",
    ))
fig_sc.update_layout(
    height=350, margin=dict(l=0, r=0, t=8, b=8),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
    xaxis=dict(title="Mean human reviewer rating", showgrid=True, gridcolor=BORDER,
               tickfont=dict(size=11, color=SUBTEXT)),
    yaxis=dict(title="log(1 + citations)", showgrid=True, gridcolor=BORDER,
               tickfont=dict(size=11, color=SUBTEXT)),
)
st.plotly_chart(fig_sc, use_container_width=True)
st.markdown("---")

# ── 3d. MISSED GEMS + HUMAN CONSENSUS WRONG ─────────────────────────────────
st.markdown("#### 2d. Missed Gems and Human Consensus Errors")
st.markdown('<p class="explainer">'
            '<b>Missed gems</b> (left): high-citation papers the regime failed to select — '
            'in the citation ideal but not picked. Where available, reviewer rejection tags '
            'show why they were passed over. '
            '<b>Human consensus errors</b> (right): papers selected by both this regime '
            '<i>and</i> AC decisions, yet absent from the citation ideal — cases where human '
            'consensus agreed but the citation signal says they were wrong.</p>',
            unsafe_allow_html=True)

merged = pool_df.dropna(subset=["openalex_citations"])

# Fetch rejection_tags directly from eval_table to avoid stale cache issues
_rej_map = eval_table.set_index("paper_id")["rejection_tags"] \
    if "rejection_tags" in eval_table.columns else None

# Missed gems: ideal only, sorted by citations
missed_raw = merged[merged["quadrant"] == "ideal only"].nlargest(10, "openalex_citations")[
    ["paper_id", "title", "year", "openalex_citations", "mean_rating"]].copy()
missed_raw = missed_raw.round({"openalex_citations": 0, "mean_rating": 2})
if _rej_map is not None:
    missed_raw["rejection_tags"] = missed_raw["paper_id"].map(_rej_map).fillna("(accepted by AC — no rejection tag)")
missed_raw = missed_raw.drop(columns=["paper_id"])
missed_raw.columns = (["Title", "Year", "Citations", "Avg rating"] +
                      (["Rejection tags"] if _rej_map is not None else []))

# Human consensus errors: regime ∩ AC but NOT in citation ideal
consensus_wrong = merged[
    merged["paper_id"].isin(regime_ids & ac_ids) &
    ~merged["paper_id"].isin(ideal_ids)
].nlargest(10, "openalex_citations")[
    ["title", "year", "openalex_citations", "mean_rating"]].copy()
consensus_wrong = consensus_wrong.round({"openalex_citations": 0, "mean_rating": 2})
consensus_wrong.columns = ["Title", "Year", "Citations", "Avg rating"]

col_l, col_r = st.columns(2)
n_missed = len(merged[merged["quadrant"] == "ideal only"])
n_wrong  = len(merged[merged["paper_id"].isin(regime_ids & ac_ids) & ~merged["paper_id"].isin(ideal_ids)])
col_l.markdown(f"**Missed gems** — high-impact papers regime didn't select (n={n_missed})")
col_l.dataframe(missed_raw, use_container_width=True, hide_index=True)
col_r.markdown(f"**Human consensus errors** — both regime & AC accepted, but below citation ideal (n={n_wrong})")
col_r.dataframe(consensus_wrong, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COVARIATE HETEROGENEITY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 3 — Covariate Heterogeneity</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">'
            'Do LLM committee results hold across subgroups, or does the advantage concentrate '
            'in a particular type of paper or author team? This section first audits coverage '
            '(what fraction of papers have each covariate), then asks whether regime recall '
            'advantages are stable or driven by a specific subgroup.</p>',
            unsafe_allow_html=True)

_RDD_PATH = "data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"
_COV_PATH = "outputs/paper_author_covariates.csv"
_ET_PATH  = "outputs/eval_table.csv"

@st.cache_data
def _load_hetero_full(src="OpenAlex", _v=3):
    cov_path = "outputs/paper_author_covariates.csv"
    if not os.path.exists(cov_path):
        return None
    et  = _et(src)
    cov = pd.read_csv(cov_path)
    df  = et.merge(cov, on="paper_id", how="left")
    df["log_cites"] = np.log1p(df["openalex_citations"].fillna(0))
    return df

_het_df = _load_hetero_full(citation_source)

if _het_df is None:
    st.info("Run `python src/build_author_covariates.py` to enable this section.")
else:
    total_papers = len(_het_df)
    n_matched = _het_df["top_institution_flag"].notna().sum()

    # ── 3a. COVERAGE TABLE ────────────────────────────────────────────────────
    st.markdown("#### 3a. Covariate Coverage")
    st.markdown('<p class="explainer">'
                'Covariate data comes from OpenAlex author profiles. '
                '28.5% of papers have no author match (likely lower-tier venues with fewer indexed authors). '
                'These papers have a much lower acceptance rate (12.9%) vs matched papers (42%), '
                'so coverage is <b>non-random</b> — estimates for author-flag covariates apply to '
                'a higher-quality slice of the pool.</p>', unsafe_allow_html=True)

    cov_rows = [
        {"Covariate": "Any author data",       "N with data": n_matched,
         "% covered": f"{n_matched/total_papers:.1%}",
         "Note": "Missing = no OpenAlex author match; acceptance rate 12.9% vs 42% for matched"},
        {"Covariate": "top_institution_flag",  "N with data": n_matched,
         "% covered": f"{n_matched/total_papers:.1%}",
         "Note": "Imputed 0 when institution name missing (30% of matched authors)"},
        {"Covariate": "industry_flag",         "N with data": n_matched,
         "% covered": f"{n_matched/total_papers:.1%}",
         "Note": "Imputed 0 when institution name missing"},
        {"Covariate": "us_team_flag",          "N with data": n_matched,
         "% covered": f"{n_matched/total_papers:.1%}",
         "Note": "Imputed 0 when country missing (~30% of matched authors)"},
        {"Covariate": "max_h_index",           "N with data": int(_het_df["max_h_index"].notna().sum()),
         "% covered": f"{_het_df['max_h_index'].notna().mean():.1%}",
         "Note": "h_index missing for some authors in OpenAlex"},
        {"Covariate": "team_size",             "N with data": n_matched,
         "% covered": f"{n_matched/total_papers:.1%}",
         "Note": "Count of authors on the OpenReview submission"},
    ]
    st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 3b. RECALL ADVANTAGE BY SUBGROUP ─────────────────────────────────────
    st.markdown("#### 3b. LLM Committee recall@10% advantage vs Human AC — by subgroup")
    st.markdown('<p class="explainer">'
                'Each bar shows the percentage-point gap between LLM Committee recall@10% and '
                'Human AC recall@10% within a subgroup. Positive = LLM outperforms. '
                'If the advantage is stable across "yes" and "no" halves, the result is not '
                'driven by that covariate.</p>', unsafe_allow_html=True)

    @st.cache_data
    def _recall_by_subgroup(years_tuple, mode_key, impute, excl, src="OpenAlex"):
        llm = LLMCommittee(); ac = HumanActual()
        rows = []
        base_df = pd.concat([
            prepare_pool(yr, mode_key, impute, excl, src) for yr in years_tuple
        ], ignore_index=True)
        cov = pd.read_csv("outputs/paper_author_covariates.csv")
        base_df = base_df.merge(cov, on="paper_id", how="left")

        for gcol in ["field", "top_institution_flag", "industry_flag", "us_team_flag"]:
            for gval, grp in base_df.groupby(gcol):
                if grp.empty: continue
                # Need to compute recall per year then average
                llm_rec, ac_rec = [], []
                for yr in years_tuple:
                    g = grp[grp["year"] == yr].copy()
                    n_acc = int(g["decision"].str.startswith("Accept", na=False).sum())
                    if n_acc < 3: continue
                    try:
                        llm_rec.append(compute_metrics(llm.select(g, n_acc), g, mode_key)["recall_at_10"])
                        ac_rec.append(compute_metrics(ac.select(g, n_acc), g, mode_key)["recall_at_10"])
                    except Exception:
                        pass
                if not llm_rec: continue
                label = str(gval).replace("_", " ")
                if gcol != "field":
                    label = gcol.replace("_flag","").replace("_"," ") + f"={'yes' if gval else 'no'}"
                rows.append({
                    "covariate": gcol, "value": str(gval), "label": label,
                    "llm_recall": np.mean(llm_rec),
                    "ac_recall":  np.mean(ac_rec),
                    "advantage_pp": (np.mean(llm_rec) - np.mean(ac_rec)) * 100,
                    "n_years": len(llm_rec),
                })
        return pd.DataFrame(rows)

    _sub_df = _recall_by_subgroup(tuple(years), mode, impute_zeros, exclude_top_decile, citation_source)

    if not _sub_df.empty:
        # Plot field separately (categorical, not binary)
        _field_df = _sub_df[_sub_df["covariate"] == "field"].sort_values("advantage_pp")
        _flag_df  = _sub_df[_sub_df["covariate"] != "field"].sort_values("advantage_pp")

        col3b1, col3b2 = st.columns([1, 1.3])

        with col3b1:
            st.markdown("**By field**")
            fig_adv_field = go.Figure()
            fig_adv_field.add_trace(go.Bar(
                x=_field_df["advantage_pp"],
                y=_field_df["label"].str.replace("_"," "),
                orientation="h",
                marker_color=[COLORS["LLM Committee (Gemma)"] if v >= 0 else COLORS["Human (AC decisions)"]
                              for v in _field_df["advantage_pp"]],
                marker_line_width=0,
                text=[f"{v:+.1f}pp" for v in _field_df["advantage_pp"]],
                textposition="outside", textfont=dict(size=10, color=TEXT),
                hovertemplate="%{y}: LLM advantage = %{x:+.1f}pp<extra></extra>",
            ))
            fig_adv_field.add_vline(x=0, line_color=SUBTEXT, line_width=1.5)
            fig_adv_field.update_layout(
                height=220, margin=dict(l=0, r=60, t=4, b=4),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="pp vs Human AC", showgrid=True, gridcolor=BORDER,
                           ticksuffix="pp", tickfont=dict(size=10, color=SUBTEXT)),
                yaxis=dict(showgrid=False, tickfont=dict(size=10, color=TEXT)),
                showlegend=False,
            )
            st.plotly_chart(fig_adv_field, use_container_width=True)

        with col3b2:
            st.markdown("**By author flag (yes vs no)**")
            fig_adv_flag = go.Figure()
            fig_adv_flag.add_trace(go.Bar(
                x=_flag_df["advantage_pp"],
                y=_flag_df["label"],
                orientation="h",
                marker_color=[COLORS["LLM Committee (Gemma)"] if v >= 0 else COLORS["Human (AC decisions)"]
                              for v in _flag_df["advantage_pp"]],
                marker_line_width=0,
                text=[f"{v:+.1f}pp  LLM={r:.0%} / AC={a:.0%}"
                      for v, r, a in zip(_flag_df["advantage_pp"], _flag_df["llm_recall"], _flag_df["ac_recall"])],
                textposition="outside", textfont=dict(size=10, color=TEXT),
                hovertemplate="%{y}: LLM advantage = %{x:+.1f}pp<extra></extra>",
            ))
            fig_adv_flag.add_vline(x=0, line_color=SUBTEXT, line_width=1.5)
            fig_adv_flag.update_layout(
                height=280, margin=dict(l=0, r=200, t=4, b=4),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="pp vs Human AC", showgrid=True, gridcolor=BORDER,
                           ticksuffix="pp", tickfont=dict(size=10, color=SUBTEXT)),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color=TEXT)),
                showlegend=False,
            )
            st.plotly_chart(fig_adv_flag, use_container_width=True)

    st.markdown("---")

    # ── 3c. CONTROLLED RECALL — field×year FE regression ─────────────────────
    st.markdown("#### 3c. Does controlling for field affect regime rankings?")
    st.markdown('<p class="explainer">'
                'Within-field recall@10% for each regime, averaged across years. '
                'If ranking is stable across fields, covariate control does not change the conclusion. '
                'Right: a single pooled regression — citation_pct_rank ~ committee_rating × field + year FE '
                '— replaces a per-field×year Spearman-ρ scan (which silently dropped any cell with &lt;5 '
                'papers) with one model and a proper significance test of whether the rating→citation '
                'slope actually differs by field.</p>', unsafe_allow_html=True)

    @st.cache_data
    def _field_recall(years_tuple, mode_key, impute, excl, src="OpenAlex"):
        rows = []
        _regs = [LLMCommittee(), HumanActual(), HumanScore()]
        et_sub = _et(src)
        for field in et_sub["field"].dropna().unique():
            for yr in years_tuple:
                pf = prepare_pool(yr, mode_key, impute, excl, src)
                pf = pf[pf["field"] == field].copy()
                n_acc = int(pf["decision"].str.startswith("Accept", na=False).sum())
                if n_acc < 3: continue
                for reg in _regs:
                    try:
                        sel = reg.select(pf, n_acc)
                        m = compute_metrics(sel, pf, mode_key)
                        rows.append({"field": field, "year": yr, "regime": reg.name,
                                     "recall_at_10": m["recall_at_10"], "n": len(pf)})
                    except Exception:
                        pass
        return pd.DataFrame(rows)

    _fc_df = _field_recall(tuple(years), mode, impute_zeros, exclude_top_decile, citation_source)

    if not _fc_df.empty:
        # Grouped bar: recall@10% by field, three regimes
        _mean_fc = _fc_df.groupby(["field","regime"])["recall_at_10"].mean().reset_index()
        fig_fc = go.Figure()
        _reg_order = [LLMCommittee().name, HumanActual().name, HumanScore().name]
        for reg in _reg_order:
            sub = _mean_fc[_mean_fc["regime"] == reg]
            if sub.empty: continue
            fig_fc.add_trace(go.Bar(
                x=sub["field"].str.replace("_"," "), y=sub["recall_at_10"],
                name=reg.replace("Human (","").rstrip(")"),
                marker_color=color_map.get(reg, RANDOM_COLOR),
                marker_line_width=0,
            ))
        fig_fc.update_layout(
            barmode="group", height=280,
            margin=dict(l=0, r=0, t=4, b=4),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
            xaxis=dict(tickangle=-15, tickfont=dict(size=10, color=TEXT), showgrid=False),
            yaxis=dict(title="Recall@10%", showgrid=True, gridcolor=BORDER,
                       tickformat=".0%", tickfont=dict(size=10, color=SUBTEXT)),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("---")

        # ── Pooled field×year FE regression (replaces the old per-cell ρ scan) ──
        st.markdown("###### citation_pct_rank ~ committee_rating × field + year FE")
        from fuzzy_rdd import cat_dummies, wls_hc1, wls_hc1_full
        from scipy.stats import t as _tdist

        def _stars(p):
            if np.isnan(p): return ""
            if p < 0.01: return "***"
            if p < 0.05: return "**"
            if p < 0.10: return "*"
            return ""

        def _fmt_cell(beta, se, p):
            return f"{beta:+.4f}{_stars(p)}<br><span style='color:{SUBTEXT};font-size:11px'>({se:.4f})</span>"

        _reg_df = eval_table[eval_table["year"].isin(years)].dropna(
            subset=["committee_rating", "citation_pct_rank", "field"]).copy()

        if len(_reg_df) >= 30:
            _fields = sorted(_reg_df["field"].unique())
            _ref_field = _fields[0]
            _n = len(_reg_df)
            _fld_d = cat_dummies(_reg_df["field"].values)
            _yr_d = cat_dummies(_reg_df["year"].values)
            _x = _reg_df["committee_rating"].values
            _y = _reg_df["citation_pct_rank"].values

            def _fit(X):
                beta, cov = wls_hc1_full(X, _y, np.ones(_n))
                se = np.sqrt(np.diag(cov))
                df_resid = _n - X.shape[1]
                p = np.array([2 * (1 - _tdist.cdf(abs(b / s), df=df_resid)) if s > 0 else np.nan
                              for b, s in zip(beta, se)])
                yhat = X @ beta
                r2 = 1 - np.sum((_y - yhat) ** 2) / np.sum((_y - _y.mean()) ** 2)
                return beta, se, p, cov, r2

            # Model (1): main effects only — committee_rating + field FE + year FE
            _X0 = np.column_stack([np.ones(_n), _x, _fld_d, _yr_d])
            _b0, _se0, _p0, _cov0, _r20 = _fit(_X0)

            # Model (2): + committee_rating × field interaction
            _inter = [(_fld_d[:, j] * _x) for j in range(_fld_d.shape[1])]
            _X1 = np.column_stack([_X0] + _inter) if _inter else _X0
            _b1, _se1, _p1, _cov1, _r21 = _fit(_X1)
            _k0 = _X0.shape[1]

            # ── Standard regression table (stargazer-style) ─────────────────
            col3c_tbl, col3c_plot = st.columns([1.1, 1])

            with col3c_tbl:
                _rows_html = [
                    ("committee_rating", _b0[1], _se0[1], _p0[1], _b1[1], _se1[1], _p1[1]),
                ]
                for j, f in enumerate(_fields[1:]):
                    _rows_html.append((
                        f"Field: {f.replace('_',' ')}",
                        _b0[2 + j], _se0[2 + j], _p0[2 + j],
                        _b1[2 + j], _se1[2 + j], _p1[2 + j],
                    ))
                _inter_rows = [("&nbsp;&nbsp;(reference field)", None, None, None, None, None, None)]
                for j, f in enumerate(_fields[1:]):
                    _inter_rows.append((
                        f"committee_rating × {f.replace('_',' ')}",
                        None, None, None,
                        _b1[_k0 + j], _se1[_k0 + j], _p1[_k0 + j],
                    ))

                def _row(label, b0, s0, p0, b1, s1, p1):
                    c0 = _fmt_cell(b0, s0, p0) if b0 is not None else ""
                    c1 = _fmt_cell(b1, s1, p1) if b1 is not None else ""
                    return (f"<tr><td style='padding:3px 8px;text-align:left'>{label}</td>"
                           f"<td style='padding:3px 8px;text-align:center'>{c0}</td>"
                           f"<td style='padding:3px 8px;text-align:center'>{c1}</td></tr>")

                _html = [f"<table style='width:100%;border-collapse:collapse;font-size:12.5px;color:{TEXT}'>",
                        f"<tr style='border-top:2px solid {TEXT};border-bottom:1px solid {BORDER}'>",
                        "<td style='padding:3px 8px'></td>",
                        "<td style='padding:3px 8px;text-align:center;font-weight:600'>(1)</td>",
                        "<td style='padding:3px 8px;text-align:center;font-weight:600'>(2)</td></tr>"]
                for r in _rows_html:
                    _html.append(_row(*r))
                _html.append(f"<tr><td colspan='3' style='padding:2px 8px;font-weight:600;"
                             f"border-top:1px solid {BORDER}'>committee_rating × field</td></tr>")
                for r in _inter_rows:
                    _html.append(_row(*r))
                _html.append(f"<tr style='border-top:1px solid {BORDER}'>"
                             f"<td style='padding:3px 8px'>N</td>"
                             f"<td style='padding:3px 8px;text-align:center'>{_n:,}</td>"
                             f"<td style='padding:3px 8px;text-align:center'>{_n:,}</td></tr>")
                _html.append(f"<tr><td style='padding:3px 8px'>R²</td>"
                             f"<td style='padding:3px 8px;text-align:center'>{_r20:.3f}</td>"
                             f"<td style='padding:3px 8px;text-align:center'>{_r21:.3f}</td></tr>")
                _html.append(f"<tr><td style='padding:3px 8px'>Year FE</td>"
                             f"<td style='padding:3px 8px;text-align:center'>Yes</td>"
                             f"<td style='padding:3px 8px;text-align:center'>Yes</td></tr>")
                _html.append(f"<tr style='border-bottom:2px solid {TEXT}'>"
                             f"<td style='padding:3px 8px'>Field FE</td>"
                             f"<td style='padding:3px 8px;text-align:center'>Yes</td>"
                             f"<td style='padding:3px 8px;text-align:center'>Yes</td></tr>")
                _html.append("</table>")
                st.markdown("".join(_html), unsafe_allow_html=True)
                st.caption(
                    f"Dependent variable: citation_pct_rank. HC1-robust SEs in parentheses. "
                    f"*p&lt;0.10, **p&lt;0.05, ***p&lt;0.01. Reference field: {_ref_field.replace('_',' ')}. "
                    "(1) pooled slope; (2) allows the slope to vary by field."
                )

            # ── Coefficient plot: implied slope per field, with 95% CI ───────
            with col3c_plot:
                _plot_rows = []
                for j, f in enumerate([_ref_field] + list(_fields[1:])):
                    if f == _ref_field:
                        slope, var = _b1[1], _cov1[1, 1]
                    else:
                        idx = _k0 + (j - 1)
                        slope = _b1[1] + _b1[idx]
                        var = _cov1[1, 1] + _cov1[idx, idx] + 2 * _cov1[1, idx]
                    se = np.sqrt(max(var, 0))
                    _plot_rows.append({"field": f.replace("_", " "), "slope": slope,
                                       "lo": slope - 1.96 * se, "hi": slope + 1.96 * se})
                _plot_df = pd.DataFrame(_plot_rows).sort_values("slope")
                fig_coef = go.Figure()
                fig_coef.add_trace(go.Scatter(
                    x=_plot_df["slope"], y=_plot_df["field"], mode="markers",
                    marker=dict(color=COLORS["LLM Committee (Gemma)"], size=9),
                    error_x=dict(type="data", symmetric=False,
                                array=_plot_df["hi"] - _plot_df["slope"],
                                arrayminus=_plot_df["slope"] - _plot_df["lo"]),
                ))
                fig_coef.add_vline(x=0, line_dash="dot", line_color=SUBTEXT)
                fig_coef.update_layout(
                    title=dict(text="Implied slope by field (committee_rating → citation_pct_rank)",
                              font=dict(size=11, color=SUBTEXT), x=0),
                    height=280, margin=dict(l=0, r=10, t=32, b=4),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Slope (95% CI)", showgrid=True, gridcolor=BORDER,
                              tickfont=dict(size=10, color=SUBTEXT)),
                    yaxis=dict(tickfont=dict(size=10, color=TEXT)),
                )
                st.plotly_chart(fig_coef, use_container_width=True)
                st.caption("Field-specific slope = committee_rating coefficient + that field's "
                          "interaction term (Model 2); CI from the full HC1 covariance, not just "
                          "each coefficient's own SE, since it's a linear combination of two.")
        else:
            st.caption("Not enough field-tagged papers with both committee_rating and "
                      "citation_pct_rank to fit the pooled regression for this selection.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FUZZY RDD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 4 — Fuzzy RDD</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">'
            'Does ICLR acceptance <i>cause</i> additional citations, or does the correlation '
            'just reflect paper quality? Running variable = score_centered (mean_rating − '
            'year-specific cutoff). Treatment is fuzzy: ACs don\'t follow ratings perfectly. '
            'Instrument: above = 1{score_centered ≥ 0}. Outcome: log(1 + citations), '
            'from the citation source selected in the sidebar. OpenAlex undercounts accepted '
            'papers more than rejected ones (median 3.5× vs 2.0× vs S2) — biasing the OA-based '
            'LATE toward zero — so compare both sources here.</p>',
            unsafe_allow_html=True)

@st.cache_data
def _load_rdd_sample(src="OpenAlex", _v=3):
    rdd_path = "data/OpenAlex/openalex_rdd_dashboard.csv"
    if not os.path.exists(rdd_path):
        return None
    raw = pd.read_csv(rdd_path)
    dm = raw[
        (raw["year"] <= 2020)
        & raw["in_year_specific_rdd_sample"].astype(bool)
        & raw["openalex_matched"].astype(bool)
    ].copy()
    if src == "Semantic Scholar" and os.path.exists(S2_CSV):
        # same RDD sample, outcome swapped to S2 counts; papers without an
        # S2 match (~7%) are dropped rather than imputed
        s2 = pd.read_csv(S2_CSV)
        ok = s2[s2["s2_citations"].notna() &
                ((s2["method"] == "arxiv_batch") | (s2["title_sim"].fillna(0) >= 0.9))]
        dm = dm.merge(ok[["paper_id", "s2_citations"]].drop_duplicates("paper_id"),
                      on="paper_id", how="inner")
        dm["openalex_cited_by_count"] = dm["s2_citations"]
    dm["lcites"] = np.log1p(dm["openalex_cited_by_count"].fillna(0))
    dm["accepted"] = dm["accepted"].astype(int)
    _field = pd.read_csv("outputs/eval_table.csv")[["paper_id", "field"]]
    dm = dm.merge(_field, on="paper_id", how="left")
    return dm

@st.cache_data
def _rdd_year_bscatter(yr, n_bins=16, src="OpenAlex"):
    dm = _load_rdd_sample(src)
    if dm is None: return None, None
    sub = dm[dm["year"] == yr].copy()
    bw = sub["bandwidth"].iloc[0]
    sub = sub[sub["score_centered"].abs() <= bw].copy()
    bins = pd.cut(sub["score_centered"], bins=n_bins)
    bs = (sub.groupby(bins, observed=True)
             .agg(score=("score_centered","mean"),
                  p_accept=("accepted","mean"),
                  lcites=("lcites","mean"),
                  n=("score_centered","size"))
             .dropna().reset_index(drop=True))
    return bs, bw

@st.cache_data
def _rdd_all_specs(src="OpenAlex"):
    dm = _load_rdd_sample(src)
    if dm is None: return [], [], []
    from fuzzy_rdd import run_specs_constant, run_specs
    pooled_lc = [r for h in [0.5, 0.75, 1.0]
                 if (r := run_specs_constant(dm, h, f"Pooled ±{h:.2f}"))]
    pooled_field_lc = [r for h in [0.5, 0.75, 1.0]
                       if (r := run_specs_constant(dm, h, f"Pooled ±{h:.2f} (+field FE)", field_col="field"))]
    yr_lc = []
    for yr in [2018, 2019, 2020]:
        sub = dm[dm["year"] == yr]
        bw = sub["bandwidth"].iloc[0]
        r = run_specs_constant(sub, bw, f"{yr}  h={bw:.2f}")
        if r: yr_lc.append({**r, "year": yr})
    return pooled_lc, yr_lc, pooled_field_lc

_rdd_dm = _load_rdd_sample(citation_source)
if _rdd_dm is None:
    st.info("RDD data not found.")
else:
    _pooled_specs, _yr_specs, _pooled_field_specs = _rdd_all_specs(citation_source)

    # ── 4a. SCORE DISTRIBUTION — heaping diagnostic ─────────────────────────
    st.markdown("#### 4a. Score distribution by year — masspoints diagnostic")
    st.markdown('<p class="explainer">'
                '2020 reviewers used integer ratings; score_centered heaps at ±0.5. '
                'Almost no papers fall in (−0.25, 0.25), making local linear extrapolation '
                'at the cutoff unreliable for 2020. Local constant (mean difference within ±h) '
                'is robust to this and is the preferred estimator throughout.</p>',
                unsafe_allow_html=True)

    _YR_COLORS = {2018: COLORS["Human (AC decisions)"],
                  2019: COLORS["Human (score top-N)"],
                  2020: COLORS["LLM Committee (Gemma)"]}
    fig_dist = go.Figure()
    for yr in [2018, 2019, 2020]:
        sub = _rdd_dm[_rdd_dm["year"] == yr]
        fig_dist.add_trace(go.Histogram(
            x=sub["score_centered"], name=str(yr),
            nbinsx=48, opacity=0.6,
            marker_color=_YR_COLORS[yr], marker_line_width=0,
        ))
    fig_dist.add_vline(x=0, line_dash="dash", line_color=TEXT, line_width=1.5,
                       annotation_text="cutoff", annotation_font=dict(size=10, color=TEXT))
    fig_dist.update_layout(
        barmode="overlay", height=220, margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        xaxis=dict(title="score_centered", showgrid=True, gridcolor=BORDER,
                   tickfont=dict(size=10, color=SUBTEXT)),
        yaxis=dict(title="N papers", showgrid=True, gridcolor=BORDER,
                   tickfont=dict(size=10, color=SUBTEXT)),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── 4b. YEAR-SPECIFIC BINSCATTER ─────────────────────────────────────────
    st.markdown("#### 4b. First stage and reduced form by year")
    st.markdown('<p class="explainer">'
                'Each column = one year. Left = P(accepted) vs score_centered (first stage). '
                'Right = log(1+cites) vs score_centered (reduced form). '
                'Circles = binned means; vertical line = cutoff. '
                'Clear jumps in both panels confirm the instrument is valid.</p>',
                unsafe_allow_html=True)

    _yr_cols = st.columns(3)
    for i, yr in enumerate([2018, 2019, 2020]):
        bs, bw = _rdd_year_bscatter(yr, src=citation_source)
        with _yr_cols[i]:
            st.markdown(f"**{yr}** (h={bw:.2f})")
            if bs is None or bs.empty:
                st.caption("No data"); continue
            _bl = bs[bs["score"] < 0]
            _br = bs[bs["score"] >= 0]
            _cl = "#94A3B8"
            _cr = _YR_COLORS[yr]
            # first stage
            fig_yr_fs = go.Figure()
            for _side, _c in [(_bl, _cl), (_br, _cr)]:
                fig_yr_fs.add_trace(go.Scatter(
                    x=_side["score"], y=_side["p_accept"], mode="markers",
                    marker=dict(color=_c, size=7), showlegend=False,
                    hovertemplate="score=%{x:.2f}<br>P(acc)=%{y:.0%}<extra></extra>",
                ))
            fig_yr_fs.add_vline(x=0, line_dash="dash", line_color=SUBTEXT, line_width=1)
            fig_yr_fs.update_layout(
                height=180, margin=dict(l=0, r=0, t=4, b=4),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="score_centered", showgrid=True, gridcolor=BORDER,
                           zeroline=False, tickfont=dict(size=9, color=SUBTEXT)),
                yaxis=dict(title="P(accepted)", range=[0,1.05], tickformat=".0%",
                           showgrid=True, gridcolor=BORDER, tickfont=dict(size=9, color=SUBTEXT)),
            )
            st.plotly_chart(fig_yr_fs, use_container_width=True)
            # reduced form
            fig_yr_rf = go.Figure()
            for _side, _c in [(_bl, _cl), (_br, _cr)]:
                fig_yr_rf.add_trace(go.Scatter(
                    x=_side["score"], y=_side["lcites"], mode="markers",
                    marker=dict(color=_c, size=7), showlegend=False,
                    hovertemplate="score=%{x:.2f}<br>log(1+c)=%{y:.2f}<extra></extra>",
                ))
            fig_yr_rf.add_vline(x=0, line_dash="dash", line_color=SUBTEXT, line_width=1)
            fig_yr_rf.update_layout(
                height=180, margin=dict(l=0, r=0, t=4, b=4),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="score_centered", showgrid=True, gridcolor=BORDER,
                           zeroline=False, tickfont=dict(size=9, color=SUBTEXT)),
                yaxis=dict(title="log(1+cites)", showgrid=True, gridcolor=BORDER,
                           tickfont=dict(size=9, color=SUBTEXT)),
            )
            st.plotly_chart(fig_yr_rf, use_container_width=True)

    st.markdown("---")

    # ── 4c. LATE ESTIMATES ───────────────────────────────────────────────────
    st.markdown("#### 4c. Fuzzy LATE estimates (local constant, year FE, HC1 SEs)")
    st.markdown('<p class="explainer">'
                'Local constant = compares mean outcomes just above vs just below the cutoff within ±h. '
                'No slope extrapolation → robust to 2020 heaping. '
                'Year-specific bandwidth uses each year\'s own h. '
                'All F-stats > 100 confirm a strong instrument.</p>',
                unsafe_allow_html=True)

    col4c1, col4c2 = st.columns(2)

    with col4c1:
        st.markdown("**Year-specific** (year-specific bandwidth, preferred)")
        if _yr_specs:
            _yr_tbl = pd.DataFrame([{
                "Year": r["year"], "N": r["N"],
                "FS Δ": f"{r['FS_jump']:+.3f}",
                "FS F": r["FS_F"],
                "RF Δ": f"{r['RF_jump']:+.3f}",
                "LATE": r["LATE"],
                "95% CI": r["CI_95"],
            } for r in _yr_specs])
            st.dataframe(_yr_tbl, use_container_width=True, hide_index=True)

    with col4c2:
        st.markdown("**Pooled** (2018–2020 together, varying window)")
        if _pooled_specs:
            _pool_tbl = pd.DataFrame([{
                "Window": r["spec"], "N": r["N"],
                "FS Δ": f"{r['FS_jump']:+.3f}",
                "FS F": r["FS_F"],
                "RF Δ": f"{r['RF_jump']:+.3f}",
                "LATE": r["LATE"],
                "95% CI": r["CI_95"],
            } for r in _pooled_specs])
            st.dataframe(_pool_tbl, use_container_width=True, hide_index=True)

    if _pooled_field_specs:
        st.markdown("**Robustness: pooled + field FE** (additive covariate, not a per-field LATE)")
        st.markdown('<p class="explainer">'
                    'Field fixed effects added as an additive covariate to soak up citation-level '
                    'variance unrelated to treatment (same logic as covariate adjustment in an RCT) — '
                    'not a per-field treatment interaction, which the near-cutoff sample is too thin to '
                    'support reliably. Rows with no field tag are dropped, so N is smaller than the '
                    'no-field spec above; compare LATE and CI width directly.</p>', unsafe_allow_html=True)
        _pf_tbl = pd.DataFrame([{
            "Window": r["spec"], "N": r["N"],
            "FS Δ": f"{r['FS_jump']:+.3f}",
            "FS F": r["FS_F"],
            "RF Δ": f"{r['RF_jump']:+.3f}",
            "LATE": r["LATE"],
            "95% CI": r["CI_95"],
        } for r in _pooled_field_specs])
        st.dataframe(_pf_tbl, use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="explainer">'
        '<b>Interpretation</b>: LATE ≈ 0.8–1.2 log-citation units across all specs '
        '(≈ 2.2–3.3× citations). This is the causal effect of acceptance on citations '
        'for <i>complier</i> papers near the rating cutoff — papers that would not have been '
        'accepted if their score had been marginally lower. '
        'Effect is largest in 2018 (LATE=1.20) and declines with recency, consistent with '
        'older papers having had more time to accumulate citations. '
        '⚠ Local linear specs (not shown) have F≈0–5 due to 2020 heaping; '
        'they are excluded here as uninformative.</p>',
        unsafe_allow_html=True)

    st.caption(
        f"Sample: {len(_rdd_dm):,} papers (ICLR 2018–2020, OpenAlex-matched, within year-specific bandwidth). "
        f"Pooled bandwidth h = {_rdd_dm['bandwidth'].median():.2f} (median). "
        "McCrary density test: no significant manipulation (β≈0.0, p>0.05)."
    )

st.markdown('<p class="section-header">Section 5 — Leakage Test</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">'
            'The LLM committee was run in 2024–2026 on ICLR 2018–2020 papers — its training data '
            'may already know which papers became famous. If so, "LLM beats human" could reflect '
            'memorized hindsight, not judgment. Six tests below: '
            'decision-recall probe (LAP), a probe-validity placebo check, a fame-recall probe, '
            'a masked re-review ablation, a leakage-excluded re-run of the headline comparison, '
            'and an abstract-completion extraction probe. '
            'Scripts: <code>src/leakage_lap_v1.py</code>, <code>leakage_fame_v1.py</code>, '
            '<code>leakage_controls.py</code>, <code>leakage_masked_rereview.py</code>, '
            '<code>leakage_exclusion_eval.py</code>, <code>leakage_abstract_completion_v1.py</code>.</p>',
            unsafe_allow_html=True)

def _wilson_ci(x, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p, max(0.0, center - half), min(1.0, center + half)


@st.cache_data
def _load_leakage(_v=1):
    out = {}
    for key, path in [
        ("lap", "outputs/leakage_lap_v1.csv"),
        ("fame", "outputs/leakage_fame_v1.csv"),
        ("controls", "outputs/leakage_controls.csv"),
        ("masked", "outputs/leakage_masked_rereview.csv"),
        ("exclusion", "outputs/leakage_exclusion_eval.csv"),
        ("exclusion_s2", "outputs/leakage_exclusion_eval_s2.csv"),
        ("threshold_sweep", "outputs/leakage_threshold_sweep.csv"),
        ("abstract", "outputs/leakage_abstract_completion_v1.csv"),
    ]:
        out[key] = pd.read_csv(path) if os.path.exists(path) else None
    return out

_leak = _load_leakage(_v=2)

if _leak["lap"] is None or _leak["fame"] is None:
    st.info("Run `python src/leakage_lap_v1.py --full` and `leakage_fame_v1.py --full` to enable this section.")
else:
    _lap = _leak["lap"][_leak["lap"]["lap"].notna()]
    _fame = _leak["fame"][_leak["fame"]["fame"].notna()]

    # ── 5a. RECALL SUMMARY ───────────────────────────────────────────────────
    st.markdown("#### 5a. How much does the model recall?")
    _c1, _c2, _c3, _c4 = st.columns(4)
    _lap_commit_n = int((_lap["lap"] >= 0.5).sum())
    _p, _lo, _hi = _wilson_ci(_lap_commit_n, len(_lap))
    _c1.metric("Decision recall (LAP ≥ 0.5)", f"{_p:.1%}",
               help=f"Title+year only — model confidently states accept/reject. "
                    f"95% CI [{_lo:.1%}, {_hi:.1%}], N={len(_lap):,}.")
    _committed = _lap[_lap["lap"] >= 0.5].copy()
    _committed["true_accept"] = _committed["decision"].str.startswith("Accept", na=False)
    _committed["said_accept"] = _committed["ud"] > 0
    _dir_n = int((_committed["true_accept"] == _committed["said_accept"]).sum())
    _p, _lo, _hi = _wilson_ci(_dir_n, len(_committed))
    _c2.metric("Decision-direction accuracy", f"{_p:.1%}",
               help=f"Among committed answers: is the direction (accept vs reject) correct? "
                    f"95% CI [{_lo:.1%}, {_hi:.1%}], N={len(_committed):,}.")
    _fame_commit_n = int((_fame["fame"] >= 0.5).sum())
    _p, _lo, _hi = _wilson_ci(_fame_commit_n, len(_fame))
    _c3.metric("Fame recall (FAME ≥ 0.5)", f"{_p:.1%}",
               help=f"Title+year only — model states whether the paper is widely cited (top 10%). "
                    f"95% CI [{_lo:.1%}, {_hi:.1%}], N={len(_fame):,}.")
    _fcommitted = _fame[_fame["fame"] >= 0.5].copy()
    _fcommitted["true_top"] = _fcommitted["citation_pct_rank"] >= 0.9
    _fcommitted["said_high"] = _fcommitted["fame_ud"] > 0
    _fdir_n = int((_fcommitted["true_top"] == _fcommitted["said_high"]).sum())
    _p, _lo, _hi = _wilson_ci(_fdir_n, len(_fcommitted))
    _c4.metric("Fame-direction accuracy", f"{_p:.1%}",
               help=f"Among committed answers: is high/low cited correctly identified? "
                    f"95% CI [{_lo:.1%}, {_hi:.1%}], N={len(_fcommitted):,}.")
    st.markdown('<p class="explainer">'
                'Decision-direction accuracy is near chance — the model doesn\'t reliably remember '
                '<i>who</i> accepted a paper. Fame-direction accuracy is well above chance — it does '
                'reliably remember <i>which papers became prominent</i>. That is the sharper leakage '
                'channel for a citation-based ground truth.</p>', unsafe_allow_html=True)
    st.caption(
        f"N: LAP probed on {len(_lap):,} papers ({_lap['year'].min()}–{_lap['year'].max()}), "
        f"{(_lap['decision'].str.startswith('Accept', na=False)).mean():.0%} accepts; "
        f"FAME probed on {len(_fame):,} papers, "
        f"{len(_committed):,} committed on LAP / {len(_fcommitted):,} committed on FAME. "
        "Probe accuracy columns use OpenAlex ground truth fixed at probe time (unaffected by the "
        "sidebar citation-source toggle). OpenAlex undercounts (median 2.9× vs S2, worse for "
        "accepted papers), so fame-direction accuracy is a lower bound."
    )

    if _leak["controls"] is not None:
        _ctrl = _leak["controls"]
        st.markdown("##### Probe validity (placebo controls)")
        st.markdown('<p class="explainer">'
                    'N is power/precision-justified, not arbitrary (see '
                    '<code>src/leakage_power_analysis.py</code>, '
                    '<code>outputs/leakage_power_analysis.md</code>): fabricated-title N sized so the '
                    '95% CI on the false-positive rate clears the real commit rate with margin; '
                    'wrong-year N sized via TOST equivalence testing (a non-significant pilot result at '
                    'N=30 is not itself evidence of "no difference").</p>', unsafe_allow_html=True)
        _fake = _ctrl[(_ctrl["probe"] == "fabricated") & (_ctrl["answer"] != "ERROR")]
        if len(_fake):
            _fx = int((_fake["lap"] >= 0.5).sum())
            _fp, _flo, _fhi = _wilson_ci(_fx, len(_fake))
            _rp, _rlo, _rhi = _wilson_ci(_lap_commit_n, len(_lap))
            st.caption(f"Fabricated titles (N={len(_fake)}): confident answer "
                       f"{_fp:.1%} of the time (95% CI [{_flo:.1%}, {_fhi:.1%}]), vs. {_rp:.1%} "
                       f"(95% CI [{_rlo:.1%}, {_rhi:.1%}]) on real papers (N={len(_lap):,}). "
                       "CIs don't overlap — the probe measures real recall, not acquiescence.")

        _wy_rows = []
        for _ptype in sorted(p for p in _ctrl["probe"].unique() if p.startswith("wrong_year")):
            _wy = _ctrl[(_ctrl["probe"] == _ptype) & (_ctrl["answer"] != "ERROR")]
            _cmp = _wy.merge(_lap[["paper_id", "lap"]], left_on="probe_id", right_on="paper_id",
                             suffixes=("_wy", "_correct"))
            if not len(_cmp):
                continue
            _diff = _cmp["lap_correct"] - _cmp["lap_wy"]
            _se = _diff.std() / np.sqrt(len(_diff))
            _offset_label = "+1yr" if _ptype == "wrong_year" else _ptype.replace("wrong_year", "")
            _wy_rows.append({
                "offset": _offset_label, "N": len(_diff),
                "mean diff (correct − wrong-year)": f"{_diff.mean():+.4f}",
                "95% CI": f"[{_diff.mean() - 1.96 * _se:+.4f}, {_diff.mean() + 1.96 * _se:+.4f}]",
            })
        if _wy_rows:
            st.dataframe(pd.DataFrame(_wy_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Equivalence check: does asking with the wrong year change recall? Both offsets' "
                "95% CIs sit inside a ±0.05 equivalence band around zero — the probe tracks memory "
                "of the paper, not of the year we asked about."
            )

    st.markdown("---")

    # ── 5b. WHAT PREDICTS RECALL ─────────────────────────────────────────────
    st.markdown("#### 5b. What predicts recall — citations, not reviewer opinion")
    st.markdown('<p class="explainer">'
                'Regressing recall on log(1+citations) and human mean_rating jointly: citations stay '
                'significant, mean_rating drops out. Recall tracks citation-linked fame specifically — '
                'not "the model likes what reviewers liked."</p>', unsafe_allow_html=True)

    _rc = eval_table.merge(_lap[["paper_id", "lap"]], on="paper_id") \
                    .merge(_fame[["paper_id", "fame"]], on="paper_id", how="left")
    _rc["log_cites"] = np.log1p(_rc["openalex_citations"])
    _rc_d = _rc.dropna(subset=["log_cites", "mean_rating", "lap", "fame"])

    def _multi_ols(y):
        n_ = len(_rc_d)
        X = np.column_stack([np.ones(n_), _rc_d["log_cites"], _rc_d["mean_rating"]])
        yv = _rc_d[y].values
        beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        sigma2 = np.dot(resid, resid) / (n_ - 3)
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sigma2)
        from scipy.stats import t as _tdist
        p = [2 * (1 - _tdist.cdf(abs(b / s), df=n_ - 3)) for b, s in zip(beta, se)]
        return beta, p

    _col5b1, _col5b2 = st.columns(2)
    for _col, _y, _label in [(_col5b1, "lap", "LAP"), (_col5b2, "fame", "FAME")]:
        with _col:
            _beta, _p = _multi_ols(_y)
            st.dataframe(pd.DataFrame({
                "term": ["log(1+citations)", "human mean_rating"],
                "β": [f"{_beta[1]:+.4f}", f"{_beta[2]:+.4f}"],
                "p": [f"{_p[1]:.2g}", f"{_p[2]:.2g}"],
            }), use_container_width=True, hide_index=True)
            st.caption(f"{_label} ~ log_cites + mean_rating  (N={len(_rc_d):,})")

    # ── 5b (visual). Binscatter: mean confidence per log-citation decile ─────
    st.markdown("###### Continuous view: mean recall confidence by citation decile")
    _rc_d = _rc_d.copy()
    _rc_d["cite_bin"] = pd.qcut(_rc_d["log_cites"], 10, labels=False, duplicates="drop")
    fig_bs = go.Figure()
    for _y, _label, _color in [("lap", "LAP (decision recall)", COLORS["LLM Committee (Gemma)"]),
                               ("fame", "FAME (fame recall)", COLORS["LLM Decision Head"])]:
        _bs = _rc_d.groupby("cite_bin").agg(
            x=("log_cites", "mean"), y=(_y, "mean"), n=(_y, "size"), se=(_y, "sem")
        ).reset_index()
        fig_bs.add_trace(go.Scatter(
            x=_bs["x"], y=_bs["y"], mode="lines+markers", name=_label,
            marker=dict(color=_color, size=8),
            line=dict(color=_color),
            error_y=dict(type="data", array=1.96 * _bs["se"], visible=True),
        ))
    fig_bs.update_layout(
        height=320, margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
        xaxis=dict(title="log(1+citations), decile-mean", showgrid=True, gridcolor=BORDER,
                  tickfont=dict(size=10, color=SUBTEXT)),
        yaxis=dict(title="Mean confidence (0–1)", showgrid=True, gridcolor=BORDER,
                  tickfont=dict(size=10, color=SUBTEXT)),
    )
    st.plotly_chart(fig_bs, use_container_width=True)
    st.caption(
        f"Binscatter: papers grouped into 10 equal-N bins by log(1+citations), each point is the "
        f"bin mean confidence ± 95% CI (N={len(_rc_d):,} per probe, ~{len(_rc_d) // 10:,} papers/bin). "
        "A rising line is the continuous version of the OLS β above — visualizes that recall "
        "confidence climbs with citation rank rather than relying on the linear coefficient alone."
    )

    _top = eval_table[eval_table["citation_pct_rank"] >= 0.9].merge(_lap[["paper_id", "lap"]], on="paper_id").merge(_fame[["paper_id", "fame"]], on="paper_id", how="left")
    _bot = eval_table[eval_table["citation_pct_rank"] <= 0.1].merge(_lap[["paper_id", "lap"]], on="paper_id").merge(_fame[["paper_id", "fame"]], on="paper_id", how="left")
    _decile_tbl = pd.DataFrame({
        "citation decile": ["Top 10% by citations", "Bottom 10% by citations"],
        "LAP ≥ 0.5 rate": [f"{(_top['lap'] >= 0.5).mean():.1%}", f"{(_bot['lap'] >= 0.5).mean():.1%}"],
        "FAME ≥ 0.5 rate": [f"{(_top['fame'] >= 0.5).mean():.1%}", f"{(_bot['fame'] >= 0.5).mean():.1%}"],
        "N": [len(_top), len(_bot)],
    })
    st.dataframe(_decile_tbl, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 5c. MASKED RE-REVIEW ──────────────────────────────────────────────────
    if _leak["masked"] is not None and len(_leak["masked"]):
        st.markdown("#### 5c. Masked re-review — does the score survive identity ablation?")
        st.markdown('<p class="explainer">'
                    'Same committee rubric scored twice per paper: normal (title+abstract) vs. masked '
                    '(no title, abstract paraphrased with proper nouns genericized — identity ablated, '
                    'content preserved). If memorized papers lose more score under masking, part of '
                    'their rating rode on recognizing the paper, not judging it.</p>',
                    unsafe_allow_html=True)
        _mk = _leak["masked"].copy()
        _mk["hi_lap"] = _mk["lap"] >= 0.5
        _mk["delta"] = _mk["score_original"] - _mk["score_masked"]
        _mk_summary = _mk.groupby(_mk["hi_lap"].map({True: "High-LAP (memorized)", False: "Low-LAP (not memorized)"})) \
                         .agg(N=("delta", "size"), mean_delta=("delta", "mean")).reset_index() \
                         .rename(columns={"hi_lap": "Group"})
        fig_mask = go.Figure(go.Bar(
            x=_mk_summary["Group"], y=_mk_summary["mean_delta"],
            marker_color=[COLORS["LLM Committee (Gemma)"], RANDOM_COLOR],
            text=[f"{v:+.2f}" for v in _mk_summary["mean_delta"]], textposition="outside",
        ))
        fig_mask.update_layout(
            height=260, margin=dict(l=0, r=0, t=4, b=4),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Mean score drop under masking (orig − masked)",
                      showgrid=True, gridcolor=BORDER, tickfont=dict(size=10, color=SUBTEXT)),
            xaxis=dict(tickfont=dict(size=11, color=TEXT)),
        )
        st.plotly_chart(fig_mask, use_container_width=True)
        st.dataframe(_mk_summary.rename(columns={"mean_delta": "mean score drop"}),
                    use_container_width=True, hide_index=True)
        st.caption(f"N={len(_mk)} total ({_mk_summary['N'].to_dict()}). Score drop is larger for "
                   "memorized papers — consistent with identity recall inflating the original score.")
        st.markdown("---")

    # ── 5d. LEAKAGE-EXCLUDED HEADLINE ────────────────────────────────────────
    _use_s2_excl = citation_source == "Semantic Scholar" and _leak["exclusion_s2"] is not None
    if _use_s2_excl:
        _leak = dict(_leak, exclusion=_leak["exclusion_s2"])
    if _leak["exclusion"] is not None:
        st.markdown("#### 5d. Headline comparison, with memorized papers excluded")
        st.markdown('<p class="explainer">'
                    'All regimes re-run twice: full pool, and with every paper the model recalls '
                    '(LAP or FAME ≥ 0.5) removed and N rescaled. This is the number that should be '
                    'quoted as the headline — the full-pool number overstates the LLM regimes\' edge.</p>',
                    unsafe_allow_html=True)
        _exc = _leak["exclusion"]
        _piv = _exc.pivot_table(index="regime", columns="pool", values="lift", aggfunc="mean").reset_index()
        if {"full", "leakage_excluded"} <= set(_piv.columns):
            _piv["delta"] = _piv["leakage_excluded"] - _piv["full"]
            _order = ["Human (AC decisions)", "Human (score top-N)", "Human (disagreement-adjusted, λ=+1)",
                      "LLM Decision Head", "LLM Committee (Gemma)"]
            _piv["regime"] = pd.Categorical(_piv["regime"], categories=[r for r in _order if r in _piv["regime"].values], ordered=True)
            _piv = _piv.sort_values("regime")

            _boot_path = ("outputs/leakage_exclusion_bootstrap_s2.csv" if _use_s2_excl
                          else "outputs/leakage_exclusion_bootstrap_openalex.csv")
            _boot = pd.read_csv(_boot_path) if os.path.exists(_boot_path) else None
            if _boot is not None:
                # bootstrap points as bar heights so bars and CIs share one convention
                _bl = _boot[_boot["stat"] == "lift"].set_index(["regime", "pool"])
                for _pool_col in ("full", "leakage_excluded"):
                    _piv[_pool_col] = [_bl.loc[(r, _pool_col), "point"] if (r, _pool_col) in _bl.index
                                       else v for r, v in zip(_piv["regime"], _piv[_pool_col])]
                _err = {p: dict(
                    array=[_bl.loc[(r, p), "hi"] - _bl.loc[(r, p), "point"] if (r, p) in _bl.index else 0
                           for r in _piv["regime"]],
                    arrayminus=[_bl.loc[(r, p), "point"] - _bl.loc[(r, p), "lo"] if (r, p) in _bl.index else 0
                                for r in _piv["regime"]],
                    type="data", visible=True, color=SUBTEXT, thickness=1.2, width=4,
                ) for p in ("full", "leakage_excluded")}
                _piv["delta"] = _piv["leakage_excluded"] - _piv["full"]
            else:
                _err = {"full": None, "leakage_excluded": None}
            fig_exc = go.Figure()
            fig_exc.add_trace(go.Bar(name="Full pool", x=_piv["regime"], y=_piv["full"],
                                     marker_color=RANDOM_COLOR, error_y=_err["full"]))
            fig_exc.add_trace(go.Bar(name="Leakage-excluded", x=_piv["regime"], y=_piv["leakage_excluded"],
                                     marker_color=[COLORS.get(r, IDEAL_COLOR) for r in _piv["regime"]],
                                     error_y=_err["leakage_excluded"]))
            fig_exc.update_layout(
                barmode="group", height=340, margin=dict(l=0, r=0, t=4, b=4),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
                yaxis=dict(title="Mean lift over random", showgrid=True, gridcolor=BORDER,
                          tickfont=dict(size=10, color=SUBTEXT)),
                xaxis=dict(tickfont=dict(size=10, color=TEXT)),
            )
            st.plotly_chart(fig_exc, use_container_width=True)

            _disp = _piv[["regime", "full", "leakage_excluded", "delta"]].copy()
            _disp.columns = ["Regime", "Full-pool lift", "Leakage-excluded lift", "Δ"]
            for c in ["Full-pool lift", "Leakage-excluded lift", "Δ"]:
                _disp[c] = _disp[c].map(lambda v: f"{v:+.3f}" if c == "Δ" else f"{v:.3f}")
            st.dataframe(_disp, use_container_width=True, hide_index=True)

            if _boot is not None:
                st.markdown("##### Bootstrap 95% CIs — is the LLM–human gap real?")
                _gaps = _boot[_boot["stat"] == "gap"].copy()
                _gaps["estimate [95% CI]"] = _gaps.apply(
                    lambda r: f"{r['point']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]", axis=1)
                _gaps["p (bootstrap)"] = _gaps["p_boot"].map(
                    lambda p: "<0.001" if p == 0 else f"{p:.3f}")
                _gt = _gaps.pivot_table(index="regime", columns="pool",
                                        values="estimate [95% CI]", aggfunc="first")
                _gp = _gaps.pivot_table(index="regime", columns="pool",
                                        values="p (bootstrap)", aggfunc="first")
                _gap_disp = pd.DataFrame({
                    "LLM − human gap": _gt.index,
                    "Full pool": _gt["full"].values,
                    "p": _gp["full"].values,
                    "Leakage-excluded": _gt["leakage_excluded"].values,
                    "p ": _gp["leakage_excluded"].values,
                })
                st.dataframe(_gap_disp, use_container_width=True, hide_index=True)
                st.caption(
                    "Paired percentile bootstrap over papers (B=2,000), conditional on realized "
                    "selections; same replicate draw for both regimes in each gap, so shared noise "
                    "cancels. When bootstrap results exist, the bars and error bars above use the "
                    "bootstrap point estimates and 95% CIs (analytic per-replicate random "
                    "baselines), so chart, table, and CIs share one convention. "
                    "Script: src/leakage_exclusion_bootstrap.py."
                )

            _pool_ids = set(eval_table["paper_id"])
            _probed_ids = (set(_lap["paper_id"]) | set(_fame["paper_id"])) & _pool_ids
            _excluded_ids = (set(_lap.loc[_lap["lap"] >= 0.5, "paper_id"]) |
                             set(_fame.loc[_fame["fame"] >= 0.5, "paper_id"])) & _pool_ids
            st.caption(
                f"Δ < 0 means the regime's edge shrinks once memorized papers are removed — that gap "
                f"is the measured leakage tax. Probe coverage: {len(_probed_ids):,}/{len(_pool_ids):,} "
                f"papers ({len(_probed_ids) / len(_pool_ids):.1%}); {len(_excluded_ids):,} excluded as "
                f"memorized (LAP or FAME ≥ 0.5). "
                + ("Semantic Scholar ground truth (follows the sidebar toggle; "
                   "src/leakage_exclusion_eval.py --citation-source s2). Under S2, exclusion "
                   "helps every regime — famous excluded papers carry even more of the citation "
                   "mass — but helps humans far more: the LLM Committee's full-pool edge over "
                   "Human AC nearly vanishes on the leakage-excluded pool."
                   if _use_s2_excl else
                   "OpenAlex ground truth. LLM regimes shrink the most; human regimes are ~flat. "
                   "Toggle the sidebar citation source to see the S2 version.")
            )

        if _leak["threshold_sweep"] is not None and len(_leak["threshold_sweep"]):
            st.markdown("##### Is 0.5 doing special work? Threshold-sensitivity sweep")
            st.markdown('<p class="explainer">'
                        'The 0.5 cutoff is a necessary discretization — a paper has to be in or out of '
                        'the re-run pool — but the specific value is arbitrary. Re-running the exclusion '
                        'at every cutoff from 0.1 to 0.9 (no new API calls — pure recompute over already-'
                        'collected LAP/FAME scores) checks whether the result is sensitive to that choice.</p>',
                        unsafe_allow_html=True)
            _sw = _leak["threshold_sweep"]
            _sw_order = ["Human (AC decisions)", "Human (score top-N)", "Human (disagreement-adjusted, λ=+1)",
                        "LLM Decision Head", "LLM Committee (Gemma)"]
            fig_sw = go.Figure()
            for _regime in [r for r in _sw_order if r in _sw["regime"].values]:
                _rd = _sw[_sw["regime"] == _regime].sort_values("threshold")
                if not len(_rd):
                    continue
                fig_sw.add_trace(go.Scatter(
                    x=_rd["threshold"], y=_rd["delta"], mode="lines+markers", name=_regime,
                    marker=dict(color=COLORS.get(_regime, IDEAL_COLOR), size=6),
                    line=dict(color=COLORS.get(_regime, IDEAL_COLOR)),
                ))
            fig_sw.add_hline(y=0, line_dash="dot", line_color=SUBTEXT)
            fig_sw.update_layout(
                height=320, margin=dict(l=0, r=0, t=4, b=4),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                xaxis=dict(title="Exclusion threshold (LAP or FAME ≥ x)", showgrid=True,
                          gridcolor=BORDER, tickfont=dict(size=10, color=SUBTEXT)),
                yaxis=dict(title="Δ lift (leakage-excluded − full)", showgrid=True,
                          gridcolor=BORDER, tickfont=dict(size=10, color=SUBTEXT)),
            )
            st.plotly_chart(fig_sw, use_container_width=True)
            _n_range = _sw["n_excluded"].agg(["min", "max"])
            st.caption(
                f"Δ is essentially flat across thresholds 0.1–0.9 for every regime (excluded-pool size "
                f"ranges only {_n_range['min']:,}–{_n_range['max']:,} papers over that range — LAP/FAME "
                "scores cluster near 0 or 1, so most of the range doesn't relabel any papers). The 0.5 "
                "cutoff isn't cherry-picked — the conclusion would be the same at any threshold in this band. "
                "Sweep is precomputed with OpenAlex ground truth (not affected by the citation-source toggle)."
            )

    # ── 5e. ABSTRACT-COMPLETION EXTRACTION PROBE ─────────────────────────────
    if _leak["abstract"] is not None and len(_leak["abstract"]):
        st.markdown("---")
        st.markdown("#### 5e. Abstract completion — is the paper's *text* in the weights?")
        st.markdown('<p class="explainer">'
                    'Given only title, year, and the first abstract sentence, the model writes the '
                    'rest (greedy decode). The continuation is scored against the true abstract vs '
                    '5 same-field×year decoy abstracts: ROUGE-L margin (target − best decoy) is the '
                    'soft signal, verbatim 8-gram hits are the hard signal. "Extractable" = beats '
                    'all decoys AND ≥1 verbatim 8-gram — the Carlini-style training-data-extraction '
                    'criterion. Stratified sample of ~300 papers by citation decile.</p>',
                    unsafe_allow_html=True)
        _ab = _leak["abstract"]
        _ab = _ab[_ab["rougeL_target"].notna()].copy()

        _e1, _e2, _e3, _e4 = st.columns(4)
        _ex_n = int(_ab["extractable"].sum())
        _p, _lo, _hi = _wilson_ci(_ex_n, len(_ab))
        _e1.metric("Extractable papers", f"{_p:.1%}",
                   help=f"{_ex_n}/{len(_ab)} papers. 95% CI [{_lo:.1%}, {_hi:.1%}].")
        from scipy import stats as _sps
        _sp_m = _sps.spearmanr(_ab["citation_pct_rank"], _ab["rougeL_margin"])
        _e2.metric("Gradient: ROUGE-L margin", f"ρ=+{_sp_m.statistic:.3f}",
                   help=f"Spearman vs citation rank, p={_sp_m.pvalue:.3g}. Positive = famous "
                        "papers' text is preferentially memorized.")
        _sp_8 = _sps.spearmanr(_ab["citation_pct_rank"], _ab["eight_target"])
        _e3.metric("Gradient: verbatim 8-grams", f"ρ=+{_sp_8.statistic:.3f}",
                   help=f"Spearman vs citation rank, p={_sp_8.pvalue:.3g}.")
        _top2 = _ab[_ab["decile"] >= 8]
        _e4.metric("Extractable in top 2 deciles", f"{_top2['extractable'].mean():.1%}",
                   help=f"vs {_ab[_ab['decile'] < 8]['extractable'].mean():.1%} in deciles 0–7 "
                        f"(N={len(_top2)} / {len(_ab) - len(_top2)}).")

        _by_dec = _ab.groupby("decile").agg(
            n=("extractable", "size"), pct_extractable=("extractable", "mean"),
            margin=("rougeL_margin", "mean")).reset_index()
        fig_ab = go.Figure()
        fig_ab.add_trace(go.Bar(
            x=_by_dec["decile"], y=_by_dec["pct_extractable"] * 100,
            marker_color=[COLORS["LLM Committee (Gemma)"] if d >= 8 else RANDOM_COLOR
                          for d in _by_dec["decile"]],
            text=[f"{v:.0%}" if v else "" for v in _by_dec["pct_extractable"]],
            textposition="outside",
        ))
        fig_ab.update_layout(
            height=280, margin=dict(l=0, r=0, t=4, b=4),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
            xaxis=dict(title="Citation decile (0 = least cited)", dtick=1,
                       tickfont=dict(size=10, color=SUBTEXT)),
            yaxis=dict(title="% extractable", showgrid=True, gridcolor=BORDER,
                       tickfont=dict(size=10, color=SUBTEXT)),
        )
        st.plotly_chart(fig_ab, use_container_width=True)
        st.caption(
            "Every extractable paper sits in the top two citation deciles — verbatim-level "
            "confirmation of the fame-recall finding. ROUGE-L margin correlates with FAME "
            "recall but not LAP decision recall: what's in the weights is the paper and its "
            "prominence, not its accept/reject outcome. One-sided test: the model is "
            "instruction-tuned, which suppresses regurgitation — a null does NOT prove the "
            "text is absent from the weights."
        )
        _exhibit = _ab[_ab["extractable"] == 1].nlargest(5, "eight_target").merge(
            eval_table[["paper_id", "title", "openalex_citations"]], on="paper_id", how="left")
        if len(_exhibit):
            _exhibit_disp = _exhibit[["title", "openalex_citations", "rougeL_margin", "eight_target"]].copy()
            _exhibit_disp.columns = ["Title", "Citations", "ROUGE-L margin", "8-gram hit rate"]
            _exhibit_disp["Citations"] = _exhibit_disp["Citations"].map(
                lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
            _exhibit_disp["ROUGE-L margin"] = _exhibit_disp["ROUGE-L margin"].map("{:+.3f}".format)
            _exhibit_disp["8-gram hit rate"] = _exhibit_disp["8-gram hit rate"].map("{:.1%}".format)
            st.dataframe(_exhibit_disp, use_container_width=True, hide_index=True)
            st.caption("Most-extractable papers. Generated/reference texts archived in "
                       "outputs/leakage_abstract_completion_texts.jsonl.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Human reviews are from ICLR 2018–2020 via OpenReview. "
           "LLM reviews generated by a Gemma-4-31B committee pipeline — see sidebar note for details.")
