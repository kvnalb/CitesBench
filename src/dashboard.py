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
from scipy.stats import gaussian_kde

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

@st.cache_data
def load_eval_table(_v=2):  # bump _v to bust cache on redeploy
    et = pd.read_csv("outputs/eval_table.csv")
    rej_path = "outputs/outlier_reviews.csv"
    if os.path.exists(rej_path):
        rej = pd.read_csv(rej_path)[["title", "rejection_tags"]].drop_duplicates("title")
        et = et.merge(rej, on="title", how="left")
    else:
        et["rejection_tags"] = pd.NA
    return et

try:
    df_static = load_results()
    eval_table = load_eval_table()
except FileNotFoundError:
    st.error("Run `python src/run_eval.py` first.")
    st.stop()

BASELINE_CACHE = "outputs/baselines_cache.csv"

@st.cache_data
def prepare_pool(year, mode, impute_zeros):
    pool = eval_table[eval_table["year"] == year].copy()
    if impute_zeros:
        pool["openalex_citations"] = pool["openalex_citations"].fillna(0)
        for field, grp in pool.groupby("field"):
            mask = pool["field"].eq(field) & pool["openalex_citations"].notna()
            pool.loc[mask, "citation_pct_rank"] = pool.loc[mask, "openalex_citations"].rank(pct=True)
    return pool

@st.cache_data
def get_baselines(year, mode, impute_zeros):
    key = f"{year}_{mode}_{int(impute_zeros)}"
    if os.path.exists(BASELINE_CACHE):
        cached = pd.read_csv(BASELINE_CACHE)
        hit = cached[cached["key"] == key]
        if not hit.empty:
            rand  = dict(zip(hit[hit["which"]=="random"]["metric"], hit[hit["which"]=="random"]["value"]))
            ideal = dict(zip(hit[hit["which"]=="ideal"]["metric"],  hit[hit["which"]=="ideal"]["value"]))
            n     = int(hit["n"].values[0])
            return rand, ideal, n
    pool = prepare_pool(year, mode, impute_zeros)
    n = eval_table[eval_table["year"].eq(year) &
                   eval_table["decision"].str.startswith("Accept", na=False)].shape[0]
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

def compute_live(regime, year, mode, impute_zeros):
    pool = prepare_pool(year, mode, impute_zeros)
    rand, ideal_vals, n = get_baselines(year, mode, impute_zeros)
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Controls")

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

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Median/mean citations computed over OpenAlex-matched papers only "
                   "(accepts ~89%, rejects ~63%). Recall metrics unaffected.")
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
            live_rows.extend(compute_live(regime, year, mode, impute_zeros))
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
color_map = {r: list(COLORS.values())[i % len(COLORS)] for i, r in enumerate(regimes)}
metrics   = [m for m in METRIC_LABELS if m in dff["metric"].unique()]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# CitesBench — Reviewer Regime Comparison")
st.markdown(f"<span style='color:{SUBTEXT};font-size:15px'>ICLR 2018–2020  ·  "
            f"{selected_year}  ·  Raw citations</span>", unsafe_allow_html=True)
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
# SECTION 2 — LEAKAGE-ROBUST CHECK (TOP DECILE EXCLUDED)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 2 — Leakage-Robust Check</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">LLMs trained on recent data may recognise highly-cited '
            'papers by reputation, inflating performance at the top end. This table repeats '
            'the evaluation after removing the top-10% of papers by citations from each '
            'year\'s pool (N adjusted to accepted papers in the remaining pool). '
            'If LLM regimes still outperform human AC here, the signal is harder to '
            'attribute to memorisation of famous papers.</p>', unsafe_allow_html=True)

_td_rows = []
_cutoff_info = {}
for _yr in all_years:
    _pool = prepare_pool(_yr, mode, impute_zeros)
    _known = _pool.dropna(subset=["openalex_citations"])
    if _known.empty:
        continue
    _cut = _known["openalex_citations"].quantile(0.90)
    _cutoff_info[_yr] = int(_cut)
    _fpool = _pool[
        _pool["openalex_citations"].isna() | (_pool["openalex_citations"] <= _cut)
    ].copy()
    _n_f = _fpool[_fpool["decision"].str.startswith("Accept", na=False)].shape[0]
    if _n_f == 0:
        continue
    _rand_f  = random_baseline(_fpool, _n_f, mode)
    _ideal_f = ideal_baseline(_fpool, _n_f, mode)
    for _reg in all_regimes:
        try:
            _sel = _reg.select(_fpool, _n_f)
            _mets = compute_metrics(_sel, _fpool, mode)
            for _met, _val in _mets.items():
                _td_rows.append({
                    "regime": _reg.name, "year": _yr, "metric": _met,
                    "value": _val,
                    "random_value": _rand_f.get(_met, np.nan),
                    "ideal_value":  _ideal_f.get(_met, np.nan),
                })
        except Exception:
            pass

_td_df = pd.DataFrame(_td_rows)
if not _td_df.empty:
    if selected_year == "All years":
        _td_df = _td_df.groupby(["regime", "metric"])[
            ["value", "random_value", "ideal_value"]].mean().reset_index()
    else:
        _td_df = _td_df[_td_df["year"] == int(selected_year)]

    _show_m = [m for m in
               ["median_citations", "mean_log_citations", "recall_at_5", "recall_at_10"]
               if m in _td_df["metric"].unique() and m in dff["metric"].unique()]

    _td_piv = _td_df[_td_df["regime"].isin(regimes)].pivot_table(
        index="regime", columns="metric", values="value")
    _full_piv = dff[dff["regime"].isin(regimes)].pivot_table(
        index="regime", columns="metric", values="value")

    _td_piv   = _td_piv.loc[[r for r in regime_order if r in _td_piv.index],
                              [m for m in _show_m if m in _td_piv.columns]].round(3)
    _full_piv = _full_piv.loc[[r for r in regime_order if r in _full_piv.index],
                                [m for m in _show_m if m in _full_piv.columns]].round(3)

    _td_piv.rename(  columns={m: f"{METRIC_SHORT[m]} †" for m in _show_m if m in METRIC_SHORT}, inplace=True)
    _full_piv.rename(columns={m: METRIC_SHORT[m]         for m in _show_m if m in METRIC_SHORT}, inplace=True)
    _td_piv.index   = _td_piv.index.str.replace("Human (", "").str.rstrip(")")
    _full_piv.index = _full_piv.index.str.replace("Human (", "").str.rstrip(")")

    # Interleave columns: Metric (full), Metric † (excl), ...
    _cols_ord = []
    for m in _show_m:
        if m not in METRIC_SHORT: continue
        lbl = METRIC_SHORT[m]
        if lbl in _full_piv.columns:     _cols_ord.append(lbl)
        if f"{lbl} †" in _td_piv.columns: _cols_ord.append(f"{lbl} †")
    _combined = pd.concat([_full_piv, _td_piv], axis=1)[[c for c in _cols_ord if c in pd.concat([_full_piv, _td_piv], axis=1).columns]]

    _rand_ref = {}
    _ideal_ref = {}
    for m in _show_m:
        if m not in METRIC_SHORT: continue
        lbl = METRIC_SHORT[m]
        _rand_ref[lbl]        = dff[dff["metric"]==m]["random_value"].mean()
        _rand_ref[f"{lbl} †"] = _td_df[_td_df["metric"]==m]["random_value"].mean()
        _ideal_ref[lbl]        = dff[dff["metric"]==m]["ideal_value"].mean()
        _ideal_ref[f"{lbl} †"] = _td_df[_td_df["metric"]==m]["ideal_value"].mean()
    _ref_td = pd.DataFrame([_rand_ref, _ideal_ref],
                           index=["— Random baseline", "— Ideal ceiling"]).round(3)
    _ref_td = _ref_td[[c for c in _cols_ord if c in _ref_td.columns]]

    if _cutoff_info:
        st.caption("† top-decile excluded  ·  cutoffs: " +
                   "  ·  ".join(f"{yr}: ≥{cut:,} cites" for yr, cut in sorted(_cutoff_info.items())))
    st.dataframe(pd.concat([_combined, _ref_td]), use_container_width=True)
else:
    st.info("No citation data available for top-decile exclusion.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REGIME DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 3 — Regime Deep Dive</p>', unsafe_allow_html=True)
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
def compute_quadrants(regime_name, lam, years_tuple, mode, impute_zeros):
    """Returns pool_df with 'quadrant' column and sets of IDs."""
    pools, regime_ids, ideal_ids, ac_ids = [], set(), set(), set()
    for year in years_tuple:
        pool = prepare_pool(year, mode, impute_zeros)
        _, _, n = get_baselines(year, mode, impute_zeros)
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
        dive_regime_name, lam, tuple(dive_years), mode, impute_zeros)

quad_order = ["regime ∩ ideal", "ideal only", "regime only", "neither"]

# ── 3a. CONFUSION MATRICES ────────────────────────────────────────────────────
st.markdown("#### 3a. Confusion Matrices")
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
st.markdown("#### 3b. Flipped Papers vs Human AC — Citation Residuals")
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
st.markdown("#### 3c. Human Reviewer Score vs Citations")
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
st.markdown("#### 3d. Missed Gems and Human Consensus Errors")
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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Human reviews are from ICLR 2018–2020 via OpenReview. "
           "LLM reviews generated by a Gemma-4-31B committee pipeline — see sidebar note for details.")
