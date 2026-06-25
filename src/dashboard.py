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
from regimes.llm_neutral import LLMNeutral
from regimes.llm_ensemble import LLMEnsemble
from regimes.llm_positive import LLMPositive

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CitesBench — Reviewer Regime Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "Human (AC decisions)":                  "#2563EB",
    "Human (score top-N)":                   "#D97706",
    "Human (disagreement-adjusted)":         "#0D9488",
    "LLM1 (neutral)":                        "#DC2626",
    "LLM2 (ensemble)":                       "#059669",
    "LLM3 (positive advocate)":              "#7C3AED",
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
def load_eval_table(): return pd.read_csv("outputs/eval_table.csv")

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

# ── Regime list ───────────────────────────────────────────────────────────────
all_regimes = [HumanActual(), HumanScore(), HumanDisagree(lam),
               LLMNeutral(), LLMEnsemble(), LLMPositive()]
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

def _pct_gap(regime, drawdown=show_drawdown):
    scores = []
    for metric in metrics:
        sub = dff[(dff["regime"] == regime) & (dff["metric"] == metric)]
        if sub.empty: continue
        v, rand, ideal = sub["value"].values[0], sub["random_value"].values[0], sub["ideal_value"].values[0]
        gap = ideal - rand
        if gap and not np.isnan(gap):
            scores.append((ideal - v) / gap * 100 if drawdown else (v - rand) / gap * 100)
    return np.mean(scores) if scores else 0

def bar_chart(scores_df, title_label, drawdown=False):
    """Horizontal bar chart showing % of gap closed (or drawdown)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores_df["score"],
        y=scores_df["label"],
        orientation="h",
        marker_color=[color_map.get(r, "#94A3B8") for r in scores_df["regime"]],
        marker_line_width=0,
        text=[f"{v:.0f}%" for v in scores_df["score"]],
        textposition="outside",
        textfont=dict(size=12, color=TEXT),
        hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=40 * len(scores_df) + 60,
        margin=dict(l=0, r=80, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(range=[0, 115] if not drawdown else [-5, 110],
                   autorange="reversed" if drawdown else True,
                   showgrid=True, gridcolor=BORDER,
                   ticksuffix="%", tickfont=dict(color=SUBTEXT, size=11),
                   zeroline=True, zerolinecolor=BORDER),
        yaxis=dict(showgrid=False, tickfont=dict(color=TEXT, size=12)),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# CitesBench — Reviewer Regime Comparison")
st.markdown(f"<span style='color:{SUBTEXT};font-size:15px'>ICLR 2018–2020  ·  "
            f"{selected_year}  ·  Raw citations</span>", unsafe_allow_html=True)
st.markdown("")

st.markdown('<p class="section-header">Section 1 — Overall Performance</p>', unsafe_allow_html=True)
c_label, c_note = st.columns([2, 3])
mode_label = "drawdown from ideal (lower = better, 0% = ideal)" if show_drawdown \
             else "% of ideal gap closed (higher = better, 100% = ideal)"
c_label.markdown(f'<p class="explainer">Each bar = <b>{mode_label}</b>, '
                 'averaged equally across all 5 metrics (median cites, mean log cites, '
                 'recall @1/5/10%). This is a summary score — see the table below for '
                 'per-metric breakdown.</p>', unsafe_allow_html=True)

regime_order = sorted(regimes, key=lambda r: _pct_gap(r, show_drawdown),
                      reverse=not show_drawdown)
scores_df = pd.DataFrame([{
    "regime": r,
    "label":  r.replace("Human (", "").rstrip(")").replace("LLM", "LLM "),
    "score":  _pct_gap(r, show_drawdown),
} for r in regime_order])

st.plotly_chart(bar_chart(scores_df, "Overall", show_drawdown), use_container_width=True)

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
# SECTION 2 — METRIC BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Section 2 — Metric Breakdown</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">Different metrics capture different aspects of quality. '
            'Median citations rewards consistently good picks; recall @1% rewards finding '
            'the rare breakthrough papers. Select a metric to see per-regime performance '
            'on that dimension. Bars show % of gap closed; raw values shown in parentheses. '
            'The dashed line is random selection; the dotted line is the theoretical ceiling '
            '(top-N papers by citations).</p>', unsafe_allow_html=True)

sel_metric = st.selectbox("Select metric", metrics, format_func=lambda m: METRIC_LABELS[m])

sub_rows = []
for r in regime_order:
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
              LLMNeutral(), LLMEnsemble(), LLMPositive()]:
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

# ── 3b. CITATION KDE BY QUADRANT ─────────────────────────────────────────────
st.markdown("#### 3b. Citation Distribution by Quadrant")
st.markdown('<p class="explainer">'
            'Each curve shows the distribution of citation counts (log scale) for one quadrant. '
            'The key diagnostic: if <b>regime only</b> (green) overlaps with '
            '<b>ideal only</b> (amber) — the regime is making different but equally impactful picks. '
            'If it overlaps with <b>neither</b> (grey) — the regime is selecting low-impact papers '
            'that the citation-based ideal also excludes. Curves are normalized so shape is '
            'visible regardless of group size.</p>', unsafe_allow_html=True)

known = pool_df.dropna(subset=["openalex_citations"])
fig_kde = go.Figure()
for quad in quad_order:
    vals = known[known["quadrant"] == quad]["openalex_citations"].values
    if len(vals) < 5: continue
    log_vals = np.log1p(vals)
    xs = np.linspace(log_vals.min(), log_vals.max(), 200)
    kde = gaussian_kde(log_vals, bw_method=0.4)
    fig_kde.add_trace(go.Scatter(
        x=xs, y=kde(xs), mode="lines", name=f"{quad} (n={len(vals)})",
        line=dict(color=QUAD_COLORS[quad], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({int(QUAD_COLORS[quad][1:3],16)},"
                  f"{int(QUAD_COLORS[quad][3:5],16)},"
                  f"{int(QUAD_COLORS[quad][5:7],16)},0.08)",
    ))

fig_kde.update_layout(
    height=280, margin=dict(l=0, r=0, t=8, b=8),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
    xaxis=dict(title="log(1 + citations)", showgrid=True, gridcolor=BORDER,
               tickfont=dict(size=11, color=SUBTEXT)),
    yaxis=dict(title="density", showgrid=False, tickfont=dict(size=11, color=SUBTEXT)),
)
st.plotly_chart(fig_kde, use_container_width=True)
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

# ── 3d. LLM SCORE vs HUMAN SCORE (LLM regimes only) ─────────────────────────
llm_score_col = {
    "LLM1 (neutral)":          "llm_neutral_rating",
    "LLM2 (ensemble)":         "llm_mean_rating",
    "LLM3 (positive advocate)":"llm_positive_rating",
}
if dive_regime_name in llm_score_col:
    col = llm_score_col[dive_regime_name]
    st.markdown("#### 3d. LLM Score vs Human Reviewer Score")
    st.markdown('<p class="explainer">'
                'X-axis = mean human reviewer rating; Y-axis = LLM score for this regime. '
                'Where the LLM diverges from humans (off-diagonal clusters) tells you where '
                'it is making independent judgments. Papers in the upper-left '
                '(LLM liked it, humans didn\'t) are the regime\'s "champion picks" — '
                'check whether those are in the ideal set (blue) or not (green).</p>',
                unsafe_allow_html=True)

    llm_df = pool_df.dropna(subset=["mean_rating", col]).copy()
    fig_llm = go.Figure()
    for quad in quad_order:
        sub = llm_df[llm_df["quadrant"] == quad]
        fig_llm.add_trace(go.Scatter(
            x=sub["mean_rating"], y=sub[col], mode="markers",
            name=f"{quad} (n={len(sub)})",
            marker=dict(color=QUAD_COLORS[quad], size=5, opacity=0.6,
                        line=dict(width=0)),
            hovertemplate="Human: %{x:.1f}<br>LLM: %{y:.1f}<extra>" + quad + "</extra>",
        ))
    # diagonal agreement line
    mn = max(llm_df["mean_rating"].min(), llm_df[col].min())
    mx = min(llm_df["mean_rating"].max(), llm_df[col].max())
    fig_llm.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
        line=dict(color=SUBTEXT, width=1, dash="dash"), showlegend=False,
        hoverinfo="skip"))
    fig_llm.update_layout(
        height=350, margin=dict(l=0, r=0, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        xaxis=dict(title="Mean human reviewer rating", showgrid=True, gridcolor=BORDER,
                   tickfont=dict(size=11, color=SUBTEXT)),
        yaxis=dict(title=f"LLM score ({dive_regime_name})", showgrid=True, gridcolor=BORDER,
                   tickfont=dict(size=11, color=SUBTEXT)),
    )
    st.plotly_chart(fig_llm, use_container_width=True)
    st.markdown("---")

# ── 3e. MISSED GEMS + UNIQUE FINDS ───────────────────────────────────────────
st.markdown("#### 3e. Missed Gems and Unique Finds")
st.markdown('<p class="explainer">'
            '<b>Missed gems</b> (left): high-impact papers the regime failed to select — '
            'these were in the citation ideal but not picked by this regime. '
            '<b>Unique finds</b> (right): papers this regime selected that are not in the '
            'citation ideal — sorted by citations to show how wrong (or right) the unique '
            'picks were. Hover for paper title.</p>', unsafe_allow_html=True)

merged = pool_df.dropna(subset=["openalex_citations"])

missed = merged[merged["quadrant"] == "ideal only"].nlargest(10, "openalex_citations")[
    ["title","year","openalex_citations","mean_rating"]].round(2)
missed.columns = ["Title","Year","Citations","Avg reviewer score"]
missed["Title"] = missed["Title"].str[:70] + "…"

unique = merged[merged["quadrant"] == "regime only"].nlargest(10, "openalex_citations")[
    ["title","year","openalex_citations","mean_rating"]].round(2)
unique.columns = ["Title","Year","Citations","Avg reviewer score"]
unique["Title"] = unique["Title"].str[:70] + "…"

col_l, col_r = st.columns(2)
col_l.markdown(f"**Missed gems** — top-impact papers regime didn't pick (n={len(merged[merged['quadrant']=='ideal only'])})")
col_l.dataframe(missed, use_container_width=True, hide_index=True)
col_r.markdown(f"**Unique finds** — regime's picks outside the ideal set (n={len(merged[merged['quadrant']=='regime only'])})")
col_r.dataframe(unique, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("**LLM1–3** reviews (neutral, ensemble, positive-advocate) are synthetically "
           "generated reviews from the OpenReview dataset authors, not produced by this project. "
           "Human reviews are from ICLR 2018–2020 via OpenReview.")
