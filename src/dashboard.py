"""
Reviewer regime comparison dashboard.
Run: streamlit run src/dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from metrics import METRIC_LABELS, compute_metrics
from baselines import random_baseline, ideal_baseline
from regimes.human_actual import HumanActual
from regimes.human_score import HumanScore
from regimes.human_disagree import HumanDisagree

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICLR Reviewer Regimes",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
REGIME_COLORS = {
    "Human (AC decisions)":          "#2563EB",  # blue
    "Human (score top-N)":           "#D97706",  # amber
    "Human (penalize disagreement)": "#0D9488",  # teal
    "Human (reward disagreement)":   "#9333EA",  # purple
}
RANDOM_COLOR  = "#94A3B8"
IDEAL_COLOR   = "#1E293B"
BG            = "#F8FAFC"
CARD_BG       = "#FFFFFF"
BORDER        = "#E2E8F0"
TEXT          = "#0F172A"
SUBTEXT       = "#64748B"

METRIC_SHORT = {
    "median_citations":   "Median cites",
    "mean_log_citations": "Mean log(1+c)",
    "top1_count":         "Top 1% count",
    "top5_count":         "Top 5% count",
    "top10_count":        "Top 10% count",
    "recall_at_1":        "Recall @1%",
    "recall_at_5":        "Recall @5%",
    "recall_at_10":       "Recall @10%",
}

st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: {BG}; }}
  [data-testid="stSidebar"] {{ background: {CARD_BG}; border-right: 1px solid {BORDER}; }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}
  h1 {{ color: {TEXT}; font-weight: 700; letter-spacing: -0.5px; }}
  h2, h3 {{ color: {TEXT}; font-weight: 600; }}
  .metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
  }}
  .winner-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: {SUBTEXT};
    margin-bottom: 2px;
  }}
  .winner-name {{ font-size: 14px; font-weight: 700; color: {TEXT}; }}
  .section-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: {SUBTEXT};
    margin-bottom: 12px;
    margin-top: 8px;
  }}
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    return pd.read_csv("outputs/eval_results.csv")

@st.cache_data
def load_eval_table():
    return pd.read_csv("outputs/eval_table.csv")

try:
    df_static = load_results()
    eval_table = load_eval_table()
except FileNotFoundError:
    st.error("Run `python src/run_eval.py` first.")
    st.stop()

@st.cache_data
def prepare_pool(year, mode, impute_zeros):
    pool = eval_table[eval_table["year"] == year].copy()
    if impute_zeros:
        pool["openalex_citations"] = pool["openalex_citations"].fillna(0)
        # recompute pct rank with zeros included
        for (field,), grp in pool.groupby(["field"]):
            mask = pool["field"].eq(field) & pool["openalex_citations"].notna()
            pool.loc[mask, "citation_pct_rank"] = pool.loc[mask, "openalex_citations"].rank(pct=True)
    if mode == "normalized":
        pool = pool[pool["citation_pct_rank"].notna()].copy()
    return pool

@st.cache_data
def get_baselines(year, mode, impute_zeros):
    pool = prepare_pool(year, mode, impute_zeros)
    n_accepts = eval_table[
        eval_table["year"].eq(year) & eval_table["decision"].str.startswith("Accept", na=False)
    ].shape[0]
    n = n_accepts if mode == "raw" else int(round(n_accepts * len(pool) / eval_table[eval_table["year"] == year].shape[0]))
    return random_baseline(pool, n, mode), ideal_baseline(pool, n, mode), n

def compute_live(regime, year, mode, impute_zeros):
    pool = prepare_pool(year, mode, impute_zeros)
    rand, ideal_vals, n = get_baselines(year, mode, impute_zeros)
    selected = regime.select(pool, n)
    metrics = compute_metrics(selected, pool, mode)
    rows = []
    for metric, value in metrics.items():
        rv, iv = rand.get(metric, np.nan), ideal_vals.get(metric, np.nan)
        rows.append({
            "regime": regime.name, "year": year, "metric": metric, "mode": mode,
            "value": value, "random_value": rv, "ideal_value": iv,
            "lift": (value - rv) / abs(rv) if rv and rv != 0 else np.nan,
            "drawdown": (iv - value) / abs(iv) if iv and iv != 0 else np.nan,
        })
    return rows

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Controls")

mode = "raw"  # field×year normalization pending full field tagging

year_opts = ["All years"] + sorted(df_static["year"].dropna().unique().astype(int).tolist())
selected_year = st.sidebar.selectbox("Year", year_opts)

st.sidebar.markdown("---")
lam = st.sidebar.slider(
    "λ (disagreement weight)", min_value=-3.0, max_value=3.0, value=1.0, step=0.25,
    help="score = mean_rating + λ × rating_std  |  λ > 0: boost contested papers  |  λ < 0: prefer consensus  |  λ = 0: pure score",
)

impute_zeros = st.sidebar.checkbox(
    "Impute 0 citations for unmatched papers", value=False,
    help="~37% of papers have no OpenAlex citation match. Off: exclude them from metrics. On: count them as 0 — penalizes regimes that accept uncited/unknown papers.",
)

st.sidebar.markdown("---")
show_ideal    = st.sidebar.checkbox("Show ideal ceiling", value=True)
show_random   = st.sidebar.checkbox("Show random baseline", value=True)
show_drawdown = st.sidebar.checkbox("Show drawdown from ideal", value=False)

# ── Build dff ─────────────────────────────────────────────────────────────────
years = sorted(eval_table["year"].unique().astype(int).tolist()) if selected_year == "All years" else [int(selected_year)]

all_regimes_live = [
    HumanActual(), HumanScore(), HumanDisagree(lam),
]

if impute_zeros:
    # recompute everything live when imputation changes the ground truth
    live_rows = []
    for year in years:
        for regime in all_regimes_live:
            try:
                live_rows.extend(compute_live(regime, year, mode, impute_zeros))
            except Exception:
                pass
    dff = pd.DataFrame(live_rows)
    if not dff.empty and selected_year == "All years":
        dff = dff.groupby(["regime","metric","mode"])[["value","random_value","ideal_value","lift","drawdown"]].mean().reset_index()
else:
    # use precomputed CSV for fixed regimes, live for disagreement
    static_names = ["Human (AC decisions)", "Human (score top-N)"]
    dff_static = df_static[df_static["mode"].eq(mode) & df_static["regime"].isin(static_names)].copy()
    if selected_year != "All years":
        dff_static = dff_static[dff_static["year"] == int(selected_year)]
    else:
        dff_static = dff_static.groupby(["regime","metric","mode"])[["value","random_value","ideal_value","lift","drawdown"]].mean().reset_index()

    live_rows = []
    for year in years:
        for regime in [HumanDisagree(lam)]:
            try:
                live_rows.extend(compute_live(regime, year, mode, impute_zeros))
            except Exception:
                pass
    dff_live = pd.DataFrame(live_rows)
    if not dff_live.empty and selected_year == "All years":
        dff_live = dff_live.groupby(["regime","metric","mode"])[["value","random_value","ideal_value","lift","drawdown"]].mean().reset_index()
    dff = pd.concat([dff_static, dff_live], ignore_index=True) if not dff_live.empty else dff_static

if dff.empty:
    st.warning("No data — try raw mode or re-run run_eval.py")
    st.stop()

if mode == "normalized":
    try:
        tagged = eval_table["citation_pct_rank"].notna().sum()
        total = len(eval_table)
        if tagged < total:
            pct = tagged / total * 100
            st.warning(f"Field tagging is {pct:.0f}% complete ({tagged:,}/{total:,} papers). Normalized results are partial — re-run `build_eval_table.py` + `run_eval.py` once tagging finishes.")
    except Exception:
        pass


COLOR_LIST = ["#2563EB", "#D97706", "#0D9488", "#9333EA"]
all_regimes = dff["regime"].unique().tolist()
regime_color = {r: COLOR_LIST[i % len(COLOR_LIST)] for i, r in enumerate(all_regimes)}
regimes = all_regimes
metrics = [m for m in METRIC_LABELS if m in dff["metric"].unique()]

# ── Header ────────────────────────────────────────────────────────────────────
year_label = selected_year if selected_year != "All years" else "2018–2020 avg"
st.markdown(f"# Reviewer Regime Comparison")
st.markdown(f"<span style='color:{SUBTEXT};font-size:15px'>ICLR · {year_label} · {'Raw citations' if mode == 'raw' else 'Field × year normalized'}</span>", unsafe_allow_html=True)
st.markdown("")

# ── Normalized performance chart (hero) ───────────────────────────────────────
# Scale: 0% = random, 100% = ideal
st.markdown('<div class="section-label">Overall performance — % of gap closed vs. random baseline (100% = ideal)</div>', unsafe_allow_html=True)

norm_rows = []
for regime in regimes:
    scores = []
    for metric in metrics:
        row = dff[(dff["regime"] == regime) & (dff["metric"] == metric)]
        if row.empty:
            continue
        v, rand, ideal = row["value"].values[0], row["random_value"].values[0], row["ideal_value"].values[0]
        gap = ideal - rand
        pct = (v - rand) / gap * 100 if gap != 0 else 0
        scores.append(pct)
    if scores:
        norm_rows.append({"regime": regime, "score": np.mean(scores)})

norm_df = pd.DataFrame(norm_rows).sort_values("score", ascending=True)

fig_hero = go.Figure()
fig_hero.add_trace(go.Bar(
    x=norm_df["score"],
    y=norm_df["regime"].str.replace("Human (", "").str.rstrip(")"),
    orientation="h",
    marker_color=[regime_color.get(r, "#94A3B8") for r in norm_df["regime"]],
    marker_line_width=0,
    text=[f"{v:.0f}%" for v in norm_df["score"]],
    textposition="outside",
    textfont=dict(size=13, color=TEXT),
    hovertemplate="%{y}<br>%{x:.1f}% of gap closed<extra></extra>",
))
fig_hero.update_layout(
    height=180,
    margin=dict(l=0, r=60, t=8, b=8),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        range=[0, 115],
        showgrid=True, gridcolor=BORDER, gridwidth=1,
        zeroline=True, zerolinecolor=BORDER,
        ticksuffix="%", tickfont=dict(color=SUBTEXT, size=11),
    ),
    yaxis=dict(showgrid=False, tickfont=dict(color=TEXT, size=12)),
    showlegend=False,
)
st.plotly_chart(fig_hero, use_container_width=True)

# ── Raw numbers table (sorted by overall performance) ─────────────────────────
pivot = (
    dff[dff["regime"].isin(regimes)]
    .pivot_table(index="regime", columns="metric", values="value")
)
display_cols = [m for m in METRIC_LABELS if m in pivot.columns]

# compute % gap closed per regime to use as sort key
def _pct_gap(regime):
    scores = []
    for metric in display_cols:
        sub = dff[(dff["regime"] == regime) & (dff["metric"] == metric)]
        if sub.empty: continue
        v, rand, ideal = sub["value"].values[0], sub["random_value"].values[0], sub["ideal_value"].values[0]
        gap = ideal - rand
        if gap and not np.isnan(gap):
            scores.append((v - rand) / gap * 100)
    return np.mean(scores) if scores else 0

regime_order = sorted(regimes, key=_pct_gap, reverse=True)
pivot_display = pivot.loc[[r for r in regime_order if r in pivot.index], display_cols].round(3).rename(columns=METRIC_SHORT)
pivot_display.index = pivot_display.index.str.replace("Human (", "").str.rstrip(")")
st.dataframe(pivot_display, use_container_width=True)

st.markdown("---")

# ── Per-metric drill-down ─────────────────────────────────────────────────────
selected_metric = st.selectbox(
    "Metric drill-down",
    metrics,
    format_func=lambda m: METRIC_LABELS[m],
    index=0,
)

def metric_gap_chart(metric):
    sub_rows = []
    for regime in regimes:
        sub = dff[(dff["regime"] == regime) & (dff["metric"] == metric)]
        if sub.empty:
            continue
        v    = sub["value"].values[0]
        rand = sub["random_value"].values[0]
        ideal= sub["ideal_value"].values[0]
        gap  = ideal - rand
        pct  = (v - rand) / gap * 100 if gap and not np.isnan(gap) else 0
        sub_rows.append({"regime": regime, "pct": pct, "value": v, "rand": rand, "ideal": ideal})

    sub_df = pd.DataFrame(sub_rows).sort_values("pct", ascending=True)
    short  = sub_df["regime"].str.replace("Human (", "").str.rstrip(")")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub_df["pct"],
        y=short,
        orientation="h",
        marker_color=[regime_color.get(r, "#94A3B8") for r in sub_df["regime"]],
        marker_line_width=0,
        text=[f"{p:.0f}%  ({v:.3f})" for p, v in zip(sub_df["pct"], sub_df["value"])],
        textposition="outside",
        textfont=dict(size=12, color=TEXT),
        hovertemplate="%{y}<br>%{x:.1f}% of gap closed<extra></extra>",
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=120, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            range=[0, 120],
            showgrid=True, gridcolor=BORDER, gridwidth=1,
            zeroline=True, zerolinecolor=BORDER,
            ticksuffix="%", tickfont=dict(color=SUBTEXT, size=11),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color=TEXT, size=12)),
    )
    return fig

st.plotly_chart(metric_gap_chart(selected_metric), use_container_width=True)

