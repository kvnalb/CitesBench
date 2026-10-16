"""
Reviewer regime comparison dashboard.
Run: streamlit run src/app/dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # app_style

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from metrics import METRIC_LABELS, compute_metrics
from baselines import random_baseline, ideal_baseline
from regimes.human_actual import HumanActual
from regimes.human_score import HumanScore
from regimes.human_disagree import HumanDisagree
from regimes.llm_committee import LLMCommittee
from regimes.llm_deepseek import LLMDeepSeek
from app_style import (COLORS, RANDOM_COLOR, IDEAL_COLOR, BORDER, TEXT, SUBTEXT,
                   _abs, EVAL_TABLE, CITATIONS_CSV, apply_css)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CitesBench — Reviewer Regime Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

# ── Design tokens ─────────────────────────────────────────────────────────────
QUAD_COLORS = {
    "regime ∩ ideal":  "#2563EB",
    "ideal only":      "#F59E0B",
    "regime only":     "#10B981",
    "neither":         "#CBD5E1",
}

METRIC_SHORT = {
    "median_citations":   "Median cites",
    "mean_log_citations": "Mean log(1+c)",
    "recall_at_1":        "Recall @1%",
    "recall_at_5":        "Recall @5%",
    "recall_at_10":       "Recall @10%",
}

apply_css()

# ── Data ─────────────────────────────────────────────────────────────────────
# Ground truth is outputs/citations.csv, reached through eval_table.csv, which
# build_eval_table.py joins against it. There is deliberately no second definition
# of the outcome in this file: the old sidebar toggle rebuilt it in-app from
# s2_citations_full.csv (S2 v1, a 0.9 similarity gate, no tier rule), which by the
# end did the opposite of what its label said — the option marked "OpenAlex" was
# already serving canonical S2 counts, and choosing "Semantic Scholar" *overwrote*
# them with the weaker v1 numbers. See build_citations.py for why one table wins.


@st.cache_data
def load_results(): return pd.read_csv(_abs("outputs/eval_results.csv"))


@st.cache_data
def load_eval_table(_v=4):
    et = pd.read_csv(EVAL_TABLE, low_memory=False)
    rej_path = _abs("outputs/outlier_reviews.csv")
    if os.path.exists(rej_path):
        rej = pd.read_csv(rej_path)[["title", "rejection_tags"]].drop_duplicates("title")
        et = et.merge(rej, on="title", how="left")
    else:
        et["rejection_tags"] = pd.NA
    return et


def _et(*_ignored):
    return load_eval_table()


def _provenance(et):
    """(source string, tier mix, coverage) read off the table — never asserted here."""
    src = et["citation_source"].dropna()
    src = src.iloc[0] if len(src) else "unknown"
    tiers = et["citation_tier"].value_counts().to_dict()
    n = int(et["s2_citations"].notna().sum())
    return src, tiers, n


st.sidebar.markdown("## Controls")

try:
    df_static = load_results()
    eval_table = _et()
except FileNotFoundError:
    st.error("Run `python src/analysis/run_eval.py` first.")
    st.stop()

# eval_table is a derived file. If the canonical table has moved underneath it the
# numbers on this page are silently from the previous build, so say so rather than
# rendering them straight.
if os.path.exists(CITATIONS_CSV) and \
        os.path.getmtime(CITATIONS_CSV) > os.path.getmtime(EVAL_TABLE):
    st.sidebar.error(
        "`outputs/citations.csv` is newer than `outputs/eval_table.csv`. The figures "
        "below are from the previous build. Re-run:\n\n"
        "```bash\npython src/build/build_eval_table.py\n```")

_src, _tiers, _cov = _provenance(eval_table)
_accepted = eval_table["decision"].str.startswith("Accept", na=False)
_coverage_acc = eval_table.loc[_accepted, "s2_citations"].notna().mean()
_coverage_rej = eval_table.loc[~_accepted, "s2_citations"].notna().mean()
_coverage_gap = 100 * (_coverage_acc - _coverage_rej)

BASELINE_CACHE = _abs("outputs/baselines_cache.csv")

# The cached random/ideal baselines are only valid for the citation numbers they were
# computed against, and the old keys ("2018_raw_0_0") did not record which those were.
# When eval_table swapped from OpenAlex to S2 the pre-existing rows kept matching, so
# S2 outcomes were being scored against OpenAlex-era baselines and every lift and
# drawdown on this page was wrong. The source is part of the key now, which strands
# the stale rows instead of hitting them.
_BASELINE_SRC = str(_src)

@st.cache_data
def prepare_pool(year, mode, impute_zeros, exclude_top_decile=False):
    et = _et()
    pool = et[et["year"] == year].copy()
    if impute_zeros:
        pool["s2_citations"] = pool["s2_citations"].fillna(0)
        for field, grp in pool.groupby("field"):
            mask = pool["field"].eq(field) & pool["s2_citations"].notna()
            pool.loc[mask, "citation_pct_rank"] = pool.loc[mask, "s2_citations"].rank(pct=True)
    if exclude_top_decile:
        known = pool.dropna(subset=["s2_citations"])
        if not known.empty:
            cut = known["s2_citations"].quantile(0.90)
            pool = pool[pool["s2_citations"].isna() | (pool["s2_citations"] <= cut)].copy()
    return pool

@st.cache_data
def get_baselines(year, mode, impute_zeros, exclude_top_decile=False):
    key = f"{year}_{mode}_{int(impute_zeros)}_{int(exclude_top_decile)}_{_BASELINE_SRC}"
    if os.path.exists(BASELINE_CACHE):
        cached = pd.read_csv(BASELINE_CACHE)
        hit = cached[cached["key"] == key]
        if not hit.empty:
            rand  = dict(zip(hit[hit["which"]=="random"]["metric"], hit[hit["which"]=="random"]["value"]))
            ideal = dict(zip(hit[hit["which"]=="ideal"]["metric"],  hit[hit["which"]=="ideal"]["value"]))
            n     = int(hit["n"].values[0])
            return rand, ideal, n
    pool = prepare_pool(year, mode, impute_zeros, exclude_top_decile)
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

def compute_live(regime, year, mode, impute_zeros, exclude_top_decile=False):
    pool = prepare_pool(year, mode, impute_zeros, exclude_top_decile)
    rand, ideal_vals, n = get_baselines(year, mode, impute_zeros, exclude_top_decile)
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
    help=f"{len(eval_table)-_cov:,} of {len(eval_table):,} papers have no tier A/B match. "
         "Off: exclude from metrics. On: count as 0. Leave off — a paper we could not "
         "find is not a paper with no citations.")
show_drawdown = st.sidebar.checkbox("Show drawdown from ideal", value=False,
    help="Off: % of gap closed (higher = better). On: % left on table vs ideal (lower = better).")
exclude_top_decile = st.sidebar.checkbox("Exclude top-10% by citations (sensitivity check)", value=False,
    help="Removes the top citation decile per year and adjusts N. "
         "LLMs may recognise famous papers from training data — stripping them tests whether "
         "the signal survives after excluding highly cited work. This does not rule out memorisation.")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Ground truth:** `outputs/citations.csv` → `eval_table.csv`.  \n"
    f"Source `{_src}`. Matched {_cov:,}/{len(eval_table):,} ({_cov/len(eval_table):.1%}) — "
    f"tier A {_tiers.get('A', 0):,} (arXiv/DOI id), tier B {_tiers.get('B', 0):,} "
    f"(title ≥0.95 + year window + shared author). "
    + ("Unmatched citations are imputed as zero.  \n" if impute_zeros else
       "Unmatched citations are excluded from citation metrics.  \n") +
    f"Citation coverage is {_coverage_acc:.1%} accepted vs {_coverage_rej:.1%} rejected "
    f"({_coverage_gap:.1f} percentage points). Coverage is measured before optional imputation."
)
st.sidebar.markdown("---")
_heads = eval_table["decision_head_model"].dropna().value_counts()
st.sidebar.caption(
    "**LLM regimes:** committee ratings and decision-head probabilities are separate "
    "selectors. Decision heads in the loaded data: "
    + "; ".join(f"{model}: {count:,}" for model, count in _heads.items()) + "."
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
            live_rows.extend(compute_live(regime, year, mode, impute_zeros, exclude_top_decile))
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
_mode_label = "Top-10% excluded (sensitivity check)" if exclude_top_decile else "Full citation pool"
st.markdown(f"<span style='color:{SUBTEXT};font-size:15px'>ICLR 2018–2020  ·  "
            f"{selected_year}  ·  {_mode_label}</span>", unsafe_allow_html=True)
st.caption(
    "Exploratory live comparisons use one tie ordering and include additional selectors. "
    "All-years metrics average the three annual results equally. The paper headline "
    "below averages 200 random tie orderings and includes the single-call baseline. "
    "Use that snapshot when quoting the paper; the two displays need not match."
)
with st.expander("Paper headline: saved exhibit values"):
    headline_path = _abs("outputs/figures/fig2_headline.csv")
    if os.path.exists(headline_path):
        headline = pd.read_csv(headline_path)
        st.dataframe(headline[["regime", "metric", "value", "tie_lo", "tie_hi",
                               "share_tie_broken"]], hide_index=True)
        st.caption("Frozen paper exhibit; sidebar controls do not change these values. "
                   "tie_lo/tie_hi are the observed range over 200 tie orderings, "
                   "not sampling confidence intervals or exhaustive bounds.")
        if os.path.getmtime(EVAL_TABLE) > os.path.getmtime(headline_path):
            st.warning("The evaluation data is newer than this saved headline. "
                       "Revalidate the exhibit before quoting it.")
    else:
        st.info("The paper headline exhibit has not been generated.")
st.markdown("")

st.markdown('<p class="section-header">Section 1 — Performance by Metric</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">Select a metric to compare regimes. '
            + ('Bars show % of ideal gap remaining (lower = better, 0% = ideal); '
               if show_drawdown else
               'Bars show % of ideal gap closed (higher = better, 100% = ideal); ') +
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
def compute_quadrants(regime_name, lam, years_tuple, mode, impute_zeros, exclude_top_decile=False):
    """Returns pool_df with 'quadrant' column and sets of IDs."""
    pools, regime_ids, ideal_ids, ac_ids = [], set(), set(), set()
    for year in years_tuple:
        pool = prepare_pool(year, mode, impute_zeros, exclude_top_decile)
        _, _, n = get_baselines(year, mode, impute_zeros, exclude_top_decile)
        regime = get_regime(regime_name)
        try:
            sel = set(regime.select(pool, n))
        except Exception:
            sel = set()
        known = pool.dropna(subset=["s2_citations"])
        cutoff_n = math.ceil(1 * n / 100 * 10)  # top-N = same N as acceptance
        cutoff_n = n  # ideal = top-N by citations = same count as selected
        ideal = set(known.nlargest(n, "s2_citations")["paper_id"])
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
        dive_regime_name, lam, tuple(dive_years), mode, impute_zeros, exclude_top_decile)

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
    selected = selected & all_ids
    ground_truth = ground_truth & all_ids
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
citation_ids = set(pool_df.loc[pool_df["s2_citations"].notna(), "paper_id"])
cm_ideal = cm_stats(regime_ids, ideal_ids, citation_ids)
cm_ac    = cm_stats(regime_ids, ac_ids,    all_ids)

col1, col2 = st.columns(2)
col1.caption(f"Citation outcomes available for {len(citation_ids):,} of {len(all_ids):,} submissions.")
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

_resid_df = pool_df.dropna(subset=["mean_rating", "s2_citations"]).copy()
_resid_df["log_cites"] = np.log1p(_resid_df["s2_citations"])

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

scatter_df = pool_df.dropna(subset=["mean_rating","s2_citations"]).copy()
scatter_df["log_cites"] = np.log1p(scatter_df["s2_citations"])

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

merged = pool_df.dropna(subset=["s2_citations"])

# Fetch rejection_tags directly from eval_table to avoid stale cache issues
_rej_map = eval_table.set_index("paper_id")["rejection_tags"] \
    if "rejection_tags" in eval_table.columns else None

# Missed gems: ideal only, sorted by citations
missed_raw = merged[merged["quadrant"] == "ideal only"].nlargest(10, "s2_citations")[
    ["paper_id", "title", "year", "s2_citations", "mean_rating"]].copy()
missed_raw = missed_raw.round({"s2_citations": 0, "mean_rating": 2})
if _rej_map is not None:
    missed_raw["rejection_tags"] = missed_raw["paper_id"].map(_rej_map).fillna("(accepted by AC — no rejection tag)")
missed_raw = missed_raw.drop(columns=["paper_id"])
missed_raw.columns = (["Title", "Year", "Citations", "Avg rating"] +
                      (["Rejection tags"] if _rej_map is not None else []))

# Human consensus errors: regime ∩ AC but NOT in citation ideal
consensus_wrong = merged[
    merged["paper_id"].isin(regime_ids & ac_ids) &
    ~merged["paper_id"].isin(ideal_ids)
].nlargest(10, "s2_citations")[
    ["title", "year", "s2_citations", "mean_rating"]].copy()
consensus_wrong = consensus_wrong.round({"s2_citations": 0, "mean_rating": 2})
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
def _load_hetero_full(_v=4):
    cov_path = "outputs/paper_author_covariates.csv"
    if not os.path.exists(cov_path):
        return None
    et  = _et()
    cov = pd.read_csv(cov_path)
    df  = et.merge(cov, on="paper_id", how="left")
    df["log_cites"] = np.log1p(df["s2_citations"].fillna(0))
    return df

_het_df = _load_hetero_full()

if _het_df is None:
    st.info("Run `python src/build/build_author_covariates.py` to enable this section.")
else:
    total_papers = len(_het_df)
    n_matched = _het_df["top_institution_flag"].notna().sum()

    # ── 3a. COVERAGE TABLE ────────────────────────────────────────────────────
    st.markdown("#### 3a. Covariate Coverage")
    st.markdown('<p class="explainer">'
                'Team size and h-index come from Semantic Scholar, which covers 98.6% of accepted '
                'and 95.8% of rejected papers. The institution flags come from OpenAlex, which '
                'matches 89.0% of accepted but only 62.7% of rejected papers, so coverage is '
                '<b>non-random</b>: estimates for the flags apply to a higher-quality slice of the '
                'pool. h-index and institutions are as of 2026, after the decision.</p>',
                unsafe_allow_html=True)

    cov_rows = [
        {"Covariate": "Any author data",       "N with data": n_matched,
         "% covered": f"{n_matched/total_papers:.1%}",
         "Note": "OpenAlex author match; drives the three flags below"},
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
         "Note": "Semantic Scholar, read in 2026 (post-decision)"},
        {"Covariate": "team_size",             "N with data": int(_het_df["team_size"].notna().sum()),
         "% covered": f"{_het_df['team_size'].notna().mean():.1%}",
         "Note": "Semantic Scholar author count"},
    ]
    st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)

    st.markdown("###### Field × year cell sizes (N papers)")
    _cells = pd.crosstab(eval_table["field"].fillna("(no field label)"),
                         eval_table["year"].astype(int), margins=True, margins_name="Total")
    _cells.columns = _cells.columns.astype(str)
    st.dataframe(_cells, use_container_width=True)
    st.caption(
        "Read this before trusting any field×year estimate. (1) 83% of 2020 papers have no field "
        "label, so every field-FE or field-normalized specification effectively drops most of "
        "2020 — 'field FE' results are largely 2018–2019 results, a sample-composition change as "
        "much as a control. (2) Cells are extremely unbalanced: computer_vision has 15–37 papers "
        "per year and is the alphabetical reference category in the 3c interaction model, so "
        "per-field slope contrasts against it are fragile. (3) citation_pct_rank (the normalized "
        "ground truth) is undefined for unlabeled papers — the field-normalized toggle silently "
        "restricts to this 43% subsample."
    )

    st.markdown("---")

    # ── 3b. RECALL ADVANTAGE BY SUBGROUP ─────────────────────────────────────
    st.markdown("#### 3b. LLM Committee recall@10% advantage vs Human AC — by subgroup")
    st.markdown('<p class="explainer">'
                'Each bar shows the percentage-point gap between LLM Committee recall@10% and '
                'Human AC recall@10% within a subgroup. Positive = LLM outperforms. '
                'If the advantage is stable across "yes" and "no" halves, the result is not '
                'driven by that covariate.</p>', unsafe_allow_html=True)

    @st.cache_data
    def _recall_by_subgroup(years_tuple, mode_key, impute, excl):
        llm = LLMCommittee(); ac = HumanActual()
        rows = []
        base_df = pd.concat([
            prepare_pool(yr, mode_key, impute, excl) for yr in years_tuple
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

    _sub_df = _recall_by_subgroup(tuple(years), mode, impute_zeros, exclude_top_decile)

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
    def _field_recall(years_tuple, mode_key, impute, excl):
        rows = []
        _regs = [LLMCommittee(), HumanActual(), HumanScore()]
        et_sub = _et()
        for field in et_sub["field"].dropna().unique():
            for yr in years_tuple:
                pf = prepare_pool(yr, mode_key, impute, excl)
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

    _fc_df = _field_recall(tuple(years), mode, impute_zeros, exclude_top_decile)

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
        from analysis.fuzzy_rdd import cat_dummies, wls_hc1, wls_hc1_full
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

# Sections 4 (fuzzy RDD), 5 (leakage) and 6 (provenance) live on their own pages
# under src/app/pages/.

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Human reviews are from ICLR 2018–2020 via OpenReview. "
           "LLM reviews generated by a Gemma-4-31B committee pipeline — see sidebar note for details.")
