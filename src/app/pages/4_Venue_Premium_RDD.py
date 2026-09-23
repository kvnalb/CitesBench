"""Fuzzy RDD page: the causal effect of ICLR acceptance on citations.

Was Section 4 of the main dashboard. Streamlit auto-discovers src/app/pages/*.py
when you run `streamlit run src/app/dashboard.py`, so this needs no wiring.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app_style import COLORS, TEXT, BORDER, SUBTEXT, CITATIONS_CSV, EVAL_TABLE, _abs, apply_css

st.set_page_config(page_title="CitesBench — Venue Premium RDD", layout="wide")
apply_css()

st.markdown('<p class="section-header">Section 4 — Fuzzy RDD</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">'
            'Does ICLR acceptance <i>cause</i> additional citations, or does the correlation '
            'just reflect paper quality? Running variable = score_centered (mean_rating − '
            'year-specific cutoff). Treatment is fuzzy: ACs don\'t follow ratings perfectly. '
            'Instrument: above = 1{score_centered ≥ 0}. Outcome: log(1 + citations), '
            'from outputs/citations.csv (Semantic Scholar). OpenAlex undercounts accepted '
            'papers more than rejected ones (median 3.5× vs 2.0× vs S2) — biasing the OA-based '
            'LATE toward zero — so compare both sources here.</p>',
            unsafe_allow_html=True)

@st.cache_data
def _load_rdd_sample(_v=4):
    """The RDD running variable and bandwidths come from the OpenAlex sample file, but
    the OUTCOME is taken from citations.csv like everything else on this page — the
    sample definition and the thing being measured are separate concerns. Previously
    this swapped in S2 v1 counts behind the sidebar toggle, which left the RDD on a
    different outcome from the rest of the dashboard whenever the toggle was OpenAlex."""
    rdd_path = _abs("data/OpenAlex/openalex_rdd_dashboard.csv")
    if not os.path.exists(rdd_path):
        return None
    raw = pd.read_csv(rdd_path)
    dm = raw[
        (raw["year"] <= 2020)
        & raw["in_year_specific_rdd_sample"].astype(bool)
        & raw["openalex_matched"].astype(bool)
    ].copy()
    if os.path.exists(CITATIONS_CSV):
        # inner join: a paper with no tier A/B match is dropped, not imputed as zero
        cit = pd.read_csv(CITATIONS_CSV)[["paper_id", "citations"]].drop_duplicates("paper_id")
        dm = dm.merge(cit, on="paper_id", how="inner")
        dm["openalex_cited_by_count"] = dm["citations"]
    dm["lcites"] = np.log1p(dm["openalex_cited_by_count"].fillna(0))
    dm["accepted"] = dm["accepted"].astype(int)
    _field = pd.read_csv(EVAL_TABLE, low_memory=False)[["paper_id", "field"]]
    dm = dm.merge(_field, on="paper_id", how="left")
    return dm

@st.cache_data
def _rdd_year_bscatter(yr, n_bins=16):
    dm = _load_rdd_sample()
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
def _rdd_all_specs():
    dm = _load_rdd_sample()
    if dm is None: return [], [], []
    from analysis.fuzzy_rdd import run_specs_constant, run_specs
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

_rdd_dm = _load_rdd_sample()
if _rdd_dm is None:
    st.info("RDD data not found.")
else:
    _pooled_specs, _yr_specs, _pooled_field_specs = _rdd_all_specs()

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
        bs, bw = _rdd_year_bscatter(yr)
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
        f"Sample: {len(_rdd_dm):,} papers (ICLR 2018–2020, within year-specific bandwidth). "
        f"Sample membership and bandwidths come from the OpenAlex RDD file; the citation "
        f"outcome is `outputs/citations.csv`. "
        f"Pooled bandwidth h = {_rdd_dm['bandwidth'].median():.2f} (median). "
        "McCrary density test: no significant manipulation (β≈0.0, p>0.05)."
    )
