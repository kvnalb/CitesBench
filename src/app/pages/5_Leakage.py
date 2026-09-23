"""Leakage test page: is the LLM judging papers or recalling them?

Was Section 5 of the main dashboard. Streamlit auto-discovers src/app/pages/*.py
when you run `streamlit run src/app/dashboard.py`, so this needs no wiring.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app_style import COLORS, IDEAL_COLOR, RANDOM_COLOR, TEXT, BORDER, SUBTEXT, EVAL_TABLE, apply_css

st.set_page_config(page_title="CitesBench — Leakage Test", layout="wide")
apply_css()

eval_table = st.cache_data(pd.read_csv)(EVAL_TABLE, low_memory=False)

st.markdown('<p class="section-header">Section 5 — Leakage Test</p>', unsafe_allow_html=True)
st.markdown('<p class="explainer">'
            'The LLM committee was run in 2024–2026 on ICLR 2018–2020 papers — its training data '
            'may already know which papers became famous. If so, "LLM beats human" could reflect '
            'memorized hindsight, not judgment. Six tests below: '
            'decision-recall probe (LAP), a probe-validity placebo check, a fame-recall probe, '
            'a masked re-review ablation, a leakage-excluded re-run of the headline comparison, '
            'and an abstract-completion extraction probe. '
            'Scripts: <code>src/probes/leakage_lap_v1.py</code>, <code>leakage_fame_v1.py</code>, '
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
    st.info("Run `python src/probes/leakage_lap_v1.py --full` and `leakage_fame_v1.py --full` to enable this section.")
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
        "Probe accuracy columns use OpenAlex ground truth fixed at probe time. "
        "OpenAlex undercounts (median 2.9× vs S2, worse for "
        "accepted papers), so fame-direction accuracy is a lower bound."
    )

    if _leak["controls"] is not None:
        _ctrl = _leak["controls"]
        st.markdown("##### Probe validity (placebo controls)")
        st.markdown('<p class="explainer">'
                    'N is power/precision-justified, not arbitrary (see '
                    '<code>src/analysis/leakage_power_analysis.py</code>, '
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
    _rc["log_cites"] = np.log1p(_rc["s2_citations"])
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
    # One ground truth now, so the S2 exclusion arm is the only arm — it is used
    # whenever it has been built, rather than following a sidebar toggle.
    _use_s2_excl = _leak["exclusion_s2"] is not None
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
                    "Script: src/analysis/leakage_exclusion_bootstrap.py."
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
                + ("Semantic Scholar ground truth ("
                   "src/analysis/leakage_exclusion_eval.py --citation-source s2). Under S2, exclusion "
                   "helps every regime — famous excluded papers carry even more of the citation "
                   "mass — but helps humans far more: the LLM Committee's full-pool edge over "
                   "Human AC nearly vanishes on the leakage-excluded pool."
                   if _use_s2_excl else
                   "Legacy OpenAlex ground truth — the S2 exclusion arm has not been built. "
                   "Run src/analysis/leakage_exclusion_eval.py --citation-source s2.")
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
            eval_table[["paper_id", "title", "s2_citations"]], on="paper_id", how="left")
        if len(_exhibit):
            _exhibit_disp = _exhibit[["title", "s2_citations", "rougeL_margin", "eight_target"]].copy()
            _exhibit_disp.columns = ["Title", "Citations", "ROUGE-L margin", "8-gram hit rate"]
            _exhibit_disp["Citations"] = _exhibit_disp["Citations"].map(
                lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
            _exhibit_disp["ROUGE-L margin"] = _exhibit_disp["ROUGE-L margin"].map("{:+.3f}".format)
            _exhibit_disp["8-gram hit rate"] = _exhibit_disp["8-gram hit rate"].map("{:.1%}".format)
            st.dataframe(_exhibit_disp, use_container_width=True, hide_index=True)
            st.caption("Most-extractable papers. Generated/reference texts archived in "
                       "outputs/leakage_abstract_completion_texts.jsonl.")
