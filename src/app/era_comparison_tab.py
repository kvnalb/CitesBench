"""
Era comparison tab: the 2018-2020 vs 2025 result under a suite of measures.

Reads only. Everything here is produced by src/analysis/metric_suite.py and
src/analysis/did_leakage.py; this module renders and does not recompute, so the
dashboard cannot disagree with the committed numbers.
"""
import os

import pandas as pd
import streamlit as st

METRICS_CSV = "outputs/metric_suite.csv"
DID_CSV = "outputs/metric_suite_did.csv"
SUITE_MD = "outputs/metric_suite.md"
DID_MD = "outputs/did_leakage.md"
ERA_PNG = "outputs/era_comparison.png"
AIVH_PNG = "outputs/ai_vs_human_2025.png"

PRETTY = {
    "spearman_rho": "Spearman ρ", "kendall_tau_b": "Kendall τ-b",
    "somers_d": "Somers' D", "auc_top10": "Top-decile AUC",
    "recall_at_10": "Recall@10%", "ndcg_at_10": "NDCG@10%",
}
FAMILY = {
    "spearman_rho": "rank agreement", "kendall_tau_b": "rank agreement",
    "somers_d": "rank agreement", "auc_top10": "decision-relevant",
    "recall_at_10": "decision-relevant", "ndcg_at_10": "decision-relevant",
}


def _missing(paths):
    return [p for p in paths if not os.path.exists(p)]


@st.cache_data
def _load(path):
    return pd.read_csv(path)


def _headline(did):
    """Share of the 2018-2020 advantage that survives into 2025, per metric.
    Unitless, so metrics on different scales are comparable side by side."""
    d = did.copy()
    d["retained"] = d["adv_2025"] / d["adv_1820"]
    d["metric_name"] = d["metric"].map(PRETTY)
    d["family"] = d["metric"].map(FAMILY)
    return d


def render():
    st.title("Era comparison — 2018–2020 vs 2025")
    st.caption(
        "Can the LLM committee rank accepted ICLR papers by eventual citation "
        "impact, does it beat the humans, and is its 2018–2020 advantage real "
        "skill or memorisation? Accepted papers only on both sides."
    )

    gone = _missing([METRICS_CSV, DID_CSV])
    if gone:
        st.warning(
            "Results not built yet. Run:\n\n"
            "```bash\npython src/analysis/metric_suite.py --tier-a-only\n```\n\n"
            f"Missing: {', '.join(gone)}"
        )
        return

    metrics, did = _load(METRICS_CSV), _load(DID_CSV)
    h = _headline(did)

    st.subheader("Why more than one measure")
    st.markdown(
        "Spearman ρ was carrying the whole argument and is a weak single choice "
        "here: it weights a swap at ranks 800/900 like one at 3/400 when the "
        "decision is *which 10% to spotlight*; it has no units; our scores take "
        "only 15–21 distinct values, so it largely summarises how ties break; and "
        "its attenuation under a noisier outcome is the exact disputed step in the "
        "leakage argument. The measures below fail in different ways, so agreement "
        "across them is worth more than any one of them."
    )

    # ---------------------------------------------------------------- per era
    st.subheader("Each selector, each era")
    era = st.radio("Era", sorted(metrics["era"].unique()), horizontal=True)
    m = metrics[metrics["era"] == era].set_index("selector")
    show = m[list(PRETTY)].rename(columns=PRETTY)
    show["Effect (pct pts / SD)"] = m["ols_pctpts_per_sd"].round(2).astype(str) \
        + " ± " + m["ols_se"].round(2).astype(str)
    show["NB log-IRR / SD"] = m["nb_log_irr_per_sd"].round(3).astype(str) \
        + " ± " + m["nb_se"].round(3).astype(str)
    st.dataframe(show.style.format(precision=3), use_container_width=True)
    st.caption(
        f"n = {int(m['n'].iloc[0]):,} papers. Effect and NB standard errors are "
        "clustered by field — earlier unclustered numbers in this repo read as "
        "optimistic. AUC 0.5 and Recall@10% of 10% are the no-skill baselines."
    )

    # ------------------------------------------------------------ the headline
    st.subheader("Does the committee's edge over humans survive into 2025?")
    st.markdown(
        "Within an era the LLM and the humans are scored against the **identical** "
        "outcome, so the citation window cancels in the within-era difference. "
        "Comparing that difference across eras isolates what only the LLM could "
        "have had — having seen the papers. Below: the share of its 2018–2020 "
        "advantage that survives, which is unitless and so comparable across "
        "measures. **1.0 means no erosion.**"
    )
    control = st.selectbox("Control group", sorted(h["control"].unique()))
    hc = h[h["control"] == control].set_index("metric_name")
    st.bar_chart(hc["retained"], height=260)

    tbl = hc[["family", "adv_1820", "adv_2025", "retained", "did",
              "ci_lo", "ci_hi", "frac_le_0"]].rename(columns={
        "family": "family", "adv_1820": "advantage 2018–2020",
        "adv_2025": "advantage 2025", "retained": "share retained",
        "did": "DiD", "ci_lo": "CI low", "ci_hi": "CI high",
        "frac_le_0": "frac of resamples ≤ 0"})
    st.dataframe(tbl.style.format(precision=3), use_container_width=True)
    st.caption(
        "DiD = advantage 2018–2020 minus advantage 2025, bootstrapped by resampling "
        "papers within each era. A DiD above zero means the committee leads the "
        "humans by more where it has plausibly seen the papers."
    )

    # --------------------------------------------------------------- the rest
    st.subheader("Full write-ups")
    c1, c2 = st.columns(2)
    for col, path, label in [(c1, SUITE_MD, "Metric suite"),
                             (c2, DID_MD, "Leakage difference-in-differences")]:
        with col:
            with st.expander(label, expanded=False):
                if os.path.exists(path):
                    st.markdown(open(path).read())
                else:
                    st.info(f"{path} not built yet.")

    st.subheader("Figures")
    for path, cap in [(ERA_PNG, "2018–2020 vs 2025"),
                      (AIVH_PNG, "2025: committee vs human reviewers vs area chairs")]:
        if os.path.exists(path):
            st.image(path, caption=cap, use_container_width=True)

    with st.expander("What this does not establish", expanded=False):
        st.markdown(
            "- **Text provenance is not ruled out.** The LLM's input differs across "
            "eras (archive OCR vs ReviewArena markdown) while the humans read PDFs "
            "in both. `docs/reviewarena_text_quality.md` compared ReviewArena 2020 "
            "to ReviewArena 2025 — never archive-vs-ReviewArena, because the "
            "archive's fulltext was unavailable. Re-running the committee on "
            "ReviewArena's 2,213 papers from 2020 would isolate this exactly.\n"
            "- **Citation coverage is still unequal** between eras while the "
            "title-match fetch runs; magnitudes will move.\n"
            "- **Both eras condition on acceptance**, which is a collider. The "
            "selection is the same in kind but acceptance rates differ.\n"
            "- **The direct test is the age-window measure** — 2018–2020 citations "
            "truncated to 18 months, from the S2 edge list (issue #11). The DiD is "
            "the argument; that would be the proof."
        )
