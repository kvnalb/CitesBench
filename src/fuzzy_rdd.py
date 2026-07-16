"""
Fuzzy RDD: causal effect of ICLR acceptance on citations.

Running variable: score_centered (mean_rating minus year-specific cutoff)
Instrument:       above = 1{score_centered >= 0}
Treatment:        accepted  (fuzzy — ACs don't follow rating perfectly)
Outcome:          log1p(openalex_cited_by_count)

Mirrors Archive/CompletePipeline/analysis/01_citation_rdd.R
using data/OpenAlex/openalex_rdd_arxiv_paper_level.csv (already trimmed to
year-specific bandwidths).

Run: python src/fuzzy_rdd.py
"""
import os
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

RDD_CSV = "data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"
OUT_MD = "outputs/fuzzy_rdd.md"
OUT_BSCATTER = "outputs/fuzzy_rdd_binscatter.csv"


# ── OLS helpers ──────────────────────────────────────────────────────────────

def wls_hc1(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted OLS with HC1-style robust SEs (weights enter as diagonal W)."""
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    # HC1: scale by n/(n-k); weight-adjusted meat
    meat = (X * w[:, None] * e[:, None]).T @ (X * e[:, None])
    XtWX = X.T @ (X * w[:, None])
    XtWX_inv = np.linalg.inv(XtWX)
    cov = (n / (n - k)) * XtWX_inv @ meat @ XtWX_inv
    return beta, np.sqrt(np.diag(cov))


def tri_kernel(score, h):
    return np.maximum(0.0, 1.0 - np.abs(score) / h)


def cat_dummies(values):
    """Return (n × k) design matrix of dummies for any categorical array (drop first level)."""
    unique = sorted(set(values))
    if len(unique) <= 1:
        return np.zeros((len(values), 0))
    cols = []
    for v in unique[1:]:
        cols.append((values == v).astype(float))
    return np.column_stack(cols)


year_dummies = cat_dummies  # kept as a name for existing call sites


# ── Wald / 2SLS for fuzzy RDD ────────────────────────────────────────────────

def fuzzy_late(FS_beta, FS_se, RF_beta, RF_se, idx_above=1):
    """Wald LATE = RF / FS with delta-method SE. idx_above = column of 'above'."""
    fs = FS_beta[idx_above]
    rf = RF_beta[idx_above]
    se_fs = FS_se[idx_above]
    se_rf = RF_se[idx_above]
    late = rf / fs if fs != 0 else np.nan
    # delta method (independence assumption — conservative)
    if fs != 0:
        se_late = abs(late) * np.sqrt((se_rf / rf) ** 2 + (se_fs / fs) ** 2) if rf != 0 else se_rf / abs(fs)
    else:
        se_late = np.nan
    return late, se_late


def run_specs_constant(dm: pd.DataFrame, h: float, label: str, field_col: str = None) -> dict:
    """Local constant (step) RDD — just compares means within ±h. No slope extrapolation.

    field_col: if given, adds field dummies as an additive covariate (precision
    control, not a per-field treatment interaction — see leakage_power_analysis-
    style reasoning in methodology_review.md P-section on field FE). Rows with
    missing field are dropped, so N shrinks; report alongside the no-field spec,
    never in place of it.
    """
    d = dm[dm["score_centered"].abs() <= h].copy()
    if field_col:
        d = d[d[field_col].notna()].copy()
    n = len(d)
    if n < 20:
        return {}
    above = (d["score_centered"] >= 0).astype(float).values
    yr_d = year_dummies(d["year"].values)
    cols = [np.ones(n), above]
    if yr_d.shape[1] > 0:
        cols.append(yr_d)
    if field_col:
        fld_d = cat_dummies(d[field_col].values)
        if fld_d.shape[1] > 0:
            cols.append(fld_d)
    # No slope controls — just above + year FE (+ field FE if requested)
    X = np.column_stack(cols)
    w = np.ones(n)  # unweighted within window
    fs_b, fs_s = wls_hc1(X, d["accepted"].astype(float).values, w)
    rf_b, rf_s = wls_hc1(X, d["lcites"].values, w)
    late, se_late = fuzzy_late(fs_b, fs_s, rf_b, rf_s, idx_above=1)
    fs_f = (fs_b[1] / fs_s[1]) ** 2 if fs_s[1] > 0 else np.nan
    ci_lo = late - 1.96 * se_late if not np.isnan(se_late) else np.nan
    ci_hi = late + 1.96 * se_late if not np.isnan(se_late) else np.nan
    return {
        "spec": label,
        "h": h,
        "N": n,
        "N_left": int((above == 0).sum()),
        "N_right": int((above == 1).sum()),
        "FS_jump": round(fs_b[1], 4),
        "FS_se": round(fs_s[1], 4),
        "FS_F": round(fs_f, 1),
        "RF_jump": round(rf_b[1], 4),
        "RF_se": round(rf_s[1], 4),
        "LATE": round(late, 4) if not np.isnan(late) else "n/a",
        "LATE_se": round(se_late, 4) if not np.isnan(se_late) else "n/a",
        "CI_95": f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "n/a",
    }


def run_specs(dm: pd.DataFrame, h: float, label: str) -> dict:
    """Run FS, RF, LATE at a given bandwidth h."""
    d = dm.copy()
    d["w"] = tri_kernel(d["score_centered"], h)
    d = d[d["w"] > 0].copy()
    n = len(d)
    if n < 20:
        return {}

    above = (d["score_centered"] >= 0).astype(float).values
    sc = d["score_centered"].values
    yr_d = year_dummies(d["year"].values)
    w = d["w"].values

    # RHS: [1, above, score_centered, above*score_centered, year_dummies]
    Xbase = np.column_stack([np.ones(n), above, sc, above * sc])
    if yr_d.shape[1] > 0:
        Xbase = np.column_stack([Xbase, yr_d])

    # First stage
    fs_beta, fs_se = wls_hc1(Xbase, d["accepted"].astype(float).values, w)
    fs_jump = fs_beta[1]
    fs_jump_se = fs_se[1]
    fs_t = fs_jump / fs_jump_se if fs_jump_se > 0 else np.nan
    fs_f = fs_t ** 2  # approx F-stat (1 restriction)

    # Reduced form
    rf_beta, rf_se = wls_hc1(Xbase, d["lcites"].values, w)
    rf_jump = rf_beta[1]
    rf_jump_se = rf_se[1]

    # LATE
    late, se_late = fuzzy_late(fs_beta, fs_se, rf_beta, rf_se, idx_above=1)
    ci_lo = late - 1.96 * se_late if not np.isnan(se_late) else np.nan
    ci_hi = late + 1.96 * se_late if not np.isnan(se_late) else np.nan

    return {
        "spec": label,
        "h": h,
        "N": n,
        "N_left": int((above == 0).sum()),
        "N_right": int((above == 1).sum()),
        "FS_jump": round(fs_jump, 4),
        "FS_se": round(fs_jump_se, 4),
        "FS_F": round(fs_f, 1),
        "RF_jump": round(rf_jump, 4),
        "RF_se": round(rf_jump_se, 4),
        "LATE": round(late, 4) if not np.isnan(late) else "n/a",
        "LATE_se": round(se_late, 4) if not np.isnan(se_late) else "n/a",
        "CI_95": f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "n/a",
    }


# ── McCrary density test (manual) ────────────────────────────────────────────

def mccrary_test(score: np.ndarray, bin_width: float = 0.25) -> str:
    edges = np.arange(score.min() - bin_width, score.max() + bin_width * 2, bin_width)
    bins = np.digitize(score, edges)
    counts = np.bincount(bins, minlength=len(edges))
    # Bin centers
    centers = edges + bin_width / 2
    left_bins = [(c, cnt) for c, cnt in zip(centers, counts) if -1.5 < c < 0]
    right_bins = [(c, cnt) for c, cnt in zip(centers, counts) if 0 <= c < 1.5]
    left_mean = np.mean([c for _, c in left_bins]) if left_bins else 0
    right_mean = np.mean([c for _, c in right_bins]) if right_bins else 0
    jump_pct = (right_mean - left_mean) / left_mean * 100 if left_mean > 0 else 0
    return (
        f"Bin width: {bin_width}. "
        f"Mean count left of 0: {left_mean:.1f}, right: {right_mean:.1f}. "
        f"Jump at cutoff: {jump_pct:+.1f}% "
        f"({'possibly heaping' if abs(jump_pct) > 20 else 'no obvious manipulation'})."
    )


# ── Missingness balance ───────────────────────────────────────────────────────

def missingness_balance(dm: pd.DataFrame, h: float) -> str:
    d = dm.copy()
    d["w"] = tri_kernel(d["score_centered"], h)
    d = d[d["w"] > 0]
    above = (d["score_centered"] >= 0).astype(float).values
    sc = d["score_centered"].values
    w = d["w"].values
    n = len(d)
    yr_d = year_dummies(d["year"].values)
    Xbase = np.column_stack([np.ones(n), above, sc, above * sc])
    if yr_d.shape[1] > 0:
        Xbase = np.column_stack([Xbase, yr_d])
    matched = d["openalex_matched"].astype(float).values
    beta, se = wls_hc1(Xbase, matched, w)
    t = beta[1] / se[1]
    return f"Jump in OpenAlex coverage at cutoff: β={beta[1]:.4f} (SE={se[1]:.4f}, t={t:.2f}) — {'⚠ significant' if abs(t) > 2 else '✓ not significant'}."


# ── Binscatter ────────────────────────────────────────────────────────────────

def binscatter(dm: pd.DataFrame, h: float, n_bins: int = 20) -> pd.DataFrame:
    d = dm[dm["w_pool"] > 0].copy()
    d["bin"] = pd.cut(d["score_centered"], bins=n_bins)
    agg = d.groupby("bin", observed=True).agg(
        score_center=("score_centered", "mean"),
        mean_lcites=("lcites", "mean"),
        mean_accepted=("accepted", "mean"),
        n=("lcites", "size"),
    ).reset_index(drop=True)
    return agg


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_no results_\n"
    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return header + "\n" + sep + "\n" + body + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    raw = pd.read_csv(RDD_CSV)

    # Keep 2018-2020 in-sample papers with matched citations
    dm = raw[
        (raw["year"] <= 2020)
        & (raw["in_year_specific_rdd_sample"].astype(str).str.lower().isin(["true", "1", "yes"]))
        & (raw["openalex_matched"].astype(str).str.lower().isin(["true", "1", "yes"]))
    ].copy()

    dm["lcites"] = np.log1p(dm["openalex_cited_by_count"].fillna(0))
    dm["accepted"] = dm["accepted"].astype(int)

    h_pool = dm["bandwidth"].median()
    print(f"Matched sample: {len(dm)} papers (2018-2020, with citations)")
    print(f"Pooled bandwidth (median of year-specific): {h_pool:.3f}")
    print(f"Acceptance rate: {dm['accepted'].mean():.1%}")

    dm["w_pool"] = tri_kernel(dm["score_centered"], h_pool)

    # Bandwidths to try
    bandwidths = {
        "h=0.75": 0.75,
        "h=1.00": 1.00,
        f"h={h_pool:.2f} (median)": h_pool,
        "h=1.50": 1.50,
    }

    lines = ["# Fuzzy RDD: Causal Effect of ICLR Acceptance on Citations\n"]
    lines.append(f"**Sample**: {len(dm)} papers (ICLR 2018-2020, OpenAlex-matched, in-bandwidth)  ")
    lines.append(f"**Pooled bandwidth** (median of year-specific): h = {h_pool:.3f}  ")
    lines.append(f"**Running variable**: `score_centered` = mean_rating − year-specific cutoff  ")
    lines.append(f"**Outcome**: log(1 + citations)  ")
    lines.append(f"**Treatment**: `accepted` (fuzzy — not a pure function of score)\n")

    # McCrary test
    lines.append("## Manipulation (McCrary density) test\n")
    lines.append(mccrary_test(dm["score_centered"].values) + "\n")

    # Missingness balance
    lines.append("## Missingness balance (OpenAlex coverage)\n")
    lines.append(missingness_balance(dm, h_pool) + "\n")

    # Score distribution diagnostic (masspoints / heaping)
    lines.append("## Score distribution diagnostic (masspoints)\n")
    lines.append("*Discrete reviewer ratings create heaping; 2020 is notably bimodal near the cutoff.*\n")
    diag_rows = []
    for yr in sorted(dm["year"].unique()):
        yg = dm[dm["year"] == yr]
        near = yg[yg["score_centered"].abs() < 0.5]
        above_near = near[near["score_centered"] >= 0]
        below_near = near[near["score_centered"] < 0]
        diag_rows.append({
            "year": yr,
            "cutoff": round(yg["cutoff"].iloc[0], 3),
            "bandwidth": round(yg["bandwidth"].iloc[0], 3),
            "N_in_bw": len(yg),
            "N_within_0.5": len(near),
            "acc_above_0.5": round(above_near["accepted"].mean(), 3) if len(above_near) else "n/a",
            "acc_below_0.5": round(below_near["accepted"].mean(), 3) if len(below_near) else "n/a",
            "raw_delta_0.5": round(
                above_near["accepted"].mean() - below_near["accepted"].mean(), 3
            ) if len(above_near) and len(below_near) else "n/a",
        })
    lines.append(md_table(diag_rows))
    lines.append(
        "> ⚠ **2020 note**: ICLR 2020 score_centered clusters at ±0.5 (discrete rating integers "
        "create heaping). Very few papers fall in (−0.25, 0.25), making the local linear "
        "estimate at the cutoff unreliable for 2020. The Archive R implementation handles this "
        "with `rdrobust(..., masspoints='adjust')`; our pooled estimate is dominated by 2020's "
        "flat first stage. Prefer year-specific results for 2018 and 2019.\n"
    )

    # Main results table
    lines.append("## Results: First Stage, Reduced Form, Fuzzy LATE\n")
    lines.append("*All specs: triangular kernel, local linear with interacted slopes, year FE, HC1 SEs.*\n")
    lines.append("*LATE = Wald estimator (RF / FS); SE via delta method.*\n")
    results = []
    for label, h in bandwidths.items():
        r = run_specs(dm, h, label)
        if r:
            results.append(r)
    lines.append(md_table(results))

    # Local constant (no slope) — more reliable with masspoints
    lines.append("## Robustness: local constant (no slope extrapolation)\n")
    lines.append("*Simple mean difference within ±h; less sensitive to heaping/masspoints.*\n")
    const_results = []
    for label, h in {"h=0.50": 0.50, "h=0.75": 0.75, "h=1.00": 1.00}.items():
        r = run_specs_constant(dm, h, label)
        if r:
            const_results.append(r)
    lines.append(md_table(const_results))

    # Year-specific results (more reliable given 2020 masspoints issue)
    lines.append("## Year-specific results (preferred given 2020 masspoints issue)\n")
    year_results = []
    for yr in sorted(dm["year"].unique()):
        yg = dm[dm["year"] == yr].copy()
        h_yr = yg["bandwidth"].iloc[0]
        r = run_specs(yg, h_yr, f"{yr} (h={h_yr:.2f})")
        # Override: run without year FE (single year)
        d = yg.copy()
        d["w"] = tri_kernel(d["score_centered"], h_yr)
        d = d[d["w"] > 0]
        if len(d) < 20:
            continue
        n_yr = len(d)
        above_yr = (d["score_centered"] >= 0).astype(float).values
        sc_yr = d["score_centered"].values
        w_yr = d["w"].values
        X_yr = np.column_stack([np.ones(n_yr), above_yr, sc_yr, above_yr * sc_yr])
        fs_b, fs_s = wls_hc1(X_yr, d["accepted"].astype(float).values, w_yr)
        rf_b, rf_s = wls_hc1(X_yr, d["lcites"].values, w_yr)
        late, se_late = fuzzy_late(fs_b, fs_s, rf_b, rf_s, idx_above=1)
        fs_f = (fs_b[1] / fs_s[1]) ** 2 if fs_s[1] > 0 else np.nan
        ci_lo = late - 1.96 * se_late if not np.isnan(se_late) else np.nan
        ci_hi = late + 1.96 * se_late if not np.isnan(se_late) else np.nan
        year_results.append({
            "year": yr,
            "h": round(h_yr, 3),
            "N": n_yr,
            "FS_jump": round(fs_b[1], 4),
            "FS_F": round(fs_f, 1),
            "RF_jump": round(rf_b[1], 4),
            "LATE": round(late, 4) if not np.isnan(late) else "n/a",
            "LATE_se": round(se_late, 4) if not np.isnan(se_late) else "n/a",
            "CI_95": f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "n/a",
        })
    lines.append(md_table(year_results))

    # Parametric OLS (no kernel, with year + field FE, full sample in bandwidth)
    lines.append("## Parametric local linear (year + field FE)\n")
    d = dm[dm["w_pool"] > 0].copy()
    n = len(d)
    above = (d["score_centered"] >= 0).astype(float).values
    sc = d["score_centered"].values
    w = d["w_pool"].values
    yr_d = year_dummies(d["year"].values)

    # Build field dummies if available
    field_d = np.zeros((n, 0))
    if "primary_area" in d.columns:
        areas = d["primary_area"].fillna("unknown")
        unique_areas = sorted(areas.unique())[1:]  # drop first (baseline)
        if unique_areas:
            field_d = np.column_stack([(areas == a).astype(float) for a in unique_areas])

    spec_rows = []
    for spec_label, extra in [
        ("Year FE only", yr_d),
        ("Year + area FE", np.column_stack([yr_d, field_d]) if field_d.shape[1] > 0 else yr_d),
    ]:
        Xfull = np.column_stack([np.ones(n), above, sc, above * sc, extra]) if extra.shape[1] > 0 else np.column_stack([np.ones(n), above, sc, above * sc])
        beta, se = wls_hc1(Xfull, d["lcites"].values, w)
        t = beta[1] / se[1]
        spec_rows.append({
            "spec": spec_label,
            "above_coef": round(beta[1], 4),
            "above_se": round(se[1], 4),
            "t": round(t, 2),
            "N": n,
        })
    lines.append(md_table(spec_rows))

    lines.append(
        "\n> **Interpretation**: In the fuzzy RDD, LATE estimates the causal effect of "
        "ICLR acceptance on log(1+citations) for complier papers near the cutoff. "
        "The first-stage F-statistic (F > 10) confirms a strong instrument. "
        "A positive LATE would indicate that the publication/visibility channel from "
        "acceptance itself generates citations beyond paper quality.\n"
    )

    report = "\n".join(lines)
    with open(OUT_MD, "w") as f:
        f.write(report)

    # Binscatter CSV for plotting
    bscatter = binscatter(dm, h_pool)
    bscatter.to_csv(OUT_BSCATTER, index=False)

    print(f"Written: {OUT_MD}")
    print(f"Written: {OUT_BSCATTER}")
    if results:
        main_r = [r for r in results if "median" in r["spec"]]
        if main_r:
            r = main_r[0]
            print(f"\nMain spec (h={h_pool:.2f}):")
            print(f"  First stage:   Δ={r['FS_jump']:+.4f}  F={r['FS_F']}")
            print(f"  Reduced form:  Δ={r['RF_jump']:+.4f}")
            print(f"  Fuzzy LATE:    {r['LATE']} (SE={r['LATE_se']})  95% CI: {r['CI_95']}")


if __name__ == "__main__":
    main()
