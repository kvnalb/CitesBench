"""
Three standard robustness exercises for the venue-premium RD.

Motivated by one observation: the estimate moved from +0.250 to +0.342 when year
fixed effects entered. A valid RD should be insensitive to covariates, because
covariates are already balanced at the cutoff, so movement means either the
covariates are NOT balanced (a design problem) or years were pooled that should
not have been (a bookkeeping problem). These three exercises tell them apart.

A. BALANCE TEST. Run the same RD with a pre-determined characteristic as the
   outcome. Crossing a score threshold cannot change a paper's topic or its
   abstract length, so a jump in either is causally impossible and therefore an
   artifact. No jumps means covariate adjustment can only change precision.

B. PER-YEAR ESTIMATES, PRECISION-WEIGHTED. Each year has its own cutoff (5.67,
   6.00, 5.50) and its own accept rate, so a pooled "above cutoff" dummy partly
   measures which year a paper came from. Estimating within year removes that by
   construction. The years are then combined with inverse-variance weights, and
   Cochran's Q asks whether they are estimating a common parameter at all.

C. SPECIFICATION CURVE. Bandwidth x polynomial order x covariates x kernel, every
   combination, sorted by point estimate. Reporting one specification invites
   "why that one"; reporting the whole space answers it in advance.

Every estimate here is the FUZZY premium, the citation jump divided by the
acceptance jump, not the reduced form. See src/analysis/venue_premium_rdd.py.

Run: python src/analysis/venue_premium_robustness.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs                      # noqa: E402
from venue_premium_rdd import load, design, ols_hc1, BW       # noqa: E402

OUT_BALANCE = "outputs/venue_premium_balance.csv"
OUT_PERYEAR = "outputs/venue_premium_by_year.csv"
OUT_SPECCURVE = "outputs/venue_premium_speccurve.csv"
OUT_MD = "outputs/venue_premium_robustness.md"
FIG_BALANCE = "outputs/figures/venue_premium_balance"
FIG_PERYEAR = "outputs/figures/venue_premium_by_year"
FIG_SPECCURVE = "outputs/figures/venue_premium_speccurve"

BANDWIDTHS = [0.50, 0.75, 1.00, 1.25, 1.50]
POLYS = [1, 2]
COVARIATE_SETS = [("none", None), ("year", ["year"]), ("year x topic", ["year", "topic"])]
KERNELS = ["uniform", "triangular"]


# --------------------------------------------------------------- estimation
def design_p(d, poly, fe):
    """Design matrix with polynomial order `poly` on each side of the cutoff."""
    cols = [np.ones(len(d)), d.above.to_numpy()]
    names = ["const", "treat"]
    for p in range(1, poly + 1):
        cols += [(d.r ** p).to_numpy(), (d.above * d.r ** p).to_numpy()]
        names += [f"r{p}", f"above_x_r{p}"]
    if fe:
        key = d[fe[0]].astype(str)
        for c in fe[1:]:
            key = key + "_" + d[c].astype(str)
        dm = pd.get_dummies(key, drop_first=True).to_numpy(dtype=float)
        cols += [dm[:, j] for j in range(dm.shape[1])]
        names += [f"fe{j}" for j in range(dm.shape[1])]
    return np.column_stack(cols), names


def wls_hc1(X, y, w):
    """Weighted OLS with an HC1 sandwich, weights entering as a diagonal."""
    Xw = X * w[:, None]
    XtX_inv = np.linalg.pinv(X.T @ Xw)
    b = XtX_inv @ (Xw.T @ y)
    e = y - X @ b
    n, k = X.shape
    return b, e, XtX_inv, n / max(n - k, 1)


def fuzzy(d, poly=1, fe=None, kernel="uniform", h=None):
    """Fuzzy-RD premium on a sample: (citation jump) / (acceptance jump).

    Both stages share one design and one sample, so the delta-method variance can
    use the true covariance between the two coefficients rather than dropping it.
    """
    s = d.dropna(subset=["y"]).copy()
    if h is not None:
        s = s[s.r.abs() <= h]
    if len(s) < 60 or s.above.nunique() < 2:
        return dict(late=np.nan, se=np.nan, rf=np.nan, fs=np.nan, F=np.nan, n=len(s))

    hh = h if h is not None else s.h.max()
    w = np.ones(len(s)) if kernel == "uniform" else (1 - (s.r.abs() / hh)).to_numpy()
    X, names = design_p(s, poly, fe)
    i = names.index("treat")

    b_rf, e_rf, inv, scale = wls_hc1(X, s.y.to_numpy(), w)
    b_fs, e_fs, _, _ = wls_hc1(X, s.D.to_numpy(), w)

    def sw(ea, eb):
        return (inv @ (X * (w * ea * eb)[:, None]).T @ (X * w[:, None])
                @ inv * scale)[i, i]

    v_rf, v_fs, cov = sw(e_rf, e_rf), sw(e_fs, e_fs), sw(e_rf, e_fs)
    rf, fsj = b_rf[i], b_fs[i]
    if fsj == 0:
        return dict(late=np.nan, se=np.nan, rf=rf, fs=fsj, F=np.nan, n=len(s))
    late = rf / fsj
    var = v_rf / fsj ** 2 + rf ** 2 * v_fs / fsj ** 4 - 2 * rf * cov / fsj ** 3
    return dict(late=late, se=np.sqrt(max(var, 0)), rf=rf, fs=fsj,
                F=(fsj ** 2 / v_fs) if v_fs > 0 else np.nan, n=len(s))


# ------------------------------------------------------------------ A. balance
def balance(d):
    """Same RD, pre-determined characteristics as outcomes. Jumps are impossible."""
    import sqlite3
    d = d.copy()
    con = sqlite3.connect("data/gen_review.db")
    ab = pd.read_sql("SELECT id AS paper_id, abstract FROM SUBMISSION", con)
    con.close()
    d["abstract_len"] = d.paper_id.map(
        ab.set_index("paper_id").abstract.fillna("").str.len())
    tests = [("number of reviews", "n_reviews"),
             ("abstract length (chars)", "abstract_len"),
             ("reviewer disagreement (sd)", "rating_std")]
    # topic is categorical: test the share in each of the largest clusters
    top = d.topic.value_counts().head(4).index
    for t in top:
        d[f"topic_{t}"] = (d.topic == t).astype(float)
        tests.append((f"share in topic {t}", f"topic_{t}"))
    # year composition is the pooling problem stated as a balance test
    for y in spec.YEARS[:-1]:
        d[f"is_{y}"] = (d.year == y).astype(float)
        tests.append((f"share from {y}", f"is_{y}"))

    rows = []
    for label, col in tests:
        s = d.dropna(subset=[col])
        X, names = design_p(s, 1, None)
        b, e, inv, scale = wls_hc1(X, s[col].to_numpy(), np.ones(len(s)))
        i = names.index("treat")
        se = np.sqrt((inv @ (X * (e ** 2)[:, None]).T @ X @ inv * scale)[i, i])
        sd = float(s[col].std()) or 1.0
        rows.append({"characteristic": label, "jump": b[i], "se": se,
                     "t": b[i] / se, "n": len(s),
                     # in SD units, so a character count and a 0/1 share are
                     # comparable on one axis
                     "jump_sd": b[i] / sd, "se_sd": se / sd})
    t = pd.DataFrame(rows)
    t["fails"] = t.t.abs() > 1.96
    return t


# ----------------------------------------------------------------- B. per year
def per_year(d):
    rows = []
    for year in spec.YEARS:
        r = fuzzy(d[d.year == year], poly=1, fe=None, h=BW[year])
        rows.append({"year": year, **r})
    t = pd.DataFrame(rows)

    ok = t.dropna(subset=["late", "se"])
    ok = ok[ok.se > 0]
    w = 1 / ok.se ** 2
    pooled = float((ok.late * w).sum() / w.sum())
    pooled_se = float(np.sqrt(1 / w.sum()))
    # Cochran's Q: do the years differ by more than their own noise allows?
    Q = float((w * (ok.late - pooled) ** 2).sum())
    dof = max(len(ok) - 1, 1)
    return t, pooled, pooled_se, Q, dof


# ------------------------------------------------------------- C. spec curve
def spec_curve(d):
    rows = []
    for h in BANDWIDTHS:
        for poly in POLYS:
            for cov_name, fe in COVARIATE_SETS:
                for kern in KERNELS:
                    r = fuzzy(d, poly=poly, fe=fe, kernel=kern, h=h)
                    rows.append({"h": h, "poly": poly, "covariates": cov_name,
                                 "kernel": kern, **r})
    t = pd.DataFrame(rows).dropna(subset=["late"])
    return t.sort_values("late").reset_index(drop=True)


# ---------------------------------------------------------------- rendering
def fig_balance(t):
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN, 2.9))
    y = np.arange(len(t))[::-1]
    ax.errorbar(t.jump_sd, y, xerr=1.96 * t.se_sd, fmt="o", ms=4, lw=1.3,
                capsize=2.5, color=fs.BLUE, zorder=3)
    for xi, yi, bad in zip(t.jump_sd, y, t.fails):
        if bad:
            ax.scatter([xi], [yi], s=42, color=fs.VERMILLION, zorder=4)
    ax.axvline(0, color=fs.INK, lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(t.characteristic, fontsize="small")
    ax.set_xlabel("Jump at the cutoff, in SD of the characteristic "
                  "(should be zero)")
    fs.clean(ax, xgrid=True)
    fs.frame(fig, top_in=0.10, bottom_in=0.44, left=0.34, right=0.98)
    fig.savefig(FIG_BALANCE + ".pdf"); fig.savefig(FIG_BALANCE + ".png", dpi=200)
    plt.close(fig)


def fig_per_year(t, pooled, pooled_se):
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN, 2.5))
    ok = t.dropna(subset=["late"])
    y = np.arange(len(ok))[::-1]
    ax.errorbar(ok.late, y, xerr=1.96 * ok.se, fmt="o", ms=5, lw=1.4, capsize=3,
                color=fs.BLUE, zorder=3)
    ax.errorbar([pooled], [-1], xerr=[1.96 * pooled_se], fmt="D", ms=6, lw=1.6,
                capsize=3, color=fs.VERMILLION, zorder=3)
    ax.axvline(0, color=fs.INK, lw=1.0, zorder=2)
    ax.set_yticks(list(y) + [-1])
    ax.set_yticklabels([str(v) for v in ok.year] + ["precision-weighted"],
                       fontsize="small")
    ax.set_xlabel("Venue premium, log points")
    fs.clean(ax, xgrid=True)
    fs.frame(fig, top_in=0.10, bottom_in=0.44, left=0.30, right=0.98)
    fig.savefig(FIG_PERYEAR + ".pdf"); fig.savefig(FIG_PERYEAR + ".png", dpi=200)
    plt.close(fig)


def fig_spec_curve(t):
    """Weak first stages produce arbitrarily large ratios, so the axis is set from
    the F >= 10 specifications and anything outside it is drawn as a clipped
    marker. Letting those few dominate the axis hides the whole curve."""
    fs.apply()
    fig, (ax, axm) = plt.subplots(2, 1, figsize=(fs.TEXT_WIDTH_IN, 4.4),
                                  gridspec_kw={"height_ratios": [2.1, 1.5]},
                                  sharex=True)
    x = np.arange(len(t))
    lo, hi = t.late - 1.96 * t.se, t.late + 1.96 * t.se
    sig = (lo > 0) | (hi < 0)
    strong = t[t.F >= 10]
    pad = 0.6 * max(np.ptp(strong.late) if len(strong) else 1.0, 1.0)
    ylo, yhi = strong.late.min() - pad, strong.late.max() + pad
    ax.vlines(x, lo.clip(ylo, yhi), hi.clip(ylo, yhi), color=fs.GRID, lw=1.4,
              zorder=2)
    inside = t.late.between(ylo, yhi)
    ax.scatter(x[(~sig & inside).values], t.late[(~sig & inside).values], s=9,
               color=fs.MUTED, zorder=3, label="CI includes 0")
    ax.scatter(x[(sig & inside).values], t.late[(sig & inside).values], s=9,
               color=fs.BLUE, zorder=3, label="CI excludes 0")
    ax.scatter(x[(~inside).values], t.late[~inside].clip(ylo, yhi), s=16,
               marker="^", color=fs.VERMILLION, zorder=4,
               label=f"off scale, weak first stage ({int((~inside).sum())})")
    ax.set_ylim(ylo, yhi)
    ax.axhline(0, color=fs.INK, lw=1.0, zorder=4)
    ax.set_ylabel("Venue premium, log points")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    fs.clean(ax)

    # which choice each specification used
    marks = ([(f"h = {h}", t.h == h) for h in BANDWIDTHS]
             + [(f"poly {p}", t.poly == p) for p in POLYS]
             + [(c, t.covariates == c) for c, _ in COVARIATE_SETS]
             + [(k, t.kernel == k) for k in KERNELS])
    for row, (label, mask) in enumerate(marks):
        yy = len(marks) - row - 1
        axm.scatter(x[mask.values], np.full(mask.sum(), yy), s=5,
                    color=fs.BLUE, zorder=3)
    axm.set_yticks(range(len(marks)))
    axm.set_yticklabels([m[0] for m in marks][::-1], fontsize=6.5)
    axm.set_xlabel("Specification, sorted by estimate")
    axm.set_ylim(-0.7, len(marks) - 0.3)
    fs.clean(axm)
    axm.yaxis.grid(False)

    fs.frame(fig, top_in=0.10, bottom_in=0.42, left=0.20, right=0.98, hspace=0.10)
    fig.savefig(FIG_SPECCURVE + ".pdf"); fig.savefig(FIG_SPECCURVE + ".png", dpi=200)
    plt.close(fig)


def build():
    os.makedirs("outputs/figures", exist_ok=True)
    d = load()

    bal = balance(d)
    yr, pooled, pooled_se, Q, dof = per_year(d)
    sc = spec_curve(d)

    bal.to_csv(OUT_BALANCE, index=False)
    yr.to_csv(OUT_PERYEAR, index=False)
    sc.to_csv(OUT_SPECCURVE, index=False)
    fig_balance(bal); fig_per_year(yr, pooled, pooled_se); fig_spec_curve(sc)

    sig = ((sc.late - 1.96 * sc.se > 0) | (sc.late + 1.96 * sc.se < 0)).mean()
    strong = sc[sc.F >= 10]          # conventional weak-instrument screen
    L = ["# Venue premium: balance, per-year estimates, specification curve", "",
         "## A. Balance test", "",
         "The same RD with a pre-determined characteristic as the outcome. A jump "
         "is causally impossible, so a significant one is evidence the design "
         "picks up more than the cutoff.", "",
         "| characteristic | jump | SE | t | n | |", "|---|---:|---:|---:|---:|---|"]
    for r in bal.itertuples():
        L.append(f"| {r.characteristic} | {r.jump:+.3f} | {r.se:.3f} | {r.t:+.2f} | "
                 f"{r.n:,} | {'**FAILS**' if r.fails else 'ok'} |")
    L += ["", f"{int(bal.fails.sum())} of {len(bal)} characteristics show a "
          "significant jump.", "",
          "## B. Per-year premium", "",
          "| year | premium | SE | 95% CI | first-stage F | n |",
          "|---|---:|---:|---|---:|---:|"]
    for r in yr.itertuples():
        if np.isnan(r.late):
            L.append(f"| {r.year} | — | — | not estimable | — | {r.n:,} |")
            continue
        L.append(f"| {r.year} | {r.late:+.3f} | {r.se:.3f} | "
                 f"[{r.late - 1.96 * r.se:+.2f}, {r.late + 1.96 * r.se:+.2f}] | "
                 f"{r.F:.1f} | {r.n:,} |")
    L += [f"| **precision-weighted** | **{pooled:+.3f}** | **{pooled_se:.3f}** | "
          f"**[{pooled - 1.96 * pooled_se:+.2f}, {pooled + 1.96 * pooled_se:+.2f}]** "
          "| | |", "",
          f"Cochran's Q = {Q:.2f} on {dof} df. Q much larger than df means the "
          "years are not estimating a common premium and the weighted average is "
          "a summary rather than a parameter.", "",
          "## C. Specification curve", "",
          f"{len(sc)} specifications: bandwidth x polynomial order x covariates x "
          f"kernel. Estimates run {sc.late.min():+.2f} to {sc.late.max():+.2f}, "
          f"median {sc.late.median():+.2f}. "
          f"{sig:.0%} have a 95% interval excluding zero, and "
          f"{(sc.late > 0).mean():.0%} are positive.", "",
          "The extreme values are weak-first-stage artifacts: the premium divides "
          "by the acceptance jump, so a specification whose first stage is near "
          "zero produces an arbitrarily large ratio. Screening on the conventional "
          f"F >= 10 leaves {len(strong)} of {len(sc)} specifications, running "
          f"{strong.late.min():+.2f} to {strong.late.max():+.2f} with median "
          f"{strong.late.median():+.2f}. That is the range to quote."]
    open(OUT_MD, "w").write("\n".join(L) + "\n")

    print(bal.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print()
    print(yr[["year", "late", "se", "F", "n"]].to_string(
        index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nprecision-weighted {pooled:+.3f} (se {pooled_se:.3f})  "
          f"Q={Q:.2f} on {dof} df")
    print(f"\nspec curve: {len(sc)} specs, {sc.late.min():+.2f} to "
          f"{sc.late.max():+.2f}, {sig:.0%} exclude zero")
    print(f"  F>=10 only: {len(strong)} specs, {strong.late.min():+.2f} to "
          f"{strong.late.max():+.2f}, median {strong.late.median():+.2f}")
    print(f"\n-> {OUT_MD}\n-> {FIG_BALANCE}.pdf\n-> {FIG_PERYEAR}.pdf"
          f"\n-> {FIG_SPECCURVE}.pdf")
    return bal, yr, sc, pooled, pooled_se, Q


def demo():
    bal, yr, sc, pooled, pooled_se, Q = build()

    # Inverse-variance weighting must land inside the range of what it averages,
    # and be at least as precise as the best single year.
    ok = yr.dropna(subset=["late"])
    assert ok.late.min() - 1e-9 <= pooled <= ok.late.max() + 1e-9, \
        f"pooled {pooled} outside the per-year range"
    assert pooled_se <= ok.se.min() + 1e-9, \
        "pooling should not be less precise than the best single year"

    # The spec curve exists to show the spread, so it must actually vary.
    assert sc.late.nunique() > 5, "specification curve is degenerate"
    print(f"\nok — {len(bal)} balance tests ({int(bal.fails.sum())} fail), "
          f"{len(ok)} year estimates pooled to {pooled:+.3f} (se {pooled_se:.3f}), "
          f"{len(sc)} specifications")


if __name__ == "__main__":
    demo()
