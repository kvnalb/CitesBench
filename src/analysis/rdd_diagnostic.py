"""
Whether a regression discontinuity can identify the ICLR venue premium.

Three panels, each testing one requirement of the design:

  A  support      is there data arbitrarily close to the threshold
  B  first stage  does the probability of acceptance JUMP at the threshold
  C  stability    does the estimate settle down as the bandwidth shrinks

Runs on the canonical outcome (outputs/citations.csv via eval_table), unlike
src/analysis/fuzzy_rdd.py, which still reads the old OpenAlex pull with its
26.3 pp accept/reject coverage differential.

The cutoff is taken from data/OpenAlex/openalex_rdd_arxiv_paper_level.csv, which
is where the archived R pipeline left it. Worth stating plainly: no script in
this repo computes that value, so it was reverse-engineered from the very
decisions the design proposes to instrument.

Run: python src/analysis/rdd_diagnostic.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs   # noqa: E402

RDD_CSV = "data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"
OUT_DIR = "outputs/figures"
OUT_A = "outputs/figures/rdd_a_support"
OUT_B = "outputs/figures/rdd_b_first_stage"
OUT_C = "outputs/figures/rdd_c_stability"
OUT_CSV = "outputs/rdd_diagnostic.csv"

TITLES = {
    OUT_A: "Support of the running variable: mass points in mean review score",
    OUT_B: "First stage: acceptance probability along the review-score axis, by year",
    OUT_C: "Acceptance premium across bandwidths",
}

BANDWIDTHS = [0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50]
MIN_MASS = 5          # mass points thinner than this are noise, not structure


def load():
    """Pool with the running variable, centred on the imputed year cutoff."""
    cut = (pd.read_csv(RDD_CSV, low_memory=False)
             .groupby("year")["cutoff"].first().to_dict())
    et = spec.read_eval_table()
    et = et[et.mean_rating.notna()].copy()
    et["cutoff"] = et.year.map(cut)
    et["z"] = (et.mean_rating - et.cutoff).round(4)      # running variable
    et["above"] = (et.z >= 0).astype(float)              # the instrument
    et["d"] = et.accepted.astype(float)                  # the treatment
    et["y"] = np.log1p(et[spec.OUTCOME])                 # the outcome
    return et.dropna(subset=["z", "cutoff"])


def wald(d, h):
    """Fuzzy-RD Wald estimate at bandwidth h: local linear, triangular kernel,
    slopes allowed to differ either side, HC1 standard errors, delta-method SE
    on the ratio. Returns (first-stage jump, F, LATE, SE, n)."""
    s = d[(d.z.abs() <= h) & d.y.notna()].copy()
    if len(s) < 30 or s.above.nunique() < 2:
        return (np.nan,) * 4 + (len(s),)
    w = 1 - (s.z.abs() / h)                              # triangular
    X = np.column_stack([np.ones(len(s)), s.above, s.z, s.above * s.z])
    W = w.to_numpy()

    def fit(yv):
        Xw = X * W[:, None]
        XtX = X.T @ Xw
        if np.linalg.cond(XtX) > 1e12:
            return None
        b = np.linalg.solve(XtX, Xw.T @ yv)
        e = yv - X @ b
        meat = (X * (W * e)[:, None]).T @ (X * (W * e)[:, None])
        inv = np.linalg.inv(XtX)
        V = inv @ meat @ inv * len(s) / max(len(s) - X.shape[1], 1)
        return b, np.sqrt(np.diag(V))

    fs_fit = fit(s.d.to_numpy())
    rf_fit = fit(s.y.to_numpy())
    if fs_fit is None or rf_fit is None:
        return (np.nan,) * 4 + (len(s),)
    (bf, sef), (br, ser) = fs_fit, rf_fit
    jump, jump_se = bf[1], sef[1]
    rf, rf_se = br[1], ser[1]
    F = (jump / jump_se) ** 2 if jump_se > 0 else np.nan
    late = rf / jump if jump != 0 else np.nan
    # delta method on a ratio, ignoring the covariance term (optimistic)
    se = (abs(late) * np.sqrt((rf_se / rf) ** 2 + (jump_se / jump) ** 2)
          if rf != 0 and jump != 0 else np.nan)
    return jump, F, late, se, len(s)


# ------------------------------------------------------------------ panels
def panel_a(ax, d):
    """Where the data actually sits. Bars are mass points of the running variable."""
    for i, year in enumerate(spec.YEARS):
        s = d[d.year == year]
        m = s.groupby("z").size()
        m = m[(np.abs(m.index.values) <= 1.4) & (m.values >= MIN_MASS)]
        ax.scatter(m.index, [i] * len(m), s=m.values * 1.4,
                   color=fs.BLUE, alpha=0.65, linewidths=0, zorder=3)
        for z, n in m.items():
            # only the substantial clumps get labelled: the thin ones collide with
            # their neighbours and carry no argument
            if abs(z) <= 0.6 and n >= 20:
                # nudged off the cutoff rule, which otherwise splits the label
                ax.annotate(f"{n}", (z, i), xytext=(0, 10 if z else 13), fontsize=6.5,
                            textcoords="offset points", ha="center", color=fs.INK,
                            zorder=6)
    ax.axvline(0, color=fs.VERMILLION, lw=1.2, zorder=2)
    ax.set_yticks(range(len(spec.YEARS)))
    ax.set_yticklabels(spec.YEARS)
    ax.set_xlabel("Mean review score minus the imputed cutoff")
    ax.set_ylim(-0.6, len(spec.YEARS) - 0.4)


def panel_b(ax, d):
    """The first stage. A usable design needs a step here, not a ramp."""
    for year, colr in zip(spec.YEARS, [fs.BLUE, fs.VERMILLION, fs.BLUISHGREEN]):
        s = d[d.year == year]
        g = s.groupby("z").agg(n=("d", "size"), p=("d", "mean"))
        g = g[(np.abs(g.index.values) <= 1.4) & (g.n.values >= MIN_MASS)]
        ax.plot(g.index, g.p, marker="o", ms=4, lw=1.6, color=colr, label=str(year))
    ax.axvline(0, color=fs.MUTED, ls=(0, (3, 2)), lw=1.1, zorder=2)
    ax.set_xlabel("Mean review score minus the imputed cutoff")
    ax.set_ylabel("P(accepted)")
    ax.set_ylim(-0.04, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, ncol=3, loc="upper left")


def panel_c(ax, rows):
    """The estimate against bandwidth. Convergence is the thing to look for."""
    r = rows[rows.late.notna()]
    ax.axhline(0, color=fs.MUTED, lw=1.0, zorder=2)
    ax.errorbar(r.h, r.late, yerr=1.96 * r.se, fmt="o", ms=5, lw=1.4,
                capsize=3, color=fs.BLUE, zorder=3)
    for _, x in r.iterrows():
        ax.annotate(f"F={x.F:.0f}", (x.h, x.late), xytext=(0, 11), fontsize=6.5,
                    textcoords="offset points", ha="center", color=fs.MUTED)
    ax.set_xlabel("Bandwidth h (rating points either side of the cutoff)")
    ax.set_ylabel("Fuzzy-RD estimate, log points")


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = load()

    rows = pd.DataFrame(
        [dict(zip(["jump", "F", "late", "se", "n"], wald(d, h)), h=h)
         for h in BANDWIDTHS])
    rows.to_csv(OUT_CSV, index=False)

    for out, draw, height in [(OUT_A, lambda ax: panel_a(ax, d), 2.0),
                              (OUT_B, lambda ax: panel_b(ax, d), 2.6),
                              (OUT_C, lambda ax: panel_c(ax, rows), 2.6)]:
        fs.apply()
        fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN, height))
        draw(ax)
        fs.clean(ax, xgrid=True)
        fs.frame(fig, top_in=0.14, bottom_in=0.48, left=0.13, right=0.98)
        fs.add_title(fig, TITLES[out])
        fig.savefig(out + ".pdf")
        fig.savefig(out + ".png", dpi=200)
        plt.close(fig)

    print(rows.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    for year in spec.YEARS:
        s = d[d.year == year]
        print(f"{year}: {int((s.z.abs() < 0.10).sum()):>4} papers within 0.10 of the "
              f"cutoff, {int((s.z.abs() < 0.25).sum()):>4} within 0.25, "
              f"{s.z.nunique()} distinct values of the running variable")
    print(f"\n-> {OUT_A}.pdf\n-> {OUT_B}.pdf\n-> {OUT_C}.pdf\n-> {OUT_CSV}")
    return rows


def demo():
    rows = build().dropna(subset=["late"])

    # A first stage DOES exist — an earlier version of this file asserted otherwise
    # and the data refused. The jump is 0.18 to 0.26 and F reaches 47 at the widest
    # bandwidth. The design still fails, on three sharper grounds:

    # 1. It is weak exactly where the design needs it, at small h.
    tight = rows[rows.h <= 0.50]
    assert (tight.F < 10).all(), \
        f"first stage is not weak at small h: {tight[['h','F']].to_dict('records')}"

    # 2. F RISES with bandwidth, which is the signature of a step function absorbing
    #    a slope rather than detecting a discontinuity. A real jump does not get
    #    easier to see as you look further from it.
    assert rows.F.is_monotonic_increasing, "F should rise with h under misspecification"

    # 3. No bandwidth yields an informative interval. This is the claim that matters:
    #    zero and a sevenfold citation premium are both inside every CI.
    rows["lo"], rows["hi"] = rows.late - 1.96 * rows.se, rows.late + 1.96 * rows.se
    assert ((rows.lo < 0) & (rows.hi > 0)).all(), "some CI excluded zero"

    # 4. The estimate does not converge as h shrinks — it drifts with h, so the
    #    number reported would be a choice of bandwidth, not a measurement.
    span = rows.late.max() - rows.late.min()
    assert span > 0.5, f"estimate is more stable in h than claimed (span {span:.2f})"

    print(f"\nok — first stage F {rows.F.min():.1f} to {rows.F.max():.1f} rising with h; "
          f"every CI contains zero; point estimate spans {span:.2f} log points "
          f"across bandwidths ({rows.late.min():+.2f} to {rows.late.max():+.2f})")


if __name__ == "__main__":
    demo()
