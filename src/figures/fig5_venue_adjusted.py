"""
Figure 5: the regime comparison after netting out the venue premium.

WHY THIS EXHIBIT EXISTS. Citations are realised after the decision, and acceptance
itself raises them. Write

    Y_i = Y_i(0) + D_i * tau

where D_i is historical ICLR acceptance and tau is the venue premium. The area
chairs' admitted set is by definition {i : D_i = 1}, so they collect the premium on
every paper they pick. An LLM regime collects it only on the papers it agrees with
the humans about. The raw comparison is therefore biased TOWARD the humans, and the
size of the bias depends on tau.

WHY IT IS A CURVE AND NOT A NUMBER. src/analysis/venue_premium_rdd.py estimates
tau, and src/analysis/venue_premium_robustness.py shows that estimate is not tight:
the per-year precision-weighted figure is +1.24 with a 95% interval of [-0.05,
2.53], and screening the specification curve at F >= 10 leaves a range of +0.07 to
+1.61. A single adjusted number would hide all of that. So this figure sweeps tau
across the plausible range and shows what each regime's metric does, with the
estimate and its interval marked on the axis.

The reading it supports is a breakdown value: the tau at which a conclusion flips,
compared against the range the data admits.

HOW THE ADJUSTMENT WORKS. To recover Y(0) we remove the premium from papers the
humans accepted, in the units the premium is estimated in (log points):

    log1p(c_adj) = log1p(c) - tau   for accepted papers, unchanged for rejected

then convert back to counts so the median and recall metrics stay on their own
scales. Rejected papers are untouched, which keeps the counterfactual on the side
of the data that was never treated.

Run: python src/figures/fig5_venue_adjusted.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs   # noqa: E402

PREMIUM_CSV = "outputs/venue_premium_by_year.csv"
SPECCURVE_CSV = "outputs/venue_premium_speccurve.csv"
OUT_PDF = "outputs/figures/fig5_venue_adjusted.pdf"
OUT_PNG = "outputs/figures/fig5_venue_adjusted.png"
OUT_CSV = "outputs/figures/fig5_venue_adjusted.csv"

TAUS = np.round(np.arange(0.0, 2.01, 0.20), 2)
METRICS = [("median_citations", "Median citations"),
           ("mean_log_citations", "Mean log(1 + citations)")]
# Fewer orderings than Figure 2: this sweeps 11 values of tau, and the tie interval
# is not what the figure is about.
N_SHUFFLE = 60


def premium_range():
    """The estimate to mark on the axis, and the range to shade.

    Point and interval come from the per-year precision-weighted estimate, which
    is the defensible one: pooling years shares one slope across three different
    cutoffs and a balance test rejects it. The shaded band is the F >= 10 subset of
    the specification curve, which is the honest spread across implementations.
    """
    out = {"point": np.nan, "lo": np.nan, "hi": np.nan,
           "band_lo": np.nan, "band_hi": np.nan}
    if os.path.exists(PREMIUM_CSV):
        y = pd.read_csv(PREMIUM_CSV).dropna(subset=["late", "se"])
        y = y[y.se > 0]
        if len(y):
            w = 1 / y.se ** 2
            p = float((y.late * w).sum() / w.sum())
            se = float(np.sqrt(1 / w.sum()))
            out.update(point=p, lo=p - 1.96 * se, hi=p + 1.96 * se)
    if os.path.exists(SPECCURVE_CSV):
        sc = pd.read_csv(SPECCURVE_CSV)
        strong = sc[sc.F >= 10]
        if len(strong):
            out.update(band_lo=float(strong.late.min()),
                       band_hi=float(strong.late.max()))
    return out


def adjust(et, tau):
    """Remove `tau` log points from accepted papers' citations, in place on a copy."""
    d = et.copy()
    acc = d.accepted & d[spec.OUTCOME].notna()
    d.loc[acc, spec.OUTCOME] = np.maximum(
        np.expm1(np.log1p(d.loc[acc, spec.OUTCOME]) - tau), 0.0)
    return d


def build():
    os.makedirs("outputs/figures", exist_ok=True)
    et = spec.read_eval_table()
    pr = premium_range()

    rows = []
    for tau in TAUS:
        d = adjust(et, tau)
        for metric, _ in METRICS:
            for r in spec.HEADLINE:
                v = spec.metric_over_orderings(d, r, metric,
                                               n_shuffle=N_SHUFFLE)[0]
                rows.append({"tau": tau, "metric": metric, "regime": r.label,
                             "key": r.key, "value": v})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    fs.apply(ncols=2)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharex=True)
    for ax, (metric, unit) in zip(axes, METRICS):
        sub = res[res.metric == metric]
        if not np.isnan(pr["band_lo"]):
            ax.axvspan(pr["band_lo"], pr["band_hi"], color=fs.GRID, zorder=1)
        if not np.isnan(pr["point"]):
            ax.axvline(pr["point"], color=fs.MUTED, ls=(0, (3, 2)), lw=1.1, zorder=2)
        for r in spec.HEADLINE:
            s = sub[sub.key == r.key]
            ax.plot(s.tau, s.value, lw=1.8, color=r.color, zorder=3,
                    label=r.label.split(" (")[0])
        ax.set_xlabel(r"Venue premium $\tau$ removed, log points")
        ax.set_ylabel(unit)
        fs.clean(ax, xgrid=True)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")

    fs.frame(fig, top_in=0.10, bottom_in=0.46, left=0.11, right=0.99, wspace=0.34)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    # crossings: the tau at which each LLM regime overtakes the area chairs
    print(f"premium: point {pr['point']:+.2f} "
          f"[{pr['lo']:+.2f}, {pr['hi']:+.2f}], "
          f"F>=10 band [{pr['band_lo']:+.2f}, {pr['band_hi']:+.2f}]\n")
    cross = []
    for metric, _ in METRICS:
        ac = res[(res.metric == metric) & (res.key == "human_ac")].set_index("tau").value
        for r in spec.HEADLINE[1:]:
            s = res[(res.metric == metric) & (res.key == r.key)].set_index("tau").value
            ahead = s > ac
            first = next((t for t in TAUS if ahead.loc[t]), None)
            cross.append({"metric": metric, "regime": r.label,
                          "tau_overtakes_AC": first,
                          "ahead_at_tau_0": bool(ahead.loc[0.0])})
    c = pd.DataFrame(cross)
    print(c.to_string(index=False))
    print(f"\n-> {OUT_PDF}\n-> {OUT_PNG}\n-> {OUT_CSV}")
    return res, pr, c


def demo():
    res, pr, c = build()

    # Removing the premium can only lower the area chairs' measured outcome, since
    # every paper they picked is one the premium is being taken off.
    ac = res[(res.metric == "mean_log_citations") & (res.key == "human_ac")]
    ac = ac.sort_values("tau").value.to_numpy()
    assert ac[-1] < ac[0], "the AC curve must fall as tau rises"
    assert np.all(np.diff(ac) <= 1e-9), "the AC curve must be monotone in tau"

    # An LLM regime that never overlapped the human slate would be flat. All three
    # overlap heavily, so all three should move, just less than the ACs do.
    for r in spec.HEADLINE[1:]:
        s = res[(res.metric == "mean_log_citations") & (res.key == r.key)]
        s = s.sort_values("tau").value.to_numpy()
        drop_llm, drop_ac = s[0] - s[-1], ac[0] - ac[-1]
        assert 0 <= drop_llm < drop_ac, \
            f"{r.key} should fall, but by less than the ACs: {drop_llm} vs {drop_ac}"

    assert not np.isnan(pr["point"]), "premium estimate missing — run the RD scripts"
    print(f"\nok — {len(TAUS)} values of tau, AC mean-log falls "
          f"{ac[0]:.3f} -> {ac[-1]:.3f}; premium estimate {pr['point']:+.2f}")


if __name__ == "__main__":
    demo()
