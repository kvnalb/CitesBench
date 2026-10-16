"""
Figure 4: Table 2's contrast, drawn.

Table 2 regresses log(1 + citations) on a selected dummy with year fixed effects,
so its regressor is binary and there is no continuous axis to scatter the raw data
against. What there IS to plot is the bootstrap: each of the N draws resamples
papers within year, re-runs every regime's selection on the resampled pool, and
refits. One draw therefore gives one coefficient per regime, and those coefficients
are paired — they come from the same resampled pool.

Plotting the pair is the scatter this table wants. Each point is one draw: the
area chairs' coefficient on x, an LLM regime's on y. The 45-degree line is
"identical performance on this draw". A point below the line is a draw where the
area chairs did better. The share of points below the line IS Table 2's
`frac_le_0` column, so the figure and the table cannot disagree.

This is more informative than a dot-and-whisker of the two marginal intervals,
which is the other obvious choice: the marginals overlap heavily, but the draws
are positively correlated because a resample that happens to contain more
high-citation papers lifts every regime at once. The paired view removes that
common component, which is exactly what the contrast row does arithmetically.

Reads the cached draws written by table2_regression.py rather than recomputing
them, and refuses to run if that cache was built for different inputs.

Run: python src/figures/fig4_contrast_scatter.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs   # noqa: E402
import table2_regression as t2             # noqa: E402

DRAWS_CSV = t2.DRAWS_CSV
TABLE2_CSV = t2.OUT_CSV
OUT_PDF = "outputs/figures/fig4_contrast_scatter.pdf"
TITLE = "Paired bootstrap draws: council and single call against the area chairs"
OUT_PNG = "outputs/figures/fig4_contrast_scatter.png"
OUT_CSV = "outputs/figures/fig4_contrast_scatter.csv"

# y-axis regimes, each plotted against the area chairs on x
AGAINST = [spec.BY_KEY["llm_council"], spec.BY_KEY["llm_single"]]
# llm_multi is in HEADLINE but omitted here: two square panels fit the
# text width, three do not, and the ladder's ends are the comparison.
BASE = spec.BY_KEY["human_ac"]


def load():
    if not os.path.exists(DRAWS_CSV):
        raise SystemExit(f"{DRAWS_CSV} missing — run "
                         "python src/figures/table2_regression.py first")
    d = pd.read_csv(DRAWS_CSV)
    fp = t2.fingerprint()
    if d.get("fingerprint", pd.Series([None])).iloc[0] != fp:
        raise SystemExit(
            f"{DRAWS_CSV} was built for different inputs — rerun "
            "python src/figures/table2_regression.py --refresh")
    return d


def build():
    os.makedirs("outputs/figures", exist_ok=True)
    d = load()

    fs.apply(ncols=2)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75))

    recs = []
    for ax, r in zip(axes, AGAINST):
        x, y = d[BASE.key].to_numpy(), d[r.key].to_numpy()
        below = float((y - x <= 0).mean())

        # Each panel gets its own square window around its own cloud. Sharing one
        # scale across both panels put the council's draws in a corner, because the
        # single call sits half a log point lower and stretched the range. The
        # reference here is the diagonal, not a common axis, so per-panel limits
        # cost nothing and the spread stays legible.
        pad = 0.10 * max(np.ptp(x), np.ptp(y))   # ndarray.ptp went in NumPy 2.0
        lo = min(x.min(), y.min()) - pad
        hi = max(x.max(), y.max()) + pad

        # 45 degrees is equal performance on that draw, so it is the reference the
        # eye should measure against — drawn under the points, not over them.
        ax.plot([lo, hi], [lo, hi], color=fs.MUTED, ls=(0, (4, 3)), lw=1.0, zorder=2)
        ax.scatter(x, y, s=7, color=r.color, alpha=0.55, linewidths=0, zorder=3)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(BASE.label)
        ax.set_ylabel(r.label)
        fs.clean(ax, xgrid=True)

        recs.append({"regime": r.label, "key": r.key, "n_draws": len(d),
                     "mean_diff": float((y - x).mean()),
                     "frac_at_or_below_45": below,
                     "corr_with_base": float(np.corrcoef(x, y)[0, 1])})

    res = pd.DataFrame(recs)
    res.to_csv(OUT_CSV, index=False)

    fs.frame(fig, top_in=0.10, bottom_in=0.45, left=0.11, right=0.99, wspace=0.28)
    fs.add_title(fig, TITLE)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {OUT_PDF}\n-> {OUT_PNG}\n-> {OUT_CSV}")
    return res


def demo():
    res = build().set_index("key")

    # The whole point of the paired view: the share of points below the diagonal
    # must equal Table 2's frac_le_0, or the figure is telling a different story
    # than the table it accompanies.
    t = pd.read_csv(TABLE2_CSV).set_index("key")
    got = res.loc["llm_council", "frac_at_or_below_45"]
    want = float(t.loc["contrast", "frac_le_0"])
    assert abs(got - want) < 1e-9, (
        f"scatter says {got:.3f} of draws at or below the diagonal, Table 2's "
        f"contrast row says {want:.3f}")

    # Draws are paired by construction, so a resample that lifts one regime lifts
    # the others. If that correlation vanished the pairing would be pointless and
    # a plain dot-and-whisker would say the same thing.
    assert (res["corr_with_base"] > 0.3).all(), \
        f"draws should be positively correlated, got {res['corr_with_base'].to_dict()}"

    print(f"\nok — council at or below the diagonal on {got:.0%} of draws, "
          f"matching Table 2; draw correlation with the area chairs "
          + ", ".join(f"{k} {v:.2f}" for k, v in res["corr_with_base"].items()))


if __name__ == "__main__":
    demo()
