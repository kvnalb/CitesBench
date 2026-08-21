"""
The three context figures the RD section needs: the deck's Figures 2, 10 and 11.

src/analysis/rdd_diagnostic.py covers the deck's Figure 1 (running-variable support)
and Figure 3 (first stage by year). src/analysis/venue_premium_rdd.py covers Figure
12 (the two binscatters) and the specification table. This script fills the rest.

  rdd_d  the rating distribution split by decision. The cutoff is visible and soft,
         which is the reason the design has to be fuzzy rather than sharp.
  rdd_e  observability against the running variable. This is the figure that decides
         whether the outcome can be read at all.
  rdd_f  preprint date relative to the decision date, for papers that reached arXiv.

WHY rdd_e MATTERS MOST. The deck could only see citations for arXiv-matched papers,
and arXiv coverage climbs with reviewer rating: on our data 42.0% in the lowest
rating bin against 87.2% in the highest. An outcome observed that selectively is
not an outcome. It is a second decision rule stacked on the first.

Our citation coverage does not have that shape, because S2 title matching reaches
papers that never posted a preprint: 90.8% in the lowest bin against 98.8% in the
highest. The figure plots both series so the gap between them is the argument.

DECISION DATES. OpenReview notification dates, hardcoded because they are three
constants that will never change. A few days either way does not move a
distribution whose quartiles are months wide.

rdd_f IS RESTRICTED TO ARXIV-MATCHED PAPERS. S2's publicationDate is the record's
date, which for an arXiv preprint is the v1 posting and for a journal record is the
issue date. Mixing the two would compare a preprint clock against a publication
clock. So the panel keeps only papers with an arXiv ID and says so on the axis.

Run: python src/analysis/rd_context.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs  # noqa: E402

MASTER = "outputs/paper_master.parquet"

OUT_D = "outputs/figures/rdd_d_rating_by_decision"
OUT_E = "outputs/figures/rdd_e_observability"
OUT_F = "outputs/figures/rdd_f_preprint_timing"
OUT_CSV = "outputs/rd_context.csv"

TITLES = {
    OUT_D: "Distribution of mean review score, by final decision",
    OUT_E: ("Outcome observability along the review-score axis: arXiv record "
            "versus citation count"),
    OUT_F: ("Preprint posting date relative to the decision date, "
            "arXiv-matched papers"),
}

# OpenReview notification dates for each cycle.
DECISION_DATE = {2018: "2018-01-29", 2019: "2018-12-20", 2020: "2019-12-19"}

RATING_BINS = [0, 3, 4, 5, 6, 7, 10]
MIN_BIN = 30          # a bin thinner than this is a point estimate, not a rate


def load():
    d = pd.read_parquet(MASTER)
    d = d[d.year.isin(spec.YEARS)].copy()
    d["accepted"] = d.decision.str.lower().str.contains("accept")
    d["pub_date"] = pd.to_datetime(d.s2_publication_date, errors="coerce")
    dec = d.year.map(DECISION_DATE).pipe(pd.to_datetime)
    d["days_from_decision"] = (d.pub_date - dec).dt.days
    d["has_arxiv"] = d.s2_arxiv_id.notna()
    d["has_cites"] = d[spec.OUTCOME].notna()
    return d


def panel_d(ax, d):
    """Figure 2: the rating distribution by decision. Soft cutoff, hence fuzzy."""
    # bin width 1/3: most papers have three reviews, so the mean lands on thirds.
    # Half-point bins alias against that and produce a sawtooth that looks like a
    # rendering fault rather than the discreteness it is. rdd_a shows the mass
    # points directly.
    edges = np.arange(1, 10.34, 1 / 3)
    for lab, m, colr in [("Rejected", ~d.accepted, fs.VERMILLION),
                         ("Accepted", d.accepted, fs.BLUE)]:
        ax.hist(d.loc[m, "mean_rating"].dropna(), bins=edges, color=colr,
                histtype="step", lw=1.8, label=lab, zorder=3)
    ax.set_xlabel("Mean review score")
    ax.set_ylabel("Submissions")
    ax.legend(frameon=False, loc="upper right")


def observability(d):
    d = d.assign(bin=pd.cut(d.mean_rating, RATING_BINS))
    g = d.groupby("bin", observed=True).agg(
        n=("paper_id", "size"),
        arxiv=("has_arxiv", "mean"),
        cites=("has_cites", "mean"),
        mid=("mean_rating", "mean"))
    return g[g.n >= MIN_BIN]


def panel_e(ax, g):
    """Figure 10, with the series the deck could not draw.

    arXiv coverage is steeply selected on rating. Citation coverage is nearly flat,
    because S2 title matching does not need a preprint. The vertical distance
    between the two lines is what changed between the deck and this paper.
    """
    ax.plot(g.mid, g.arxiv, marker="o", ms=5, lw=1.8, color=fs.VERMILLION,
            label="Has an arXiv record", zorder=3)
    ax.plot(g.mid, g.cites, marker="s", ms=5, lw=1.8, color=fs.BLUE,
            label="Has a citation count", zorder=3)
    ax.set_xlabel("Mean review score")
    ax.set_ylabel("Share of submissions")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower right")


def panel_f(ax, d):
    """Figure 11: preprint posting relative to the decision, arXiv-matched only."""
    s = d[d.has_arxiv & d.days_from_decision.notna()]
    edges = np.arange(-730, 731, 45)
    # step outlines, not filled bars: two overlaid fills hide each other exactly
    # where the distributions differ
    for lab, m, colr in [("Rejected", ~s.accepted, fs.VERMILLION),
                         ("Accepted", s.accepted, fs.BLUE)]:
        v = s.loc[m, "days_from_decision"].clip(-730, 730)
        med = int(s.loc[m, "days_from_decision"].median())
        ax.hist(v, bins=edges, color=colr, density=True, histtype="step",
                lw=1.8, zorder=3,
                label=f"{lab} (median {med:+d}d)".replace("-", "\u2212"))
    ax.axvline(0, color=fs.INK, ls=(0, (3, 2)), lw=1.2, zorder=4)
    ax.set_xlabel("Days from decision to preprint (arXiv-matched papers)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")
    return s


def build():
    os.makedirs("outputs/figures", exist_ok=True)
    d = load()
    g = observability(d)
    g.to_csv(OUT_CSV)

    for out, draw, height in [(OUT_D, lambda ax: panel_d(ax, d), 2.3),
                              (OUT_E, lambda ax: panel_e(ax, g), 2.5),
                              (OUT_F, lambda ax: panel_f(ax, d), 2.4)]:
        fs.apply()
        fig, ax = plt.subplots(figsize=(5.5, height))
        draw(ax)
        fs.clean(ax)
        fs.frame(fig, top_in=0.10, bottom_in=0.44, left=0.13, right=0.99)
        fs.add_title(fig, TITLES[out])
        fig.savefig(out + ".pdf")
        fig.savefig(out + ".png", dpi=200)
        plt.close(fig)

    print(g.to_string(float_format=lambda v: f"{v:.3f}"))
    s = d[d.has_arxiv & d.days_from_decision.notna()]
    print(f"\npreprint timing, arXiv-matched n={len(s):,}: "
          f"accepted median {s[s.accepted].days_from_decision.median():+.0f}d, "
          f"rejected {s[~s.accepted].days_from_decision.median():+.0f}d")
    print(f"posted before the decision: "
          f"accepted {(s[s.accepted].days_from_decision < 0).mean():.1%}, "
          f"rejected {(s[~s.accepted].days_from_decision < 0).mean():.1%}")
    print(f"\n-> {OUT_D}.pdf\n-> {OUT_E}.pdf\n-> {OUT_F}.pdf\n-> {OUT_CSV}")
    return d, g


def demo():
    d, g = build()

    # The deck's problem: arXiv coverage climbs with rating. If this ever stops
    # being true the figure has lost its reason to exist.
    assert g.arxiv.iloc[-1] - g.arxiv.iloc[0] > 0.30, \
        f"arXiv coverage gradient only {g.arxiv.iloc[-1] - g.arxiv.iloc[0]:.3f}"

    # Our fix: the citation series must be both higher and flatter everywhere.
    assert (g.cites > g.arxiv).all(), "citation coverage should beat arXiv coverage"
    assert (g.cites.iloc[-1] - g.cites.iloc[0]) < (g.arxiv.iloc[-1] - g.arxiv.iloc[0]) / 3, \
        "citation coverage should be far flatter than arXiv coverage"

    # Decision dates must land inside the posting distribution, not outside it.
    s = d[d.has_arxiv & d.days_from_decision.notna()]
    frac = (s.days_from_decision < 0).mean()
    assert 0.05 < frac < 0.95, \
        f"{frac:.1%} posted pre-decision — check DECISION_DATE"

    print(f"\nok — arXiv coverage {g.arxiv.iloc[0]:.1%} to {g.arxiv.iloc[-1]:.1%} "
          f"across rating bins, citations {g.cites.iloc[0]:.1%} to "
          f"{g.cites.iloc[-1]:.1%}; {frac:.1%} of arXiv-matched papers posted "
          "before the decision")


if __name__ == "__main__":
    demo()
