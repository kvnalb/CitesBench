"""
Figure 1: how the evaluation works.

A schematic, not a plot — but every number on it is read from
outputs/figures/table1_sample.csv rather than typed in, so the methodology figure
cannot drift away from the sample it describes. Build Table 1 first.

The one design decision worth a figure is the middle panel: every regime returns
EXACTLY n paper ids, with n pinned to that year's real accept count. Without that,
a regime could look good by being more permissive, and "which selector is better"
would collapse into "which selector accepted more". Pinning n makes the regimes
comparable to each other and to what the conference actually did, and it is why the
area chairs can appear as a regime at all rather than as a separate baseline.

Drawn with plain matplotlib patches. No graphviz, no mermaid, no new dependency.

Run: python src/figures/table1_sample.py && python src/figures/fig1_design.py
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs

TABLE1 = spec.TABLE1_CSV
OUT_PDF = "outputs/figures/fig1_design.pdf"
OUT_PNG = "outputs/figures/fig1_design.png"

BOX_FC = "#f7f7f5"


def facts():
    t = spec.read_table1()
    a = t[t.year == "all"].iloc[0]
    per = t[t.year != "all"]
    return a, per


def box(ax, x, y, w, h, title, lines, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.4, edgecolor=edge, facecolor=BOX_FC, zorder=2))
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top", zorder=3,
            fontsize="medium", fontweight="bold", color=edge)
    ax.text(x + w / 2, y + h - 0.135, "\n".join(lines), ha="center", va="top",
            zorder=3, fontsize="small", color=fs.INK, linespacing=1.65)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=17,
                                 linewidth=1.4, color=fs.MUTED, zorder=4))


def build():
    os.makedirs("outputs/figures", exist_ok=True)
    a, per = facts()
    yrs = "   ".join(f"{int(r.year)}: {r.submissions:,} / {r.accepts:,}"
                     for r in per.itertuples())

    # One hue for all three boxes: these are pipeline steps, not regimes, and the
    # categorical slots mean "area chairs / council / single call" everywhere else
    # in the paper. The step numbers carry the sequence.
    #
    # No headline, deck or source line anywhere in this module. Every figure here
    # emits marks and axis labels only; the caption is written in the LaTeX
    # document, by the author, from the CSV each script writes beside its PDF.
    fs.apply()
    fig, ax = plt.subplots(figsize=(5.5, 1.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    w, h, y = 0.28, 0.78, 0.08
    box(ax, 0.03, y, w, h, "1.  The pool", [
        "ICLR 2018-2020, every submission",
        f"{a.submissions:,} papers",
        f"{a.accepts:,} accepted   {a.submissions - a.accepts:,} rejected",
    ], fs.BLUE)

    box(ax, 0.36, y, w, h, "2.  Each regime selects n", [
        "select(pool, n) -> exactly n ids",
        "",
        "n = that year's accept count",
        yrs.replace("   ", "\n"),
    ], fs.BLUE)

    box(ax, 0.69, y, w, h, "3.  Score against outcome", [
        "Semantic Scholar citations",
        f"tier A+B, {a.cite_coverage:.1%} of the pool",
        f"accepted {a.coverage_accepted:.1%} / rejected {a.coverage_rejected:.1%}",
        f"differential {a.coverage_differential_pp:.1f} pp",
        "",
        "median, mean log(1+c), recall@k",
    ], fs.BLUE)

    arrow(ax, 0.317, 0.353, y + h / 2)
    arrow(ax, 0.647, 0.683, y + h / 2)

    fs.frame(fig, top_in=0.06, bottom_in=0.06, left=0.0, right=1.0)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"-> {OUT_PDF}\n-> {OUT_PNG}")
    return a


def demo():
    a = build()
    # the figure states the sample; if it ever disagrees with Table 1 it is wrong
    assert a.submissions == 4567 and a.accepts == sum(spec.N_PINNED.values())
    assert a.coverage_differential_pp < 10
    print(f"ok — schematic drawn from Table 1 "
          f"({a.submissions:,} papers, {a.accepts:,} accepts, "
          f"{a.cite_coverage:.1%} outcome coverage)")


if __name__ == "__main__":
    demo()
