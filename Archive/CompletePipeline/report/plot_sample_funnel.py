#!/usr/bin/env python3
"""
Sample selection funnel for ICLR 2018-2020 RDD sample.

Counts (from rdd_sample_year_specific_bandwidth_with_openalex_citations.csv):
                    Accepted   Rejected
  In RDD sample        1170       1221
  arXiv matched         964        615
  OpenAlex citations    946        566

Per year:
  2018   total=512   acc=219  rej=293
  2019   total=896   acc=414  rej=482
  2020   total=983   acc=537  rej=446

Output: plots/sample_funnel.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

# Colours (match theme.scss)
ACCENT = "#1f3a5f"
ACC_FACE = "#d1fae5"
ACC_BORDER = "#059669"
REJ_FACE = "#fee2e2"
REJ_BORDER = "#b91c1c"
TOP_FACE = "#dbeafe"
TOP_BORDER = "#1f3a5f"
GREY = "#6b7280"


def box(ax, x, y, w, h, text, face, border, *, fontsize=11, fontweight="bold"):
    patch = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=face,
        edgecolor=border,
        linewidth=1.3,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color="#111827",
        linespacing=1.3,
        zorder=4,
    )


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.3):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw, mutation_scale=14, alpha=0.85
        ),
    )


def render() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        6.0,
        9.55,
        "Sample Selection Funnel  ·  ICLR 2018–2020, year-specific RDD bandwidth",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    ax.plot([3.2, 8.8], [9.28, 9.28], color=ACCENT, lw=1.0, alpha=0.45)

    # Top box
    top_y = 8.3
    box(
        ax,
        6.0,
        top_y,
        4.5,
        0.85,
        "Total papers in RDD sample\n n = 2,391",
        TOP_FACE,
        TOP_BORDER,
        fontsize=12,
    )

    # Split to accepted / rejected
    acc_x, rej_x = 3.1, 8.9
    split_y = 6.8
    box(
        ax,
        acc_x,
        split_y,
        3.6,
        0.85,
        "Accepted\n n = 1,170  (48.9%)",
        ACC_FACE,
        ACC_BORDER,
        fontsize=12,
    )
    box(
        ax,
        rej_x,
        split_y,
        3.6,
        0.85,
        "Rejected\n n = 1,221  (51.1%)",
        REJ_FACE,
        REJ_BORDER,
        fontsize=12,
    )

    arrow(ax, 5.0, top_y - 0.45, acc_x + 0.7, split_y + 0.45)
    arrow(ax, 7.0, top_y - 0.45, rej_x - 0.7, split_y + 0.45)

    # arXiv row
    arxiv_y = 5.3
    box(
        ax,
        acc_x,
        arxiv_y,
        3.6,
        0.85,
        "arXiv matched\n 964   (82.4% of accepted)",
        ACC_FACE,
        ACC_BORDER,
        fontsize=11,
    )
    box(
        ax,
        rej_x,
        arxiv_y,
        3.6,
        0.85,
        "arXiv matched\n 615   (50.4% of rejected)",
        REJ_FACE,
        REJ_BORDER,
        fontsize=11,
    )

    arrow(ax, acc_x, split_y - 0.43, acc_x, arxiv_y + 0.43)
    arrow(ax, rej_x, split_y - 0.43, rej_x, arxiv_y + 0.43)

    # OpenAlex row
    oa_y = 3.8
    box(
        ax,
        acc_x,
        oa_y,
        3.6,
        0.85,
        "OpenAlex citations\n 946   (80.9% of accepted)",
        ACC_FACE,
        ACC_BORDER,
        fontsize=11,
    )
    box(
        ax,
        rej_x,
        oa_y,
        3.6,
        0.85,
        "OpenAlex citations\n 566   (46.4% of rejected)",
        REJ_FACE,
        REJ_BORDER,
        fontsize=11,
    )

    arrow(ax, acc_x, arxiv_y - 0.43, acc_x, oa_y + 0.43)
    arrow(ax, rej_x, arxiv_y - 0.43, rej_x, oa_y + 0.43)

    # Per-year breakdown strip
    year_rows = [
        ("2018", 512, 219, 293),
        ("2019", 896, 414, 482),
        ("2020", 983, 537, 446),
    ]
    strip_y = 2.25
    ax.text(
        1.4,
        strip_y + 0.55,
        "Per-year totals",
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="600",
        color="#374151",
    )
    col_x = [2.3, 4.8, 7.3, 9.8]
    headers = ["Year", "Total", "Accepted", "Rejected"]
    for x, h in zip(col_x, headers):
        ax.text(
            x,
            strip_y,
            h,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="600",
            color="#111827",
        )
    ax.plot([1.7, 10.4], [strip_y - 0.22, strip_y - 0.22], color="#d1d5db", lw=1)
    for i, (yr, tot, acc, rej) in enumerate(year_rows):
        y = strip_y - 0.5 - i * 0.35
        ax.text(col_x[0], y, yr, ha="center", va="center", fontsize=10, color="#374151")
        ax.text(col_x[1], y, f"{tot:,}", ha="center", va="center", fontsize=10, color="#374151")
        ax.text(col_x[2], y, f"{acc:,}", ha="center", va="center", fontsize=10, color=ACC_BORDER)
        ax.text(col_x[3], y, f"{rej:,}", ha="center", va="center", fontsize=10, color=REJ_BORDER)

    # Footnote on auxiliary embeddings
    ax.text(
        6.0,
        0.55,
        "Auxiliary embeddings  ·  SPECTER2 (topic FEs, abstract-level)  ·  all-MiniLM-L6-v2 (review-text similarity)",
        ha="center",
        va="center",
        fontsize=10,
        color="#4b5563",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "sample_funnel.pdf", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "sample_funnel.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("Saved sample_funnel")


if __name__ == "__main__":
    render()
