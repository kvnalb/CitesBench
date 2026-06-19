#!/usr/bin/env python3
"""
Methodology diagram for within-paper pairwise review-text similarity.

Shows the pipeline as four stages:
  1. Pick two reviews of the same paper.
  2. Split each into atomic topics (25–900 chars).
  3. Embed with all-MiniLM-L6-v2 and compute the cosine matrix.
  4. Threshold at 0.5 per row / per column → balanced overlap.

Output: plots/similarity_method.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

# Palette
ACCENT = "#1f3a5f"
A_FACE = "#dbeafe"
A_BD = "#1f3a5f"
B_FACE = "#fef3c7"
B_BD = "#b45309"
HL = "#fde68a"
GREY = "#6b7280"
RULE = "#d1d5db"


def rounded(ax, x, y, w, h, face, border, lw=1.2, z=3):
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=face,
        edgecolor=border,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.4):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw, mutation_scale=14, alpha=0.9
        ),
    )


def render() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Title
    ax.text(
        7.0,
        6.85,
        "Within-paper review similarity  ·  balanced overlap at cosine ≥ 0.5",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    ax.plot([2.2, 11.8], [6.6, 6.6], color=ACCENT, lw=0.9, alpha=0.45)

    # ── Stage 1 · Two reviews ─────────────────────────────────────
    # Left column: two review cards stacked, same paper
    ax.text(
        1.8,
        6.15,
        "① Two reviews of the same paper",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="600",
        color="#111827",
    )

    # Review A
    rounded(ax, 0.25, 4.25, 3.1, 1.60, A_FACE, A_BD)
    ax.text(0.45, 5.72, "Review  A", fontsize=10, fontweight="bold", color=A_BD)
    a_topics = [
        "a₁  no ablation on retrieval module",
        "a₂  claims generalize beyond tested domain",
        "a₃  missing Llama-3 baseline",
    ]
    for i, t in enumerate(a_topics):
        ax.text(0.45, 5.40 - i * 0.34, t, fontsize=8.8, color="#111827")

    # Review B
    rounded(ax, 0.25, 2.45, 3.1, 1.60, B_FACE, B_BD)
    ax.text(0.45, 3.92, "Review  B", fontsize=10, fontweight="bold", color=B_BD)
    b_topics = [
        "b₁  ablation on retrieval is absent",
        "b₂  writing is clear and well-scoped",
        "b₃  evaluation only on one benchmark",
    ]
    for i, t in enumerate(b_topics):
        ax.text(0.45, 3.60 - i * 0.34, t, fontsize=8.8, color="#111827")

    # Source note
    ax.text(
        1.8,
        2.15,
        "Reviews split into atomic topics\n(25–900 chars, strengths / weaknesses / questions)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=GREY,
        style="italic",
        linespacing=1.35,
    )

    # ── Stage 2 · Embed ───────────────────────────────────────────
    ax.text(
        5.6,
        6.15,
        "② Embed each topic",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="600",
        color="#111827",
    )

    # Small "vector" pill per topic
    def vec_pill(x, y, label, face, border):
        w, h = 1.55, 0.36
        rounded(ax, x, y, w, h, face, border, lw=0.9)
        ax.text(x + 0.08, y + h / 2, label, fontsize=8.4, va="center", color="#111827")
        # tiny vector glyph
        for k in range(6):
            ax.plot(
                [x + 0.78 + k * 0.1, x + 0.78 + k * 0.1],
                [y + 0.07, y + h - 0.07],
                color=border,
                lw=0.6,
                alpha=0.55,
            )

    # A vectors
    for i, lab in enumerate(["a₁ →", "a₂ →", "a₃ →"]):
        vec_pill(4.4, 5.40 - i * 0.44, lab, A_FACE, A_BD)
    # B vectors
    for i, lab in enumerate(["b₁ →", "b₂ →", "b₃ →"]):
        vec_pill(4.4, 3.60 - i * 0.44, lab, B_FACE, B_BD)

    ax.text(
        5.2,
        2.15,
        "sentence-transformers · all-MiniLM-L6-v2\nnormalised 384-d vectors",
        ha="center",
        va="top",
        fontsize=8.5,
        color=GREY,
        style="italic",
        linespacing=1.35,
    )

    # arrows review → vector pills
    arrow(ax, 3.4, 5.05, 4.35, 5.05)
    arrow(ax, 3.4, 3.25, 4.35, 3.25)

    # ── Stage 3 · Cosine matrix ──────────────────────────────────
    ax.text(
        8.6,
        6.15,
        "③ Pairwise cosine",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="600",
        color="#111827",
    )

    # Similarity matrix (synthetic illustrative values)
    sim = np.array(
        [
            [0.82, 0.11, 0.38],
            [0.17, 0.29, 0.56],
            [0.44, 0.18, 0.22],
        ]
    )
    mx, my = 7.25, 3.15  # bottom-left corner
    cell = 0.55
    # cells
    for i in range(3):
        for j in range(3):
            val = sim[i, j]
            # colour by value (light blue gradient; stronger for ≥ 0.5)
            if val >= 0.5:
                fc = "#93c5fd"
                ec = "#1d4ed8"
                lw = 1.4
            else:
                fc = "#eff6ff"
                ec = "#cbd5e1"
                lw = 0.6
            rect = mpatches.Rectangle(
                (mx + j * cell, my + (2 - i) * cell),
                cell,
                cell,
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                mx + j * cell + cell / 2,
                my + (2 - i) * cell + cell / 2,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#111827",
                fontweight="600" if val >= 0.5 else "400",
            )

    # axis labels
    for j, lab in enumerate(["b₁", "b₂", "b₃"]):
        ax.text(
            mx + j * cell + cell / 2,
            my + 3 * cell + 0.12,
            lab,
            ha="center",
            va="bottom",
            fontsize=9,
            color=B_BD,
            fontweight="600",
        )
    for i, lab in enumerate(["a₁", "a₂", "a₃"]):
        ax.text(
            mx - 0.12,
            my + (2 - i) * cell + cell / 2,
            lab,
            ha="right",
            va="center",
            fontsize=9,
            color=A_BD,
            fontweight="600",
        )

    ax.text(
        mx + 1.5 * cell,
        my + 3 * cell + 0.55,
        "cos(A, B)",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#111827",
        fontstyle="italic",
    )
    ax.text(
        mx + 1.5 * cell,
        my - 0.30,
        "dark cells ≥ 0.5 threshold",
        ha="center",
        va="top",
        fontsize=8.2,
        color=GREY,
        style="italic",
    )

    # arrows pills → matrix
    arrow(ax, 6.1, 5.05, mx + 0.2, my + 2.8 * cell)
    arrow(ax, 6.1, 3.25, mx + 0.2, my + 0.3 * cell)

    # ── Stage 4 · Balanced overlap ───────────────────────────────
    ax.text(
        11.9,
        6.15,
        "④ Row / column match rate",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="600",
        color="#111827",
    )

    # A→B rate: fraction of A-topics (rows) with any cell ≥ 0.5
    # Rows: a1 has 0.82 (yes), a2 has 0.56 (yes), a3 has no ≥ 0.5 (0.44 max) → 2/3
    # Cols: b1 has 0.82 (yes), b2 has no ≥ 0.5 (0.29 max), b3 has 0.56 (yes) → 2/3
    rows = ["a₁  max = 0.82  ✔", "a₂  max = 0.56  ✔", "a₃  max = 0.44  ✘"]
    cols = ["b₁  max = 0.82  ✔", "b₂  max = 0.29  ✘", "b₃  max = 0.56  ✔"]

    # A→B block
    rounded(ax, 10.1, 4.55, 3.75, 1.35, A_FACE, A_BD, lw=0.9)
    ax.text(10.2, 5.77, "A → B  (per row)", fontsize=9.5, fontweight="bold", color=A_BD)
    for i, r in enumerate(rows):
        ax.text(10.2, 5.42 - i * 0.28, r, fontsize=8.6, color="#111827")
    ax.text(
        13.8,
        4.67,
        "= 2 / 3",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=A_BD,
    )

    # B→A block
    rounded(ax, 10.1, 2.80, 3.75, 1.35, B_FACE, B_BD, lw=0.9)
    ax.text(10.2, 4.02, "B → A  (per column)", fontsize=9.5, fontweight="bold", color=B_BD)
    for i, c in enumerate(cols):
        ax.text(10.2, 3.67 - i * 0.28, c, fontsize=8.6, color="#111827")
    ax.text(
        13.8,
        2.92,
        "= 2 / 3",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=B_BD,
    )

    # Arrow matrix → blocks
    arrow(ax, mx + 3 * cell + 0.1, my + 2.3 * cell, 10.05, 5.20)
    arrow(ax, mx + 3 * cell + 0.1, my + 0.7 * cell, 10.05, 3.45)

    # ── Final equation strip ─────────────────────────────────────
    rounded(ax, 3.2, 0.55, 7.6, 1.15, "#f8fafc", ACCENT, lw=1.2)
    ax.text(
        7.0,
        1.42,
        "Balanced overlap",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        7.0,
        0.97,
        "balanced  =  ½ · ( A→B  +  B→A )   =   ½ · ( 2/3  +  2/3 )   =   0.67",
        ha="center",
        va="center",
        fontsize=12,
        color="#111827",
        family="monospace",
    )

    # Arrows into equation
    arrow(ax, 11.7, 4.55, 10.5, 1.72, color="#9ca3af")
    arrow(ax, 11.7, 2.80, 10.5, 1.72, color="#9ca3af")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "similarity_method.pdf", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "similarity_method.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("Saved similarity_method")


if __name__ == "__main__":
    render()
