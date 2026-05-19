#!/usr/bin/env python3
"""
Schematic of the fuzzy-RDD LLM-tiebreaker counterfactual design.

Five-step flowchart:
  sample → borderline band (±δ) → two decision rules (human vs LLM) →
  volume-matched accepted sets → Poisson citation outcome + Δ per flip.

Output: plots/tiebreaker_design.{pdf,png}
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

ACCENT = "#1f3a5f"
GREY = "#6b7280"

C_SAMPLE = "#dbeafe"
C_SAMPLE_BD = "#1f3a5f"
C_BAND = "#fde68a"
C_BAND_BD = "#b45309"
C_HUMAN = "#d1fae5"
C_HUMAN_BD = "#059669"
C_LLM = "#fce7f3"
C_LLM_BD = "#be185d"
C_OUT = "#ede9fe"
C_OUT_BD = "#6d28d9"


def rounded(ax, x, y, w, h, face, border, lw=1.2, z=3):
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=face,
        edgecolor=border,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.5):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw, mutation_scale=15, alpha=0.9
        ),
    )


def label(ax, x, y, title, sub, title_color="#111827", title_size=11.5,
          sub_size=9.5, sub_color="#1f2937"):
    ax.text(x, y + 0.22, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=title_color)
    ax.text(x, y - 0.28, sub, ha="center", va="center",
            fontsize=sub_size, color=sub_color, linespacing=1.35)


def render() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Title
    ax.text(7.0, 6.15,
            "LLM-tiebreaker counterfactual  ·  same papers, two decision rules, volume-matched",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111827")
    ax.plot([2.2, 11.8], [5.88, 5.88], color=ACCENT, lw=1.0, alpha=0.45)

    # ── Node 1: Sample ───────────────────────────────
    rounded(ax, 0.4, 3.65, 2.6, 1.2, C_SAMPLE, C_SAMPLE_BD)
    label(ax, 1.7, 4.25, "All papers",
          "ICLR 2018–2020\nn = 2,391", title_size=11)

    # ── Node 2: Borderline band ──────────────────────
    rounded(ax, 3.7, 3.55, 3.0, 1.4, C_BAND, C_BAND_BD)
    label(ax, 5.2, 4.25, "Borderline band",
          "mean_rating ∈\n[cutoff − δ,  cutoff + δ]\nδ ∈ {0.25, 0.5, 0.75}",
          title_size=11)

    arrow(ax, 3.05, 4.25, 3.66, 4.25)

    # ── Node 3: Two decision rules (stacked) ────────
    # Human rule (top)
    rounded(ax, 7.4, 4.45, 3.1, 1.15, C_HUMAN, C_HUMAN_BD)
    label(ax, 8.95, 5.00, "Human rule",
          "accept  ⟺  mean_rating ≥ cutoff",
          title_size=11, sub_size=9.2)

    # LLM tiebreaker (bottom)
    rounded(ax, 7.4, 2.80, 3.1, 1.35, C_LLM, C_LLM_BD)
    label(ax, 8.95, 3.55, "LLM tiebreaker",
          "outside band:  same as human\ninside band:  top-N by LLM rating\n(same N as human)",
          title_size=11, sub_size=9.0)

    arrow(ax, 6.72, 4.50, 7.38, 5.00)
    arrow(ax, 6.72, 4.00, 7.38, 3.45)

    # Volume-match note between the two rules
    ax.text(8.95, 4.32, "volume-matched within band",
            ha="center", va="center", fontsize=8.3, fontstyle="italic", color=GREY)

    # ── Node 4: Outcome ──────────────────────────────
    rounded(ax, 11.1, 3.45, 2.7, 1.55, C_OUT, C_OUT_BD)
    ax.text(12.45, 4.70, "Outcome", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#111827")
    ax.text(12.45, 4.20,
            "Poisson  cites  ~  accepted\n+ mean_rating\n| year × topic FE",
            ha="center", va="center", fontsize=9, color="#1f2937",
            family="monospace", linespacing=1.3)
    ax.text(12.45, 3.60, "Δ cites per flipped accept",
            ha="center", va="center", fontsize=9, color=C_OUT_BD,
            fontweight="600")

    arrow(ax, 10.55, 5.00, 11.10, 4.55)
    arrow(ax, 10.55, 3.45, 11.10, 4.05)

    # ── Bottom note: topic construction ──────────────
    ax.text(
        7.0, 1.80,
        "Topics via k-means (k = 20) on SPECTER2 abstract embeddings;  "
        "paper set fixed across policies → no between-paper confounds;  "
        "same accept count → no volume confound.",
        ha="center", va="center",
        fontsize=9.5, color="#4b5563", style="italic", linespacing=1.45,
    )

    # Decorative row underneath noting what identifies what
    ax.text(
        7.0, 1.05,
        "Identification: the comparison is who gets picked, not how many.",
        ha="center", va="center",
        fontsize=10.5, color=ACCENT, fontweight="600",
    )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tiebreaker_design.pdf", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "tiebreaker_design.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("Saved tiebreaker_design")


if __name__ == "__main__":
    render()
