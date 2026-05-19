#!/usr/bin/env python3
"""
Slim Coarse pipeline diagram — horizontal flow, 5 stages.

Produces an overview plus one highlighted variant per stage so that
slides can pair a zoomed-in pipeline view with prompt / output examples.

Outputs (PDF + PNG):
  pipeline_overview   — all stages equal weight
  pipeline_focus      — Focus Notes stage highlighted
  pipeline_persona    — Persona Reviews highlighted
  pipeline_committee  — Committee Synthesis highlighted
  pipeline_decision   — Decision Head highlighted
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"

# Palette, tuned to match the Inter/navy slide theme
C_INPUT = "#dbeafe"
C_INPUT_BD = "#1f3a5f"
C_LLM = "#fff1db"
C_LLM_BD = "#b45309"
C_HEAD = "#fee2e2"
C_HEAD_BD = "#b91c1c"
ACCENT = "#1f3a5f"

STAGES = [
    {
        "id": "input",
        "title": "Paper",
        "sub": "PDF + OpenReview\nmetadata",
        "color": C_INPUT,
        "border": C_INPUT_BD,
        "model": "",
        "calls": "",
    },
    {
        "id": "focus",
        "title": "Focus Notes",
        "sub": "intro · method\ncontribution",
        "color": C_LLM,
        "border": C_LLM_BD,
        "model": "Gemma-4-31B",
        "calls": "3 LLM calls",
    },
    {
        "id": "persona",
        "title": "Persona Reviews",
        "sub": "empiricist · theorist\nsystems · novelty",
        "color": C_LLM,
        "border": C_LLM_BD,
        "model": "Gemma-4-31B",
        "calls": "4 LLM calls",
    },
    {
        "id": "committee",
        "title": "Committee Synth.",
        "sub": "merged review text +\nweighted scores",
        "color": C_LLM,
        "border": C_LLM_BD,
        "model": "Gemma-4-31B",
        "calls": "1 LLM call",
    },
    {
        "id": "decision",
        "title": "Decision Head",
        "sub": "accept / reject\n+ evidence reasons",
        "color": C_HEAD,
        "border": C_HEAD_BD,
        "model": "DeepSeek V3.1  /\nGemma-2-9B",
        "calls": "1 LLM call",
    },
]


def render(highlight: Optional[str], outname: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    n = len(STAGES)
    # Evenly space box centres between x = 1.5 and x = 12.5
    xs = [1.5 + i * (11.0 / (n - 1)) for i in range(n)]
    y_center = 2.55
    box_w = 2.15
    box_h = 1.55

    # Connecting arrows
    for i in range(n - 1):
        x1 = xs[i] + box_w / 2 + 0.02
        x2 = xs[i + 1] - box_w / 2 - 0.02
        neighbors = {STAGES[i]["id"], STAGES[i + 1]["id"]}
        alpha = 0.95 if (highlight is None or highlight in neighbors) else 0.2
        ax.annotate(
            "",
            xy=(x2, y_center),
            xytext=(x1, y_center),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#6b7280",
                lw=1.6,
                alpha=alpha,
                mutation_scale=14,
            ),
        )

    # Stages
    for i, st in enumerate(STAGES):
        is_hl = st["id"] == highlight
        is_fade = highlight is not None and not is_hl
        alpha = 0.28 if is_fade else 1.0
        linewidth = 2.8 if is_hl else 1.0

        # Outer shadow for highlight
        if is_hl:
            shadow = mpatches.FancyBboxPatch(
                (xs[i] - box_w / 2 - 0.06, y_center - box_h / 2 - 0.06),
                box_w + 0.12,
                box_h + 0.12,
                boxstyle="round,pad=0.04,rounding_size=0.14",
                facecolor="#fde68a",
                edgecolor="none",
                alpha=0.55,
                zorder=2,
            )
            ax.add_patch(shadow)

        box = mpatches.FancyBboxPatch(
            (xs[i] - box_w / 2, y_center - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=st["color"],
            edgecolor=st["border"],
            linewidth=linewidth,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(box)

        # Title (bold)
        ax.text(
            xs[i],
            y_center + 0.30,
            st["title"],
            ha="center",
            va="center",
            fontsize=14.5,
            fontweight="bold",
            color="#111827",
            alpha=alpha,
            zorder=4,
        )

        # Subtext (two lines)
        ax.text(
            xs[i],
            y_center - 0.28,
            st["sub"],
            ha="center",
            va="center",
            fontsize=10.5,
            color="#1f2937",
            alpha=alpha * 0.88,
            linespacing=1.25,
            zorder=4,
        )

        # Model label above the box
        if st["model"]:
            ax.text(
                xs[i],
                y_center + box_h / 2 + 0.28,
                st["model"],
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#4b5563",
                style="italic",
                alpha=alpha * 0.9,
                linespacing=1.2,
            )

        # Calls label below
        if st["calls"]:
            ax.text(
                xs[i],
                y_center - box_h / 2 - 0.18,
                st["calls"],
                ha="center",
                va="top",
                fontsize=9.5,
                color="#4b5563",
                fontweight="500",
                alpha=alpha * 0.9,
            )

    # Header
    title_text = "Slim Coarse Pipeline  ·  9 LLM calls per paper"
    ax.text(
        7.0,
        4.65,
        title_text,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )

    # Decorative rule under title
    ax.plot([4.5, 9.5], [4.42, 4.42], color=ACCENT, lw=1.0, alpha=0.5)

    # Caption strip at the bottom indicating the three groups
    strip_y = 0.75
    ax.text(
        xs[0],
        strip_y,
        "Input",
        ha="center",
        va="center",
        fontsize=9.5,
        color=C_INPUT_BD,
        fontweight="600",
        alpha=0.8,
    )
    committee_mid = (xs[1] + xs[3]) / 2
    ax.text(
        committee_mid,
        strip_y,
        "Stage 1 · Committee",
        ha="center",
        va="center",
        fontsize=9.5,
        color=C_LLM_BD,
        fontweight="600",
        alpha=0.85,
    )
    ax.text(
        xs[4],
        strip_y,
        "Stage 2 · Decision Head",
        ha="center",
        va="center",
        fontsize=9.5,
        color=C_HEAD_BD,
        fontweight="600",
        alpha=0.85,
    )
    # Underline segments for each group
    underline_y = 1.08
    ax.plot(
        [xs[0] - box_w / 2, xs[0] + box_w / 2],
        [underline_y, underline_y],
        color=C_INPUT_BD,
        lw=2.0,
        alpha=0.35,
    )
    ax.plot(
        [xs[1] - box_w / 2, xs[3] + box_w / 2],
        [underline_y, underline_y],
        color=C_LLM_BD,
        lw=2.0,
        alpha=0.35,
    )
    ax.plot(
        [xs[4] - box_w / 2, xs[4] + box_w / 2],
        [underline_y, underline_y],
        color=C_HEAD_BD,
        lw=2.0,
        alpha=0.35,
    )

    fig.tight_layout()
    pdf_path = PLOT_DIR / f"{outname}.pdf"
    png_path = PLOT_DIR / f"{outname}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved {outname}")


if __name__ == "__main__":
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    render(highlight=None, outname="pipeline_overview")
    render(highlight="focus", outname="pipeline_focus")
    render(highlight="persona", outname="pipeline_persona")
    render(highlight="committee", outname="pipeline_committee")
    render(highlight="decision", outname="pipeline_decision")
