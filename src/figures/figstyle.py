"""
One borrowed chart style, shared by the plot scripts.

Not a bespoke design system — seaborn's theme system does the work. `context`
scales type AND spacing together, which is the thing that makes hand-tuned
matplotlib look off: fonts get changed but padding, tick length and line width
don't, so the proportions drift. Setting a context fixes all of them at once.

  "notebook"  screen / slides at this figure size  (what we use)
  "paper"     two-column journal figure — switch with FIG_CONTEXT=paper

For a submission the alternative is SciencePlots (`plt.style.use(["science",
"nature", "no-latex"])`), which targets journal column widths. Kept out of the
default because it is tuned for 3.5in figures and these are read on screens.

Colors are the validated categorical slots (worst adjacent CVD dE 24.7 protan,
checked with the palette validator, not by eye).
"""
import os

import matplotlib.pyplot as plt
import seaborn as sns

# validated categorical slots 1-3
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def apply(context=None):
    """Set the theme. Call once, before creating a figure."""
    sns.set_theme(
        context=context or os.environ.get("FIG_CONTEXT", "notebook"),
        style="whitegrid",
        font=FONT_STACK[0],
        rc={
            "font.sans-serif": FONT_STACK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cfcfca",
            "axes.labelcolor": MUTED,
            "text.color": INK,
            "xtick.color": MUTED, "ytick.color": MUTED,
            "grid.color": "#e5e5e1",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
        },
    )


# every text element and the plot area share this left edge
LEFT = 0.055


def title_block(fig, title, deck=None, y=0.965):
    """Left-aligned headline + deck, on the shared left edge."""
    fig.suptitle(title, x=LEFT, y=y, ha="left", fontsize="large",
                 fontweight="bold", color=INK)
    if deck:
        fig.text(LEFT, y - 0.055, deck, ha="left", va="top",
                 fontsize="small", color=MUTED, linespacing=1.5)


def source(fig, text, y=0.018):
    fig.text(LEFT, y, text, ha="left", va="bottom", fontsize="x-small",
             color=MUTED, linespacing=1.5)


def axis_note(ax, text):
    """Horizontal unit note above the plot instead of a rotated y-label — the
    data-journalism convention, and it stops long labels being clipped."""
    ax.set_ylabel("")
    ax.annotate(text, xy=(0, 1.03), xycoords="axes fraction", ha="left",
                va="bottom", fontsize="small", color=MUTED)


def clean(ax, xgrid=False):
    """Horizontal gridlines only, no box."""
    sns.despine(ax=ax, left=True, bottom=False)
    ax.xaxis.grid(xgrid)
    ax.yaxis.grid(True)
    ax.tick_params(length=0)


def label_ends(ax, ends, x, min_gap, pad="  "):
    """Direct-label series at their right end, nudging apart any that collide.
    ends: list of [y, text, color]. Replaces a legend box."""
    ends = sorted(ends, key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
    for yy, text, col in ends:
        ax.annotate(pad + text, (x, yy), color=col, fontsize="small",
                    va="center", ha="left", fontweight="bold",
                    annotation_clip=False)


def demo():
    apply("talk")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], color=BLUE)
    title_block(fig, "Title", "Deck")
    source(fig, "Source: test")
    assert plt.rcParams["font.sans-serif"][0] == FONT_STACK[0]
    assert plt.rcParams["axes.titlelocation"] == "left"
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    demo()
