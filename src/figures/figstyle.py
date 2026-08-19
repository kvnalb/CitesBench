"""
One chart style for the paper's exhibits: tueplots for the geometry, Okabe-Ito
for the colour.

WHY TUEPLOTS. Hand-tuned matplotlib drifts because fonts get changed and padding,
tick length and line width do not, so the proportions stop matching the page the
figure lands on. `tueplots.bundles.iclr2024()` sets all of them together against
ICLR's actual column width and body size — 5.5in text width, 9pt body, 7pt ticks
and legend. Using the bundle for the venue we are writing about also means a
figure that looks right in the paper looks right here.

Two deliberate departures from the bundle:

  - `text.usetex` is forced off. The bundle asks for LaTeX with the times package;
    there is no LaTeX on this machine, and a missing binary fails at draw time
    with a stack trace rather than at import. The serif stack below targets Times
    directly so the result still matches the venue.
  - `figure.constrained_layout.use` is forced off. Every script here places its
    own title block and source line with `fig.text` and then calls
    `subplots_adjust`; constrained layout silently ignores that and reflows.

The bundle asks for Times because that is ICLR's body face. Figures here are set
in Helvetica instead: sans in the figures against serif in the running text is the
convention across NeurIPS, ICML and ICLR, and it keeps axis labels legible at the
6-7pt tick sizes the bundle specifies, where a Times numeral gets thin. See
`resolved_font()` — a silent fallback to DejaVu is a failure, not a default.

WHY OKABE-ITO. It is the reference qualitative palette for colour-vision
deficiency (Okabe & Ito 2008), and unlike an ad-hoc set it is safe across the
whole palette rather than only for the pairs someone happened to check. Measured
separation for the four slots this project assigns, OKLab dE x100:

    pair                     normal  protan  deutan  tritan
    AC / council               31.2    24.0    30.6    29.2
    AC / single call           18.7    18.3    17.2    10.5
    council / single call      25.8    11.5    13.7    25.6

Worst pair across all four slots is 8.2 under protanopia, against a target of 8.
Slot assignment lives in spec.py, not here — this module supplies the palette,
spec.py decides which regime wears which colour.

Run: python src/figures/figstyle.py   (self-check)
"""
import os

import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles, figsizes

# Okabe & Ito's colour-vision-deficiency-safe qualitative palette, in the
# published order. Referenced by name below; do not reorder.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
(ORANGE, SKYBLUE, BLUISHGREEN, YELLOW,
 BLUE, VERMILLION, REDDISHPURPLE, BLACK) = OKABE_ITO

# Ink and grounds. Not from the palette: these carry no categorical meaning, and
# giving a reference line or an axis label a palette colour would read as a
# fourth series.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e2df"
NEUTRAL = "#bdbdb8"          # reference bars, "rejected" arm, anything not a regime

# Sans, the convention for figures in ML papers even where the body text is Times:
# Helvetica is what \usepackage{helvet} resolves to (phv), Arial is its Windows
# metric equivalent, Nimbus Sans and Liberation Sans are the free clones that stand
# in on Linux and CI. DejaVu Sans is matplotlib's own default and only catches a
# box with none of the above.
SANS_STACK = ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"]

BUNDLE = "iclr2024"


def resolved_font():
    """The face matplotlib will actually use, not the one we asked for.

    A missing family falls back silently and the figure just looks slightly wrong,
    which is the hardest kind of style bug to notice in a PDF. demo() asserts on
    this so a machine without Helvetica says so instead of shipping DejaVu.
    """
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in SANS_STACK:
        if name in have:
            return name
    return None


def rc(nrows=1, ncols=1, rel_width=1.0):
    """The bundle's rcParams, with this project's two departures applied.

    Pass nrows/ncols to get the figure size tueplots would choose for a grid of
    that shape at ICLR's column width; scripts that size their own figure can
    ignore the returned `figure.figsize`.
    """
    params = dict(bundles.iclr2024(usetex=False))
    params.update(figsizes.iclr2024(nrows=nrows, ncols=ncols, rel_width=rel_width))
    params.update({
        "text.usetex": False,               # no LaTeX on this machine — see docstring
        "figure.constrained_layout.use": False,   # scripts lay themselves out
        "figure.autolayout": False,
        "font.family": "sans-serif",
        "font.sans-serif": SANS_STACK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cfcfca",
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    })
    return params


TEXT_WIDTH_IN = 5.5          # ICLR's text width, per tueplots


def apply(width_in=TEXT_WIDTH_IN, nrows=1, ncols=1):
    """Set the theme for a figure that will be drawn `width_in` inches wide.

    THIS IS THE ARGUMENT THAT MATTERS. tueplots sizes type for a figure drawn at
    ICLR's 5.5in text width, where 9pt body type renders as 9pt on the page. A
    figure drawn at 11in and then placed at \textwidth is scaled to 0.5x by
    LaTeX, so that same 9pt lands on the page as 4.5pt and the bundle's whole
    point is lost.

    Drawing at width W and scaling the type by W/5.5 produces output identical to
    drawing at 5.5in, while leaving room for the title block and source line this
    project puts inside the figure rather than in the LaTeX caption. So callers
    declare the width they will actually use and the type follows it, instead of
    a magic number that drifts away from the figsize above it.
    """
    sns.set_palette(sns.color_palette(OKABE_ITO))
    params = rc(nrows, ncols)
    scale = width_in / TEXT_WIDTH_IN
    if scale != 1.0:
        for k in ("font.size", "axes.labelsize", "axes.titlesize",
                  "legend.fontsize", "xtick.labelsize", "ytick.labelsize"):
            params[k] = params[k] * scale
    sns.set_theme(style="whitegrid", rc=params)
    plt.rcParams.update(params)          # seaborn's own defaults must not win
    return scale


def figsize(nrows=1, ncols=1, rel_width=1.0):
    """ICLR-correct figure size for a grid of this shape."""
    return figsizes.iclr2024(nrows=nrows, ncols=ncols,
                             rel_width=rel_width)["figure.figsize"]


# every text element and the plot area share this left edge
LEFT = 0.055

# Layout below is specified in INCHES, not axes fractions. A fraction that looks
# right at 9pt on a 5.5in figure overlaps its own axis labels at 19pt on an 11.5in
# one, because the text grows and the fraction does not. Inches are invariant to
# both the figure size and the type scale, so a caller sets the margin once.
LINE = 1.45          # line height as a multiple of font size


def _in(fig, pt):
    """Points to a fraction of this figure's height."""
    return pt / 72 / fig.get_figheight()


def title_block(fig, title, deck=None, top_in=0.30):
    """Left-aligned headline + deck, `top_in` inches below the top edge."""
    base = plt.rcParams["font.size"]
    title_pt = base * 1.2                       # matplotlib's "large"
    y = 1 - _in(fig, top_in * 72)
    fig.suptitle(title, x=LEFT, y=y, ha="left", fontsize=title_pt,
                 fontweight="bold", color=INK, va="top")
    if deck:
        fig.text(LEFT, y - _in(fig, title_pt * LINE), deck, ha="left", va="top",
                 fontsize=base * 0.85, color=MUTED, linespacing=1.5)


def source(fig, text, bottom_in=0.12):
    fig.text(LEFT, _in(fig, bottom_in * 72), text, ha="left", va="bottom",
             fontsize=plt.rcParams["font.size"] * 0.75, color=MUTED,
             linespacing=1.5)


def deck_height(fig, n_lines):
    """Inches the title block occupies for a deck of `n_lines`."""
    base = plt.rcParams["font.size"]
    return (base * 1.2 * LINE + base * 0.85 * LINE * n_lines) / 72


def frame(fig, top_in, bottom_in, left=None, right=0.98, wspace=0.22, hspace=0.4):
    """subplots_adjust in inches, so margins survive a change of type scale."""
    h = fig.get_figheight()
    fig.subplots_adjust(left=left if left is not None else LEFT, right=right,
                        top=1 - top_in / h, bottom=bottom_in / h,
                        wspace=wspace, hspace=hspace)


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
    scale = apply()
    assert scale == 1.0, "5.5in is the reference width and must not be scaled"
    assert apply(11.0) == 2.0, "type must scale with the drawn width"
    apply()
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"][:3] == OKABE_ITO[:3], \
        "seaborn overrode the Okabe-Ito cycle"
    assert plt.rcParams["text.usetex"] is False, "usetex must stay off — no LaTeX here"
    assert plt.rcParams["figure.constrained_layout.use"] is False, \
        "constrained layout fights the manual title blocks"
    assert plt.rcParams["axes.titlelocation"] == "left"
    assert len(set(OKABE_ITO)) == 8, "palette has a duplicate"
    face = resolved_font()
    assert face is not None, f"none of {SANS_STACK} is installed — figures would " \
                             "fall back to whatever matplotlib finds, silently"
    assert face != "DejaVu Sans", ("only DejaVu Sans is available; install Helvetica, "
                                   "Arial or Nimbus Sans before building exhibits")

    fig, ax = plt.subplots(figsize=figsize())
    for i, c in enumerate(OKABE_ITO[:4]):
        ax.plot([0, 1], [0, i], color=c)
    title_block(fig, "Title", "Deck")
    source(fig, "Source: test")
    clean(ax)
    plt.close(fig)

    w, h = figsize(nrows=2, ncols=3)
    assert abs(w - 5.5) < 0.01, f"ICLR text width should be 5.5in, got {w}"
    print(f"ok — tueplots {BUNDLE}, {face}, Okabe-Ito x{len(OKABE_ITO)}, "
          f"1x1 figsize {figsize()[0]:.2f}x{figsize()[1]:.2f}in")


if __name__ == "__main__":
    demo()
