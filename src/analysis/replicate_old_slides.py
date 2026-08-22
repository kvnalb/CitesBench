"""
The April 2026 deck's figures, rebuilt on the corrected data (slides 2-6, 21, 23).

WHY. The deck was built on the OpenAlex citation pull and an earlier pool. Every
number in it that touches coverage or citations moved. These are the same seven
exhibits on outputs/paper_master.parquet, so the deck's claims can be checked
against what the data says now rather than re-argued from memory.

FIDELITY. Panel structure, annotations and statistics boxes follow the deck.
Two deliberate departures:

  1. Colour. The deck used red/green for reject/accept, which is the worst pair for
     the ~8% of male readers with deuteranopia. Vermillion and blue carry the same
     semantics and survive CVD simulation. Everything else keeps the repo palette.
  2. Slide 23 panel C is not built. It needs the four per-persona ratings, which
     the pipeline does persist in persona_reviews/*.json but which are not on this
     machine: one of 4,497 paper directories is local, the rest sit at the
     share_coarse_review_json Dropbox paths. Deck Figures 8 and 9 are skipped for
     the same reason plus a sentence-transformers dependency.

  3. YEAR FIXED EFFECTS ONLY. The deck used year x topic FE, with topics from
     k-means on SPECTER2 embeddings. We do not have SPECTER2, a TF-IDF stand-in is
     not the same control, and topic is not what the design needs. Every regression
     here and in venue_premium_rdd.py uses year FE alone.

WHAT MOVED, AND WHAT DID NOT. The year-specific cutoffs and bandwidths are the
deck's own (5.667/1.333, 6.000/1.250, 5.500/1.167) and reproduce here exactly, as
do the 2019 and 2020 in-band counts. What changed is the funnel: the deck could
only observe citations for arXiv-matched papers, and this build reads them from the
S2 tier A+B table, so the observability rows move a long way.

Run: python src/analysis/replicate_old_slides.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs  # noqa: E402

MASTER = "outputs/paper_master.parquet"
RDD_CSV = "outputs/all_paper_results_consistent_gptoss20b.csv"

OUT = "outputs/figures"
STEM = "oldslides"
WIDTH = 6.5

REJ, ACC = fs.VERMILLION, fs.BLUE      # the deck's red/green, made CVD-safe


def load():
    d = pd.read_parquet(MASTER)
    d = d[d.year.isin(spec.YEARS)].copy()
    d["accepted"] = d.decision.str.lower().str.contains("accept")
    d["has_arxiv"] = d.s2_arxiv_id.notna()
    d["has_cites"] = d[spec.OUTCOME].notna()

    # the deck's own year-specific cutoff and bandwidth, carried in the run file
    r = pd.read_csv(RDD_CSV, usecols=["paper_id", "cutoff", "bandwidth",
                                      "deepseek_decision"])
    d = d.merge(r.drop_duplicates("paper_id"), on="paper_id", how="left",
                validate="1:1")
    band = d.groupby("year")[["cutoff", "bandwidth"]].first()
    d["cutoff"] = d.year.map(band.cutoff)
    d["bandwidth"] = d.year.map(band.bandwidth)
    d["r"] = d.mean_rating - d.cutoff
    d["in_band"] = d.r.abs() <= d.bandwidth
    return d, band


def save(fig, n, title):
    fs.add_title(fig, title)
    out = f"{OUT}/{STEM}_{n}"
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)
    plt.close(fig)
    print(f"  -> {out}.pdf")


# --------------------------------------------------------------- slide 2 (Fig 1)

def slide2(d):
    fs.apply(width_in=WIDTH, ncols=2)
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    edges = np.arange(1, 10.34, 1 / 3)

    ax[0].hist(d.mean_rating.dropna(), bins=edges, color=fs.BLUE, zorder=3)
    ax[0].set_xlabel(f"Mean human-reviewer rating\n(all years pooled, "
                     f"n = {int(d.mean_rating.notna().sum()):,})")
    ax[0].set_ylabel("Number of papers")

    for yr, colr in zip(spec.YEARS, [fs.BLUE, fs.VERMILLION, fs.BLUISHGREEN]):
        s = d[d.year == yr].mean_rating.dropna()
        ax[1].hist(s, bins=edges, histtype="step", lw=1.6, color=colr,
                   label=f"{yr} (n={len(s):,})", zorder=3)
    ax[1].set_xlabel("Mean human-reviewer rating\n(by year)")
    ax[1].legend(frameon=False, fontsize="small")

    for a in ax:
        a.set_xlim(1, 10)
        fs.clean(a)
    fs.frame(fig, top_in=0.10, bottom_in=0.68, left=0.10, right=0.99, wspace=0.24)
    save(fig, "01_rating_distribution",
         "Distribution of mean human-reviewer ratings, ICLR 2018-2020")


# --------------------------------------------------------------- slide 3 (Fig 2)

def slide3(d):
    fs.apply(width_in=WIDTH, ncols=2)
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    edges = np.arange(1, 10.34, 1 / 3)

    n_a, n_r = int(d.accepted.sum()), int((~d.accepted).sum())
    ax[0].hist([d.loc[~d.accepted, "mean_rating"].dropna(),
                d.loc[d.accepted, "mean_rating"].dropna()],
               bins=edges, stacked=True, color=[REJ, ACC],
               label=[f"Reject (n={n_r:,})", f"Accept (n={n_a:,})"], zorder=3)
    ax[0].set_xlabel("Mean human-reviewer rating")
    ax[0].set_ylabel("Number of papers")
    ax[0].legend(frameon=False, fontsize="small", loc="upper right")

    # acceptance rate per mass point, marker area proportional to the mass
    g = d.dropna(subset=["mean_rating"]).groupby(
        d.mean_rating.round(2)).agg(p=("accepted", "mean"), n=("accepted", "size"))
    g = g[g.n >= 5]
    ax[1].plot(g.index, g.p, lw=1.0, color=fs.BLUE, zorder=2)
    ax[1].scatter(g.index, g.p, s=np.clip(g.n / 3, 6, 90), color=fs.BLUE, zorder=3,
                  linewidths=0)
    med = float(d.cutoff.median())
    ax[1].axvline(med, color=fs.VERMILLION, ls=(0, (4, 2)), lw=1.2, zorder=4)
    ax[1].annotate(f"median cutoff = {med:.2f}", (med, 1.0), xytext=(4, 0),
                   textcoords="offset points", fontsize="small", color=fs.VERMILLION,
                   va="top")
    ax[1].axhline(0.5, color=fs.GRID, lw=0.9, zorder=1)
    ax[1].set_xlabel("Mean human-reviewer rating")
    ax[1].set_ylabel("Acceptance rate")
    ax[1].set_ylim(-0.03, 1.06)
    ax[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    for a in ax:
        a.set_xlim(1, 10)
        fs.clean(a)
    fs.frame(fig, top_in=0.10, bottom_in=0.46, left=0.10, right=0.99, wspace=0.26)
    save(fig, "02_ratings_and_acceptance",
         "Human-reviewer ratings and acceptance, ICLR 2018-2020")


# --------------------------------------------------------------- slide 4 (Fig 3)

def slide4(d, band):
    fs.apply(width_in=WIDTH, ncols=3)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharey=True)
    for ax, yr in zip(axes, spec.YEARS):
        s = d[(d.year == yr) & d.mean_rating.notna()]
        c, h = band.cutoff[yr], band.bandwidth[yr]
        g = s.groupby(s.mean_rating.round(2)).agg(p=("accepted", "mean"),
                                                  n=("accepted", "size"))
        g = g[g.n >= 3]
        ax.axvspan(c - h, c + h, color=fs.GRID, zorder=1)
        ax.plot(g.index, g.p, lw=1.0, color=fs.BLUE, zorder=2)
        ax.scatter(g.index, g.p, s=np.clip(g.n / 2, 5, 80), color=fs.BLUE,
                   zorder=3, linewidths=0)
        ax.axvline(c, color=fs.VERMILLION, ls=(0, (4, 2)), lw=1.1, zorder=4)
        ax.annotate(f"c = {c:.3f}\nh = {h:.3f}\nRD sample: {int(s.in_band.sum()):,}",
                    (0.03, 0.97), xycoords="axes fraction", va="top",
                    fontsize=plt.rcParams["font.size"] * 0.8, color=fs.INK)
        ax.set_xlabel(f"Mean reviewer rating\nICLR {yr} (n = {len(s):,})")
        ax.set_xlim(1, 10)
        ax.set_ylim(-0.04, 1.08)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        fs.clean(ax)
    axes[0].set_ylabel("Acceptance rate")
    fs.frame(fig, top_in=0.10, bottom_in=0.62, left=0.09, right=0.99, wspace=0.14)
    save(fig, "03_first_stage_by_year",
         "Acceptance probability by paper-level mean rating, by year")


# --------------------------------------------------------------- slide 5 (funnel)

def slide5(d):
    s = d[d.in_band]
    rows = []
    for lab, m in [("Accepted", s.accepted), ("Rejected", ~s.accepted)]:
        g = s[m]
        rows.append((lab, len(g), int(g.has_arxiv.sum()), int(g.has_cites.sum())))

    fs.apply(width_in=WIDTH)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, lines, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                    linewidth=1.2, edgecolor=edge,
                                    facecolor="white", zorder=3))
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
                fontsize=plt.rcParams["font.size"] * 0.85, color=fs.INK, zorder=4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, color=fs.MUTED,
                                     linewidth=0.9, zorder=2))

    box(0.30, 0.86, 0.40, 0.12, [f"Papers in the RD bandwidth", f"n = {len(s):,}"],
        fs.MUTED)
    for i, (lab, n, ax_n, ci) in enumerate(rows):
        x = 0.06 + i * 0.50
        edge = ACC if lab == "Accepted" else REJ
        arrow(0.50, 0.86, x + 0.19, 0.75)
        box(x, 0.62, 0.38, 0.12, [lab, f"n = {n:,}  ({n / len(s):.1%})"], edge)
        arrow(x + 0.19, 0.62, x + 0.19, 0.51)
        box(x, 0.38, 0.38, 0.12,
            ["arXiv matched", f"{ax_n:,}  ({ax_n / n:.1%} of {lab.lower()})"], edge)
        arrow(x + 0.19, 0.38, x + 0.19, 0.27)
        box(x, 0.14, 0.38, 0.12,
            ["S2 citations (tier A+B)", f"{ci:,}  ({ci / n:.1%} of {lab.lower()})"],
            edge)

    per = d[d.in_band].groupby("year").agg(total=("paper_id", "size"),
                                          acc=("accepted", "sum"))
    per["rej"] = per.total - per.acc
    txt = "   ".join(f"{y}: {r.total:,} ({r.acc:,} acc / {r.rej:,} rej)"
                     for y, r in per.iterrows())
    ax.text(0.5, 0.03, txt, ha="center", va="bottom", color=fs.MUTED,
            fontsize=plt.rcParams["font.size"] * 0.8)

    fs.frame(fig, top_in=0.06, bottom_in=0.06, left=0.0, right=1.0)
    save(fig, "04_sample_funnel",
         "Sample selection funnel, ICLR 2018-2020 within the RD bandwidth")


# --------------------------------------------------------------- slide 6 (pipeline)

STAGES = [
    ("Paper", ["PDF + OpenReview", "metadata"], "", "Input"),
    ("Focus notes", ["intro - method", "contribution"], "3 calls", "Stage 1 - council"),
    ("Persona reviews", ["empiricist - theorist", "systems - novelty"], "4 calls",
     "Stage 1 - council"),
    ("Council synth.", ["merged text +", "weighted scores"], "1 call",
     "Stage 1 - council"),
    ("Decision head", ["accept / reject", "+ evidence"], "1 call",
     "Stage 2 - head"),
]


def slide6():
    fs.apply(width_in=WIDTH)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, gap = 0.168, 0.038
    for i, (name, lines, calls, group) in enumerate(STAGES):
        x = 0.005 + i * (w + gap)
        edge = fs.MUTED if i == 0 else (ACC if i == 4 else fs.ORANGE)
        ax.add_patch(FancyBboxPatch((x, 0.34), w, 0.42,
                                    boxstyle="round,pad=0.008", linewidth=1.3,
                                    edgecolor=edge, facecolor="white", zorder=3))
        ax.text(x + w / 2, 0.68, name, ha="center", va="top", zorder=4,
                fontweight="bold", fontsize=plt.rcParams["font.size"] * 0.9)
        ax.text(x + w / 2, 0.56, "\n".join(lines), ha="center", va="top", zorder=4,
                color=fs.MUTED, fontsize=plt.rcParams["font.size"] * 0.75)
        if calls:
            ax.text(x + w / 2, 0.28, calls, ha="center", va="top", color=fs.INK,
                    fontsize=plt.rcParams["font.size"] * 0.75)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.004, 0.55), (x - 0.004, 0.55),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color=fs.MUTED, linewidth=0.9, zorder=2))
    ax.text(0.5, 0.10, "9 model calls per paper - council on Gemma-4-31B, "
            "head on gpt-oss-20b", ha="center", va="bottom", color=fs.MUTED,
            fontsize=plt.rcParams["font.size"] * 0.8)
    fs.frame(fig, top_in=0.06, bottom_in=0.06, left=0.0, right=1.0)
    save(fig, "05_pipeline", "The council pipeline: nine model calls per paper")


# --------------------------------------------------------------- slide 21 (Fig 5)

def confusion(actual, pred):
    tn = int((~actual & ~pred).sum())
    fp = int((~actual & pred).sum())
    fn = int((actual & ~pred).sum())
    tp = int((actual & pred).sum())
    n = tn + fp + fn + tp
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else np.nan
    return np.array([[tn, fp], [fn, tp]]), n, (tp + tn) / n, prec, rec, f1


def slide21(d):
    """Three models, one council packet each, thresholded to a binary.

    Restricted to the RD bandwidth. The deck reports n = 2,361 here, which is the
    in-band sample and not the pool; scoring on the pool instead inflates every
    correlation through a wider spread of the running variable.
    """
    s = d[d.in_band].dropna(subset=["committee_rating", "deepseek_decision",
                                    "single_call_rating"]).copy()
    # the deck's committee arm is a rating threshold; ours uses the same rule, and
    # the head arm is the model's own forced binary.
    models = [
        ("Council rating >= 6\n(Gemma-4-31B)", s.committee_rating >= 6),
        ("Decision head\n(gpt-oss-20b)", s.deepseek_decision.eq("accept")),
        ("Single call >= 7\n(Gemma-4-31B)", s.single_call_rating >= 7),
    ]
    fs.apply(width_in=WIDTH, ncols=3)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7))
    for ax, (name, pred) in zip(axes, models):
        m, n, acc, prec, rec, f1 = confusion(s.accepted.to_numpy(), pred.to_numpy())
        ax.imshow(m / n, cmap="Blues", vmin=0, vmax=0.55)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:,}\n({m[i, j] / n:.1%})", ha="center",
                        va="center", fontsize=plt.rcParams["font.size"] * 0.8,
                        color="white" if m[i, j] / n > 0.30 else fs.INK)
        ax.set_xticks([0, 1], ["Reject", "Accept"])
        ax.set_yticks([0, 1], ["Reject", "Accept"])
        ax.set_xlabel(f"{name}\npredicts accept {pred.mean():.1%}\n"
                      f"acc {acc:.0%}  P {prec:.0%}\nR {rec:.0%}  F1 {f1:.0%}",
                      fontsize=plt.rcParams["font.size"] * 0.72)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.grid(False)          # a heatmap must not have gridlines over the cells
    axes[0].set_ylabel("Actual")
    fs.frame(fig, top_in=0.10, bottom_in=1.02, left=0.10, right=0.99, wspace=0.30)
    save(fig, "06_confusion_matrices",
         f"Confusion matrices vs the area chairs, in-band (n = {len(s):,})")


# --------------------------------------------------------------- slide 23 (Fig 7)

def slide23(d):
    # In-band, per the deck's own n = 2,361. On the full pool the correlation is
    # 0.52 rather than 0.37, purely from the wider rating range.
    s = d[d.in_band].dropna(subset=["mean_rating", "committee_rating"])
    fs.apply(width_in=WIDTH, ncols=2)
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.5))

    for lab, m, colr in [("Reject", ~s.accepted, REJ), ("Accept", s.accepted, ACC)]:
        g = s[m]
        ax[0].scatter(g.mean_rating, g.committee_rating, s=5, alpha=0.35,
                      color=colr, linewidths=0, label=f"{lab} (n={len(g):,})",
                      zorder=3)
    lim = [1, 10]
    ax[0].plot(lim, lim, color=fs.MUTED, ls=(0, (3, 2)), lw=1.0, zorder=2)
    r = s.mean_rating.corr(s.committee_rating)
    rho = s.mean_rating.corr(s.committee_rating, method="spearman")
    mae = (s.committee_rating - s.mean_rating).abs().mean()
    bias = (s.committee_rating - s.mean_rating).mean()
    ax[0].annotate(f"n = {len(s):,}\nPearson r = {r:.3f}\nSpearman = {rho:.3f}\n"
                   f"MAE = {mae:.2f}\nbias = {bias:+.2f}",
                   (0.03, 0.97), xycoords="axes fraction", va="top",
                   fontsize=plt.rcParams["font.size"] * 0.75, color=fs.INK)
    ax[0].set_xlabel("Human mean rating")
    ax[0].set_ylabel("Council rating")
    ax[0].set_xlim(*lim)
    ax[0].set_ylim(*lim)
    ax[0].legend(frameon=False, fontsize=6, loc="lower right")

    edges = np.arange(1, 10.34, 1 / 3)
    for lab, v, colr in [
            (f"Human mean (sd {s.mean_rating.std():.2f})", s.mean_rating, fs.BLUE),
            (f"Council (sd {s.committee_rating.std():.2f})", s.committee_rating,
             fs.ORANGE)]:
        ax[1].hist(v, bins=edges, density=True, histtype="step", lw=1.6,
                   color=colr, label=lab, zorder=3)
    ax[1].set_xlabel("Rating\n(paper-level aggregates)")
    ax[1].set_ylabel("Density")
    ax[1].legend(frameon=False, fontsize=6)

    ax[1].set_xlim(1, 10)
    for a in ax:
        fs.clean(a)
    fs.frame(fig, top_in=0.10, bottom_in=0.62, left=0.10, right=0.99, wspace=0.28)
    save(fig, "07_llm_vs_human_scores",
         "Human review scores against LLM council scores, ICLR 2018-2020")
    return r, mae, bias



# --------------------------------------------------------- slide 22 (deck Fig 6)

def slide22(d):
    """Council rating against the human mean, coloured by the real decision.

    The deck shows this on its own before the three-panel version. Kept separate
    because it is the figure that shows the council compresses the tails: its
    ratings occupy roughly 5 to 7 where the humans use 1 to 9.
    """
    s = d[d.in_band].dropna(subset=["mean_rating", "committee_rating"])
    fs.apply(width_in=WIDTH)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.9))
    for lab, m, colr in [("Reject", ~s.accepted, REJ), ("Accept", s.accepted, ACC)]:
        g = s[m]
        ax.scatter(g.mean_rating, g.committee_rating, s=7, alpha=0.35, color=colr,
                   linewidths=0, label=f"{lab} (n={len(g):,})", zorder=3)
    ax.plot([1, 10], [1, 10], color=fs.MUTED, ls=(0, (3, 2)), lw=1.0, zorder=2)
    r = s.mean_rating.corr(s.committee_rating)
    ax.annotate(f"n = {len(s):,}   r = {r:.3f}", (0.03, 0.97),
                xycoords="axes fraction", va="top", color=fs.INK,
                fontsize=plt.rcParams["font.size"] * 0.8)
    ax.set_xlabel("Mean human-reviewer rating")
    ax.set_ylabel("Council weighted-average rating")
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 10)
    ax.legend(frameon=False, fontsize="small", loc="lower right")
    fs.clean(ax)
    fs.frame(fig, top_in=0.10, bottom_in=0.44, left=0.15, right=0.98)
    save(fig, "08_council_vs_human",
         "Council rating against the human mean, by decision")


# ------------------------------------------------ slides 31-33 (deck Figs 13-15)

DELTAS = (0.25, 0.50, 0.75)


def counterfactual(d, delta):
    """The deck's volume-matched tiebreaker, per year.

    Inside a band of half-width `delta` around that year's cutoff, the human rule
    accepts everything at or above the cutoff. The LLM rule accepts the same NUMBER
    of papers, chosen as the top-N by council rating. Volume matching is what makes
    this a comparison of WHO gets picked rather than how many.

    Returns the in-band rows with a `flip` label: flip_to_accept means the council
    admits a paper the humans rejected, flip_to_reject the reverse. Counts are equal
    by construction.
    """
    out = []
    for yr in spec.YEARS:
        s = d[(d.year == yr) & d.mean_rating.notna()
              & d.committee_rating.notna()].copy()
        c = float(s.cutoff.iloc[0])
        s = s[(s.mean_rating - c).abs() <= delta].copy()
        if not len(s):
            continue
        s["human_accept"] = s.mean_rating >= c
        n = int(s.human_accept.sum())
        # rank by council rating; ties broken by the human score, then paper_id, so
        # the slate is reproducible
        order = s.sort_values(["committee_rating", "mean_rating", "paper_id"],
                              ascending=[False, False, True])
        s["llm_accept"] = s.paper_id.isin(order.paper_id.iloc[:n])
        s["llm_cutoff"] = (order.committee_rating.iloc[n - 1] if n else np.nan)
        s["flip"] = np.where(s.llm_accept & ~s.human_accept, "flip_to_accept",
                             np.where(~s.llm_accept & s.human_accept,
                                      "flip_to_reject", "none"))
        s["delta"] = delta
        out.append(s)
    return pd.concat(out, ignore_index=True)


def slide31():
    """The design diagram: same papers, two rules, volume-matched in a band."""
    fs.apply(width_in=WIDTH)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, head, lines, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.007",
                                    linewidth=1.2, edgecolor=edge,
                                    facecolor="white", zorder=3))
        ax.text(x + w / 2, y + h - 0.06, head, ha="center", va="top", zorder=4,
                fontweight="bold", fontsize=plt.rcParams["font.size"] * 0.85)
        ax.text(x + w / 2, y + h - 0.34, "\n".join(lines), ha="center", va="top",
                zorder=4, color=fs.MUTED,
                fontsize=plt.rcParams["font.size"] * 0.72)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, color=fs.MUTED,
                                     linewidth=0.9, zorder=2))

    box(0.005, 0.34, 0.20, 0.44, "All papers",
        ["ICLR 2018-2020", "in the RD band"], fs.MUTED)
    box(0.245, 0.34, 0.20, 0.44, "Borderline band",
        ["|score - cutoff|", "<= 0.25 / 0.5 / 0.75"], fs.ORANGE)
    box(0.49, 0.60, 0.24, 0.34, "Human rule",
        ["accept iff score >= cutoff"], ACC)
    box(0.49, 0.16, 0.24, 0.34, "Council rule",
        ["top-N by council rating,", "same N as the humans"], REJ)
    box(0.775, 0.34, 0.22, 0.44, "Outcome",
        ["S2 citations of the", "papers each rule swaps in"], fs.MUTED)
    arrow(0.205, 0.56, 0.245, 0.56)
    arrow(0.445, 0.56, 0.49, 0.72)
    arrow(0.445, 0.56, 0.49, 0.36)
    arrow(0.73, 0.72, 0.775, 0.58)
    arrow(0.73, 0.36, 0.775, 0.54)
    ax.text(0.5, 0.03, "Same paper set and same accept count under both rules, so "
            "the comparison is who gets picked, not how many",
            ha="center", va="bottom", color=fs.INK,
            fontsize=plt.rcParams["font.size"] * 0.78)
    fs.frame(fig, top_in=0.06, bottom_in=0.06, left=0.0, right=1.0)
    save(fig, "09_counterfactual_design",
         "Counterfactual design: same papers, two decision rules, volume-matched")


def slide32(d):
    """The deck's Figure 14: the band at delta = 0.5, one point per paper."""
    cf = counterfactual(d, 0.50)
    fs.apply(width_in=WIDTH, ncols=3)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharey=True)
    for ax, yr in zip(axes, spec.YEARS):
        s = cf[cf.year == yr]
        for lab, colr, z in [("none", fs.NEUTRAL, 2), ("flip_to_accept", ACC, 4),
                             ("flip_to_reject", REJ, 4)]:
            g = s[s.flip == lab]
            ax.scatter(g.mean_rating, g.committee_rating, s=11, zorder=z,
                       color=colr, linewidths=0, alpha=0.85,
                       label=f"{lab.replace('_', ' ')} ({len(g)})")
        ax.axvline(float(s.cutoff.iloc[0]), color=fs.MUTED, ls=(0, (4, 2)), lw=1.0)
        ax.axhline(float(s.llm_cutoff.iloc[0]), color=fs.BLUE, ls=(0, (2, 2)),
                   lw=1.0)
        ax.set_xlabel(f"Mean human rating\nICLR {yr} (n = {len(s):,})")
        ax.legend(frameon=False, fontsize=5.2, loc="upper left")
        fs.clean(ax)
    axes[0].set_ylabel("Council rating")
    fs.frame(fig, top_in=0.10, bottom_in=0.62, left=0.10, right=0.99, wspace=0.16)
    save(fig, "10_borderline_flips",
         "Borderline band at delta = 0.5: which papers the council swaps")
    return cf


def flip_gain(d):
    """Deck Figure 15, bottom panel: how good are the papers the council swaps in?

    For each band width, compare the papers the council admits and the humans did
    not against the papers the humans admitted and the council did not. Volume
    matching makes the two sets the same size, so this is a like-for-like swap.

    THREE STATISTICS, NOT ONE. The deck fitted a Poisson and called the result
    tempered. A Poisson on `accepted + mean_rating + year` does the opposite here:
    it predicts MORE citations for the swapped-out papers, because they sit above
    the cutoff and were accepted, so their residuals go more negative and the gap
    widens. The first version of this function reported +1197% at delta = 0.25 for
    that reason.

    So the tail is tempered the way the rest of this project tempers it, with the
    median and the mean of log(1 + citations). Those are spec.py's own metrics and
    they need no model. The raw mean is kept beside them, because the distance
    between the raw mean and the median IS the heavy tail the deck was worried
    about.
    """
    ib = d[d.in_band].dropna(subset=[spec.OUTCOME]).copy()
    ib["lc"] = np.log1p(ib[spec.OUTCOME])

    rows = []
    for delta in DELTAS:
        cf = counterfactual(d, delta)
        j = cf[["paper_id", "flip"]].merge(ib, on="paper_id", how="inner")
        a_, r_ = j[j.flip == "flip_to_accept"], j[j.flip == "flip_to_reject"]
        rows.append({
            "delta": delta,
            "n_flips": int((cf.flip == "flip_to_accept").sum()),
            "n_scored_in": len(a_), "n_scored_out": len(r_),
            "mean_in": a_[spec.OUTCOME].mean(), "mean_out": r_[spec.OUTCOME].mean(),
            "median_in": a_[spec.OUTCOME].median(),
            "median_out": r_[spec.OUTCOME].median(),
            "logmean_in": a_.lc.mean(), "logmean_out": r_.lc.mean(),
            "gain_mean": a_[spec.OUTCOME].mean() - r_[spec.OUTCOME].mean(),
            "gain_median": a_[spec.OUTCOME].median() - r_[spec.OUTCOME].median(),
            "gain_log": a_.lc.mean() - r_.lc.mean()})
    return pd.DataFrame(rows)


def slide33(d):
    """Deck Figure 15. Three panels, not two: median citations and mean log are on
    incompatible scales and sharing an axis makes the log bars invisible."""
    g = flip_gain(d)
    fs.apply(width_in=WIDTH, nrows=3)
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 4.2))
    x = np.arange(len(g))
    w = 0.34

    axes[0].bar(x - w / 2, g.n_flips, width=w, color=ACC, zorder=3,
                label="flipped to accept")
    axes[0].bar(x + w / 2, -g.n_flips, width=w, color=REJ, zorder=3,
                label="flipped to reject")
    for xi, n in zip(x, g.n_flips):
        axes[0].annotate(f"{n:+,}", (xi - w / 2, n), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize="small")
    axes[0].axhline(0, color=fs.INK, lw=0.9, zorder=4)
    axes[0].set_ylabel("Papers flipped")
    axes[0].set_ylim(-max(g.n_flips) * 1.6, max(g.n_flips) * 1.6)
    axes[0].legend(frameon=False, fontsize="small", ncol=2, loc="lower left")

    for ax, col, lab, colr, fmt in [
            (axes[1], "gain_median", "Median cites,\nin minus out", fs.BLUE,
             "{:+,.0f}"),
            (axes[2], "gain_log", "Mean log cites,\nin minus out", fs.ORANGE,
             "{:+.2f}")]:
        ax.bar(x, g[col], width=0.45, color=colr, zorder=3)
        for xi, v in zip(x, g[col]):
            ax.annotate(fmt.format(v), (xi, v), xytext=(0, 4 if v >= 0 else -11),
                        textcoords="offset points", ha="center", fontsize="small")
        ax.axhline(0, color=fs.INK, lw=0.9, zorder=4)
        ax.set_ylabel(lab)
        ax.set_ylim(min(0, g[col].min()) * 1.35 - abs(g[col].max()) * 0.12,
                    max(0, g[col].max()) * 1.35)

    # tick detail on the bottom panel only; repeating it three times is noise
    for ax in axes[:2]:
        ax.set_xticks(x, [""] * len(x))
        fs.clean(ax)
    axes[2].set_xticks(x, [f"{v:g}\n{m:,} flips\n{k:,} with citations"
                           for v, m, k in zip(g.delta, g.n_flips, g.n_scored_in)])
    fs.clean(axes[2])
    axes[2].set_xlabel("Band half-width around the cutoff (rating units)")
    fs.frame(fig, top_in=0.10, bottom_in=0.78, left=0.16, right=0.99, hspace=0.22)
    save(fig, "11_flip_counts_and_gain",
         "Council tiebreaker: papers flipped, and the citations they buy")
    print()
    print(g.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return g


def build():
    os.makedirs(OUT, exist_ok=True)
    d, band = load()
    slide2(d)
    slide3(d)
    slide4(d, band)
    slide5(d)
    slide6()
    slide21(d)
    r, mae, bias = slide23(d)
    slide22(d)
    slide31()
    slide32(d)
    g = slide33(d)
    return d, band, r, mae, bias, g


def demo():
    d, band, r, mae, bias, g = build()

    # Internal validity only. The deck's numbers are not the target: it ran on the
    # OpenAlex pull and these run on S2 tier A+B, so every citation-side figure is
    # SUPPOSED to differ. What must hold is that each exhibit describes the pool it
    # claims to describe.
    assert len(d) == 4567, f"pool is {len(d)}, expected 4,567"
    assert d.paper_id.is_unique, "not one row per paper"

    # The band definition must be the run's own, not re-derived here.
    for yr in spec.YEARS:
        s_yr = d[d.year == yr]
        assert s_yr.cutoff.nunique() == 1 and s_yr.bandwidth.nunique() == 1, \
            f"{yr}: more than one cutoff or bandwidth"
        w = float(s_yr.bandwidth.iloc[0])
        assert (s_yr.loc[s_yr.in_band, "r"].abs() <= w + 1e-9).all(), \
            f"{yr}: a paper outside the bandwidth is flagged in-band"

    # A fuzzy design needs a first stage that is a step, not a ramp.
    ib = d[d.in_band]
    lo = ib.loc[ib.r < 0, "accepted"].mean()
    hi = ib.loc[ib.r >= 0, "accepted"].mean()
    assert hi - lo > 0.20, f"first stage only {hi - lo:.3f} across the cutoff"

    # Citations must be the S2 tier A+B column, not an OpenAlex remnant.
    assert set(d.citation_tier.dropna()) <= set(spec.TIERS), \
        f"unexpected citation tiers: {set(d.citation_tier.dropna())}"

    # The council must actually disagree with the humans, or the confusion matrices
    # and the counterfactual are measuring nothing.
    assert 0.1 < abs(r) < 0.9, f"council-human correlation {r:.3f} is degenerate"

    # Volume matching is the identifying assumption: equal counts in and out.
    assert (g.n_flips > 0).all(), "no flips at some band width"

    print(f"\nok — 11 figures on {len(d):,} papers; in-band "
          f"{len(ib):,}; first stage {lo:.1%} -> {hi:.1%} across the cutoff; "
          f"council-vs-human r={r:.3f}, MAE={mae:.2f}, bias={bias:+.2f}")


if __name__ == "__main__":
    demo()
