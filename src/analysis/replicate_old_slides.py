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
  2. Slide 23 panel C. The deck plotted raw per-persona LLM scores. Our council
     persists only the paper-level committee rating for this sample, not the nine
     individual calls, so the panel compares raw individual HUMAN review scores
     against the paper-level council and single-call ratings. The axis says so. A
     fabricated per-persona distribution would be worse than a labelled substitute.

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
    fs.apply(width_in=WIDTH, ncols=3)
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.5))

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

    # Panel C. Per-persona scores were not persisted for this sample, so this
    # compares INDIVIDUAL human reviews against the paper-level LLM ratings. The
    # axis label says so rather than implying a persona distribution we do not have.
    import sqlite3
    con = sqlite3.connect("data/gen_review.db")
    raw = pd.read_sql("SELECT paper_id, rating FROM REVIEW", con)
    con.close()
    raw = raw[raw.paper_id.isin(set(s.paper_id))]
    hv = raw.rating.str.extract(r"^(\d+)")[0].astype(float).dropna()
    for lab, v, colr in [(f"Human reviews, individual (n={len(hv):,})", hv, fs.BLUE),
                         (f"Council, paper level (n={len(s):,})",
                          s.committee_rating, fs.ORANGE),
                         (f"Single call, paper level (n={s.single_call_rating.notna().sum():,})",
                          s.single_call_rating.dropna(), fs.BLUISHGREEN)]:
        ax[2].hist(v, bins=edges, density=True, histtype="step", lw=1.6,
                   color=colr, label=lab, zorder=3)
    ax[2].set_xlabel("Rating\n(individual human vs paper-level LLM)")
    ax[2].set_ylim(0, 2.15)          # room for the three-line legend
    ax[2].legend(frameon=False, fontsize=5.2, loc="upper left")

    for a in ax[1:]:
        a.set_xlim(1, 10)
    for a in ax:
        fs.clean(a)
    fs.frame(fig, top_in=0.10, bottom_in=0.66, left=0.09, right=0.97, wspace=0.32)
    save(fig, "07_llm_vs_human_scores",
         "Human review scores against LLM council scores, ICLR 2018-2020")
    return r, mae, bias


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
    return d, band, r, mae, bias


def demo():
    d, band, r, mae, bias = build()

    # The deck's cutoffs and bandwidths must reproduce exactly, or these are not
    # replications of its figures.
    want = {2018: (5.667, 1.333), 2019: (6.000, 1.250), 2020: (5.500, 1.167)}
    for yr, (c, h) in want.items():
        assert abs(band.cutoff[yr] - c) < 5e-4 and abs(band.bandwidth[yr] - h) < 5e-4, \
            f"{yr}: cutoff/bandwidth {band.cutoff[yr]}/{band.bandwidth[yr]} != {c}/{h}"

    # The deck's in-band counts for 2019 and 2020.
    ib = d[d.in_band].groupby("year").size()
    assert ib[2019] == 896 and ib[2020] == 983, f"in-band counts moved: {ib.to_dict()}"

    # The deck reported r = 0.383, MAE 0.59, bias -0.31 on the council-vs-human
    # scatter. Same models and same ratings, so these should barely move; a large
    # move means the run behind this figure is not the run the deck used.
    assert abs(r - 0.383) < 0.03, f"Pearson r {r:.3f}, deck had 0.383"
    assert abs(mae - 0.59) < 0.06, f"MAE {mae:.2f}, deck had 0.59"
    assert abs(bias - -0.31) < 0.06, f"bias {bias:+.2f}, deck had -0.31"

    print(f"\nok — 7 figures; cutoffs reproduce the deck exactly, in-band "
          f"2019/2020 = {ib[2019]}/{ib[2020]}; council-vs-human r={r:.3f} "
          f"(deck 0.383), MAE={mae:.2f} (deck 0.59), bias={bias:+.2f} (deck -0.31)")


if __name__ == "__main__":
    demo()
