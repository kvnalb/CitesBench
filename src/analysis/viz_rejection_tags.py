import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("outputs/outlier_reviews.csv")

tag_counts = df["rejection_tags"].str.split(", ").explode().value_counts()

TAG_META = {
    "novelty":         ("#4e79a7", "Too incremental; field considered it\nan insufficient advance over prior work"),
    "empirics":        ("#f28e2b", "Weak evaluation: missing ablations,\nsmall datasets, or unconvincing results"),
    "significance":    ("#e15759", "Contribution judged too narrow\nor its impact unclear"),
    "soundness":       ("#76b7b2", "Theoretical gaps, unsubstantiated\nclaims, or logical flaws"),
    "baselines":       ("#59a14f", "Missing or unfair comparisons\nto prior methods"),
    "clarity":         ("#edc948", "Presentation too unclear to\nassess the contribution"),
    "related_work":    ("#b07aa1", "Insufficient engagement with or\ncomparison to related work"),
    "significance":    ("#e15759", "Contribution judged too narrow\nor its impact unclear"),
    "reproducibility": ("#ff9da7", "Missing details needed to\nreproduce the work"),
    "framing":         ("#9c755f", "Wrong venue or scope mismatch\n(e.g. software, survey at ICLR)"),
}

tags_ordered = tag_counts.index.tolist()
colors = [TAG_META.get(t, ("#aaaaaa", ""))[0] for t in tags_ordered]
counts = tag_counts.values

fig = plt.figure(figsize=(13, 8))
fig.patch.set_facecolor("#fafafa")
gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1], wspace=0.06)

ax = fig.add_subplot(gs[0])
bars = ax.barh(tags_ordered[::-1], counts[::-1], color=colors[::-1],
               edgecolor="white", linewidth=0.8, height=0.65)

for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
            str(count), va="center", ha="left", fontsize=11, color="#333333")

ax.set_xlabel("Papers tagged (163 outliers, multi-tag)", fontsize=11, color="#444")
ax.set_xlim(0, max(counts) * 1.15)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=11)
ax.tick_params(axis="x", labelsize=9, color="#aaa")
ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.6)
ax.set_axisbelow(True)
ax.set_facecolor("#fafafa")

title = "Why did ICLR reject papers that became highly cited?"
subtitle = (
    "Outlier criterion: rejected papers with citations > 75th pct of accepted papers from same year\n"
    "Funnel: 3,041 rejects 2018–2020  →  1,904 with citation data (62.6%)  →  151 outliers (5.0% of rejects / 7.9% of those with data)"
)
fig.text(0.03, 0.97, title, fontsize=14, fontweight="bold", color="#111", va="top")
fig.text(0.03, 0.91, subtitle, fontsize=8.5, color="#666", va="top", linespacing=1.6)

ax_leg = fig.add_subplot(gs[1])
ax_leg.axis("off")
ax_leg.set_facecolor("#fafafa")

legend_title = ax_leg.text(0.05, 0.97, "Tag definitions", fontsize=10,
                            fontweight="bold", transform=ax_leg.transAxes,
                            va="top", color="#222")

y = 0.89
for tag in tags_ordered:
    color, desc = TAG_META.get(tag, ("#aaaaaa", tag))
    patch = mpatches.FancyBboxPatch((0.03, y - 0.025), 0.06, 0.045,
                                     boxstyle="round,pad=0.005",
                                     facecolor=color, edgecolor="none",
                                     transform=ax_leg.transAxes)
    ax_leg.add_patch(patch)
    ax_leg.text(0.13, y, f"$\\bf{{{tag}}}$\n{desc}",
                transform=ax_leg.transAxes, va="center",
                fontsize=7.8, color="#333", linespacing=1.45)
    y -= 0.105

out = "outputs/rejection_tags.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fafafa")
print(f"saved {out}")
