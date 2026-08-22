"""
Strip plot: one dot per paper-tag pair, x = citations (log), y = rejection tag.
Papers with multiple tags appear in each relevant row.
Top papers labelled. Accept p75 threshold shown as a reference band.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("outputs/outlier_reviews.csv")
df = df[df["rejection_tags"].notna()].copy()

# explode to one row per tag
df["tags_list"] = df["rejection_tags"].str.split(", ")
exploded = df.explode("tags_list").rename(columns={"tags_list": "tag"})

TAG_ORDER = ["novelty", "empirics", "significance", "soundness",
             "baselines", "clarity", "related_work", "framing", "reproducibility"]
TAG_COLORS = {
    "novelty":         "#4e79a7",
    "empirics":        "#f28e2b",
    "significance":    "#e15759",
    "soundness":       "#76b7b2",
    "baselines":       "#59a14f",
    "clarity":         "#edc948",
    "related_work":    "#b07aa1",
    "framing":         "#9c755f",
    "reproducibility": "#ff9da7",
}

exploded = exploded[exploded.tag.isin(TAG_ORDER)].copy()
exploded["y"] = exploded["tag"].map({t: i for i, t in enumerate(TAG_ORDER)})

# jitter within each row
rng = np.random.default_rng(42)
exploded["y_jit"] = exploded["y"] + rng.uniform(-0.35, 0.35, len(exploded))

# accept p75 per year — use median across years as reference line
p75_median = df.groupby("year")["accept_p75"].first().median()

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

# per-year accept p75 thresholds
year_p75 = df.groupby("year")["accept_p75"].first()
year_colors = {2018: "#999999", 2019: "#bbbbbb", 2020: "#cccccc"}
for year, p75 in year_p75.items():
    ax.axvline(p75, color=year_colors[year], lw=1, ls="--", zorder=1, alpha=0.8)
    ax.text(p75, len(TAG_ORDER) - 0.1, f"{year}\n({p75:.0f})",
            ha="center", va="top", fontsize=6.5, color="#777")

# dots
for tag in TAG_ORDER:
    sub = exploded[exploded.tag == tag]
    ax.scatter(sub.s2_citations, sub.y_jit,
               color=TAG_COLORS[tag], alpha=0.7, s=30, linewidths=0,
               zorder=2)

# label top 10 papers, stacked vertically on the right side
top = df.nlargest(10, "s2_citations").sort_values("s2_citations", ascending=False)
x_label = df.s2_citations.max() * 1.6
y_positions = np.linspace(len(TAG_ORDER) - 0.5, -0.5, len(top))

for (_, row), y_text in zip(top.iterrows(), y_positions):
    primary_tag = row["rejection_tags"].split(", ")[0]
    if primary_tag not in TAG_ORDER:
        continue
    y_dot = TAG_ORDER.index(primary_tag)
    short = row["title"][:44] + ("…" if len(row["title"]) > 44 else "")
    ax.text(x_label, y_text,
            f"{short} ({int(row['s2_citations'])})",
            fontsize=6.5, color="#333", va="center")

ax.set_xscale("log")
ax.set_xlim(left=df.s2_citations.min() * 0.7,
            right=df.s2_citations.max() * 12)
ax.set_yticks(range(len(TAG_ORDER)))
ax.set_yticklabels(TAG_ORDER, fontsize=11)
ax.set_xlabel("Citations (log scale)", fontsize=11)
ax.set_title("Highly-cited rejected papers: citations vs. stated rejection reason",
             fontsize=13, fontweight="bold", pad=12)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color="#e0e0e0", lw=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=8, framealpha=0.6, loc="lower right",
          title="Dashed lines = accept p75\nthreshold per year", title_fontsize=7)

fig.text(
    0.5, -0.01,
    "Each dot = one paper. Papers with multiple rejection tags appear in each relevant row. "
    "Dashed lines = accept p75 citation threshold per year (outlier criterion).",
    ha="center", fontsize=8.5, color="#666",
)

plt.tight_layout()
plt.savefig("outputs/outlier_scatter.png", dpi=150, bbox_inches="tight", facecolor="#fafafa")
print("saved outputs/outlier_scatter.png")
