import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlite3

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/OpenAlex/openalex_rdd_arxiv_paper_level.csv")
raw["accepted"] = raw["decision"].str.lower().str.contains("accept", na=False)
raw = raw[raw.year.isin([2018, 2019, 2020])]

con = sqlite3.connect("data/gen_review.db")
db = pd.read_sql("SELECT when_submitted as year, decision FROM SUBMISSION WHERE when_submitted IN (2018,2019,2020)", con)
con.close()
db["accepted"] = db["decision"].str.lower().str.contains("accept", na=False)
totals = db.groupby(["year", "accepted"]).size()  # denominator: all submissions in DB that year

df = raw[raw.openalex_cited_by_count.notna()].copy()

bins = np.logspace(0, np.log10(df.openalex_cited_by_count.max() + 1), 40)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, year in zip(axes, [2018, 2019, 2020]):
    sub = df[df.year == year]
    for label, mask, color, acc in [("Accepted", sub.accepted, "steelblue", True), ("Rejected", ~sub.accepted, "tomato", False)]:
        n_found = mask.sum()
        n_total = totals.get((year, acc), 0)
        ax.hist(sub[mask].openalex_cited_by_count, bins=bins, alpha=0.6,
                label=f"{label} ({n_found}/{n_total})", color=color)
    ax.set_xscale("log")
    ax.set_title(f"{year}  (n={len(sub)})")
    ax.set_xlabel("Citations (log scale)")
    ax.set_ylabel("Papers")
    ax.legend()

plt.tight_layout()
fig.text(0.5, -0.02, "Parentheses show (papers with OpenAlex citation data / total submissions) for that decision bucket.",
         ha="center", fontsize=9, color="gray")
plt.savefig("outputs/cite_hist.png", dpi=150, bbox_inches="tight")
print("saved outputs/cite_hist.png")
