import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlite3

os.makedirs("outputs", exist_ok=True)

cites = pd.read_csv("output/citations_2018_2020.csv")
cites = cites[cites.status == "found"]

con = sqlite3.connect("data/gen_review.db")
sub = pd.read_sql(
    "SELECT id, when_submitted as year, decision FROM SUBMISSION WHERE when_submitted IN (2018,2019,2020)", con
)
con.close()
sub["accepted"] = sub["decision"].str.lower().str.contains("accept", na=False)

df = sub.merge(cites[["paper_id", "openalex_citations"]], left_on="id", right_on="paper_id", how="inner")
totals = sub.groupby(["year", "accepted"]).size()

bins = np.logspace(0, np.log10(df.openalex_citations.max() + 1), 40)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, year in zip(axes, [2018, 2019, 2020]):
    sub_yr = df[df.year == year]
    for label, mask, color, acc in [
        ("Accepted", sub_yr.accepted, "steelblue", True),
        ("Rejected", ~sub_yr.accepted, "tomato", False),
    ]:
        n_found = mask.sum()
        n_total = totals.get((year, acc), 0)
        ax.hist(sub_yr[mask].openalex_citations, bins=bins, alpha=0.6,
                label=f"{label} ({n_found}/{n_total})", color=color)
    ax.set_xscale("log")
    ax.set_title(f"{year}  (n={len(sub_yr)})")
    ax.set_xlabel("Citations (log scale)")
    ax.set_ylabel("Papers")
    ax.legend()

plt.tight_layout()
fig.text(
    0.5, -0.02,
    "Parentheses: papers with OpenAlex citation data (DOI + title search) / total submissions. "
    "Coverage: accepts 88.8% · rejects 62.6%.",
    ha="center", fontsize=9, color="gray",
)
plt.savefig("outputs/cite_hist.png", dpi=150, bbox_inches="tight")
print("saved outputs/cite_hist.png")
