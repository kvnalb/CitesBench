"""
Tag each outlier paper with 1-3 rejection reason labels using Claude.
Writes a `rejection_tags` column back to outputs/outlier_reviews.csv.

Tags: novelty, empirics, clarity, reproducibility, baselines,
      soundness, significance, framing, related_work
"""
import os
import pandas as pd
import anthropic

TAGS = ["novelty", "empirics", "clarity", "reproducibility",
        "baselines", "soundness", "significance", "framing", "related_work"]

SYSTEM = f"""You classify academic paper rejection reasons.
Given the reviewer weaknesses and program chair decision note for a rejected paper,
return 1-3 comma-separated tags from this fixed list (no other words):
{', '.join(TAGS)}

Pick only the tags that are primary drivers of rejection. Order them by importance."""

client = anthropic.Anthropic()
df = pd.read_csv("outputs/outlier_reviews.csv")

# skip if already tagged
if "rejection_tags" not in df.columns:
    df["rejection_tags"] = None

todo = df[df["rejection_tags"].isna()].index
print(f"Tagging {len(todo)} papers...")

for i, idx in enumerate(todo):
    row = df.loc[idx]
    text = f"WEAKNESSES:\n{row.get('weaknesses','')}\n\nPC DECISION:\n{row.get('pc_decision_note','')}"
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        system=SYSTEM,
        messages=[{"role": "user", "content": text[:4000]}],
    )
    tags = resp.content[0].text.strip()
    df.at[idx, "rejection_tags"] = tags
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(todo)}")
        df.to_csv("outputs/outlier_reviews.csv", index=False)  # checkpoint

df.to_csv("outputs/outlier_reviews.csv", index=False)
print("Done. Tag distribution:")
print(df["rejection_tags"].str.split(", ").explode().value_counts().to_string())
