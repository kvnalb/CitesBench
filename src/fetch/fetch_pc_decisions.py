"""
Fetch Program Chair decision notes from OpenReview API for the top N outlier papers
and append them to outlier_reviews.csv.
"""
import os
import re
import time
import urllib.request
import json
import pandas as pd

os.makedirs("outputs", exist_ok=True)

EMAIL = open("OpenAlex.txt").read().strip()

df = pd.read_csv("outputs/outlier_reviews.csv")

# pull forum id from pdf URL: https://openreview.net/pdf?id=HJjvxl-Cb -> HJjvxl-Cb
import sqlite3
con = sqlite3.connect("data/gen_review.db")
subs = pd.read_sql("SELECT title, pdf FROM SUBMISSION", con)
con.close()
df = df.merge(subs, on="title", how="left")
df["forum_id"] = df["pdf"].str.extract(r"id=([^&]+)")

def fetch_decision(forum_id):
    url = f"https://api.openreview.net/notes?forum={forum_id}&trash=true"
    req = urllib.request.Request(url, headers={"User-Agent": f"research/1.0 ({EMAIL})"})
    for attempt in range(4):
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt * 5
                print(f"    rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"[fetch error: {e}]"
        except Exception as e:
            return f"[fetch error: {e}]"
    else:
        return "[fetch error: rate limit exceeded after retries]"
    for note in data.get("notes", []):
        inv = note.get("invitation", "")
        if any(k in inv for k in ("Decision", "Meta_Review", "Program_Chairs")):
            c = note.get("content", {})
            parts = [c.get("decision", ""), c.get("comment", ""), c.get("metareview", "")]
            return "\n".join(p for p in parts if p).strip()
    return "[no decision note found]"

todo = df[df["pc_decision_note"].str.startswith("[fetch error", na=False)].copy()
print(f"Fetching PC decisions for {len(todo)} remaining outliers...")
for idx, row in todo.iterrows():
    fid = row["forum_id"]
    result = fetch_decision(fid)
    df.at[idx, "pc_decision_note"] = result
    print(f"  {row['title'][:60]}... → {result[:60]}")
    time.sleep(2)

df = df.drop(columns=["pdf", "forum_id"])
df.to_csv("outputs/outlier_reviews.csv", index=False)
print(f"\nUpdated outputs/outlier_reviews.csv with pc_decision_note column.")
