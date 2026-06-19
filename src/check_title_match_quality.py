"""
For papers matched via title search (no DOI in citations CSV), fetch the
OpenAlex display_name and compare it to the OpenReview title.

Streams results to disk as it goes (resumable).
Pauses for 60s every 950 requests to avoid OpenAlex's 1000/min limit.
"""
import os, time, urllib.request, json
import pandas as pd
from difflib import SequenceMatcher

os.makedirs("outputs", exist_ok=True)
OUT = "outputs/title_match_quality.csv"
EMAIL = open("OpenAlex.txt").read().strip()

cites  = pd.read_csv("output/citations_2018_2020.csv")
found  = cites[cites.status == "found"].copy()
no_doi = found[found.doi.isna()][["paper_id", "title", "openalex_id"]].copy()
print(f"No-DOI matches to check: {len(no_doi)}")

# resume: skip already-fetched rows
if os.path.exists(OUT):
    done_ids = set(pd.read_csv(OUT)["paper_id"])
    no_doi = no_doi[~no_doi.paper_id.isin(done_ids)]
    print(f"Resuming — {len(done_ids)} already done, {len(no_doi)} remaining")

def fetch_display_name(openalex_id: str) -> str | None:
    work_id = openalex_id.replace("https://openalex.org/", "")
    url = f"https://api.openalex.org/works/{work_id}?select=id,display_name&mailto={EMAIL}"
    req = urllib.request.Request(url, headers={"User-Agent": f"research/1.0 ({EMAIL})"})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return data.get("display_name")
    except Exception:
        return None

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

write_header = not os.path.exists(OUT)
with open(OUT, "a") as f:
    if write_header:
        f.write("paper_id,title,openalex_id,openalex_title,sim\n")

    for i, row in enumerate(no_doi.itertuples(), 1):
        # pause before hitting the 1000/min ceiling
        if i % 950 == 0:
            print(f"  {i} requests — pausing 65s to reset rate limit...")
            time.sleep(65)

        oa_title = fetch_display_name(row.openalex_id)
        s = sim(row.title, oa_title) if oa_title else float("nan")

        safe = lambda t: '"' + str(t).replace('"', '""') + '"'
        s_str = f"{s:.4f}" if oa_title else ""
        f.write(f"{row.paper_id},{safe(row.title)},{row.openalex_id},"
                f"{safe(oa_title) if oa_title else ''},{s_str}\n")
        f.flush()

        time.sleep(0.12)
        if i % 100 == 0:
            print(f"  {i}/{len(no_doi)}")

# ── report ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT)
s = df["sim"].dropna()
print(f"\nTitle similarity (n={len(s)}):")
print(f"  exact  (=1.00):   {(s==1.0).sum():4d}  ({(s==1.0).mean():.1%})")
print(f"  ≥0.95:            {(s>=0.95).sum():4d}  ({(s>=0.95).mean():.1%})")
print(f"  0.80–0.95:        {((s>=0.80)&(s<0.95)).sum():4d}  ({((s>=0.80)&(s<0.95)).mean():.1%})")
print(f"  <0.80 (suspect):  {(s<0.80).sum():4d}  ({(s<0.80).mean():.1%})")

print("\nWorst matches (sim < 0.80):")
for _, r in df[df.sim < 0.80].sort_values("sim").head(20).iterrows():
    print(f"  {r.sim:.2f}  OR: {str(r.title)[:60]!r}")
    print(f"         OA: {str(r.openalex_title)[:60]!r}")
