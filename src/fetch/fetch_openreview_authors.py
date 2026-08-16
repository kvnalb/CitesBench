"""
Author names for ICLR 2018 and 2019 submissions, from OpenReview.

Why only those two years: S2 title matches are verified by shared-author surname,
and the author lists come from ReviewArena (2020 onward, and complete — it is a
full venue dump, so it carries rejected submissions too) plus OpenAlex and arXiv.
2018 and 2019 predate ReviewArena, so they fall back to OpenAlex, whose coverage
of non-arXiv rejected papers is ~63%. The result was a tier-B pass rate of 58-63%
on the 2018/2019 title-match tail against 99% for 2020 — and since that tail is
disproportionately rejected papers, the shortfall would reintroduce differential
attrition at the tiering gate, right after we removed it at the fetch gate.

ICLR 2018/2019 live on the v1 API (api.openreview.net), not api2. Submissions are
"Blind_Submission" but are de-anonymized after decisions, so the author lists are
present and complete, rejects included.

Output schema matches outputs/paper_author_names_reviewarena.csv so load_inputs
treats the two alike.

Run: python src/fetch/fetch_openreview_authors.py
"""
import argparse
import os
import sys

import pandas as pd
from dotenv import dotenv_values

OUT_CSV = "outputs/paper_author_names_openreview.csv"
BASEURL = "https://api.openreview.net"          # v1: 2018/2019 are not on api2
INVITATION = "ICLR.cc/{year}/Conference/-/Blind_Submission"
YEARS = (2018, 2019)


def client():
    import openreview
    env = dotenv_values(".env")
    u = env.get("OPENREVIEW_USERNAME") or os.environ.get("OPENREVIEW_USERNAME")
    p = env.get("OPENREVIEW_PASSWORD") or os.environ.get("OPENREVIEW_PASSWORD")
    if not (u and p):
        sys.exit("ERROR: OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD not set in .env")
    return openreview.Client(baseurl=BASEURL, username=u, password=p)


def fetch_year(c, year, page=1000):
    import openreview
    inv = INVITATION.format(year=year)
    notes = list(openreview.tools.iterget_notes(c, invitation=inv))
    rows = []
    for n in notes:
        for name in (n.content.get("authors") or []):
            name = str(name).strip()
            if name and name.lower() != "anonymous":
                rows.append((n.id, year, name))
    print(f"  {year}: {len(notes):,} submissions -> {len(rows):,} author rows")
    return rows


def build(out_csv=OUT_CSV, years=YEARS):
    os.makedirs("outputs", exist_ok=True)
    c = client()
    rows = []
    for y in years:
        rows += fetch_year(c, y)
    out = pd.DataFrame(rows, columns=["paper_id", "year", "author_name"]).drop_duplicates()
    out.to_csv(out_csv, index=False)
    print(f"{len(out):,} rows for {out.paper_id.nunique():,} papers -> {out_csv}")
    return out


def report(out):
    """Coverage against our corpus, split by decision — the number that matters."""
    ev = pd.read_csv("outputs/eval_table.csv", low_memory=False)
    ev = ev[ev["year"].isin(YEARS)]
    ev["has"] = ev["paper_id"].isin(set(out["paper_id"]))
    ev["accepted"] = ev["decision"].str.startswith("Accept", na=False)
    print("\ncoverage of our 2018/2019 corpus:")
    print(ev.groupby(["year", "accepted"])["has"].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()
    df = build(a.out)
    assert df.paper_id.nunique() > 1000, "suspiciously few papers"
    assert not df.author_name.str.contains("Anonymous", case=False).any()
    report(df)
