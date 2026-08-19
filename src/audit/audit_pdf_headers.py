"""
Does the first line of the extracted paper text give away the decision?

OpenReview PDFs carry a running header. A submission under review reads

    Under review as a conference paper at ICLR 2018

and the camera-ready version of an accepted paper reads

    Published as a conference paper at ICLR 2018

If the archive's extracted text preserves that header, then every model that read
the text — the 9-call council, the single-call baseline, any future regime — could
read the outcome off line 1 instead of judging the paper. That is not subtle
memorisation; it is the label printed at the top of the input.

This script does not fix anything. It measures how far the problem goes: how many
papers carry a header, whether the header agrees with the recorded decision, and
what a regime would score by doing nothing but pattern-matching that one line.

The regexes are case-insensitive and tolerant of runs of whitespace, because the
text came out of a PDF extractor and line-wrapping is not guaranteed. The header is
looked for on the first non-empty line, and separately anywhere in the first
SCAN_LINES lines — a title page that put the title first would otherwise read as
"no header" when the giveaway is still one line down.

Two text sources, because the leak is a property of the PDFs rather than of one
extractor, and showing that requires checking both:

    --source archive       data/OpenReview/*/fulltext/<paper_id>.txt, 2018-2020
    --source reviewarena   the ReviewArena parquet, 2020 and 2025

2020 appears in both, which makes it a replication rather than a second opinion.

Run: python src/audit/audit_pdf_headers.py
     python src/audit/audit_pdf_headers.py --years 2018 2019 2020
     python src/audit/audit_pdf_headers.py --source reviewarena --years 2020 2025
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_TABLE = "outputs/eval_table.csv"
TEXT_DIRS = [
    "data/OpenReview/rdd_bandwidth_2018_2020__gemma4_dedicated_stage1/fulltext",
    "data/OpenReview/full_2018_2020_remaining/fulltext",
]
OUT_CSV = "outputs/audits/pdf_header_leakage.csv"
OUT_MD = "outputs/audits/pdf_header_leakage.md"
OUT_CSV_RA = "outputs/audits/pdf_header_leakage_reviewarena.csv"
OUT_MD_RA = "outputs/audits/pdf_header_leakage_reviewarena.md"

SCAN_LINES = 5

PUBLISHED = re.compile(r"published\s+as\s+a\s+conference\s+paper\s+at\s+iclr\s*(\d{4})?", re.I)
UNDER_REVIEW = re.compile(r"under\s+review\s+as\s+a\s+conference\s+paper\s+at\s+iclr\s*(\d{4})?", re.I)
# accepted-but-not-poster variants exist in the wild; count them rather than
# silently filing them under "other"
WORKSHOP = re.compile(r"accepted\s+as\s+a\s+workshop\s+(paper|contribution)\s+at\s+iclr", re.I)


def text_index(dirs):
    """paper_id -> path, largest file wins when a paper appears in several dirs."""
    idx = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.txt")):
            pid = os.path.basename(p)[:-4]
            if os.path.getsize(p) > idx.get(pid, ("", -1))[1]:
                idx[pid] = (p, os.path.getsize(p))
    return {k: v[0] for k, v in idx.items()}


def classify(line):
    if PUBLISHED.search(line):
        return "published"
    if UNDER_REVIEW.search(line):
        return "under_review"
    if WORKSHOP.search(line):
        return "workshop"
    return "other"


def _head_lines(path):
    lines = []
    with open(path, errors="replace") as f:
        for _ in range(SCAN_LINES * 4):          # blank lines are cheap, read past them
            ln = f.readline()
            if not ln:
                break
            if ln.strip():
                lines.append(ln.strip())
            if len(lines) >= SCAN_LINES:
                break
    return lines


def heads_archive(years):
    """paper_id -> first SCAN_LINES non-empty lines, from the archive .txt files."""
    return {pid: _head_lines(path) for pid, path in text_index(TEXT_DIRS).items()}


def heads_reviewarena(years):
    """Same, from the ReviewArena parquet. Its `markdown` column is OCR'd PDF, so the
    running header survives there too if it survived the PDF."""
    import glob as _glob
    files = sorted(_glob.glob("data/ReviewArena/raw/data/*.parquet"))
    if not files:
        sys.exit("no ReviewArena parquet found")
    d = pd.concat([pd.read_parquet(f, columns=["forum_id", "year", "markdown"])
                   for f in files], ignore_index=True)
    d = d[d.year.isin(years) & d.markdown.notna()]
    out = {}
    for pid, md in zip(d.forum_id, d.markdown):
        out[pid] = [l.strip() for l in str(md).split("\n") if l.strip()][:SCAN_LINES]
    return out


def head_verdict(lines):
    """(first non-empty line, its class, best class within SCAN_LINES lines)."""
    if not lines:
        return "", "empty", "empty"
    first = lines[0]
    within = next((c for c in (classify(l) for l in lines) if c != "other"), "other")
    return first, classify(first), within


def _decisions(years, source):
    """Papers and decisions. eval_table only covers 2018-2020, so ReviewArena's own
    decision column carries the later years."""
    ev = pd.read_csv(EVAL_TABLE, low_memory=False)
    ev = ev[ev.year.isin(years)][["paper_id", "year", "decision"]]
    missing = [y for y in years if y not in set(ev.year)]
    if missing and source == "reviewarena":
        import glob as _glob
        files = sorted(_glob.glob("data/ReviewArena/raw/data/*.parquet"))
        d = pd.concat([pd.read_parquet(f, columns=["forum_id", "year", "decision"])
                       for f in files], ignore_index=True)
        d = d[d.year.isin(missing)].rename(columns={"forum_id": "paper_id"})
        ev = pd.concat([ev, d[["paper_id", "year", "decision"]]], ignore_index=True)
    ev = ev.copy()
    ev["accepted"] = ev.decision.astype(str).str.startswith("Accept")
    return ev


def build(years, source="archive", out_csv=None, out_md=None):
    os.makedirs("outputs/audits", exist_ok=True)
    out_csv = out_csv or (OUT_CSV if source == "archive" else OUT_CSV_RA)
    out_md = out_md or (OUT_MD if source == "archive" else OUT_MD_RA)
    ev = _decisions(years, source)
    heads = (heads_archive if source == "archive" else heads_reviewarena)(years)

    rows = []
    for r in ev.itertuples(index=False):
        lines = heads.get(r.paper_id)
        if lines is None:
            rows.append({"paper_id": r.paper_id, "year": r.year, "decision": r.decision,
                         "accepted": r.accepted, "first_line": "", "header": "no_text",
                         "header_in_head": "no_text"})
            continue
        first, head, within = head_verdict(lines)
        rows.append({"paper_id": r.paper_id, "year": r.year, "decision": r.decision,
                     "accepted": r.accepted, "first_line": first[:200],
                     "header": head, "header_in_head": within})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    have = df[df.header != "no_text"]
    L = ["# Does the first line give away the decision?", "",
         f"Source: **{source}**. Years {', '.join(map(str, years))}. "
         f"{len(have):,} of {len(df):,} papers have extracted text.", "",
         "Generated by `python src/audit/audit_pdf_headers.py` — do not hand-edit.", ""]

    for yr in years:
        d = have[have.year == yr]
        if d.empty:
            continue
        ct = pd.crosstab(d.accepted.map({True: "accepted", False: "rejected"}), d.header)
        L += [f"## {yr}  (n = {len(d):,})", "", "First line:", "",
              ct.to_markdown(), ""]
        pub = d[d.header == "published"]
        und = d[d.header == "under_review"]
        if len(pub) or len(und):
            # what a regime scores by reading line 1 and nothing else
            tp = int(pub.accepted.sum()); fp = int((~pub.accepted).sum())
            tn = int((~und.accepted).sum()); fn = int(und.accepted.sum())
            tot = tp + fp + tn + fn
            acc = (tp + tn) / tot if tot else float("nan")
            L += [f"- `published` header: {len(pub):,} papers, "
                  f"**{pub.accepted.mean():.1%} accepted**",
                  f"- `under review` header: {len(und):,} papers, "
                  f"**{und.accepted.mean():.1%} accepted**",
                  f"- predicting accept from the header alone: **{acc:.1%} accurate** "
                  f"on the {tot:,} papers that carry one", ""]
        if (d.header != d.header_in_head).any():
            n = int((d.header != d.header_in_head).sum())
            L += [f"- {n:,} papers whose first line is not the header but which carry "
                  f"one within {SCAN_LINES} lines", ""]
    open(out_md, "w").write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"-> {out_csv}\n-> {out_md}")
    return df


def demo():
    df = build([2018])
    have = df[df.header != "no_text"]
    assert len(have) > 500, f"expected most 2018 papers to have text, got {len(have)}"
    # the example from the ticket, checked by hand: accepted 2018, camera-ready header
    row = df[df.paper_id == "B1ae1lZRb"]
    if len(row):
        assert row.header.iloc[0] == "published", row.header.iloc[0]
        assert bool(row.accepted.iloc[0])
    assert set(df.header) <= {"published", "under_review", "workshop", "other",
                              "empty", "no_text"}
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2018])
    ap.add_argument("--source", choices=["archive", "reviewarena"], default="archive")
    a = ap.parse_args()
    demo() if len(sys.argv) == 1 else build(a.years, a.source)
