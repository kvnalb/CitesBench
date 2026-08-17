"""
The one citation table. Everything downstream reads this and nothing else.

Before this file there were three definitions of the outcome variable and no
canonical one: build_eval_table read OpenAlex with `status=='found'`, the
dashboard reconstructed the column in-app from S2 v1 with a 0.9 similarity gate,
and compare_eras used S2 v2 tiers. The dashboard computed metrics live rather
than reading eval_results.csv, so two different answers to "what does regime X
score" already existed and nothing detected the disagreement.

Why S2 and not OpenAlex, in one number: OpenAlex citation coverage is 89.0% for
accepted papers and 62.7% for rejected — a 26.3 point differential. For a paper
whose claim is "this regime selects better from a pool of accepts and rejects,"
that is disqualifying: a regime reaching into the reject pile is punished by
measurement rather than by quality, which flatters the human baseline. S2 v2
runs 0.8 points. That single comparison is the reason this file exists.

Tiers come from the fetcher's own assign_tiers, imported rather than
reimplemented, so the rule cannot drift between the fetch and the analysis:
  A  arXiv-ID or DOI match
  B  title_sim >= 0.95, year within [-1,+3], >=1 shared author surname
  C  weaker — carried but excluded from the primary outcome by default
Unmatched papers are DROPPED, never imputed as zero. A paper we could not find
is not a paper with no citations.

`source` is a pinned string, not a guess. It reads s2_api_<window> today; when
the S2 bulk release lands (#11) it becomes s2_bulk_<release_id> and nothing
downstream changes. The window is honest about what we have: the API pull was
made incrementally and carries no per-row timestamp, so we record the span
rather than pretending to a single instant.

Run: python src/build/build_citations.py [--include-tier-c]
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fetch"))
from fetch_citations_s2_v2 import assign_tiers

OUT_CSV = "outputs/citations.csv"

# (raw pull, eval table it was fetched against, era label). The *_authored file is
# preferred where it exists — it carries the backfilled author_overlap.
SOURCES = [
    ("outputs/s2_citations_v2_authored.csv", "outputs/s2_citations_v2.csv",
     "outputs/eval_table.csv", "2018-2020"),
    ("outputs/s2_citations_2025_authored.csv", "outputs/s2_citations_2025.csv",
     "outputs/eval_table_2025.csv", "2025"),
]
COLS = ["paper_id", "year", "citations", "influential", "source", "tier",
        "matched_by", "title_sim", "author_overlap", "s2_paper_id", "s2_corpus_id",
        "fetched_window"]


def fetch_window(paths):
    """Honest provenance: the span the pull was made over, from git history where
    available, else file mtimes. Not a single instant, because it was not one."""
    dates = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            out = subprocess.run(["git", "log", "--follow", "--format=%ad",
                                  "--date=short", "--", p],
                                 capture_output=True, text=True, timeout=20).stdout.split()
            dates += out
        except Exception:
            pass
        dates.append(pd.Timestamp(os.path.getmtime(p), unit="s").strftime("%Y-%m-%d"))
    if not dates:
        return "unknown"
    lo, hi = min(dates), max(dates)
    return lo if lo == hi else f"{lo}..{hi}"


def load_era(preferred, fallback, eval_table, era):
    path = preferred if os.path.exists(preferred) else fallback
    if not os.path.exists(path):
        print(f"  {era}: SKIPPED — neither {preferred} nor {fallback} exists")
        return None, path
    raw = pd.read_csv(path, low_memory=False)
    ev = pd.read_csv(eval_table, low_memory=False)[["paper_id", "year", "decision"]]
    tiered = assign_tiers(raw, ev)
    print(f"  {era}: {path}  {len(tiered):,} rows  "
          f"tiers {tiered['tier'].value_counts().to_dict()}")
    return tiered, path


def build(out_csv=OUT_CSV, include_c=False):
    keep = ["A", "B", "C"] if include_c else ["A", "B"]
    frames, paths = [], []
    for preferred, fallback, ev, era in SOURCES:
        t, used = load_era(preferred, fallback, ev, era)
        paths.append(used)
        if t is None:
            continue
        d = t[t["tier"].isin(keep) & t["s2_citations"].notna()].copy()
        d["era"] = era
        frames.append(d)

    if not frames:
        sys.exit("no source files found — run the fetcher first")

    all_ = pd.concat(frames, ignore_index=True)
    win = fetch_window([p for p in paths if p])

    out = pd.DataFrame({
        "paper_id": all_["paper_id"],
        "year": all_["year"],
        "citations": all_["s2_citations"].astype(float),
        "influential": pd.to_numeric(all_.get("s2_influential"), errors="coerce"),
        "source": f"s2_api_{win}",
        "tier": all_["tier"],
        "matched_by": all_["query_method"],
        "title_sim": pd.to_numeric(all_["title_sim"], errors="coerce"),
        "author_overlap": pd.to_numeric(all_["author_overlap"], errors="coerce"),
        "s2_paper_id": all_["s2_paper_id"],
        "s2_corpus_id": all_["s2_corpus_id"],
        "fetched_window": win,
    })[COLS].drop_duplicates("paper_id")

    os.makedirs("outputs", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\n{len(out):,} papers -> {out_csv}   source={out['source'].iloc[0]}")
    return out


def report(out):
    """Coverage by decision is the number this file exists to protect.

    Read the two eras differently. 2018-2020 is the primary sample and both sides
    must be complete — a differential there is a defect. 2025 was fetched
    accepted-only on purpose (it is an appendix robustness arm, and the committee
    was only ever run on accepted 2025 papers), so its reject coverage is whatever
    the earlier ID-only pass happened to catch and a large differential there is
    scope, not breakage."""
    for ev_path, era, note in [
            ("outputs/eval_table.csv", "2018-2020", "primary sample — both sides must be complete"),
            ("outputs/eval_table_2025.csv", "2025", "accepted-only by design; reject figure is not a defect")]:
        if not os.path.exists(ev_path):
            continue
        ev = pd.read_csv(ev_path, low_memory=False)
        ev = ev[ev["decision"].notna()]
        ev["acc"] = ev["decision"].str.startswith("Accept", na=False)
        ev["has"] = ev["paper_id"].isin(set(out["paper_id"]))
        g = ev.groupby("acc")["has"].mean()
        if len(g) == 2:
            print(f"  {era}: accepted {g[True]:.1%}  rejected {g[False]:.1%}  "
                  f"differential {abs(g[True]-g[False])*100:.1f} pp")
            print(f"    ({note})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--include-tier-c", action="store_true",
                    help="carry weak matches too (sensitivity arm; off by default)")
    a = ap.parse_args()
    df = build(a.out, a.include_tier_c)
    assert df.paper_id.is_unique
    assert df.citations.notna().all(), "unmatched papers must be dropped, not imputed"
    print("\ncoverage by decision:")
    report(df)
