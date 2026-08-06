"""
Replace the placebo probe titles with real arXiv titles that were never ICLR submissions.

The first placebo design recombined halves of real ICLR titles from the same arm. It failed
in both directions: the midpoint split cuts mid-phrase, producing ungrammatical titles that
a model may decline for being malformed rather than unrecognised (deflating the fabrication
floor); and distinctive prefixes survive intact ("DeCo:", "STRAP:"), so a model may
recognise the real paper the head came from and answer from genuine recall (inflating it).

v2 uses real, coherent arXiv titles from the same categories and year window, screened
against every title in gen_review.db (all 32,652 submissions, 2018-2025) so none was ever
an OpenReview submission under that title. A model asserting an ICLR accept/reject outcome
for one of these is inventing a venue for a paper it may well know — a cleaner failure to
interpret than inventing a paper.

Known residual: some of these papers may have been ICLR submissions under a *different*
title, and title-level screening cannot catch that. Rate is bounded by the retitling
analysis (outputs/arxiv_fuzzy_report.md) at roughly a fifth of unmatched submissions.

Only the placebo rows of the probe plan change; lap / fame / wrongyear stay byte-identical,
so only the placebo calls need re-running.

Run: python src/rebuild_placebo.py
"""
import os
import re
import glob
import sqlite3

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PLAN = "outputs/samples/oos_probe_plan.csv"
PAPERS = "outputs/samples/oos_papers.csv"
DUMP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--librarian-bots--arxiv-metadata-snapshot/"
    "snapshots/*/data/*.parquet")
CATS = ("cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML", "cs.NE")
SEED = 42


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


def year_of(versions, update_date):
    try:
        if isinstance(versions, (list, tuple)) and versions and isinstance(versions[0], dict):
            m = re.search(r"\b(19|20)\d{2}\b", str(versions[0].get("created")))
            if m:
                return int(m.group(0))
    except Exception:
        pass
    m = re.search(r"\b(19|20)\d{2}\b", str(update_date or ""))
    return int(m.group(0)) if m else None


def main():
    rng = np.random.default_rng(SEED)
    plan = pd.read_csv(PLAN)
    papers = pd.read_csv(PAPERS)

    con = sqlite3.connect("data/gen_review.db")
    banned = set(norm(t) for t in pd.read_sql("SELECT title FROM SUBMISSION", con)["title"])
    con.close()
    print(f"screening against {len(banned):,} OpenReview titles")

    # year window per arm, so a placebo title is contemporaneous with the arm it lands in
    windows = {a: (int(g["year"].min()) - 2, int(g["year"].max())) for a, g in papers.groupby("arm")}
    pool = {a: [] for a in windows}
    need = {a: int((plan["arm"].eq(a) & plan["probe"].eq("placebo")).sum()) for a in windows}
    target = {a: need[a] * 12 for a in windows}          # oversample, then subsample

    for path in sorted(glob.glob(DUMP_GLOB)):
        pf = pq.ParquetFile(path)
        cols = [c for c in ["title", "categories", "versions", "update_date"] if c in pf.schema.names]
        for batch in pf.iter_batches(batch_size=50_000, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d["title"])):
                cats = str(d["categories"][i] or "")
                if not any(c in cats for c in CATS):
                    continue
                t = str(d["title"][i] or "").strip()
                w = t.split()
                if not (6 <= len(w) <= 20) or norm(t) in banned:
                    continue
                y = year_of(d.get("versions", [None] * len(d["title"]))[i],
                            d.get("update_date", [None] * len(d["title"]))[i])
                if y is None:
                    continue
                for a, (lo, hi) in windows.items():
                    if lo <= y <= hi and len(pool[a]) < target[a]:
                        pool[a].append(t)
        if all(len(pool[a]) >= target[a] for a in windows):
            break
        print(f"  pool sizes {[f'{a}:{len(v)}' for a, v in pool.items()]} "
              f"after {os.path.basename(path)}", flush=True)

    plan = plan.copy()
    for a in windows:
        cand = list(dict.fromkeys(pool[a]))
        if len(cand) < need[a]:
            raise SystemExit(f"only {len(cand)} candidates for arm {a}, need {need[a]}")
        pick = [cand[i] for i in rng.choice(len(cand), size=need[a], replace=False)]
        mask = plan["arm"].eq(a) & plan["probe"].eq("placebo")
        plan.loc[mask, "probe_title"] = pick
        print(f"  {a}: {need[a]} placebo titles from {len(cand):,} candidates "
              f"(years {windows[a][0]}-{windows[a][1]})")

    # A frozen sample that gets rewritten in place is not frozen. Keep the prior
    # version and stamp the revision so the design doc's claim stays checkable.
    if os.path.exists(PLAN):
        v = 1
        while os.path.exists(f"{PLAN[:-4]}_v{v}.csv"):
            v += 1
        pd.read_csv(PLAN).to_csv(f"{PLAN[:-4]}_v{v}.csv", index=False)
        print(f"  archived prior plan to {PLAN[:-4]}_v{v}.csv")
    plan["placebo_source"] = np.where(plan["probe"].eq("placebo"), "arxiv_real_nonICLR", "")
    plan["placebo_revised_at"] = pd.Timestamp.now(tz="UTC").date().isoformat()
    plan.to_csv(PLAN, index=False)
    print(f"\nRewrote {PLAN} — placebo rows only")
    for r in plan[plan["probe"].eq("placebo")].head(5).itertuples():
        print(f"  [{r.arm}] {r.probe_title[:88]}")


if __name__ == "__main__":
    main()
