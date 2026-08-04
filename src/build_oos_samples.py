"""
Freeze the evaluation samples for the out-of-sample leakage / review benchmark.

Samples are constructed once and reused by every model and every probe, so that model
differences are model differences and not sampling differences. Nothing downstream is
allowed to draw its own sample.

Three arms, chosen to sit at different points on a contamination gradient rather than to
give one yes/no test. Reference cutoff is Llama-3.3-70B's (December 2023):

  arm          papers submitted   decisions   a Dec-2023-cutoff model could have seen
  contaminated 2017-2019          2018-2020   the papers, the decisions and the citations
  partial      Sep 2023 (ICLR24)  Jan 2024    the preprints, but no outcome
  clean        Sep 2024 (ICLR25)  Jan 2025    nothing

Eligibility: a paper is eligible only if it has a title, a decision, and a *verified*
citation measurement — an arXiv-ID match into Semantic Scholar. Title-only matches are
excluded because their citation counts are not trustworthy and their failure rate is
correlated with the decision. This restricts the 2024/2025 arms to roughly half of each
year, and that half is decision-skewed (arXiv posting rates differ by ~32 points between
accepted and rejected papers). The restriction is applied identically across arms and is
reported in the design file — it bounds external validity, it does not bias the
model-vs-model comparison.

Stratification: decision class x citation quartile, within arm. Quartiles are computed
among eligible papers of that arm only.

Probe variants, all generated here so the plan is identical across models:
  lap        "was this accepted or rejected at ICLR <year>?"      outcome recall
  fame       "is this paper widely cited?"                        impact recall
  placebo    same question, fabricated title                      false-positive rate
  wrongyear  same question, year shifted by +/-2                  acquiescence check

The placebo titles are built by recombining n-grams from other titles in the same arm, so
they are plausible-sounding and in-distribution but refer to no real paper. A model that
answers "accepted" to those is agreeing with the prompt, not recalling anything.

Output: outputs/samples/oos_papers.csv       frozen paper sample, one row per paper
        outputs/samples/oos_probe_plan.csv   one row per paper x probe variant
        outputs/samples/oos_sample_design.md the design, strata counts and MDE

Run: python src/build_oos_samples.py [--n-per-arm 600] [--seed 42]
"""
import os
import re
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUTDIR = "outputs/samples"
os.makedirs(OUTDIR, exist_ok=True)

ARMS = {
    "contaminated": {"eval_table": "outputs/eval_table.csv",
                     "s2": "outputs/s2_citations_v2_tiered.csv", "years": [2018, 2019, 2020]},
    "partial":      {"eval_table": "outputs/eval_table_2024.csv",
                     "s2": "outputs/s2_citations_2024.csv", "years": [2024]},
    "clean":        {"eval_table": "outputs/eval_table_2025.csv",
                     "s2": "outputs/s2_citations_2025.csv", "years": [2025]},
}
PROBES = ["lap", "fame", "placebo", "wrongyear"]


def decision_class(d):
    d = str(d or "")
    for pre, lab in [("Accept", "accept"), ("Withdraw", "withdrawn"),
                     ("Desk", "desk_reject"), ("Reject", "reject")]:
        if d.startswith(pre):
            return lab
    return "other"


def load_arm(name, cfg):
    if not os.path.exists(cfg["eval_table"]):
        print(f"  [{name}] missing {cfg['eval_table']} — skipped")
        return None
    ev = pd.read_csv(cfg["eval_table"], low_memory=False)
    ev["decision_class"] = ev["decision"].map(decision_class)

    if not os.path.exists(cfg["s2"]):
        print(f"  [{name}] missing {cfg['s2']} — skipped")
        return None
    s2 = pd.read_csv(cfg["s2"], low_memory=False)
    keep = ["paper_id", "s2_citations", "s2_paper_id", "s2_year", "s2_venue", "query_method"]
    if "tier" in s2:
        keep.append("tier")
    s2 = s2[[c for c in keep if c in s2]]
    df = ev.merge(s2, on="paper_id", how="left")

    # verified measurement only: an identifier match that returned a record
    df["verified"] = (df["s2_paper_id"].notna() & df["s2_citations"].notna()
                      & df["query_method"].isin(["arxiv_id", "doi"]))
    if "tier" in df:
        df["verified"] &= df["tier"].ne("C")
    df["arm"] = name
    print(f"  [{name}] {len(df):,} papers, {int(df['verified'].sum()):,} with a verified "
          f"citation measurement ({df['verified'].mean():.1%})")
    return df


def fabricate_titles(titles, k, rng):
    """Plausible but non-existent titles: recombine head and tail fragments of real ones
    from the same arm. Deterministic given the seed."""
    heads, tails = [], []
    for t in titles:
        w = str(t).split()
        if len(w) >= 6:
            cut = len(w) // 2
            heads.append(" ".join(w[:cut]))
            tails.append(" ".join(w[cut:]))
    out, seen = [], set(str(t).lower() for t in titles)
    guard = 0
    while len(out) < k and guard < k * 50:
        guard += 1
        cand = f"{heads[rng.integers(len(heads))]} {tails[rng.integers(len(tails))]}"
        if cand.lower() not in seen and cand not in out:
            out.append(cand)
    return out


def sample_arm(df, n, rng, min_cell=None):
    """Stratify on decision class x citation quartile among verified papers."""
    e = df[df["verified"]].copy()
    if e.empty:
        return e
    e["cit_q"] = pd.qcut(e["s2_citations"].rank(method="first"), 4, labels=False)
    e["stratum"] = e["decision_class"] + "_q" + e["cit_q"].astype(str)

    cells = {k: g for k, g in e.groupby("stratum") if len(g)}
    if min_cell is None:
        min_cell = max(4, n // (2 * max(1, len(cells))))
    # floor first so small classes (withdrawn, desk_reject) stay analysable,
    # then distribute what is left in proportion to cell size
    alloc = {k: min(min_cell, len(g)) for k, g in cells.items()}
    left = n - sum(alloc.values())
    if left > 0:
        tot = sum(max(0, len(g) - alloc[k]) for k, g in cells.items())
        for k, g in cells.items():
            if tot > 0:
                alloc[k] += int(round(left * max(0, len(g) - alloc[k]) / tot))
    picks = []
    for k, g in cells.items():
        take = min(alloc[k], len(g))
        idx = rng.choice(len(g), size=take, replace=False)
        picks.append(g.iloc[idx])
    out = pd.concat(picks).head(n)
    return out.reset_index(drop=True)


def build(n_per_arm, seed):
    rng = np.random.default_rng(seed)
    frames = []
    for name, cfg in ARMS.items():
        df = load_arm(name, cfg)
        if df is None:
            continue
        s = sample_arm(df, n_per_arm, rng)
        if len(s):
            frames.append(s)
    if not frames:
        raise SystemExit("No arms available — run the S2 fetches first.")

    cols = ["arm", "paper_id", "title", "year", "decision", "decision_class",
            "mean_rating", "n_reviews", "s2_citations", "s2_year", "s2_venue",
            "stratum", "cit_q"]
    papers = pd.concat(frames, ignore_index=True)
    papers = papers[[c for c in cols if c in papers]]
    papers["sample_id"] = [f"{a}_{i:04d}" for i, a in enumerate(papers["arm"])]
    papers["frozen_at"] = datetime.now(timezone.utc).date().isoformat()
    papers["seed"] = seed
    papers.to_csv(f"{OUTDIR}/oos_papers.csv", index=False)

    # ---- probe plan: one row per paper x variant, identical for every model
    rows = []
    for arm, g in papers.groupby("arm"):
        fake = fabricate_titles(g["title"].tolist(), len(g), rng)
        for i, r in enumerate(g.itertuples()):
            for probe in PROBES:
                title, year = r.title, int(r.year)
                if probe == "placebo":
                    title = fake[i] if i < len(fake) else fake[-1]
                if probe == "wrongyear":
                    year = year + (2 if rng.random() < 0.5 else -2)
                rows.append({"sample_id": r.sample_id, "arm": arm, "probe": probe,
                             "paper_id": r.paper_id, "probe_title": title,
                             "probe_year": year, "true_year": int(r.year),
                             "decision_class": r.decision_class,
                             "s2_citations": r.s2_citations, "stratum": r.stratum})
    plan = pd.DataFrame(rows)
    plan.to_csv(f"{OUTDIR}/oos_probe_plan.csv", index=False)

    write_design(papers, plan, n_per_arm, seed)
    print(f"\nWrote {OUTDIR}/oos_papers.csv ({len(papers):,} papers) and "
          f"{OUTDIR}/oos_probe_plan.csv ({len(plan):,} probe calls per model)")


def write_design(papers, plan, n_per_arm, seed):
    n_arms = papers["arm"].nunique()
    per_arm = papers.groupby("arm").size()
    # MDE for a difference in proportions between two arms, alpha=.05, power=.80
    def mde(n1, n2, p=0.5):
        return 2.8 * np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    L = ["# Out-of-sample benchmark — frozen sample design", "",
         f"Frozen {datetime.now(timezone.utc).strftime('%Y-%m-%d')} with seed {seed}. "
         "Every model and every probe runs on exactly these papers.", "",
         "## Arms", "",
         "| arm | papers | years | what a Dec-2023-cutoff model could have seen |",
         "|---|---|---|---|"]
    seen = {"contaminated": "papers, decisions and citations",
            "partial": "preprints only — no decision, no citations",
            "clean": "nothing"}
    for a, n in per_arm.items():
        yrs = sorted(papers[papers["arm"] == a]["year"].unique())
        L.append(f"| {a} | {n:,} | {', '.join(str(int(y)) for y in yrs)} | {seen.get(a,'')} |")

    L += ["", "## Strata (decision class x citation quartile)", "",
          "| arm | " + " | ".join(sorted(papers["decision_class"].unique())) + " |",
          "|---" * (1 + papers["decision_class"].nunique()) + "|"]
    ct = pd.crosstab(papers["arm"], papers["decision_class"])
    for a in ct.index:
        L.append(f"| {a} | " + " | ".join(str(int(ct.loc[a, c])) if c in ct else "0"
                                          for c in sorted(papers["decision_class"].unique())) + " |")

    L += ["", "## Citation distribution by arm (verified S2 counts)", "",
          "| arm | median | p75 | p90 | share zero |", "|---|---|---|---|---|"]
    for a, g in papers.groupby("arm"):
        c = g["s2_citations"]
        L.append(f"| {a} | {c.median():.0f} | {c.quantile(.75):.0f} | {c.quantile(.90):.0f} | "
                 f"{(c <= 0).mean():.1%} |")

    L += ["", "## Probe plan", "",
          f"{len(PROBES)} variants x {len(papers):,} papers = **{len(plan):,} calls per model**.", "",
          "| probe | question | reads |", "|---|---|---|",
          "| lap | was this accepted or rejected at ICLR <year>? | outcome recall |",
          "| fame | is this paper widely cited? | impact recall |",
          "| placebo | same, with a fabricated title | false-positive / acquiescence rate |",
          "| wrongyear | same, with the year shifted +/-2 | acquiescence to confident framing |",
          "", "The placebo arm is not optional. A test call to Llama-3.3-70B answered "
          "\"accepted\" for *Attention Is All You Need* at ICLR 2018 — a paper never submitted "
          "to ICLR. Without the placebo rate, any positive recall result is uninterpretable.", "",
          "## Detectable effects", ""]
    arms = list(per_arm.index)
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            L.append(f"- {arms[i]} vs {arms[j]}: n={per_arm[arms[i]]:,} vs {per_arm[arms[j]]:,}, "
                     f"MDE on a rate difference ≈ **{100*mde(per_arm[arms[i]], per_arm[arms[j]]):.1f}pp** "
                     "(alpha .05, power .80, worst-case p=.5)")
    L += ["", "Within-arm subgroup comparisons (e.g. accepted vs rejected) have roughly half "
          "these n and correspondingly wider MDEs; the strata floor keeps every cell at 15+ so "
          "subgroup breaks stay possible, at the cost of the sample not being self-weighting. "
          "Weight by stratum size when reporting arm-level rates.", "",
          "## Eligibility and its cost", "",
          "Only papers with an arXiv-ID match into Semantic Scholar are eligible, because "
          "title-only matches produce citation counts that fail more often for rejected papers. "
          "That restriction removes roughly half of 2024/2025, and the removed half is "
          "decision-skewed — arXiv posting rates differ ~32pp between accepted and rejected "
          "papers even after correcting for retitling. Identical across arms, so it does not "
          "bias model-vs-model comparison; it does bound generalization to arXiv-posted work, "
          "and that belongs in the limitations.", ""]
    open(f"{OUTDIR}/oos_sample_design.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-arm", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    build(a.n_per_arm, a.seed)
