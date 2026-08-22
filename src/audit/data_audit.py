"""
Full-pipeline data quality audit.

Walks every stage of the CitesBench data flow and flags issues:

  Stage 0  source        data/gen_review.db (SUBMISSION / REVIEW / GENAI_REVIEW)
  Stage 1  external      OpenAlex + Semantic Scholar fetches, author/venue pulls
  Stage 2  annotation    LLM-generated labels (fields, committee, decision head, leakage probes)
  Stage 3  join          outputs/eval_table.csv — the one flat table everything downstream reads
  Stage 4  results       eval_results / leakage_* / bootstrap artifacts
  Stage 5  plumbing      staleness, provenance, convention violations

Writes outputs/data_audit.md and outputs/data_audit.html. Every finding carries
the file(s) to open and a command to reproduce it.

Run: python src/audit/data_audit.py [--no-html]

# ponytail: checks are flat functions appending to one list; no plugin registry
# until someone needs to run a subset.
"""
import os
import re
import html
import json
import argparse
import sqlite3
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

DB = "data/gen_review.db"
YEARS = (2018, 2019, 2020)

# Every path the pipeline touches, with the script that produces it.
# Used for the staleness DAG in stage 5.
PRODUCERS = {
    "output/citations_2018_2020.csv": ("src/fetch/fetch_citations_openalex.py", [DB]),
    "outputs/paper_fields.csv": ("src/build/tag_fields.py", [DB]),
    "outputs/eval_table.csv": ("src/build/build_eval_table.py",
                               [DB, "output/citations_2018_2020.csv", "outputs/paper_fields.csv"]),
    "outputs/s2_citations_full.csv": ("src/fetch/fetch_citations_s2_full.py",
                                      ["outputs/eval_table.csv",
                                       "data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"]),
    "outputs/eval_results.csv": ("src/analysis/run_eval.py", ["outputs/eval_table.csv"]),
    "outputs/leakage_lap_v1.csv": ("src/probes/leakage_lap_v1.py", ["outputs/eval_table.csv"]),
    "outputs/leakage_fame_v1.csv": ("src/probes/leakage_fame_v1.py", ["outputs/eval_table.csv"]),
    "outputs/leakage_exclusion_eval.csv": ("src/analysis/leakage_exclusion_eval.py",
                                           ["outputs/eval_table.csv", "outputs/leakage_lap_v1.csv",
                                            "outputs/leakage_fame_v1.csv"]),
    "outputs/leakage_exclusion_eval_s2.csv": ("src/analysis/leakage_exclusion_eval.py",
                                              ["outputs/s2_citations_full.csv",
                                               "outputs/leakage_lap_v1.csv"]),
    "outputs/paper_author_covariates.csv": ("src/build/build_author_covariates.py",
                                            ["outputs/author_stats.csv", "outputs/paper_author_ids.csv"]),
    "outputs/hetero_analysis.md": ("src/analysis/hetero_analysis.py",
                                   ["outputs/eval_table.csv", "outputs/paper_author_covariates.csv"]),
    # NB: table1_summary_stats.py only prints to stdout — the .tex was written by hand
    # or by redirection, so it has no producer edge and is excluded from the staleness DAG.
    "outputs/leakage_threshold_sweep.csv": ("src/analysis/leakage_threshold_sweep.py",
                                            ["outputs/eval_table.csv", "outputs/leakage_lap_v1.csv"]),
}

FIELD_TAXONOMY = {"nlp", "computer_vision", "generative_models",
                  "reinforcement_learning", "theory_methods"}

ISSUES = []
STAGE_NOTES = {}   # stage -> list of (label, value) descriptive stats


def flag(stage, severity, title, detail, where, how=""):
    """severity: blocker | major | minor | info"""
    ISSUES.append(dict(stage=stage, severity=severity, title=title, detail=detail,
                       where=where if isinstance(where, list) else [where], how=how))


def note(stage, label, value):
    STAGE_NOTES.setdefault(stage, []).append((label, str(value)))


def read(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def parse_rating(s):
    """Same parse build_eval_table.py uses: '6: Marginally above...' -> 6.0"""
    return s.str.extract(r"^(\d+)")[0].astype(float)


# ---------------------------------------------------------------- stage 0: source DB

def audit_source():
    S = "0 · Source (gen_review.db)"
    if not os.path.exists(DB):
        flag(S, "blocker", "Source database missing", f"{DB} not found.", DB)
        return None
    con = sqlite3.connect(DB)
    sub = pd.read_sql("SELECT id AS paper_id, title, abstract, decision, "
                      "when_submitted AS year, source_id FROM SUBMISSION", con)
    rev = pd.read_sql("SELECT paper_id, reviewer_id, rating, confidence FROM REVIEW", con)
    gen = pd.read_sql("SELECT paper_id, type, rating FROM GENAI_REVIEW", con)
    con.close()

    note(S, "SUBMISSION rows", f"{len(sub):,}")
    note(S, "REVIEW rows", f"{len(rev):,}")
    note(S, "GENAI_REVIEW rows", f"{len(gen):,}")

    in_window = sub[sub["year"].isin(YEARS)]
    note(S, f"SUBMISSION in {YEARS[0]}-{YEARS[-1]}", f"{len(in_window):,}")
    note(S, "year values present", sorted(sub["year"].dropna().unique().tolist()))

    if len(in_window) < len(sub):
        flag(S, "info", "Database is wider than the study window",
             f"{len(sub):,} submissions total, only {len(in_window):,} are "
             f"{YEARS[0]}-{YEARS[-1]}. Every pipeline script filters on "
             "`when_submitted IN (2018,2019,2020)`; anything that forgets the filter "
             "silently mixes in other years/venues.",
             ["src/build/build_eval_table.py#L22-L26", "src/probes/leakage_abstract_completion_v1.py#L144"],
             "sqlite3 data/gen_review.db 'select when_submitted, count(*) from SUBMISSION group by 1'")

    # --- orphan reviews
    orphan = rev[~rev["paper_id"].isin(set(sub["paper_id"]))]
    note(S, "REVIEW rows w/ no SUBMISSION", f"{len(orphan):,}")
    if len(orphan):
        flag(S, "major", "REVIEW rows reference missing submissions",
             f"{len(orphan):,} review rows ({pct(len(orphan), len(rev))}) have a paper_id "
             "absent from SUBMISSION. The declared foreign key points at a table named "
             "`PAPER` that does not exist, so SQLite never enforced it.",
             ["data/gen_review.db", "src/build/build_eval_table.py#L28-L31"],
             "sqlite3 data/gen_review.db 'select count(*) from REVIEW r left join SUBMISSION s "
             "on r.paper_id=s.id where s.id is null'")

    # --- rating parseability (the number every human regime is built on)
    rw = rev[rev["paper_id"].isin(set(in_window["paper_id"]))].copy()
    rw["num"] = parse_rating(rw["rating"])
    bad = rw[rw["num"].isna()]
    note(S, "in-window review rows", f"{len(rw):,}")
    note(S, "unparseable ratings", f"{len(bad):,}")
    if len(bad):
        samples = bad["rating"].value_counts().head(5).to_dict()
        decision_shaped = bad["rating"].str.match(r"^(Accept|Reject|Invite|Desk)").fillna(False)
        sev = "blocker" if decision_shaped.mean() > 0.5 else "major"
        flag(S, sev, "The REVIEW table contains decision rows masquerading as reviews",
             f"{len(bad):,} of {len(rw):,} in-window REVIEW rows ({pct(len(bad), len(rw))}) have a "
             f"`rating` that is not a score at all — it is the paper's decision string. Most common "
             f"values: {samples}. These are the area chair's decision notes, scraped into the same "
             "table as reviewer records. They become NaN under the "
             r"`^(\d+)` parse, so mean_rating is unaffected, but n_reviews counts them: "
             "`.agg(n_reviews='count')` counts non-NaN parsed ratings, while anything that counts "
             "REVIEW rows directly (Table 1's reviews-per-paper, the outlier analysis' "
             "confidence_mean) is inflated by one row per paper.",
             ["src/build/build_eval_table.py#L36-L41", "src/analysis/table1_summary_stats.py#L31",
              "src/analysis/outlier_analysis.py#L18"],
             "sqlite3 data/gen_review.db \"select rating, count(*) from REVIEW where "
             "rating not glob '[0-9]*' group by 1 order by 2 desc limit 10\"")

    if len(rw):
        lo, hi = rw["num"].min(), rw["num"].max()
        note(S, "rating range (parsed)", f"{lo} – {hi}")
        if hi > 10 or lo < 1:
            flag(S, "major", "Reviewer rating outside the 1-10 scale",
                 f"Parsed ratings span {lo}-{hi}. ICLR 2018-2020 used 1-10; out-of-range values "
                 "mean the string prefix is not the score for some rows.",
                 "src/build/build_eval_table.py#L36")

    # --- review count per paper
    counts = rw.groupby("paper_id").size()
    no_rev = set(in_window["paper_id"]) - set(counts.index)
    note(S, "papers with 0 reviews", f"{len(no_rev):,}")
    note(S, "papers with 1 review", f"{int((counts == 1).sum()):,}")
    if no_rev:
        flag(S, "major", "In-window papers with no reviews at all",
             f"{len(no_rev):,} papers have zero REVIEW rows, so mean_rating and rating_std are NaN. "
             "`HumanScore` ranks by mean_rating and drops them; `HumanActual` still counts them if "
             "their decision starts with 'Accept', so the two regimes see different pools.",
             ["src/regimes/human_score.py", "src/regimes/human_actual.py#L8"],
             "python -c \"import pandas as pd;d=pd.read_csv('outputs/eval_table.csv');"
             "print(d[d.mean_rating.isna()][['paper_id','year','decision']])\"")
    if (counts == 1).sum():
        flag(S, "minor", "Single-review papers get rating_std = 0",
             f"{int((counts == 1).sum()):,} papers have exactly one review. pandas std(ddof=1) is NaN "
             "there and build_eval_table fills it with 0, i.e. 'perfect reviewer agreement'. "
             "The disagreement regimes (mean ± λ·std) therefore treat a single review as maximum "
             "consensus, which biases them toward those papers.",
             ["src/build/build_eval_table.py#L43", "src/regimes/human_disagree.py"])

    # --- decision labels
    dec = in_window["decision"].value_counts(dropna=False)
    note(S, "decision labels", dec.to_dict())
    accept_like = [d for d in dec.index if isinstance(d, str) and d.startswith("Accept")]
    other = [d for d in dec.index if d not in accept_like]
    if other:
        flag(S, "minor", "Non-'Accept*' decision labels are all pooled as rejects",
             "Accept detection everywhere is `decision.str.startswith('Accept')`. That puts "
             f"{ {d: int(dec[d]) for d in other} } on the reject side, including "
             "'Invite to Workshop Track' (a partial accept) and 'Desk Reject' (never reviewed). "
             "Desk rejects have no reviews and no real quality signal but still sit in the pool "
             "that random/ideal baselines are drawn from.",
             ["src/build/build_eval_table.py#L85", "src/analysis/run_eval.py#L23-L25",
              "src/regimes/human_actual.py#L8"],
             "sqlite3 data/gen_review.db 'select decision, count(*) from SUBMISSION where "
             "when_submitted in (2018,2019,2020) group by 1 order by 2 desc'")
    if in_window["decision"].isna().any():
        flag(S, "major", "Submissions with NULL decision",
             f"{int(in_window['decision'].isna().sum()):,} in-window papers have no decision. "
             "`decision` is the only nullable column in SUBMISSION.",
             "data/gen_review.db")

    # --- text fields the LLM probes depend on ('' passes NOT NULL)
    empty_abs = in_window[in_window["abstract"].fillna("").str.strip() == ""]
    short_abs = in_window[in_window["abstract"].fillna("").str.len().between(1, 200)]
    note(S, "empty abstracts (in window)", f"{len(empty_abs):,}")
    note(S, "abstracts < 200 chars", f"{len(short_abs):,}")
    if len(empty_abs) or len(short_abs):
        flag(S, "major" if len(empty_abs) else "minor",
             "Empty or stub abstracts feed the memorization probes",
             f"{len(empty_abs):,} empty and {len(short_abs):,} very short (<200 char) abstracts. "
             "The abstract-completion probe splits on the first sentence and ROUGE-scores the rest; "
             "an empty or one-line abstract produces a degenerate score rather than an error, and "
             "masked re-review has nothing to mask.",
             ["src/probes/leakage_abstract_completion_v1.py#L144", "src/probes/leakage_masked_rereview.py#L93"],
             "sqlite3 data/gen_review.db \"select count(*) from SUBMISSION where "
             "when_submitted in (2018,2019,2020) and trim(abstract)=''\"")

    # --- duplicate titles (same paper resubmitted / scrape duplication)
    dup_titles = (in_window.assign(t=in_window["title"].str.lower().str.strip())
                  .groupby("t").filter(lambda g: len(g) > 1))
    note(S, "papers sharing a title", f"{len(dup_titles):,}")
    if len(dup_titles):
        flag(S, "minor", "Duplicate titles inside the study window",
             f"{len(dup_titles):,} in-window papers share a normalized title with another paper "
             f"({dup_titles['t'].nunique():,} distinct titles). Both the OpenAlex and S2 fetchers "
             "match by title for unmatched papers, so duplicates can be assigned the same external "
             "record and double-count its citations.",
             ["src/fetch/fetch_citations_s2_full.py#L107-L145", "src/fetch/fetch_rejected_venues_s2_title.py"],
             "sqlite3 data/gen_review.db \"select lower(trim(title)) t, count(*) c from SUBMISSION "
             "where when_submitted in (2018,2019,2020) group by t having c>1\"")

    # --- GENAI_REVIEW shape
    if len(gen):
        gen_win = gen[gen["paper_id"].isin(set(in_window["paper_id"]))]
        per = gen_win.groupby("paper_id")["type"].nunique()
        note(S, "GENAI_REVIEW types", gen["type"].value_counts().to_dict())
        note(S, "GENAI_REVIEW papers", f"{gen['paper_id'].nunique():,}")
        note(S, "GENAI_REVIEW rows in window", f"{len(gen_win):,}")
        incomplete = int((per < 3).sum())
        if incomplete:
            flag(S, "minor", "GENAI_REVIEW personas incomplete for some papers",
                 f"{incomplete:,} papers have fewer than 3 persona rows. "
                 "build_eval_table no longer reads GENAI_REVIEW at all, so this affects nothing "
                 "papers' means come from a different persona mix than the rest.",
                 "src/build/build_eval_table.py#L45-L51")
        gbad = gen_win.assign(num=parse_rating(gen_win["rating"]))
        gbad = gbad[gbad["num"].isna()]
        if len(gbad):
            flag(S, "major", "GENAI_REVIEW ratings that do not parse",
                 f"{len(gbad):,} of {len(gen_win):,} in-window generated ratings fail the numeric "
                 "extraction — LLM output that drifted from the expected 'N: label' format.",
                 "src/build/build_eval_table.py#L44")
        orphan_gen = gen[~gen["paper_id"].isin(set(sub["paper_id"]))]
        if len(orphan_gen):
            flag(S, "minor", "GENAI_REVIEW rows with no matching submission",
                 f"{len(orphan_gen):,} generated-review rows reference an unknown paper_id.",
                 "data/gen_review.db")
    return sub, rev, gen


# ------------------------------------------------------- stage 1: external citation data

def audit_external(sub):
    S = "1 · External data (OpenAlex / S2)"
    oa = read("output/citations_2018_2020.csv")
    s2 = read("outputs/s2_citations_full.csv")
    ev = read("outputs/eval_table.csv")

    if oa is None:
        flag(S, "blocker", "OpenAlex citation file missing",
             "output/citations_2018_2020.csv is the citation input to build_eval_table.py.",
             "src/build/build_eval_table.py#L53")
    else:
        note(S, "OpenAlex rows", f"{len(oa):,}")
        note(S, "OpenAlex status", oa["status"].value_counts().to_dict())
        found = (oa["status"] == "found").sum()
        note(S, "OpenAlex match rate", pct(found, len(oa)))
        if oa["paper_id"].duplicated().any():
            flag(S, "major", "Duplicate paper_id in the OpenAlex citation file",
                 f"{int(oa['paper_id'].duplicated().sum()):,} duplicate rows. The fetcher appends "
                 "incrementally for resumability; a resumed run that mis-computes the done-set "
                 "re-appends. Duplicates fan out rows on the merge in build_eval_table.",
                 ["src/fetch/fetch_citations_openalex.py", "src/build/build_eval_table.py#L67"],
                 "python -c \"import pandas as pd;d=pd.read_csv('output/citations_2018_2020.csv');"
                 "print(d[d.paper_id.duplicated(keep=False)].sort_values('paper_id'))\"")
        if (oa["openalex_citations"].fillna(0) < 0).any():
            flag(S, "major", "Negative OpenAlex citation counts", "Impossible values present.",
                 "output/citations_2018_2020.csv")

        # differential missingness by decision — the bias that matters
        if ev is not None and "decision" in ev:
            j = oa.merge(ev[["paper_id", "decision"]], on="paper_id", how="left")
            j["accepted"] = j["decision"].fillna("").str.startswith("Accept")
            rate = j.groupby("accepted")["status"].apply(lambda s: (s == "found").mean())
            note(S, "OpenAlex match rate by decision",
                 {("accept" if k else "reject"): f"{v:.1%}" for k, v in rate.items()})
            if len(rate) == 2 and abs(rate.iloc[0] - rate.iloc[1]) > 0.03:
                flag(S, "blocker", "OpenAlex coverage is differential by decision",
                     f"Match rate {rate.get(True, float('nan')):.1%} for accepted vs "
                     f"{rate.get(False, float('nan')):.1%} for rejected. Unmatched papers get "
                     "openalex_citations = NaN and are dropped from every metric, so the accept and "
                     "reject pools are not the same population. Any accept-vs-reject citation "
                     "comparison on the OpenAlex source inherits that selection.",
                     ["src/build/build_eval_table.py#L73", "src/metrics.py#L20-L24",
                      "outputs/citation_source_comparison.md"],
                     "python src/analysis/compare_citation_sources.py")

    if s2 is None:
        flag(S, "major", "Semantic Scholar citation file missing",
             "outputs/s2_citations_full.csv is the corrected ground truth used by the "
             "dashboard's source toggle and every bootstrap.",
             "src/fetch/fetch_citations_s2_full.py")
    else:
        note(S, "S2 rows", f"{len(s2):,}")
        note(S, "S2 methods", s2["method"].value_counts(dropna=False).to_dict())
        miss = s2["s2_citations"].isna().sum()
        note(S, "S2 rows with no citation count", f"{miss:,} ({pct(miss, len(s2))})")
        if s2["paper_id"].duplicated().any():
            flag(S, "major", "Duplicate paper_id in the S2 citation file",
                 f"{int(s2['paper_id'].duplicated().sum()):,} duplicates from incremental appends.",
                 "src/fetch/fetch_citations_s2_full.py#L74")

        # title-match quality: the main wrong-paper risk
        tm = s2[s2["title_sim"].notna()]
        if len(tm):
            note(S, "S2 title-matched rows", f"{len(tm):,}")
            note(S, "S2 title_sim quartiles",
                 {k: round(v, 3) for k, v in tm["title_sim"].quantile([.05, .25, .5]).items()})
            weak = tm[tm["title_sim"] < 0.90]
            if len(weak):
                flag(S, "major", "S2 title matches accepted with no similarity threshold",
                     f"{len(weak):,} of {len(tm):,} title-matched papers have title_sim < 0.90 "
                     f"(min {tm['title_sim'].min():.3f}). The fetcher records title_sim but never "
                     "filters on it, and no downstream consumer filters either — those citation "
                     "counts may belong to a different paper.",
                     ["src/fetch/fetch_citations_s2_full.py#L107-L145",
                      "src/analysis/leakage_exclusion_eval.py#L98", "src/analysis/table1_summary_stats.py#L46"],
                     "python -c \"import pandas as pd;d=pd.read_csv('outputs/s2_citations_full.csv');"
                     "print(d[d.title_sim<0.9].sort_values('title_sim').head(30)"
                     "[['paper_id','title_sim','s2_title','s2_citations']])\"")

        # year sanity: an S2 record years off the submission year is probably the wrong work
        if ev is not None:
            j = s2.merge(ev[["paper_id", "year"]], on="paper_id", how="left")
            j["dy"] = j["s2_year"] - j["year"]
            wrong = j[j["dy"].notna() & ((j["dy"] < -1) | (j["dy"] > 4))]
            note(S, "S2 year within [-1, +4] of submission",
                 pct(len(j[j['dy'].between(-1, 4)]), int(j['dy'].notna().sum())))
            if len(wrong):
                flag(S, "major", "S2 matches with an implausible publication year",
                     f"{len(wrong):,} papers matched to an S2 record dated more than a year before "
                     "or four years after submission. Combined with an unfiltered title match this "
                     "is the signature of a wrong-paper join.",
                     "src/fetch/fetch_citations_s2_full.py",
                     "python -c \"import pandas as pd;s=pd.read_csv('outputs/s2_citations_full.csv');"
                     "e=pd.read_csv('outputs/eval_table.csv')[['paper_id','year','title']];"
                     "j=s.merge(e,on='paper_id');print(j[(j.s2_year-j.year>4)|(j.s2_year-j.year<-1)]"
                     "[['paper_id','year','s2_year','title','s2_title']].head(30))\"")

        # OA vs S2 divergence
        if oa is not None:
            j = (oa[["paper_id", "openalex_citations", "status"]]
                 .merge(s2[["paper_id", "s2_citations"]], on="paper_id", how="inner"))
            j = j[(j["status"] == "found") & j["s2_citations"].notna()]
            j = j[j["openalex_citations"] > 0]
            if len(j):
                ratio = (j["s2_citations"] / j["openalex_citations"])
                note(S, "S2/OpenAlex citation ratio (median)", f"{ratio.median():.2f}x")
                note(S, "papers where S2 > 3x OpenAlex",
                     f"{int((ratio > 3).sum()):,} ({pct(int((ratio > 3).sum()), len(j))})")
                if ratio.median() > 1.5 or ratio.median() < 0.67:
                    flag(S, "blocker", "The two citation sources disagree systematically",
                         f"Median S2/OpenAlex ratio is {ratio.median():.2f}x over {len(j):,} papers "
                         f"with both counts; {int((ratio > 3).sum()):,} papers differ by more than 3x. "
                         "Both files are live in the codebase and the dashboard toggles between them, "
                         "so the headline effect size depends on which one a reader picks. Results "
                         "must always be reported with the source named.",
                         ["outputs/citation_source_comparison.md", "src/analysis/compare_citation_sources.py",
                          "src/app/dashboard.py#L70-L80", "docs/PROJECT_OVERVIEW.md"],
                         "open outputs/citation_source_comparison.md")

    # snapshot dating
    if oa is not None or s2 is not None:
        stamps = {p: datetime.fromtimestamp(os.path.getmtime(p)).date().isoformat()
                  for p in ["output/citations_2018_2020.csv", "outputs/s2_citations_full.csv"]
                  if os.path.exists(p)}
        note(S, "citation snapshot dates (file mtime)", stamps)
        if len(set(stamps.values())) > 1:
            flag(S, "minor", "Citation snapshots were taken on different dates",
                 f"{stamps}. Citation counts accrue continuously, so cross-source comparisons "
                 "conflate 'different index' with 'different as-of date'. Neither file records a "
                 "fetch timestamp column — mtime is the only provenance available.",
                 ["output/citations_2018_2020.csv", "outputs/s2_citations_full.csv"])

    # author / venue side pulls
    for path, key in [("outputs/author_stats.csv", "author_id"),
                      ("outputs/paper_author_ids.csv", None),
                      ("outputs/paper_venues.csv", "paper_id"),
                      ("outputs/paper_author_covariates.csv", "paper_id")]:
        df = read(path)
        if df is None:
            continue
        note(S, f"{os.path.basename(path)} rows", f"{len(df):,}")
        if key and df[key].duplicated().any():
            flag(S, "major", f"Duplicate {key} in {os.path.basename(path)}",
                 f"{int(df[key].duplicated().sum()):,} duplicate keys in an append-mode fetch output. "
                 "Merging on this key fans out rows downstream.",
                 [path, "src/build/build_author_covariates.py#L88-L89"])

    cov = read("outputs/paper_author_covariates.csv")
    if cov is not None and ev is not None:
        note(S, "author-covariate coverage of eval_table",
             pct(cov["paper_id"].nunique(), len(ev)))
        if cov["paper_id"].nunique() < 0.9 * len(ev):
            flag(S, "major", "Author covariates cover only part of the corpus",
                 f"{cov['paper_id'].nunique():,} of {len(ev):,} papers "
                 f"({pct(cov['paper_id'].nunique(), len(ev))}) have author covariates, because the "
                 "chain runs through an arXiv/OpenAlex match. Papers never posted to arXiv drop out, "
                 "and that is correlated with the decision being studied — the heterogeneity section "
                 "is estimated on a selected subsample.",
                 ["src/build/build_author_covariates.py", "src/analysis/hetero_analysis.py#L141",
                  "docs/PROJECT_OVERVIEW.md"])


# ---------------------------------------------------------- stage 2: LLM-produced labels

def audit_annotations(sub):
    S = "2 · LLM annotations"
    in_window = sub[sub["year"].isin(YEARS)] if sub is not None else None
    n_corpus = len(in_window) if in_window is not None else None

    # --- field tags
    f = read("outputs/paper_fields.csv")
    if f is None:
        flag(S, "major", "Field tags missing", "outputs/paper_fields.csv not found.",
             "src/build/tag_fields.py")
    else:
        note(S, "field-tagged papers", f"{len(f):,}")
        note(S, "field distribution", f["field"].value_counts().to_dict())
        if n_corpus:
            note(S, "field coverage", pct(len(f), n_corpus))
            if len(f) < 0.95 * n_corpus:
                flag(S, "blocker", "Field tagging is incomplete",
                     f"{len(f):,} of {n_corpus:,} papers tagged ({pct(len(f), n_corpus)}). "
                     "tag_fields.py stopped early rather than finishing. citation_pct_rank is "
                     "computed inside field×year groups, so every untagged paper has a NaN "
                     "normalized outcome and is dropped from the entire 'normalized' mode.",
                     ["src/build/tag_fields.py", "src/build/build_eval_table.py#L77-L80",
                      "src/analysis/run_eval.py#L34-L40"],
                     "python src/build/tag_fields.py   # resumes from outputs/paper_fields.csv")
        stray = set(f["field"].dropna().unique()) - FIELD_TAXONOMY
        if stray:
            flag(S, "major", "Field labels outside the declared taxonomy",
                 f"Unexpected labels: {sorted(stray)}. The prompt's docstring in tag_fields.py "
                 "lists a 10-field taxonomy while FIELDS defines 5 — the two disagree, so labels "
                 "from an older run can survive in the append-mode output file.",
                 ["src/build/tag_fields.py#L1-L10", "src/build/tag_fields.py#L20"])
        if f["id"].duplicated().any():
            flag(S, "major", "Duplicate paper in the field-tag file",
                 f"{int(f['id'].duplicated().sum()):,} papers tagged more than once (append-mode "
                 "output re-run without a clean done-set).", "src/build/tag_fields.py#L117")
        top = f["field"].value_counts(normalize=True).iloc[0]
        if top > 0.5:
            flag(S, "major", "One field bucket dominates the taxonomy",
                 f"'{f['field'].value_counts().index[0]}' is {top:.0%} of tagged papers. It is "
                 "defined in the prompt as 'everything else', so field fixed effects and "
                 "field-stratified results contrast one large heterogeneous bucket against four "
                 "small specific ones.",
                 ["src/build/tag_fields.py#L20-L32", "src/analysis/hetero_analysis.py"])
        flag(S, "major", "Field labels have never been validated",
             "No hand-labeled sample, no accuracy number, and no cross-check against the "
             "arxiv_categories column already present in the OpenAlex paper-level file. Every "
             "field-stratified result rests on an unmeasured classifier.",
             ["src/build/tag_fields.py",
              "data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"],
             "python -c \"import pandas as pd;d=pd.read_csv("
             "'data/OpenAlex/openalex_rdd_arxiv_paper_level.csv',low_memory=False);"
             "print(d.arxiv_categories.notna().sum())\"")

        # per-year coverage: the gap is not uniform
        if in_window is not None:
            j = in_window.merge(f.rename(columns={"id": "paper_id"}), on="paper_id", how="left")
            byyear = j.groupby("year")["field"].apply(lambda s: f"{s.notna().mean():.0%}").to_dict()
            note(S, "field coverage by year", byyear)
            worst = j.groupby("year")["field"].apply(lambda s: s.notna().mean()).min()
            if worst < 0.5:
                flag(S, "blocker", "Field coverage collapses in at least one year",
                     f"Coverage by year: {byyear}. A year that is mostly untagged cannot support "
                     "field×year normalization at all — the 'normalized' mode for that year is "
                     "computed on a small, non-random remainder.",
                     ["src/build/build_eval_table.py#L77-L80", "src/app/dashboard.py"],
                     "python -c \"import pandas as pd;d=pd.read_csv('outputs/eval_table.csv');"
                     "print(pd.crosstab(d.year,d.field.notna()))\"")

    # --- committee / decision-head scores
    apr = read("data/archive/all_paper_results.csv")
    if apr is None:
        flag(S, "major", "LLM pipeline result table missing",
             "data/archive/all_paper_results.csv carries committee_rating and deepseek_p_accept.",
             "data/README.md")
    else:
        note(S, "LLM pipeline papers", f"{len(apr):,}")
        if n_corpus:
            note(S, "LLM pipeline coverage", pct(len(apr), n_corpus))
            if len(apr) < n_corpus:
                flag(S, "minor", "LLM regimes do not cover the whole corpus",
                     f"{len(apr):,} of {n_corpus:,} papers have committee/decision-head scores. "
                     "Papers without a score are dropped by the LLM regimes but remain in the pool "
                     "the human regimes and the random baseline draw from.",
                     ["src/regimes/llm_committee.py", "src/regimes/llm_deepseek.py",
                      "src/baselines.py"])
        if "deepseek_p_accept" in apr:
            p = apr["deepseek_p_accept"]
            note(S, "deepseek_p_accept range", f"{p.min()} – {p.max()}")
            bad = p[(p < 0) | (p > 1)]
            if len(bad):
                flag(S, "blocker", "deepseek_p_accept outside [0,1]",
                     f"{len(bad):,} rows carry an out-of-range probability.",
                     "data/archive/all_paper_results.csv")
            ties = p.value_counts()
            if len(ties) and ties.iloc[0] / len(p) > 0.1:
                flag(S, "major", "deepseek_p_accept is heavily tied",
                     f"The single value {ties.index[0]} covers {ties.iloc[0]:,} papers "
                     f"({ties.iloc[0]/len(p):.0%}). Top-N selection on a mostly-tied score is "
                     "resolved by row order, not by the model — the regime becomes partly "
                     "arbitrary and is not reproducible across a re-sort.",
                     ["src/regimes/llm_deepseek.py", "src/metrics.py#L33-L39"],
                     "python -c \"import pandas as pd;print(pd.read_csv("
                     "'data/archive/all_paper_results.csv').deepseek_p_accept.value_counts().head())\"")
        if "committee_rating" in apr:
            c = apr["committee_rating"]
            note(S, "committee_rating range", f"{c.min()} – {c.max()}")
            note(S, "committee_rating nulls", f"{int(c.isna().sum()):,}")
        for col, label in [("deepseek_http_error", "decision-head HTTP errors")]:
            if col in apr:
                n = apr[col].notna().sum() if apr[col].dtype == object else int(apr[col].fillna(0).astype(bool).sum())
                note(S, label, f"{n:,}")
                if n:
                    flag(S, "major", "Decision-head calls that failed with an HTTP error",
                         f"{n:,} papers recorded a transport error. Their p_accept is missing or "
                         "partial and the regime silently ranks without them.",
                         "data/archive/all_paper_results.csv")
        # Model identity. Dedicated-endpoint suffixes are noise; a different base model is not.
        def base_model(s):
            s = str(s).split("/")[-1]
            return re.sub(r"-[0-9a-f]{8}$", "", s)

        for col, label in [("committee_model", "committee"),
                           ("decision_head_model", "decision head")]:
            if col not in apr:
                continue
            bases = apr[col].map(base_model).value_counts()
            note(S, f"{col} (endpoint suffix stripped)", bases.to_dict())
            if apr[col].nunique() > len(bases):
                flag(S, "minor", f"{label.capitalize()} ratings come from many serving endpoints",
                     f"{apr[col].nunique()} distinct `{col}` strings collapse to {len(bases)} base "
                     f"model(s) {list(bases.index)} — the rest are per-endpoint hashes. Harmless for "
                     "the model identity, but it means the run was spread over many dedicated "
                     "endpoints, so any endpoint-level config drift is untracked.",
                     "data/archive/all_paper_results.csv")
            if len(bases) > 1:
                flag(S, "blocker", f"Two different {label} models are pooled into one regime",
                     f"`{col}` resolves to {bases.to_dict()}. The regime is presented as a single "
                     "system but is two different models scored on two disjoint halves of the "
                     "corpus. Any per-year or per-field comparison is partly a comparison between "
                     "models, and the single-model-family caveat in the project notes understates "
                     "the problem — the split is inside the headline number.",
                     ["data/archive/all_paper_results.csv", "src/regimes/llm_deepseek.py",
                      "docs/PROJECT_OVERVIEW.md"],
                     "python -c \"import pandas as pd;print(pd.read_csv("
                     f"'data/archive/all_paper_results.csv').groupby('{col}').size())\"")

    # --- leakage probes
    ev = read("outputs/eval_table.csv")
    for path, pcol, cols in [
        ("outputs/leakage_lap_v1.csv", "lap", ["p_accept", "p_reject", "p_unknown"]),
        ("outputs/leakage_fame_v1.csv", "fame", ["p_high", "p_low", "p_unknown"]),
    ]:
        d = read(path)
        if d is None:
            flag(S, "minor", f"{os.path.basename(path)} missing",
                 "A leakage probe output is absent; the exclusion eval silently skips it.",
                 "src/analysis/leakage_exclusion_eval.py#L125-L135")
            continue
        note(S, f"{os.path.basename(path)} rows", f"{len(d):,}")
        if ev is not None:
            covg = d["paper_id"].nunique() / len(ev)
            note(S, f"{pcol} probe coverage", f"{covg:.1%}")
            if covg < 0.90:
                flag(S, "major", f"{pcol.upper()} probe coverage below the script's own 90% bar",
                     f"{d['paper_id'].nunique():,} of {len(ev):,} papers probed ({covg:.1%}). "
                     "leakage_exclusion_eval.py prints a warning that results are directional "
                     "below 90% coverage, and that warning currently applies.",
                     [path, "src/analysis/leakage_exclusion_eval.py"],
                     f"python {PRODUCERS.get(path, ('src/probes/leakage_lap_v1.py', []))[0]} --full")
        have = [c for c in cols if c in d]
        if len(have) == 3:
            tot = d[have].sum(axis=1)
            off = d[(tot - 1).abs() > 0.02]
            note(S, f"{pcol} probability mass in [0.98,1.02]", pct(len(d) - len(off), len(d)))
            if len(off):
                flag(S, "major" if len(off) > 0.01 * len(d) else "minor",
                     f"{pcol.upper()} probe probabilities do not sum to 1",
                     f"{len(off):,} of {len(d):,} rows ({pct(len(off), len(d))}) have p_* summing to "
                     f"{tot[(tot-1).abs()>0.02].median():.3f} (median of the offenders). The probe "
                     "reads answer probabilities from logprobs over a restricted token set; mass "
                     "leaking to other tokens means the renormalization did not fire for those rows, "
                     f"so their {pcol} is on a different scale from the rest — and {pcol} is compared "
                     "against a hard 0.5 exclusion cutoff.",
                     [path, "src/probes/leakage_lap_v1.py", "src/analysis/leakage_threshold_sweep.py"],
                     f"python -c \"import pandas as pd;d=pd.read_csv('{path}');"
                     f"print((d[{have}].sum(axis=1)).describe())\"")
        if pcol in d and d[pcol].notna().any():
            note(S, f"{pcol} at/above 0.5 (excluded by default)",
                 f"{int((d[pcol] >= 0.5).sum()):,} ({pct(int((d[pcol] >= 0.5).sum()), len(d))})")

    # Abstract-completion probe. NB: `extractable` is the memorization *signal*
    # (target ROUGE beat every decoy AND an 8-gram hit), not a failure flag.
    ac = read("outputs/leakage_abstract_completion_v1.csv")
    if ac is not None:
        note(S, "abstract-completion probe rows", f"{len(ac):,}")
        if "extractable" in ac:
            note(S, "papers flagged extractable (memorization signal)",
                 f"{int(ac['extractable'].astype(bool).sum()):,} "
                 f"({pct(int(ac['extractable'].astype(bool).sum()), len(ac))})")
        empty = ac[ac["gen_chars"].fillna(0) <= 0] if "gen_chars" in ac else ac.iloc[:0]
        if len(empty):
            flag(S, "minor", "Abstract-completion calls that returned no text",
                 f"{len(empty):,} of {len(ac):,} rows have gen_chars <= 0 and were scored anyway.",
                 "src/probes/leakage_abstract_completion_v1.py#L244")
        if len(ac) < 300:
            flag(S, "minor", "Abstract-completion probe is short of its sampled N",
                 f"{len(ac):,} rows against a 300-paper sample. The {300 - len(ac)} missing papers "
                 "returned no scoreable output and were dropped without being recorded, so the "
                 "denominator of the memorization rate is the surviving sample, not the drawn one.",
                 ["outputs/leakage_abstract_completion_v1.csv",
                  "outputs/leakage_abstract_completion_report.md"])
        if len(ac) < 0.1 * 4567:
            flag(S, "major", "Memorization conclusions rest on a 300-paper subsample",
                 f"The hardest memorization probe covers {len(ac):,} of ~4,567 papers "
                 f"({pct(len(ac), 4567)}). It is the probe the project leans on to argue the LLM "
                 "regimes are not recalling famous papers, and at this N a null result has wide "
                 "confidence bounds — especially since the exclusion eval uses the LAP/FAME probes' "
                 "0.5 cutoff, which this probe does not validate.",
                 ["outputs/leakage_abstract_completion_report.md",
                  "src/probes/leakage_abstract_completion_v1.py"])


# ------------------------------------------------------------ stage 3: the join table

def audit_join(sub, rev):
    S = "3 · Join (eval_table.csv)"
    ev = read("outputs/eval_table.csv")
    if ev is None:
        flag(S, "blocker", "eval_table.csv missing",
             "Every downstream script reads outputs/eval_table.csv.",
             "src/build/build_eval_table.py")
        return
    note(S, "rows", f"{len(ev):,}")
    note(S, "columns", len(ev.columns))
    note(S, "column non-null counts",
         {c: int(ev[c].notna().sum()) for c in ev.columns})

    if ev["paper_id"].duplicated().any():
        flag(S, "blocker", "Duplicate paper_id in eval_table",
             f"{int(ev['paper_id'].duplicated().sum()):,} duplicates — a many-to-one merge fanned out.",
             "src/build/build_eval_table.py#L64-L70")

    # --- the big one: columns on disk that the builder does not produce
    src = open("src/build/build_eval_table.py").read()
    declared = set(re.findall(r'"([a-z_0-9]+)"', src)) | set(re.findall(r"'([a-z_0-9]+)'", src))
    orphan_cols = [c for c in ev.columns if c not in declared]
    if orphan_cols:
        flag(S, "blocker", "eval_table.csv holds columns build_eval_table.py cannot produce",
             f"Columns present on disk but never written by the builder: {orphan_cols}. "
             "They were merged in by hand or by a script that no longer exists in src/ "
             "(the values match data/archive/all_paper_results.csv). Re-running "
             "`python src/build/build_eval_table.py` regenerates the file without them and silently "
             "breaks every LLM regime, the heterogeneity section and the exclusion eval. "
             "The build step is not reproducible.",
             ["src/build/build_eval_table.py", "outputs/eval_table.csv",
              "data/archive/all_paper_results.csv", "src/regimes/llm_committee.py",
              "src/analysis/hetero_analysis.py#L31"],
             "python -c \"import pandas as pd;print([c for c in "
             "pd.read_csv('outputs/eval_table.csv').columns])\"")

    # --- recompute what the builder claims to compute, and diff
    if sub is not None and rev is not None:
        base = sub[sub["year"].isin(YEARS)][["paper_id", "year", "decision"]]
        note(S, "papers in DB window vs eval_table", f"{len(base):,} vs {len(ev):,}")
        missing = set(base["paper_id"]) - set(ev["paper_id"])
        extra = set(ev["paper_id"]) - set(base["paper_id"])
        if missing or extra:
            flag(S, "major", "eval_table paper set does not match the source window",
                 f"{len(missing):,} DB papers absent from eval_table, {len(extra):,} eval_table "
                 "papers absent from the DB window. The table is a stale snapshot of the source.",
                 ["src/build/build_eval_table.py#L22-L26", "outputs/eval_table.csv"])

        rw = rev[rev["paper_id"].isin(set(base["paper_id"]))].copy()
        rw["num"] = parse_rating(rw["rating"])
        agg = (rw.groupby("paper_id")["num"]
               .agg(mean_rating_rebuilt="mean", n_reviews_rebuilt="count").reset_index())
        chk = ev[["paper_id", "mean_rating", "n_reviews"]].merge(agg, on="paper_id", how="left")
        drift = chk[(chk["mean_rating"] - chk["mean_rating_rebuilt"]).abs() > 1e-6]
        note(S, "mean_rating rows matching a fresh rebuild",
             pct(len(chk) - len(drift), len(chk)))
        if len(drift):
            flag(S, "blocker", "eval_table.mean_rating does not match a rebuild from the DB",
                 f"{len(drift):,} of {len(chk):,} papers differ. The committed table was built from "
                 "a different DB state than the one on disk, so every human-score regime, the RDD "
                 "cutoff and Table 1 are computed on numbers that no longer reproduce.",
                 ["src/build/build_eval_table.py#L36-L43", "outputs/eval_table.csv"],
                 "python src/build/build_eval_table.py   # then diff against the committed file")

    # --- citation column integrity
    if "openalex_citations" in ev:
        n = ev["openalex_citations"].notna().sum()
        note(S, "openalex_citations coverage", pct(n, len(ev)))
        if n < 0.9 * len(ev):
            flag(S, "major", "A third of the corpus has no citation outcome",
                 f"{len(ev) - n:,} of {len(ev):,} papers have NaN openalex_citations. metrics.py "
                 "drops NaNs from the selected set but computes the top-k denominator from "
                 "`quality.dropna()`, so recall@k is measured against the matched subset while "
                 "regimes select from the full pool. Coverage differences between regimes therefore "
                 "move recall without any change in selection quality.",
                 ["src/metrics.py#L20-L39", "src/build/build_eval_table.py#L73"])

    # --- normalized outcome
    if "citation_pct_rank" in ev:
        n = ev["citation_pct_rank"].notna().sum()
        note(S, "citation_pct_rank coverage", pct(n, len(ev)))
        if "field" in ev:
            both = ev[ev["field"].notna() & ev["openalex_citations"].notna()]
            if n < len(both):
                flag(S, "minor", "citation_pct_rank missing for papers that have the inputs",
                     f"{len(both) - n:,} papers have both a field and a citation count but no "
                     "percentile rank.", "src/build/build_eval_table.py#L77-L80")
            tiny = (both.groupby(["field", "year"]).size().pipe(lambda s: s[s < 30]))
            if len(tiny):
                flag(S, "major", "Percentile ranks computed inside very small field×year cells",
                     f"{len(tiny):,} field×year cells have fewer than 30 papers "
                     f"(smallest {int(tiny.min())}). A percentile rank inside a 5-paper cell takes "
                     "only 5 distinct values, so the normalized outcome is coarse and unstable "
                     "exactly where field stratification is used.",
                     ["src/build/build_eval_table.py#L77-L80", "src/app/dashboard.py"],
                     "python -c \"import pandas as pd;d=pd.read_csv('outputs/eval_table.csv');"
                     "print(pd.crosstab(d.field,d.year))\"")

    # --- N per year, the pinned budget every regime must return
    if "decision" in ev:
        acc = ev[ev["decision"].fillna("").str.startswith("Accept")].groupby("year").size()
        note(S, "N accepts per year (pinned N)", acc.to_dict())
        for col in ["committee_rating", "deepseek_p_accept", "mean_rating"]:
            if col not in ev:
                continue
            short = {int(y): int(g[col].notna().sum()) for y, g in ev.groupby("year")
                     if int(g[col].notna().sum()) < int(acc.get(y, 0))}
            if short:
                flag(S, "blocker", f"Fewer scored papers than the pinned N for {col}",
                     f"Year → scored papers, where scored < N accepts: {short} vs N {acc.to_dict()}. "
                     "A regime ranking on this column cannot return n ids and will either assert or "
                     "pad with unscored papers.",
                     ["src/analysis/run_eval.py#L23-L25", f"src/regimes/"])


# -------------------------------------------------------------- stage 4: result artifacts

def audit_results():
    S = "4 · Results artifacts"
    ev = read("outputs/eval_table.csv")
    res = read("outputs/eval_results.csv")

    # which regimes does the code actually register?
    reg_src = open("src/regimes/__init__.py").read()
    registered = re.findall(r"^from \.(\w+) import (\w+)", reg_src, re.M)
    in_all = re.search(r"ALL_REGIMES\s*=\s*\[(.*?)\]", reg_src, re.S)
    active = re.findall(r"(\w+)\(\)", in_all.group(1)) if in_all else []
    on_disk = sorted(f[:-3] for f in os.listdir("src/regimes")
                     if f.endswith(".py") and f != "__init__.py")
    note(S, "regime modules on disk", on_disk)
    note(S, "regimes in ALL_REGIMES", active)
    # dashboard.py keeps its own hardcoded regime list instead of using ALL_REGIMES
    dash = open("src/app/dashboard.py").read() if os.path.exists("src/app/dashboard.py") else ""
    dash_regimes = sorted(set(re.findall(r"from regimes\.\w+ import (\w+)", dash)))
    note(S, "regimes the dashboard imports directly", dash_regimes)

    unwired = [m for m in on_disk if m not in [r[0] for r in registered]]
    if unwired:
        flag(S, "major", "Two independent regime registries that have drifted apart",
             f"`ALL_REGIMES` in regimes/__init__.py holds {active}, while dashboard.py ignores it "
             f"and imports {dash_regimes} directly at dashboard.py:213. Modules not registered at "
             f"all: {unwired}. The dashboard is therefore correct and current — it computes "
             "committee and decision-head metrics live from eval_table.csv — but `run_eval.py`, the "
             "only batch/CLI path, still evaluates the three superseded persona regimes and cannot "
             "reproduce a single number the dashboard displays. Whichever list a reader trusts "
             "changes the answer.",
             ["src/regimes/__init__.py#L13-L28", "src/analysis/run_eval.py#L14",
              "src/app/dashboard.py#L17-L21", "src/app/dashboard.py#L213-L214"],
             "python -c \"import sys;sys.path.insert(0,'src');from regimes import ALL_REGIMES;"
             "print([r.name for r in ALL_REGIMES])\"")

    if res is None:
        flag(S, "minor", "eval_results.csv missing", "run_eval.py has not been run.",
             "src/analysis/run_eval.py")
    else:
        note(S, "eval_results rows", f"{len(res):,}")
        note(S, "regimes in eval_results", sorted(res["regime"].unique().tolist()))
        counts = res.groupby("regime").size()
        note(S, "rows per regime", counts.to_dict())
        if counts.nunique() > 1:
            flag(S, "major", "Uneven row counts per regime in eval_results",
                 f"{counts.to_dict()}. Regimes with half the rows were only evaluated in one of the "
                 "two modes (raw / normalized) — a skip path fired mid-run and nothing recorded why. "
                 "Any table built by pivoting this file compares regimes on different metric sets.",
                 ["outputs/eval_results.csv", "src/analysis/run_eval.py#L29-L45"],
                 "python -c \"import pandas as pd;d=pd.read_csv('outputs/eval_results.csv');"
                 "print(d.pivot_table(index='regime',columns='mode',values='value',aggfunc='count'))\"")
        stale_names = [r for r in res["regime"].unique()
                       if re.match(r"LLM\d", str(r))]
        if stale_names:
            flag(S, "major", "eval_results.csv is from the superseded persona regimes",
                 f"Regimes present: {sorted(res['regime'].unique().tolist())}. These are the "
                 "neutral/ensemble/positive personas the project replaced with the committee and "
                 "decision-head regimes; the committed file contains neither current regime. Scope: "
                 "the Streamlit dashboard does NOT read these numbers — it recomputes every regime "
                 "live from eval_table.csv (dashboard.py:160-170), so what you see in the app is "
                 "current. The stale file matters for anyone reading the CSV directly, quoting it "
                 "into a paper table, or diffing runs.",
                 ["outputs/eval_results.csv", "src/regimes/__init__.py",
                  "src/app/dashboard.py#L160-L170", "docs/PROJECT_OVERVIEW.md"],
                 "python src/analysis/run_eval.py   # after fixing ALL_REGIMES")

        # ...and the dashboard requires the file to exist while never using its contents
        if "df_static = load_results()" in dash and "df_static" not in dash.replace(
                "df_static = load_results()", ""):
            flag(S, "minor", "The dashboard loads eval_results.csv and never uses it",
                 "`df_static = load_results()` at dashboard.py:108 is assigned once and referenced "
                 "nowhere else, but its FileNotFoundError is what triggers the "
                 "'Run python src/analysis/run_eval.py first' hard stop. So the app refuses to start without "
                 "a file whose contents it ignores — which is also why nobody noticed the file went "
                 "stale.",
                 ["src/app/dashboard.py#L106-L112", "src/app/dashboard.py#L64-L65"])

    # bootstrap artifacts: same shape, and no accidental sensitivity run in the headline slot
    boots = {p: read(f"outputs/{p}") for p in os.listdir("outputs")
             if p.startswith("leakage_exclusion_bootstrap") and p.endswith(".csv")}
    for name, d in boots.items():
        if d is None:
            continue
        note(S, f"{name} rows", f"{len(d):,}")
        if {"lo", "hi", "point"} <= set(d.columns):
            bad = d[(d["point"] < d["lo"]) | (d["point"] > d["hi"])]
            if len(bad):
                flag(S, "major", f"Point estimate outside its own CI in {name}",
                     f"{len(bad):,} rows. The point estimate and the bootstrap percentiles were "
                     "not produced by the same run.",
                     [f"outputs/{name}", "src/analysis/leakage_exclusion_bootstrap.py"])
    shapes = {n: len(d) for n, d in boots.items() if d is not None}
    if len(set(shapes.values())) > 1:
        flag(S, "minor", "Bootstrap output files disagree in shape",
             f"{shapes}. The four source×venue-premium variants should be parallel; a differing row "
             "count usually means one file is a leftover sensitivity run, which is exactly the bug "
             "the last integrity audit found.",
             ["outputs/", "outputs/findings_integrity_check.md"])

    # is the reported headline reproducible from the files?
    ex = read("outputs/leakage_exclusion_eval.csv")
    ex2 = read("outputs/leakage_exclusion_eval_s2.csv")
    if ex is not None and ex2 is not None and len(ex) == len(ex2):
        note(S, "exclusion eval variants", "OpenAlex + S2 present, same shape")
        if "value" in ex and (ex["value"].reset_index(drop=True)
                              .equals(ex2["value"].reset_index(drop=True))):
            flag(S, "blocker", "The OpenAlex and S2 exclusion evals are byte-identical",
                 "Two files that should differ by ground-truth source carry the same values — one "
                 "was written with the wrong `--source` flag.",
                 ["outputs/leakage_exclusion_eval.csv", "outputs/leakage_exclusion_eval_s2.csv"])


# ------------------------------------------------------------------- stage 5: plumbing

def audit_plumbing():
    S = "5 · Plumbing & provenance"

    # staleness DAG
    stale = []
    for out, (script, inputs) in PRODUCERS.items():
        if not os.path.exists(out):
            continue
        t_out = os.path.getmtime(out)
        for src_path in list(inputs) + [script]:
            if os.path.exists(src_path) and os.path.getmtime(src_path) > t_out:
                stale.append((out, src_path,
                              datetime.fromtimestamp(t_out).date().isoformat(),
                              datetime.fromtimestamp(os.path.getmtime(src_path)).date().isoformat()))
    note(S, "stale output/input pairs", len(stale))
    if stale:
        lines = "; ".join(f"{o} ({to}) older than {i} ({ti})" for o, i, to, ti in stale[:12])
        flag(S, "major", "Outputs older than the inputs or code that produce them",
             f"{len(stale)} stale pairs. {lines}"
             + (" …" if len(stale) > 12 else "") +
             " Nothing in the repo enforces rebuild order, so a chart can be several source "
             "revisions behind the data it claims to show.",
             ["outputs/", "src/"],
             "ls -lt outputs/ | head -30")

    # convention: data written outside outputs/
    if os.path.isdir("output"):
        files = sorted(os.listdir("output"))
        flag(S, "major", "A second output directory `output/` sits outside the convention",
             f"CLAUDE.md states all generated files live in `outputs/`, but `output/` exists with "
             f"{files} — and `output/citations_2018_2020.csv` is a required *input* to "
             "build_eval_table.py, so the OpenAlex ground truth is the one artifact stored off-path. "
             "Easy to miss in a backup, easy to shadow with a typo.",
             ["CLAUDE.md", "src/build/build_eval_table.py#L53", "src/analysis/outlier_analysis.py#L27",
              "src/analysis/cite_hist.py#L9"],
             "ls -l output/")

    # scripts that write into data/ (declared read-only)
    writers = []
    for root, _, files in os.walk("src"):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            body = open(p, errors="ignore").read()
            if re.search(r"to_csv\(\s*[\"']data/", body) or re.search(r"open\(\s*[\"']data/[^\"']+[\"']\s*,\s*[\"'][wa]", body):
                writers.append(p)
    if writers:
        flag(S, "major", "Scripts write into data/, which CLAUDE.md declares read-only",
             f"{writers}", writers)

    # provenance of the critical inputs
    untracked = []
    for p in [DB, "data/archive/all_paper_results.csv", "output/citations_2018_2020.csv",
              "outputs/eval_table.csv", "outputs/s2_citations_full.csv",
              "outputs/paper_fields.csv"]:
        if not os.path.exists(p):
            continue
        r = subprocess.run(["git", "ls-files", "--error-unmatch", p],
                           capture_output=True, text=True)
        if r.returncode != 0:
            untracked.append(p)
    note(S, "critical data files not tracked by git", untracked or "none")
    if untracked:
        flag(S, "major", "Critical data files are untracked",
             f"{untracked} are not in git, so there is no version history for the inputs behind "
             "every committed number. A silent overwrite is unrecoverable and undetectable.",
             [".gitignore"] + untracked,
             "git status --short data outputs output | head")

    # file-type sanity: a .md that is not markdown
    for p in ["data/CLAUDE.md", "data/README.md", "docs/PROJECT_OVERVIEW.md"]:
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            head = fh.read(4)
        if head[:2] == b"PK":
            flag(S, "major", f"{p} is a ZIP archive, not markdown",
                 f"{p} starts with the PK zip magic bytes. Anything that opens it as project "
                 "documentation (including CLAUDE.md auto-loading, which reads data/CLAUDE.md as "
                 "instructions) gets binary garbage.",
                 p, f"file {p} && unzip -l {p} | head")

    # secrets hygiene
    if os.path.exists(".env"):
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"],
                                 capture_output=True, text=True).returncode == 0
        if tracked:
            flag(S, "blocker", ".env is tracked by git", "Secrets are in version control.", ".env")
    hard = []
    for root, _, files in os.walk("src"):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                body = open(p, errors="ignore").read()
                for m in re.finditer(r"[\"'](sk-[A-Za-z0-9_\-]{12,}|[A-Za-z0-9]{32,})[\"']", body):
                    hard.append(f"{p}:{body[:m.start()].count(chr(10)) + 1}")
    if hard:
        flag(S, "major", "Possible hardcoded credential in src/",
             f"Literal high-entropy strings at {hard[:10]}. Verify each is not a key.", hard[:10])


# ----------------------------------------------------------------------- reporting

PIPELINE_MAP = [
    ("Sources (nothing in the repo produces these)",
     ["data/gen_review.db — OpenReview scrape: SUBMISSION / REVIEW / GENAI_REVIEW",
      "data/archive/all_paper_results.csv + data/paper_manifest.csv — committee & decision-head LLM run",
      "data/OpenAlex/ — arXiv↔OpenAlex match, batch cache, RDD paper-level file",
      "OpenAlex API, Semantic Scholar API, Together AI, Anthropic (live fetches)"]),
    ("Fetch / enrich",
     ["fetch_citations_openalex.py → output/citations_2018_2020.csv",
      "fetch_citations_s2_full.py → outputs/s2_citations_full.csv",
      "fetch_rejected_venues_s2{,_title}.py → outputs/rejected_venues_s2*.csv",
      "fetch_author_stats.py → outputs/author_stats.csv, paper_author_ids.csv, paper_venues.csv",
      "fetch_pc_decisions.py → appends pc_decision_note to outputs/outlier_reviews.csv"]),
    ("Annotate (LLM-generated labels)",
     ["tag_fields.py → outputs/paper_fields.csv",
      "tag_rejection_reasons.py → rejection_tags column in outputs/outlier_reviews.csv",
      "leakage_lap_v1.py / leakage_fame_v1.py → LAP / FAME probe CSVs",
      "leakage_controls.py, leakage_masked_rereview.py, leakage_abstract_completion_v1.py"]),
    ("Transform / join",
     ["build_eval_table.py → outputs/eval_table.csv  (DB + citations + fields + GENAI ratings,"
      " plus field×year citation_pct_rank)",
      "build_author_covariates.py → outputs/paper_author_covariates.csv",
      "compare_citation_sources.py → outputs/citation_source_comparison.{csv,md}",
      "check_title_match_quality.py → outputs/title_match_quality.csv"]),
    ("Analyse / report",
     ["run_eval.py (regimes/ × metrics.py × baselines.py) → outputs/eval_results.csv",
      "leakage_exclusion_eval.py, leakage_threshold_sweep.py, leakage_exclusion_bootstrap.py",
      "fuzzy_rdd.py, hetero_analysis.py, outlier_analysis.py, table1_summary_stats.py",
      "dashboard.py (Streamlit), viz_*.py, cite_hist.py"]),
]

SEV_ORDER = {"blocker": 0, "major": 1, "minor": 2, "info": 3}
SEV_LABEL = {"blocker": "Blocker", "major": "Major", "minor": "Minor", "info": "Info"}


def where_link(w):
    """'src/x.py#L12' -> (path, display, anchor)"""
    m = re.match(r"([^#]+)#L(\d+)(?:-L?(\d+))?$", w)
    if m:
        return m.group(1), f"{m.group(1)}:{m.group(2)}" + (f"-{m.group(3)}" if m.group(3) else ""), w
    return w, w, w


def write_markdown(path):
    ordered = sorted(ISSUES, key=lambda i: (SEV_ORDER[i["severity"]], i["stage"]))
    counts = {s: sum(1 for i in ISSUES if i["severity"] == s) for s in SEV_ORDER}
    L = [f"# Data audit — CitesBench",
         "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
         "`python src/audit/data_audit.py`.",
         "",
         f"**{counts['blocker']} blockers · {counts['major']} major · "
         f"{counts['minor']} minor · {counts['info']} info**",
         "",
         "Severity means: **blocker** — a number in the paper is wrong or not reproducible; "
         "**major** — a real bias or fragility that must be disclosed or fixed; "
         "**minor** — a defensible choice that is currently undocumented; **info** — context.",
         "",
         "## Pipeline map", ""]
    for head, items in PIPELINE_MAP:
        L.append(f"**{head}**")
        L.append("")
        L += [f"- `{it}`" for it in items]
        L.append("")
    L += ["## Findings", ""]
    for n, i in enumerate(ordered, 1):
        L.append(f"### {n}. [{SEV_LABEL[i['severity']]}] {i['title']}")
        L.append(f"*Stage {i['stage']}*")
        L.append("")
        L.append(i["detail"])
        L.append("")
        L.append("**Where to look:** " + ", ".join(
            f"[`{d}`]({p})" for p, d, _ in (where_link(w) for w in i["where"])))
        if i["how"]:
            L.append("")
            L.append("**Reproduce:**")
            L.append("```bash")
            L.append(i["how"])
            L.append("```")
        L.append("")
    L += ["## Stage statistics", ""]
    for stage in sorted(STAGE_NOTES):
        L.append(f"### {stage}")
        L.append("")
        L.append("| measure | value |")
        L.append("|---|---|")
        for k, v in STAGE_NOTES[stage]:
            L.append(f"| {k} | {v.replace('|', '\\|')} |")
        L.append("")
    open(path, "w").write("\n".join(L))


HTML_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e0dc;--card:#fff;
--blocker:#b3261e;--major:#b06000;--minor:#5c6bc0;--info:#5f7a5f;}
@media(prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eceae6;--mut:#9a9791;
--line:#2e2e2e;--card:#1c1c1c;--blocker:#f2837b;--major:#e0a458;--minor:#9fa8da;--info:#9bbf9b;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:28px;margin:0 0 6px}h2{font-size:19px;margin:44px 0 14px;
padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 26px}
.tally{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 8px}
.pill{border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:13px;
background:var(--card);cursor:pointer;user-select:none}
.pill.off{opacity:.35}.pill b{font-variant-numeric:tabular-nums}
.legend{color:var(--mut);font-size:13px;margin:14px 0 0}
.issue{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--mut);
border-radius:6px;padding:14px 16px;margin:12px 0}
.issue[data-sev=blocker]{border-left-color:var(--blocker)}
.issue[data-sev=major]{border-left-color:var(--major)}
.issue[data-sev=minor]{border-left-color:var(--minor)}
.issue[data-sev=info]{border-left-color:var(--info)}
.issue h3{margin:0;font-size:16px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.tag{font-size:11px;letter-spacing:.06em;text-transform:uppercase;font-weight:700}
[data-sev=blocker] .tag{color:var(--blocker)}[data-sev=major] .tag{color:var(--major)}
[data-sev=minor] .tag{color:var(--minor)}[data-sev=info] .tag{color:var(--info)}
.stage{color:var(--mut);font-size:12px;margin:4px 0 10px}
.issue p{margin:0 0 10px}
.where{font-size:13px;color:var(--mut)}
code{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
background:rgba(127,127,127,.12);padding:1px 5px;border-radius:3px}
pre{overflow-x:auto;background:rgba(127,127,127,.10);padding:10px 12px;border-radius:5px;
font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;margin:8px 0 0}
details summary{cursor:pointer;color:var(--mut);font-size:13px;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);
vertical-align:top}th{color:var(--mut);font-weight:600}
td:last-child{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;word-break:break-word}
.scroll{overflow-x:auto}
"""

HTML_JS = """
document.querySelectorAll('.pill[data-filter]').forEach(p=>p.onclick=()=>{
  p.classList.toggle('off');
  const on=[...document.querySelectorAll('.pill[data-filter]:not(.off)')]
    .map(x=>x.dataset.filter);
  document.querySelectorAll('.issue').forEach(i=>{
    i.style.display=on.includes(i.dataset.sev)?'':'none';});
});
"""


def write_html(path):
    e = html.escape
    ordered = sorted(ISSUES, key=lambda i: (SEV_ORDER[i["severity"]], i["stage"]))
    counts = {s: sum(1 for i in ISSUES if i["severity"] == s) for s in SEV_ORDER}
    P = [f"<title>Data audit — CitesBench</title><style>{HTML_CSS}</style>",
         "<div class=wrap>",
         "<h1>Data audit — CitesBench</h1>",
         f"<p class=sub>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
         "by <code>python src/audit/data_audit.py</code>. Click a pill to filter.</p>",
         "<div class=tally>"]
    for s in SEV_ORDER:
        P.append(f"<span class=pill data-filter={s}><b>{counts[s]}</b> {SEV_LABEL[s].lower()}</span>")
    P += ["</div>",
          "<p class=legend><b>blocker</b> — a reported number is wrong or not reproducible · "
          "<b>major</b> — a real bias or fragility to fix or disclose · "
          "<b>minor</b> — defensible but undocumented · <b>info</b> — context.</p>",
          "<h2>Pipeline map</h2>"]
    for head, items in PIPELINE_MAP:
        P.append(f"<h3>{e(head)}</h3><ul>")
        P += [f"<li><code>{e(it)}</code></li>" for it in items]
        P.append("</ul>")
    P.append("<h2>Findings</h2>")
    for n, i in enumerate(ordered, 1):
        P.append(f"<div class=issue data-sev={i['severity']}>")
        P.append(f"<h3><span class=tag>{SEV_LABEL[i['severity']]}</span>"
                 f"<span>{n}. {e(i['title'])}</span></h3>")
        P.append(f"<div class=stage>Stage {e(i['stage'])}</div>")
        P.append(f"<p>{e(i['detail'])}</p>")
        links = " · ".join(f"<code>{e(d)}</code>" for _, d, _ in
                           (where_link(w) for w in i["where"]))
        P.append(f"<div class=where>Where to look: {links}</div>")
        if i["how"]:
            P.append("<details><summary>Reproduce</summary>"
                     f"<pre>{e(i['how'])}</pre></details>")
        P.append("</div>")
    P.append("<h2>Stage statistics</h2>")
    for stage in sorted(STAGE_NOTES):
        P.append(f"<h3>{e(stage)}</h3><div class=scroll><table>"
                 "<tr><th>measure</th><th>value</th></tr>")
        for k, v in STAGE_NOTES[stage]:
            P.append(f"<tr><td>{e(k)}</td><td>{e(v)}</td></tr>")
        P.append("</table></div>")
    P.append(f"</div><script>{HTML_JS}</script>")
    open(path, "w").write("\n".join(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--json", action="store_true", help="also dump findings as JSON")
    args = ap.parse_args()

    src = audit_source()
    sub, rev, gen = src if src else (None, None, None)
    audit_external(sub)
    audit_annotations(sub)
    audit_join(sub, rev)
    audit_results()
    audit_plumbing()

    write_markdown("outputs/data_audit.md")
    if not args.no_html:
        write_html("outputs/data_audit.html")
    if args.json:
        json.dump(ISSUES, open("outputs/data_audit.json", "w"), indent=1)

    counts = {s: sum(1 for i in ISSUES if i["severity"] == s) for s in SEV_ORDER}
    print(f"{len(ISSUES)} findings — " +
          ", ".join(f"{counts[s]} {s}" for s in SEV_ORDER))
    for i in sorted(ISSUES, key=lambda i: SEV_ORDER[i["severity"]]):
        print(f"  [{i['severity']:<7}] {i['stage'][:1]} · {i['title']}")
    print("\nWrote outputs/data_audit.md" + ("" if args.no_html else " + outputs/data_audit.html"))


def _selfcheck():
    """Smallest thing that fails if the report plumbing breaks."""
    ISSUES.clear()
    flag("0 · t", "blocker", "T", "d", ["src/x.py#L4-L9"], "echo hi")
    assert where_link("src/x.py#L4-L9") == ("src/x.py", "src/x.py:4-9", "src/x.py#L4-L9")
    write_markdown("/tmp/_a.md"); write_html("/tmp/_a.html")
    assert "src/x.py:4-9" in open("/tmp/_a.md").read()
    assert "data-sev=blocker" in open("/tmp/_a.html").read()
    print("selfcheck ok")


if __name__ == "__main__":
    if os.environ.get("AUDIT_SELFCHECK"):
        _selfcheck()
    else:
        main()
