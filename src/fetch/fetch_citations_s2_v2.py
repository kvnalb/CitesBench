"""
Semantic Scholar citation refetch, v2 — one uniform code path for every paper.

Why v2: v1 (fetch_citations_s2_full.py) reused outputs/rejected_venues_s2_title.csv
wholesale as `method=title_cached`. That block is 1,832 papers, 100% rejected, 0
accepted, median 1 citation and 38% zero/null, with no year and no venue recorded —
while rejects fetched through the other two paths have median 25-42. So the accept
side and the reject side of the outcome variable were produced by different
pipelines, and the accept-reject citation gap moves 2.3x depending on which one you
believe. Coverage asymmetry was the exact flaw S2 was brought in to fix for OpenAlex.

v2 fetches all 4,567 papers through the same rules, records the evidence needed to
verify each match (S2 paperId, externalIds, authors, year, venue, publicationTypes),
and probes for duplicate stub records wherever a paper comes back with <=1 citation.

Match tiers (assigned in --report, so thresholds are tunable without refetching):
  A  arXiv-ID or DOI match. Title drift allowed (papers get retitled) but flagged.
  B  title_sim >= 0.95, s2_year within [-1,+3] of submission, >=1 shared author.
  C  anything weaker — excluded from the primary outcome, kept for sensitivity.

Collisions (issue #46): all 26 in this sample are cross-year resubmissions sharing
one arXiv preprint. The accepted submission wins the record, because citations
accrue to the published version; see the comment in assign_tiers for the AdamW case
that made the old title_sim tie-break give a rejected paper 2,738 citations.

Output: outputs/s2_citations_v2.csv (incremental, resumable)
Report: outputs/s2_attribution_report.md  (python src/fetch/fetch_citations_s2_v2.py --report)

Run: python src/fetch/fetch_citations_s2_v2.py [--limit N] [--report]

# ponytail: no S2 key in .env, so SLEEP is the unauthenticated 100-req/5-min budget.
# Set SEMANTIC_SCHOLAR_API_KEY to drop it to 1.0s (~1h instead of ~4h).
"""
import os
import re
import csv
import sys
import time
import json
import argparse
from difflib import SequenceMatcher

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OUT_CSV = "outputs/s2_citations_v2.csv"
REPORT = "outputs/s2_attribution_report.md"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = ("paperId,externalIds,title,year,venue,publicationTypes,"
          "citationCount,influentialCitationCount,authors.name")

S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
HEADERS = {"x-api-key": S2_KEY} if S2_KEY else {}
SLEEP = 1.0 if S2_KEY else 3.1

COLS = ["paper_id", "query_method", "probe_used", "s2_paper_id", "s2_arxiv", "s2_doi",
        "s2_corpus_id", "s2_title", "title_sim", "author_overlap", "n_known_authors",
        "s2_year", "s2_venue", "s2_pubtypes", "s2_citations", "s2_influential"]


def norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).strip()


def sim(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def last_names(authors):
    """'Yann LeCun; Y. Bengio' -> {'lecun', 'bengio'}. Initials are dropped."""
    out = set()
    for a in authors:
        parts = [p for p in re.split(r"[^A-Za-z\-]+", str(a)) if len(p) > 1]
        if parts:
            out.add(parts[-1].lower())
    return out


def get(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(SLEEP * (i + 2))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 503):        # rate limited — back off and retry
            time.sleep(SLEEP * (i + 2) * 2)
            continue
        return None                            # 404 = no match, not an error
    return None


def load_inputs(min_cos=0.70, eval_table="outputs/eval_table.csv"):
    """arXiv IDs come from the two-pass resolver (exact-title, then abstract-verified
    fuzzy), not from the old RDD-subsample file — that file only ever covered 1,579 of
    the 4,567 papers and its coverage was correlated with the decision."""
    ev = pd.read_csv(eval_table, low_memory=False)[["paper_id", "title", "year", "decision"]]

    ids, known, id_source = {}, {}, {}
    res_path = "outputs/arxiv_resolution.csv"
    if not os.path.exists(res_path):
        sys.exit("Run src/fetch/resolve_arxiv_ids.py first.")
    res = pd.read_csv(res_path, low_memory=False,
                      dtype={"arxiv_id": str, "paper_id": str})
    res = res[res["matched"].eq(1)]
    for r in res.itertuples():
        ids[r.paper_id] = r.arxiv_id
        id_source[r.paper_id] = f"arxiv_{r.match_rule}"
        if isinstance(getattr(r, "arxiv_authors", None), str):
            known.setdefault(r.paper_id, set()).update(
                last_names(re.split(r",| and ", r.arxiv_authors)))

    fz_path = "outputs/arxiv_fuzzy_candidates.csv"
    if os.path.exists(fz_path):
        fz = pd.read_csv(fz_path, dtype={"arxiv_id": str, "paper_id": str})
        fz = fz[fz["rank"].eq(1) & (fz["abstract_cos"] >= min_cos)]
        for r in fz.itertuples():
            if r.paper_id in ids:          # exact pass wins
                continue
            ids[r.paper_id] = r.arxiv_id
            id_source[r.paper_id] = f"arxiv_fuzzy_cos{min_cos}"
            if isinstance(r.arxiv_authors, str):
                known.setdefault(r.paper_id, set()).update(
                    last_names(re.split(r",| and ", r.arxiv_authors)))

    # DOIs as a secondary identifier where OpenAlex has one.
    dois = {}
    ax_path = "data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"
    if os.path.exists(ax_path):
        ax = pd.read_csv(ax_path, low_memory=False).drop_duplicates("paper_id")
        if "openalex_doi" in ax:
            dois = (ax.set_index("paper_id")["openalex_doi"].dropna()
                    .str.replace(r"^https?://doi\.org/", "", regex=True).to_dict())

    # Author names from OpenAlex too — a source independent of both arXiv and S2.
    # OpenAlex author names, and ReviewArena's. Neither alone is enough: OpenAlex
    # covers 3,264 papers all in 2018-2020, ReviewArena covers 2020 onward. A 2025
    # paper with no arXiv preprint has authors from ReviewArena only — and that is
    # precisely the population reaching the title-match path, so without it every
    # 2025 title match scores author_overlap=0 and assign_tiers demotes it to C.
    for pa_path in ("outputs/paper_author_ids.csv",
                    "outputs/paper_author_names_reviewarena.csv",
                    "outputs/paper_author_names_openreview.csv"):   # 2018/2019
        if os.path.exists(pa_path):
            pa = pd.read_csv(pa_path)
            for pid, g in pa.groupby("paper_id")["author_name"]:
                known.setdefault(pid, set()).update(last_names(g.dropna().tolist()))

    in_scope = set(ev["paper_id"]) & set(ids)
    print(f"  arXiv IDs available for {len(in_scope):,}/{len(ev):,} in-scope papers "
          f"({len(in_scope)/len(ev):.1%})")
    return ev, ids, dois, known


def record(pid, method, probe, res, title, known_auth):
    """One output row. res is an S2 paper dict or None."""
    if not res:
        return dict(zip(COLS, [pid, method, int(probe)] + [""] * 13))
    ext = res.get("externalIds") or {}
    s2_auth = last_names([a.get("name", "") for a in (res.get("authors") or [])])
    overlap = len(s2_auth & known_auth) if known_auth else ""
    return {
        "paper_id": pid, "query_method": method, "probe_used": int(probe),
        "s2_paper_id": res.get("paperId") or "",
        "s2_arxiv": ext.get("ArXiv") or "", "s2_doi": ext.get("DOI") or "",
        "s2_corpus_id": ext.get("CorpusId") or "",
        "s2_title": res.get("title") or "",
        "title_sim": round(sim(title, res.get("title") or ""), 4),
        "author_overlap": overlap, "n_known_authors": len(known_auth) if known_auth else 0,
        "s2_year": res.get("year") or "", "s2_venue": res.get("venue") or "",
        "s2_pubtypes": "|".join(res.get("publicationTypes") or []),
        "s2_citations": res.get("citationCount"),
        "s2_influential": res.get("influentialCitationCount"),
    }


def stub_probe(title, known_auth):
    """S2 often keeps an orphan stub next to the merged record. Search by title and
    take the best-attributed record, preferring the one with the most citations."""
    js = get(SEARCH_URL, {"query": title, "limit": 5, "fields": FIELDS})
    time.sleep(SLEEP)
    best, best_key = None, None
    for c in (js or {}).get("data", []) or []:
        s = sim(title, c.get("title") or "")
        if s < 0.90:
            continue
        ov = len(last_names([a.get("name", "") for a in (c.get("authors") or [])]) & known_auth) \
            if known_auth else 0
        key = (ov > 0, c.get("citationCount") or 0, s)
        if best_key is None or key > best_key:
            best, best_key = c, key
    return best


def fetch(limit=None, ids_only=False, eval_table="outputs/eval_table.csv", out_csv=None):
    global OUT_CSV
    if out_csv:
        OUT_CSV = out_csv
    ev, ids, dois, known = load_inputs(eval_table=eval_table)
    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV)["paper_id"])
    todo = ev[~ev["paper_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(done):,} already fetched, {len(todo):,} to go "
          f"(sleep {SLEEP}s/req, no API key)" if not S2_KEY else
          f"{len(done):,} done, {len(todo):,} to go")

    new = not os.path.exists(OUT_CSV)
    fh = open(OUT_CSV, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()
        fh.flush()

    # --- pass 1: identifier batch (arXiv id, else DOI) --------------------------
    by_id = [(r.paper_id, r.title, f"ARXIV:{ids[r.paper_id]}") for r in todo.itertuples()
             if r.paper_id in ids] + \
            [(r.paper_id, r.title, f"DOI:{dois[r.paper_id]}") for r in todo.itertuples()
             if r.paper_id not in ids and r.paper_id in dois]
    resolved, failed = {}, set()
    for i in range(0, len(by_id), 500):
        chunk = by_id[i:i + 500]
        res, ok = None, False
        for attempt in range(5):
            try:
                r = requests.post(BATCH_URL, params={"fields": FIELDS},
                                  json={"ids": [c[2] for c in chunk]},
                                  headers=HEADERS, timeout=90)
            except requests.RequestException:
                time.sleep(SLEEP * (attempt + 2) * 2)
                continue
            if r.status_code == 200:
                res, ok = r.json(), True
                break
            time.sleep(SLEEP * (attempt + 2) * 2)
        if not ok or not isinstance(res, list):
            # Do NOT record these as misses — an API failure is not a missing paper.
            # They stay out of the done-set and the next run retries them.
            failed.update(c[0] for c in chunk)
            print(f"  batch {i // 500 + 1}: FAILED after retries, {len(chunk)} papers deferred")
            time.sleep(SLEEP)
            continue
        for (pid, title, _), item in zip(chunk, res):
            resolved[pid] = item
        print(f"  batch {i // 500 + 1}: {sum(x is not None for x in res)}/{len(chunk)} resolved")
        time.sleep(SLEEP)

    # --- pass 2: per paper — title match where no id, then stub probe -----------
    for n, r in enumerate(todo.itertuples(), 1):
        pid, title = r.paper_id, r.title
        if pid in failed:
            continue
        ka = known.get(pid, set())
        if pid in resolved:
            res, method = resolved[pid], ("arxiv_id" if pid in ids else "doi")
        elif ids_only:
            continue                       # leave for the keyed run; don't half-write it
        else:
            res = get(MATCH_URL, {"query": title, "fields": FIELDS})
            res = (res or {}).get("data", [None])[0] if isinstance(res, dict) else None
            method = "title_match"
            time.sleep(SLEEP)

        probe = False
        cites = (res or {}).get("citationCount")
        if not ids_only and (res is None or (cites is not None and cites <= 1)):
            alt = stub_probe(title, ka)
            if alt and (res is None or (alt.get("citationCount") or 0) > (cites or 0)):
                res, probe = alt, True

        w.writerow(record(pid, method, probe, res, title, ka))
        fh.flush()
        if n % 25 == 0:
            print(f"  {n:,}/{len(todo):,}  {pid}  cites={(res or {}).get('citationCount')}"
                  f"{' [probed]' if probe else ''}", flush=True)
    fh.close()
    print(f"Done. Wrote {OUT_CSV}")


# ------------------------------------------------------------------ tiering + report

def assign_tiers(df, ev):
    d = df.merge(ev[["paper_id", "year", "decision"]], on="paper_id", how="left")
    d["accepted"] = d["decision"].fillna("").str.startswith("Accept")
    d["dy"] = pd.to_numeric(d["s2_year"], errors="coerce") - d["year"]
    sim_ok = d["title_sim"].fillna(0) >= 0.95
    yr_ok = d["dy"].between(-1, 3)
    au_ok = pd.to_numeric(d["author_overlap"], errors="coerce").fillna(0) > 0

    d["tier"] = "C"
    d.loc[sim_ok & yr_ok & au_ok, "tier"] = "B"
    d.loc[d["query_method"].isin(["arxiv_id", "doi"]) & d["s2_paper_id"].notna()
          & d["s2_paper_id"].ne(""), "tier"] = "A"
    d.loc[d["s2_paper_id"].isna() | d["s2_paper_id"].eq(""), "tier"] = "none"
    d["title_changed"] = (d["tier"].eq("A") & (d["title_sim"].fillna(1) < 0.90))

    # Collisions: one S2 record claimed by several ICLR papers. Every one of the 26
    # groups in the 2018-2020 sample is a CROSS-YEAR RESUBMISSION — the same paper
    # sent to ICLR twice, sharing one arXiv preprint and therefore one S2 record.
    # None are same-year. So this is not an S2 defect and not a matching error.
    #
    # `accepted` sorts FIRST, and that ordering is the fix for issue #46. Sorting on
    # title_sim alone hands the citations to whichever of our titles matches S2's
    # stored title, and that title goes stale. arXiv 1711.05101 went to ICLR 2018 as
    # "Fixing Weight Decay Regularization in Adam" (rejected), was retitled
    # "Decoupled Weight Decay Regularization", and was accepted at ICLR 2019. S2
    # still stores the old title, so title_sim scored the REJECTED submission 1.0 and
    # the accepted one 0.709: the rejected paper took all 2,738 citations and the
    # accepted paper got NaN. Two harms in the same direction, on the exact axis this
    # benchmark measures. Deep Imitative Models (160) and Double Neural CFR (54)
    # failed the same way.
    #
    # An S2 record's citations accrue to the PUBLISHED version. If one colliding
    # submission was accepted at ICLR, that is the published version, whatever title
    # S2 kept. Losers still drop to tier C: we cannot attribute a record to a
    # submission it does not describe, and NaN is the honest answer.
    dup = d[d["s2_paper_id"].astype(str).ne("")].groupby("s2_paper_id")["paper_id"].nunique()
    clash = set(dup[dup > 1].index)
    d["collision"] = d["s2_paper_id"].isin(clash)
    for sid, g in d[d["collision"]].groupby("s2_paper_id"):
        keep = g.sort_values(["accepted", "title_sim", "author_overlap"],
                             ascending=False).index[0]
        d.loc[[i for i in g.index if i != keep], "tier"] = "C"

    check_collisions(d)
    return d


def check_collisions(d):
    """The invariant issue #46 broke: no collision group may pay a rejected
    submission while an accepted sibling is demoted to tier C."""
    bad = []
    for sid, g in d[d["collision"]].groupby("s2_paper_id"):
        win = g[g["tier"].isin(["A", "B"])]
        if len(win) == 1 and not win.iloc[0]["accepted"] \
                and g[g["tier"].eq("C")]["accepted"].any():
            bad.append((sid, win.iloc[0]["paper_id"]))
    assert not bad, f"collision pays a rejected paper over an accepted one: {bad}"

    n_acc = int(d[d["collision"]].groupby("s2_paper_id")["accepted"].any().sum())
    print(f"collisions: {int(d['collision'].sum())} papers over "
          f"{d.loc[d['collision'], 's2_paper_id'].nunique()} S2 records, "
          f"{n_acc} groups containing an accepted paper, "
          f"{int((d['collision'] & d['tier'].eq('C')).sum())} demoted to tier C")


def report(eval_table="outputs/eval_table.csv"):
    # ponytail: tiered/report paths derive from OUT_CSV so a 2025 report cannot
    # clobber the 2018-2020 one. Both were hardcoded.
    tiered = OUT_CSV.replace(".csv", "_tiered.csv")
    report_md = OUT_CSV.replace(".csv", "_attribution.md")
    df = pd.read_csv(OUT_CSV)
    ev = pd.read_csv(eval_table, low_memory=False)[["paper_id", "year", "decision"]]
    d = assign_tiers(df, ev)
    d.to_csv(tiered, index=False)

    import numpy as np
    tier_by_dec = pd.crosstab(d["tier"], d["accepted"], normalize="columns").round(3)
    counts = pd.crosstab(d["tier"], d["accepted"])
    usable = d[d["tier"].isin(["A", "B"])]

    def stats(g):
        c = pd.to_numeric(g["s2_citations"], errors="coerce")
        return pd.Series({"n": len(g), "median": c.median(),
                          "mean_log1p": np.log1p(c.dropna()).mean(),
                          "pct_zero_or_null": 100 * c.fillna(-1).le(0).mean()})

    ids_only = not d["query_method"].eq("title_match").any()
    L = ["# S2 attribution report (v2)", "",
         ("**ID-matched papers only** — title matching and stub probes are deferred until the "
          "API key arrives, so the papers missing here are disproportionately rejected ones.\n"
          if ids_only else ""),
         f"{len(d):,} papers fetched through one code path. "
         f"Tier A = ID match, B = verified title match, C = weak/demoted, none = no record.", "",
         "## Tier composition by decision", "",
         "| tier | rejected | accepted | n rejected | n accepted |", "|---|---|---|---|---|"]
    for t in ["A", "B", "C", "none"]:
        if t in tier_by_dec.index:
            L.append(f"| {t} | {tier_by_dec.loc[t, False]:.1%} | {tier_by_dec.loc[t, True]:.1%} | "
                     f"{counts.loc[t, False]:,} | {counts.loc[t, True]:,} |")
    gapA = abs(tier_by_dec.loc["A", True] - tier_by_dec.loc["A", False]) if "A" in tier_by_dec.index else 0
    L += ["", f"**Tier A share differs by {gapA:.1%} across the decision boundary.** "
              "Anything above a few points has to be disclosed and carried as a sensitivity arm — "
              "it is the same asymmetry that disqualified the OpenAlex counts.", "",
          "## Citation distribution, usable tiers (A+B) only", "",
          "| group | n | median | mean log1p | % zero/null |", "|---|---|---|---|---|"]
    for lab, g in [("rejected", usable[~usable["accepted"]]), ("accepted", usable[usable["accepted"]])]:
        s = stats(g)
        L.append(f"| {lab} | {s['n']:.0f} | {s['median']:.1f} | {s['mean_log1p']:.3f} | "
                 f"{s['pct_zero_or_null']:.1f}% |")
    L += ["", "## Diagnostics", "",
          f"- stub probe changed the record for **{int(d['probe_used'].fillna(0).sum()):,}** papers",
          f"- ID matches where the title changed between submission and publication: "
          f"**{int(d['title_changed'].sum()):,}** (expected; not errors)",
          f"- S2 records claimed by more than one ICLR paper: "
          f"**{int(d['collision'].sum()):,}** papers involved, "
          f"**{int((d['collision'] & d['tier'].eq('C')).sum()):,}** demoted to tier C",
          f"- papers with no S2 record at all: **{int(d['tier'].eq('none').sum()):,}**",
          f"- author-overlap check possible for **{int((d['n_known_authors'] > 0).sum()):,}** papers "
          f"({100 * (d['n_known_authors'] > 0).mean():.0f}%)", ""]
    open(report_md, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {report_md} and {tiered}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--eval-table", default="outputs/eval_table.csv")
    ap.add_argument("--out")
    ap.add_argument("--ids-only", action="store_true",
                    help="fetch only papers with an arXiv/DOI id (works without an API key); "
                         "skip title matching and stub probes until the key arrives")
    a = ap.parse_args()
    if a.out:
        OUT_CSV = a.out
    report(a.eval_table) if a.report else fetch(a.limit, a.ids_only, a.eval_table, a.out)
