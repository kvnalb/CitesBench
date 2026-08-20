"""
Fetch the rich Semantic Scholar metadata for paper IDs we already resolved (issue #44).

WHY THIS IS CHEAP. src/fetch/fetch_citations_s2_v2.py already matched 4,477 of the
4,567 papers to an S2 paper ID and recorded the match tier. It asked S2 for nine
fields, because its job was to count citations. This script reuses those IDs and
asks for everything else. It does NO matching: an ID that script did not vet does
not get queried here, by title or otherwise. 4,477 IDs at 500 per batch is about
ten POSTs.

WHAT COMES BACK. Measured on a 60-ID probe (30 accepted, 30 rejected), everything
below lands at 100% except author affiliations at 31.7%, which is worse than the
39.5% the OpenAlex pull already gives us. Institutions are not fixable from here.

WHY FIELD PROVENANCE STAYS SPLIT. s2FieldsOfStudy carries a `source` per label:
`external` (from the publisher) and `s2-fos-model` (S2's own classifier). Merging
them would hide which papers are labelled by a model. They stay in two columns,
and s2_primary_field prefers the external label.

POST-TREATMENT WARNING. s2_*_h_index and s2_sum_author_citation_count are read at
fetch time, in 2026. For a 2018 submission they are downstream of the decision:
acceptance raised them. They are fine as descriptive columns and are NOT valid as
pre-determined covariates in a balance test without an as-of-submission
reconstruction. The OpenAlex covariates in this repo have the same defect.

THE OUTCOME VARIABLE IS NOT TOUCHED. citationCount comes back in the same response
and is written as s2_citations_refetched so drift against the frozen
`openalex_citations` can be inspected. It is never substituted for it.

S2 REDIRECTS. Two of the 4,451 IDs come back under a different canonical paperId,
because S2 has merged those arXiv preprints into unrelated records (one into a 2014
paper literally titled "Deep neural networks", one into a materials-science paper on
silicon colour design). Both were tier A at title_sim 1.0, so this is S2's merge, not
our matcher's. The join key is therefore the ID we ASKED for, kept in s2_paper_id,
with whatever came back recorded in s2_paper_id_returned. Rows where the two differ
carry metadata for the wrong paper and should be dropped by any consumer; the demo
counts them.

RESUMABLE. Each chunk is appended to the CSV as it returns, per the external-API
rule in CLAUDE.md. On restart the script reads the output and skips IDs already
present.

Run: python src/fetch/fetch_s2_metadata.py [--limit 500]
"""
import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs", exist_ok=True)

MATCH_CSV = "outputs/s2_citations_v2_authored_tiered.csv"
OUT_CSV = "outputs/s2_paper_metadata.csv"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

FIELDS = ("paperId,title,abstract,year,publicationDate,venue,publicationVenue,journal,"
          "publicationTypes,externalIds,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,"
          "tldr,citationCount,influentialCitationCount,referenceCount,isOpenAccess,"
          "authors.authorId,authors.name,authors.affiliations,authors.hIndex,"
          "authors.paperCount,authors.citationCount")

CHUNK = 500
S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
HEADERS = {"x-api-key": S2_KEY} if S2_KEY else {}
SLEEP = 1.0 if S2_KEY else 3.1

COLS = ["s2_paper_id", "s2_paper_id_returned", "s2_title", "s2_abstract", "s2_tldr", "s2_year",
        "s2_publication_date", "s2_venue_name", "s2_venue_type", "s2_journal",
        "s2_pubtypes", "s2_is_open_access", "s2_oa_pdf_url",
        "s2_fields_external", "s2_fields_model", "s2_primary_field",
        "s2_reference_count", "s2_citations_refetched", "s2_influential_refetched",
        "n_s2_authors", "s2_first_author_h_index", "s2_max_h_index",
        "s2_mean_h_index", "s2_sum_author_paper_count",
        "s2_sum_author_citation_count", "s2_affiliations", "s2_arxiv_id", "s2_doi"]


def _fields(rec):
    """Split s2FieldsOfStudy by its `source`, keeping model labels distinguishable."""
    ext, mod = [], []
    for f in rec.get("s2FieldsOfStudy") or []:
        cat, src = f.get("category"), f.get("source")
        if not cat:
            continue
        (ext if src == "external" else mod).append(cat)
    # fieldsOfStudy is the older flat list; fold it into external, it has no model tag
    for cat in rec.get("fieldsOfStudy") or []:
        if cat not in ext:
            ext.append(cat)
    return sorted(set(ext)), sorted(set(mod))


def flatten(rec, requested):
    """One S2 paper record -> one flat row, keyed on the ID we asked for. Author
    fields collapse here so the table stays one row per paper and the downstream
    merge can be 1:1."""
    ext, mod = _fields(rec)
    authors = rec.get("authors") or []
    h = [a["hIndex"] for a in authors if a.get("hIndex") is not None]
    pc = [a["paperCount"] for a in authors if a.get("paperCount") is not None]
    cc = [a["citationCount"] for a in authors if a.get("citationCount") is not None]
    affs = sorted({s for a in authors for s in (a.get("affiliations") or [])})
    venue = rec.get("publicationVenue") or {}
    journal = rec.get("journal") or {}
    ids = rec.get("externalIds") or {}
    oa = rec.get("openAccessPdf") or {}
    tldr = rec.get("tldr") or {}
    return {
        "s2_paper_id": requested,
        "s2_paper_id_returned": rec.get("paperId"),
        "s2_title": rec.get("title"),
        "s2_abstract": rec.get("abstract"),
        "s2_tldr": tldr.get("text"),
        "s2_year": rec.get("year"),
        "s2_publication_date": rec.get("publicationDate"),
        "s2_venue_name": venue.get("name") or rec.get("venue"),
        "s2_venue_type": venue.get("type"),
        "s2_journal": journal.get("name"),
        "s2_pubtypes": "; ".join(rec.get("publicationTypes") or []) or None,
        "s2_is_open_access": rec.get("isOpenAccess"),
        "s2_oa_pdf_url": oa.get("url"),
        "s2_fields_external": "; ".join(ext) or None,
        "s2_fields_model": "; ".join(mod) or None,
        # external label wins; the model label is the fallback, never a silent merge
        "s2_primary_field": (ext or mod or [None])[0],
        "s2_reference_count": rec.get("referenceCount"),
        "s2_citations_refetched": rec.get("citationCount"),
        "s2_influential_refetched": rec.get("influentialCitationCount"),
        "n_s2_authors": len(authors) or None,
        "s2_first_author_h_index": authors[0].get("hIndex") if authors else None,
        "s2_max_h_index": max(h) if h else None,
        "s2_mean_h_index": sum(h) / len(h) if h else None,
        "s2_sum_author_paper_count": sum(pc) if pc else None,
        "s2_sum_author_citation_count": sum(cc) if cc else None,
        "s2_affiliations": "; ".join(affs) or None,
        "s2_arxiv_id": ids.get("ArXiv"),
        "s2_doi": ids.get("DOI"),
    }


def wanted_ids(limit=None):
    m = pd.read_csv(MATCH_CSV)
    ids = m.loc[m.s2_paper_id.notna(), "s2_paper_id"].drop_duplicates().tolist()
    return ids[:limit] if limit else ids


def fetch(limit=None):
    ids = wanted_ids(limit)
    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV, usecols=["s2_paper_id"]).s2_paper_id)
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} IDs wanted, {len(done)} already fetched, {len(todo)} to go")

    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        r = requests.post(BATCH_URL, params={"fields": FIELDS},
                          json={"ids": chunk}, headers=HEADERS, timeout=120)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} on chunk at {start}: {r.text[:200]}",
                  file=sys.stderr)
            time.sleep(SLEEP * 3)
            continue
        # S2 returns records in request order, null for anything it cannot resolve
        got = r.json()
        rows = [flatten(rec, rid) for rid, rec in zip(chunk, got) if rec]
        # append immediately: one chunk on disk is one chunk we never refetch
        pd.DataFrame(rows, columns=COLS).to_csv(
            OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)
        print(f"  +{len(rows)} rows (chunk {start // CHUNK + 1}, "
              f"{len(got) - len(rows)} null)")
        time.sleep(SLEEP)

    d = pd.read_csv(OUT_CSV)
    print(f"\n-> {OUT_CSV}: {len(d)} rows x {len(d.columns)} columns")
    return d


def demo(limit=None):
    d = fetch(limit)
    ids = wanted_ids(limit)

    assert d.s2_paper_id.is_unique, "duplicate s2_paper_id in the output"
    missing = set(ids) - set(d.s2_paper_id)
    assert not missing, f"{len(missing)} requested IDs never came back"

    empty = [c for c in d.columns if d[c].notna().sum() == 0]
    assert not empty, f"columns are 100% null: {empty}"

    fos = d.s2_primary_field.notna().mean()
    assert fos > 0.95, f"field-of-study coverage only {fos:.1%}"

    # The reason for fetching this at all: a field label whose coverage does not
    # depend on the decision. paper_fields.csv has a 6.0pp gap; this should not.
    m = pd.read_csv(MATCH_CSV)[["s2_paper_id", "accepted"]].dropna()
    j = d.merge(m.drop_duplicates("s2_paper_id"), on="s2_paper_id", validate="1:1")
    acc = j.loc[j.accepted == 1, "s2_primary_field"].notna().mean()
    rej = j.loc[j.accepted == 0, "s2_primary_field"].notna().mean()
    gap = 100 * (acc - rej)
    assert abs(gap) < 2.0, f"field coverage differential {gap:.1f}pp, expected under 2"

    redirects = int((d.s2_paper_id != d.s2_paper_id_returned).sum())
    assert redirects < 0.01 * len(d), f"{redirects} redirects, too many to shrug at"

    print(f"\nok — {len(d)} papers; field coverage {fos:.1%} "
          f"(accept {acc:.1%} vs reject {rej:.1%}, {gap:+.1f}pp); "
          f"affiliations {d.s2_affiliations.notna().mean():.1%}; "
          f"{redirects} S2 redirects to drop")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="fetch only the first N IDs (smoke test)")
    demo(p.parse_args().limit)
