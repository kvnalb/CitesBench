# ICLR Local RD Exploration

This folder contains an exploratory local regression-discontinuity-style workflow
for ICLR review data across all years available in the local SQLite database.

The first-pass goal is descriptive and diagnostic:

- aggregate paper-level reviewer scores across all available ICLR years
- plot review scores and acceptance rates by year
- scan for year-specific acceptance jumps in paper-level mean reviewer rating
- pool the year-specific centered scores and fit local linear models with
  fixed effects

This is not a clean causal RD setup. It is an exploratory local-jump analysis.

## Important Constraints

- The local SQLite database contains ICLR submissions from 2018 through 2025.
- There are no Area Chair or Senior Area Chair identifiers in the local DB, so
  chair fixed effects are not available.
- `primary_area` is blank in 2018-2023 and populated in 2024-2025 only.
- Review-form score dimensions change over time, but `rating` is present in all
  years and is the common running variable used here.
- Decision labels also change over time, so the script normalizes them into a
  binary accept/reject flag and excludes withdrawn/workshop outcomes from the
  main analysis.

## Workflow

The main script is:

- [run_iclr_local_rdd.py](/Users/schoudhary42/Library/CloudStorage/Dropbox-Personal/Projects/LLMReview/Code/Desing/iclr_local_rdd/run_iclr_local_rdd.py)
- [fetch_arxiv_metadata.py](/Users/schoudhary42/Library/CloudStorage/Dropbox-Personal/Projects/LLMReview/Code/Desing/iclr_local_rdd/fetch_arxiv_metadata.py)
- [download_arxiv_metadata_snapshot.py](/Users/schoudhary42/Library/CloudStorage/Dropbox-Personal/Projects/LLMReview/Code/Desing/iclr_local_rdd/download_arxiv_metadata_snapshot.py)
- [match_arxiv_metadata_dump.py](/Users/schoudhary42/Library/CloudStorage/Dropbox-Personal/Projects/LLMReview/Code/Desing/iclr_local_rdd/match_arxiv_metadata_dump.py)
- [fuzzy_match_arxiv_metadata_dump.py](/Users/schoudhary42/Library/CloudStorage/Dropbox-Personal/Projects/LLMReview/Code/Desing/iclr_local_rdd/fuzzy_match_arxiv_metadata_dump.py)
- [fetch_openreview_metadata.py](/Users/schoudhary42/Library/CloudStorage/Dropbox-Personal/Projects/LLMReview/Code/Desing/iclr_local_rdd/fetch_openreview_metadata.py)
- [fetch_openreview_yearly_submissions.py](/Users/schoudhary42/Library/CloudStorage/Dropbox-Personal/Projects/LLMReview/Code/Desing/iclr_local_rdd/fetch_openreview_yearly_submissions.py)
- [fetch_crossref_metadata.py](/Users/schoudhary42/Library/CloudStorage/Dropbox-Personal/Projects/LLMReview/Code/Desing/iclr_local_rdd/fetch_crossref_metadata.py)

It performs the following steps:

1. Load all review rows from `data/LLM-Reviewer-03042026/data/gen_review.db`.
2. Parse reviewer `rating` values and aggregate them to the paper level.
3. Normalize decision labels into `accept`, `reject`, `withdrawn`, `workshop`,
   or `other`.
4. Plot score distributions and acceptance-vs-score curves by year.
5. Scan candidate cutoffs within each year using a local linear weighted model.
6. Choose an exploratory year-specific cutoff based on the largest absolute jump.
7. Select a year-specific bandwidth around each year-specific cutoff using a
   local-fit cross-validation criterion with an explicit locality preference.
8. Export the resulting year-specific RDD sample.
9. Center scores by the year-specific cutoff and fit pooled local-linear models:
   - no fixed effects
   - year fixed effects
   - year-by-area fixed effects when available, falling back to year only

## Outputs

By default, the script writes to:

- `Output/Design/iclr_local_rdd/`

Key outputs include:

- `paper_level_all_years.csv`
- `yearly_summary.csv`
- `yearly_cutoff_scan.csv`
- `yearly_best_cutoffs.csv`
- `yearly_bandwidth_scan.csv`
- `yearly_selected_bandwidths.csv`
- `paper_level_with_year_specific_windows.csv`
- `rdd_sample_year_specific_bandwidth.csv`
- `rdd_sample_summary_by_year.csv`
- `pooled_local_models.csv`
- `fig_score_distribution_by_year.png`
- `fig_acceptance_vs_score_by_year.png`
- `fig_centered_acceptance_pooled.png`

The arXiv metadata pipeline writes to:

- `rawdata/Design/arXiv/`

Key arXiv outputs include:

- `query_manifest.csv`
- `arxiv_best_matches.csv`
- `arxiv_candidate_matches.csv`
- `<input_stem>_with_arxiv_best_match.csv`
- `arxiv_query_summary.json`
- `query_cache/*.xml`
- `query_cache/*.json`

The local dump workflow additionally uses:

- `rawdata/Design/arXiv/dump/`
- `arxiv_dump_best_matches.csv`
- `arxiv_dump_candidate_matches.csv`
- `<input_stem>_with_arxiv_dump_match.csv`
- `arxiv_dump_match_summary.json`
- `arxiv_dump_fuzzy_candidate_matches.csv`
- `arxiv_dump_fuzzy_best_matches.csv`
- `arxiv_dump_combined_best_matches.csv`
- `<input_stem>_with_arxiv_dump_combined_match.csv`
- `arxiv_dump_fuzzy_match_summary.json`

The OpenReview metadata pipeline writes to:

- `rawdata/Design/OpenReview/`

Key OpenReview outputs include:

- `query_manifest.csv`
- `openreview_note_metadata.csv`
- `<input_stem>_with_openreview_metadata.csv`
- `openreview_query_summary.json`
- `notes/*.json`
- optionally `forum_notes/*.json`
- optionally `invitations/*.json`
- `openreview_yearly_submissions.csv`
- `year_page_manifest.csv`
- `<input_stem>_with_openreview_yearly_submissions.csv`
- `openreview_yearly_submissions_summary.json`
- `year_pages/<year>/page_*.json`

The Crossref DOI-recovery pipeline writes to:

- `rawdata/Design/Crossref/`

Key Crossref outputs include:

- `query_manifest.csv`
- `crossref_candidate_matches.csv`
- `crossref_best_matches.csv`
- `<input_stem>_with_crossref_best_match.csv`
- `crossref_query_summary.json`
- `query_cache/*.json`

## Usage

Run from the repo root:

```bash
python3 Code/Desing/iclr_local_rdd/run_iclr_local_rdd.py
```

Optional arguments:

```bash
python3 Code/Desing/iclr_local_rdd/run_iclr_local_rdd.py \
  --bandwidth 0.5 \
  --bandwidth-min 0.35 \
  --bandwidth-max 1.5 \
  --bandwidth-cv-tolerance-frac 0.10 \
  --candidate-min 4.5 \
  --candidate-max 6.5
```

To query arXiv metadata for the current RDD sample and cache raw responses:

```bash
python3 Code/Desing/iclr_local_rdd/fetch_arxiv_metadata.py
```

Because the local ICLR files do not include author names, the arXiv matcher is
conservative by default:

- automatic matches require an exact normalized-title match
- near-exact title matches are retained in `arxiv_candidate_matches.csv` for review
- raw Atom responses are cached in `rawdata/Design/arXiv/query_cache/`
- the request loop stays within arXiv's legacy API rate guidance and backs off on
  transient denials before retrying

The default input is `rdd_sample_year_specific_bandwidth.csv`. To run on a
different paper file, pass `--input-csv`.

To prefer a local metadata dump over live API calls:

```bash
python3 Code/Desing/iclr_local_rdd/download_arxiv_metadata_snapshot.py
python3 Code/Desing/iclr_local_rdd/match_arxiv_metadata_dump.py
python3 Code/Desing/iclr_local_rdd/fuzzy_match_arxiv_metadata_dump.py
```

The downloader currently targets the Hugging Face mirror of the arXiv metadata
snapshot because it is directly downloadable in this environment, but the local
matcher also accepts Kaggle-style JSONL dump files if you provide `--dump-path`.

The fuzzy pass only scans papers that were still unmatched after exact-title
matching. It uses conservative blocking keys, then separates:

- `fuzzy_high_confidence`: safe to auto-promote into the combined match file
- `fuzzy_review_candidate`: plausible suggestions retained for manual review

To fetch public OpenReview metadata for the current arXiv-unmatched set:

```bash
python3 Code/Desing/iclr_local_rdd/fetch_openreview_metadata.py
```

This OpenReview script defaults to:

- `rawdata/Design/arXiv/arxiv_dump_combined_best_matches.csv` as input
- `rawdata/Design/OpenReview/` as the raw output directory
- querying only rows still unmatched after the arXiv pass
- caching one raw `notes?id=<paper_id>` API response per paper under `notes/`

Optional flags:

```bash
python3 Code/Desing/iclr_local_rdd/fetch_openreview_metadata.py \
  --query-mode all \
  --fetch-thread \
  --fetch-invitations
```

The script uses Playwright because direct unauthenticated HTTP calls to
OpenReview can be denied in this environment even when the public forum pages
and browser-context API calls are accessible.

To fetch the full public ICLR submission metadata year by year, then merge it
back to the current analytic file:

```bash
python3 Code/Desing/iclr_local_rdd/fetch_openreview_yearly_submissions.py
```

This bulk workflow uses the year-level submission invitations we verified from
OpenReview:

- ICLR 2018-2023: v1 `Blind_Submission`
- ICLR 2024-2025: v2 domain-scoped `Submission`

It is much more efficient than querying `forum?id=<paper_id>` one paper at a
time, and it also preserves raw paginated API responses under
`rawdata/Design/OpenReview/year_pages/`.

To query Crossref for DOI candidates using the OpenReview-enriched metadata:

```bash
python3 Code/Desing/iclr_local_rdd/fetch_crossref_metadata.py
```

This defaults to the OpenReview-enriched arXiv working file and only queries
rows that:

- were still unmatched in the arXiv pass
- now have public OpenReview metadata available

The Crossref matcher is conservative by default:

- automatic matches require an exact normalized-title match
- near-exact matches are retained for review
- raw JSON responses are cached under `rawdata/Design/Crossref/query_cache/`

If you have an email address you want to identify to Crossref, pass:

```bash
python3 Code/Desing/iclr_local_rdd/fetch_crossref_metadata.py --mailto you@example.com
```

## Interpretation

Treat the outputs here as an organized first look at whether local acceptance
jumps exist in the paper-level mean review score distribution, and how stable
those jumps are across years. Because we do not observe chair assignments or a
known deterministic acceptance threshold, the cutoff scan is exploratory rather
than identified.
