# Complete Pipeline

This folder is a consolidated copy of the code used to build the RDD review-analysis presentation.

Defaults in the copied scripts now point to `OutputNew/` instead of the original scattered output locations.

## Layout

- `design/`
  - RDD sample construction
  - arXiv matching
  - OpenReview metadata ingestion
  - OpenAlex citation enrichment
  - timing / missingness diagnostics
- `llm/`
  - slim committee pipeline
  - decision-head evaluation
  - Gemma shard orchestration
  - human-vs-LLM evaluation scripts
  - vendored `coarse/` package used by the slim pipeline
- `analysis/`
  - citation RDD
  - embeddings
  - citation prediction
  - tiebreaker counterfactual
- `report/`
  - figure/table builders for the RDD coarse presentation
- `presentation/`
  - copied slide source and build assets
- `citation_imputation/`
  - copied imputation note and supporting outputs
- `prompts/`
  - persona prompt files used by the slim review pipeline

## Main Phases

1. Build the year-specific RDD sample:
   - `python3 Code/CompletePipeline/design/run_iclr_local_rdd.py`
2. Enrich with arXiv / OpenReview / OpenAlex:
   - `python3 Code/CompletePipeline/design/download_arxiv_metadata_snapshot.py`
   - `python3 Code/CompletePipeline/design/match_arxiv_metadata_dump.py`
   - `python3 Code/CompletePipeline/design/fuzzy_match_arxiv_metadata_dump.py`
   - `python3 Code/CompletePipeline/design/fetch_openreview_yearly_submissions.py`
   - `python3 Code/CompletePipeline/design/fetch_openalex_citations_from_arxiv_matches.py`
3. Run the slim LLM review pipeline:
   - `.venv-coarse/bin/python Code/CompletePipeline/llm/07_run_rdd_bandwidth_coarse_reviews.py`
   - `.venv-coarse/bin/python Code/CompletePipeline/llm/08_orchestrate_gemma_shards.py`
   - `.venv-coarse/bin/python Code/CompletePipeline/llm/09_queue_follow_on_wave.py`
4. Run evaluation / derived empirics:
   - `python3 Code/CompletePipeline/analysis/03a_embed_abstracts.py`
   - `Rscript Code/CompletePipeline/analysis/01_citation_rdd.R`
   - `Rscript Code/CompletePipeline/analysis/03b_predict_citations.R`
   - `Rscript Code/CompletePipeline/analysis/run_policy_counterfactual.R`
   - `.venv-coarse/bin/python Code/CompletePipeline/llm/12_eval_cached_decision_heads.py`
   - `.venv-coarse/bin/python Code/CompletePipeline/llm/13_eval_review_topic_overlap.py`
   - `.venv-coarse/bin/python Code/CompletePipeline/llm/15_eval_human_persona_embedding_overlap.py`
   - `.venv-coarse/bin/python Code/CompletePipeline/llm/16_eval_human_bucket_persona_overlap.py`
5. Rebuild presentation figures:
   - `python3 Code/CompletePipeline/report/compute_pairwise_review_similarity.py`
   - `Rscript Code/CompletePipeline/report/plot_rdd_accept_and_cites.R`
   - `Rscript Code/CompletePipeline/report/plot_rdd_citations_jump.R`
   - `python3 Code/CompletePipeline/report/plot_confusion_matrices.py`
   - `python3 Code/CompletePipeline/report/plot_committee_vs_human_scores.py`
   - `python3 Code/CompletePipeline/report/plot_human_vs_llm_scatter_density.py`

## Output Layout

The copied scripts now target:

- `OutputNew/Design/`
- `OutputNew/rawdata/Design/`
- `OutputNew/Empirics/`
- `OutputNew/Coarse/`
- `OutputNew/LLMOutput/`
- `OutputNew/Report/RDD_Coarse/plots/`
- `OutputNew/Playground/fuzzy_rdd_llm_tiebreaker/`

## Notes

- Human review inputs still come from the existing `processed/` tree.
- API-key paths still point to the existing repo-root key files such as `key.txt` and `OpenAlex.txt`.
- The copied presentation source in `presentation/slides.qmd` now points at `OutputNew/`.
- Verified in this workspace:
  - Python syntax for the copied `.py` files via `py_compile`
  - `--help` runs for `design/fetch_openalex_citations_from_arxiv_matches.py`
  - `--help` runs for `llm/07_run_rdd_bandwidth_coarse_reviews.py`
