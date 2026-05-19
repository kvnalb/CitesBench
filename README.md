# LLMReviewer Code

This repo now contains three layers of code:

- The original lightweight PDF evaluator kept under [`Code/`](Code)
- The active research code at the repo root, including `paper_review/`, `pairwise_review/`, `Coarse/`, and the numbered compatibility launchers
- The consolidated end-to-end RDD pipeline in [`CompletePipeline/`](CompletePipeline)

## Main areas

- `CompletePipeline/`
  End-to-end pipeline for the RDD presentation workflow: OpenReview/arXiv/OpenAlex data assembly, slim LLM review generation, human-vs-LLM evaluation, report plots, and slide source.
- `paper_review/`
  Paper-level and abstract-level review generation and comparison code.
- `pairwise_review/`
  Pairwise ranking, Swiss/anchor scheduling, and model-sweep utilities.
- `Coarse/`
  Earlier coarse-review and committee/decision-head pipeline code.
- Top-level numbered scripts
  Compatibility launchers preserved so older commands still run.
- `Code/`
  Legacy minimal evaluator that discovers PDFs, builds a prompt, calls an LLM, and saves JSON outputs.

## Recommended entrypoint

For the consolidated workflow, start with [`CompletePipeline/README.md`](CompletePipeline/README.md).

That folder is organized into:

- `design/` for OpenReview/arXiv/OpenAlex ingestion and RDD sample construction
- `llm/` for slim committee generation, orchestration, and decision-head evaluation
- `analysis/` for citation and counterfactual analysis
- `report/` for the figures and tables used by the presentation
- `presentation/` for the Quarto slide deck

## Legacy minimal evaluator

The original GitHub repo contents remain available:

- [`Code/run.py`](Code/run.py)
- [`Code/evaluate_pdfs.py`](Code/evaluate_pdfs.py)
- [`Code/barebones.py`](Code/barebones.py)
- [`prompts/evaluation_prompt.txt`](prompts/evaluation_prompt.txt)

Those files support a small standalone batch evaluator using environment variables and `requirements.txt`.
