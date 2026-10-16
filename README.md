# CitesBench

Do LLM reviewers pick better papers than human reviewers? Reviewer selection regimes
(human discretionary, human score-based, LLM variants) are evaluated against
citation-based ground truth on ICLR 2018–2020, with leakage probes to check how much
the LLM is recalling rather than judging.

## Quick start

```bash
pip install -r requirements.txt
```

Create a `.env` (gitignored) with your API keys — every script loads it via
`python-dotenv`.

Run everything from the repo root:

```bash
python src/<group>/<script>.py
streamlit run src/app/dashboard.py
```

Outputs land in `outputs/`.

## Where things are

`MANIFEST.md` is the authoritative index. It lists every script, what each one writes,
and which script produced each file under `outputs/` and `data/`. It is generated from
the code and is not in git, so build it locally:

```bash
python src/audit/build_manifest.py
```

This README stays high-level on purpose.

Directory conventions and the rules scripts follow live in [CLAUDE.md](CLAUDE.md).

## Data

`data/` and `outputs/` are gitignored. A fresh clone has only the code and a few
tracked outputs. The raw data comes from four public sources:

| source | what we use | where it lands |
|---|---|---|
| GenReview dataset | ICLR submissions and human reviews (SQLite) | `data/gen_review.db` |
| [ReviewArena](https://huggingface.co/datasets/anonymousNeurIPS2026submission4281/reviewarena-eval) | full paper text as markdown (ICLR 2025) | `data/ReviewArena/raw/data/*.parquet` |
| OpenReview API | author names, PDFs, PC decision notes | `src/fetch/fetch_openreview_*.py`, `fetch_pc_decisions.py` |
| Semantic Scholar API | citation counts, the outcome (`s2_citations`) | `src/fetch/fetch_citations_s2_v2.py` |

- `data/OpenAlex/`: OpenAlex citation data, kept only for the source comparison.
- `OpenAlex.txt`: email for the OpenAlex polite pool (higher rate limits).

Outputs of the LLM runs (committee reviews, probe traces) are not in git. They sit
on the project's internal Dropbox. To make them again, see *Reproducing the LLM runs*.

Everything in `data/` is read-only except to `src/fetch/` scripts.

## Core pipeline

These are the steps behind the dashboard's headline comparison. Run them in order,
from the repo root:

```bash
python src/fetch/resolve_arxiv_ids.py        # OpenReview submission -> arXiv id
python src/fetch/fetch_citations_s2_v2.py    # Semantic Scholar citations (resumable)
python src/build/build_citations.py          # outputs/citations.csv, the one citation table
python src/build/build_eval_table.py         # outputs/eval_table.csv, one row per submission
python src/analysis/run_eval.py              # every regime x year -> outputs/eval_results.csv
streamlit run src/app/dashboard.py           # dashboard + pages/
```

Paper exhibits: `python src/figures/check_inputs.py`, then `src/figures/fig*.py` and
`table*.py`. Every choice they depend on is in `src/figures/spec.py`.

The other scripts are side analyses (leakage probes, venue-premium RDD,
heterogeneity, era comparison). MANIFEST.md shows which output each one feeds.

## Reproducing the LLM runs

All model calls go to Together AI's OpenAI-compatible API. Put `TOGETHER_API_KEY`
in `.env`. Model IDs are in `MODELS` in `src/llm.py`.

- **Committee review (9 calls per paper):** `src/probes/slim_pipeline.py`, driven by
  `python src/probes/run_slim_pipeline.py --model gemma --n 5` (a smoke test; drop
  `--n` for the full run). It is a port of the pipeline in `Archive/CompletePipeline/`
  that produced the 2018-2020 committee outputs. `tests/test_slim_pipeline_matches_archive.py`
  checks that the port matches.
- **Single-call baseline:** `src/probes/run_single_call_baseline.py`. It needs the
  2018-2020 full text (`--text-dir`). The full command is in the script's docstring.
  Start with `--dry-run`, then `--n 2`.
- **Leakage probes:** `src/probes/leakage_*.py` and `run_oos_probes.py`.

Every run appends one row per call and skips IDs already done, so a crash costs
nothing on restart.

**Another provider:** any OpenAI-compatible endpoint works. Change the base URL and the
model IDs. The URL is hardcoded in each probe script. List them with:

```bash
grep -rn "api.together.xyz" src
```

### Schema

**SUBMISSION:** `id`, `title`, `abstract`, `when_submitted`, `primary_area`, `decision`, `source_id`, `pdf`

**REVIEW:** `paper_id`, `reviewer_id`, `rating`, `confidence`, `binocular_score`, `recommendation`, `review_text`

## Contributing

Branch → PR → merge; `main` is never committed to directly. Work spanning more than
one session gets a GitHub issue first. See [CLAUDE.md](CLAUDE.md#workflow).
