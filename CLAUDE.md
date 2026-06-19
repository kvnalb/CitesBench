# Codebase conventions

## Directory layout

```
src/         all runnable scripts; add subdirs (src/analysis/, src/fetch/) when a natural cluster forms
data/        read-only inputs (DB, CSVs, raw downloads) — never written by scripts
outputs/     all generated files: plots, processed CSVs, logs
Archive/     historical code kept for reference; nothing here is imported or run
```

## Output discipline

Every script writes its outputs under `outputs/`. Create the directory on the fly:

```python
import os; os.makedirs("outputs", exist_ok=True)
```

Never write outputs to the repo root or `data/`.

## Script conventions

- Scripts are standalone (`python src/cite_hist.py`), no package install required.
- Accept `--db-path` / `--output-dir` flags when the path is likely to vary; hardcode sensible defaults otherwise.
- One script = one logical step. Name with a verb: `cite_hist.py`, `fetch_citations.py`.
- Use `data/` and `outputs/` paths relative to the repo root; always run scripts from the repo root.

## Evaluation dashboard

Compares reviewer selection regimes (human discretionary, human score-based, LLM variants) against citation-based ground truth across ICLR 2018-2020.

- N per year = actual accept count for that year (pinned, same across all regimes)
- Every regime implements `select(papers_df, n) -> List[paper_id]` returning exactly n IDs
- Metrics: median citations, mean log(1+citations), count in true top 1/5/10%, recall@k
- Baselines: random (1000 runs averaged), ideal (top-N by citations)
- Reports lift over random and drawdown from ideal per metric per regime
- **Field normalization toggle**: dashboard supports both raw citations and field×year normalized citation percentile ranks as the ground truth signal. Fields: nlp, computer_vision, generative_models, reinforcement_learning, theory_methods.

## Secrets

All secrets live in `.env` (gitignored). Every script loads it at the top:

```python
from dotenv import load_dotenv
load_dotenv()
```

Never hardcode keys. Never read `os.environ["KEY"]` without `load_dotenv()` first.

## External API scripts

Any script that calls an external API in a loop must write results incrementally — one row per API call, appended immediately to the output CSV. Never accumulate in memory and flush at the end. This makes scripts resumable by default: on restart, read the output file, skip already-processed IDs, continue from where it left off.

## What not to add

- No intermediate abstraction layers unless two scripts share >10 lines of identical logic.
- No config files for values that don't change across runs.
- `Archive/` is append-only: move old scripts in, never import from it.

## Data files

- `data/gen_review.db` — OpenReview submissions + reviews (SQLite)
- `data/OpenAlex/` — citation data fetched from OpenAlex API
- `OpenAlex.txt` — email for OpenAlex polite pool (not a secret, but don't commit credentials)
