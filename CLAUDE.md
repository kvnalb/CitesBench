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

## What not to add

- No intermediate abstraction layers unless two scripts share >10 lines of identical logic.
- No config files for values that don't change across runs.
- `Archive/` is append-only: move old scripts in, never import from it.

## Data files

- `data/gen_review.db` — OpenReview submissions + reviews (SQLite)
- `data/OpenAlex/` — citation data fetched from OpenAlex API
- `OpenAlex.txt` — email for OpenAlex polite pool (not a secret, but don't commit credentials)
