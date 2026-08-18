# Codebase conventions

## Directory layout

```
src/fetch/      pulls from an external API or dump; incremental and resumable
src/build/      builds the analysis tables and frozen samples from fetched data
src/probes/     sends prompts to a model (leakage / recall / review probes)
src/analysis/   reads tables, computes results, writes reports and figures
src/figures/    the paper's figures and tables -> outputs/figures/; figstyle.py lives here
src/app/        Streamlit dashboard + pages/ (reads only, never writes results)
src/audit/      audits the repo itself: data quality, prompt export, MANIFEST
src/*.py        shared modules imported by the above: prompts, metrics, baselines
src/regimes/    one selection regime per file, all implementing select()
prompts/        every prompt as a .txt template — no prompt text inside a script
data/           read-only inputs (DB, CSVs, raw downloads) — never written by scripts
outputs/        all generated files: plots, processed CSVs, logs, traces
Archive/        historical code kept for reference; nothing here is imported or run
MANIFEST.md     generated: every script, and which script produced each output
```

A script goes in the directory matching what it does, not what it is about. New
group only when a file fits none of the seven.

Adding a group means registering it in `GROUPS` in `src/audit/build_manifest.py`.
Without that the whole directory is invisible to the manifest and everything it
writes is reported as an orphan — the script count going DOWN after adding scripts
is the symptom.

## MANIFEST.md must never be stale

`MANIFEST.md` is the answer to "where did this file come from": every output with its
producing script and its consumers, files written by more than one script, and the
files whose provenance is not reconstructable. It is generated — never hand-edit it.

**Regenerate it in the same turn as any of these, before reporting the work done:**

- adding, deleting, renaming, or moving a script under `src/`
- changing where a script reads from or writes to (a new `outputs/` or `data/` path,
  or a changed one)
- adding a prompt template under `prompts/`
- any change to the directory layout

```bash
python src/audit/build_manifest.py
```

A `Stop` hook in `.claude/settings.json` also runs this at the end of every turn, so
the file self-heals if it is forgotten. Do not treat the hook as the primary path:
it runs after the response is written, so its output is not seen or verified before
the work is reported. Run it yourself and read the selfcheck line — it reports how
many files are attributable and flags multi-writer files, which is information worth
acting on rather than skipping past.

If a new output does not show up attributed, the path was assembled inline instead of
being named in a module-level constant. Fix the script, not the manifest.

## Output discipline

Every script writes its outputs under `outputs/`. Create the directory on the fly:

```python
import os; os.makedirs("outputs", exist_ok=True)
```

Never write outputs to the repo root or `data/`.

## Script conventions

- Scripts are standalone (`python src/analysis/cite_hist.py`), no package install required.
- Accept `--db-path` / `--output-dir` flags when the path is likely to vary; hardcode sensible defaults otherwise.
- One script = one logical step. Name with a verb: `cite_hist.py`, `fetch_citations.py`.
- Use `data/` and `outputs/` paths relative to the repo root; always run scripts from the repo root.
- To import a shared module from a script one level down, bootstrap on `src/`:
  ```python
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from prompts import load
  ```
- Prefer a module-level constant for every output path (`OUT_CSV = "outputs/x.csv"`).
  `build_manifest.py` derives provenance from those literals, so a path assembled
  inline is invisible to it.

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

## Workflow

- Never commit to `main`. Branch per unit of work, PR, merge, delete the branch.
  Branches are short-lived — if one is more than ~10 commits ahead of `main`, it
  should have been a PR already.
- Anything spanning more than one session gets a GitHub issue first, written as the
  next concrete step rather than a theme ("add per-field recall@k to the dashboard",
  not "improve metrics"). Reference it in the PR body so merging closes it.
- Commit messages: `type: imperative summary` — `feat:`, `fix:`, `docs:`, `refactor:`,
  `perf:`, `data:`. One logical change per commit.
- README stays high-level: setup, data, schema, pointers. The script inventory lives
  in `MANIFEST.md` and is generated — do not duplicate it into the README.

## What not to add

- No PR templates, CODEOWNERS, issue labels, or branch protection — process for
  coordinating people, and this repo has one. Add when a second contributor arrives.
- No CI. The `Stop` hook already keeps `MANIFEST.md` honest locally; a server-side
  check earns its keep only once someone else can push.
- No intermediate abstraction layers unless two scripts share >10 lines of identical logic.
- No config files for values that don't change across runs.
- `Archive/` is append-only: move old scripts in, never import from it.

## Data files

- `data/gen_review.db` — OpenReview submissions + reviews (SQLite)
- `data/OpenAlex/` — citation data fetched from OpenAlex API
- `OpenAlex.txt` — email for OpenAlex polite pool (not a secret, but don't commit credentials)
