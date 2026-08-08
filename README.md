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

See [MANIFEST.md](MANIFEST.md) — generated from the code, never hand-edited. It lists
every script, what each one writes, and which script produced each file under
`outputs/` and `data/`. That is the authoritative index; this README stays high-level
on purpose.

Directory conventions and the rules scripts follow live in [CLAUDE.md](CLAUDE.md).

## Data

- `data/gen_review.db` — OpenReview submissions + reviews (SQLite)
- `data/OpenAlex/` — citation data from OpenAlex
- `OpenAlex.txt` — email for the OpenAlex polite pool (higher rate limits)

Everything in `data/` is read-only except to `src/fetch/` scripts.

### Schema

**SUBMISSION:** `id`, `title`, `abstract`, `when_submitted`, `primary_area`, `decision`, `source_id`, `pdf`

**REVIEW:** `paper_id`, `reviewer_id`, `rating`, `confidence`, `binocular_score`, `recommendation`, `review_text`

## Contributing

Branch → PR → merge; `main` is never committed to directly. Work spanning more than
one session gets a GitHub issue first. See [CLAUDE.md](CLAUDE.md#workflow).
