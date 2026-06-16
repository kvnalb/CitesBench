# LLM Reviewer

Analysis of LLM-generated paper reviews against human reviewer data from ICLR 2018–2020.

## Quick Start

```bash
pip install -r requirements.txt
```

Run all scripts from the repo root. Outputs land in `outputs/`.

## Scripts

### `src/cite_hist.py`

Plots citation count distributions for accepted vs. rejected papers (2018–2020).

```bash
python src/cite_hist.py
```

**Inputs:**
- `data/OpenAlex/openalex_rdd_arxiv_paper_level.csv`
- `data/gen_review.db`

**Output:** `outputs/cite_hist.png`

## File Structure

```
src/         runnable scripts
data/        read-only inputs (gen_review.db, OpenAlex CSVs)
outputs/     generated plots and CSVs (git-ignored)
Archive/     old pipeline code kept for reference
```

## Data

- `data/gen_review.db` — OpenReview submissions + reviews (SQLite)
- `data/OpenAlex/` — citation data from OpenAlex API
- `OpenAlex.txt` — email for OpenAlex polite pool (higher rate limits)

## Database Schema

**SUBMISSION:** `id`, `title`, `abstract`, `when_submitted`, `primary_area`, `decision`, `source_id`, `pdf`

**REVIEW:** `paper_id`, `reviewer_id`, `rating`, `confidence`, `binocular_score`, `recommendation`, `review_text`
