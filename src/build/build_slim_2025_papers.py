"""
Freeze the ICLR 2025 population the slim 9-call pipeline runs on.

ReviewArena is the only source here with full paper text — the OpenReview DB carries
a `pdf` URL and nothing else, so the 9-call pipeline (which reviews sections, not
abstracts) can only run on papers present in the parquet dump.

2025 is accepts-only in ReviewArena: 3,703 papers, all with text, median 79k chars.
The 5,019 rejected 2025 submissions in gen_review.db have no full text anywhere
locally, so any eval built on this list is accept-only and cannot speak to
accept-vs-reject separation. That caveat travels with the file.

`run_order` is a seeded shuffle so that "first 5 papers" is a reproducible smoke
test and not whatever the parquet happened to sort by (which is decision-correlated).

The markdown itself is NOT copied here — 3,703 x 79k chars is ~290MB. The runner
reads text from the parquet on demand, keyed by paper_id.

Outputs:
  outputs/samples/slim_2025_papers.csv   paper_id, title, decision, chars, run_order

Run: python src/build/build_slim_2025_papers.py
"""
import os
import glob

import pandas as pd

PARQUET_GLOB = "data/ReviewArena/raw/data/*.parquet"
OUT_CSV = "outputs/samples/slim_2025_papers.csv"
YEAR = 2025
SEED = 42
MIN_CHARS = 1000  # below this the "full text" is a stub, not a paper

os.makedirs("outputs/samples", exist_ok=True)


def load_year(year=YEAR):
    """All ReviewArena rows for a year. Kept separate so the runner can reuse it."""
    files = sorted(glob.glob(PARQUET_GLOB))
    if not files:
        raise SystemExit(f"ERROR: no parquet under {PARQUET_GLOB}")
    cols = ["forum_id", "year", "title", "decision", "markdown", "markdown_chars",
            "primary_area", "num_reviews"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return d[d.year == year]


def main():
    d = load_year()
    keep = d[d.decision.str.contains("Accept", na=False) & (d.markdown_chars >= MIN_CHARS)]
    keep = keep.rename(columns={"forum_id": "paper_id"})

    # seeded shuffle -> run_order; smoke tests take run_order < 5
    keep = keep.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    keep["run_order"] = range(len(keep))

    out = keep[["paper_id", "title", "decision", "markdown_chars", "primary_area",
                "num_reviews", "run_order"]]
    out.to_csv(OUT_CSV, index=False)

    print(f"wrote {OUT_CSV}: {len(out)} papers")
    print(out.decision.value_counts().to_string())
    print(f"chars: median {int(out.markdown_chars.median()):,} "
          f"p95 {int(out.markdown_chars.quantile(0.95)):,} "
          f"max {int(out.markdown_chars.max()):,}")
    dropped = len(d) - len(out)
    if dropped:
        print(f"dropped {dropped} rows (non-accept or under {MIN_CHARS} chars)")


if __name__ == "__main__":
    main()
