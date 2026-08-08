# Can ReviewArena markdown be used for the 2025 run?

**Verdict: yes — conditional on heading normalization, and with one confound that this
audit could not close.**

Issue [#9](https://github.com/kvnalb/CitesBench/issues/9). Produced by
`python src/audit/audit_reviewarena_text.py` over all 5,916 ICLR papers ReviewArena has
for 2020 and 2025. No LLM calls; every measure is deterministic and reproducible.

## Why the question exists

The 2018–2020 reviews in `data/archive/all_paper_results.csv` were generated from text
produced by the archive's `extract_file` (Docling PDF extraction). A 2025 run would use
ReviewArena's `markdown` column, which is OCR'd PDF text of visibly different character.
Same pipeline and same model still means the model reads different text — and if the
text source alone moves the score, a 2025 number is not comparable to a 2018–2020 one.

## What could not be measured, and why

**The archive's extracted text is not in this repo.** `all_paper_results.csv` carries
`fulltext_path` / `pdf_path` pointing at an external share
(`OutputNew/LLMReviewPipeline_AllProcessed_Share/papers/…`); none of those files exist
locally, and the column is empty in the CSV that is here.

So the direct test — same paper, archive text vs ReviewArena text, measured side by
side — was not possible. Everything below is the strongest available substitute:
applying the archive's *own* acceptance criteria to ReviewArena text, and comparing
2020 (a year the archive also processed) against 2025.

## Results

| | 2020 | 2025 |
|---|---|---|
| papers | 2,213 | 3,703 |
| median chars | 49,225 | 78,674 |
| median garble ratio | 0.0 | 0.0 |
| % over archive's garble threshold (0.005) | **0.0** | **0.0** |
| median sections, raw text | **1** | **1** |
| median sections, after `normalize()` | **15** | **18** |
| % passing `_check_extraction_quality` | **100.0** | **100.0** |
| % with a typed INTRODUCTION | 99.0 | 99.6 |
| % with a typed METHODOLOGY | 53.3 | 64.8 |
| % with a typed CONCLUSION | 80.1 | 82.8 |
| % with a REFERENCES section | 99.7 | 99.6 |
| **% where the method call fires** | **99.9** | **100.0** |
| **% where the intro call fires** | **100.0** | **100.0** |

## Reading the results

**1. Heading normalization is mandatory, not an optimization.** The median paper has
**one** section before normalization — ReviewArena's "markdown" has no `#` headings, so
`_parse_sections_from_markdown` returns a single untyped blob. Fed raw, the pipeline
would review an abstract and an undifferentiated mass of text for every paper, and
would report nothing unusual while doing it. After `src/build/normalize_paper_markdown.py`
promotes numbered (`4 EXPERIMENTS`) and keyword (`INTRODUCTION`) lines to `## `, the
median rises to 15–18 sections and no paper is left sectionless.

**2. The text passes the archive's own quality gate outright.** Zero papers exceed the
garble threshold, and 100% satisfy `_check_extraction_quality` (sections exist, ≥500
chars of section text). By the standard the archive applied to its own inputs, this
text is acceptable.

**3. Call counts are stable, which is what comparability requires.** The typed-METHODOLOGY
rate looks alarming (53% vs 65%, an 11-point year gap) but is the wrong metric.
`_method_sections` falls back from typed METHODOLOGY → title-keyword match
(`method|approach|model|architecture|training|setup|experiment|evaluation|result|analysis`)
→ any non-boilerplate section. The method call therefore fires for **99.9–100%** of
papers in both years. This is corroborated externally: `committee_llm_calls` is exactly
**8 for all 4,497 archived papers**, i.e. the archive never skipped this stage either.
Every 2025 paper will receive the same call sequence every 2020 paper did.

## The confound that remains open

Two limitations, stated plainly:

**The garble metric does not measure the errors this text actually has.** `garble_ratio`
detects a specific character class — `®`, `õ`, `/C12`, `glyph[...]` — the signature of
pre-2005 PDF encodings. It does not detect word-level OCR corruption, which is what
ReviewArena text visibly contains: `DIMENSION-SPECIFE STUCHASTIC SUB-GRAPHS`, `CIFARIO`
for CIFAR-10, `GCNI` for GCNII. A score of 0.0 means "free of the artifacts the archive
knew to look for", **not** "clean". This audit therefore establishes that the text is no
worse than the archive's stated bar, not that it is good.

**Section-typing quality differs by year even though call counts do not.** For 46% of
2020 papers and 35% of 2025 papers, the method stage receives a fallback selection
rather than a genuine methodology section. The number of calls is identical; the text
inside one of them is chosen by a different rule, at a rate that differs by 11 points
between the years being compared.

## Recommendation

Use ReviewArena markdown for the 2025 run, with `normalize()` applied. It clears the
archive's bar on every criterion the archive defined.

Treat 2025-vs-2018–2020 comparisons as **provisional** until one of these closes the gap:

1. **Retrieve the archive share** (`OutputNew/LLMReviewPipeline_AllProcessed_Share/`)
   and run the true test — same 2020 papers, archive text vs ReviewArena text, through
   the identical pipeline, comparing output scores rather than input statistics. This is
   the only measurement that answers the original question, and it is cheap once the
   files exist.
2. **Run the 2020 overlap as a control** — score ReviewArena's 2020 papers with the
   restored pipeline and compare against the archived 2020 scores for those same
   papers. Any systematic shift is the text-source effect, measured directly.

Option 2 is available now and needs no external data. It should run before, or
alongside, the 2025 run — not after it.
