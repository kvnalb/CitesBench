# Prompts

Every prompt sent to a model lives here as a `.txt` file. No prompt text in any
script under `src/` — scripts call `load("<path>", **placeholders)` from
[`src/prompts.py`](../src/prompts.py).

| File | Used by | What it asks |
|---|---|---|
| `recall/lap_oneword.txt` | `leakage_lap_v1.py`, `leakage_controls.py` | accepted / rejected / unknown, one word, no reasoning |
| `recall/fame_oneword.txt` | `leakage_fame_v1.py` | high / low / unknown citation recall, one word |
| `recall/lap_cot.txt` | `run_oos_probes.py` (lap, placebo, wrongyear) | two sentences of recall, then `ANSWER: accepted/rejected/unknown` |
| `recall/fame_cot.txt` | `run_oos_probes.py` (fame) | two sentences of recall, then `ANSWER: high/low/unknown` |
| `recall/abstract_completion.txt` | `leakage_abstract_completion_v1.py` | continue the abstract from its first sentence (verbatim-memorization probe) |
| `review/iclr_review_calibrated.txt` | `leakage_masked_rereview.py --rubric calibrated` (default) | 5 dimensions on a 0-5 float scale + rationale, with anti-bias warnings and 5 few-shot examples; takes `{year}` |
| `review/score_system.txt` | `leakage_masked_rereview.py --rubric simple` | single ICLR 1-10 quality score, JSON only |
| `review/citation_prediction_system.txt` | `leakage_eval_v1.py` | predicted 5-year citation count, JSON only |
| `review/paraphrase_abstract.txt` | `leakage_masked_rereview.py` | rewrite an abstract, strip identifying names |
| `review/title_abstract_body.txt` | `leakage_masked_rereview.py`, `leakage_eval_v1.py` | user message: title + abstract |
| `review/abstract_only_body.txt` | `leakage_masked_rereview.py` | user message: masked abstract only |

## Conventions

- Placeholders are `{name}`. Substitution is a plain string replace, not
  `str.format` — abstracts contain braces and `$…$` LaTeX that `.format` would
  choke on. A template may therefore contain literal JSON like `{"score": …}`.
- Exactly one trailing newline is stripped on load; every prompt we send ends
  mid-line. Do not reflow a paragraph onto multiple lines — a newline inside a
  paragraph changes the bytes sent, and `prompt_sha1` in
  `outputs/leakage_ledger.csv` will no longer match past runs.
- The Llama `<|start_header_id|>` wrapper stays in `run_oos_probes.py`. It is
  endpoint plumbing for the raw `/v1/completions` path, not a prompt.
- `Archive/` is untouched: it holds the 9-call committee pipeline's prompts
  (`Archive/Coarse/slim_coarse_pipeline.py`), which nothing imports or runs.

## Check

    python src/prompts.py                          # templates load, placeholders resolve
    python src/build_leakage_ledger.py --selfcheck  # prompt_sha1 still matches past runs
