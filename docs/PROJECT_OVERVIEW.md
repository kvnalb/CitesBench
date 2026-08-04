# CitesBench — Project Overview

Compiled 2026-07-22 from the full git history (54 commits, 2026-01-30 to
2026-07-17) and the current codebase on branch `kunal`. This is a working
summary of what has been built and found, not a paper draft.

## 1. What the project is

CitesBench evaluates whether different ways of selecting which papers to
accept — human area chairs, human reviewer scores, an LLM committee, an LLM
decision head — pick papers that turn out to matter, using citation counts as
an outcome measure independent of the review process itself. The corpus is
every ICLR 2018–2020 submission with its OpenReview reviews and decision.

Every selection method implements one interface:

```
select(papers_df, n) -> List[paper_id]
```

`n` is pinned to the real historical accept count for that year, so every
regime is compared on equal footing. Regimes currently defined
(`src/regimes/`):

| Regime | Selection rule |
|---|---|
| `HumanActual` | The real AC accept/reject decisions |
| `HumanScore` | Top-N by mean human reviewer rating |
| `HumanDisagree` (+ `_reward`/`_penalize` variants) | Mean rating ± λ·rating_std |
| `LLMCommittee` | Top-N by a Gemma-4-31B multi-persona committee rating |
| `LLMDeepSeek` (Decision Head) | Top-N by DeepSeek-V3.1/GPT-oss-20b accept probability |
| `LLMNeutral` / `LLMEnsemble` / `LLMPositive` | Earlier single-persona LLM regimes (superseded, dead code as of `bb1f37f`) |

Metrics (`src/metrics.py`): median citations, mean log(1+citations), recall
at top 1/5/10%, each reported as lift over a 1000-run random baseline and
drawdown from an oracle "pick the top-N by citations" ceiling
(`src/baselines.py`).

## 2. Timeline, phase by phase

### Phase 0 — inherited scaffolding (Jan–Jun 2026, commits `9df896a`…`e7e0d4b`)
An earlier "LLM Paper Evaluator" / "paper_review and pairwise_review" codebase
was folded in, including an opt-in "editorial dedup" LLM pass with an A/B
harness and a safety check that refuses to run if the editorial judge and the
editor model are the same model (`01e8145`, `5aa3f52`, `90cf3da`). This
became the source of the coarse committee/decision-head pipeline referenced
throughout later work. Repo reorganized into `src/`/`outputs/`/`CLAUDE.md`
conventions at `4b19298`.

### Phase 1 — dashboard v1 and baseline metrics (`6df2831`…`905b2a5`, Jun 19–25)
Built the Streamlit dashboard, fixed baseline-aggregation bugs (pooling across
years, single-review std, top-k ties), added random/ideal baselines and
drawdown-from-ideal. Section 2 added: confusion matrices, citation-residual
scatter for flipped papers, missed-gems table, human-consensus-error table.
Three now-dead LLM persona regimes (neutral/ensemble/positive-advocate) were
added and later removed once the committee/decision-head regimes replaced
them (`bb1f37f`).

### Phase 2 — outlier & rejection-reason analysis (parallel track)
`outlier_analysis.py` identifies rejected papers that beat the 75th
percentile of same-year accepted-paper citations; `fetch_pc_decisions.py`
pulls the AC's decision note for each; `tag_rejection_reasons.py` uses Claude
to tag each with 1–3 reasons from a fixed taxonomy (novelty, empirics,
clarity, reproducibility, baselines, soundness, significance, framing,
related_work); `viz_outlier_scatter.py`/`viz_rejection_tags.py` visualize the
distribution. This produced `outputs/rejection_tags.png` and the "missed
gems" table surfaced in the dashboard.

### Phase 3 — committee/decision-head regimes + heterogeneity + RDD (`695889d`…`0d29df8`, Jul 2–10)
Replaced the dead persona regimes with `LLMCommittee` and `LLMDeepSeek`. Added
Section 3 (covariate heterogeneity: field coverage, subgroup recall@10%
advantage, field-controlled regression) and Section 4 (fuzzy regression
discontinuity around the acceptance score cutoff, `src/fuzzy_rdd.py`) to the
dashboard.

### Phase 4 — leakage/memorization test suite (`3b0b9db`, `31907cc`, Jul 10–11)
The central methodological worry: an LLM trained on internet text may already
know which ICLR papers became famous, making its "review" partly recall
rather than judgment. Built as five linked probes:

- **LAP** (Lookahead Propensity, `leakage_lap_v1.py`) — title-only query, "was
  this accepted?", answer read from logprobs. LAP = P(accept)+P(reject)
  (commitment), U-D = P(accept)−P(reject) (direction).
- **FAME** (`leakage_fame_v1.py`) — same mechanism, asking "is this widely
  cited?" instead of accept/reject — the outcome-side twin, since the actual
  dependent variable is citations, not decisions.
- **Placebo controls** (`leakage_controls.py`, sized by
  `leakage_power_analysis.py`) — fabricated-title false-positive rate (CI
  half-width design) and wrong-year equivalence test (TOST) to rule out the
  model just acquiescing to any confident-sounding prompt.
- **Masked re-review** (`leakage_masked_rereview.py`) — the identity-ablation
  mechanism test: score every paper twice, once with title+verbatim abstract,
  once with no title and a paraphrased abstract (proper nouns replaced by
  generic descriptors). If predictive power survives masking, it isn't
  memorization.
- **Exclusion eval + threshold sweep** (`leakage_exclusion_eval.py`,
  `leakage_threshold_sweep.py`) — re-run every regime on the pool with
  LAP/FAME ≥ 0.5 papers dropped, then sweep that 0.5 cutoff to check the
  result isn't an artifact of one arbitrary threshold.

### Phase 5 — statistical rigor pass (Jul 16, commits `5bea9f3`…`a600f23`)
Added field fixed effects to Sections 3c/4c, an econ-style regression table
with a coefficient plot, and a field×year cell-count matrix (flagging that
2020 is 83% unlabeled, so field-stratified results are effectively an
18–19 sample).

### Phase 6 — the citation ground-truth crisis and correction (Jul 16, `abe2580`, `d894210`, `14c58c9`, `d8e991b`, `5c7d7ac`)
A DDSP trace inspection surfaced a large OpenAlex/Semantic-Scholar citation
mismatch (78 vs. 485). Systematic audit
(`compare_citation_sources.py`) found OpenAlex indexes 98.6% of the corpus as
arXiv-preprint-only records, undercounting citations to the published venue
by a median 2.9×, differentially by decision (3.47× accepted vs. 2.00×
rejected — because accepted papers have a published version to lose
citations to). A full Semantic Scholar refetch
(`fetch_citations_s2_full.py`) raised coverage from 71.5% to 93.0%
(symmetric by decision) and became the corrected ground truth, wired into
the dashboard as a sidebar source toggle threading through every section
(`_et`, `src=` params on every cached loader). Paired bootstrap CIs
(B=2,000, `leakage_exclusion_bootstrap.py`) were added to the exclusion
headline, and a venue-premium add-back (`(1+c)·e^LATE − 1` applied to
rejected papers' citations, using the RDD-estimated causal effect of
acceptance) was implemented as a first-pass correction for the fact that
citation counts are partly caused by the decision being evaluated, not just
correlated with quality.

### Phase 7 — abstract-completion memorization probe (`abe2580`, Jul 16)
An independent, harder memorization test: give the model a title, year, and
first sentence of the true abstract, let it complete the rest, score the
completion's ROUGE-L against the true continuation vs. 5 same-field-year
decoys, and check for verbatim 8-gram overlap. N=297 (3 of 300 sampled papers
returned no scoreable output).

### Phase 8 — integrity audit (Jul 17, `d382751`)
Every numeric claim in that week's findings summary
(`docs/notes/07162026_findings.txt`) was independently recomputed from the
underlying CSVs and re-run from source scripts. Found and fixed one real
data-integrity bug (the committed S2 venue-premium bootstrap file held a
stale sensitivity run, not the headline B=2,000 run) and eight small
transcription errors in the prose (none changed a conclusion). Full
methodology audit in `outputs/findings_integrity_check.md`.

### Phase 9 — paper-writing prep (this week, Jul 17–22, uncommitted)
Target venue identified: NeurIPS 2026 workshop *AI-Native Academia*, Track 3
("AI-Assisted Peer Review and Reviewer Accountability"), 9-page long-paper
format, deadline 2026-08-29. `outputs/table1_summary_stats.tex` +
`src/table1_summary_stats.py` built as the paper's Table 1 (full corpus vs.
RDD-sample summary statistics, cross-checked against the raw DB and both
citation sources). A narrative skeleton (abstract → intro → data → methods →
results → discussion) was mapped out but not yet written to disk. The field
classifier (`tag_fields.py`) was found to be incomplete (2,726/4,567 papers
tagged, stalled on an uncaught crash rather than a deliberate stop) and
un-validated (no ground-truth accuracy check exists); an alternative
(OpenAlex Concepts/Topics) was live-tested and rejected as noisier and
taxonomically mismatched to the study's ML-subfield categories.

## 3. What the current headline result actually is

Not one number — three, depending on two methodological choices, and the
paper's honest framing needs to be built around that fact rather than around
picking the most flattering one:

| Ground truth | Venue-premium adjustment | Committee − Human AC lift (leakage-excluded pool) |
|---|---|---|
| OpenAlex | none | **+0.29** [+0.08, +0.56], p=.007 |
| Semantic Scholar | none | **+0.03** [−0.24, +0.31], p=.83 (not distinguishable from zero) |
| Semantic Scholar | RDD LATE = 1.285 | **+0.65** [+0.42, +0.89], p<.001 |

The middle row is the corrected-ground-truth answer as of today. The bottom
row depends on extrapolating a margin-identified causal estimate to every
rejected paper, which is flagged below as the single most important open
methods problem.

## 4. Empirical limitations

**Structural — not fixable with more compute in this project's timeframe:**
- Single venue (ICLR only) and a single 3-year window (2018–2020), both tied
  to the one scraped review database (`data/gen_review.db`) available.
  Generalizing to other venues means acquiring a new dataset, not running
  more analysis.
- The RDD sample, author-count covariates, and part of the citation pipeline
  all require an arXiv-matched paper. Papers never posted to arXiv are
  excluded, and that exclusion is unlikely to be random (e.g., authors who
  stop investing in a paper after rejection). This can be characterized
  (report arXiv-posting rate by decision) but not eliminated.
- Any citation count is a snapshot as of the fetch date; numbers will drift
  as citations accrue. Already handled by dating every fetch, not otherwise
  fixable.

**Real methods gaps, currently unresolved:**
- **Venue-premium extrapolation.** The reversal to +0.65 applies one
  RDD-estimated LATE uniformly to all 3,041 rejected papers, including clear
  rejects far from the margin. A stronger design already sits half-built:
  for the ~21% of rejected papers later published elsewhere (observable via
  the S2 venue match), use their *actual* subsequent citations instead of an
  extrapolated counterfactual. This is the highest-priority open item — it
  is the paper's current headline number, and it hasn't been checked against
  the more defensible design yet.
- **Field classifier is incomplete and unvalidated.** 1,841/4,567 papers
  (mostly 2020) have no field label because `tag_fields.py` stalled on an
  uncaught exception rather than finishing; and no accuracy number exists
  for the 2,726 papers it did tag, because no hand-labeled validation set
  has been built. The `theory_methods` bucket (64% of tagged papers) is
  defined as a catch-all, so field-stratified results currently contrast one
  large heterogeneous bucket against four much smaller, more specific ones.
- **Single model family.** The "AI" regimes are one Gemma committee and one
  DeepSeek/GPT-oss decision head. Nothing yet establishes whether the
  measured effect generalizes across model providers — the planned
  multi-model robustness appendix (4–5 models, single-LLM and committee
  configurations) has not been run.
- **No fixed citation window.** Papers from 2018 have had three more years to
  accumulate citations than 2020 papers; the analysis currently uses raw
  cumulative counts (year fixed effects partially absorb this, but a
  fixed-window comparison, e.g. citations in the first 5 years, would be
  cleaner and hasn't been built — it requires per-citation-event dates, not
  just totals, so it's the most API-expensive item on this list).
- **Leakage probe coverage is partial.** LAP/FAME probes and the
  abstract-completion probe each cover 300–948 of 4,567 papers; the
  exclusion-eval scripts print an explicit warning that results are
  directional below 90% probe coverage, and a `--full` run has not been
  executed.
- **Mechanism story is underdeveloped.** The "why does the committee
  outperform" question (inter-persona score correlation, marginal value of
  committee size) is analyzable right now from data already in
  `GENAI_REVIEW` (three persona-level scores per paper already logged) but
  has not yet been computed; the "what happens at N=10 committee members"
  extension would require new LLM calls.

## 5. Suggested next steps, roughly in priority order

1. **RDD-with-observed-outcomes fix for the venue premium.** No new API
   calls needed — the S2 venue data already identifies which rejected papers
   were later published elsewhere. Re-engineer
   `leakage_exclusion_eval.py`/`leakage_exclusion_bootstrap.py` to substitute
   observed post-rejection citations where available, falling back to the
   extrapolated premium only for papers with no later venue. Directly
   determines whether the +0.65 headline survives a more defensible design.
2. **Finish and validate field tagging.** Re-run `tag_fields.py` to
   completion (expect it to crash and need restarting; consider wrapping the
   retry loop in a try/except so one bad response doesn't kill the whole
   job), then hand-label a 30–50 paper stratified sample to get a real
   accuracy number, and cross-tabulate against `arxiv_categories` (already
   fetched, zero-cost) as a second, larger-N validation check.
3. **Mechanism analysis from existing data.** Compute inter-persona score
   correlation and a first variance decomposition directly from
   `GENAI_REVIEW`'s three logged persona scores — zero new API calls, and it
   is the paper's current biggest unwritten section (Table 3 in the
   narrative skeleton).
4. **Full-corpus leakage probe run.** Execute the `--full` mode already
   coded into `leakage_lap_v1.py`/`leakage_fame_v1.py` to move the
   memorization-robustness claim from "directional on a stratified sample"
   to corpus-wide.
5. **Multi-model robustness appendix.** Re-run the committee and
   decision-head pipelines with 4–5 different model families on a
   subsample, to address the single-model-family external-validity gap
   before submission.
6. **Fixed citation window,** only if time and rate-limit budget remain
   after the above — requires fetching per-citing-paper dates, the most
   expensive remaining item.

Track for submission: NeurIPS 2026 workshop *AI-Native Academia*, Track 3,
deadline **2026-08-29 AoE**, 9-page long paper (references excluded),
non-archival via OpenReview.
