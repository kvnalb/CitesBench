# CitesBench — First-Principles Methodology Review

What the dashboard claims to answer: *would an alternative selection regime (score-based,
disagreement-adjusted, LLM committee, DeepSeek) have chosen papers that turned out more
impactful than the ones human Area Chairs actually accepted at ICLR 2018–2020?*

Ground truth = top-N papers by citations ("the ideal"). Each regime picks exactly N
(= actual accept count). Metrics = overlap with ideal, median/mean citations, recall@k.

Below: every structural problem with that design, ranked by severity, with the best fix and
a cheap good-enough fix for each.

---

## TIER 1 — Existential. Fix before any PI takes the ranking seriously.

### P1. The ground truth is caused by the treatment → the design penalizes deviation from AC

This is the deepest problem and it is fatal if unaddressed.

Acceptance *causes* citations (visibility, proceedings, credentialing, author promotion — the
"venue premium," documented for journal "twins"). Two consequences compound:

1. **The citation-ideal is biased toward accepted papers.** Top-N-by-citations over-counts
   papers AC accepted, because acceptance inflated their citations. So the Human-AC regime
   overlaps the ideal partly *by construction*, not by merit.
2. **Every regime's unique picks are rejected papers with suppressed citations.** When a regime
   deviates from AC, its unique selections are drawn from the rejected pool — and those papers'
   observed citations are artificially low *because they were rejected*. We never observe their
   counterfactual (accepted-world) citation count.

Net: **the evaluation is rigged toward the status quo.** Any regime that would have made
genuinely different (and possibly better) decisions is mechanically punished, because deviation
means picking papers whose citations were suppressed by the very rejection the regime is trying
to overturn. This guts the entire premise — the point of the exercise is to evaluate regimes
that decide *differently* than AC.

**Best solution — restrict the credible comparison to the RDD margin.** Around the acceptance
threshold, accept/reject is quasi-random (fuzzy RDD; running variable = mean reviewer rating,
cutoff = the bar). Within ±ε of the bar, rejected and accepted papers are exchangeable, so their
citations are comparable and deviation is no longer differentially penalized. **You may already
have this**: `all_paper_results.csv` has a `share_source_group = "old_rdd"` (2,361 papers,
ratings restricted to 4.3–7.25 — exactly the margin). Investigate that artifact first.

**Alternative best solution — impute the counterfactual.** Estimate the venue premium γ (via the
RDD jump, or DML controlling for reviewer scores + abstract embeddings + field + year) and add γ
back to rejected papers' citations before computing the ideal and all metrics. Identified at the
margin; an assumption-laden extrapolation for clear rejects.

**Cheap good-enough.** Run the headline comparison only on the borderline band (e.g. papers
within one rating point of the bar), and label everything on the full set "exploratory,
status-quo-biased." Honest, no new modeling.

---

### P2. LLM-regime hindsight leakage — TESTED (2026-07-10), quantified, not fatal

The LLM committee / DeepSeek scores were generated in 2024–2026 by models whose training data
*includes* which ICLR 2018–2020 papers became famous (Soft Actor-Critic, etc.). An LLM "review"
of a 2018 paper is not a blind review — it can reflect memorized hindsight about impact. If so,
"LLM beats human" could be pure leakage: the model already knows the answer.

**Update — a five-layer leakage test suite was built and run on the full corpus.** Scripts:
`src/leakage_lap_v1.py`, `src/leakage_fame_v1.py`, `src/leakage_controls.py`,
`src/leakage_masked_rereview.py`, `src/leakage_exclusion_eval.py`. Reports:
`outputs/leakage_lap_report.md`, `outputs/leakage_fame_report.md`,
`outputs/leakage_masked_report.md`, `outputs/leakage_exclusion_eval.csv`.

1. **Decision recall (LAP, adapted from Gao/Jiang/Yan 2026).** Title+year-only query to Gemma-4-31B
   (the same model used for the committee): "was this paper accepted or rejected?" Read soft
   probability of commitment from logprobs. N=4,189 (full corpus). 21.5% of papers get a confident
   answer. Detection regression `log(1+cites) ~ committee_rating + LAP + LAP×committee_rating`:
   β₃ = 0.46 (p < 0.001) — the committee rating is more accurate specifically on memorized papers.
   Decomposition (residualize committee_rating on human mean_rating): foresight on non-memorized
   (LAP=0) papers is still significant (β=0.87, p≈0) and larger than the contaminated share
   (β=0.67, p<0.001) — genuine signal survives, but contamination is real.

2. **Probe validity (placebo controls) — scaled to power-justified N (2026-07-11).** The original
   pilot (30 fabricated + 30 wrong-year) was a convenience sample with no power analysis; see
   `src/leakage_power_analysis.py` / `outputs/leakage_power_analysis.md` for the sizing. Fabricated
   titles scaled to N=150 (sized so the 95% CI on the false-positive rate clears the real commit
   rate with >2x margin); wrong-year scaled to N≈300/offset across two offsets (+1, −1), sized via
   TOST equivalence testing at a ±0.05 margin (a non-significant pilot result at N=30 is not itself
   evidence of "no difference" — it needed an equivalence test, not just a null result).
   **Result:** 150 fabricated titles → confident answer only 0.7% of the time (95% CI [0.1%, 3.7%]),
   vs. 18.9% (95% CI [17.8%, 20.1%]) on real papers — non-overlapping CIs. Wrong-year: mean diff
   (correct − wrong-year LAP) = −0.003 (N=328, 95% CI [−0.040, +0.034]) at +1yr and +0.020 (N=300,
   95% CI [−0.016, +0.056]) at −1yr — both comfortably inside the ±0.05 equivalence band. The probe
   measures real memory, not acquiescence, and doesn't reduce to sensitivity to our framing.

3. **Fame recall — the sharper channel.** Decision-direction accuracy is near chance (52–58%), but
   a parallel probe ("is this paper widely cited — top 10%?") is **85% accurate** when the model
   commits (36% of papers, N=4,193 full corpus). What's memorized is prominence, not the accept/
   reject decision — the more direct threat to a citation-based ground truth.

4. **What predicts recall — citations specifically, not general quality.** Regressing LAP/FAME on
   `log(1+citations)` and human `mean_rating` jointly: citations remain significant (p≈0) while
   `mean_rating` drops out (p=0.13 for LAP, p=0.42 for FAME) — recall tracks citation-linked fame,
   not "the model likes what reviewers liked." Top-citation-decile papers are recalled at 3–4×
   the rate of bottom-decile papers (LAP: 51.5% vs 15.5%; FAME: 64.9% vs 26.1%).

5. **Masked re-review (causal test).** Same committee rubric scored twice per paper: once normal
   (title + abstract), once with title removed and abstract paraphrased by a different model family
   with proper nouns genericized (identity ablated, content preserved). N=119. Memorized (high-LAP)
   papers lose significantly more score under masking than non-memorized papers (Δ = +1.31 vs
   +0.78, one-sided p = 0.027) — part of the inflated score rides on recognizing the paper, not
   just judging its content.

6. **Leakage-excluded headline re-run.** All regimes re-evaluated excluding every paper flagged
   memorized on either channel (LAP or FAME ≥ 0.5; 1,564/4,567 papers excluded, 98.5% probe
   coverage). Lift over random, full vs. excluded:

   | Regime | Full | Excluded | Δ |
   |---|---|---|---|
   | LLM Committee (Gemma) | 1.40 | 1.18 | **−0.22** |
   | LLM Decision Head | 1.26 | 1.07 | −0.18 |
   | Human AC | 0.95 | 0.89 | −0.06 |
   | Human Score-based | 0.85 | 0.86 | +0.02 |

**Verdict.** Contamination is real and now quantified: roughly a quarter of the LLM committee's
apparent lift over random (0.22 of 1.40) is attributable to memorized papers. The thesis survives
its removal — the LLM committee still beats every human regime on papers it cannot be shown to
recall (1.18 vs. 0.86–0.89) — but the full-sample number (1.40) should never be reported without
the excluded number alongside it. **Any headline claim should cite lift = 1.18 (leakage-excluded),
not 1.40, with the 0.22 gap reported as the measured leakage tax.**

**Residual caveat.** LAP/FAME ≥ 0.5 is a threshold, not a hard boundary — a paper just below it
isn't guaranteed clean, and the exclusion test can only remove memorization the probes actually
elicit. Not fatal, but keep this caveat in any writeup.

**Threshold-sensitivity check (2026-07-11, `src/leakage_threshold_sweep.py`, no new API calls).**
Re-ran the exclusion eval at every cutoff from 0.1 to 0.9. Excluded-pool size barely moves
(1,563–1,568 papers) because LAP/FAME scores cluster near 0 or 1, and per-regime Δ (leakage-excluded
minus full lift) is flat across the whole range — e.g. LLM Committee stays at roughly −0.21 to −0.22
throughout. 0.5 isn't cherry-picked; the same conclusion holds at any threshold in this band.

---

### P3. No uncertainty quantification

Every regime is a point estimate (overlap 0.62, etc.). With ~500 papers/year and heavy-tailed
citations, the gap between 0.62 and 0.60 may be noise. The random baseline is averaged over 1000
runs (good), but the regimes themselves have no CIs, and "All years" averaging hides between-year
variance. A ranking without error bars invites over-reading.

**Best solution.** Paired bootstrap over papers: resample the pool (same resample across all
regimes), recompute every metric, repeat 1000×. Report CIs on each bar *and* on every
regime-minus-regime difference. Only call a difference real if its CI clears zero.

**Cheap good-enough.** Bootstrap CIs on the headline metric only; show them as error bars on the
Section-1 bars.

---

## TIER 2 — Serious confounds in the outcome measure.

### P4. Citations are not age-normalized (2018 papers had 2 extra years to accumulate)

Raw citation counts conflate merit with exposure time. A 2018 paper and a 2020 paper are not
comparable on total citations. The field×year percentile partly handles year, but via noisy small
cells and error-prone LLM field tags (and field was found not to matter, p=0.36, on incomplete
data).

**Best solution — fixed citation window.** Count citations in a fixed K-year window from
publication (e.g. first 36 months) for every paper. Equal exposure for all → the cleanest removal
of the 2018-vs-2020 gap, standard in bibliometrics. OpenAlex exposes `counts_by_year` per work —
**check whether that was stored; if only the total count was saved, re-fetch with
`counts_by_year`.** This is the highest-value cheap fix in the whole document.

**Cheap good-enough.** Drop field (it doesn't matter), use citation percentile rank *within year*
only — removes year, avoids noisy field cells and dependence on LLM field tags.

---

### P5. Coverage / missingness bias

Two distinct holes:
- **LLM pipeline covers 3,494/4,567.** Missing papers are U-shaped: clear accepts and clear
  rejects under-covered, contested middle well-covered (coverage: <3 rating 51%, 5–6 rating 97%,
  >7 rating 51%). LLM regimes can only be scored on the covered subset, which over-represents the
  hard middle and changes the base rate.
- **~37% of papers have no OpenAlex citation match.** The "impute zeros" toggle is blunt:
  unmatched-because-title-failure (could be high-impact) vs unmatched-because-never-published
  (genuinely ~0) are opposite cases treated identically.

**Best solution.** (a) Evaluate *all* regimes on the common covered subset (3,494) with N
re-derived within it — never compare an LLM regime on 3,494 against a human regime on 4,567,
different denominators. Or complete the pipeline on the missing 1,073 (abstracts exist in the DB).
(b) Fix citation matching first (DOI / OpenAlex ID / fuzzy title) to shrink the 37%; impute only as
last resort and always show impute-vs-drop as a robustness band.

**Cheap good-enough.** Common-subset evaluation + report both impute and drop side by side.

---

## TIER 3 — Design and interpretation.

### P6. Fixed N erases the threshold question and the precision/recall curve

Pinning N = actual accepts reduces everything to "which N papers" (ranking), discards "where to
set the bar" (calibration), and bakes in the human acceptance rate as correct. A regime with a
great ranking but bad calibration looks identical to one with both.

**Best solution.** Report threshold-free ranking metrics: AUC / AUPRC against
citation-ideal-membership, NDCG (rank-weighted, rewards nailing the very top), Spearman between
regime score and citations. The N-pinned overlap becomes one point on a full precision-recall
curve.

---

### P7. The "overall performance" number averages heterogeneous metrics with arbitrary weights

Equal-weighting median citations + mean-log + recall@1/5/10 into one bar has no decision-theoretic
basis. Conferences care most about *not missing breakthroughs* (recall at the very top), which
equal-weighting dilutes.

**Best solution.** State the objective explicitly and let the user pick the utility (e.g.
"minimize missed top-1%" → weight recall@top). Don't collapse heterogeneous metrics without a
stated utility function; if in doubt, show them separately and refuse to average.

---

### P8. Regimes see different information → not a fair "reviewing" comparison

Human AC: full reviews + rebuttals + discussion. Score-top-N: mean rating only. LLM/DeepSeek:
paper text. "LLM beats human" may just mean "this input set was richer/leakier," not "better
judgment."

**Best solution.** Document each regime's input set explicitly. Headline comparison only between
regimes with comparable inputs; treat the rest as exploratory. (Interacts with P2 — leakage is the
sharpest version of this.)

---

### P9. Human-AC is a degenerate "regime" — it's the reference, not a competitor

HumanActual returns the accepted set (n by definition), has no ranking, no tunable threshold, and
folds in desk-rejects / workshop invites / ethics flags that aren't quality judgments.

**Best solution.** Frame AC as the status-quo reference point ("here is what actually happened"),
and express every other regime as lift/drawdown relative to *both* AC and the citation-ideal — not
as a peer on a leaderboard.

---

### P10. Researcher degrees of freedom / garden of forking paths

λ slider × impute toggle × drawdown toggle × year × 6 regimes × 5 metrics = a large space where
*something* will look good somewhere, with no pre-registered primary endpoint.

**Best solution.** Pre-specify ONE primary metric + ONE primary comparison set before looking
(suggested: recall of citation-top-N, 3-yr-windowed, year-normalized, within the RDD margin, on the
common covered subset). Report that as the headline; everything else is explicitly labeled
exploratory/robustness. State the pre-specification on the dashboard.

---

## What a defensible version looks like (minimum bar)

The study is salvageable, but the headline result needs all four of:
1. **P1** — RDD-margin restriction (or venue-premium add-back) so deviation isn't punished.
2. **P2** — leakage screen for LLM regimes, or they're not evidence about reviewing. **DONE** —
   full-corpus five-layer test run; report lift = 1.18 (leakage-excluded), not 1.40.
3. **P3** — bootstrap CIs so rankings clear noise.
4. **P4** — fixed citation window so 2018 ≠ unfair advantage over 2020.

Cheapest path to "credible enough to share": P4 (citation window) + P3 (bootstrap CIs) +
P1-cheap (run headline on the borderline band, label the rest status-quo-biased) + P2 (done).
That's a few days of work and converts the dashboard from "suggestive" to "defensible."

## Framing, regardless of method

Even fully deconfounded, the 2014 NeurIPS experiment found reviewer scores have ~0 correlation
with citations *among accepted papers* (0.051; 0.22 among rejected). The honest claim CitesBench
can make is **comparative** ("regime A aligns with citation impact better than regime B on the same
outcome"), used with the deconfounded outcome as a robustness check — never "we recovered true
paper quality."

