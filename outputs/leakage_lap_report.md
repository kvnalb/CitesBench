# LAP Leakage Test — google/gemma-4-31B-it

## Design (Gao, Jiang, Yan 2026 — adapted)

Decision-only recall query: title + year, NO abstract. Model answers: accepted/rejected/unknown.
LAP = P(accepted) + P(rejected). High LAP = model memorized the outcome.

**N = 3216** papers with both committee_rating and openalex_citations.

---

## LAP Distribution

| Statistic | Value |
|---|---|
| Mean LAP | 0.215 |
| Fraction LAP ≥ 0.50 | 21.5% |
| Fraction LAP ≥ 0.95 (saturated) | 21.5% |
| Fraction with any recall (LAP > 0.05) | 21.5% |
| Recall accuracy on committed answers (LAP ≥ 0.5) | 58.3% |

---

## Regression 1 — Validation (existence of recall)

**Y = log(1+citations) ~ (U-D)**

Does the directional recall signal (title-only, no content) predict actual citations?
Any predictive power here can only come from training-time memorization.

| Coefficient | Estimate | p-value |
|---|---|---|
| Slope on (U-D) | 0.6850 | 3.067e-25 |

✅ Recall channel is directionally informative (p < 0.05) — model has absorbed outcome-relevant content.

---

## Regression 2 — Detection (does memorization flow into forecast?)

**Y = log(1+citations) ~ β₁·committee_rating + β₂·LAP + β₃·(LAP × committee_rating)**

β₃ > 0 with p < 0.05 = contamination confirmed: committee_rating is more accurate
precisely when the model has memorized the paper's outcome.

| Term | β | SE | p-value |
|---|---|---|---|
| Intercept | -4.6069 | 0.2817 | 0 |
| committee_rating (β₁) | 1.3622 | 0.0532 | 0 |
| LAP (β₂) | -2.2698 | 0.5386 | 2.578e-05 |
| LAP × committee_rating (β₃) | 0.4649 | 0.0968 | 1.653e-06 |

---

## Regression 3 — Decomposition (genuine foresight vs contaminated share)

Residualize committee_rating on human mean_rating (what the LLM adds beyond
reviewers), then:

**Y = log(1+citations) ~ mean_rating + resid + LAP + (resid × LAP)**   (N = 3216)

The `resid` coefficient is the LLM's excess predictive power on papers with
NO decision recall (LAP=0) — the defensible "genuine foresight" estimate.
The interaction is the share concentrated on memorized papers — contamination.

| Term | β | SE | p-value |
|---|---|---|---|
| mean_rating | 0.5224 | 0.0157 | 0 |
| resid (foresight at LAP=0) | 0.8667 | 0.0589 | 0 |
| LAP | 0.2266 | 0.0602 | 0.0001702 |
| resid × LAP (contamination) | 0.6669 | 0.1057 | 3.231e-10 |

---

## Verdict: **CONTAMINATION DETECTED**

- β₃ = 0.4649 (p=1.653e-06)
- Foresight at LAP=0: 0.8667 (p=0); contaminated share: 0.6669 (p=3.231e-10)

Caveat: LAP measures recall of the accept/reject *decision*, not of *fame*.
See leakage_fame_v1 for the citation-prominence recall probe, and
leakage_controls for probe validity (placebo) checks.
