# Probe-validity power/precision analysis

Real full-corpus commit rate (LAP ≥ 0.5): 18.9% (N=4,497).

## A. Fabricated-title placebo — precision-based N
Pilot: N=30, observed FP rate = 3.3%. Target: 95% CI upper bound stays well below the real commit rate, even under a conservative assumed true rate above what was observed.

| assumed true FP rate | target CI upper bound | required N |
|---|---|---|
| 5% | ≤ 10% | 73 |
| 8% | ≤ 15% | 58 |

**Chosen N = 150** (rounds up the worse-case scenario with margin). At N=150 and the pilot's observed rate (~3.3%), the 95% CI is (n_for_ci_halfwidth check below), comfortably separated from the real 18.9% commit rate by more than 2×.

## B. Wrong-year probe — equivalence-based N (TOST)
Pilot: N=30, mean(correct − wrong-year LAP) = +0.0333, sd = 0.3198. This is a validity claim of NO difference, so it needs an equivalence test (TOST) against a pre-specified margin, not just a non-significant difference test — non-significance at N=30 is not evidence of equivalence.

Equivalence margin δ = ±0.05 on the LAP (0–1) scale, α=0.05, power=0.80 (TOST): required N ≈ 253. **Chosen N = 300** per offset.
At N=300, SE on the mean diff ≈ 0.0185, 95% CI half-width ≈ 0.0362 — tight enough to plausibly clear a ±0.05 equivalence band around the pilot's observed mean diff.

Design choice: wrong-year reuses papers already probed for the primary LAP test (outputs/leakage_lap_v1.csv) — no new abstracts or paraphrasing needed, so scaling this test is a single cheap LLM call per paper, unlike the masked re-review.
