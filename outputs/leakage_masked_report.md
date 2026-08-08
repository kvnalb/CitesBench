# Masked Re-Review — identity ablation (google/gemma-4-31B-it)

Within-paper design, N = 122. Each paper scored twice on the same 1-10 rubric:
**original** (title + verbatim abstract) vs **masked** (no title, abstract
paraphrased by meta-llama/Llama-3.3-70B-Instruct-Turbo with proper names genericized).

## Citation-predictive power by arm (Spearman ρ of score vs log citations)

| Stratum | N | original | masked |
|---|---|---|---|
| all | 122 | 0.620 (p=2.67e-14) | 0.357 (p=5.53e-05) |
| high-LAP (memorized) | 59 | 0.668 (p=7.48e-09) | 0.437 (p=0.000532) |
| low-LAP (not memorized) | 63 | 0.530 (p=8.05e-06) | 0.282 (p=0.025) |

## Masking-induced score deflation

Score drop under masking: high-LAP mean Δ = +1.31, low-LAP mean Δ = +0.78 (Mann-Whitney one-sided p = 0.0266)

## Reading

- Predictive power **survives masking** (masked ρ ≈ original ρ, incl. on
  low-LAP papers) → judgment is content-driven; memorization is not the story.
- Predictive power **collapses under masking**, or high-LAP papers lose
  disproportionately more → the original scores rode on identity recall.

Caveat: masking removes the title's semantic content along with its identity
value, so a small ρ drop is expected even with zero leakage; the LAP-stratified
contrast (does high-LAP drop MORE?) is the identifying comparison.
