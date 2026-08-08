# Abstract-Completion Probe — verbatim memorization (google/gemma-4-31B-it)

Title + year + first abstract sentence → model writes the rest (greedy decode).
Continuation scored against the true remainder vs 5 same-field×year decoys.
N = 297 (stratified by citation decile).

## Memorization gradient by citation decile

| Decile | N | ROUGE-L margin (target − decoy) | 8-gram hit rate | % extractable |
|---|---|---|---|---|
| 0 | 30 | +0.0437 | 0.0000 | 0.0% |
| 1 | 30 | +0.0519 | 0.0000 | 0.0% |
| 2 | 30 | +0.0517 | 0.0000 | 0.0% |
| 3 | 30 | +0.0648 | 0.0000 | 0.0% |
| 4 | 30 | +0.0566 | 0.0000 | 0.0% |
| 5 | 30 | +0.0553 | 0.0000 | 0.0% |
| 6 | 29 | +0.0549 | 0.0000 | 0.0% |
| 7 | 30 | +0.0588 | 0.0000 | 0.0% |
| 8 | 29 | +0.0710 | 0.0023 | 10.3% |
| 9 | 29 | +0.0682 | 0.0046 | 6.9% |

Spearman ρ (citation rank vs ROUGE-L margin): **+0.142** (p=0.0144)
Spearman ρ (citation rank vs 8-gram hit rate): **+0.187** (p=0.00121)

## Convergent validity — does verbatim memorization track outcome recall?

| Probe | N overlap | Spearman ρ vs ROUGE-L margin | p |
|---|---|---|---|
| LAP (decision recall) | 297 | +0.081 | 0.164 |
| FAME (fame recall) | 297 | +0.125 | 0.0317 |

## Exhibit — most-extractable papers

| Title | Citations | ROUGE-L margin | 8-gram rate |
|---|---|---|---|
| GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Languag | 3949 | +0.148 | 0.068 |
| Reformer: The Efficient Transformer | 323 | +0.254 | 0.067 |
| Deep Online Learning Via Meta-Learning: Continual Adaptation for Model | 90 | +0.067 | 0.029 |
| Convergence Guarantees for RMSProp and ADAM in Non-Convex Optimization | 83 | +0.050 | 0.026 |
| DELTA: DEEP LEARNING TRANSFER USING FEATURE MAP WITH ATTENTION FOR CON | 88 | +0.122 | 0.010 |
| Dynamic Graph Representation Learning via Self-Attention Networks | 65 | +0.256 | 0.000 |
| Tree2Tree Learning with Memory Unit | 1 | +0.201 | 0.000 |
| Global Optimality Conditions for Deep Neural Networks | 56 | +0.188 | 0.000 |
| Sample Efficient Deep Neuroevolution in Low Dimensional Latent Space | 3 | +0.161 | 0.000 |
| Weakly-supervised Knowledge Graph Alignment with Adversarial Learning | 9 | +0.159 | 0.000 |
| Learning Implicit Generative Models by Teaching Explicit Ones | 6 | +0.157 | 0.000 |
| Cost-Sensitive Robustness against Adversarial Examples | 13 | +0.152 | 0.000 |
| Latent Topic Conversational Models | 4 | +0.152 | 0.000 |
| LEARNING TO PROPAGATE LABELS: TRANSDUCTIVE PROPAGATION NETWORK FOR FEW | 488 | +0.148 | 0.000 |
| Deli-Fisher GAN: Stable and Efficient Image Generation With Structured | 0 | +0.145 | 0.000 |

## Reading

- "Extractable" = target ROUGE-L beats ALL 5 decoys AND ≥1 verbatim 8-gram
  hit. Overall extractable rate: **1.7%**.
- Positive citation-rank gradient = famous papers' text is preferentially in the
  weights — verbatim-level confirmation of the fame-recall finding.
- **One-sided test**: google/gemma-4-31B-it is instruction-tuned, which suppresses regurgitation.
  Positive results are strong evidence of memorization; a null does NOT prove the
  text is absent from the weights.
