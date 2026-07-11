# Fame-Recall Probe — google/gemma-4-31B-it

Title-only recall of citation prominence (the outcome-side twin of the LAP
decision probe). N = 1922.

| Statistic | Value |
|---|---|
| Mean FAME (commitment) | 0.357 |
| Fraction FAME ≥ 0.5 | 35.6% |
| Recall accuracy on committed answers | 85.4% |
| Spearman ρ (fame U-D vs actual citation rank) | 0.123 (p=6.94e-08) |

## Fame recall by actual citation decile

|   decile |       n |   mean_fame |   frac_said_high |
|---------:|--------:|------------:|-----------------:|
|        0 | 160.000 |       0.265 |            0.000 |
|        1 | 229.000 |       0.297 |            0.013 |
|        2 | 178.000 |       0.298 |            0.006 |
|        3 | 194.000 |       0.294 |            0.021 |
|        4 | 191.000 |       0.243 |            0.026 |
|        5 | 194.000 |       0.294 |            0.015 |
|        6 | 189.000 |       0.338 |            0.058 |
|        7 | 194.000 |       0.418 |            0.057 |
|        8 | 191.000 |       0.451 |            0.136 |
|        9 | 202.000 |       0.649 |            0.431 |

A positive Spearman ρ means the model can identify highly-cited papers from
the title alone — direct evidence that fame is memorized and available to
contaminate any citation-adjacent judgment. Use `fame` alongside `lap` as an
exclusion criterion in leakage_exclusion_eval.
