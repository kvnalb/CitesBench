# OpenAlex vs Semantic Scholar citation counts

Corpus: papers with matched arXiv ID and OpenAlex citations (n=1383, S2 matched: 90.4%).

| Statistic | Value |
|---|---|
| Median S2/OA ratio | 2.88 |
| Mean S2/OA ratio | 4.37 |
| Share with S2 > 2x OA | 70.3% |
| Share with S2 > 5x OA | 25.0% |
| Spearman ρ (OA vs S2) | 0.833 (p=9.9e-323) |
| Top-decile label flips (within-year) | 6.5% |
| Median ratio, accepted / rejected | 3.47 / 2.00 |

## Ratio by whether S2 linked a published (non-arXiv) DOI

| s2_has_pub_doi   |   count |   median |   mean |
|:-----------------|--------:|---------:|-------:|
| False            | 1146.00 |     2.93 |   4.32 |
| True             |  104.00 |     2.09 |   4.94 |

## Worst undercounts

| paper_id   |   year |   oa_citations |   s2_citations |   ratio | s2_venue                                                                |
|:-----------|-------:|---------------:|---------------:|--------:|:------------------------------------------------------------------------|
| rJXMpikCZ  |   2018 |           8340 |          27156 |       3 | International Conference on Learning Representations                    |
| rJzIBfZAb  |   2018 |           1541 |          15115 |      10 | International Conference on Learning Representations                    |
| S1p31z-Ab  |   2018 |           1791 |          12201 |       7 | North American Chapter of the Association for Computational Linguistics |
| r1Ddp1-Rb  |   2018 |           4760 |          12038 |       3 | International Conference on Learning Representations                    |
| Hk99zCeAb  |   2018 |           1559 |           8646 |       6 | International Conference on Learning Representations                    |
| SkeHuCVFDr |   2020 |           2040 |           9030 |       4 | International Conference on Learning Representations                    |
| rJ4km2R5t7 |   2019 |           3949 |           8739 |       2 | BlackboxNLP@EMNLP                                                       |
| rygGQyrFvH |   2020 |           1111 |           4354 |       4 | International Conference on Learning Representations                    |
| rklz9iAcKQ |   2019 |            332 |           3038 |       9 | International Conference on Learning Representations                    |
| SktLlGbRZ  |   2018 |            630 |           3282 |       5 | International Conference on Machine Learning                            |
| r1gR2sC9FX |   2019 |            164 |           2458 |      15 | International Conference on Machine Learning                            |
| S1v4N2l0-  |   2018 |           1540 |           3608 |       2 | International Conference on Learning Representations                    |
| HkgEQnRqYQ |   2019 |            771 |           2803 |       4 | International Conference on Learning Representations                    |
| r1lUOzWCW  |   2018 |             99 |           2094 |      21 | International Conference on Learning Representations                    |
| HkxLXnAcFQ |   2019 |            188 |           2004 |      11 | International Conference on Learning Representations                    |
