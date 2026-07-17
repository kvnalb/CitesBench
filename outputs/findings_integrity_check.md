# Integrity check — docs/notes/07162026_findings.txt

Date: 2026-07-17. Every numeric claim in the findings file was recomputed from the
underlying data files or re-run from the source scripts (`/opt/anaconda3/bin/python`,
repo root). No Together API calls were made; the abstract-completion generations were
verified from their saved outputs, not regenerated.

## Verdict

- **1 data-integrity failure, now fixed**: the committed `outputs/leakage_exclusion_bootstrap_s2_vp.csv` held the wrong run (see below). The *claims* in the findings file were correct; the *file* was not.
- **8 transcription errors corrected** in the findings txt (all small; none change any conclusion).
- Everything else reproduces exactly, including all 12 RDD specs, all bootstrap CIs, the venue-premium reversal, the S2/OA audit, venue coverage, and the field×year cell matrix.

---

## Critical issue: wrong file committed as the S2 venue-premium bootstrap

`outputs/leakage_exclusion_bootstrap_s2_vp.csv` (as committed in `5c7d7ac`) contained the
**premium=0.88, B=200 sensitivity run**, not the headline **premium=1.285, B=2000** run:
its excluded-pool Committee−AC gap read +0.490 [0.168, 0.690] — exactly the claimed 0.88
sensitivity row. The post-sensitivity file restore on 2026-07-16 restored the wrong content.

**Fix applied**: re-ran
`python src/leakage_exclusion_bootstrap.py --citation-source s2 --venue-premium 1.285 --B 2000`
(seed 20260716, deterministic). The regenerated file reproduces every claimed number exactly:
Committee−AC +0.649 [+0.418, +0.886] p<.001; AC lift +0.247; Committee lift +0.896;
Committee−score +0.503; Head−AC +0.580. The corrected file is on disk and needs a commit.
`outputs/leakage_exclusion_bootstrap_openalex_vp.csv` was checked the same way and was correct.

Both sensitivity claims were also re-run (B=200, outputs kept in the session scratchpad,
not `outputs/`, to avoid the same overwrite): premium 0.88 → +0.490 [0.168, 0.690];
premium 1.70 → +0.748 [0.524, 1.003]. Both match the findings file exactly.

---

## Corrections applied to the findings txt

| Location | Was | Is (recomputed) | Source of truth |
|---|---|---|---|
| Bullet 2 / probe section | FAME rho +0.133 (p=0.024) | **+0.125 (p=0.032)** | `leakage_abstract_completion_v1.csv` ⋈ `leakage_fame_v1.csv`, n=297; matches `leakage_abstract_completion_report.md` |
| Bullet 2 / probe section | LAP p=0.18 (rho +0.079) | **p=0.16 (rho +0.081)** | same join with `leakage_lap_v1.csv` |
| Probe section | decile 8: 10.7%, decile 9: 7.1% extractable | **10.3% (3/29), 6.9% (2/29)** | groupby(decile) on the probe CSV; matches the report md |
| Probe section | mean margin decile 8 = +0.072 | **+0.071** (0.0710) | same |
| Bullet 9 | RDD LATE "rises 30–50%" under S2 | **rises 27–48%** (h=0.75: 1.285/1.010=1.27; h=0.5: 1.394/0.940=1.48; h=1.0: 1.33) | re-ran `run_specs_constant`, see below |
| Audit section | 98.6% of records carry an arXiv DOI | **98.8%** of audit-corpus records that have an OA DOI (10.48550/\*); 95.2% if DOI-missing records count in the denominator | `data/OpenAlex/openalex_rdd_arxiv_paper_level.csv` ⋈ eval_table, arXiv-matched + OA-cited rows (n=1,383) |
| Refetch section | OA median accepted 34 | **34.5** (ratio 34.5/6 = 5.8x unchanged) | eval_table medians by decision |
| Refetch section | fame-direction accuracy n=948, 85.2%/87.2%; fame_ud rho 0.116/0.124 | **n=938, 85.3%/86.2%; rho 0.117/0.127** on the final S2 file (committed answers, resp. all probed papers, restricted to papers ranked under both sources) | original numbers were computed against the still-growing `s2_citations_full.csv` on 7/16; not recoverable, final-file values substituted |
| 3c section | OA committee beta se 0.010 | **se 0.009** (0.0094) | statsmodels HC1 re-run, N=1,922 |
| OA exclusion eval | Human AC 0.945 full / 0.900 excluded | **0.948 / 0.891** from `leakage_exclusion_eval.csv` (simulated random baseline); 0.900 is the bootstrap analytic-baseline point — the two conventions differ by <1% and the txt mixed them | pivot of the eval CSV vs `leakage_exclusion_bootstrap_openalex.csv` |

---

## Section-by-section verification

### Abstract-completion probe (bullets 1–2)
**How tested**: recomputed from `outputs/leakage_abstract_completion_v1.csv` (N=297 confirmed):
extractable rate `df.extractable.mean()`, per-decile rates, Spearman correlations
(`scipy.stats.spearmanr`), exhibit rows joined to eval_table titles.
**Result**: 1.7% (5/297) extractable ✓; extractable papers only in deciles 8–9, deciles 0–7 at 0.0% ✓;
rank~margin +0.142 (p=0.0144) ✓; rank~8-gram +0.187 (p=0.0012) ✓; GLUE 3,949 OA cites / 6.8% 8-grams ✓;
Reformer margin +0.254 / 6.7% ✓; margin positive in all 10 deciles, min +0.044 (d0), max +0.071 (d8) ✓
(after corrections above). The generation step itself was not re-run (Together API); scoring columns
are deterministic functions of the saved generations in `leakage_abstract_completion_texts.jsonl`.
**Repro**: `python src/leakage_abstract_completion_v1.py --report-only` regenerates the report from the CSV.

### Fame trace sample
**How tested**: parsed `outputs/leakage_fame_traces_sample30.jsonl`, counted `answer_correct`.
**Result**: 5 TP, 7 TN, 4 FP, 1 FN ✓; 10 unknown answers + 3 committed answers with `answer_correct=None` ✓;
FP year-ranks {0.815, 0.830, 0.843, 0.885} — matches "0.815 to 0.885" ✓; FN = Lite Transformer,
130 citations, rank 0.9415 ✓. The qualitative claim (FP traces retrieve facts absent from the prompt)
was established by reading the traces on 7/16 and was not re-verified quantitatively.

### DDSP spot check
**How tested**: local files only. eval_table `B1x1ma4tDr` → 78 OA citations ✓;
`s2_citations_full.csv` → 485 via title_match, title_sim=1.0 ✓.
**Not independently re-verified**: the OA reverse-citation query (78), and the title/author identity
match across dblp/arXiv/OA/S2 — both are live API spot checks performed 2026-07-16.
Repro: `https://api.semanticscholar.org/graph/v1/paper/ARXIV:2001.04643?fields=citationCount,title,authors`
and `https://api.openalex.org/works?filter=cites:W2995233853` (live counts will have drifted).

### Citation-source audit (bullets 3–4)
**How tested**: replicated the `report()` conventions in `src/compare_citation_sources.py`
(ratio = S2 / max(OA,1) on matched rows) against `outputs/citation_source_comparison.csv`.
**Result**: n=1,383, match 90.4% ✓; median ratio 2.88 ✓; mean 4.37 ✓; >2x 70.3% ✓; >5x 25.0% ✓;
Spearman 0.833 ✓; decile flips 6.5% ✓; median ratio accepted 3.47 / rejected 2.00 ✓;
worst undercount rJXMpikCZ 8,340 vs 27,156 ✓; all 15 worst undercounts have a non-blank S2 venue ✓.
**Repro**: `python src/compare_citation_sources.py --report-only`.

### Full S2 refetch (bullets 5–6)
**How tested**: recomputed from `outputs/s2_citations_full.csv` under the standard gate
(arxiv_batch, or title_sim ≥ 0.9, with non-null count) merged onto eval_table.
**Result**: match rate 91.9% ✓ (4,197/4,567 rows with title_sim ≥ 0.9 — the fetch log's own stat;
coverage is higher, 93.0%, because arXiv-batch matches below sim 0.9 still count);
coverage 93.0% vs OA 71.5% ✓; by decision 93.1/93.0 vs 89.0/62.7 ✓; medians S2 112 vs 5 (22.4x) ✓;
mean-log gaps 2.645 vs 1.491 ✓; both-covered n=3,060, Spearman 0.919 ✓; top-decile flips 180 (5.9%),
using within-year ranks computed on the both-covered subset ✓; prior title run 1,777/1,922 matched,
1,265 blank venue, 920 gained counts ✓. The MAG-only probe (Noise-Based Regularizers) is a 7/16
live API spot check, not re-verified.
**Repro**: gate + merges are the same 6 lines used in `src/leakage_exclusion_bootstrap.py:load_eval_table`.

### Exclusion eval + bootstrap CIs (bullets 7–8)
**How tested**: pivoted `outputs/leakage_exclusion_eval{,_s2}.csv` (mean lift by regime×pool);
read all four `outputs/leakage_exclusion_bootstrap_*.csv` directly.
**Result**: every S2 eval lift ✓ (full: AC 1.648, score 1.476, disagree 1.114, Committee 2.015,
Head 1.806; excluded: 2.061/1.748/2.088/1.764); OA Committee 1.397/1.182 ✓ (AC corrected, see table).
Bootstrap: OA full +0.451 [0.300, 0.583] ✓, OA excluded +0.293 [0.075, 0.564] p=.007 ✓;
S2 full +0.371 [0.184, 0.523] ✓, S2 excluded +0.027 [−0.238, 0.311] p=.833 ✓;
Committee−score excluded +0.339 [0.082, 0.607] p=.011 ✓; Head−AC −0.300 [−0.577, −0.044] p=.017 ✓;
Head−score +0.012 p=.946 ✓; OA Head−AC +0.182 [−0.152, 0.499] p=.282 ✓.
**Repro**: `python src/leakage_exclusion_bootstrap.py [--citation-source s2]` — fixed seed, exact.

### Venue-premium add-back (bullet 10)
**How tested**: full re-run (see Critical issue above). All claimed numbers reproduce exactly,
including both sensitivity bounds.
**Repro**: `python src/leakage_exclusion_bootstrap.py --citation-source s2 --venue-premium 1.285 --B 2000`
(≈2 min; sensitivity: `--venue-premium 0.88|1.70 --B 200`, and move the output aside — the
output filename does not encode the premium value, which is what caused the original overwrite).

### Venue coverage (bullet 11)
**How tested**: S2 "real venue" = gated match with non-blank venue not containing "arXiv";
OA = `outputs/paper_venues.csv` venue_type present and ≠ repository.
**Result**: accepted 1,382/1,526 (90.6%) vs OA 498 (32.6%) ✓; rejected 651/3,041 (21.4%) vs OA 211 (6.9%) ✓;
top rejected venues ICML 154, ICLR 72, NeurIPS 49, AAAI 20, IJCNN 18, IJCAI 17, ECCV 13, JMLR 10 ✓.

### Section 3c regressions
**How tested**: re-ran `citation_pct_rank ~ committee_rating [+ mean_rating] + C(field) + C(year)`
and the interaction spec with statsmodels, HC1 SEs, field-labeled rows only; S2 variant swaps counts
and recomputes the field×year rank.
**Result**: OA N=1,922 beta 0.266 ✓ (se corrected to 0.009); S2 N=2,493 beta 0.284 (se 0.009) ✓;
horse race OA +0.194/+0.063 ✓, S2 +0.175/+0.088 ✓; all eight interaction coefficients and p-values ✓
(OA: RL +0.146 p=.016, theory +0.123 p=.030, gen +0.093 p=.12, nlp +0.100 p=.11;
S2: gen +0.077 p=.049, nlp +0.109 p=.008, RL +0.149 p=.0001, theory +0.109 p=.0005).

### Section 4 fuzzy RDD (bullet 9)
**How tested**: rebuilt the RDD sample exactly as `src/dashboard.py:_load_rdd_sample` does
(`data/OpenAlex/openalex_rdd_dashboard.csv`, in-sample + OA-matched, S2 inner-join swap) and re-ran
`fuzzy_rdd.run_specs_constant` for all 12 specs (2 sources × field FE on/off × h ∈ {0.5, 0.75, 1.0}).
**Result**: all 12 LATEs, CIs, and Ns match to the printed precision ✓ (e.g. S2 h=0.75: 1.285
[0.875, 1.695], N=953; OA: 1.010 [0.656, 1.364], N=1,053; S2+FE h=0.5: 1.824 [0.942, 2.707], N=408).
FS F range 75.9–673.9 ✓ ("76 to 674"); FS jump 0.420–0.636 ✓ ("0.42 to 0.64").
The "30–50% larger" phrasing corrected to 27–48%.

### Field × year cells
**How tested**: `pd.crosstab(eval_table.field, eval_table.year)`.
**Result**: every cell ✓ (cv 18/37/15, gen 106/153/35, nlp 93/111/20, RL 127/208/54,
theory 591/910/248); 2020 unlabeled 1,841/2,213 ✓; label share 59.7% ✓; rank coverage 42.8% ✓.

---

## Claims not independently re-verifiable from local data

1. DDSP identity match (title + 4 authors) across dblp/arXiv/OA/S2 — live API check, 2026-07-16.
2. OA reverse-citation count for W2995233853 — live API, counts drift.
3. The single MAG-only-record probe — live API.
4. The fame trace qualitative claim (FPs retrieve facts absent from the prompt) — established by manual trace reading.
5. Original fame-direction accuracy at n=948 — computed against a mid-fetch snapshot of
   `s2_citations_full.csv` that no longer exists; final-file values substituted in the txt.
