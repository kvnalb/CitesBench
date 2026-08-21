# CitesBench: what the data supports, and what it does not

A status note for PIs. Written before the figures go out, so the numbers below are
the ones the figures will show.

---

## 1. The claim the draft can make today

The benchmark works. The sample is complete. The outcome measure is sound.

Three of the four claims in the draft hold. One does not hold yet.

| Draft claim | Status |
|---|---|
| Peer review can be scored as a selection function | **Holds.** 4,567 submissions, n pinned per year, every regime returns exactly n papers. |
| LLM regimes beat human area chairs on recall of the top papers | **Holds.** Council 84.5% vs 73.1% at the top 10%. |
| LLM regimes beat human area chairs on average citation quality | **Does not hold.** The difference is not distinguishable from zero. |
| We can estimate the ICLR acceptance premium | **Holds, with a wide interval.** +1.33 log points, standard error 0.66. |

One problem blocks publication. Section 5 gives it.

---

## 2. What the data covers

We use six sources. Each row below is one source. The last column is the gap
between accepted and rejected papers.

| Source | What it gives us | Coverage | Gap |
|---|---|---:|---:|
| OpenReview submission | abstract, decision, year | 100.0% | +0.0 pp |
| OpenReview reviews | scores, disagreement, review count | 99.7% | +0.4 pp |
| Our council run | council rating | 98.5% | +1.4 pp |
| Our single-call run | single-call rating | 98.3% | +1.5 pp |
| Semantic Scholar record | venue, team size, h-index, date | 96.8% | +2.6 pp |
| Semantic Scholar citations | **the outcome** | 96.3% | +4.2 pp |
| OpenReview keywords | keyword count | 90.9% | +3.1 pp |
| OpenAlex author record | country and team flags | 71.5% | +26.3 pp |
| OpenReview confidence | reviewer confidence | 51.3% | +5.6 pp |
| OpenAlex institution | named institution, industry flag | 39.5% | +42.9 pp |

### Why the gap column matters

The benchmark asks one question. Does a regime pick better papers from a pool of
accepts and rejects? If we find data more easily for accepted papers, then a
regime gets a reward for picking accepted papers. The reward is not for quality.
The gap column measures that risk for each source.

The outcome has a gap of 4.2 pp. This is small. The old OpenAlex citation pull had
a gap of 26.3 pp. We replaced it for this reason.

The two OpenAlex author sources have large gaps. Do not use them for a main result.

### Coverage is a property of the source, not the variable

Ten sources give ten numbers. Every variable inherits the number of its source.
This is why the coverage table has ten rows and not twenty-four.

---

## 3. What we ran, and what each run shows

### 3.1 The main comparison

Three regimes on the same pool. Four metrics.

| Metric | Human ACs | Council (9 calls) | Single call (1 call) |
|---|---:|---:|---:|
| Median citations | 123.2 | 131.0 | 110.6 |
| Mean log(1 + citations) | 4.791 | 4.786 | 4.538 |
| Recall of the top 1% | 77.2% | 94.9% | 97.2% |
| Recall of the top 10% | 73.1% | 84.5% | 78.9% |

### 3.2 The regression behind the comparison

We bootstrap the selection 400 times, with year fixed effects.

Council minus area chairs is **−0.049 log points**. The 95% interval is
**[−0.153, +0.040]**. The interval contains zero. 88% of draws put the council at
or below the area chairs.

**Read this carefully.** On average citation quality, the council does not beat the
humans. It also does not lose to them. The draft must not claim a win here.

### 3.3 Where the LLM regimes do win

Recall of the top papers. The council finds 84.5% of the true top 10%. The area
chairs find 73.1%. This gap is large and it is stable.

This is a different objective from average quality. A regime can find more of the
best papers and still not raise the average. The draft should state both results
and say they answer different questions.

### 3.4 Ties are part of the result

An LLM gives coarse scores. Many papers tie at the cutoff. The regime does not
choose between them.

The single call must break ties for **77%** of its admitted set. The council must
break ties for **11%**.

So the single call's 97.2% recall is not a decision the model made. It is mostly a
coin flip. We report a range for each regime, not a single number. The range width
is the regime's resolution. This is a finding, not noise.

### 3.5 The venue premium

Acceptance raises citations by itself. We must remove that effect before we compare
regimes.

We use a fuzzy regression discontinuity design at the score cutoff.

| Estimate | Value |
|---|---|
| Pooled, all years | +1.41 log points (se 0.60), 95% CI [+0.25, +2.58] |
| Per year, precision-weighted | +1.33 log points (se 0.66) |
| Specification curve, strong instrument only | +0.07 to +1.61, median +0.89 |

The design passes its balance tests. Nine covariates, two failures. Both failures
are year dummies. This is a pooling artifact and not a design failure.

### 3.6 What happens after we remove the premium

We sweep the premium from 0 to 2 log points and re-run the comparison.

Every point where a regime overtakes the area chairs sits **inside** the interval
we estimated for the premium. So the ranking is not robust to the correction.

**Say this plainly in the paper.** Do not pick one premium value and report the
ranking at that value.

### 3.7 Memorization

The model saw these papers in training. We ran three probes.

| Probe | What it asks | Result |
|---|---|---|
| LAP | Do you recall the accept or reject decision? | 21.5% commit. 58.3% correct. Near chance. |
| FAME | Do you recall that this paper is famous? | 35.6% commit. 85.4% correct. |
| Abstract completion | Can you write the rest of the abstract? | Verbatim recall only in the top two citation deciles. |

FAME is the concern. The correlation between recalled fame and true citation rank
is +0.12 and highly significant. In the top citation decile, the model calls 43% of
papers famous. In the bottom decile it calls 0% famous.

So the model does not recall decisions. It does recall fame. Fame is our outcome.

We report results with and without the recalled papers. The effect gets smaller. It
does not disappear.

---

## 4. What we can infer

1. Human area chairs are weak at finding the top papers. They find 73% of the true
   top 10%. This agrees with Cortes and Lawrence.
2. An LLM council finds more of the top papers than the area chairs do.
3. An LLM council does not produce a better average outcome than the area chairs.
4. Nine calls beat one call on average quality. One call is not a cheap substitute.
5. Most of the single call's apparent success comes from tie-breaking, not from
   judgment.
6. Acceptance at ICLR raises later citations. The best estimate is about +1.3 log
   points. The interval is wide.

---

## 5. What we must be careful about

### 5.1 The council read the decisions (blocking)

We audited the text files that we gave the council. **282 of 300 files (94.0%)
contain the ICLR decision in a running header or footer.**

So the council did not review a blind submission. It read a paper that states its
own outcome.

This affects every council number in Section 3. We must re-run the council on
anonymised text before we submit. `src/build/anonymize_fulltext.py` exists for this.

This is the only blocking problem in the draft.

### 5.2 Citation age is not adjusted

A 2018 paper has eight years to collect citations. A 2020 paper has six. We do not
correct for this.

Each year is selected on its own, so the comparison inside a year is fair. The
pooled median is not.

We now hold the publication date to the day for 96.8% of papers. So we can fix this.
It is work, not a blocker.

### 5.3 Field normalization is not available

Reviewers will ask for field-adjusted citations. We cannot give it.

- Our own field labels cover 59.7% of papers. The gap is 6.0 pp. 1,749 of 2,726
  labels are one catch-all class.
- Semantic Scholar labels 96.8% of papers with a 2.6 pp gap. But **94.7% of them
  say "Computer Science"**. This is not a partition. Normalizing inside it is the
  same as not normalizing.

State in the paper that the outcome is raw citations. Give the reason.

### 5.4 Institution heterogeneity is not available

Institution data covers 39.5% of papers. The gap is 42.9 pp. Accepted papers are
almost three times as likely to have a named institution.

Any cut by institution measures the data, not the papers. Do not run it.

Team seniority is available. Semantic Scholar gives h-index and team size for 96.8%
of papers, with a 2.8 pp gap. Use that instead.

### 5.5 The pool holds duplicate papers

52 papers (1.14%) are the same paper, submitted in two different years. 45 were
rejected. 7 were accepted.

This does not break the design. We pin n per year and select each year alone. But a
few papers get two chances at acceptance. Put the number in the sample table.

### 5.6 One citation bug is fixed, and it was directional

Two of our submissions can match one Semantic Scholar record. Our old rule gave the
record to the better title match. Semantic Scholar stores stale titles, so the rule
gave three accepted papers' citations to the rejected version.

The worst case was AdamW. The rejected 2018 submission took 2,738 citations. The
accepted 2019 paper got nothing.

We fixed the rule. The accepted paper now wins the record. The effect on the
headline is small, at most +0.6 pp, and it helps the humans.

---

## 6. Decisions we need from you

1. **Re-run the council on anonymised text?** This is the blocking item. It costs
   compute and about a day.
2. **Adjust for citation age?** We have the dates. A common window will shrink the
   sample or need a model.
3. **Report the raw ranking, or only the premium-adjusted sweep?** We recommend the
   sweep, because the ranking flips inside the premium's own interval.
4. **Include the memorization probes as a main table or an appendix?** They show
   attenuation, not immunity.
