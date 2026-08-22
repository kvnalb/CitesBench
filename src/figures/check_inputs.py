"""
Validity gate for the 2018-2020 exhibits. Exits non-zero on failure.

Run before building figures and before handing anything to a reader:

    python src/figures/check_inputs.py

Every check here corresponds to a failure that actually happened in this repo,
not to a failure that could be imagined. Both real ones were silent and were
caught by a person noticing an implausible number, which is not a control:

  - `eval_results.csv` stacks raw counts and normalized percentile ranks in one
    `value` column. A groupby that forgot the filter averaged a median of 184.0
    with a rank of 0.75 and reported 92.4, which looked like a finding.
  - `baselines_cache.csv` was keyed without the citation source, so when the
    outcome swapped from OpenAlex to S2 the stale rows kept matching and every
    lift on the dashboard was computed against the wrong baseline.

The checks are assertions over the built tables rather than a schema framework.
A framework would have caught neither of the above; these do.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figures import spec  # noqa: E402

MAX_COVERAGE_DIFFERENTIAL_PP = 5.0   # measured 3.9; a regression past 5 is a stop
MIN_DISTINCT_SCORES = 2              # a constant score is not a regime

failures, notes = [], []


def check(label, ok, detail=""):
    (notes if ok else failures).append((label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))


def main():
    print(f"eval_table : {spec.EVAL_TABLE}")
    print(f"mode       : {spec.MODE}    tiers: {'+'.join(spec.TIERS)}\n")

    et = spec.read_eval_table()

    print("pool")
    check("one row per paper", not et["paper_id"].duplicated().any())
    check("years are exactly the primary sample",
          set(et["year"].unique()) == set(spec.YEARS),
          f"{sorted(et['year'].unique())}")
    for year in spec.YEARS:
        n = int(et[(et["year"] == year) & et["accepted"]].shape[0])
        check(f"n({year}) is the actual accept count",
              n == spec.N_PINNED[year], f"{n} vs pinned {spec.N_PINNED[year]}")

    print("\noutcome")
    src = et["citation_source"].dropna().unique()
    # Two fetch windows in one column means two snapshot dates in one comparison,
    # and citations grow. This is the check that stops a silent mixed-vintage join.
    check("exactly one citation fetch window", len(src) == 1, str(list(src)))

    tiers = set(et["citation_tier"].dropna().unique())
    check(f"tiers are a subset of {spec.TIERS}", tiers <= set(spec.TIERS), str(sorted(tiers)))

    acc = et["accepted"]
    cov = et[spec.OUTCOME].notna()
    diff = abs(cov[acc].mean() - cov[~acc].mean()) * 100
    check(f"pooled coverage differential under {MAX_COVERAGE_DIFFERENTIAL_PP} pp",
          diff < MAX_COVERAGE_DIFFERENTIAL_PP,
          f"{diff:.1f} pp  (acc {cov[acc].mean():.1%}, rej {cov[~acc].mean():.1%})")
    # Per year, reported rather than asserted: 2018 runs 5.4 pp on 935 papers, which
    # is thinner data rather than a worse match rule. The pooled figure is what the
    # paper quotes, so that is what gates; a year drifting is visible here.
    for year in spec.YEARS:
        y = et[et["year"] == year]
        ya, yc = y["accepted"], y[spec.OUTCOME].notna()
        print(f"        {year}: {abs(yc[ya].mean() - yc[~ya].mean()) * 100:.1f} pp "
              f"(acc {yc[ya].mean():.1%}, rej {yc[~ya].mean():.1%}, n={len(y):,})")

    # A paper we could not match must stay NaN. If an unmatched paper ever carries
    # a 0 it will rank last instead of being excluded, which quietly rewards any
    # regime that avoided it — and imputation has been proposed here more than once.
    unmatched_zero = int((et["citation_tier"].isna() & et[spec.OUTCOME].eq(0)).sum())
    check("unmatched papers are NaN, never imputed as zero", unmatched_zero == 0,
          f"{unmatched_zero} unmatched rows carry 0")

    print("\nregimes")
    for r in spec.REGIMES:
        if r.score is None:
            check(f"{r.label}: selects by decision, no score needed", True)
            continue
        check(f"{r.label}: score column `{r.score}` present",
              r.score in et.columns)
        if r.score not in et.columns:
            continue
        s = et[r.score]
        check(f"{r.label}: score varies", s.nunique() >= MIN_DISTINCT_SCORES,
              f"{s.nunique()} distinct values, {s.notna().mean():.1%} coverage")
        for year in spec.YEARS:
            p = et[et["year"] == year]
            n = spec.N_PINNED[year]
            check(f"{r.label}: {year} has n scored papers to rank",
                  int(p[r.score].notna().sum()) >= n,
                  f"{int(p[r.score].notna().sum())} scored, need {n}")

    print("\nresolution  (reported, not asserted — a coarse regime is a finding)")
    for year in spec.YEARS:
        p, n = et[et["year"] == year], spec.N_PINNED[year]
        for r in spec.REGIMES:
            if r.score is None:
                continue
            own, sup, tied = spec.resolution(p[r.score], n)
            print(f"  {year}  {r.label:<28} {sup:>4}/{n} of the slate "
                  f"({sup/n:>5.1%}) supplied by the tie-break, {tied} tied at cutoff")

    print("\nderived tables")
    if os.path.exists(spec.EVAL_RESULTS):
        # One scale in one column. run_eval.py no longer writes the field-normalized
        # arm by default (its labels cover 59.7% of the pool with a 6.0 pp
        # accept/reject differential), so a second mode appearing here means someone
        # ran --include-normalized and left the output in place.
        modes = set(pd.read_csv(spec.EVAL_RESULTS)["mode"].unique())
        check("eval_results holds one mode only", modes == {spec.MODE},
              f"found {sorted(modes)}; rerun python src/analysis/run_eval.py"
              if modes != {spec.MODE} else "")
        res = spec.read_eval_results()
        dup = res.duplicated(subset=["regime", "year", "metric"]).sum()
        # More than one row per key is exactly the shape that let a mean over two
        # incompatible scales pass as a value.
        check("one row per (regime, year, metric) after the mode filter", dup == 0,
              f"{dup} duplicate keys")
    else:
        check("eval_results.csv exists", False, "run src/analysis/run_eval.py")

    if os.path.exists(spec.CITATIONS):
        stale = os.path.getmtime(spec.CITATIONS) > os.path.getmtime(spec.EVAL_TABLE)
        check("eval_table is not older than citations.csv", not stale,
              "citations.csv is newer — rerun src/build/build_eval_table.py"
              if stale else "")

    # Labels are the join key between exhibit CSVs. fig2 said "LLM council
    # (9 calls)" while fig3 said "LLM council", so joining them returned nothing.
    csvs = [os.path.join(spec.FIG_DIR, f) for f in sorted(os.listdir(spec.FIG_DIR))
            if f.endswith(".csv")] if os.path.isdir(spec.FIG_DIR) else []
    seen = {}
    for path in csvs:
        d = pd.read_csv(path)
        if "regime" in d.columns:
            seen[os.path.basename(path)] = set(d["regime"].dropna().unique())
    if seen:
        known = {r.label for r in spec.REGIMES} | spec.RESERVED_LABELS
        for name, labels in seen.items():
            check(f"{name}: regime labels are canonical", labels <= known,
                  f"unknown: {sorted(labels - known)}" if labels - known else "")

    print(f"\n{len(notes)} passed, {len(failures)} failed")
    if failures:
        print("\nfailures:")
        for label, detail in failures:
            print(f"  - {label}" + (f": {detail}" if detail else ""))
        sys.exit(1)
    print("inputs are consistent with src/figures/spec.py")


if __name__ == "__main__":
    main()
