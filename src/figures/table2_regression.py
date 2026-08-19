"""
Table 2: the regression behind Figure 2.

Figure 2 shows three bars. This asks whether the differences between them survive
sampling noise, and reports the one contrast the paper actually claims:
council minus area chairs.

SPECIFICATION. For each regime R, on the pooled 2018-2020 pool of matched papers:

    log1p(citations)_i = a_y + b * selected_R,i + e_i

with year fixed effects a_y. b is the gap in log points between the papers R picks
and the papers it passes over. The contrast of interest is b_council - b_AC,
estimated on the same resamples so the two are perfectly correlated and their
difference has an honest interval.

WHY BOOTSTRAP AND NOT CLUSTERED SEs. The two clusterings available are both broken.
Field is 40% missing and its levels run from n=70 to n=1,749 — an earlier version of
this analysis reported SE 0.0000 because it clustered on a constant. Year gives
three clusters, far below where the cluster-robust asymptotics work. So SEs come
from resampling papers within year (stratified, so the year composition is fixed),
which needs no cluster-count assumption.

The resample also re-runs each regime's selection, which folds the tie-break spread
from Figure 2 into the same interval rather than reporting two separate
uncertainties for one quantity. That matters here: the naive regime's slate is ~50%
decided by ties, so a bootstrap that held selection fixed would understate it badly.

Unmatched papers (170 of 4,567) are dropped, never imputed as zero.

Run: python src/figures/table2_regression.py
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regimes.human_actual import HumanActual
from regimes.llm_committee import LLMCommittee
from regimes.llm_ensemble import LLMEnsemble

EVAL_TABLE = "outputs/eval_table.csv"
OUT_CSV = "outputs/figures/table2_regression.csv"
DRAWS_CSV = "outputs/figures/table2_bootstrap_draws.csv"
OUT_TEX = "outputs/figures/table2_regression.tex"

YEARS = [2018, 2019, 2020]
N_BOOT = 400
SEED = 0

REGIMES = [(HumanActual(), "Human (area chairs)", False),
           (LLMCommittee(), "LLM council (9 calls)", True),
           (LLMEnsemble(), "Naive LLM (1 prompt)", True)]


def fit(pool, selected_ids):
    """Within-year demeaned OLS of log1p(cites) on a selected dummy.

    Demeaning by year IS the year fixed effect for a single regressor, and it keeps
    this to numpy rather than building a design matrix per bootstrap draw.
    """
    d = pool.dropna(subset=["openalex_citations"]).copy()
    d["y"] = np.log1p(d.openalex_citations)
    d["x"] = d.paper_id.isin(selected_ids).astype(float)
    d["y"] -= d.groupby("year")["y"].transform("mean")
    d["x"] -= d.groupby("year")["x"].transform("mean")
    vx = float((d.x ** 2).sum())
    return float((d.x * d.y).sum() / vx) if vx > 0 else np.nan


def select_all(pool):
    """One selection per regime on a given pool, n pinned to that year's accepts."""
    out = {}
    for regime, _, _ in REGIMES:
        ids = []
        for yr in YEARS:
            p = pool[pool.year == yr]
            n = int(p.accepted.sum())
            if n == 0 or len(p) < n:
                continue
            ids += regime.select(p, n)
        out[regime.name] = set(ids)
    return out


def fingerprint():
    """What the cached draws are only valid for.

    Everything the draws depend on: the input bytes, the draw count, the seed, the
    year set and which regimes were run. A cache keyed on less than this is the
    baselines_cache.csv failure — that file kept its rows valid-looking after the
    citation column was swapped underneath it, and served OpenAlex-era baselines
    against S2 outcomes without a word.
    """
    h = hashlib.sha1(open(EVAL_TABLE, "rb").read())
    h.update(repr((N_BOOT, SEED, YEARS, [r[0].name for r in REGIMES])).encode())
    return h.hexdigest()[:16]


def load_draws(fp):
    """Cached per-draw coefficients, or None if they are for something else."""
    if not os.path.exists(DRAWS_CSV):
        return None
    d = pd.read_csv(DRAWS_CSV)
    if d.get("fingerprint", pd.Series([None])).iloc[0] != fp or len(d) != N_BOOT:
        return None
    return d


def compute_draws(et, fp):
    rng = np.random.default_rng(SEED)
    draws = {r[0].name: [] for r in REGIMES}
    contrasts = []
    for _ in range(N_BOOT):
        # stratified by year: resample papers within year so year composition, and
        # therefore n per year, is held fixed across draws
        idx = np.concatenate([rng.choice(g.index.values, len(g), replace=True)
                              for _, g in et.groupby("year")])
        b = et.loc[idx]
        sel = select_all(b)
        vals = {k: fit(b, v) for k, v in sel.items()}
        for k, v in vals.items():
            draws[k].append(v)
        contrasts.append(vals[LLMCommittee().name] - vals[HumanActual().name])

    d = pd.DataFrame(draws)
    d["contrast"] = contrasts
    d["fingerprint"] = fp
    d.to_csv(DRAWS_CSV, index=False)
    return d


def build(refresh=False):
    os.makedirs("outputs/figures", exist_ok=True)
    et = pd.read_csv(EVAL_TABLE, low_memory=False)
    et = et[et.year.isin(YEARS)].copy()
    et["accepted"] = et.decision.str.startswith("Accept", na=False)

    point = {k: fit(et, v) for k, v in select_all(et).items()}

    fp = fingerprint()
    d = None if refresh else load_draws(fp)
    if d is None:
        print(f"computing {N_BOOT} bootstrap draws -> {DRAWS_CSV}")
        d = compute_draws(et, fp)
    else:
        print(f"reusing {len(d):,} cached draws from {DRAWS_CSV} (fingerprint {fp})")
    draws = {r[0].name: d[r[0].name].tolist() for r in REGIMES}
    contrasts = d["contrast"].tolist()

    rows = []
    for regime, label, tie in REGIMES:
        a = np.array(draws[regime.name], dtype=float)
        a = a[~np.isnan(a)]
        rows.append({
            "regime": regime.name, "label": label,
            "coef": point[regime.name],
            "se": a.std(ddof=1),
            "ci_lo": np.percentile(a, 2.5), "ci_hi": np.percentile(a, 97.5),
            "tie_broken": tie, "n_boot": len(a),
        })
    c = np.array(contrasts, dtype=float); c = c[~np.isnan(c)]
    rows.append({
        "regime": "contrast", "label": "Council - area chairs",
        "coef": point[LLMCommittee().name] - point[HumanActual().name],
        "se": c.std(ddof=1), "ci_lo": np.percentile(c, 2.5),
        "ci_hi": np.percentile(c, 97.5), "tie_broken": True, "n_boot": len(c),
        # share of resamples on the wrong side of zero — a p-value's honest cousin
        "frac_le_0": float((c <= 0).mean()),
    })
    t = pd.DataFrame(rows)
    t.to_csv(OUT_CSV, index=False)

    body = "\n".join(
        f"{r.label} & {r.coef:.3f} & ({r.se:.3f}) & [{r.ci_lo:.3f}, {r.ci_hi:.3f}] \\\\"
        + ("\n\\midrule" if r.regime == "Naive LLM (1 prompt)" else "")
        for r in t.itertuples())
    open(OUT_TEX, "w").write(
        "% generated by src/figures/table2_regression.py — do not hand-edit\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "& Coef. & (SE) & 95\\% CI \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        f"% dependent variable log1p(citations), year FE, {N_BOOT} stratified "
        "bootstrap draws that re-run selection\n")

    with pd.option_context("display.width", 200):
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {OUT_CSV}\n-> {OUT_TEX}")
    return t


def demo(refresh=False):
    t = build(refresh).set_index("regime")
    for r in ("Human (AC decisions)", "LLM Committee (Gemma)", "LLM2 (ensemble)"):
        assert t.loc[r, "ci_lo"] > 0, f"{r} should beat the papers it passes over"
    assert t.loc["LLM Committee (Gemma)", "coef"] > t.loc["LLM2 (ensemble)", "coef"]
    # the cache must exist and be stamped for THIS input after a build, or a later
    # run would silently reuse draws belonging to different data
    cached = pd.read_csv(DRAWS_CSV)
    assert cached.fingerprint.iloc[0] == fingerprint(), "cache stamped for other data"
    assert len(cached) == N_BOOT
    con = t.loc["contrast"]
    verdict = ("distinguishable from zero" if con.ci_lo > 0 or con.ci_hi < 0
               else "NOT distinguishable from zero")
    print(f"ok — council - AC = {con.coef:+.3f} log points, "
          f"95% CI [{con.ci_lo:.3f}, {con.ci_hi:.3f}] — {verdict}; "
          f"{con.frac_le_0:.0%} of draws <= 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the draws even if the cache is still valid")
    demo(ap.parse_args().refresh)
