"""
Is the committee's 2018-2020 advantage real skill, or memorisation?

The era comparison on its own cannot say. The committee scores rho 0.49 against
citations on 2018-2020 and 0.23 on 2025, but two explanations fit equally well:
the model has seen the older papers (leakage), or 18 months of citations is simply
a noisier outcome than 7 years (age), which attenuates any correlation.

This separates them with a difference-in-differences, using the humans as a
control. The key property: within an era, the LLM and the human reviewers are
scored against the *identical* outcome. So whatever the citation window does to
measurement quality, it does to both. It cancels in the within-era difference.

  Delta_era = rho(LLM) - rho(human)          within-era advantage, age-cancelled
  DiD       = Delta_1820 - Delta_2025        how much more the LLM leads where it
                                             has plausibly seen the papers

Age alone predicts DiD = 0: both selectors degrade together and the advantage is
preserved. Leakage predicts DiD > 0: only the selector that could have memorised
outcomes loses ground when the papers are new.

The human series is also directly informative — it is an estimate of what the
shorter citation window costs a selector that cannot have memorised anything.

Both the area chair tier and the mean reviewer score are used as controls; they
are independent human signals and agree.

Caveats this does NOT address: the eras differ in acceptance rate, reviewer count
and citation coverage, and the human review process itself changed between 2020
and 2025. DiD assumes those do not differentially advantage the LLM.

Run: python src/analysis/did_leakage.py [--tier-a-only]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compare_eras as ce

OUT_MD = "outputs/did_leakage.md"
OUT_CSV = "outputs/did_leakage.csv"
EVAL_2025 = "outputs/eval_table_2025.csv"

# 2018 called the top tier "Talk"; 2019-2020 called it "Oral". Same rung.
TIER = {"Accept (Poster)": 0, "Accept (Spotlight)": 1,
        "Accept (Talk)": 2, "Accept (Oral)": 2}
N_BOOT = 4000
SEED = 0


def load(tier_a_only):
    tiers = ["A"] if tier_a_only else ["A", "B"]
    ev = pd.read_csv(ce.EVAL_1820, low_memory=False)
    acc = ev[ev["decision"].str.startswith("Accept", na=False)]
    d1 = ce.load_era(acc[["paper_id", "year", "committee_rating", "mean_rating",
                          "decision"]], ce.TIER_1820, "committee_rating", tiers)
    d1["ac"] = d1["decision"].map(TIER)

    r = pd.read_csv(ce.RATE_2025)
    e2 = pd.read_csv(EVAL_2025, low_memory=False)[["paper_id", "mean_rating", "decision"]]
    d2 = ce.load_era(r[["paper_id", "year", "rating"]], ce.TIER_2025, "rating",
                     tiers).merge(e2, on="paper_id", how="left")
    d2["ac"] = d2["decision"].map(TIER)
    return (d1.dropna(subset=["mean_rating", "ac"]),
            d2.dropna(subset=["mean_rating", "ac"]))


def _rho(x, cp):
    return stats.spearmanr(x, cp)[0]


def advantage(d, human_col):
    cp = d["cite_pct"].to_numpy()
    return _rho(d["rating"].to_numpy(), cp) - _rho(d[human_col].to_numpy(), cp)


def boot_ratio(d1, d2, human_col, rng):
    """Ratio-of-ratios. Outcome noise attenuates correlations MULTIPLICATIVELY
    (rho_obs ~ lambda * rho_true), so the additive DiD is the wrong model if the
    two selectors sit at very different rho. lambda is estimated from the control,
    then used to predict where the LLM should land in 2025."""
    a1 = d1[["rating", human_col, "cite_pct"]].to_numpy()
    a2 = d2[["rating", human_col, "cite_pct"]].to_numpy()
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s1 = a1[rng.integers(0, len(a1), len(a1))]
        s2 = a2[rng.integers(0, len(a2), len(a2))]
        lam_num, lam_den = _rho(s2[:, 1], s2[:, 2]), _rho(s1[:, 1], s1[:, 2])
        if lam_den <= 0.01:
            out[i] = np.nan
            continue
        predicted = _rho(s1[:, 0], s1[:, 2]) * (lam_num / lam_den)
        out[i] = predicted - _rho(s2[:, 0], s2[:, 2])     # shortfall vs prediction
    return out[~np.isnan(out)]


def boot_did(d1, d2, human_col, rng):
    """Resample papers within each era independently; recompute the DiD."""
    a1 = d1[["rating", human_col, "cite_pct"]].to_numpy()
    a2 = d2[["rating", human_col, "cite_pct"]].to_numpy()
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s1 = a1[rng.integers(0, len(a1), len(a1))]
        s2 = a2[rng.integers(0, len(a2), len(a2))]
        d_1820 = _rho(s1[:, 0], s1[:, 2]) - _rho(s1[:, 1], s1[:, 2])
        d_2025 = _rho(s2[:, 0], s2[:, 2]) - _rho(s2[:, 1], s2[:, 2])
        out[i] = d_1820 - d_2025
    return out


def main(tier_a_only=False):
    d1, d2 = load(tier_a_only)
    rng = np.random.default_rng(SEED)
    rows, L = [], []

    L += ["# Is the committee's 2018-2020 edge skill or memorisation?", "",
          f"Accepted papers only. Tier {'A' if tier_a_only else 'A+B'} citations. "
          f"n = {len(d1):,} (2018-2020) and {len(d2):,} (2025).", "",
          "## Spearman rho against the within-year citation percentile", "",
          "| selector | 2018-2020 | 2025 | change |", "|---|---|---|---|"]
    for lbl, col in [("LLM committee", "rating"), ("Human reviewers", "mean_rating"),
                     ("Area chair tier", "ac")]:
        r1 = _rho(d1[col], d1["cite_pct"])
        r2 = _rho(d2[col], d2["cite_pct"])
        L.append(f"| {lbl} | {r1:.3f} | {r2:.3f} | {r2 - r1:+.3f} |")
        rows.append({"selector": lbl, "rho_1820": r1, "rho_2025": r2, "change": r2 - r1})

    L += ["", "## Difference-in-differences", "",
          "| control | Δ 2018-2020 | Δ 2025 | DiD | 95% CI | P(DiD<=0) |",
          "|---|---|---|---|---|---|"]
    for lbl, col in [("vs human reviewers", "mean_rating"), ("vs area chair tier", "ac")]:
        a1, a2 = advantage(d1, col), advantage(d2, col)
        b = boot_did(d1, d2, col, rng)
        lo, hi = np.percentile(b, [2.5, 97.5])
        n_le = int((b <= 0).sum())
        p_txt = f"{n_le}/{len(b):,} resamples" if n_le else f"0/{len(b):,} (p < {1/len(b):.0e})"
        L.append(f"| {lbl} | {a1:+.3f} | {a2:+.3f} | **{a1 - a2:+.3f}** | "
                 f"[{lo:+.3f}, {hi:+.3f}] | {p_txt} |")
        rows.append({"selector": f"DiD {lbl}", "rho_1820": a1, "rho_2025": a2,
                     "change": a1 - a2, "ci_lo": lo, "ci_hi": hi,
                     "p_le_0": (b <= 0).mean()})

    # multiplicative model: what the controls predict for the LLM in 2025
    L += ["", "## Multiplicative (attenuation) model", "",
          "Outcome noise scales correlations down proportionally rather than "
          "subtracting a constant, so the additive DiD above is not the only "
          "reasonable model. Estimating the attenuation factor from each control "
          "and applying it to the LLM's 2018-2020 rho:", "",
          "| control | attenuation λ | predicted LLM 2025 | observed | shortfall | 95% CI |",
          "|---|---|---|---|---|---|"]
    llm25 = _rho(d2["rating"], d2["cite_pct"])
    llm18 = _rho(d1["rating"], d1["cite_pct"])
    for lbl, col in [("human reviewers", "mean_rating"), ("area chair tier", "ac")]:
        lam = _rho(d2[col], d2["cite_pct"]) / _rho(d1[col], d1["cite_pct"])
        pred = llm18 * lam
        b = boot_ratio(d1, d2, col, rng)
        lo, hi = np.percentile(b, [2.5, 97.5])
        L.append(f"| {lbl} | {lam:.2f} | {pred:.3f} | {llm25:.3f} | "
                 f"**{pred - llm25:+.3f}** | [{lo:+.3f}, {hi:+.3f}] |")
        rows.append({"selector": f"ratio-model vs {lbl}", "rho_1820": llm18,
                     "rho_2025": llm25, "change": pred - llm25,
                     "ci_lo": lo, "ci_hi": hi})

    L += ["", "## Reading it", "",
          "The direction is the same under both models and both controls: the "
          "committee falls further than a purely-noisier outcome can explain, "
          "whether that outcome is modelled as subtracting a constant or scaling "
          "everything down. The size is model-dependent and belongs in the paper "
          "as a range, roughly +0.13 to +0.24 of rho, never as a point estimate.", "",
          "The strength of the evidence differs by model, and this should not be "
          "smoothed over. The additive DiD is decisive (0 of 4,000 resamples at or "
          "below zero, both controls). The multiplicative model is weaker: with the "
          "area chair control the shortfall excludes zero, but with the human-reviewer "
          "control its 95% interval **includes** zero. That is expected — lambda is a "
          "ratio of two small correlations, so it is estimated imprecisely — but it "
          "means the multiplicative form corroborates the additive result rather than "
          "independently establishing it.", "",
          "## Rivals this does NOT rule out", "",
          "**Text provenance.** The DiD cancels anything hitting both selectors "
          "equally within an era, but the LLM's *input* differs across eras — "
          "archive OCR text in 2018-2020, ReviewArena markdown in 2025 — while the "
          "humans read real PDFs in both. docs/reviewarena_text_quality.md shows "
          "ReviewArena text clears every gate the archive itself used (0% garble, "
          "100% extraction-quality pass, identical call counts), but it compared "
          "ReviewArena 2020 against ReviewArena 2025; the archive's own fulltext was "
          "never available, so archive-vs-ReviewArena has never been measured. It "
          "also states plainly that garble_ratio does not detect the word-level OCR "
          "corruption ReviewArena visibly has.", "",
          "The decisive test is cheap and uses data already on disk: ReviewArena "
          "covers 2,213 papers from 2020, all of which have archive committee "
          "results. Re-running the committee on ReviewArena's 2020 text isolates "
          "text provenance exactly — same papers, same era, same leakage, only the "
          "text changes.", ""]

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    open(OUT_MD, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a-only", action="store_true")
    main(ap.parse_args().tier_a_only)
