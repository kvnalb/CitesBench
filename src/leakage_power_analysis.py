"""
Power/precision analysis for the probe-validity controls (leakage_controls.py).

Two separate questions, two separate designs:

1. Fabricated-title placebo — a PRECISION problem. We want the 95% CI on the
   false-positive commit rate to sit clearly below the real full-corpus commit
   rate (~18.9%, from outputs/leakage_lap_v1.csv), even under a conservative
   (higher than the observed 3.3%) assumed true rate. Solved via CI-halfwidth
   sample size, not a hypothesis test — there's no natural null to test against
   other than "no better than chance recall," and we care about the bound, not
   a p-value.

2. Wrong-year probe — an EQUIVALENCE problem. The claim is that wrong-year LAP
   is NOT meaningfully different from correct-year LAP (probe measures memory
   of the paper, not of our framing). That means the pilot's non-significant
   result is not itself evidence for the claim — absence of evidence is not
   evidence of absence. Sized via TOST (two one-sided tests) at a pre-specified
   equivalence margin on the LAP scale.

Run: python src/leakage_power_analysis.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

os.makedirs("outputs", exist_ok=True)
OUT_MD = "outputs/leakage_power_analysis.md"

Z_95 = norm.ppf(0.975)


def n_for_ci_halfwidth(p, halfwidth):
    return Z_95**2 * p * (1 - p) / halfwidth**2


def main():
    lap = pd.read_csv("outputs/leakage_lap_v1.csv")
    lap_valid = lap[lap["lap"].notna()]
    real_rate = (lap_valid["lap"] >= 0.5).mean()

    ctrl = pd.read_csv("outputs/leakage_controls.csv")
    fake_pilot = ctrl[(ctrl["probe"] == "fabricated") & (ctrl["answer"] != "ERROR")]
    wy_pilot = ctrl[(ctrl["probe"] == "wrong_year") & (ctrl["answer"] != "ERROR")]
    wy_cmp = wy_pilot.merge(lap_valid[["paper_id", "lap"]], left_on="probe_id",
                            right_on="paper_id", suffixes=("_wy", "_correct"))
    wy_diff = wy_cmp["lap_correct"] - wy_cmp["lap_wy"]

    lines = ["# Probe-validity power/precision analysis", ""]
    lines.append(f"Real full-corpus commit rate (LAP ≥ 0.5): {real_rate:.1%} (N={len(lap_valid):,}).")
    lines.append("")

    # ── A. Fabricated-title placebo ──────────────────────────────────────────
    lines.append("## A. Fabricated-title placebo — precision-based N")
    lines.append(
        f"Pilot: N={len(fake_pilot)}, observed FP rate = {(fake_pilot['lap'] >= 0.5).mean():.1%}. "
        "Target: 95% CI upper bound stays well below the real commit rate, even under a "
        "conservative assumed true rate above what was observed."
    )
    scenarios = [(0.05, 0.10), (0.08, 0.15)]
    lines.append("")
    lines.append("| assumed true FP rate | target CI upper bound | required N |")
    lines.append("|---|---|---|")
    n_chosen = 0
    for p_assumed, target_upper in scenarios:
        hw = target_upper - p_assumed
        n = n_for_ci_halfwidth(p_assumed, hw)
        n_chosen = max(n_chosen, n)
        lines.append(f"| {p_assumed:.0%} | ≤ {target_upper:.0%} | {n:.0f} |")
    n_fake_target = int(np.ceil(n_chosen / 10) * 10)
    n_fake_target = max(n_fake_target, 150)
    p_obs = (fake_pilot["lap"] >= 0.5).mean() if len(fake_pilot) else 0.033
    se_obs = np.sqrt(p_obs * (1 - p_obs) / n_fake_target)
    lines.append("")
    lines.append(
        f"**Chosen N = {n_fake_target}** (rounds up the worse-case scenario with margin). "
        f"At N={n_fake_target} and the pilot's observed rate (~{p_obs:.1%}), the projected 95% CI "
        f"is ({max(0, p_obs - Z_95 * se_obs):.1%}, {p_obs + Z_95 * se_obs:.1%}) — comfortably "
        f"separated from the real {real_rate:.1%} commit rate by more than 2×."
    )

    # ── B. Wrong-year equivalence ────────────────────────────────────────────
    lines.append("")
    lines.append("## B. Wrong-year probe — equivalence-based N (TOST)")
    lines.append(
        f"Pilot: N={len(wy_cmp)}, mean(correct − wrong-year LAP) = {wy_diff.mean():+.4f}, "
        f"sd = {wy_diff.std():.4f}. This is a validity claim of NO difference, so it needs "
        "an equivalence test (TOST) against a pre-specified margin, not just a non-significant "
        "difference test — non-significance at N=30 is not evidence of equivalence."
    )
    delta = 0.05
    z_a = norm.ppf(0.95)
    z_b = norm.ppf(0.80)
    n_tost = (z_a + z_b) ** 2 * wy_diff.std() ** 2 / delta ** 2
    n_wy_target = int(np.ceil(n_tost / 10) * 10)
    n_wy_target = max(n_wy_target, 300)
    lines.append("")
    lines.append(
        f"Equivalence margin δ = ±{delta} on the LAP (0–1) scale, α=0.05, power=0.80 (TOST): "
        f"required N ≈ {n_tost:.0f}. **Chosen N = {n_wy_target}** per offset."
    )
    se_chosen = wy_diff.std() / np.sqrt(n_wy_target)
    lines.append(
        f"At N={n_wy_target}, SE on the mean diff ≈ {se_chosen:.4f}, 95% CI half-width ≈ "
        f"{Z_95 * se_chosen:.4f} — tight enough to plausibly clear a ±{delta} equivalence band "
        "around the pilot's observed mean diff."
    )
    lines.append("")
    lines.append(
        "Design choice: wrong-year reuses papers already probed for the primary LAP test "
        "(outputs/leakage_lap_v1.csv) — no new abstracts or paraphrasing needed, so scaling "
        "this test is a single cheap LLM call per paper, unlike the masked re-review."
    )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT_MD}")
    print(f"\nTARGETS: n_fake={n_fake_target}  n_wrongyear_per_offset={n_wy_target}")


if __name__ == "__main__":
    main()
