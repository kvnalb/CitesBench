"""
Probe-validity (placebo) controls for the LAP leakage test.

Two probes against the same model + prompt as leakage_lap_v1:
  1. fabricated — plausible but nonexistent ICLR-style titles. Any commitment
     (LAP > 0) is a false positive: the probe's acquiescence base rate.
  2. wrong_year — real already-probed titles asked with the wrong year (+1).
     If recall is genuine memory of the venue-year outcome, commitment should
     match the correct-year probe (model recalls the paper, not our framing).

Output: outputs/leakage_controls.csv (incremental, resumable) + printed summary.

Run: python src/leakage_controls.py [--smoke] [--n-fake 30] [--n-wrongyear 30]
"""
import os
import sys
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from leakage_lap_v1 import probe_one, recall_prompt, MODEL

load_dotenv()
os.makedirs("outputs", exist_ok=True)

OUT_CSV = "outputs/leakage_controls.csv"
LAP_CSV = "outputs/leakage_lap_v1.csv"

# Fabricated but plausible ICLR-style titles (no such papers exist).
FAKE_TITLES = [
    "Spectral Gating Networks for Efficient Sequence Modeling",
    "Adversarially Regularized Curriculum Distillation",
    "Latent Manifold Alignment for Cross-Domain Few-Shot Learning",
    "Gradient Surgery with Momentum Reprojection for Multi-Task Learning",
    "Stochastic Depth Annealing in Very Deep Residual Networks",
    "Contrastive Predictive Routing for Modular Neural Computation",
    "Entropy-Constrained Policy Iteration with Learned Temperature",
    "Hierarchical Attention Bottlenecks for Compositional Generalization",
    "Implicit Neural Priors for Unsupervised Anomaly Segmentation",
    "Cyclical Weight Averaging Beyond Flat Minima",
    "Bayesian Mode Connectivity in Overparameterized Networks",
    "Sparse Hypernetwork Distillation for On-Device Adaptation",
    "Invariant Risk Extrapolation under Covariate Drift",
    "Neural Rejection Sampling for Amortized Inference",
    "Dual-Critic Advantage Decomposition in Cooperative Multi-Agent RL",
    "Structured Dropout as Implicit Ensemble Distillation",
    "Metric-Aware Prototype Refinement for Long-Tailed Recognition",
    "Recurrent Energy Matching for Score-Based Generative Models",
    "Permutation-Equivariant Value Factorization for Coordination Games",
    "Lipschitz-Certified Attention for Robust Sequence Classification",
    "Self-Supervised Frequency Decoupling for Domain Generalization",
    "Monotone Operator Splitting for Deep Equilibrium Training",
    "Curvature-Adaptive Learning Rates via Hessian Sketching",
    "Counterfactual Data Augmentation with Learned Interventions",
    "Asynchronous Knowledge Routing in Mixture-of-Expert Transformers",
    "Variational Subgoal Discovery for Sparse-Reward Exploration",
    "Topology-Preserving Graph Coarsening for Scalable GNNs",
    "Calibrated Uncertainty Propagation in Deep State-Space Models",
    "Prototype-Guided Memory Consolidation for Continual Learning",
    "Reweighted Score Matching under Heavy-Tailed Noise",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="5 probes total")
    parser.add_argument("--n-fake", type=int, default=30)
    parser.add_argument("--n-wrongyear", type=int, default=30)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        sys.exit("ERROR: TOGETHER_API_KEY not set in .env")

    # Build probe list: (probe_type, probe_id, title, year_asked)
    probes = []
    for i, t in enumerate(FAKE_TITLES[: args.n_fake]):
        probes.append(("fabricated", f"fake_{i:03d}", t, 2019))

    if not os.path.exists(LAP_CSV):
        sys.exit(f"ERROR: {LAP_CSV} not found — run leakage_lap_v1.py first "
                 "(wrong-year probes reuse its papers).")
    lap_done = pd.read_csv(LAP_CSV)
    lap_done = lap_done[lap_done["answer"] != "ERROR"]
    titles = pd.read_csv("outputs/eval_table.csv")[["paper_id", "title"]]
    lap_done = lap_done.merge(titles, on="paper_id")
    for row in lap_done.head(args.n_wrongyear).itertuples():
        probes.append(("wrong_year", row.paper_id, row.title, int(row.year) + 1))

    if args.smoke:
        # ponytail: 3 fake + 2 wrong-year is enough to prove the plumbing
        probes = probes[:3] + [p for p in probes if p[0] == "wrong_year"][:2]

    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev["probe"], prev["probe_id"]))
    todo = [p for p in probes if (p[0], p[1]) not in done]
    print(f"Model: {MODEL}")
    print(f"Probes: {len(probes)} total, {len(done)} done, {len(todo)} to fetch")

    from openai import OpenAI
    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    lock = threading.Lock()
    counter = [0]

    with open(OUT_CSV, "a") as fout:
        if write_header:
            fout.write("probe,probe_id,year_asked,answer,p_accept,p_reject,p_unknown,lap,ud\n")

        def fetch_one(probe):
            ptype, pid, title, year = probe
            client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
            try:
                answer, p_acc, p_rej, p_unk, _ = probe_one(client, recall_prompt(title, year))
            except Exception as e:
                with lock:
                    fout.write(f"{ptype},{pid},{year},ERROR,,,,,\n")
                    fout.flush()
                print(f"  SKIP {pid}: {e}")
                return
            lap = p_acc + p_rej
            ud = p_acc - p_rej
            with lock:
                fout.write(f"{ptype},{pid},{year},{answer},"
                           f"{p_acc:.6f},{p_rej:.6f},{p_unk:.6f},{lap:.6f},{ud:.6f}\n")
                fout.flush()
                counter[0] += 1
                print(f"  {counter[0]}/{len(todo)}  [{ptype}] {pid}  answer={answer}  lap={lap:.3f}")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, p) for p in todo]
            for f in as_completed(futures):
                f.result()

    # ── Summary ────────────────────────────────────────────────────────────────
    res = pd.read_csv(OUT_CSV)
    res = res[res["answer"] != "ERROR"]
    print("\n=== Control summary ===")
    fab = res[res["probe"] == "fabricated"]
    if len(fab):
        print(f"fabricated  (n={len(fab)}): false-positive rate (LAP≥0.5) = "
              f"{(fab['lap'] >= 0.5).mean():.1%}, mean LAP = {fab['lap'].mean():.3f}")
    wy = res[res["probe"] == "wrong_year"]
    if len(wy):
        orig = pd.read_csv(LAP_CSV)
        cmp = wy.merge(orig[["paper_id", "lap"]], left_on="probe_id",
                       right_on="paper_id", suffixes=("_wrongyr", "_correct"))
        print(f"wrong_year  (n={len(wy)}): mean LAP wrong-year = {cmp['lap_wrongyr'].mean():.3f} "
              f"vs correct-year = {cmp['lap_correct'].mean():.3f}")
    print(f"\nOutput: {OUT_CSV}")


if __name__ == "__main__":
    main()
