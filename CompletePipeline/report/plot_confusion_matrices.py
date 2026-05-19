#!/usr/bin/env python3
"""
Three-panel confusion matrices:
  (A) Committee recommendation (borderline accept + strong accept → Accept)
  (B) DeepSeek V3.1 decision head
  (C) Gemma-2-9B decision head (positive-bias variant)
"""

from __future__ import annotations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
PLOT_DIR = ROOT / "OutputNew" / "Report" / "RDD_Coarse" / "plots"
EMPIRICS = ROOT / "OutputNew" / "Empirics"

COMMITTEE_RUNS = [
    "gemma_ready7_wave1_cached_v2",
    "gemma_ready8_wave2_incremental",
    "gemma_ready8_wave3_single_managed",
]

GEMMA_PRED = (
    EMPIRICS
    / "decision_head_positive_bias_gemma2_9b_all_20260421"
    / "predictions"
    / "thedatainnovati_6e25_google_gemma_2_9b_it_e9d6e73e.jsonl"
)
DEEPSEEK_PRED = (
    EMPIRICS
    / "decision_head_positive_bias_gemma2_9b_all_20260421"
    / "predictions"
    / "deepseek_v3_1_cached.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_committee(run_dirs: list[str]) -> dict[str, dict]:
    """Return {paper_id: {accepted, recommendation}}."""
    out = {}
    for run in run_dirs:
        run_dir = EMPIRICS / run
        for search_root in [run_dir] + sorted(run_dir.glob("shard_*")):
            papers = search_root / "papers"
            if not papers.is_dir():
                continue
            for p in papers.iterdir():
                cr = p / "coarse_review.json"
                pr = p / "paper_result.json"
                if not cr.exists() or not pr.exists():
                    continue
                if p.name in out:
                    continue
                c = json.loads(cr.read_text())
                r = json.loads(pr.read_text())
                out[p.name] = {
                    "accepted": r.get("accepted"),
                    "recommendation": (c.get("recommendation") or "").strip().lower(),
                }
    return out


def confusion_counts(true_labels, pred_labels):
    """Return (TP, FP, FN, TN) with Accept = positive."""
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "accept" and p == "accept")
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "reject" and p == "accept")
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "accept" and p == "reject")
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "reject" and p == "reject")
    return tp, fp, fn, tn


def plot_cm(ax, tp, fp, fn, tn, title):
    n = tp + fp + fn + tn
    matrix = np.array([[tn, fp], [fn, tp]])
    pct = 100 * matrix / n

    ax.imshow(matrix, cmap="Blues", alpha=0.6, vmin=0, vmax=max(matrix.flat))

    labels_pred = ["Reject", "Accept"]
    labels_true = ["Reject", "Accept"]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}\n({pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_pred, fontsize=10)
    ax.set_yticklabels(labels_true, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)

    # Marginal distributions
    actual_accept = (fn + tp) / n
    actual_reject = (tn + fp) / n
    pred_accept = (fp + tp) / n
    pred_reject = (tn + fn) / n

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Title above: model name only; shared "actual" line is on the figure.
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Caption below the axis: LLM-predicted class balance + metrics
    ax.text(
        0.5, -0.32,
        f"LLM predicts  ·  Accept {pred_accept:.1%}  ·  Reject {pred_reject:.1%}\n"
        f"Acc = {accuracy:.1%}   P = {precision:.1%}   R = {recall:.1%}   F1 = {f1:.1%}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9.5, fontweight="500",
        color="#1f2937",
        linespacing=1.5,
    )


# ---------- collect data ----------

# Committee
committee_data = collect_committee(COMMITTEE_RUNS)
committee_true = []
committee_pred = []
for pid, row in committee_data.items():
    true = "accept" if row["accepted"] == 1.0 else "reject"
    pred = "accept" if row["recommendation"] in ("borderline accept", "strong accept") else "reject"
    committee_true.append(true)
    committee_pred.append(pred)

# DeepSeek
ds_rows = load_jsonl(DEEPSEEK_PRED)
ds_true = [r["true_decision"] for r in ds_rows]
ds_pred = [r["decision"] for r in ds_rows]

# Gemma 2 9B
gemma_rows = load_jsonl(GEMMA_PRED)
gemma_true = [r["true_decision"] for r in gemma_rows]
gemma_pred = [r["decision"] for r in gemma_rows]

print(f"Committee: {len(committee_true)}, DeepSeek: {len(ds_true)}, Gemma-2-9B: {len(gemma_true)}")

# ---------- plot ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))

tp, fp, fn, tn = confusion_counts(committee_true, committee_pred)
plot_cm(axes[0], tp, fp, fn, tn, "(A) Committee\n(Gemma-4-31B)")

tp, fp, fn, tn = confusion_counts(ds_true, ds_pred)
plot_cm(axes[1], tp, fp, fn, tn, "(B) Decision Head\n(DeepSeek V3.1)")

tp, fp, fn, tn = confusion_counts(gemma_true, gemma_pred)
plot_cm(axes[2], tp, fp, fn, tn, "(C) Decision Head\n(Gemma-2-9B)")

fig.suptitle(
    "Confusion Matrices — ICLR 2018–2020",
    fontsize=13, fontweight="bold", y=1.02,
)

# Shared "actual" class balance line (all three models share the same sample).
n_all = len(committee_true)
actual_acc_rate = sum(1 for t in committee_true if t == "accept") / n_all
actual_rej_rate = 1 - actual_acc_rate
fig.text(
    0.5, 0.955,
    f"Actual  ·  Accept {actual_acc_rate:.1%}  ·  Reject {actual_rej_rate:.1%}   (n = {n_all:,})",
    ha="center", va="top",
    fontsize=10.5, color="#4b5563", fontweight="500",
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.22, top=0.80, wspace=0.35)
fig.savefig(PLOT_DIR / "confusion_matrices.pdf", bbox_inches="tight", dpi=300)
fig.savefig(PLOT_DIR / "confusion_matrices.png", bbox_inches="tight", dpi=300)
print(f"Saved to {PLOT_DIR / 'confusion_matrices.png'}")
