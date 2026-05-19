#!/usr/bin/env python3
"""Plot decision-head accept/reject outcomes against OpenReview RDD labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
DEFAULT_RDD_SAMPLE_CSV = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / "rdd_sample_year_specific_bandwidth.csv"
DEFAULT_DEEPSEEK = ROOT / "OutputNew" / "Empirics" / "decision_head_positive_bias_gemma2_9b_all_20260421" / "predictions" / "deepseek_v3_1_cached.jsonl"
DEFAULT_GEMMA = ROOT / "OutputNew" / "Empirics" / "decision_head_positive_bias_gemma2_9b_all_20260421" / "predictions" / "thedatainnovati_6e25_google_gemma_2_9b_it_e9d6e73e.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "Results" / "RDD" / "Coarse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accept/reject decision-head outputs.")
    parser.add_argument("--rdd-sample-csv", type=Path, default=DEFAULT_RDD_SAMPLE_CSV)
    parser.add_argument("--deepseek-predictions", type=Path, default=DEFAULT_DEEPSEEK)
    parser.add_argument("--gemma-predictions", type=Path, default=DEFAULT_GEMMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year-min", type=int, default=2018)
    parser.add_argument("--year-max", type=int, default=2023)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def boolish(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "1.0", "true", "accept", "accepted"}:
        return True
    if text in {"0", "0.0", "false", "reject", "rejected"}:
        return False
    return None


def load_rdd_sample(path: Path, year_min: int, year_max: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    counts = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            year = int(float(row["year"]))
            if year < year_min or year > year_max:
                continue
            accepted = boolish(row.get("accepted"))
            decision = normalize_decision(row.get("decision") or row.get("decision_group"))
            if accepted is None and decision is not None:
                accepted = decision == "accept"
            if accepted is None:
                counts[(year, "missing")] += 1
                continue
            paper_id = str(row["paper_id"])
            true_decision = "accept" if accepted else "reject"
            rows[paper_id] = {
                "paper_id": paper_id,
                "title": row.get("title"),
                "year": year,
                "human_rating_mean": float(row["mean_rating"]),
                "human_decision": row.get("decision"),
                "true_decision": true_decision,
                "accepted": int(accepted),
                "cutoff": float(row["cutoff"]),
                "bandwidth": float(row["bandwidth"]),
                "score_centered": float(row["score_centered"]),
            }
            counts[(year, true_decision)] += 1

    by_year: dict[str, dict[str, Any]] = {}
    for year in range(year_min, year_max + 1):
        accept = counts[(year, "accept")]
        reject = counts[(year, "reject")]
        missing = counts[(year, "missing")]
        total = accept + reject + missing
        by_year[str(year)] = {
            "total": total,
            "accept": accept,
            "reject": reject,
            "missing": missing,
            "accept_rate": round(accept / total, 6) if total else None,
        }
    n_total = sum(item["total"] for item in by_year.values())
    n_accept = sum(item["accept"] for item in by_year.values())
    n_reject = sum(item["reject"] for item in by_year.values())
    summary = {
        "source_path": str(path),
        "year_min": year_min,
        "year_max": year_max,
        "n": n_total,
        "accept": n_accept,
        "reject": n_reject,
        "missing": sum(item["missing"] for item in by_year.values()),
        "accept_rate": round(n_accept / n_total, 6) if n_total else None,
        "by_year": by_year,
    }
    return rows, summary


def normalize_decision(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("accept"):
        return "accept"
    if text.startswith("reject"):
        return "reject"
    return None


def join_rows(prediction_path: Path, human_scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pred in read_jsonl(prediction_path):
        paper_id = str(pred.get("paper_id") or "")
        human = human_scores.get(paper_id)
        if not human:
            continue
        true_decision = human["true_decision"]
        model_decision = normalize_decision(pred.get("decision"))
        if true_decision is None or model_decision is None:
            continue
        rows.append(
            {
                **human,
                "model_decision": model_decision,
                "p_accept": pred.get("p_accept"),
                "model_source": pred.get("model_label") or pred.get("model_id") or pred.get("source"),
            }
        )
    rows.sort(key=lambda row: (row["year"], row["score_centered"], row["paper_id"]))
    return rows


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cm = Counter((row["true_decision"], row["model_decision"]) for row in rows)
    tn = cm[("reject", "reject")]
    fp = cm[("reject", "accept")]
    fn = cm[("accept", "reject")]
    tp = cm[("accept", "accept")]
    n = tn + fp + fn + tp

    def div(num: float, den: float) -> float | None:
        return round(num / den, 6) if den else None

    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    return {
        "n": n,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": div(tp + tn, n),
        "balanced_accuracy": round(((recall or 0.0) + (specificity or 0.0)) / 2.0, 6),
        "precision": div(tp, tp + fp),
        "recall": recall,
        "f1": div(2 * tp, (2 * tp) + fp + fn),
        "openreview_accept_rate": div(tp + fn, n),
        "model_accept_rate": div(tp + fp, n),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id",
        "title",
        "year",
        "human_rating_mean",
        "score_centered",
        "cutoff",
        "bandwidth",
        "human_decision",
        "true_decision",
        "accepted",
        "model_decision",
        "p_accept",
        "model_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_model(rows: list[dict[str, Any]], metrics: dict[str, Any], label: str, output_prefix: Path) -> None:
    cache_root = Path("/tmp") / "llmreview_matplotlib_cache"
    (cache_root / "mpl").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260421)
    x = np.asarray([row["score_centered"] for row in rows], dtype=float)
    y_base = np.asarray([1.0 if row["model_decision"] == "accept" else 0.0 for row in rows], dtype=float)
    y = y_base + rng.uniform(-0.08, 0.08, size=len(rows))
    accepted = np.asarray([row["true_decision"] == "accept" for row in rows], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 0.9]})
    fig.suptitle(f"OpenReview Outcomes vs {label} Accept/Reject Decisions", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.scatter(
        x[~accepted],
        y[~accepted],
        s=17,
        alpha=0.46,
        color="#2f6fb0",
        edgecolors="white",
        linewidths=0.25,
        label="OpenReview reject",
    )
    ax.scatter(
        x[accepted],
        y[accepted],
        s=17,
        alpha=0.52,
        color="#c83f49",
        edgecolors="white",
        linewidths=0.25,
        label="OpenReview accept",
    )
    ax.axvline(0.0, color="#555555", linewidth=1, alpha=0.6, linestyle=":")
    ax.axhline(0.5, color="#555555", linewidth=1, alpha=0.6, linestyle=":")
    max_abs = max(0.2, float(np.nanmax(np.abs(x))) if len(x) else 1.0)
    ax.set_xlim(-max_abs * 1.04, max_abs * 1.04)
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Model reject", "Model accept"])
    ax.set_xlabel("Mean review rating minus year-specific RDD cutoff")
    ax.set_title("(a) Model decision by RDD running variable")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    stats = (
        f"n = {metrics['n']}\n"
        f"Accuracy = {metrics['accuracy']:.3f}\n"
        f"Balanced acc. = {metrics['balanced_accuracy']:.3f}\n"
        f"Precision = {metrics['precision']:.3f}\n"
        f"Recall = {metrics['recall']:.3f}\n"
        f"F1 = {metrics['f1']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        stats,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
        fontsize=10,
    )

    ax = axes[1]
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=float)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model reject", "Model accept"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OpenReview reject", "OpenReview accept"])
    ax.set_title("(b) Confusion matrix")
    for i in range(2):
        for j in range(2):
            value = int(matrix[i, j])
            pct = value / metrics["n"] if metrics["n"] else 0.0
            color = "white" if matrix[i, j] > matrix.max() * 0.55 else "#1f1f1f"
            ax.text(j, i, f"{value}\n({pct:.1%})", ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    human_scores, rdd_summary = load_rdd_sample(args.rdd_sample_csv.resolve(), args.year_min, args.year_max)
    specs = [
        ("DeepSeek", args.deepseek_predictions.resolve(), "deepseek_accept_reject"),
        ("Gemma-2-9B positive", args.gemma_predictions.resolve(), "gemma_accept_reject"),
    ]
    summary: dict[str, Any] = {"rdd_label_source": rdd_summary}
    for label, prediction_path, slug in specs:
        prediction_rows = read_jsonl(prediction_path)
        rows = join_rows(prediction_path, human_scores)
        metrics = compute_metrics(rows)
        metrics["label"] = label
        metrics["prediction_path"] = str(prediction_path)
        metrics["n_prediction_rows"] = len(prediction_rows)
        metrics["n_prediction_rows_matched_to_rdd_2018_2023"] = len(rows)
        metrics["n_prediction_rows_unmatched_to_rdd_2018_2023"] = len(prediction_rows) - len(rows)
        summary[slug] = metrics
        write_csv(args.output_dir / f"{slug}.csv", rows)
        plot_model(rows, metrics, label, args.output_dir / slug)

    (args.output_dir / "decision_head_accept_reject_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
