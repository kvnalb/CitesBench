#!/usr/bin/env python3
"""Plot human review scores against decision-head probability scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    parser = argparse.ArgumentParser(description="Plot human scores against decision-head scores.")
    parser.add_argument("--rdd-sample-csv", type=Path, default=DEFAULT_RDD_SAMPLE_CSV)
    parser.add_argument("--deepseek-predictions", type=Path, default=DEFAULT_DEEPSEEK)
    parser.add_argument("--gemma-predictions", type=Path, default=DEFAULT_GEMMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year-min", type=int, default=2018)
    parser.add_argument("--year-max", type=int, default=2023)
    parser.add_argument("--accept-threshold", type=float, default=6.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def normalize_decision(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("accept"):
        return "accept"
    if text.startswith("reject"):
        return "reject"
    return None


def boolish(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "1.0", "true", "accept", "accepted"}:
        return True
    if text in {"0", "0.0", "false", "reject", "rejected"}:
        return False
    return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def decision_probability_to_rating(p_accept: float) -> float:
    """Map accept probability to the review-score scale with p=0.5 at score 6."""
    return max(1.0, min(10.0, 1.0 + 10.0 * p_accept))


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
            true_decision = "accept" if accepted else "reject"
            paper_id = str(row["paper_id"])
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
    n = sum(item["total"] for item in by_year.values())
    accept = sum(item["accept"] for item in by_year.values())
    reject = sum(item["reject"] for item in by_year.values())
    summary = {
        "source_path": str(path),
        "year_min": year_min,
        "year_max": year_max,
        "n": n,
        "accept": accept,
        "reject": reject,
        "missing": sum(item["missing"] for item in by_year.values()),
        "accept_rate": round(accept / n, 6) if n else None,
        "by_year": by_year,
    }
    return rows, summary


def join_predictions(prediction_path: Path, rdd_rows: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    predictions = read_jsonl(prediction_path)
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        paper_id = str(pred.get("paper_id") or "")
        human = rdd_rows.get(paper_id)
        if not human:
            continue
        p_accept = parse_float(pred.get("p_accept"))
        decision = normalize_decision(pred.get("decision"))
        if p_accept is None or decision is None:
            continue
        rows.append(
            {
                **human,
                "model_decision": decision,
                "p_accept": p_accept,
                "decision_head_score": round(decision_probability_to_rating(p_accept), 6),
                "model_source": pred.get("model_label") or pred.get("model_id") or pred.get("source"),
            }
        )
    rows.sort(key=lambda row: (row["year"], row["human_rating_mean"], row["paper_id"]))
    return rows, len(predictions)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 6)


def kde_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(grid)
    if len(values) == 1:
        bandwidth = 0.12
    else:
        std = float(np.std(values, ddof=1))
        q75, q25 = np.percentile(values, [75, 25])
        iqr_sigma = float((q75 - q25) / 1.349) if q75 > q25 else std
        sigma = min(std, iqr_sigma) if iqr_sigma > 0 else std
        bandwidth = 0.9 * sigma * (len(values) ** (-1 / 5)) if sigma > 0 else 0.12
        bandwidth = max(bandwidth, 0.12)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).mean(axis=1) / (bandwidth * math.sqrt(2 * math.pi))


def summarize(rows: list[dict[str, Any]], accept_threshold: float, n_prediction_rows: int) -> dict[str, Any]:
    x = np.asarray([row["human_rating_mean"] for row in rows], dtype=float)
    y = np.asarray([row["decision_head_score"] for row in rows], dtype=float)
    diff = y - x
    openreview_accept = np.asarray([row["true_decision"] == "accept" for row in rows], dtype=bool)
    model_accept = np.asarray([row["model_decision"] == "accept" for row in rows], dtype=bool)
    return {
        "n": len(rows),
        "n_prediction_rows": n_prediction_rows,
        "n_prediction_rows_matched_to_rdd_2018_2023": len(rows),
        "n_prediction_rows_unmatched_to_rdd_2018_2023": n_prediction_rows - len(rows),
        "accept_threshold": accept_threshold,
        "score_mapping": "decision_head_score = clamp(1 + 10*p_accept, 1, 10), so p_accept=0.5 maps to score 6",
        "human_mean": round(float(np.mean(x)), 4) if len(x) else None,
        "human_std": round(float(np.std(x, ddof=1)), 4) if len(x) > 1 else None,
        "model_score_mean": round(float(np.mean(y)), 4) if len(y) else None,
        "model_score_std": round(float(np.std(y, ddof=1)), 4) if len(y) > 1 else None,
        "mae": round(float(np.mean(np.abs(diff))), 4) if len(diff) else None,
        "bias_model_minus_human": round(float(np.mean(diff)), 4) if len(diff) else None,
        "pearson": pearson(x, y),
        "spearman": pearson(rankdata(x), rankdata(y)) if len(x) else None,
        "openreview_accept_rate": round(float(np.mean(openreview_accept)), 4) if len(x) else None,
        "model_accept_rate": round(float(np.mean(model_accept)), 4) if len(y) else None,
        "decision_agreement": round(float(np.mean(openreview_accept == model_accept)), 4) if len(x) else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id",
        "title",
        "year",
        "human_decision",
        "true_decision",
        "accepted",
        "human_rating_mean",
        "score_centered",
        "cutoff",
        "bandwidth",
        "model_decision",
        "p_accept",
        "decision_head_score",
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

    x = np.asarray([row["human_rating_mean"] for row in rows], dtype=float)
    y = np.asarray([row["decision_head_score"] for row in rows], dtype=float)
    accepted = np.asarray([row["true_decision"] == "accept" for row in rows], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.suptitle(f"Human Review Scores vs {label} Decision-Head Scores", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.scatter(
        x[~accepted],
        y[~accepted],
        s=18,
        alpha=0.50,
        color="#2f6fb0",
        edgecolors="white",
        linewidths=0.25,
        label="Rejected by OpenReview",
    )
    ax.scatter(
        x[accepted],
        y[accepted],
        s=18,
        alpha=0.56,
        color="#c83f49",
        edgecolors="white",
        linewidths=0.25,
        label="Accepted by OpenReview",
    )
    ax.plot([1, 10], [1, 10], color="#d64f5a", linestyle="--", linewidth=1.4, label="Model score = human")
    ax.axvline(metrics["accept_threshold"], color="#666666", linewidth=0.9, alpha=0.65)
    ax.axhline(metrics["accept_threshold"], color="#666666", linewidth=0.9, alpha=0.65)
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 10)
    ax.set_xlabel("Human mean rating")
    ax.set_ylabel("Decision-head score from p(accept)")
    ax.set_title("(a) Paper-level score relationship")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    stats = (
        f"n = {metrics['n']}\n"
        f"Pearson r = {metrics['pearson']:.3f}\n"
        f"Spearman rho = {metrics['spearman']:.3f}\n"
        f"MAE = {metrics['mae']:.2f}\n"
        f"Bias model-human = {metrics['bias_model_minus_human']:.2f}"
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
    grid = np.linspace(1, 10, 700)
    human_density = kde_density(x, grid)
    model_density = kde_density(y, grid)
    ax.plot(
        grid,
        human_density,
        color="#4c78a8",
        linewidth=2,
        label=f"Human mean (mean={metrics['human_mean']:.2f}, sd={metrics['human_std']:.2f})",
    )
    ax.fill_between(grid, human_density, color="#4c78a8", alpha=0.17)
    ax.plot(
        grid,
        model_density,
        color="#f58518",
        linewidth=2,
        label=f"{label} score (mean={metrics['model_score_mean']:.2f}, sd={metrics['model_score_std']:.2f})",
    )
    ax.fill_between(grid, model_density, color="#f58518", alpha=0.17)
    ax.axvline(metrics["accept_threshold"], color="#666666", linestyle=":", linewidth=1.1, label="threshold 6.0")
    ax.set_xlim(1, 10)
    ax.set_xlabel("Rating-equivalent score")
    ax.set_ylabel("Density")
    ax.set_title("(b) Score distributions")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Decision-head score maps p(accept) to the 1-10 rating scale as clamp(1 + 10*p(accept), 1, 10); p(accept)=0.5 maps to score 6.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rdd_rows, rdd_summary = load_rdd_sample(args.rdd_sample_csv.resolve(), args.year_min, args.year_max)
    specs = [
        ("DeepSeek", args.deepseek_predictions.resolve(), "deepseek_human_vs_decision_head_scores"),
        ("Gemma-2-9B positive", args.gemma_predictions.resolve(), "gemma_human_vs_decision_head_scores"),
    ]
    summary: dict[str, Any] = {"rdd_label_source": rdd_summary}
    for label, prediction_path, slug in specs:
        rows, n_prediction_rows = join_predictions(prediction_path, rdd_rows)
        metrics = summarize(rows, args.accept_threshold, n_prediction_rows)
        metrics["label"] = label
        metrics["prediction_path"] = str(prediction_path)
        summary[slug] = metrics
        write_csv(args.output_dir / f"{slug}.csv", rows)
        plot_model(rows, metrics, label, args.output_dir / slug)

    (args.output_dir / "human_vs_decision_head_scores_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
