#!/usr/bin/env python3
"""Plot human mean review ratings against final LLM committee ratings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
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
DEFAULT_OUTPUT_ROOTS = [
    ROOT / "OutputNew" / "Empirics" / "gemma_ready7_wave1_cached_v2",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave2_incremental",
    ROOT / "OutputNew" / "Empirics" / "gemma_ready8_wave3_single_managed",
    ROOT / "OutputNew" / "Coarse",
]
DEFAULT_REVIEW_DB = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "Empirics" / "human_vs_llm_committee_scores_20260421"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot human mean rating vs LLM committee rating.")
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--generated-output-roots", type=Path, nargs="*", default=DEFAULT_OUTPUT_ROOTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--accept-threshold", type=float, default=6.0)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_numeric_score(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    text = str(raw).strip()
    if not text or text.lower() in {"n/a", "not applicable", "none", "nan"}:
        return None
    match = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_generated_reviews(roots: list[Path]) -> dict[str, Path]:
    """Return one generated review path per paper, choosing latest mtime on duplicates."""
    paths: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("**/coarse_review.json"):
            try:
                payload = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            paper_id = str(payload.get("paper_id") or path.parent.name)
            if not paper_id:
                continue
            old = paths.get(paper_id)
            if old is None or path.stat().st_mtime > old.stat().st_mtime:
                paths[paper_id] = path
    return paths


def load_human_scores(db_path: Path) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                r.paper_id,
                r.reviewer_id,
                r.rating,
                s.title,
                s.decision,
                s.when_submitted
            FROM REVIEW r
            JOIN SUBMISSION s ON r.paper_id = s.id
            ORDER BY r.paper_id, r.reviewer_id
            """
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["paper_id"])].append(dict(row))

    out: dict[str, dict[str, Any]] = {}
    for paper_id, paper_rows in grouped.items():
        values = [parse_numeric_score(row.get("rating")) for row in paper_rows]
        numeric = [value for value in values if value is not None]
        if not numeric:
            continue
        out[paper_id] = {
            "paper_id": paper_id,
            "title": paper_rows[0].get("title"),
            "decision": paper_rows[0].get("decision"),
            "year": paper_rows[0].get("when_submitted"),
            "n_reviews": len(paper_rows),
            "n_numeric_reviews": len(numeric),
            "human_rating_mean": round(float(np.mean(numeric)), 6),
            "human_rating_std": round(float(np.std(numeric, ddof=1)), 6) if len(numeric) > 1 else 0.0,
            "human_rating_min": round(float(min(numeric)), 6),
            "human_rating_max": round(float(max(numeric)), 6),
            "human_rating_values": numeric,
        }
    return out


def build_joined_rows(generated_paths: dict[str, Path], human_scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper_id, path in sorted(generated_paths.items()):
        human = human_scores.get(paper_id)
        if not human:
            continue
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        llm_rating = parse_numeric_score(payload.get("rating"))
        if llm_rating is None:
            continue
        rows.append(
            {
                **human,
                "llm_rating": round(float(llm_rating), 6),
                "source_decision_from_generated_file": payload.get("decision"),
                "llm_recommendation": payload.get("recommendation"),
                "llm_confidence": parse_numeric_score(payload.get("confidence")),
                "llm_soundness": parse_numeric_score(payload.get("soundness")),
                "llm_presentation": parse_numeric_score(payload.get("presentation")),
                "llm_contribution": parse_numeric_score(payload.get("contribution")),
                "generated_review_path": str(path),
            }
        )
    return rows


def is_human_accepted(decision: Any) -> bool:
    text = str(decision or "").strip().lower()
    return text.startswith("accept")


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
    density = np.exp(-0.5 * z * z).mean(axis=1) / (bandwidth * math.sqrt(2 * math.pi))
    return density


def summarize(rows: list[dict[str, Any]], accept_threshold: float) -> dict[str, Any]:
    x = np.asarray([row["human_rating_mean"] for row in rows], dtype=float)
    y = np.asarray([row["llm_rating"] for row in rows], dtype=float)
    diff = y - x
    human_accept = x >= accept_threshold
    llm_accept = y >= accept_threshold
    return {
        "created_at_utc": now_utc(),
        "n_papers": len(rows),
        "accept_threshold": accept_threshold,
        "human_mean": round(float(np.mean(x)), 4) if len(x) else None,
        "human_std": round(float(np.std(x, ddof=1)), 4) if len(x) > 1 else None,
        "llm_mean": round(float(np.mean(y)), 4) if len(y) else None,
        "llm_std": round(float(np.std(y, ddof=1)), 4) if len(y) > 1 else None,
        "mae": round(float(np.mean(np.abs(diff))), 4) if len(diff) else None,
        "bias_llm_minus_human": round(float(np.mean(diff)), 4) if len(diff) else None,
        "pearson": pearson(x, y),
        "spearman": pearson(rankdata(x), rankdata(y)) if len(x) else None,
        "human_accept_rate": round(float(np.mean(human_accept)), 4) if len(x) else None,
        "llm_accept_rate": round(float(np.mean(llm_accept)), 4) if len(y) else None,
        "decision_agreement_at_threshold": round(float(np.mean(human_accept == llm_accept)), 4) if len(x) else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id",
        "title",
        "year",
        "decision",
        "n_reviews",
        "n_numeric_reviews",
        "human_rating_mean",
        "human_rating_std",
        "human_rating_min",
        "human_rating_max",
        "llm_rating",
        "source_decision_from_generated_file",
        "llm_recommendation",
        "llm_confidence",
        "llm_soundness",
        "llm_presentation",
        "llm_contribution",
        "generated_review_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    cache_root = Path("/tmp") / "llmreview_matplotlib_cache"
    (cache_root / "mpl").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([row["human_rating_mean"] for row in rows], dtype=float)
    y = np.asarray([row["llm_rating"] for row in rows], dtype=float)
    accepted = np.asarray([is_human_accepted(row.get("decision")) for row in rows], dtype=bool)
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.suptitle("Human Review Scores vs Final LLM Committee Scores", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.scatter(
        x[~accepted],
        y[~accepted],
        s=18,
        alpha=0.46,
        color="#2f6fb0",
        edgecolors="white",
        linewidths=0.25,
        label="Rejected by humans",
    )
    ax.scatter(
        x[accepted],
        y[accepted],
        s=18,
        alpha=0.52,
        color="#c83f49",
        edgecolors="white",
        linewidths=0.25,
        label="Accepted by humans",
    )
    ax.plot([1, 10], [1, 10], color="#c44e52", linewidth=1.6, linestyle="--", label="LLM = human")
    ax.axhline(summary["accept_threshold"], color="#555555", linewidth=1, alpha=0.55)
    ax.axvline(summary["accept_threshold"], color="#555555", linewidth=1, alpha=0.55)
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 10)
    ax.set_xlabel("Human mean rating")
    ax.set_ylabel("LLM final committee rating")
    ax.set_title("(a) Paper-level score relationship")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    stats = (
        f"n = {summary['n_papers']}\n"
        f"Pearson r = {summary['pearson']:.3f}\n"
        f"Spearman rho = {summary['spearman']:.3f}\n"
        f"MAE = {summary['mae']:.2f}\n"
        f"Bias LLM-human = {summary['bias_llm_minus_human']:+.2f}"
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
    grid = np.linspace(1, 10, 600)
    human_density = kde_density(x, grid)
    llm_density = kde_density(y, grid)
    ax.plot(grid, human_density, color="#4c78a8", linewidth=2.2, label=f"Human mean (mean={summary['human_mean']:.2f}, sd={summary['human_std']:.2f})")
    ax.fill_between(grid, human_density, color="#4c78a8", alpha=0.22)
    ax.plot(grid, llm_density, color="#f58518", linewidth=2.2, label=f"LLM committee (mean={summary['llm_mean']:.2f}, sd={summary['llm_std']:.2f})")
    ax.fill_between(grid, llm_density, color="#f58518", alpha=0.24)
    ax.axvline(summary["accept_threshold"], color="#555555", linewidth=1, linestyle=":", label=f"threshold {summary['accept_threshold']:.1f}")
    ax.set_xlim(1, 10)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Density")
    ax.set_title("(b) Score distributions")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"human_vs_llm_committee_scores.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Human vs LLM Committee Scores",
        "",
        f"- Created: {summary['created_at_utc']}",
        f"- Papers: {summary['n_papers']}",
        f"- Human mean rating: {summary['human_mean']} (std {summary['human_std']})",
        f"- LLM committee rating: {summary['llm_mean']} (std {summary['llm_std']})",
        f"- Pearson: {summary['pearson']}",
        f"- Spearman: {summary['spearman']}",
        f"- MAE: {summary['mae']}",
        f"- Bias, LLM minus human: {summary['bias_llm_minus_human']}",
        f"- Human accept rate at {summary['accept_threshold']}: {summary['human_accept_rate']}",
        f"- LLM accept rate at {summary['accept_threshold']}: {summary['llm_accept_rate']}",
        f"- Threshold decision agreement: {summary['decision_agreement_at_threshold']}",
        "",
        "Outputs:",
        "",
        "- `human_vs_llm_committee_scores.png`",
        "- `human_vs_llm_committee_scores.pdf`",
        "- `human_vs_llm_committee_scores.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = collect_generated_reviews(args.generated_output_roots)
    human_scores = load_human_scores(args.review_db)
    rows = build_joined_rows(generated_paths, human_scores)
    summary = summarize(rows, args.accept_threshold)
    summary["n_generated_review_files_unique_papers"] = len(generated_paths)
    summary["n_human_score_papers"] = len(human_scores)

    write_csv(args.output_dir / "human_vs_llm_committee_scores.csv", rows)
    write_json(args.output_dir / "summary.json", summary)
    write_summary_md(args.output_dir / "summary.md", summary)
    plot(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
