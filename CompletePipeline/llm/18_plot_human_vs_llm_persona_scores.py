#!/usr/bin/env python3
"""Plot individual human review ratings against individual LLM persona ratings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
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
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "Empirics" / "human_vs_llm_persona_scores_20260421"
PERSONA_ORDER = ["empiricist", "theorist", "systems_pragmatist", "novelty_gatekeeper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot individual human scores vs LLM persona scores.")
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


def collect_generated_review_dirs(roots: list[Path]) -> dict[str, Path]:
    """Return one generated paper directory per paper, choosing latest mtime on duplicates."""
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
            if old is None or path.stat().st_mtime > (old / "coarse_review.json").stat().st_mtime:
                paths[paper_id] = path.parent
    return paths


def load_human_review_scores(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                r.paper_id,
                r.reviewer_id,
                r.rating,
                r.confidence,
                r.soundness,
                r.presentation,
                r.contribution,
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

    out: list[dict[str, Any]] = []
    for row in rows:
        rating = parse_numeric_score(row["rating"])
        if rating is None:
            continue
        out.append(
            {
                "paper_id": row["paper_id"],
                "reviewer_id": row["reviewer_id"],
                "title": row["title"],
                "year": row["when_submitted"],
                "decision": row["decision"],
                "human_rating": rating,
                "human_confidence": parse_numeric_score(row["confidence"]),
                "human_soundness": parse_numeric_score(row["soundness"]),
                "human_presentation": parse_numeric_score(row["presentation"]),
                "human_contribution": parse_numeric_score(row["contribution"]),
            }
        )
    return out


def load_persona_scores(generated_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper_id, paper_dir in sorted(generated_dirs.items()):
        coarse_path = paper_dir / "coarse_review.json"
        try:
            coarse = read_json(coarse_path)
        except (OSError, json.JSONDecodeError):
            coarse = {}
        persona_dir = paper_dir / "persona_reviews"
        if not persona_dir.exists():
            continue
        for path in sorted(persona_dir.glob("*.json")):
            try:
                payload = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            rating = parse_numeric_score(payload.get("rating"))
            if rating is None:
                continue
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": payload.get("title") or coarse.get("title") or coarse.get("source_title"),
                    "year": coarse.get("year"),
                    "source_decision_from_generated_file": coarse.get("decision"),
                    "persona": str(payload.get("persona_slug") or path.stem),
                    "llm_persona_rating": rating,
                    "llm_persona_confidence": parse_numeric_score(payload.get("confidence")),
                    "llm_persona_soundness": parse_numeric_score(payload.get("soundness")),
                    "llm_persona_presentation": parse_numeric_score(payload.get("presentation")),
                    "llm_persona_contribution": parse_numeric_score(payload.get("contribution")),
                    "llm_persona_recommendation": payload.get("recommendation"),
                    "persona_review_path": str(path),
                    "coarse_review_path": str(coarse_path),
                }
            )
    return rows


def build_pair_rows(human_rows: list[dict[str, Any]], persona_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    human_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    persona_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in human_rows:
        human_by_paper[str(row["paper_id"])].append(row)
    for row in persona_rows:
        persona_by_paper[str(row["paper_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for paper_id in sorted(set(human_by_paper) & set(persona_by_paper)):
        for human in human_by_paper[paper_id]:
            for persona in persona_by_paper[paper_id]:
                rows.append(
                    {
                        "paper_id": paper_id,
                        "title": human.get("title") or persona.get("title"),
                        "year": human.get("year") or persona.get("year"),
                        "decision": human.get("decision"),
                        "reviewer_id": human.get("reviewer_id"),
                        "persona": persona.get("persona"),
                        "human_rating": human.get("human_rating"),
                        "llm_persona_rating": persona.get("llm_persona_rating"),
                        "human_confidence": human.get("human_confidence"),
                        "llm_persona_confidence": persona.get("llm_persona_confidence"),
                        "human_soundness": human.get("human_soundness"),
                        "llm_persona_soundness": persona.get("llm_persona_soundness"),
                        "human_presentation": human.get("human_presentation"),
                        "llm_persona_presentation": persona.get("llm_persona_presentation"),
                        "human_contribution": human.get("human_contribution"),
                        "llm_persona_contribution": persona.get("llm_persona_contribution"),
                        "llm_persona_recommendation": persona.get("llm_persona_recommendation"),
                        "persona_review_path": persona.get("persona_review_path"),
                    }
                )
    return rows


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


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return round(float(np.std(values, ddof=1)), 4)


def summarize(
    *,
    human_rows: list[dict[str, Any]],
    persona_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    accept_threshold: float,
) -> dict[str, Any]:
    human_scores = [float(row["human_rating"]) for row in human_rows]
    persona_scores = [float(row["llm_persona_rating"]) for row in persona_rows]
    pair_h = np.asarray([row["human_rating"] for row in pair_rows], dtype=float)
    pair_p = np.asarray([row["llm_persona_rating"] for row in pair_rows], dtype=float)
    diff = pair_p - pair_h

    by_persona: dict[str, list[float]] = defaultdict(list)
    by_persona_pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in persona_rows:
        by_persona[str(row["persona"])].append(float(row["llm_persona_rating"]))
    for row in pair_rows:
        by_persona_pair_rows[str(row["persona"])].append(row)

    persona_summary: dict[str, dict[str, Any]] = {}
    for persona in sorted(by_persona):
        rows = by_persona_pair_rows.get(persona, [])
        x = np.asarray([row["human_rating"] for row in rows], dtype=float)
        y = np.asarray([row["llm_persona_rating"] for row in rows], dtype=float)
        persona_summary[persona] = {
            "n_persona_scores": len(by_persona[persona]),
            "n_paired_rows": len(rows),
            "mean": mean(by_persona[persona]),
            "std": std(by_persona[persona]),
            "accept_rate": round(float(np.mean(np.asarray(by_persona[persona]) >= accept_threshold)), 4),
            "paired_mae": round(float(np.mean(np.abs(y - x))), 4) if len(rows) else None,
            "paired_bias_llm_minus_human": round(float(np.mean(y - x)), 4) if len(rows) else None,
            "paired_spearman": pearson(rankdata(x), rankdata(y)) if len(rows) else None,
        }

    return {
        "created_at_utc": now_utc(),
        "accept_threshold": accept_threshold,
        "n_papers": len({row["paper_id"] for row in pair_rows}),
        "n_human_review_scores": len(human_rows),
        "n_llm_persona_scores": len(persona_rows),
        "n_human_x_persona_pairs": len(pair_rows),
        "human_mean": mean(human_scores),
        "human_std": std(human_scores),
        "llm_persona_mean": mean(persona_scores),
        "llm_persona_std": std(persona_scores),
        "paired_mae": round(float(np.mean(np.abs(diff))), 4) if len(diff) else None,
        "paired_bias_llm_minus_human": round(float(np.mean(diff)), 4) if len(diff) else None,
        "paired_pearson": pearson(pair_h, pair_p),
        "paired_spearman": pearson(rankdata(pair_h), rankdata(pair_p)) if len(pair_h) else None,
        "human_accept_rate": round(float(np.mean(np.asarray(human_scores) >= accept_threshold)), 4) if human_scores else None,
        "llm_persona_accept_rate": round(float(np.mean(np.asarray(persona_scores) >= accept_threshold)), 4) if persona_scores else None,
        "persona_counts": dict(Counter(row["persona"] for row in persona_rows)),
        "persona_summary": persona_summary,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ordered_personas(personas: list[str]) -> list[str]:
    known = [persona for persona in PERSONA_ORDER if persona in personas]
    extra = sorted(persona for persona in personas if persona not in PERSONA_ORDER)
    return known + extra


def plot(pair_rows: list[dict[str, Any]], human_rows: list[dict[str, Any]], persona_rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    cache_root = Path("/tmp") / "llmreview_matplotlib_cache"
    (cache_root / "mpl").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_h = np.asarray([row["human_rating"] for row in pair_rows], dtype=float)
    pair_p = np.asarray([row["llm_persona_rating"] for row in pair_rows], dtype=float)
    human_scores = np.asarray([row["human_rating"] for row in human_rows], dtype=float)
    persona_scores = np.asarray([row["llm_persona_rating"] for row in persona_rows], dtype=float)
    personas = ordered_personas(sorted({str(row["persona"]) for row in persona_rows}))
    persona_values = [
        [float(row["llm_persona_rating"]) for row in persona_rows if str(row["persona"]) == persona]
        for persona in personas
    ]

    fig = plt.figure(figsize=(13.5, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85])
    fig.suptitle("Human Review Scores vs LLM Persona Scores", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(grid[0, 0])
    hb = ax.hexbin(pair_h, pair_p, gridsize=32, extent=(1, 10, 1, 10), mincnt=1, cmap="viridis", linewidths=0)
    rng = np.random.default_rng(42)
    sample_n = min(len(pair_rows), 8000)
    if sample_n:
        idx = rng.choice(len(pair_rows), size=sample_n, replace=False)
        ax.scatter(
            pair_h[idx] + rng.normal(0, 0.04, size=sample_n),
            pair_p[idx] + rng.normal(0, 0.04, size=sample_n),
            s=5,
            alpha=0.10,
            color="black",
            edgecolors="none",
        )
    ax.plot([1, 10], [1, 10], color="#c44e52", linewidth=1.4, linestyle="--")
    ax.axhline(summary["accept_threshold"], color="#555555", linewidth=1, alpha=0.55)
    ax.axvline(summary["accept_threshold"], color="#555555", linewidth=1, alpha=0.55)
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 10)
    ax.set_xlabel("Individual human review rating")
    ax.set_ylabel("LLM persona rating")
    ax.set_title("All same-paper Human x Persona pairs")
    ax.grid(alpha=0.18)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Pairs per hex")
    stats = (
        f"papers = {summary['n_papers']}\n"
        f"human reviews = {summary['n_human_review_scores']}\n"
        f"persona scores = {summary['n_llm_persona_scores']}\n"
        f"pairs = {summary['n_human_x_persona_pairs']}\n"
        f"Spearman rho = {summary['paired_spearman']:.3f}\n"
        f"MAE = {summary['paired_mae']:.2f}\n"
        f"Bias LLM-human = {summary['paired_bias_llm_minus_human']:+.2f}"
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

    ax = fig.add_subplot(grid[0, 1])
    bins = np.arange(1, 10.25, 0.25)
    ax.hist(human_scores, bins=bins, density=True, alpha=0.55, color="#4c78a8", label=f"Human reviews (mean={summary['human_mean']:.2f}, sd={summary['human_std']:.2f})")
    ax.hist(persona_scores, bins=bins, density=True, alpha=0.55, color="#f58518", label=f"LLM personas (mean={summary['llm_persona_mean']:.2f}, sd={summary['llm_persona_std']:.2f})")
    ax.axvline(summary["accept_threshold"], color="#555555", linewidth=1, linestyle=":")
    ax.set_xlim(1, 10)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Density")
    ax.set_title("Raw score distributions")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=9)

    ax = fig.add_subplot(grid[1, :])
    box = ax.boxplot(persona_values, tick_labels=personas, vert=True, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ["#4c78a8", "#54a24b", "#b279a2", "#f58518"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(float(np.mean(human_scores)), color="#4c78a8", linestyle="--", linewidth=1.3, label="Human review mean")
    ax.axhline(summary["accept_threshold"], color="#555555", linestyle=":", linewidth=1, label=f"threshold {summary['accept_threshold']:.1f}")
    ax.set_ylim(1, 10)
    ax.set_ylabel("LLM persona rating")
    ax.set_title("LLM score distribution by persona")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"human_vs_llm_persona_scores.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Human vs LLM Persona Scores",
        "",
        f"- Created: {summary['created_at_utc']}",
        f"- Papers: {summary['n_papers']}",
        f"- Human review scores: {summary['n_human_review_scores']}",
        f"- LLM persona scores: {summary['n_llm_persona_scores']}",
        f"- Same-paper Human x Persona pairs: {summary['n_human_x_persona_pairs']}",
        f"- Human mean rating: {summary['human_mean']} (std {summary['human_std']})",
        f"- LLM persona mean rating: {summary['llm_persona_mean']} (std {summary['llm_persona_std']})",
        f"- Paired Spearman: {summary['paired_spearman']}",
        f"- Paired MAE: {summary['paired_mae']}",
        f"- Paired bias, LLM minus human: {summary['paired_bias_llm_minus_human']}",
        f"- Human accept rate at {summary['accept_threshold']}: {summary['human_accept_rate']}",
        f"- LLM persona accept rate at {summary['accept_threshold']}: {summary['llm_persona_accept_rate']}",
        "",
        "## Persona Summary",
        "",
        "| Persona | Scores | Mean | Std | Accept rate | Paired MAE | Paired bias | Paired rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for persona in ordered_personas(list(summary["persona_summary"])):
        row = summary["persona_summary"][persona]
        lines.append(
            f"| {persona} | {row['n_persona_scores']} | {row['mean']} | {row['std']} | "
            f"{row['accept_rate']} | {row['paired_mae']} | {row['paired_bias_llm_minus_human']} | {row['paired_spearman']} |"
        )
    lines.extend(
        [
            "",
            "Outputs:",
            "",
            "- `human_vs_llm_persona_scores.png`",
            "- `human_vs_llm_persona_scores.pdf`",
            "- `human_x_persona_score_pairs.csv`",
            "- `human_review_scores_long.csv`",
            "- `llm_persona_scores_long.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_dirs = collect_generated_review_dirs(args.generated_output_roots)
    human_rows_all = load_human_review_scores(args.review_db)
    persona_rows = load_persona_scores(generated_dirs)
    paper_ids = {row["paper_id"] for row in persona_rows}
    human_rows = [row for row in human_rows_all if row["paper_id"] in paper_ids]
    pair_rows = build_pair_rows(human_rows, persona_rows)
    summary = summarize(
        human_rows=human_rows,
        persona_rows=persona_rows,
        pair_rows=pair_rows,
        accept_threshold=args.accept_threshold,
    )
    summary["n_generated_review_files_unique_papers"] = len(generated_dirs)
    summary["n_human_score_papers_all_db"] = len({row["paper_id"] for row in human_rows_all})

    write_csv(args.output_dir / "human_review_scores_long.csv", human_rows)
    write_csv(args.output_dir / "llm_persona_scores_long.csv", persona_rows)
    write_csv(args.output_dir / "human_x_persona_score_pairs.csv", pair_rows)
    write_json(args.output_dir / "summary.json", summary)
    write_summary_md(args.output_dir / "summary.md", summary)
    plot(pair_rows, human_rows, persona_rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
