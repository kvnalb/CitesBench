#!/usr/bin/env python3
"""
Exploratory local RD-style analysis for ICLR review data across all years.

The running variable is paper-level mean reviewer rating. Because the local DB
does not expose Area Chair IDs or an official score cutoff, this script treats
the cutoff search as exploratory. It scans candidate score cutoffs within each
year, selects the largest local acceptance jump, then recenters scores by those
year-specific cutoffs for pooled local-linear models.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

def find_repo_root(anchor: str | Path) -> Path:
    start = Path(anchor).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "Code").exists() and (candidate / "Report").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


ROOT = find_repo_root(__file__)
SCRIPT_DIR = Path(__file__).resolve().parent

MPLCONFIGDIR = ROOT / "OutputNew" / "Design" / "iclr_local_rdd" / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DB_PATH = ROOT / "data" / "LLM-Reviewer-03042026" / "data" / "gen_review.db"
DEFAULT_OUTPUT_DIR = ROOT / "OutputNew" / "Design" / "iclr_local_rdd"


@dataclass
class LocalLinearResult:
    jump: float
    jump_se: float
    jump_pvalue: float
    raw_jump: float
    n_obs: int
    n_left: int
    n_right: int
    left_rate: float
    right_rate: float


def parse_score(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text or text in {"N/A", "Not applicable"}:
        return None
    match = re.match(r"^(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def classify_decision(decision: object) -> tuple[str, float]:
    if decision is None:
        return "other", np.nan
    label = str(decision).strip()
    lower = label.lower()
    if not label:
        return "other", np.nan
    if "withdrawn" in lower:
        return "withdrawn", np.nan
    if "workshop" in lower:
        return "workshop", np.nan
    if "accept" in lower:
        return "accept", 1.0
    if "reject" in lower:
        return "reject", 0.0
    return "other", np.nan


def load_review_rows(db_path: Path) -> pd.DataFrame:
    query = """
        SELECT
            s.id AS paper_id,
            s.title,
            s.when_submitted AS year,
            s.primary_area,
            s.decision,
            s.source_id,
            r.reviewer_id,
            r.rating,
            r.confidence,
            r.binocular_score
        FROM REVIEW r
        JOIN SUBMISSION s ON r.paper_id = s.id
        ORDER BY s.when_submitted, s.id, r.reviewer_id
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn)

    df["rating_num"] = df["rating"].map(parse_score)
    df["confidence_num"] = df["confidence"].map(parse_score)
    df["binocular_num"] = pd.to_numeric(df["binocular_score"], errors="coerce")
    df["primary_area"] = (
        df["primary_area"].fillna("").astype(str).str.strip().replace("", pd.NA)
    )
    return df


def build_paper_level(review_df: pd.DataFrame) -> pd.DataFrame:
    rated = review_df[review_df["rating_num"].notna()].copy()
    grouped = (
        rated.groupby(
            ["paper_id", "title", "year", "primary_area", "decision", "source_id"],
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_rating=("rating_num", "mean"),
            median_rating=("rating_num", "median"),
            std_rating=("rating_num", "std"),
            min_rating=("rating_num", "min"),
            max_rating=("rating_num", "max"),
            n_reviews=("rating_num", "size"),
            n_unique_reviewers=("reviewer_id", "nunique"),
            mean_confidence=("confidence_num", "mean"),
            mean_binocular=("binocular_num", "mean"),
        )
        .sort_values(["year", "paper_id"])
        .reset_index(drop=True)
    )

    grouped["std_rating"] = grouped["std_rating"].fillna(0.0)
    decision_info = grouped["decision"].map(classify_decision)
    grouped["decision_group"] = [x[0] for x in decision_info]
    grouped["accepted"] = [x[1] for x in decision_info]
    grouped["has_primary_area"] = grouped["primary_area"].notna()
    grouped["fe_group"] = np.where(
        grouped["primary_area"].notna(),
        grouped["year"].astype(str) + "::" + grouped["primary_area"].astype(str),
        grouped["year"].astype(str) + "::ALL",
    )
    return grouped


def analysis_sample(paper_df: pd.DataFrame) -> pd.DataFrame:
    keep = paper_df["decision_group"].isin({"accept", "reject"}) & paper_df["accepted"].notna()
    return paper_df.loc[keep].copy().reset_index(drop=True)


def yearly_summary(paper_df: pd.DataFrame) -> pd.DataFrame:
    return (
        paper_df.groupby("year", as_index=False)
        .agg(
            n_papers=("paper_id", "size"),
            accept_rate=("accepted", "mean"),
            mean_rating=("mean_rating", "mean"),
            median_rating=("mean_rating", "median"),
            min_rating=("mean_rating", "min"),
            max_rating=("mean_rating", "max"),
            with_primary_area=("has_primary_area", "sum"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )


def fit_local_linear(
    df: pd.DataFrame,
    running_col: str,
    bandwidth: float,
    fe_col: str | None = None,
    min_side: int = 20,
) -> LocalLinearResult | None:
    subset = df.loc[df[running_col].abs() <= bandwidth].copy()
    if subset.empty:
        return None

    subset["running"] = subset[running_col].astype(float)
    subset["above"] = (subset["running"] >= 0).astype(float)
    subset["interaction"] = subset["above"] * subset["running"]
    subset["weights"] = 1.0 - (subset["running"].abs() / bandwidth)

    left = subset.loc[subset["running"] < 0]
    right = subset.loc[subset["running"] >= 0]
    if len(left) < min_side or len(right) < min_side:
        return None

    design = subset[["above", "running", "interaction"]].copy()
    if fe_col is not None:
        dummies = pd.get_dummies(
            subset[fe_col].astype(str),
            prefix="fe",
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            design = pd.concat([design, dummies], axis=1)

    design = sm.add_constant(design, has_constant="add").astype(float)
    outcome = subset["accepted"].astype(float)
    model = sm.WLS(outcome, design, weights=subset["weights"].astype(float))
    result = model.fit(cov_type="HC1")

    names = result.model.exog_names
    params = pd.Series(result.params, index=names)
    bse = pd.Series(result.bse, index=names)
    pvalues = pd.Series(result.pvalues, index=names)

    return LocalLinearResult(
        jump=float(params["above"]),
        jump_se=float(bse["above"]),
        jump_pvalue=float(pvalues["above"]),
        raw_jump=float(right["accepted"].mean() - left["accepted"].mean()),
        n_obs=int(len(subset)),
        n_left=int(len(left)),
        n_right=int(len(right)),
        left_rate=float(left["accepted"].mean()),
        right_rate=float(right["accepted"].mean()),
    )


def fit_local_linear_model(
    subset: pd.DataFrame,
    fe_col: str | None = None,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    design = subset[["above", "running", "interaction"]].copy()
    if fe_col is not None:
        dummies = pd.get_dummies(
            subset[fe_col].astype(str),
            prefix="fe",
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            design = pd.concat([design, dummies], axis=1)

    design = sm.add_constant(design, has_constant="add").astype(float)
    outcome = subset["accepted"].astype(float)
    model = sm.WLS(outcome, design, weights=subset["weights"].astype(float))
    result = model.fit(cov_type="HC1")
    return result, design


def prepare_local_subset(
    df: pd.DataFrame,
    running_col: str,
    bandwidth: float,
    min_side: int = 20,
) -> pd.DataFrame | None:
    subset = df.loc[df[running_col].abs() <= bandwidth].copy()
    if subset.empty:
        return None

    subset["running"] = subset[running_col].astype(float)
    subset["above"] = (subset["running"] >= 0).astype(float)
    subset["interaction"] = subset["above"] * subset["running"]
    subset["weights"] = 1.0 - (subset["running"].abs() / bandwidth)

    left = subset.loc[subset["running"] < 0]
    right = subset.loc[subset["running"] >= 0]
    if len(left) < min_side or len(right) < min_side:
        return None
    return subset


def iter_candidate_cutoffs(
    values: Iterable[float],
    candidate_min: float,
    candidate_max: float,
) -> list[float]:
    return sorted(
        {
            round(float(value), 3)
            for value in values
            if candidate_min <= float(value) <= candidate_max
        }
    )


def iter_candidate_bandwidths(
    values: Iterable[float],
    cutoff: float,
    min_bandwidth: float,
    max_bandwidth: float,
) -> list[float]:
    candidates = sorted(
        {
            round(abs(float(value) - cutoff), 3)
            for value in values
            if min_bandwidth <= abs(float(value) - cutoff) <= max_bandwidth
        }
    )
    return [bw for bw in candidates if bw > 0]


def scan_yearly_cutoffs(
    paper_df: pd.DataFrame,
    bandwidth: float,
    candidate_min: float,
    candidate_max: float,
    min_side: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scan_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for year, year_df in paper_df.groupby("year"):
        year_df = year_df.copy()
        candidates = iter_candidate_cutoffs(
            year_df["mean_rating"].tolist(),
            candidate_min=candidate_min,
            candidate_max=candidate_max,
        )

        for cutoff in candidates:
            year_df["score_centered"] = year_df["mean_rating"] - cutoff
            result = fit_local_linear(
                year_df,
                running_col="score_centered",
                bandwidth=bandwidth,
                fe_col=None,
                min_side=min_side,
            )
            if result is None:
                continue
            scan_rows.append(
                {
                    "year": int(year),
                    "cutoff": cutoff,
                    "local_linear_jump": result.jump,
                    "local_linear_jump_se": result.jump_se,
                    "local_linear_jump_pvalue": result.jump_pvalue,
                    "raw_jump": result.raw_jump,
                    "n_obs": result.n_obs,
                    "n_left": result.n_left,
                    "n_right": result.n_right,
                    "left_rate": result.left_rate,
                    "right_rate": result.right_rate,
                    "abs_local_linear_jump": abs(result.jump),
                    "abs_raw_jump": abs(result.raw_jump),
                }
            )

    scan_df = pd.DataFrame(scan_rows)
    if scan_df.empty:
        return scan_df, scan_df

    best_df = (
        scan_df.sort_values(
            ["year", "abs_raw_jump", "local_linear_jump_pvalue", "n_obs"],
            ascending=[True, False, True, False],
        )
        .groupby("year", as_index=False)
        .head(1)
        .sort_values("year")
        .reset_index(drop=True)
    )
    return scan_df, best_df


def assign_stratified_folds(subset: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    out = subset.sort_values(["above", "running", "paper_id"]).copy()
    out["cv_fold"] = -1
    for above_value in [0.0, 1.0]:
        idx = out.index[out["above"] == above_value].tolist()
        for order, ix in enumerate(idx):
            out.at[ix, "cv_fold"] = order % n_folds
    return out


def cross_validated_brier_for_bandwidth(
    year_df: pd.DataFrame,
    cutoff: float,
    bandwidth: float,
    min_side: int,
    n_folds: int,
) -> dict[str, float] | None:
    working = year_df.copy()
    working["score_centered"] = working["mean_rating"] - cutoff
    subset = prepare_local_subset(
        working,
        running_col="score_centered",
        bandwidth=bandwidth,
        min_side=min_side,
    )
    if subset is None:
        return None

    left_n = int((subset["above"] == 0.0).sum())
    right_n = int((subset["above"] == 1.0).sum())
    n_folds = max(2, min(n_folds, left_n, right_n))
    subset = assign_stratified_folds(subset, n_folds=n_folds)

    fold_losses: list[float] = []
    for fold in range(n_folds):
        train = subset.loc[subset["cv_fold"] != fold].copy()
        test = subset.loc[subset["cv_fold"] == fold].copy()
        if train.empty or test.empty:
            continue

        result, design_train = fit_local_linear_model(train, fe_col=None)
        test_design = test[["above", "running", "interaction"]].copy()
        test_design = sm.add_constant(test_design, has_constant="add").astype(float)
        test_design = test_design.reindex(columns=design_train.columns, fill_value=0.0)

        preds = np.clip(result.predict(test_design), 0.0, 1.0)
        loss = ((test["accepted"].astype(float) - preds) ** 2) * test["weights"].astype(float)
        fold_losses.append(float(loss.sum() / test["weights"].sum()))

    if not fold_losses:
        return None

    local_result = fit_local_linear(
        working,
        running_col="score_centered",
        bandwidth=bandwidth,
        fe_col=None,
        min_side=min_side,
    )
    if local_result is None:
        return None

    return {
        "cv_brier_mean": float(np.mean(fold_losses)),
        "cv_brier_se": float(np.std(fold_losses, ddof=1) / np.sqrt(len(fold_losses)))
        if len(fold_losses) > 1
        else 0.0,
        "n_folds": int(len(fold_losses)),
        "n_obs": int(local_result.n_obs),
        "n_left": int(local_result.n_left),
        "n_right": int(local_result.n_right),
        "raw_jump": float(local_result.raw_jump),
        "local_linear_jump": float(local_result.jump),
        "local_linear_jump_se": float(local_result.jump_se),
        "local_linear_jump_pvalue": float(local_result.jump_pvalue),
        "left_rate": float(local_result.left_rate),
        "right_rate": float(local_result.right_rate),
    }


def select_yearly_bandwidths(
    paper_df: pd.DataFrame,
    best_cutoffs: pd.DataFrame,
    min_bandwidth: float,
    max_bandwidth: float,
    min_side: int,
    n_folds: int,
    cv_tolerance_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scan_rows: list[dict[str, object]] = []

    for year, year_df in paper_df.groupby("year"):
        cutoff_row = best_cutoffs.loc[best_cutoffs["year"] == year]
        if cutoff_row.empty:
            continue
        cutoff = float(cutoff_row["cutoff"].iloc[0])
        candidates = iter_candidate_bandwidths(
            year_df["mean_rating"].tolist(),
            cutoff=cutoff,
            min_bandwidth=min_bandwidth,
            max_bandwidth=max_bandwidth,
        )
        for bandwidth in candidates:
            metrics = cross_validated_brier_for_bandwidth(
                year_df=year_df,
                cutoff=cutoff,
                bandwidth=bandwidth,
                min_side=min_side,
                n_folds=n_folds,
            )
            if metrics is None:
                continue
            scan_rows.append(
                {
                    "year": int(year),
                    "cutoff": cutoff,
                    "bandwidth": bandwidth,
                    **metrics,
                }
            )

    scan_df = pd.DataFrame(scan_rows)
    if scan_df.empty:
        return scan_df, scan_df

    selected_rows: list[dict[str, object]] = []
    for year, year_scan in scan_df.groupby("year"):
        year_scan = year_scan.sort_values(["cv_brier_mean", "bandwidth"]).reset_index(drop=True)
        best_row = year_scan.iloc[0]
        threshold = float(best_row["cv_brier_mean"] * (1.0 + cv_tolerance_frac))
        eligible = year_scan.loc[year_scan["cv_brier_mean"] <= threshold].copy()
        if eligible.empty:
            eligible = year_scan.iloc[[0]].copy()
        chosen = eligible.sort_values(["bandwidth", "cv_brier_mean"]).iloc[0].to_dict()
        chosen["selection_threshold"] = threshold
        chosen["cv_tolerance_frac"] = cv_tolerance_frac
        selected_rows.append(chosen)

    selected_df = pd.DataFrame(selected_rows).sort_values("year").reset_index(drop=True)
    return scan_df, selected_df


def build_year_specific_rdd_sample(
    paper_df: pd.DataFrame,
    selected_bandwidths: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = paper_df.merge(
        selected_bandwidths[["year", "cutoff", "bandwidth"]],
        on="year",
        how="inner",
        validate="many_to_one",
    ).copy()
    merged["score_centered"] = merged["mean_rating"] - merged["cutoff"]
    merged["in_year_specific_rdd_sample"] = (
        merged["score_centered"].abs() <= merged["bandwidth"]
    )
    sample = merged.loc[merged["in_year_specific_rdd_sample"]].copy()
    sample["side"] = np.where(sample["score_centered"] < 0, "left", "right")

    summary = (
        sample.groupby("year", as_index=False)
        .agg(
            cutoff=("cutoff", "first"),
            selected_bandwidth=("bandwidth", "first"),
            n_obs=("paper_id", "size"),
            n_left=("side", lambda s: int((s == "left").sum())),
            n_right=("side", lambda s: int((s == "right").sum())),
            left_rate=("accepted", lambda s: float(sample.loc[s.index][sample.loc[s.index, "side"] == "left"]["accepted"].mean())),
            right_rate=("accepted", lambda s: float(sample.loc[s.index][sample.loc[s.index, "side"] == "right"]["accepted"].mean())),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    summary["raw_jump"] = summary["right_rate"] - summary["left_rate"]
    return merged, summary


def fit_pooled_models(
    paper_df: pd.DataFrame,
    best_cutoffs: pd.DataFrame,
    bandwidths: list[float],
    min_side: int,
) -> pd.DataFrame:
    merged = paper_df.merge(
        best_cutoffs[["year", "cutoff"]],
        on="year",
        how="inner",
        validate="many_to_one",
    ).copy()
    merged["score_centered"] = merged["mean_rating"] - merged["cutoff"]

    rows: list[dict[str, object]] = []
    specs = [
        ("none", None),
        ("year", "year"),
        ("year_area_or_year", "fe_group"),
    ]

    for bandwidth in bandwidths:
        for spec_name, fe_col in specs:
            result = fit_local_linear(
                merged,
                running_col="score_centered",
                bandwidth=bandwidth,
                fe_col=fe_col,
                min_side=min_side,
            )
            if result is None:
                continue
            rows.append(
                {
                    "bandwidth": bandwidth,
                    "spec": spec_name,
                    "local_linear_jump": result.jump,
                    "local_linear_jump_se": result.jump_se,
                    "local_linear_jump_pvalue": result.jump_pvalue,
                    "raw_jump": result.raw_jump,
                    "n_obs": result.n_obs,
                    "n_left": result.n_left,
                    "n_right": result.n_right,
                    "left_rate": result.left_rate,
                    "right_rate": result.right_rate,
                }
            )

    return pd.DataFrame(rows), merged


def plot_score_distribution_by_year(
    paper_df: pd.DataFrame,
    best_cutoffs: pd.DataFrame,
    selected_bandwidths: pd.DataFrame,
    output_path: Path,
) -> None:
    years = sorted(paper_df["year"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    bins = np.arange(1.0, 9.25, 0.25)
    cutoff_map = dict(zip(best_cutoffs["year"], best_cutoffs["cutoff"]))
    bandwidth_map = dict(zip(selected_bandwidths["year"], selected_bandwidths["bandwidth"]))

    for ax, year in zip(axes.flat, years):
        subset = paper_df.loc[paper_df["year"] == year]
        accepted = subset.loc[subset["accepted"] == 1.0, "mean_rating"]
        rejected = subset.loc[subset["accepted"] == 0.0, "mean_rating"]
        cutoff = cutoff_map.get(year)
        bandwidth = bandwidth_map.get(year)

        if cutoff is not None and bandwidth is not None:
            ax.axvspan(
                cutoff - bandwidth,
                cutoff + bandwidth,
                color="#7F7F7F",
                alpha=0.12,
                zorder=0,
            )
            ax.axvline(cutoff, color="#C44E52", linestyle="--", linewidth=1.2, zorder=1)

        ax.hist(rejected, bins=bins, alpha=0.6, label="Reject", color="#D95F02")
        ax.hist(accepted, bins=bins, alpha=0.6, label="Accept", color="#1B9E77")
        ax.set_title(f"{year}")
        if cutoff is not None and bandwidth is not None:
            ax.text(
                0.02,
                0.96,
                f"c={cutoff:.2f}\nh={bandwidth:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        ax.grid(alpha=0.2)

    for ax in axes.flat[len(years) :]:
        ax.axis("off")

    axes[0, 0].legend(frameon=False)
    fig.suptitle("Paper-Level Mean Rating Distribution by Year")
    fig.supxlabel("Mean reviewer rating")
    fig.supylabel("Paper count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_acceptance_vs_score_by_year(
    paper_df: pd.DataFrame,
    best_cutoffs: pd.DataFrame,
    selected_bandwidths: pd.DataFrame,
    output_path: Path,
) -> None:
    years = sorted(paper_df["year"].unique())
    cutoff_map = dict(zip(best_cutoffs["year"], best_cutoffs["cutoff"]))
    bandwidth_map = dict(zip(selected_bandwidths["year"], selected_bandwidths["bandwidth"]))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    for ax, year in zip(axes.flat, years):
        subset = paper_df.loc[paper_df["year"] == year]
        grouped = (
            subset.groupby("mean_rating", as_index=False)
            .agg(accept_rate=("accepted", "mean"), n_papers=("paper_id", "size"))
            .sort_values("mean_rating")
        )
        cutoff = cutoff_map.get(year)
        bandwidth = bandwidth_map.get(year)

        size_scale = 20 + 12 * np.sqrt(grouped["n_papers"].astype(float))
        if cutoff is not None and bandwidth is not None:
            ax.axvspan(
                cutoff - bandwidth,
                cutoff + bandwidth,
                color="#7F7F7F",
                alpha=0.12,
                zorder=0,
            )
        ax.plot(grouped["mean_rating"], grouped["accept_rate"], color="#4C72B0", alpha=0.6)
        ax.scatter(
            grouped["mean_rating"],
            grouped["accept_rate"],
            s=size_scale,
            color="#4C72B0",
            alpha=0.85,
        )
        if cutoff is not None:
            ax.axvline(cutoff, color="#C44E52", linestyle="--", linewidth=1.2)
        ax.set_title(f"{year}")
        if cutoff is not None and bandwidth is not None:
            ax.text(
                0.02,
                0.96,
                f"c={cutoff:.2f}\nh={bandwidth:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)

    for ax in axes.flat[len(years) :]:
        ax.axis("off")

    fig.suptitle("Acceptance Probability by Paper-Level Mean Rating")
    fig.supxlabel("Mean reviewer rating")
    fig.supylabel("Acceptance rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_centered_acceptance(
    merged_df: pd.DataFrame,
    selected_bandwidths: pd.DataFrame,
    output_path: Path,
) -> None:
    full = merged_df.copy()
    window = full.loc[full["in_year_specific_rdd_sample"]].copy()
    grouped_full = (
        full.groupby("score_centered", as_index=False)
        .agg(accept_rate=("accepted", "mean"), n_papers=("paper_id", "size"))
        .sort_values("score_centered")
    )
    grouped_window = (
        window.groupby("score_centered", as_index=False)
        .agg(accept_rate=("accepted", "mean"), n_papers=("paper_id", "size"))
        .sort_values("score_centered")
    )

    score_min = float(np.floor(full["score_centered"].min() * 4.0) / 4.0)
    score_max = float(np.ceil(full["score_centered"].max() * 4.0) / 4.0)
    bins = np.arange(score_min, score_max + 0.25, 0.25)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )

    min_bandwidth = float(selected_bandwidths["bandwidth"].min())
    median_bandwidth = float(selected_bandwidths["bandwidth"].median())
    max_bandwidth = float(selected_bandwidths["bandwidth"].max())

    for h in sorted(selected_bandwidths["bandwidth"].astype(float).tolist()):
        ax_top.axvline(-h, color="#7F7F7F", linestyle=":", linewidth=0.8, alpha=0.18, zorder=0)
        ax_top.axvline(h, color="#7F7F7F", linestyle=":", linewidth=0.8, alpha=0.18, zorder=0)
        ax_bottom.axvline(-h, color="#7F7F7F", linestyle=":", linewidth=0.8, alpha=0.18, zorder=0)
        ax_bottom.axvline(h, color="#7F7F7F", linestyle=":", linewidth=0.8, alpha=0.18, zorder=0)

    ax_top.axvspan(-min_bandwidth, min_bandwidth, color="#7F7F7F", alpha=0.10, zorder=0)
    ax_bottom.axvspan(-min_bandwidth, min_bandwidth, color="#7F7F7F", alpha=0.10, zorder=0)

    rejected_full = full.loc[full["accepted"] == 0.0, "score_centered"]
    accepted_full = full.loc[full["accepted"] == 1.0, "score_centered"]
    ax_top.hist(rejected_full, bins=bins, alpha=0.55, color="#D95F02", label="Reject")
    ax_top.hist(accepted_full, bins=bins, alpha=0.55, color="#1B9E77", label="Accept")
    ax_top.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.2, label="Estimated cutoff")
    ax_top.set_ylabel("Paper count")
    ax_top.set_title(
        "Pooled Centered Score Distribution Across All Years\n"
        f"Selected h range: {min_bandwidth:.2f} to {max_bandwidth:.2f}, median {median_bandwidth:.2f}"
    )
    ax_top.text(
        0.02,
        0.95,
        "Dotted lines: year-specific bandwidth edges\n"
        "Shaded band: tightest selected bandwidth",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax_top.grid(alpha=0.2)
    ax_top.legend(frameon=False, ncol=3)

    size_scale_full = 12 + 8 * np.sqrt(grouped_full["n_papers"].astype(float))
    size_scale_window = 16 + 10 * np.sqrt(grouped_window["n_papers"].astype(float))
    ax_bottom.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.2)
    ax_bottom.plot(
        grouped_full["score_centered"],
        grouped_full["accept_rate"],
        color="#9ECAE1",
        alpha=0.9,
        linewidth=1.4,
        label="Full support",
    )
    ax_bottom.scatter(
        grouped_full["score_centered"],
        grouped_full["accept_rate"],
        s=size_scale_full,
        color="#9ECAE1",
        alpha=0.55,
    )
    ax_bottom.plot(
        grouped_window["score_centered"],
        grouped_window["accept_rate"],
        color="#2171B5",
        alpha=0.95,
        linewidth=1.6,
        label="RDD sample",
    )
    ax_bottom.scatter(
        grouped_window["score_centered"],
        grouped_window["accept_rate"],
        s=size_scale_window,
        color="#2171B5",
        alpha=0.8,
    )
    ax_bottom.set_xlabel("Mean rating relative to estimated year-specific cutoff")
    ax_bottom.set_ylabel("Acceptance rate")
    ax_bottom.set_ylim(-0.05, 1.05)
    ax_bottom.grid(alpha=0.2)
    ax_bottom.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_console_summary(
    summary_df: pd.DataFrame,
    best_cutoffs: pd.DataFrame,
    pooled_models: pd.DataFrame,
) -> None:
    print("\nYearly summary")
    print(summary_df.to_string(index=False))

    print("\nBest exploratory cutoff by year")
    print(
        best_cutoffs[
            [
                "year",
                "cutoff",
                "raw_jump",
                "local_linear_jump",
                "local_linear_jump_se",
                "local_linear_jump_pvalue",
                "n_obs",
                "left_rate",
                "right_rate",
            ]
        ].to_string(index=False)
    )

    print("\nPooled local-linear models")
    print(pooled_models.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--secondary-bandwidth", type=float, default=0.75)
    parser.add_argument("--bandwidth-min", type=float, default=0.35)
    parser.add_argument("--bandwidth-max", type=float, default=1.5)
    parser.add_argument("--bandwidth-cv-folds", type=int, default=5)
    parser.add_argument("--bandwidth-cv-tolerance-frac", type=float, default=0.10)
    parser.add_argument("--candidate-min", type=float, default=4.5)
    parser.add_argument("--candidate-max", type=float, default=6.5)
    parser.add_argument("--min-side", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    review_df = load_review_rows(args.db_path)
    paper_df = build_paper_level(review_df)
    analytic_df = analysis_sample(paper_df)

    summary_df = yearly_summary(analytic_df)
    scan_df, best_df = scan_yearly_cutoffs(
        analytic_df,
        bandwidth=args.bandwidth,
        candidate_min=args.candidate_min,
        candidate_max=args.candidate_max,
        min_side=args.min_side,
    )
    bandwidth_scan_df, selected_bandwidths_df = select_yearly_bandwidths(
        analytic_df,
        best_cutoffs=best_df,
        min_bandwidth=args.bandwidth_min,
        max_bandwidth=args.bandwidth_max,
        min_side=args.min_side,
        n_folds=args.bandwidth_cv_folds,
        cv_tolerance_frac=args.bandwidth_cv_tolerance_frac,
    )
    paper_with_windows_df, rdd_sample_summary_df = build_year_specific_rdd_sample(
        analytic_df,
        selected_bandwidths=selected_bandwidths_df,
    )
    pooled_models, merged_df = fit_pooled_models(
        analytic_df,
        best_cutoffs=best_df,
        bandwidths=[args.bandwidth, args.secondary_bandwidth],
        min_side=args.min_side,
    )

    paper_df.to_csv(args.output_dir / "paper_level_all_years.csv", index=False)
    summary_df.to_csv(args.output_dir / "yearly_summary.csv", index=False)
    scan_df.to_csv(args.output_dir / "yearly_cutoff_scan.csv", index=False)
    best_df.to_csv(args.output_dir / "yearly_best_cutoffs.csv", index=False)
    bandwidth_scan_df.to_csv(args.output_dir / "yearly_bandwidth_scan.csv", index=False)
    selected_bandwidths_df.to_csv(args.output_dir / "yearly_selected_bandwidths.csv", index=False)
    paper_with_windows_df.to_csv(args.output_dir / "paper_level_with_year_specific_windows.csv", index=False)
    paper_with_windows_df.loc[
        paper_with_windows_df["in_year_specific_rdd_sample"]
    ].to_csv(args.output_dir / "rdd_sample_year_specific_bandwidth.csv", index=False)
    rdd_sample_summary_df.to_csv(args.output_dir / "rdd_sample_summary_by_year.csv", index=False)
    pooled_models.to_csv(args.output_dir / "pooled_local_models.csv", index=False)

    plot_score_distribution_by_year(
        analytic_df,
        best_cutoffs=best_df,
        selected_bandwidths=selected_bandwidths_df,
        output_path=args.output_dir / "fig_score_distribution_by_year.png",
    )
    plot_acceptance_vs_score_by_year(
        analytic_df,
        best_cutoffs=best_df,
        selected_bandwidths=selected_bandwidths_df,
        output_path=args.output_dir / "fig_acceptance_vs_score_by_year.png",
    )
    plot_centered_acceptance(
        paper_with_windows_df,
        selected_bandwidths=selected_bandwidths_df,
        output_path=args.output_dir / "fig_centered_acceptance_pooled.png",
    )

    print_console_summary(summary_df, best_df, pooled_models)
    print("\nSelected year-specific bandwidths")
    print(
        selected_bandwidths_df[
            [
                "year",
                "cutoff",
                "bandwidth",
                "cv_brier_mean",
                "cv_brier_se",
                "raw_jump",
                "n_obs",
                "n_left",
                "n_right",
            ]
        ].to_string(index=False)
    )

    print("\nRDD sample summary by year")
    print(rdd_sample_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
