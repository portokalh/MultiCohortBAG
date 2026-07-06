#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3 builder with explicit QA and safer output handling.

Builds:
  - Figure 3: OOF raw predicted brain age vs chronological age for imaging_only.
  - Supplementary Figure 3 aggregate: BAG/cBAG vs age, pooling all feature sets.
  - Supplementary Figure 3A-E: feature-set-specific BAG/cBAG vs age diagnostics.

Key safety changes vs the recovered version:
  - Does a full input QA over all cohort x feature-set prediction files before plotting.
  - Writes QA tables before figures.
  - Can run in --qa-only mode.
  - Uses argparse paths and run tags.
  - Optional --strict mode stops if expected prediction files/columns are missing.
  - Optional --overwrite controls whether an existing output directory can be reused.
  - Saves a complete manifest and output inventory.

Important:
  This script consumes final *_cv_oof_predictions.csv files. It does not rebuild graph
  matrices or retrain models. The word "graph" here refers to the plotted figure panels.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# Defaults
# =============================================================================
DEFAULT_RESULTS_ROOT = Path("/mnt/newStor/paros/paros_WORK/ines/results")
DEFAULT_BASE_VALIDATION_NAME = "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"

COHORT_ORDER = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

PREFIX_MAP = {
    "ADNI": "adni",
    "ADRC": "adrc",
    "HABS": "habs",
    "AD_DECODE": "addecode",
}

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

FEATURE_PANEL_MAP = {
    "A": "imaging_only",
    "B": "imaging_demographics",
    "C": "imaging_biomarkers",
    "D": "full",
    "E": "full_no_cardiovascular",
}

MAIN_FIGURE_FEATURE_SET = "imaging_only"

REQUIRED_COLUMNS = ["age_true", "pred_raw"]
OPTIONAL_BAG_COLUMNS = ["BAG_raw", "cBAG_foldwise", "cBAG", "cBAG_oof_global"]
BAG_COLUMNS = ["BAG_raw", "cBAG_foldwise", "cBAG_oof_global"]

BAG_LABELS = {
    "BAG_raw": "Raw BAG",
    "cBAG_foldwise": "Fold-wise cBAG",
    "cBAG_oof_global": "OOF-global cBAG",
}

PANEL_LABELS = {
    "ADNI": "A. ADNI",
    "ADRC": "B. ADRC",
    "HABS": "C. HABS",
    "AD_DECODE": "D. AD-DECODE",
}

SUBJECT_ID_CANDIDATES = [
    "subject_id", "subject", "participant_id", "RID", "rid", "id", "ID", "subj", "scan_id"
]


# =============================================================================
# Utilities
# =============================================================================
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(msg: str) -> None:
    print(msg, flush=True)


def get_oof_path(results_root: Path, cohort: str, feature_set: str) -> Path:
    return (
        results_root
        / RESULTS_DIR_MAP[cohort]
        / f"ablation_{feature_set}"
        / f"{PREFIX_MAP[cohort]}_{feature_set}_cv_oof_predictions.csv"
    )


def clean_numeric(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def find_subject_col(columns: List[str]) -> Optional[str]:
    for c in SUBJECT_ID_CANDIDATES:
        if c in columns:
            return c
    lower = {str(c).lower(): c for c in columns}
    for c in SUBJECT_ID_CANDIDATES:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def safe_pearsonr(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    if len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])


def format_p(p) -> str:
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def prediction_stats(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 3:
        return {"n": int(len(y_true)), "r": np.nan, "p": np.nan, "r2": np.nan,
                "mae": np.nan, "rmse": np.nan, "slope": np.nan, "intercept": np.nan}
    r, p = safe_pearsonr(y_true, y_pred)
    lr = linregress(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "r": float(r),
        "p": float(p),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
    }


def bag_stats(age, bag) -> Dict[str, float]:
    age = np.asarray(age, dtype=float)
    bag = np.asarray(bag, dtype=float)
    mask = np.isfinite(age) & np.isfinite(bag)
    age = age[mask]
    bag = bag[mask]
    if len(age) < 3:
        return {"n": int(len(age)), "r": np.nan, "p": np.nan, "r2": np.nan,
                "slope": np.nan, "intercept": np.nan}
    r, p = safe_pearsonr(age, bag)
    lr = linregress(age, bag)
    return {
        "n": int(len(age)),
        "r": float(r),
        "p": float(p),
        "r2": float(r ** 2) if np.isfinite(r) else np.nan,
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
    }


def add_prediction_metrics_box(ax, stats: Dict[str, float]) -> None:
    text = (
        f"n = {stats['n']}\n"
        f"r = {stats['r']:.3f}\n"
        f"R² = {stats['r2']:.3f}\n"
        f"MAE = {stats['mae']:.2f}\n"
        f"RMSE = {stats['rmse']:.2f}"
    )
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))


def add_bag_metrics_box(ax, stats: Dict[str, float]) -> None:
    text = (
        f"n = {stats['n']}\n"
        f"r = {stats['r']:.3f}\n"
        f"R² = {stats['r2']:.3f}\n"
        f"p = {format_p(stats['p'])}\n"
        f"slope = {stats['slope']:.3f}"
    )
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))


# =============================================================================
# QA and input loading
# =============================================================================
def qa_single_oof(path: Path, cohort: str, feature_set: str, read_nrows: Optional[int] = None) -> Dict[str, object]:
    row: Dict[str, object] = {
        "cohort": cohort,
        "feature_set": feature_set,
        "source_path": str(path),
        "exists": path.exists(),
        "status": "missing_file" if not path.exists() else "unchecked",
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else "",
    }

    if not path.exists():
        return row

    try:
        df = pd.read_csv(path, nrows=read_nrows)
    except Exception as exc:
        row["status"] = f"read_error: {exc}"
        return row

    cols = list(df.columns)
    row["n_rows"] = int(len(df))
    row["n_cols"] = int(len(cols))
    row["columns"] = ";".join(map(str, cols))

    missing_required = [c for c in REQUIRED_COLUMNS if c not in cols]
    row["missing_required_columns"] = ";".join(missing_required)
    row["has_age_true"] = "age_true" in cols
    row["has_pred_raw"] = "pred_raw" in cols

    for c in OPTIONAL_BAG_COLUMNS:
        row[f"has_{c}"] = c in cols

    subject_col = find_subject_col(cols)
    row["subject_id_column"] = subject_col if subject_col else ""

    if missing_required:
        row["status"] = "missing_required_columns"
        return row

    age = clean_numeric(df["age_true"])
    pred = clean_numeric(df["pred_raw"])
    valid = age.notna() & pred.notna()

    row["n_valid_age_pred"] = int(valid.sum())
    row["n_missing_age_true"] = int(age.isna().sum())
    row["n_missing_pred_raw"] = int(pred.isna().sum())
    row["age_min"] = float(age.min()) if age.notna().any() else np.nan
    row["age_max"] = float(age.max()) if age.notna().any() else np.nan
    row["pred_min"] = float(pred.min()) if pred.notna().any() else np.nan
    row["pred_max"] = float(pred.max()) if pred.notna().any() else np.nan

    if subject_col:
        row["n_unique_subject_ids"] = int(df[subject_col].astype(str).nunique(dropna=True))
        if "visit" in [str(c).lower() for c in cols]:
            visit_col = [c for c in cols if str(c).lower() == "visit"][0]
            row["n_duplicate_subject_visit"] = int(df.duplicated([subject_col, visit_col]).sum())
        else:
            row["n_duplicate_subject_ids"] = int(df[subject_col].astype(str).duplicated().sum())

    # Check BAG consistency when present.
    if "BAG_raw" in cols:
        bag = clean_numeric(df["BAG_raw"])
        diff = (bag - (pred - age)).abs()
        row["BAG_raw_max_abs_error_vs_pred_minus_age"] = float(diff.max(skipna=True)) if diff.notna().any() else np.nan
    else:
        row["BAG_raw_max_abs_error_vs_pred_minus_age"] = np.nan

    if "cBAG_foldwise" not in cols and "cBAG" in cols:
        row["cBAG_foldwise_source"] = "cBAG_alias"
    elif "cBAG_foldwise" in cols:
        row["cBAG_foldwise_source"] = "cBAG_foldwise"
    else:
        row["cBAG_foldwise_source"] = "missing"

    metrics = prediction_stats(age, pred)
    for k, v in metrics.items():
        row[f"pred_metric_{k}"] = v

    problems = []
    if row["n_valid_age_pred"] < 3:
        problems.append("too_few_valid_age_pred")
    if age.notna().any() and (age.min() < 0 or age.max() > 120):
        problems.append("age_outside_0_120")
    if pred.notna().any() and (pred.min() < 0 or pred.max() > 120):
        problems.append("pred_outside_0_120")
    if row.get("BAG_raw_max_abs_error_vs_pred_minus_age", 0) not in [np.nan, None]:
        try:
            if np.isfinite(row["BAG_raw_max_abs_error_vs_pred_minus_age"]) and row["BAG_raw_max_abs_error_vs_pred_minus_age"] > 1e-6:
                problems.append("BAG_raw_inconsistent_with_pred_minus_age")
        except Exception:
            pass

    row["qa_problems"] = ";".join(problems)
    row["status"] = "loaded" if not problems else "loaded_with_warnings"
    return row


def qa_all_inputs(results_root: Path) -> pd.DataFrame:
    rows = []
    for cohort in COHORT_ORDER:
        for feature_set in FEATURE_SETS:
            rows.append(qa_single_oof(get_oof_path(results_root, cohort, feature_set), cohort, feature_set))
    return pd.DataFrame(rows)


def load_oof(results_root: Path, cohort: str, feature_set: str) -> pd.DataFrame:
    path = get_oof_path(results_root, cohort, feature_set)
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF file: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["cohort"] = cohort
    df["feature_set"] = feature_set
    df["source_path"] = str(path)
    df["age_true"] = clean_numeric(df["age_true"])
    df["pred_raw"] = clean_numeric(df["pred_raw"])

    if "BAG_raw" not in df.columns:
        df["BAG_raw"] = df["pred_raw"] - df["age_true"]
    else:
        df["BAG_raw"] = clean_numeric(df["BAG_raw"])

    if "cBAG_foldwise" not in df.columns and "cBAG" in df.columns:
        df["cBAG_foldwise"] = clean_numeric(df["cBAG"])
    elif "cBAG_foldwise" in df.columns:
        df["cBAG_foldwise"] = clean_numeric(df["cBAG_foldwise"])

    if "cBAG_oof_global" in df.columns:
        df["cBAG_oof_global"] = clean_numeric(df["cBAG_oof_global"])

    return df


def load_oof_data_for_feature_set(results_root: Path, feature_set: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    rows = []
    for cohort in COHORT_ORDER:
        path = get_oof_path(results_root, cohort, feature_set)
        if not path.exists():
            rows.append({"cohort": cohort, "feature_set": feature_set, "source_path": str(path), "status": "missing_file"})
            continue
        try:
            df = load_oof(results_root, cohort, feature_set)
            dfs[cohort] = df
            rows.append({
                "cohort": cohort, "feature_set": feature_set, "n_rows": len(df), "source_path": str(path),
                "status": "loaded", "has_age_true": "age_true" in df.columns, "has_pred_raw": "pred_raw" in df.columns,
                "has_BAG_raw": "BAG_raw" in df.columns, "has_cBAG_foldwise": "cBAG_foldwise" in df.columns,
                "has_cBAG_oof_global": "cBAG_oof_global" in df.columns,
            })
        except Exception as exc:
            rows.append({"cohort": cohort, "feature_set": feature_set, "source_path": str(path), "status": f"error: {exc}"})
    return dfs, pd.DataFrame(rows)


# =============================================================================
# Figure builders
# =============================================================================
def make_figure3_predicted_vs_age(results_root: Path, figure_dir: Path, feature_set: str = MAIN_FIGURE_FEATURE_SET) -> Dict[str, str]:
    base = f"Figure3_OOF_RAW_predicted_vs_chronological_age_{feature_set}"
    out_png = figure_dir / f"{base}.png"
    out_pdf = figure_dir / f"{base}.pdf"
    out_csv = figure_dir / f"{base}_source_data.csv"
    out_stats_csv = figure_dir / f"{base}_stats.csv"
    out_availability_csv = figure_dir / f"{base}_input_availability.csv"

    dfs, availability_df = load_oof_data_for_feature_set(results_root, feature_set)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    source_rows = []
    stats_rows = []

    for i, cohort in enumerate(COHORT_ORDER):
        ax = axes[i]
        if cohort not in dfs:
            ax.set_title(f"{PANEL_LABELS.get(cohort, cohort)}: missing")
            ax.axis("off")
            continue

        df = dfs[cohort].copy()
        x = clean_numeric(df["age_true"]).values
        y = clean_numeric(df["pred_raw"]).values
        mask = np.isfinite(x) & np.isfinite(y)
        stats = prediction_stats(x, y)

        source_rows.append(pd.DataFrame({
            "figure": "Figure3", "feature_set": feature_set, "cohort": cohort,
            "age_true": x, "pred_raw": y, "source_path": df["source_path"].iloc[0],
        }))
        stats_rows.append({"figure": "Figure3", "feature_set": feature_set, "cohort": cohort,
                           "x": "age_true", "y": "pred_raw", **stats})

        ax.scatter(x[mask], y[mask], alpha=0.72, edgecolors="black", linewidth=0.3)
        if mask.sum() >= 3:
            lr = linregress(x[mask], y[mask])
            xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
            ax.plot(xx, lr.slope * xx + lr.intercept, linestyle="--", label="Fit")
            lo = min(np.nanmin(x[mask]), np.nanmin(y[mask]))
            hi = max(np.nanmax(x[mask]), np.nanmax(y[mask]))
            ax.plot([lo, hi], [lo, hi], linestyle=":", label="Identity")
            ax.set_xlim(lo - 1, hi + 1)
            ax.set_ylim(lo - 1, hi + 1)

        add_prediction_metrics_box(ax, stats)
        ax.set_title(PANEL_LABELS.get(cohort, cohort))
        ax.set_xlabel("Chronological age")
        ax.set_ylabel("Predicted age")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Figure 3. OOF raw predicted age vs chronological age ({feature_set})", fontsize=16, y=1.04)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    pd.concat(source_rows, ignore_index=True).to_csv(out_csv, index=False) if source_rows else pd.DataFrame().to_csv(out_csv, index=False)
    pd.DataFrame(stats_rows).to_csv(out_stats_csv, index=False)
    availability_df.to_csv(out_availability_csv, index=False)

    return {"figure": "Figure3", "feature_set": feature_set, "png": str(out_png), "pdf": str(out_pdf),
            "source_data": str(out_csv), "stats": str(out_stats_csv), "availability": str(out_availability_csv),
            "status": "main_figure"}


def make_supplementary_figure3_aggregate(results_root: Path, figure_dir: Path) -> Dict[str, str]:
    base = "SupplementaryFigure3_AGGREGATE_BAG_cBAG_age_dependence_all_feature_sets"
    out_png = figure_dir / f"{base}.png"
    out_pdf = figure_dir / f"{base}.pdf"
    out_csv = figure_dir / f"{base}_source_data.csv"
    out_stats_csv = figure_dir / f"{base}_stats.csv"
    out_availability_csv = figure_dir / f"{base}_input_availability.csv"

    pooled_by_cohort: Dict[str, pd.DataFrame] = {}
    availability_rows = []

    for cohort in COHORT_ORDER:
        cohort_dfs = []
        for feature_set in FEATURE_SETS:
            path = get_oof_path(results_root, cohort, feature_set)
            if not path.exists():
                availability_rows.append({"cohort": cohort, "feature_set": feature_set, "source_path": str(path), "status": "missing_file"})
                continue
            try:
                df = load_oof(results_root, cohort, feature_set)
                cohort_dfs.append(df)
                availability_rows.append({
                    "cohort": cohort, "feature_set": feature_set, "n_rows": len(df), "source_path": str(path), "status": "loaded",
                    "has_BAG_raw": "BAG_raw" in df.columns, "has_cBAG_foldwise": "cBAG_foldwise" in df.columns,
                    "has_cBAG_oof_global": "cBAG_oof_global" in df.columns,
                })
            except Exception as exc:
                availability_rows.append({"cohort": cohort, "feature_set": feature_set, "source_path": str(path), "status": f"error: {exc}"})
        if cohort_dfs:
            pooled_by_cohort[cohort] = pd.concat(cohort_dfs, ignore_index=True, sort=False)

    availability_df = pd.DataFrame(availability_rows)
    source_rows = []
    stats_rows = []

    fig, axes = plt.subplots(len(COHORT_ORDER), len(BAG_COLUMNS), figsize=(15.6, 16.8), dpi=300, squeeze=False)
    for row_idx, cohort in enumerate(COHORT_ORDER):
        for col_idx, bag_col in enumerate(BAG_COLUMNS):
            ax = axes[row_idx, col_idx]
            if cohort not in pooled_by_cohort:
                ax.set_title(f"{cohort}: missing")
                ax.axis("off")
                continue
            df = pooled_by_cohort[cohort].copy()
            if bag_col not in df.columns:
                ax.set_title(f"{cohort}: missing {bag_col}")
                ax.axis("off")
                continue
            x = clean_numeric(df["age_true"]).values
            y = clean_numeric(df[bag_col]).values
            stats = bag_stats(x, y)
            source_rows.append(pd.DataFrame({
                "supplementary_figure": "SupplementaryFigure3_AGGREGATE", "feature_set": df["feature_set"].values,
                "cohort": cohort, "brain_metric": bag_col, "age_true": x, "value": y, "source_path": df["source_path"].values,
            }))
            stats_rows.append({"supplementary_figure": "SupplementaryFigure3_AGGREGATE", "feature_set": "all_feature_sets_pooled",
                               "cohort": cohort, "x": "age_true", "y": bag_col, **stats})
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], alpha=0.35, edgecolors="black", linewidth=0.2)
            if mask.sum() >= 3:
                lr = linregress(x[mask], y[mask])
                xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                ax.plot(xx, lr.slope * xx + lr.intercept, linestyle="--")
                ax.axhline(0, linestyle=":", linewidth=1)
            add_bag_metrics_box(ax, stats)
            if row_idx == 0:
                ax.set_title(BAG_LABELS.get(bag_col, bag_col))
            ax.set_ylabel(f"{cohort}\nBrain-age gap" if col_idx == 0 else "Brain-age gap")
            ax.set_xlabel("Chronological age")
            ax.grid(alpha=0.3)

    fig.suptitle("Supplementary Figure 3. Aggregate BAG/cBAG age-dependence diagnostics across all feature sets", fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    pd.concat(source_rows, ignore_index=True).to_csv(out_csv, index=False) if source_rows else pd.DataFrame().to_csv(out_csv, index=False)
    pd.DataFrame(stats_rows).to_csv(out_stats_csv, index=False)
    availability_df.to_csv(out_availability_csv, index=False)

    return {"figure": "SupplementaryFigure3_AGGREGATE", "feature_set": "all_feature_sets_pooled", "png": str(out_png),
            "pdf": str(out_pdf), "source_data": str(out_csv), "stats": str(out_stats_csv),
            "availability": str(out_availability_csv), "status": "primary_supplementary_figure"}


def make_supplementary_figure3_individual(results_root: Path, individual_dir: Path, panel_letter: str, feature_set: str) -> Dict[str, str]:
    base = f"SupplementaryFigure3{panel_letter}_BAG_cBAG_age_dependence_{feature_set}"
    out_png = individual_dir / f"{base}.png"
    out_pdf = individual_dir / f"{base}.pdf"
    out_csv = individual_dir / f"{base}_source_data.csv"
    out_stats_csv = individual_dir / f"{base}_stats.csv"
    out_availability_csv = individual_dir / f"{base}_input_availability.csv"

    dfs, availability_df = load_oof_data_for_feature_set(results_root, feature_set)
    source_rows = []
    stats_rows = []
    fig, axes = plt.subplots(len(COHORT_ORDER), len(BAG_COLUMNS), figsize=(15.6, 16.8), dpi=300, squeeze=False)

    for row_idx, cohort in enumerate(COHORT_ORDER):
        for col_idx, bag_col in enumerate(BAG_COLUMNS):
            ax = axes[row_idx, col_idx]
            if cohort not in dfs:
                ax.set_title(f"{cohort}: missing")
                ax.axis("off")
                continue
            df = dfs[cohort].copy()
            if bag_col not in df.columns:
                ax.set_title(f"{cohort}: missing {bag_col}")
                ax.axis("off")
                continue
            x = clean_numeric(df["age_true"]).values
            y = clean_numeric(df[bag_col]).values
            stats = bag_stats(x, y)
            source_rows.append(pd.DataFrame({
                "supplementary_figure": f"SupplementaryFigure3{panel_letter}", "feature_set": feature_set,
                "cohort": cohort, "brain_metric": bag_col, "age_true": x, "value": y,
                "source_path": df["source_path"].iloc[0],
            }))
            stats_rows.append({"supplementary_figure": f"SupplementaryFigure3{panel_letter}", "feature_set": feature_set,
                               "cohort": cohort, "x": "age_true", "y": bag_col, **stats})
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], alpha=0.72, edgecolors="black", linewidth=0.3)
            if mask.sum() >= 3:
                lr = linregress(x[mask], y[mask])
                xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                ax.plot(xx, lr.slope * xx + lr.intercept, linestyle="--")
                ax.axhline(0, linestyle=":", linewidth=1)
            add_bag_metrics_box(ax, stats)
            if row_idx == 0:
                ax.set_title(BAG_LABELS.get(bag_col, bag_col))
            ax.set_ylabel(f"{cohort}\nBrain-age gap" if col_idx == 0 else "Brain-age gap")
            ax.set_xlabel("Chronological age")
            ax.grid(alpha=0.3)

    fig.suptitle(f"Supplementary Figure 3{panel_letter}. BAG/cBAG age-dependence diagnostics ({feature_set})", fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    pd.concat(source_rows, ignore_index=True).to_csv(out_csv, index=False) if source_rows else pd.DataFrame().to_csv(out_csv, index=False)
    pd.DataFrame(stats_rows).to_csv(out_stats_csv, index=False)
    availability_df.to_csv(out_availability_csv, index=False)

    return {"figure": f"SupplementaryFigure3{panel_letter}", "feature_set": feature_set, "png": str(out_png),
            "pdf": str(out_pdf), "source_data": str(out_csv), "stats": str(out_stats_csv),
            "availability": str(out_availability_csv), "status": "extended_individual_feature_set_qc"}


# =============================================================================
# Output handling
# =============================================================================
def prepare_output_dir(base_validation_dir: Path, run_tag: str, overwrite: bool, backup_existing: bool) -> Tuple[Path, Path]:
    figure_dir = base_validation_dir / "Figure3" if run_tag == "current" else base_validation_dir / "Figure3_runs" / f"Figure3_{run_tag}"
    individual_dir = figure_dir / "extended_individual_feature_sets"

    if figure_dir.exists() and any(figure_dir.iterdir()) and not overwrite and run_tag == "current":
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {figure_dir}\n"
            f"Use --overwrite, or use --run-tag {now_tag()} to write a fresh run directory."
        )

    if figure_dir.exists() and overwrite and backup_existing and run_tag == "current":
        backup_dir = base_validation_dir / f"Figure3_BACKUP_before_regen_{now_tag()}"
        log(f"Backing up existing Figure3 directory to: {backup_dir}")
        if backup_dir.exists():
            raise FileExistsError(f"Backup directory already exists: {backup_dir}")
        shutil.move(str(figure_dir), str(backup_dir))

    mkdir(figure_dir)
    mkdir(individual_dir)
    mkdir(figure_dir / "QA")
    return figure_dir, individual_dir


def write_run_metadata(figure_dir: Path, args: argparse.Namespace) -> Path:
    def make_json_safe(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "args": make_json_safe(vars(args)),
        "cohorts": COHORT_ORDER,
        "feature_sets": FEATURE_SETS,
        "required_columns": REQUIRED_COLUMNS,
        "bag_columns_plotted": BAG_COLUMNS,
    }
    out = figure_dir / "QA" / "Figure3_run_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def inventory_outputs(figure_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(figure_dir.rglob("*")):
        if p.is_file():
            rows.append({
                "path": str(p),
                "relative_path": str(p.relative_to(figure_dir)),
                "size_bytes": p.stat().st_size,
                "modified_time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            })
    return pd.DataFrame(rows)


def fail_if_strict_qa_fails(qa_df: pd.DataFrame) -> None:
    bad_status = qa_df[~qa_df["status"].isin(["loaded", "loaded_with_warnings"])]
    missing_main = qa_df[(qa_df["feature_set"] == MAIN_FIGURE_FEATURE_SET) & (qa_df["status"] != "loaded")]
    missing_cbag_oof = qa_df[qa_df.get("has_cBAG_oof_global", False) != True]

    messages = []
    if len(bad_status):
        messages.append("Some expected OOF files could not be loaded or have missing required columns.")
    if len(missing_main):
        messages.append("The main imaging_only Figure 3 has missing/invalid cohort input(s).")
    if len(missing_cbag_oof):
        messages.append("Some files lack cBAG_oof_global, so cBAG OOF-global panels would be incomplete.")

    if messages:
        preview_cols = [c for c in ["cohort", "feature_set", "status", "missing_required_columns", "source_path"] if c in qa_df.columns]
        details = qa_df[qa_df["status"] != "loaded"][preview_cols].to_string(index=False)
        raise RuntimeError("STRICT QA FAILED:\n" + "\n".join(messages) + "\n\nProblem rows:\n" + details)


# =============================================================================
# Main
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Figure 3 with QA from final OOF prediction CSV files.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--base-validation-name", default=DEFAULT_BASE_VALIDATION_NAME)
    parser.add_argument("--run-tag", default="current", help="Use 'current' for the canonical Figure3 folder, or a timestamp/name for Figure3_runs/Figure3_<tag>.")
    parser.add_argument("--qa-only", action="store_true", help="Only run input QA; do not make figures.")
    parser.add_argument("--strict", action="store_true", help="Stop if expected files/columns are missing or cBAG_oof_global is absent.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into canonical Figure3 output. Required if --run-tag current and directory is non-empty.")
    parser.add_argument("--backup-existing", action="store_true", help="When overwriting canonical Figure3, move old Figure3 to a timestamped backup first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root: Path = args.results_root
    base_validation_dir = results_root / args.base_validation_name

    log("=" * 100)
    log("FIGURE 3 BUILD WITH QA")
    log("=" * 100)
    log(f"Results root: {results_root}")
    log(f"Base validation dir: {base_validation_dir}")
    log(f"Run tag: {args.run_tag}")

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    figure_dir, individual_dir = prepare_output_dir(
        base_validation_dir=base_validation_dir,
        run_tag=args.run_tag,
        overwrite=args.overwrite,
        backup_existing=args.backup_existing,
    )
    write_run_metadata(figure_dir, args)

    qa_dir = mkdir(figure_dir / "QA")
    qa_df = qa_all_inputs(results_root)
    qa_csv = qa_dir / "Figure3_input_QA_all_expected_oof_predictions.csv"
    qa_xlsx = qa_dir / "Figure3_input_QA_all_expected_oof_predictions.xlsx"
    qa_df.to_csv(qa_csv, index=False)
    try:
        qa_df.to_excel(qa_xlsx, index=False)
    except Exception as exc:
        log(f"[WARN] Could not write QA Excel: {exc}")

    summary = qa_df.groupby(["status"], dropna=False).size().reset_index(name="n_files")
    summary_csv = qa_dir / "Figure3_input_QA_status_summary.csv"
    summary.to_csv(summary_csv, index=False)

    log("\nInput QA status summary:")
    log(summary.to_string(index=False))
    log(f"\nSaved QA: {qa_csv}")

    if args.strict:
        fail_if_strict_qa_fails(qa_df)

    if args.qa_only:
        log("\nQA-only mode: stopping before figure generation.")
        return

    manifest_rows = []
    manifest_rows.append(make_figure3_predicted_vs_age(results_root, figure_dir, MAIN_FIGURE_FEATURE_SET))
    manifest_rows.append(make_supplementary_figure3_aggregate(results_root, figure_dir))
    for panel_letter, feature_set in FEATURE_PANEL_MAP.items():
        manifest_rows.append(make_supplementary_figure3_individual(results_root, individual_dir, panel_letter, feature_set))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = figure_dir / "Figure3_SupplementaryFigure3_manifest.csv"
    manifest_xlsx = figure_dir / "Figure3_SupplementaryFigure3_manifest.xlsx"
    manifest_df.to_csv(manifest_csv, index=False)
    try:
        manifest_df.to_excel(manifest_xlsx, index=False)
    except Exception as exc:
        log(f"[WARN] Could not write manifest Excel: {exc}")

    inv = inventory_outputs(figure_dir)
    inv_csv = qa_dir / "Figure3_output_inventory.csv"
    inv.to_csv(inv_csv, index=False)

    log("\nSaved manifest:")
    log(str(manifest_csv))
    log(str(manifest_xlsx))
    log("\nSaved output inventory:")
    log(str(inv_csv))
    log("\nDone.")


if __name__ == "__main__":
    main()
