#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_Table2.py

Canonical Table 2 pipeline for the Brain Age manuscript.

Primary output
--------------
Table 2: raw out-of-fold (OOF) predicted-age performance in the cognitively
normal training/cross-validation sample.

Supplementary outputs
---------------------
Table S2: pooled raw OOF performance across cohorts.
Table S3: full-cohort descriptive performance in all available sessions.
Table S4: across-cohort model ranking based on mean within-cohort rank.
Table S5: corrected/residualized BAG diagnostic table from the combined
          bootstrap summary, when available.

Key safeguards
--------------
- Primary Table 2 uses RAW OOF predicted age only.
- Bias-corrected/cBAG/residualized metrics are not used for the primary table.
- Bootstrap confidence intervals are participant-cluster bootstraps.
- Repeated visits are clustered by participant when participant IDs can be inferred.
- The script uses explicit, auditable input paths.

Output root
-----------
$WORK/ines/results/Table2
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# SETTINGS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
RESULTS_ROOT = WORK / "ines" / "results"
OUTDIR = RESULTS_ROOT / "Table2"

VALIDATION_BASE = (
    RESULTS_ROOT
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)
COMBINED_BOOTSTRAP_CSV = VALIDATION_BASE / "combined_bootstrap_metric_summary.csv"

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full_no_cardiovascular",
    "full",
]

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

MODEL_LABELS = {
    "imaging_only": "Imaging only",
    "imaging_demographics": "Imaging + demographics",
    "imaging_biomarkers": "Imaging + biomarkers",
    "full_no_cardiovascular": "Full model without cardiovascular variables",
    "full": "Full model",
}

COHORT_LABELS = {
    "ADNI": "ADNI",
    "ADRC": "ADRC",
    "HABS": "HABS",
    "AD_DECODE": "AD-DECODE",
}

DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_SEED = 42
MIN_N = 3


# =============================================================================
# COLUMN CANDIDATES
# =============================================================================

TRUE_AGE_CANDIDATES = [
    "age_true",
    "Age",
    "age",
    "chronological_age",
    "Chronological_Age",
    "true_age",
    "y_true",
    "target_age",
    "Real_Age",
    "real_age",
    "VISIT_AGE",
    "AGE",
]

RAW_PRED_CANDIDATES = [
    "pred_raw",
    "predicted_age",
    "Predicted_Age",
    "pred_age",
    "prediction",
    "age_pred",
    "y_pred",
    "brain_age_pred",
    "predicted_brain_age",
    "brain_age",
    "BrainAge",
]

RAW_BAG_CANDIDATES = [
    "BAG_raw",
    "BAG",
    "brain_age_gap_raw",
    "brain_age_gap",
    "BrainAgeGap",
    "age_gap_raw",
    "age_gap",
]

PARTICIPANT_ID_CANDIDATES = [
    "participant_group",
    "DWI_subject",
    "DWI_subject_key",
    "Subject",
    "subject",
    "subject_id",
    "PTID",
    "RID",
    "Med_ID",
    "graph_subject",
]

SESSION_ID_CANDIDATES = [
    "graph_id",
    "DWI",
    "connectome_key",
    "connectome_full_key",
    "subject_id",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InputRecord:
    cohort: str
    feature_set: str
    analysis: str
    source_path: str
    source_type: str
    exists: bool
    rows: int = 0
    status: str = ""
    true_age_col: str = ""
    pred_age_col: str = ""
    participant_col: str = ""
    session_col: str = ""


# =============================================================================
# BASIC HELPERS
# =============================================================================

def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    exact = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    norm = {normalize_name(c): c for c in df.columns}

    for c in candidates:
        if c in exact:
            return exact[c]
    for c in candidates:
        key = str(c).lower()
        if key in lower:
            return lower[key]
    for c in candidates:
        key = normalize_name(c)
        if key in norm:
            return norm[key]
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {path}")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def format_float(x: object, digits: int = 3) -> str:
    try:
        value = float(x)
    except Exception:
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def format_ci(point: object, low: object, high: object, digits: int = 3) -> str:
    try:
        p = float(point)
        lo = float(low)
        hi = float(high)
    except Exception:
        return ""
    if not all(np.isfinite(v) for v in [p, lo, hi]):
        return format_float(point, digits)
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def ablation_dir(cohort: str, feature_set: str) -> Path:
    return RESULTS_ROOT / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}"


def normalize_subject_key(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""

    for pat in [r"(R\d+)", r"(H\d+)", r"(S\d+)", r"(ADRC\d+|D\d+)"]:
        m = re.search(pat, s, flags=re.I)
        if m:
            return m.group(1).upper()

    return re.sub(r"[^A-Za-z0-9]+", "", s).upper()


def derive_participant_from_session(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = re.sub(r"_y\d+$", "", s, flags=re.IGNORECASE)
    return normalize_subject_key(s)


# =============================================================================
# INPUT PATHS
# =============================================================================

def oof_candidates(cohort: str, feature_set: str) -> list[tuple[str, Path]]:
    root = ablation_dir(cohort, feature_set)
    prefix = PREFIX_MAP[cohort]
    return [
        ("cv_oof_predictions_csv", root / f"{prefix}_{feature_set}_cv_oof_predictions.csv"),
        ("cv_oof_predictions_xlsx", root / f"{prefix}_{feature_set}_cv_oof_predictions.xlsx"),
        ("validation_oof_csv", root / "validation_figures_oof" / "subject_level_validation_input.csv"),
        ("validation_default_csv", root / "validation_figures" / "subject_level_validation_input.csv"),
    ]


def full_candidates(cohort: str, feature_set: str) -> list[tuple[str, Path]]:
    root = ablation_dir(cohort, feature_set)
    prefix = PREFIX_MAP[cohort]
    return [
        ("full_cohort_predictions_csv", root / f"{prefix}_{feature_set}_full_cohort_predictions.csv"),
        ("full_cohort_predictions_xlsx", root / f"{prefix}_{feature_set}_full_cohort_predictions.xlsx"),
        ("metadata_all_with_predictions_csv", root / f"{prefix}_{feature_set}_metadata_all_with_predictions.csv"),
        ("validation_full_cohort_csv", root / "validation_figures_full_cohort" / "subject_level_validation_input.csv"),
    ]


# =============================================================================
# SCOREABLE TABLE EXTRACTION
# =============================================================================

def find_true_age_col(df: pd.DataFrame) -> Optional[str]:
    col = first_existing(df, TRUE_AGE_CANDIDATES)
    if col:
        return col

    for c in df.columns:
        low = normalize_name(c)
        if "age" not in low:
            continue
        if any(tok in low for tok in ["pred", "bag", "cbag", "gap", "delta", "correct"]):
            continue
        x = safe_numeric(df[c])
        if x.notna().sum() >= 10 and 35 <= x.dropna().median() <= 105:
            return c
    return None


def find_raw_pred_col(df: pd.DataFrame) -> Optional[str]:
    col = first_existing(df, RAW_PRED_CANDIDATES)
    if col:
        return col

    for c in df.columns:
        low = normalize_name(c)
        if "pred" in low and "age" in low and "bias" not in low and "correct" not in low:
            x = safe_numeric(df[c])
            if x.notna().sum() >= 10:
                return c
    return None


def find_raw_bag_col(df: pd.DataFrame) -> Optional[str]:
    col = first_existing(df, RAW_BAG_CANDIDATES)
    if col:
        return col

    for c in df.columns:
        low = normalize_name(c)
        if ("bag" in low or "gap" in low) and "cbag" not in low and "bias" not in low and "correct" not in low:
            x = safe_numeric(df[c])
            if x.notna().sum() >= 10:
                return c
    return None


def find_participant_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, PARTICIPANT_ID_CANDIDATES)


def extract_scoreable_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    age_col = find_true_age_col(df)
    pred_col = find_raw_pred_col(df)
    bag_col = find_raw_bag_col(df)
    participant_col = find_participant_col(df)
    session_col = first_existing(df, SESSION_ID_CANDIDATES)

    info = {
        "true_age_col": age_col or "",
        "pred_age_col": pred_col or "",
        "bag_col": bag_col or "",
        "participant_col": participant_col or "",
        "session_col": session_col or "",
        "prediction_source": "",
    }

    if age_col is None:
        return pd.DataFrame(), info

    y_true = safe_numeric(df[age_col])

    if pred_col is not None:
        y_pred = safe_numeric(df[pred_col])
        info["prediction_source"] = pred_col
    elif bag_col is not None:
        y_pred = y_true + safe_numeric(df[bag_col])
        info["prediction_source"] = f"{age_col}+{bag_col}"
    else:
        return pd.DataFrame(), info

    out = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=df.index)

    if participant_col is not None:
        out["participant_id"] = df[participant_col].map(normalize_subject_key)
    elif session_col is not None:
        out["participant_id"] = df[session_col].map(derive_participant_from_session)
    else:
        out["participant_id"] = np.arange(len(df)).astype(str)

    if session_col is not None:
        out["session_id"] = df[session_col].astype(str).str.strip()
    else:
        out["session_id"] = np.arange(len(df)).astype(str)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, info


def load_first_scoreable(cohort: str, feature_set: str, analysis: str) -> tuple[Optional[pd.DataFrame], InputRecord]:
    candidates = oof_candidates(cohort, feature_set) if analysis == "oof_training_controls" else full_candidates(cohort, feature_set)
    attempted: list[str] = []

    for source_type, path in candidates:
        attempted.append(str(path))
        if not path.exists():
            continue
        try:
            raw = read_table(path)
        except Exception:
            continue

        scoreable, info = extract_scoreable_table(raw)
        if scoreable.empty:
            continue

        record = InputRecord(
            cohort=cohort,
            feature_set=feature_set,
            analysis=analysis,
            source_path=str(path),
            source_type=source_type,
            exists=True,
            rows=len(raw),
            status="ok",
            true_age_col=info.get("true_age_col", ""),
            pred_age_col=info.get("prediction_source", ""),
            participant_col=info.get("participant_col", ""),
            session_col=info.get("session_col", ""),
        )
        return scoreable, record

    record = InputRecord(
        cohort=cohort,
        feature_set=feature_set,
        analysis=analysis,
        source_path="; ".join(attempted),
        source_type="none",
        exists=False,
        rows=0,
        status="missing_or_not_scoreable",
    )
    return None, record


# =============================================================================
# METRICS AND BOOTSTRAP
# =============================================================================

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tmp = pd.DataFrame({
        "y_true": safe_numeric(y_true),
        "y_pred": safe_numeric(y_pred),
    }).dropna()

    if len(tmp) < MIN_N or tmp["y_true"].nunique() < 2 or tmp["y_pred"].nunique() < 2:
        return {
            "status": "too_few_usable_values",
            "n": int(len(tmp)),
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "mean_error": np.nan,
            "sd_error": np.nan,
        }

    yt = tmp["y_true"].to_numpy(dtype=float)
    yp = tmp["y_pred"].to_numpy(dtype=float)
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1 - sse / sst) if sst > 0 else np.nan
    pearson_r, pearson_p = stats.pearsonr(yt, yp)
    slope, intercept, *_ = stats.linregress(yt, yp)

    return {
        "status": "ok",
        "n": int(len(tmp)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "slope": float(slope),
        "intercept": float(intercept),
        "mean_error": float(np.mean(err)),
        "sd_error": float(np.std(err, ddof=0)),
    }


def cluster_bootstrap_metrics(df: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    work = df[["y_true", "y_pred", "participant_id"]].copy()
    work["y_true"] = safe_numeric(work["y_true"])
    work["y_pred"] = safe_numeric(work["y_pred"])
    work = work.dropna(subset=["y_true", "y_pred"])

    point = compute_metrics(work["y_true"], work["y_pred"])
    if point["status"] != "ok":
        return point

    rng = np.random.default_rng(seed)
    units = work["participant_id"].astype(str).replace("", np.nan).dropna().unique()
    if len(units) == 0:
        work["participant_id"] = np.arange(len(work)).astype(str)
        units = work["participant_id"].unique()

    metric_names = ["mae", "rmse", "r2", "pearson_r"]
    boot_values = {m: [] for m in metric_names}
    grouped = {u: g for u, g in work.groupby(work["participant_id"].astype(str), sort=False)}

    for _ in range(n_bootstrap):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sampled = pd.concat([grouped[u] for u in sampled_units], ignore_index=True)
        m = compute_metrics(sampled["y_true"], sampled["y_pred"])
        if m["status"] != "ok":
            continue
        for name in metric_names:
            if np.isfinite(m[name]):
                boot_values[name].append(float(m[name]))

    for name in metric_names:
        values = np.asarray(boot_values[name], dtype=float)
        point[f"{name}_ci_low"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
        point[f"{name}_ci_high"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
        point[f"{name}_bootstrap_valid"] = int(len(values))

    point["n_participants"] = int(len(units))
    point["bootstrap_unit"] = "participant" if len(units) < len(work) else "row"
    point["n_bootstrap_requested"] = int(n_bootstrap)
    return point


# =============================================================================
# ANALYSIS JOBS
# =============================================================================

def analyze_one_job(job: tuple[str, str, str, int, int]) -> tuple[dict, dict, Optional[pd.DataFrame]]:
    cohort, feature_set, analysis, n_bootstrap, seed = job
    scoreable, record = load_first_scoreable(cohort, feature_set, analysis)

    row = {
        "analysis": analysis,
        "cohort": cohort,
        "feature_set": feature_set,
        "status": record.status,
        "source_type": record.source_type,
        "source_path": record.source_path,
        "true_age_col": record.true_age_col,
        "pred_age_col": record.pred_age_col,
        "participant_col": record.participant_col,
        "session_col": record.session_col,
    }

    if scoreable is None:
        return row, asdict(record), None

    scoreable = scoreable.dropna(subset=["y_true", "y_pred"]).copy()
    n_sessions = int(len(scoreable))
    n_unique_sessions = int(scoreable["session_id"].astype(str).nunique())
    n_duplicate_sessions = int(n_sessions - n_unique_sessions)
    n_participants_observed = int(scoreable["participant_id"].astype(str).replace("", np.nan).nunique(dropna=True))

    row.update({
        "n_sessions_input": n_sessions,
        "n_unique_sessions_input": n_unique_sessions,
        "n_duplicate_sessions_input": n_duplicate_sessions,
        "n_participants_input": n_participants_observed,
    })

    if record.session_col and n_duplicate_sessions > 0:
        row["status"] = "duplicate_exact_session_ids"
        row["n"] = n_sessions
        return row, asdict(record), scoreable

    metrics = cluster_bootstrap_metrics(scoreable, n_bootstrap=n_bootstrap, seed=seed)
    row.update(metrics)
    return row, asdict(record), scoreable if row.get("status") == "ok" else None


def run_jobs(analyses: list[str], n_bootstrap: int, seed: int, workers: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    jobs = []
    job_i = 0
    for analysis in analyses:
        for cohort in COHORTS:
            for feature_set in FEATURE_SETS:
                job_i += 1
                jobs.append((cohort, feature_set, analysis, n_bootstrap, seed + job_i * 1009))

    rows: dict[str, list[dict]] = {a: [] for a in analyses}
    inventory_rows: list[dict] = []
    prediction_frames: dict[tuple[str, str, str], pd.DataFrame] = {}

    if workers <= 1:
        results = [analyze_one_job(job) for job in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(analyze_one_job, job): job for job in jobs}
            for fut in as_completed(future_map):
                results.append(fut.result())

    for row, inv, frame in results:
        analysis = row["analysis"]
        rows[analysis].append(row)
        inventory_rows.append(inv)
        if frame is not None and not frame.empty and row.get("status") == "ok":
            prediction_frames[(analysis, row["cohort"], row["feature_set"])] = frame

        print(
            f"[{analysis:21s}] {row['cohort']:10s} {row['feature_set']:34s} "
            f"status={row.get('status','')} n={row.get('n',0)} source={row.get('source_type','')}"
        )

    outputs = {analysis: pd.DataFrame(rows[analysis]) for analysis in analyses}
    inventory = pd.DataFrame(inventory_rows)
    return outputs, inventory, prediction_frames


# =============================================================================
# RANKING
# =============================================================================

def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clean within-cohort and across-cohort ranks.

    Main Table 2 rank:
      1) lower MAE
      2) lower RMSE
      3) higher R²
      4) higher Pearson r

    Supplementary Table S4 rank:
      average of within-cohort ranks across cohorts.
    """
    out = df.copy()

    rank_cols = [
        "rank_within_cohort",
        "rank_mae_within_cohort",
        "rank_rmse_within_cohort",
        "rank_r2_within_cohort",
        "rank_r_within_cohort",
        "mean_rank_within_cohort",
        "mean_rank_across_cohorts",
        "rank_across_cohorts",
    ]

    for col in rank_cols:
        out[col] = np.nan

    ok = out["status"].eq("ok") if "status" in out.columns else pd.Series(False, index=out.index)
    if not ok.any():
        return out

    ranked_parts = []

    for cohort, sub in out.loc[ok].groupby("cohort", sort=False):
        sub = sub.copy()

        # Metric-specific ranks for audit.
        sub["rank_mae_within_cohort"] = sub["mae"].rank(method="min", ascending=True)
        sub["rank_rmse_within_cohort"] = sub["rmse"].rank(method="min", ascending=True)
        sub["rank_r2_within_cohort"] = sub["r2"].rank(method="min", ascending=False)
        sub["rank_r_within_cohort"] = sub["pearson_r"].rank(method="min", ascending=False)

        sub["mean_rank_within_cohort"] = sub[
            [
                "rank_mae_within_cohort",
                "rank_rmse_within_cohort",
                "rank_r2_within_cohort",
                "rank_r_within_cohort",
            ]
        ].mean(axis=1)

        # Main manuscript rank: deterministic lexicographic rank.
        # This avoids duplicate ranks and matches the corrected publishable outputs.
        sub = sub.sort_values(
            ["mae", "rmse", "r2", "pearson_r"],
            ascending=[True, True, False, False],
        ).copy()

        sub["rank_within_cohort"] = np.arange(1, len(sub) + 1)
        ranked_parts.append(sub)

    ranked = pd.concat(ranked_parts, axis=0)

    # Across-cohort rank uses final within-cohort ranks.
    across = (
        ranked.groupby("feature_set", as_index=False)
        .agg(
            mean_rank_across_cohorts=("rank_within_cohort", "mean"),
            n_cohorts_ranked=("cohort", "nunique"),
            mean_mae_across_cohorts=("mae", "mean"),
            mean_rmse_across_cohorts=("rmse", "mean"),
            mean_r2_across_cohorts=("r2", "mean"),
            mean_r_across_cohorts=("pearson_r", "mean"),
        )
        .sort_values(
            [
                "mean_rank_across_cohorts",
                "mean_mae_across_cohorts",
                "mean_rmse_across_cohorts",
                "mean_r2_across_cohorts",
                "mean_r_across_cohorts",
            ],
            ascending=[True, True, True, False, False],
        )
        .copy()
    )

    across["rank_across_cohorts"] = np.arange(1, len(across) + 1)

    ranked = ranked.merge(
        across[
            [
                "feature_set",
                "mean_rank_across_cohorts",
                "rank_across_cohorts",
            ]
        ],
        on="feature_set",
        how="left",
        suffixes=("", "_from_across"),
    )

    # Assign back by original row index. This is the critical fix.
    for col in rank_cols:
        out.loc[ranked.index, col] = ranked[col]

    return out

def make_publishable_by_cohort(df: pd.DataFrame, table_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df[df["status"].eq("ok")].copy()
    if out.empty:
        return pd.DataFrame()

    out["cohort_order"] = pd.Categorical(out["cohort"], categories=COHORTS, ordered=True)
    out = out.sort_values(["cohort_order", "rank_within_cohort"]).copy()

    return pd.DataFrame({
        "Table": table_label,
        "Cohort": out["cohort"].map(COHORT_LABELS).fillna(out["cohort"]),
        "Rank": out["rank_within_cohort"].astype("Int64"),
        "BAG model": out["feature_set"].map(MODEL_LABELS).fillna(out["feature_set"]),
        "N sessions": out["n"].astype("Int64"),
        "N participants": out["n_participants"].astype("Int64"),
        "MAE [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out["mae"], out["mae_ci_low"], out["mae_ci_high"])],
        "RMSE [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out["rmse"], out["rmse_ci_low"], out["rmse_ci_high"])],
        "R² [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out["r2"], out["r2_ci_low"], out["r2_ci_high"])],
        "Pearson r [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out["pearson_r"], out["pearson_r_ci_low"], out["pearson_r_ci_high"])],
    })


def make_publishable_pooled(df: pd.DataFrame, table_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df[df["status"].eq("ok")].copy().sort_values("overall_rank")
    if out.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "Table": table_label,
        "Rank": out["overall_rank"].astype("Int64"),
        "BAG model": out["feature_set"].map(MODEL_LABELS).fillna(out["feature_set"]),
        "N sessions": out["n"].astype("Int64"),
        "N participants": out["n_participants"].astype("Int64"),
        "MAE [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out["mae"], out["mae_ci_low"], out["mae_ci_high"])],
        "RMSE [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out["rmse"], out["rmse_ci_low"], out["rmse_ci_high"])],
        "R² [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out["r2"], out["r2_ci_low"], out["r2_ci_high"])],
        "Pearson r [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out["pearson_r"], out["pearson_r_ci_low"], out["pearson_r_ci_high"])],
        "Cohorts included": out["cohorts_included"],
    })


def make_publishable_across_cohort_rank(df: pd.DataFrame, table_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df[df["status"].eq("ok")].copy()
    if out.empty:
        return pd.DataFrame()

    summary = (
        out.groupby("feature_set", as_index=False)
        .agg(
            rank_across_cohorts=("rank_across_cohorts", "first"),
            mean_rank_across_cohorts=("mean_rank_across_cohorts", "first"),
            n_cohorts=("cohort", "nunique"),
            mean_MAE=("mae", "mean"),
            mean_RMSE=("rmse", "mean"),
            mean_R2=("r2", "mean"),
            mean_r=("pearson_r", "mean"),
        )
        .sort_values("rank_across_cohorts")
    )

    return pd.DataFrame({
        "Table": table_label,
        "Across-cohort rank": summary["rank_across_cohorts"].astype("Int64"),
        "BAG model": summary["feature_set"].map(MODEL_LABELS).fillna(summary["feature_set"]),
        "N cohorts": summary["n_cohorts"].astype("Int64"),
        "Mean within-cohort rank": summary["mean_rank_across_cohorts"].map(lambda x: format_float(x, 2)),
        "Mean MAE": summary["mean_MAE"].map(lambda x: format_float(x, 3)),
        "Mean RMSE": summary["mean_RMSE"].map(lambda x: format_float(x, 3)),
        "Mean R²": summary["mean_R2"].map(lambda x: format_float(x, 3)),
        "Mean Pearson r": summary["mean_r"].map(lambda x: format_float(x, 3)),
    })


def make_corrected_diagnostic_table() -> pd.DataFrame:
    if not COMBINED_BOOTSTRAP_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(COMBINED_BOOTSTRAP_CSV)
    required = {"cohort", "feature_set", "evaluation", "metric", "point_estimate", "ci_low", "ci_high"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df[df["evaluation"].isin(["OOF_BIAS_CORRECTED", "OOF_GLOBAL_BAG_RESIDUALIZED"])].copy()
    if df.empty:
        return pd.DataFrame()

    wide = df.pivot_table(
        index=["evaluation", "cohort", "feature_set"],
        columns="metric",
        values=["point_estimate", "ci_low", "ci_high"],
        aggfunc="first",
    ).reset_index()
    wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns.to_flat_index()]

    rows = []
    for evaluation, sub in wide.groupby("evaluation", sort=False):
        sub = sub.copy()
        parts = []
        for cohort, csub in sub.groupby("cohort", sort=False):
            csub = csub.sort_values(
                ["point_estimate_MAE", "point_estimate_RMSE", "point_estimate_R2", "point_estimate_r"],
                ascending=[True, True, False, False],
            ).copy()
            csub["Rank"] = np.arange(1, len(csub) + 1)
            parts.append(csub)
        ranked = pd.concat(parts, axis=0)
        rows.append(ranked)

    out = pd.concat(rows, axis=0)
    out["cohort_order"] = pd.Categorical(out["cohort"], categories=COHORTS, ordered=True)
    out["feature_order"] = pd.Categorical(out["feature_set"], categories=FEATURE_SETS, ordered=True)
    out = out.sort_values(["evaluation", "cohort_order", "Rank", "feature_order"])

    return pd.DataFrame({
        "Table": "Supplementary Table S5 corrected/residualized diagnostic",
        "Evaluation": out["evaluation"],
        "Cohort": out["cohort"].map(COHORT_LABELS).fillna(out["cohort"]),
        "Rank": out["Rank"].astype("Int64"),
        "BAG model": out["feature_set"].map(MODEL_LABELS).fillna(out["feature_set"]),
        "MAE [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out.get("point_estimate_MAE"), out.get("ci_low_MAE"), out.get("ci_high_MAE"))],
        "RMSE [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out.get("point_estimate_RMSE"), out.get("ci_low_RMSE"), out.get("ci_high_RMSE"))],
        "R² [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out.get("point_estimate_R2"), out.get("ci_low_R2"), out.get("ci_high_R2"))],
        "Pearson r [95% CI]": [format_ci(p, lo, hi) for p, lo, hi in zip(out.get("point_estimate_r"), out.get("ci_low_r"), out.get("ci_high_r"))],
    })


# =============================================================================
# SAVING
# =============================================================================

def save_table(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_excel(stem.with_suffix(".xlsx"), index=False)
    if not df.empty:
        stem.with_suffix(".tex").write_text(df.to_latex(index=False, escape=False))


def write_notes(publish_dir: Path) -> None:
    notes = """Table 2. Raw out-of-fold predicted-age performance by cohort and feature-set model.
Performance was evaluated in the cognitively normal training/cross-validation sample.
Values are point estimates with participant-cluster bootstrap 95% confidence intervals.
Models are ranked within each cohort using lower MAE as the primary criterion, followed by
lower RMSE, higher R², and higher Pearson r.

Supplementary Table S2. Pooled raw OOF performance across cohorts. This table is
supplementary because pooled metrics are influenced by cohort age distributions, sample size,
and cohort composition.

Supplementary Table S3. Full-cohort descriptive performance in all available sessions.
This table is descriptive and should not be interpreted as unbiased cross-validation.

Supplementary Table S4. Across-cohort model ranking. Models are ranked by their mean
within-cohort composite rank across cohorts.

Supplementary Table S5. Bias-corrected and OOF-global BAG-residualized diagnostic table.
These corrected/residualized metrics are not used as the primary model-performance estimate.
"""
    (publish_dir / "Table2_table_notes.txt").write_text(notes)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical Table 2 and supplementary performance tables.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for cohort × model bootstrap jobs.")
    parser.add_argument(
        "--analyses",
        default="oof_training_controls,full_all",
        help="Comma-separated analyses: oof_training_controls,full_all.",
    )
    args = parser.parse_args()

    analyses = [x.strip() for x in args.analyses.split(",") if x.strip()]
    valid = {"oof_training_controls", "full_all"}
    unknown = sorted(set(analyses) - valid)
    if unknown:
        raise ValueError(f"Unknown analyses: {unknown}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    analytic_dir = OUTDIR / "analytic"
    publish_dir = OUTDIR / "publishable"
    audit_dir = OUTDIR / "audit"
    for d in [analytic_dir, publish_dir, audit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("CANONICAL TABLE 2 PIPELINE")
    print(f"WORK={WORK}")
    print(f"RESULTS_ROOT={RESULTS_ROOT}")
    print(f"OUTDIR={OUTDIR}")
    print(f"Analyses={analyses}")
    print(f"Bootstraps={args.n_bootstrap}")
    print(f"Workers={args.workers}")
    print("Primary Table 2 uses RAW OOF predicted age only.")
    print("=" * 110)

    manifest = {
        "work": str(WORK),
        "results_root": str(RESULTS_ROOT),
        "output_dir": str(OUTDIR),
        "cohorts": COHORTS,
        "feature_sets": FEATURE_SETS,
        "analyses": analyses,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "workers": args.workers,
        "primary_table": "raw OOF predicted-age performance",
        "ranking": "lower MAE, lower RMSE, higher R2, higher Pearson r",
    }
    (audit_dir / "Table2_run_manifest.json").write_text(json.dumps(manifest, indent=2))

    raw_outputs, inventory, prediction_frames = run_jobs(
        analyses=analyses,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        workers=max(1, int(args.workers)),
    )

    inventory.to_csv(audit_dir / "Table2_input_inventory.csv", index=False)
    inventory.to_excel(audit_dir / "Table2_input_inventory.xlsx", index=False)

    all_publishable: dict[str, pd.DataFrame] = {}

    for analysis, df in raw_outputs.items():
        ranked = add_ranks(df)
        pooled = pooled_analysis(prediction_frames, analysis=analysis, n_bootstrap=args.n_bootstrap, seed=args.seed)

        save_table(ranked, analytic_dir / f"{analysis}_by_cohort")
        save_table(pooled, analytic_dir / f"{analysis}_pooled")

        if analysis == "oof_training_controls":
            by_label = "Table 2"
            pool_label = "Supplementary Table S2 pooled"
            by_stem = "Table2_raw_OOF_by_cohort_IEEE"
            pool_stem = "TableS2_raw_OOF_pooled_IEEE"
            across_stem = "TableS4_across_cohort_model_rank_IEEE"
        else:
            by_label = "Supplementary Table S3 full-cohort descriptive"
            pool_label = "Supplementary Table S3 pooled descriptive"
            by_stem = "TableS3_full_cohort_descriptive_IEEE"
            pool_stem = "TableS3_full_cohort_descriptive_pooled_IEEE"
            across_stem = f"{analysis}_across_cohort_model_rank"

        pub_by = make_publishable_by_cohort(ranked, by_label)
        pub_pool = make_publishable_pooled(pooled, pool_label)
        pub_across = make_publishable_across_cohort_rank(ranked, "Supplementary Table S4 across-cohort model rank")

        save_table(pub_by, publish_dir / by_stem)
        save_table(pub_pool, publish_dir / pool_stem)
        save_table(pub_across, publish_dir / across_stem)

        all_publishable[by_stem] = pub_by
        all_publishable[pool_stem] = pub_pool
        if analysis == "oof_training_controls":
            all_publishable[across_stem] = pub_across

    corrected = make_corrected_diagnostic_table()
    if not corrected.empty:
        save_table(corrected, publish_dir / "TableS5_corrected_BAG_diagnostic")
        all_publishable["TableS5_corrected_BAG_diagnostic"] = corrected

    combined_xlsx = publish_dir / "Table2_publication_tables.xlsx"
    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        wrote = False
        for name, df in all_publishable.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=name[:31], index=False)
            wrote = True
        if not wrote:
            pd.DataFrame({"message": ["No publishable rows generated."]}).to_excel(writer, sheet_name="No_rows", index=False)

    write_notes(publish_dir)

    print("\n[DONE] Outputs:")
    print("  publishable:", publish_dir)
    print("  analytic:   ", analytic_dir)
    print("  audit:      ", audit_dir)
    print("  workbook:   ", combined_xlsx)


if __name__ == "__main__":
    main()
