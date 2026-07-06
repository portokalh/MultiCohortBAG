#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Table 2 model-performance ranking using the working variable/path names.

This version preserves the working all-subjects logic from
Table2_model_performance_all_subjects.py and adds:
  - robust handling when a subset has no scoreable rows
  - controls-only sensitivity tables
  - diagnostic attempt tables
  - long file-inventory printing/saving for every candidate training/validation CSV

Primary Table 2:
  all available subjects

Supplementary Table S2:
  controls / cognitively normal subjects only, when identifiable

Metrics:
  MAE, RMSE, R2, Pearson r

Input priority per cohort/model:
  Primary all-subject Table 2:
    1) <prefix>_<feature_set>_full_cohort_predictions.csv
    2) <prefix>_<feature_set>_metadata_all_with_predictions.csv
    3) validation_figures/subject_level_validation_input.csv as fallback

  Controls-only sensitivity Table S2:
    1) <prefix>_<feature_set>_full_cohort_predictions.csv, with status labels merged
       from metadata_all_with_predictions.csv when needed
    2) <prefix>_<feature_set>_metadata_all_with_predictions.csv as fallback
    3) validation_figures/subject_level_validation_input.csv as last fallback

Controls-only uses full_cohort_predictions for the age/prediction values, then
merges status labels from metadata-enriched files. This avoids accidentally using
old metadata brain-age columns and fixes cohorts where metadata labels exist but
model prediction columns are absent.

Outputs:
  $WORK/ines/results/Table2_ModelPerformance/
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# SETTINGS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
BASE_DIR = WORK / "ines"
RESULTS_ROOT = BASE_DIR / "results"

OUTDIR = RESULTS_ROOT / "Table2_ModelPerformance"

# Figure 5 / AUC merged tables already contain harmonized cognitive-status labels.
# For controls-only sensitivity, we use these tables to define controls when the
# full-cohort prediction files do not carry diagnosis/status columns.
FIG5_MERGED_DIR = RESULTS_ROOT / "Figure5_AssociationsOnly" / "merged_tables"

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

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

# These are copied from the working all-subject Table 2 script.
AGE_TRUE_CANDIDATES = [
    "Age",
    "age",
    "chronological_age",
    "Chronological_Age",
    "true_age",
    "y_true",
    "target_age",
    "age_at_scan",
    "scan_age",
    "VISIT_AGE",
    "SUBJECT_AGE_SCREEN",
    "AGE",
]

PREDICTED_AGE_CANDIDATES = [
    "predicted_age",
    "Predicted_Age",
    "brain_age_pred",
    "BrainAgePred",
    "pred_age",
    "prediction",
    "age_pred",
    "y_pred",
    "predicted_brain_age",
    "brain_age",
    "BrainAge",
]

CBAG_CANDIDATES = [
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_global_raw_clean",
    "cBAG_global",
    "cBAG_bias_corrected",
    "cBAG_BiasCorrected",
    "cBAG",
]

BAG_CANDIDATES = [
    "BAG",
    "BAG_raw",
    "brain_age_gap",
    "brain_age_gap_raw",
    "BrainAgeGap",
    "age_gap",
    "age_gap_raw",
]

# Extra status labels only for controls-only sensitivity.
CONTROL_STATUS_CANDIDATES = [
    "cognitive_status",
    "cog_status",
    "diagnosis",
    "DX",
    "DX_bl",
    "DXCHANGE",
    "NORMCOG",
    "DEMENTED",
    "IMPNOMCI",
    "cognitive_impairment_binary",
    "cognitive_impairment",
    "impaired",
    "CN",
    "normal_control",
]

ALLOW_VALIDATION_FALLBACK = True

# Select the newest scoreable prediction table produced by the training/validation
# scripts for each cohort × feature set. This prevents Table 2 from silently using
# stale exact-name files when a newer rerun has produced updated prediction/validation
# CSVs in the same ablation directory.
USE_MOST_RECENT_MODEL_OUTPUTS = True

MIN_N = 3


# =============================================================================
# PATHS
# =============================================================================

def ablation_dir(cohort: str, feature_set: str) -> Path:
    return RESULTS_ROOT / RESULTS_DIR_MAP[cohort] / f"ablation_{feature_set}"


def full_cohort_predictions_path(cohort: str, feature_set: str) -> Path:
    prefix = PREFIX_MAP[cohort]
    return ablation_dir(cohort, feature_set) / f"{prefix}_{feature_set}_full_cohort_predictions.csv"


def metadata_all_with_predictions_path(cohort: str, feature_set: str) -> Path:
    prefix = PREFIX_MAP[cohort]
    return ablation_dir(cohort, feature_set) / f"{prefix}_{feature_set}_metadata_all_with_predictions.csv"


def validation_path(cohort: str, feature_set: str) -> Path:
    return ablation_dir(cohort, feature_set) / "validation_figures" / "subject_level_validation_input.csv"


def file_mtime_iso(path: Path) -> str:
    """Return a stable, readable modification timestamp for diagnostics."""
    try:
        return pd.to_datetime(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _dedupe_candidate_paths(candidates: list[tuple[str, Path, int]]) -> list[tuple[str, Path, int]]:
    """Deduplicate candidate files while preserving the strongest source label."""
    by_path: dict[Path, tuple[str, Path, int]] = {}
    for source_type, path, priority in candidates:
        path = Path(path)
        if path in by_path:
            old_source, old_path, old_priority = by_path[path]
            # Keep the more specific / higher-priority source label.
            if priority < old_priority:
                by_path[path] = (source_type, old_path, priority)
        else:
            by_path[path] = (source_type, path, priority)
    return list(by_path.values())


def discover_recent_prediction_files(cohort: str, feature_set: str, subset: str = "all_subjects") -> list[tuple[str, Path, int]]:
    """
    Discover prediction-like CSV/XLSX files produced by recent training/validation reruns.

    The exact canonical paths are still included, but we also scan the cohort/model
    ablation directory for newer scoreable outputs. Candidate files are later checked
    for true-age and predicted-age/BAG columns before being selected.
    """
    root = ablation_dir(cohort, feature_set)
    prefix = PREFIX_MAP[cohort]

    candidates: list[tuple[str, Path, int]] = [
        ("full_cohort_predictions", full_cohort_predictions_path(cohort, feature_set), 10),
        ("metadata_all_with_predictions", metadata_all_with_predictions_path(cohort, feature_set), 20),
    ]
    if ALLOW_VALIDATION_FALLBACK:
        candidates.append(("validation_fallback_oof_subset", validation_path(cohort, feature_set), 30))

    if not USE_MOST_RECENT_MODEL_OUTPUTS or not root.exists():
        return _dedupe_candidate_paths(candidates)

    # Conservative patterns first: these match the known outputs from the training
    # and validation scripts without accidentally selecting unrelated result tables.
    patterns = [
        f"**/{prefix}_{feature_set}_full_cohort_predictions*.csv",
        f"**/{prefix}_{feature_set}_metadata_all_with_predictions*.csv",
        "**/subject_level_validation_input*.csv",
        "**/*subject*level*validation*input*.csv",
        "**/*full_cohort*prediction*.csv",
        "**/*metadata*with*prediction*.csv",
        "**/*oof*prediction*.csv",
        "**/*out_of_fold*prediction*.csv",
        "**/*validation*prediction*.csv",
        "**/*predicted*age*.csv",
        "**/*brain*age*prediction*.csv",
        "**/*BAG*prediction*.csv",
    ]

    for pattern_i, pattern in enumerate(patterns, start=100):
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                continue
            # Do not ever use Table2 output files as new inputs.
            if "Table2_ModelPerformance" in str(path):
                continue
            source = f"recent_discovered:{pattern}"
            candidates.append((source, path, pattern_i))

    return _dedupe_candidate_paths(candidates)


def candidate_input_paths(cohort: str, feature_set: str, subset: str = "all_subjects") -> list[tuple[str, Path]]:
    """
    Return candidate prediction inputs sorted so the newest scoreable output wins.

    When USE_MOST_RECENT_MODEL_OUTPUTS=True, this scans the ablation directory and
    sorts existing candidate files by modification time descending. This is intended
    for reruns where the training/validation scripts produce newer prediction files
    than the canonical exact-name files. If no discovered files exist, the canonical
    exact-name priority is used.
    """
    candidates = discover_recent_prediction_files(cohort, feature_set, subset=subset)
    existing = [(src, path, pri) for src, path, pri in candidates if path.exists()]

    if USE_MOST_RECENT_MODEL_OUTPUTS and existing:
        existing = sorted(existing, key=lambda x: (x[1].stat().st_mtime, -x[2]), reverse=True)
    else:
        existing = sorted(existing, key=lambda x: x[2])

    return [(f"{src}; mtime={file_mtime_iso(path)}", path) for src, path, _pri in existing]

def read_table_any(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_prediction_table(path: Path) -> pd.DataFrame:
    return read_table_any(path)


# =============================================================================
# HELPERS
# =============================================================================

METRIC_DEFAULTS = {
    "n": 0,
    "mae": np.nan,
    "rmse": np.nan,
    "r2": np.nan,
    "pearson_r": np.nan,
    "pearson_p": np.nan,
    "slope_pred_vs_true": np.nan,
    "intercept_pred_vs_true": np.nan,
    "mean_error_pred_minus_true": np.nan,
    "sd_error_pred_minus_true": np.nan,
}


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
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    for c in candidates:
        nc = normalize_name(c)
        if nc in norm:
            return norm[nc]

    return None


def ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in METRIC_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    for col in ["subset", "feature_set", "cohort", "status"]:
        if col not in out.columns:
            out[col] = ""
    return out


def find_age_col(df: pd.DataFrame) -> Optional[str]:
    col = first_existing(df, AGE_TRUE_CANDIDATES)
    if col is not None:
        return col

    # Same conservative fallback as the working script.
    candidates = []
    for c in df.columns:
        low = normalize_name(c)
        if "age" not in low:
            continue
        if any(tok in low for tok in ["pred", "prediction", "brainage", "brain_age", "bag", "cbag", "gap", "delta"]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= 20 and 40 <= x.dropna().median() <= 100:
            candidates.append(c)

    if candidates:
        return candidates[0]
    return None


def find_pred_col(df: pd.DataFrame) -> Optional[str]:
    col = first_existing(df, PREDICTED_AGE_CANDIDATES)
    if col is not None:
        return col

    candidates = []
    for c in df.columns:
        low = normalize_name(c)
        if "pred" in low and "age" in low:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() >= 20 and 30 <= x.dropna().median() <= 110:
                candidates.append(c)
    return candidates[0] if candidates else None


def find_gap_col(df: pd.DataFrame) -> Optional[str]:
    col = first_existing(df, BAG_CANDIDATES)
    if col is not None:
        return col

    candidates = []
    for c in df.columns:
        low = normalize_name(c)
        if any(tok in low for tok in ["bag", "gap"]) and "cbag" not in low:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() >= 20:
                candidates.append(c)
    return candidates[0] if candidates else None


def find_cbag_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, CBAG_CANDIDATES)


def table_has_scoreable_prediction_columns(df: pd.DataFrame) -> bool:
    """Check whether a table can support age-prediction performance metrics."""
    age_col = find_age_col(df)
    if age_col is None:
        return False
    pred_col = find_pred_col(df)
    gap_col = find_gap_col(df)
    return pred_col is not None or gap_col is not None


def read_first_available(cohort: str, feature_set: str, subset: str = "all_subjects") -> tuple[Optional[pd.DataFrame], str, str]:
    """
    Read the newest scoreable candidate table for a cohort × feature-set.

    This function intentionally tries all candidate files in newest-first order,
    rather than returning the first path that merely exists. Files without true-age
    plus predicted-age/BAG columns are skipped and recorded in the printed diagnostics.
    """
    skipped = []
    for source_type, path in candidate_input_paths(cohort, feature_set, subset=subset):
        if not path.exists():
            continue
        try:
            df = read_prediction_table(path)
        except Exception as exc:
            skipped.append(f"{path.name}:read_error={exc}")
            print(f"[WARN] Could not read {path}: {exc}")
            continue

        if not table_has_scoreable_prediction_columns(df):
            skipped.append(f"{path.name}:not_scoreable")
            print(f"[SKIP not scoreable] {cohort} {feature_set} {subset}: {path}")
            continue

        if skipped:
            source_type = source_type + " ; skipped_newer_or_invalid=" + " | ".join(skipped[:8])
        return df, source_type, str(path)

    return None, "missing_or_no_scoreable_recent_input", " ; ".join(skipped[:12])

def infer_control_mask(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Identify control/CN subjects when labels are available.
    This is only used for sensitivity Table S2 and never affects primary Table 2.
    """
    # Binary impairment-like columns where 0 means control.
    for c in df.columns:
        low = normalize_name(c)
        if any(tok in low for tok in ["cognitive_impairment", "impaired", "demented", "impnomci"]):
            x = pd.to_numeric(df[c], errors="coerce")
            vals = set(x.dropna().unique())
            if x.notna().sum() > 0 and vals.issubset({0, 1, 0.0, 1.0}):
                return x.eq(0), c

    # NORMCOG: usually 1 means normal cognition.
    col = first_existing(df, ["NORMCOG", "normcog"])
    if col is not None:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() > 0:
            return x.eq(1), col

    # String status fields.
    for c in CONTROL_STATUS_CANDIDATES:
        col = first_existing(df, [c])
        if col is None:
            continue
        s = df[col].astype(str).str.strip().str.lower()
        cn = s.isin([
            "cn", "normal", "norm", "cognitively normal", "control",
            "unimpaired", "no impairment", "nc",
        ])
        if cn.sum() > 0:
            return cn, col

    # Numeric DXCHANGE fallback: 1 is commonly stable NL in ADNI.
    for c in df.columns:
        low = normalize_name(c)
        if "dxchange" in low or low == "dx":
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() > 0 and x.eq(1).sum() > 0:
                return x.eq(1), c

    return pd.Series(False, index=df.index), ""


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tmp = pd.DataFrame({
        "y_true": pd.to_numeric(y_true, errors="coerce"),
        "y_pred": pd.to_numeric(y_pred, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) < MIN_N or tmp["y_true"].nunique() < 2 or tmp["y_pred"].nunique() < 2:
        return {
            "status": "too_few_usable_values",
            "n": int(len(tmp)),
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "slope_pred_vs_true": np.nan,
            "intercept_pred_vs_true": np.nan,
            "mean_error_pred_minus_true": np.nan,
            "sd_error_pred_minus_true": np.nan,
        }

    err = tmp["y_pred"] - tmp["y_true"]
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    sse = np.sum((tmp["y_true"] - tmp["y_pred"]) ** 2)
    sst = np.sum((tmp["y_true"] - tmp["y_true"].mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else np.nan
    pearson_r, pearson_p = stats.pearsonr(tmp["y_true"], tmp["y_pred"])
    slope, intercept, *_ = stats.linregress(tmp["y_true"], tmp["y_pred"])

    return {
        "status": "ok",
        "n": int(len(tmp)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "slope_pred_vs_true": float(slope),
        "intercept_pred_vs_true": float(intercept),
        "mean_error_pred_minus_true": float(err.mean()),
        "sd_error_pred_minus_true": float(err.std(ddof=0)),
    }


def fig5_merged_path(cohort: str, feature_set: str) -> Path:
    return FIG5_MERGED_DIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv"


def make_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build conservative join keys for linking prediction rows to Figure 5 merged tables.
    We do not overwrite existing columns; helper columns are prefixed with _join_.
    """
    out = df.copy()

    def norm_rid(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        try:
            f = float(s)
            if np.isfinite(f) and f.is_integer():
                return str(int(f))
        except Exception:
            pass
        m = re.search(r"R(\d+)", s, flags=re.I)
        if m:
            return str(int(m.group(1)))
        m = re.search(r"_S_(\d+)", s, flags=re.I)
        if m:
            return str(int(m.group(1)))
        groups = re.findall(r"\d+", s)
        if groups:
            return str(int(groups[-1]))
        return np.nan

    def norm_visit(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        m = re.search(r"(?:_|-)y(\d+)", s)
        if m:
            return f"y{m.group(1)}"
        mapping = {
            "sc": "y0", "bl": "y0", "baseline": "y0", "m00": "y0", "m0": "y0", "0": "y0", "0.0": "y0",
            "m12": "y1", "12": "y1", "12.0": "y1", "y1": "y1",
            "m24": "y2", "24": "y2", "24.0": "y2", "y2": "y2",
            "m36": "y3", "36": "y3", "36.0": "y3", "y3": "y3",
            "m48": "y4", "48": "y4", "48.0": "y4", "y4": "y4",
        }
        return mapping.get(s, s)

    # Direct string keys first.
    for c in ["connectome_key", "connectome_full_key", "subject_id", "PTID", "RID", "VISCODE", "VISCODE2"]:
        hit = first_existing(out, [c])
        if hit is not None:
            out[f"_join_{normalize_name(c)}"] = out[hit].astype(str)

    # RID/visit composite if available.
    rid_col = first_existing(out, ["RID", "rid", "PTID", "ptid", "subject_id", "connectome_key", "connectome_full_key"])
    visit_col = first_existing(out, ["VISCODE2", "VISCODE", "visit", "VISIT", "connectome_key", "connectome_full_key"])
    if rid_col is not None:
        out["_join_rid_norm"] = out[rid_col].map(norm_rid)
    if visit_col is not None:
        out["_join_visit_norm"] = out[visit_col].map(norm_visit)
    if "_join_rid_norm" in out.columns and "_join_visit_norm" in out.columns:
        out["_join_rid_visit_norm"] = np.where(
            out["_join_rid_norm"].notna() & out["_join_visit_norm"].notna(),
            out["_join_rid_norm"].astype(str) + "_" + out["_join_visit_norm"].astype(str),
            np.nan,
        )

    return out



def diagnostic_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return compact diagnostics for columns that may encode cognitive/control status."""
    rows = []
    status_terms = [
        "normcog", "dement", "diagn", "dx", "group", "research", "status",
        "cog", "control", "risk", "cn", "mci", "ad", "impnomci",
    ]
    for c in df.columns:
        low = normalize_name(c)
        if not any(t in low for t in status_terms):
            continue
        s = df[c]
        vc = s.astype(str).replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan}).dropna().value_counts().head(12)
        rows.append({
            "column": c,
            "n_nonmissing": int(s.notna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            "top_values": "; ".join([f"{k}:{v}" for k, v in vc.items()]),
        })
    return pd.DataFrame(rows)


def infer_control_mask_v2(df: pd.DataFrame, cohort: str = "", feature_set: str = "") -> tuple[pd.Series, str]:
    """
    Robustly identify cognitively normal/control rows.

    Handles the enriched columns created in the Table 1 workflow:
      NORMCOG / NORMCOG_01
      DEMENTIA / DEMENTIA_01
      Research Group / DXSUM_Label / DXSUM_Label_final
      CDX_Cog
      Risk for AD_DECODE
    """
    idx = df.index

    # Explicit normal cognition binary columns: 1 means control.
    for cand in ["NORMCOG_01", "NORMCOG", "normcog"]:
        col = first_existing(df, [cand])
        if col is not None:
            x = pd.to_numeric(df[col], errors="coerce")
            if x.notna().sum() > 0 and x.eq(1).sum() > 0:
                return x.eq(1), col

    # Explicit dementia binary: 0 means not demented. Only use this if there is
    # no stricter normal-cognition label, because non-demented may include MCI.
    for cand in ["DEMENTIA_01", "DEMENTIA", "Dementia", "DEMENTED", "demented"]:
        col = first_existing(df, [cand])
        if col is not None:
            x = pd.to_numeric(df[col], errors="coerce")
            vals = set(x.dropna().unique().tolist())
            if x.notna().sum() > 0 and vals.issubset({0, 1, 0.0, 1.0}) and x.eq(0).sum() > 0:
                # Do not return immediately if a better label exists later; this is fallback.
                dementia_noncase = x.eq(0)
                dementia_source = col
                break
    else:
        dementia_noncase = None
        dementia_source = ""

    # HABS CDX_Cog: 0=CN, 1=MCI, 2=AD/Dementia.
    col = first_existing(df, ["CDX_Cog", "CDX_COG", "cdx_cog"])
    if col is not None:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() > 0 and x.eq(0).sum() > 0:
            return x.eq(0), col

    # AD_DECODE Risk: user requested Risk 0 and 1 as NORMCOG/controls.
    col = first_existing(df, ["Risk", "Risk_y", "risk_for_ad", "risk"])
    if col is not None:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() > 0 and x.isin([0, 1]).sum() > 0:
            return x.isin([0, 1]), col + " in {0,1}"

    # ADNI DXSUM labels.
    for cand in ["DXSUM_Label_final", "DXSUM_Label", "DX_bl", "DX", "Research Group", "Group", "Diagnosis", "diagnosis", "group_status", "cognitive_status"]:
        col = first_existing(df, [cand])
        if col is None:
            continue
        s = df[col].astype(str).str.strip().str.upper()
        cn = (
            s.isin(["CN", "NC", "NL", "NORMAL", "NORM", "CONTROL", "CONTROLS", "HEALTHY", "CU", "NORMCOG", "COGNITIVELY NORMAL"])
            | s.str.contains(r"\bCN\b|CONTROL|HEALTHY|NORMAL|NORMCOG|COGNITIVELY NORMAL", regex=True, na=False)
        )
        if cn.sum() > 0:
            return cn, col

    # Generic binary impairment columns: 0 means control/non-impaired.
    for c in df.columns:
        low = normalize_name(c)
        if any(tok in low for tok in ["cognitive_impairment", "impaired", "impnomci"]):
            x = pd.to_numeric(df[c], errors="coerce")
            vals = set(x.dropna().unique().tolist())
            if x.notna().sum() > 0 and vals.issubset({0, 1, 0.0, 1.0}) and x.eq(0).sum() > 0:
                return x.eq(0), c

    if dementia_noncase is not None and dementia_noncase.sum() > 0:
        return dementia_noncase, dementia_source + " == 0"

    return pd.Series(False, index=idx), ""



def metadata_status_candidate_paths(cohort: str, feature_set: str) -> list[Path]:
    """Return metadata/status-label tables newest first for controls-only merges."""
    root = ablation_dir(cohort, feature_set)
    candidates = [metadata_all_with_predictions_path(cohort, feature_set)]
    if root.exists():
        for pattern in [
            f"**/{PREFIX_MAP[cohort]}_{feature_set}_metadata_all_with_predictions*.csv",
            "**/*metadata*with*prediction*.csv",
            "**/*merged*metadata*.csv",
        ]:
            for path in root.glob(pattern):
                if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                    candidates.append(path)
    out = []
    seen = set()
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            out.append(path)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)

def merge_status_labels_from_metadata(pred_df: pd.DataFrame, cohort: str, feature_set: str) -> tuple[pd.DataFrame, str]:
    """
    Merge labels from the newest available metadata/status table.

    This uses newest-first metadata discovery rather than a single exact filename,
    so controls-only tables follow the latest training/validation rerun outputs.
    """
    pred_k = make_join_keys(pred_df)

    for meta_path in metadata_status_candidate_paths(cohort, feature_set):
        try:
            meta = read_table_any(meta_path)
        except Exception:
            continue

        # Candidate status columns worth carrying over.
        possible_status_cols = [
            c for c in meta.columns
            if any(tok in normalize_name(c) for tok in [
                "normcog", "dement", "diagn", "dx", "group", "research", "status",
                "cog", "control", "risk", "impnomci", "cdx"
            ])
        ]

        if not possible_status_cols:
            continue

        meta_k = make_join_keys(meta)

    key_candidates = [
        "_join_connectome_key",
        "_join_connectome_full_key",
        "_join_subject_id",
        "_join_rid_visit_norm",
        "_join_rid_norm",
        "_join_ptid",
        "_join_rid",
    ]

    for key in key_candidates:
        if key not in pred_k.columns or key not in meta_k.columns:
            continue
        pred_key = pred_k[key].replace("nan", np.nan)
        meta_key = meta_k[key].replace("nan", np.nan)
        if pred_key.notna().sum() == 0 or meta_key.notna().sum() == 0:
            continue

        keep_cols = [key] + [c for c in possible_status_cols if c in meta_k.columns and c not in pred_k.columns]
        if len(keep_cols) == 1:
            continue

        small = meta_k[keep_cols].dropna(subset=[key]).drop_duplicates(subset=[key], keep="first")
        test_overlap = pred_key.isin(set(small[key].astype(str))).sum()
        if test_overlap <= 0:
            continue

        out = pred_k.copy()
        out[key] = pred_key.astype(str)
        small[key] = small[key].astype(str)
        out = out.merge(small, on=key, how="left", suffixes=("", "_statusmeta"))
        return out, f"metadata_all_with_predictions status merge; key={key}; matched_rows={int(test_overlap)}; file={meta_path}"

    return pred_df, ""


def get_control_mask_from_fig5_labels(pred_df: pd.DataFrame, cohort: str, feature_set: str) -> tuple[pd.Series, str]:
    """
    Use the Figure 5/AUC merged table to identify CN/control rows, then map
    that label back to the prediction file by shared identifiers.
    """
    fig5_path = fig5_merged_path(cohort, feature_set)
    if not fig5_path.exists():
        return pd.Series(False, index=pred_df.index), ""

    try:
        fig5 = pd.read_csv(fig5_path, low_memory=False)
    except Exception:
        return pd.Series(False, index=pred_df.index), ""

    fig5_control, fig5_source = infer_control_mask_v2(fig5, cohort=cohort, feature_set=feature_set)
    if fig5_control.sum() == 0:
        fig5_control, fig5_source = infer_control_mask(fig5)
    if fig5_control.sum() == 0:
        return pd.Series(False, index=pred_df.index), ""

    pred_k = make_join_keys(pred_df)
    fig5_k = make_join_keys(fig5)

    # Try keys in order of specificity.
    key_candidates = [
        "_join_connectome_key",
        "_join_connectome_full_key",
        "_join_subject_id",
        "_join_rid_visit_norm",
        "_join_rid_norm",
        "_join_ptid",
        "_join_rid",
    ]

    for key in key_candidates:
        if key not in pred_k.columns or key not in fig5_k.columns:
            continue

        pred_key = pred_k[key].replace("nan", np.nan)
        fig_key = fig5_k[key].replace("nan", np.nan)
        valid_fig = fig_key.notna()
        valid_pred = pred_key.notna()
        if valid_fig.sum() == 0 or valid_pred.sum() == 0:
            continue

        label_map = (
            pd.DataFrame({"key": fig_key, "is_control": fig5_control})
            .dropna(subset=["key"])
            .sort_values("is_control", ascending=False)
            .drop_duplicates("key", keep="first")
        )

        mapped = pred_key.map(dict(zip(label_map["key"], label_map["is_control"])))
        if mapped.notna().sum() > 0:
            return mapped.fillna(False).astype(bool), f"Figure5:{fig5_source}; key={key}; file={fig5_path}"

    return pd.Series(False, index=pred_df.index), ""



def metrics_for_one(cohort: str, feature_set: str, subset: str = "all_subjects") -> dict:
    df, source_type, source_path = read_first_available(cohort, feature_set, subset=subset)

    row = {
        "subset": subset,
        "feature_set": feature_set,
        "cohort": cohort,
        "source_type": source_type,
        "source_path": source_path,
        "input_rows": 0 if df is None else len(df),
        "subset_rows": 0,
        "control_status_source": "",
        "true_age_col": "",
        "predicted_age_col": "",
        "gap_col_used_to_reconstruct_prediction": "",
        "cbag_col_available": "",
        "status": "missing_input" if df is None else "",
    }
    row.update(METRIC_DEFAULTS)

    if df is None:
        return row

    if subset == "controls_only":
        # For controls-only, we prefer to keep prediction values from
        # full_cohort_predictions and merge labels from metadata_all_with_predictions.
        merge_source = ""
        if source_type.startswith("full_cohort_predictions"):
            df_merged, merge_source = merge_status_labels_from_metadata(df, cohort, feature_set)
            if merge_source:
                df = df_merged
                source_type = source_type + "+metadata_status"
                row["source_type"] = source_type
                row["source_path"] = source_path + " ; " + merge_source

        control_mask, control_source = infer_control_mask_v2(df, cohort=cohort, feature_set=feature_set)

        # Backward-compatible older detector.
        if control_mask.sum() == 0:
            control_mask, control_source = infer_control_mask(df)

        # If the selected table is still missing labels, try merging labels.
        if control_mask.sum() == 0 and not source_type.startswith("full_cohort_predictions"):
            df_merged, merge_source = merge_status_labels_from_metadata(df, cohort, feature_set)
            if merge_source:
                merged_mask, merged_source = infer_control_mask_v2(df_merged, cohort=cohort, feature_set=feature_set)
                if merged_mask.sum() > 0:
                    df = df_merged
                    control_mask = merged_mask
                    control_source = merge_source + "; " + merged_source
                    row["source_type"] = source_type + "+metadata_status"
                    row["source_path"] = source_path + " ; " + merge_source

        # If still unresolved, use the same harmonized labels already used by
        # Figure 5/AUC cognitive-status analyses.
        if control_mask.sum() == 0:
            control_mask, control_source = get_control_mask_from_fig5_labels(df, cohort, feature_set)

        row["control_status_source"] = control_source
        row["subset_rows"] = int(control_mask.sum())
        if control_mask.sum() == 0:
            row["status"] = "no_controls_identified"
            diag = diagnostic_status_columns(df)
            if not diag.empty:
                row["available_status_like_columns"] = " | ".join(
                    f"{r.column} [{r.top_values}]" for r in diag.itertuples(index=False)
                )[:2000]
            return row
        df = df.loc[control_mask].copy()
    else:
        row["subset_rows"] = len(df)

    age_col = find_age_col(df)
    pred_col = find_pred_col(df)
    gap_col = find_gap_col(df)
    cbag_col = find_cbag_col(df)

    row["true_age_col"] = age_col or ""
    row["predicted_age_col"] = pred_col or ""
    row["gap_col_used_to_reconstruct_prediction"] = gap_col or ""
    row["cbag_col_available"] = cbag_col or ""

    if age_col is None:
        row["status"] = "missing_true_age"
        return row

    y_true = pd.to_numeric(df[age_col], errors="coerce")

    if pred_col is not None:
        y_pred = pd.to_numeric(df[pred_col], errors="coerce")
    elif gap_col is not None:
        y_pred = y_true + pd.to_numeric(df[gap_col], errors="coerce")
        row["predicted_age_col"] = f"{age_col}+{gap_col}"
    else:
        row["status"] = "missing_predicted_age_or_raw_BAG"
        return row

    row.update(compute_metrics(y_true, y_pred))
    return row


def rank_models(perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf = ensure_metric_columns(perf)
    ok = perf[perf["status"].eq("ok")].copy()

    if ok.empty:
        ranked_by_cohort = perf.copy()
        for c in ["rank_mae", "rank_rmse", "rank_r2", "rank_pearson_r", "mean_rank", "rank_overall_within_cohort"]:
            ranked_by_cohort[c] = np.nan
        model_summary = pd.DataFrame()
        ranked_by_metric = pd.DataFrame(columns=[
            "subset", "feature_set", "cohort", "n", "metric", "value",
            "lower_is_better", "rank_within_cohort",
        ])
        return ranked_by_cohort, model_summary, ranked_by_metric

    ranked_rows = []
    for cohort, sub in ok.groupby("cohort"):
        sub = sub.copy()
        sub["rank_mae"] = sub["mae"].rank(method="min", ascending=True)
        sub["rank_rmse"] = sub["rmse"].rank(method="min", ascending=True)
        sub["rank_r2"] = sub["r2"].rank(method="min", ascending=False)
        sub["rank_pearson_r"] = sub["pearson_r"].rank(method="min", ascending=False)
        sub["mean_rank"] = sub[["rank_mae", "rank_rmse", "rank_r2", "rank_pearson_r"]].mean(axis=1)
        sub["rank_overall_within_cohort"] = sub["mean_rank"].rank(method="min", ascending=True)
        ranked_rows.append(sub)

    ranked_by_cohort = pd.concat(ranked_rows, ignore_index=True, sort=False)

    model_summary = (
        ranked_by_cohort
        .groupby("feature_set", as_index=False)
        .agg(
            n_cohorts=("cohort", "nunique"),
            total_n=("n", "sum"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_pearson_r=("pearson_r", "mean"),
            mean_rank_mae=("rank_mae", "mean"),
            mean_rank_rmse=("rank_rmse", "mean"),
            mean_rank_r2=("rank_r2", "mean"),
            mean_rank_pearson_r=("rank_pearson_r", "mean"),
            mean_rank=("mean_rank", "mean"),
        )
        .sort_values(["mean_rank", "mean_mae"], ascending=[True, True])
    )
    model_summary["overall_model_rank"] = np.arange(1, len(model_summary) + 1)

    metric_rows = []
    metrics = [("mae", True), ("rmse", True), ("r2", False), ("pearson_r", False)]
    for metric, lower_is_better in metrics:
        sub = ok[["subset", "feature_set", "cohort", "n", metric]].copy()
        sub["metric"] = metric
        sub["value"] = sub[metric]
        sub["lower_is_better"] = lower_is_better
        sub["rank_within_cohort"] = sub.groupby("cohort")[metric].rank(
            method="min",
            ascending=lower_is_better,
        )
        metric_rows.append(sub.drop(columns=[metric]))

    ranked_by_metric = pd.concat(metric_rows, ignore_index=True, sort=False)
    return ranked_by_cohort, model_summary, ranked_by_metric


def make_global_model_rank_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank every evaluated cohort × BAG-model row globally.

    This is different from:
      - rank_overall_within_cohort: ranks models separately inside each cohort
      - pooled overall table: pools subjects across cohorts and ranks the five model specifications

    Here, all scoreable model evaluations are put into one table and ranked together.
    Rank 1 is the best global evaluation. Lower MAE and RMSE are better; higher R² and
    Pearson r are better. The main global rank is the mean of the four global metric ranks.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = ensure_metric_columns(df)
    ok = out[out["status"].eq("ok")].copy()

    if ok.empty:
        return pd.DataFrame(columns=[
            "subset", "cohort", "feature_set", "n", "mae", "rmse", "r2", "pearson_r",
            "rank_global_mae", "rank_global_rmse", "rank_global_r2", "rank_global_pearson_r",
            "global_mean_rank", "global_rank_all_evaluated", "status",
        ])

    ok["rank_global_mae"] = ok["mae"].rank(method="min", ascending=True)
    ok["rank_global_rmse"] = ok["rmse"].rank(method="min", ascending=True)
    ok["rank_global_r2"] = ok["r2"].rank(method="min", ascending=False)
    ok["rank_global_pearson_r"] = ok["pearson_r"].rank(method="min", ascending=False)
    ok["global_mean_rank"] = ok[[
        "rank_global_mae",
        "rank_global_rmse",
        "rank_global_r2",
        "rank_global_pearson_r",
    ]].mean(axis=1)
    ok["global_rank_all_evaluated"] = ok["global_mean_rank"].rank(method="min", ascending=True)

    # Stable presentation order with deterministic tie-breakers.
    ok = ok.sort_values(
        ["global_rank_all_evaluated", "global_mean_rank", "mae", "rmse", "r2", "pearson_r"],
        ascending=[True, True, True, True, False, False],
    ).copy()

    return ok


def compute_overall_pooled(subset: str) -> pd.DataFrame:
    rows = []

    for feature_set in FEATURE_SETS:
        pooled = []
        source_paths = []

        for cohort in COHORTS:
            df, source_type, source_path = read_first_available(cohort, feature_set, subset=subset)
            if df is None:
                continue

            if subset == "controls_only":
                if source_type.startswith("full_cohort_predictions"):
                    df_merged, _merge_source = merge_status_labels_from_metadata(df, cohort, feature_set)
                    if _merge_source:
                        df = df_merged
                control_mask, _ = infer_control_mask_v2(df, cohort=cohort, feature_set=feature_set)
                if control_mask.sum() == 0:
                    control_mask, _ = infer_control_mask(df)
                if control_mask.sum() == 0:
                    df_merged, _merge_source = merge_status_labels_from_metadata(df, cohort, feature_set)
                    merged_mask, _ = infer_control_mask_v2(df_merged, cohort=cohort, feature_set=feature_set)
                    if merged_mask.sum() > 0:
                        df = df_merged
                        control_mask = merged_mask
                if control_mask.sum() == 0:
                    control_mask, _ = get_control_mask_from_fig5_labels(df, cohort, feature_set)
                if control_mask.sum() == 0:
                    continue
                df = df.loc[control_mask].copy()

            age_col = find_age_col(df)
            pred_col = find_pred_col(df)
            gap_col = find_gap_col(df)
            if age_col is None:
                continue

            y_true = pd.to_numeric(df[age_col], errors="coerce")
            if pred_col is not None:
                y_pred = pd.to_numeric(df[pred_col], errors="coerce")
            elif gap_col is not None:
                y_pred = y_true + pd.to_numeric(df[gap_col], errors="coerce")
            else:
                continue

            pooled.append(pd.DataFrame({"cohort": cohort, "y_true": y_true, "y_pred": y_pred}))
            source_paths.append(f"{cohort}:{source_type}:{source_path}")

        row = {"subset": subset, "feature_set": feature_set, "cohorts_included": "", "source_paths": "; ".join(source_paths)}
        row.update(METRIC_DEFAULTS)

        if not pooled:
            row.update({"status": "no_usable_inputs"})
            rows.append(row)
            continue

        pooled_df = pd.concat(pooled, ignore_index=True, sort=False)
        row["cohorts_included"] = ",".join(sorted(pooled_df["cohort"].dropna().unique()))
        row.update(compute_metrics(pooled_df["y_true"], pooled_df["y_pred"]))
        rows.append(row)

    out = ensure_metric_columns(pd.DataFrame(rows))
    ok = out["status"].eq("ok") if "status" in out.columns else pd.Series(False, index=out.index)

    if ok.any():
        out.loc[ok, "rank_mae"] = out.loc[ok, "mae"].rank(method="min", ascending=True)
        out.loc[ok, "rank_rmse"] = out.loc[ok, "rmse"].rank(method="min", ascending=True)
        out.loc[ok, "rank_r2"] = out.loc[ok, "r2"].rank(method="min", ascending=False)
        out.loc[ok, "rank_pearson_r"] = out.loc[ok, "pearson_r"].rank(method="min", ascending=False)
        out.loc[ok, "mean_rank"] = out.loc[ok, ["rank_mae", "rank_rmse", "rank_r2", "rank_pearson_r"]].mean(axis=1)
        out.loc[ok, "overall_model_rank"] = out.loc[ok, "mean_rank"].rank(method="min", ascending=True)
    else:
        for c in ["rank_mae", "rank_rmse", "rank_r2", "rank_pearson_r", "mean_rank", "overall_model_rank"]:
            out[c] = np.nan

    return out.sort_values(["overall_model_rank", "mae"], na_position="last")


def run_subset(subset: str, prefix: str) -> dict[str, pd.DataFrame]:
    rows = []
    for feature_set in FEATURE_SETS:
        for cohort in COHORTS:
            row = metrics_for_one(cohort, feature_set, subset=subset)
            rows.append(row)
            print(
                f"[{subset} | {row.get('status', ''):>32}] {feature_set:<24} {cohort:<10} "
                f"source={row.get('source_type', ''):<32} n={row.get('n', '')}"
            )

    perf = ensure_metric_columns(pd.DataFrame(rows))
    by_cohort, model_summary, ranked_by_metric = rank_models(perf)
    global_rank = make_global_model_rank_table(by_cohort)
    overall_pooled = compute_overall_pooled(subset)

    return {
        f"{prefix}_by_cohort": by_cohort,
        f"{prefix}_global_rank_all_evaluated": global_rank,
        f"{prefix}_overall": overall_pooled,
        f"{prefix}_ranked_by_metric": ranked_by_metric,
        f"{prefix}_model_summary": model_summary,
        f"{prefix}_diagnostic_all_attempts": perf,
    }



# =============================================================================
# PUBLISHABLE TABLE FORMATTING
# =============================================================================

MODEL_LABELS = {
    "imaging_only": "Imaging only",
    "imaging_demographics": "Imaging + demographics",
    "imaging_biomarkers": "Imaging + biomarkers",
    "full": "Full model",
    "full_no_cardiovascular": "Full model without cardiovascular variables",
}

COHORT_LABELS = {
    "ADNI": "ADNI",
    "ADRC": "ADRC",
    "HABS": "HABS",
    "AD_DECODE": "AD-DECODE",
}

COHORT_ORDER_PUBLISH = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

MODEL_ORDER_PUBLISH = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]


def format_p_value(p: object) -> str:
    """Format p-values for manuscript tables."""
    try:
        p = float(p)
    except Exception:
        return ""

    if np.isnan(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_float(x: object, digits: int = 3) -> str:
    """Format numeric values consistently for manuscript tables."""
    try:
        x = float(x)
    except Exception:
        return ""

    if np.isnan(x):
        return ""
    return f"{x:.{digits}f}"


def make_publishable_by_cohort_table(
    df: pd.DataFrame,
    table_label: str,
) -> pd.DataFrame:
    """
    Create manuscript-ready by-cohort Table 2 / Supplementary Table S2.

    This table ranks the 5 BAG model specifications within each cohort.
    Therefore, Rank should be 1-5 for each cohort.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Keep only scoreable rows.
    if "status" in out.columns:
        out = out[out["status"].eq("ok")].copy()

    if out.empty:
        return pd.DataFrame()

    # Recalculate publication rank within each cohort.
    # Primary criterion: lower MAE.
    # Tie-breakers: lower RMSE, higher R2, higher Pearson r.
    out = out.sort_values(
        by=["cohort", "mae", "rmse", "r2", "pearson_r"],
        ascending=[True, True, True, False, False],
    ).copy()

    out["publication_rank_within_cohort"] = out.groupby("cohort").cumcount() + 1

    # Sort for presentation.
    out["cohort_order"] = pd.Categorical(
        out["cohort"],
        categories=COHORT_ORDER_PUBLISH,
        ordered=True,
    )

    out["model_order"] = pd.Categorical(
        out["feature_set"],
        categories=MODEL_ORDER_PUBLISH,
        ordered=True,
    )

    out = out.sort_values(
        ["cohort_order", "publication_rank_within_cohort", "model_order"]
    ).copy()

    publishable = pd.DataFrame({
        "Cohort": out["cohort"].map(COHORT_LABELS).fillna(out["cohort"]),
        "BAG model": out["feature_set"].map(MODEL_LABELS).fillna(out["feature_set"]),
        "N": out["n"].astype("Int64"),
        "MAE": out["mae"].apply(lambda x: format_float(x, 3)),
        "RMSE": out["rmse"].apply(lambda x: format_float(x, 3)),
        "R²": out["r2"].apply(lambda x: format_float(x, 3)),
        "Pearson r": out["pearson_r"].apply(lambda x: format_float(x, 3)),
        "Pearson p": out["pearson_p"].apply(format_p_value),
        "Slope": out["slope_pred_vs_true"].apply(lambda x: format_float(x, 3)),
        "Intercept": out["intercept_pred_vs_true"].apply(lambda x: format_float(x, 3)),
        "Mean error": out["mean_error_pred_minus_true"].apply(lambda x: format_float(x, 3)),
        "SD error": out["sd_error_pred_minus_true"].apply(lambda x: format_float(x, 3)),
        "Rank": out["publication_rank_within_cohort"].astype("Int64"),
    })

    publishable.insert(0, "Table", table_label)
    return publishable



def make_publishable_global_rank_table(
    df: pd.DataFrame,
    table_label: str,
) -> pd.DataFrame:
    """
    Create a manuscript-ready table that globally ranks every evaluated
    cohort × BAG-model combination.

    Rank 1 indicates the best-performing model evaluation among all scoreable
    rows across all cohorts and feature sets.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"].eq("ok")].copy()

    if out.empty:
        return pd.DataFrame()

    if "global_rank_all_evaluated" not in out.columns:
        out = make_global_model_rank_table(out)

    out = out.sort_values(
        ["global_rank_all_evaluated", "global_mean_rank", "mae", "rmse"],
        ascending=[True, True, True, True],
    ).copy()

    publishable = pd.DataFrame({
        "Table": table_label,
        "Global rank": out["global_rank_all_evaluated"].astype("Int64"),
        "Cohort": out["cohort"].map(COHORT_LABELS).fillna(out["cohort"]),
        "BAG model": out["feature_set"].map(MODEL_LABELS).fillna(out["feature_set"]),
        "N": out["n"].astype("Int64"),
        "MAE": out["mae"].apply(lambda x: format_float(x, 3)),
        "RMSE": out["rmse"].apply(lambda x: format_float(x, 3)),
        "R²": out["r2"].apply(lambda x: format_float(x, 3)),
        "Pearson r": out["pearson_r"].apply(lambda x: format_float(x, 3)),
        "Pearson p": out["pearson_p"].apply(format_p_value),
        "Global MAE rank": out["rank_global_mae"].astype("Int64"),
        "Global RMSE rank": out["rank_global_rmse"].astype("Int64"),
        "Global R² rank": out["rank_global_r2"].astype("Int64"),
        "Global Pearson r rank": out["rank_global_pearson_r"].astype("Int64"),
        "Global mean rank": out["global_mean_rank"].apply(lambda x: format_float(x, 2)),
    })

    return publishable


def make_publishable_overall_table(
    df: pd.DataFrame,
    table_label: str,
) -> pd.DataFrame:
    """
    Create manuscript-ready pooled model ranking table.

    This ranks the 5 BAG model specifications overall across pooled cohorts.
    Therefore, Rank should be 1-5.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "status" in out.columns:
        out = out[out["status"].eq("ok")].copy()

    if out.empty:
        return pd.DataFrame()

    # Recalculate overall rank across the 5 BAG model specifications.
    out = out.sort_values(
        by=["mae", "rmse", "r2", "pearson_r"],
        ascending=[True, True, False, False],
    ).copy()

    out["publication_overall_rank"] = np.arange(1, len(out) + 1)

    publishable = pd.DataFrame({
        "Table": table_label,
        "BAG model": out["feature_set"].map(MODEL_LABELS).fillna(out["feature_set"]),
        "N": out["n"].astype("Int64"),
        "MAE": out["mae"].apply(lambda x: format_float(x, 3)),
        "RMSE": out["rmse"].apply(lambda x: format_float(x, 3)),
        "R²": out["r2"].apply(lambda x: format_float(x, 3)),
        "Pearson r": out["pearson_r"].apply(lambda x: format_float(x, 3)),
        "Pearson p": out["pearson_p"].apply(format_p_value),
        "Rank": out["publication_overall_rank"].astype("Int64"),
        "Cohorts included": out["cohorts_included"] if "cohorts_included" in out.columns else "",
    })

    return publishable


def write_publishable_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write final publication-ready Table 2 and Supplementary Table S2 files.

    Outputs:
      - Table2_publishable_by_cohort.csv/xlsx/tex
      - Table2_publishable_overall.csv/xlsx/tex
      - TableS2_publishable_by_cohort.csv/xlsx/tex
      - TableS2_publishable_overall.csv/xlsx/tex
      - Table2_and_TableS2_publishable_combined.xlsx
      - Table2_TableS2_publishable_notes.txt
    """

    publish_dir = OUTDIR / "publishable"
    publish_dir.mkdir(parents=True, exist_ok=True)

    table2_by_cohort = make_publishable_by_cohort_table(
        outputs.get("Table2_model_performance_all_subjects_by_cohort"),
        table_label="Table 2",
    )

    table2_overall = make_publishable_overall_table(
        outputs.get("Table2_model_performance_all_subjects_overall"),
        table_label="Table 2 pooled",
    )

    table2_global_rank = make_publishable_global_rank_table(
        outputs.get("Table2_model_performance_all_subjects_global_rank_all_evaluated"),
        table_label="Table 2 global rank",
    )

    tables2_by_cohort = make_publishable_by_cohort_table(
        outputs.get("TableS2_model_performance_controls_only_by_cohort"),
        table_label="Supplementary Table S2",
    )

    tables2_overall = make_publishable_overall_table(
        outputs.get("TableS2_model_performance_controls_only_overall"),
        table_label="Supplementary Table S2 pooled",
    )

    tables2_global_rank = make_publishable_global_rank_table(
        outputs.get("TableS2_model_performance_controls_only_global_rank_all_evaluated"),
        table_label="Supplementary Table S2 global rank",
    )

    publishable_outputs = {
        "Table2_publishable_by_cohort": table2_by_cohort,
        "Table2_publishable_global_rank_all_evaluated": table2_global_rank,
        "Table2_publishable_overall": table2_overall,
        "TableS2_publishable_by_cohort": tables2_by_cohort,
        "TableS2_publishable_global_rank_all_evaluated": tables2_global_rank,
        "TableS2_publishable_overall": tables2_overall,
    }

    for name, table in publishable_outputs.items():
        if table is None or table.empty:
            print(f"[WARN] No publishable rows for {name}")
            continue

        csv_path = publish_dir / f"{name}.csv"
        xlsx_path = publish_dir / f"{name}.xlsx"
        tex_path = publish_dir / f"{name}.tex"

        table.to_csv(csv_path, index=False)
        table.to_excel(xlsx_path, index=False)

        latex = table.to_latex(
            index=False,
            escape=False,
            caption=name.replace("_", " "),
            label=f"tab:{name.lower()}",
        )
        tex_path.write_text(latex)

        print("[DONE]", csv_path)
        print("[DONE]", xlsx_path)
        print("[DONE]", tex_path)

    combined_xlsx = publish_dir / "Table2_and_TableS2_publishable_combined.xlsx"

    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        wrote_sheet = False
        if not table2_by_cohort.empty:
            table2_by_cohort.to_excel(writer, sheet_name="Table2_by_cohort", index=False)
            wrote_sheet = True
        if not table2_global_rank.empty:
            table2_global_rank.to_excel(writer, sheet_name="Table2_global_rank", index=False)
            wrote_sheet = True
        if not table2_overall.empty:
            table2_overall.to_excel(writer, sheet_name="Table2_pooled", index=False)
            wrote_sheet = True
        if not tables2_by_cohort.empty:
            tables2_by_cohort.to_excel(writer, sheet_name="TableS2_by_cohort", index=False)
            wrote_sheet = True
        if not tables2_global_rank.empty:
            tables2_global_rank.to_excel(writer, sheet_name="TableS2_global_rank", index=False)
            wrote_sheet = True
        if not tables2_overall.empty:
            tables2_overall.to_excel(writer, sheet_name="TableS2_pooled", index=False)
            wrote_sheet = True
        if not wrote_sheet:
            pd.DataFrame({"message": ["No publishable rows were generated."]}).to_excel(
                writer,
                sheet_name="No_rows",
                index=False,
            )

    print("[DONE]", combined_xlsx)

    notes_path = publish_dir / "Table2_TableS2_publishable_notes.txt"
    notes_path.write_text(
        "Table 2 note. BAG = brain age gap. Primary Table 2 includes all available subjects. "
        "The global-rank table ranks every evaluated cohort × BAG-model row together, with rank 1 indicating "
        "the best-performing evaluation across all scoreable rows. "
        "Models were ranked within each cohort from 1 to 5, with rank 1 indicating the best-performing "
        "model. Ranking was based primarily on lower MAE, with RMSE, R², and Pearson r used as "
        "secondary criteria. Pooled rankings were computed across the five BAG model specifications.\n\n"
        "Supplementary Table S2 note. Supplementary Table S2 repeats the model-performance analysis "
        "in controls/cognitively normal subjects only. Controls were identified using available "
        "harmonized cognitive-status labels where present. Models were ranked within each cohort "
        "from 1 to 5, with rank 1 indicating the best-performing model.\n"
    )
    print("[DONE]", notes_path)




# =============================================================================
# FILE INVENTORY / LONG LOGGING
# =============================================================================

AUDIT_EXTRA_RELATIVE_FILES = [
    # Training outputs used directly or useful for provenance checks.
    "{prefix}_{feature_set}_cv_oof_predictions.csv",
    "{prefix}_{feature_set}_full_cohort_predictions.csv",
    "{prefix}_{feature_set}_metadata_with_cv_predictions.csv",
    "{prefix}_{feature_set}_metadata_all_with_predictions.csv",
    "{prefix}_{feature_set}_cv_summary_metrics.csv",
    "{prefix}_{feature_set}_bootstrap_metric_summary.csv",
    "{prefix}_{feature_set}_residual_age_dependence.csv",
    "{prefix}_{feature_set}_full_cohort_prediction_summary.csv",
    "{prefix}_{feature_set}_training_compute_report.csv",

    # Validation outputs; these are not the primary Table 2 inputs unless used as fallback,
    # but they are printed/saved so you can confirm the most recent validation products.
    "validation_figures/subject_level_validation_input.csv",
    "validation_figures_oof/subject_level_validation_input.csv",
    "validation_figures_full_cohort/subject_level_validation_input.csv",
    "validation_figures/validation_summary.csv",
    "validation_figures_oof/validation_summary.csv",
    "validation_figures_full_cohort/validation_summary.csv",
]

AUDIT_COLUMN_KEYWORDS = [
    "age", "pred", "bag", "cbag", "gap", "real_age", "true_age", "y_true", "y_pred",
    "normcog", "dement", "diagn", "dx", "group", "status", "control", "cog", "risk",
]


def file_mtime_iso(path: Path) -> str:
    """Return file modification time as ISO-like local timestamp."""
    if not path.exists():
        return ""
    try:
        return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def classify_columns(columns: Sequence[str]) -> dict:
    """Compactly classify columns relevant for Table 2/cBAG checks."""
    cols = [str(c) for c in columns]
    low = {c: c.lower() for c in cols}

    def contains_any(needles):
        return [c for c in cols if any(n in low[c] for n in needles)]

    return {
        "n_columns": len(cols),
        "true_age_candidates_found": [c for c in cols if c in AGE_TRUE_CANDIDATES or normalize_name(c) in {normalize_name(x) for x in AGE_TRUE_CANDIDATES}],
        "predicted_age_candidates_found": [c for c in cols if c in PREDICTED_AGE_CANDIDATES or normalize_name(c) in {normalize_name(x) for x in PREDICTED_AGE_CANDIDATES}],
        "bag_candidates_found": [c for c in cols if c in BAG_CANDIDATES or normalize_name(c) in {normalize_name(x) for x in BAG_CANDIDATES} or ("bag" in low[c] and "cbag" not in low[c]) or "gap" in low[c]],
        "cbag_candidates_found": [c for c in cols if c in CBAG_CANDIDATES or normalize_name(c) in {normalize_name(x) for x in CBAG_CANDIDATES} or "cbag" in low[c]],
        "status_like_columns_found": [c for c in cols if any(k in low[c] for k in ["normcog", "dement", "diagn", "dx", "group", "status", "control", "cog", "risk", "mci"])],
        "all_relevant_columns_found": [c for c in cols if any(k in low[c] for k in AUDIT_COLUMN_KEYWORDS)],
    }


def audit_one_file(cohort: str, feature_set: str, label: str, path: Path, table2_priority: int | None = None) -> dict:
    """Inspect one candidate CSV path without loading full data."""
    row = {
        "cohort": cohort,
        "feature_set": feature_set,
        "label": label,
        "table2_priority": table2_priority if table2_priority is not None else "",
        "path": str(path),
        "exists": bool(path.exists()),
        "mtime": file_mtime_iso(path),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "read_status": "missing" if not path.exists() else "",
        "n_rows_preview": "",
        "n_columns": 0,
        "true_age_candidates_found": "",
        "predicted_age_candidates_found": "",
        "bag_candidates_found": "",
        "cbag_candidates_found": "",
        "status_like_columns_found": "",
        "all_relevant_columns_found": "",
    }

    if not path.exists():
        return row

    try:
        # Read a preview only. This is fast and enough to print columns.
        preview = pd.read_csv(path, low_memory=False, nrows=5)
        cls = classify_columns(preview.columns)
        row.update({
            "read_status": "ok",
            "n_rows_preview": len(preview),
            "n_columns": cls["n_columns"],
            "true_age_candidates_found": "; ".join(cls["true_age_candidates_found"]),
            "predicted_age_candidates_found": "; ".join(cls["predicted_age_candidates_found"]),
            "bag_candidates_found": "; ".join(cls["bag_candidates_found"]),
            "cbag_candidates_found": "; ".join(cls["cbag_candidates_found"]),
            "status_like_columns_found": "; ".join(cls["status_like_columns_found"]),
            "all_relevant_columns_found": "; ".join(cls["all_relevant_columns_found"]),
        })
    except Exception as exc:
        row["read_status"] = f"read_error: {exc}"

    return row


def make_file_inventory() -> pd.DataFrame:
    """
    Build a long file inventory for every cohort × feature-set candidate file.

    This includes:
      - files the Table 2 script actually tries in priority order
      - related training outputs that should exist if training completed
      - related validation subject-level outputs, including oof and full_cohort folders
      - Figure 5 merged files used for controls-only status labels
    """
    rows = []

    for cohort in COHORTS:
        prefix = PREFIX_MAP[cohort]
        for feature_set in FEATURE_SETS:
            ab = ablation_dir(cohort, feature_set)

            # Exact Table 2 candidate inputs in the order used by metrics_for_one.
            seen_paths = set()
            for priority, (label, path) in enumerate(candidate_input_paths(cohort, feature_set, subset="all_subjects"), start=1):
                rows.append(audit_one_file(cohort, feature_set, f"TABLE2_PRIMARY_{label}", path, table2_priority=priority))
                seen_paths.add(str(path))

            # Exact controls-only candidates, including status-merge priority labels.
            for priority, (label, path) in enumerate(candidate_input_paths(cohort, feature_set, subset="controls_only"), start=1):
                key = str(path)
                label2 = f"TABLES2_CONTROLS_{label}"
                if key in seen_paths:
                    # Still add a row, because the priority/role differs.
                    pass
                rows.append(audit_one_file(cohort, feature_set, label2, path, table2_priority=priority))
                seen_paths.add(key)

            # Extra training/validation provenance CSVs.
            for rel in AUDIT_EXTRA_RELATIVE_FILES:
                rel_path = rel.format(prefix=prefix, feature_set=feature_set)
                path = ab / rel_path
                if str(path) in seen_paths:
                    continue
                rows.append(audit_one_file(cohort, feature_set, f"TRAIN_VALIDATION_EXTRA:{rel_path}", path))
                seen_paths.add(str(path))

            # Figure 5 merged controls labels used only if needed.
            fig5 = fig5_merged_path(cohort, feature_set)
            rows.append(audit_one_file(cohort, feature_set, "CONTROL_LABEL_SOURCE:Figure5 merged table", fig5))

    out = pd.DataFrame(rows)
    if not out.empty:
        out["exists_sort"] = out["exists"].astype(int)
        out = out.sort_values(["cohort", "feature_set", "exists_sort", "label", "path"], ascending=[True, True, False, True, True]).drop(columns=["exists_sort"])
    return out


def save_and_print_file_inventory() -> pd.DataFrame:
    """Print a long file audit and save it as CSV/XLSX/TXT before metric computation."""
    audit_dir = OUTDIR / "file_inventory"
    audit_dir.mkdir(parents=True, exist_ok=True)
    inventory = make_file_inventory()

    csv_path = audit_dir / "Table2_input_file_inventory_long.csv"
    xlsx_path = audit_dir / "Table2_input_file_inventory_long.xlsx"
    txt_path = audit_dir / "Table2_input_file_inventory_long.txt"
    missing_path = audit_dir / "Table2_missing_input_files.csv"
    cbag_path = audit_dir / "Table2_files_with_cBAG_or_BAG_columns.csv"

    inventory.to_csv(csv_path, index=False)
    try:
        inventory.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save Excel inventory {xlsx_path}: {exc}")

    missing = inventory[~inventory["exists"]].copy() if not inventory.empty else pd.DataFrame()
    missing.to_csv(missing_path, index=False)

    cbag_bag = inventory[
        inventory.get("cbag_candidates_found", pd.Series(dtype=str)).fillna("").ne("")
        | inventory.get("bag_candidates_found", pd.Series(dtype=str)).fillna("").ne("")
    ].copy() if not inventory.empty else pd.DataFrame()
    cbag_bag.to_csv(cbag_path, index=False)

    # Human-readable long log.
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 2 INPUT FILE INVENTORY — LONG LOG")
    lines.append("=" * 120)
    lines.append(f"WORK={WORK}")
    lines.append(f"RESULTS_ROOT={RESULTS_ROOT}")
    lines.append(f"OUTDIR={OUTDIR}")
    lines.append("")

    if inventory.empty:
        lines.append("No inventory rows were generated.")
    else:
        for row in inventory.itertuples(index=False):
            exists_flag = "FOUND" if row.exists else "MISSING"
            lines.append(f"[{exists_flag}] {row.cohort} | {row.feature_set} | {row.label}")
            lines.append(f"  path: {row.path}")
            if row.exists:
                lines.append(f"  modified: {row.mtime} | size_bytes: {row.size_bytes} | read_status: {row.read_status}")
                lines.append(f"  true_age: {row.true_age_candidates_found}")
                lines.append(f"  predicted_age: {row.predicted_age_candidates_found}")
                lines.append(f"  BAG: {row.bag_candidates_found}")
                lines.append(f"  cBAG: {row.cbag_candidates_found}")
                lines.append(f"  status/control: {row.status_like_columns_found}")
                lines.append(f"  relevant columns: {row.all_relevant_columns_found}")
            lines.append("")

    txt = "\n".join(lines)
    txt_path.write_text(txt)

    print("\n" + txt)
    print("[DONE]", csv_path)
    print("[DONE]", xlsx_path)
    print("[DONE]", txt_path)
    print("[DONE]", missing_path)
    print("[DONE]", cbag_path)

    return inventory

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FINAL TABLE 2 MODEL PERFORMANCE")
    print("=" * 100)
    print("Input root:", RESULTS_ROOT)
    print("Output:", OUTDIR)
    print("Uses working path/variable names from Table2_model_performance_all_subjects.py")
    print("Validation fallback enabled:", ALLOW_VALIDATION_FALLBACK)

    # Print and save a long audit of every candidate CSV before computing metrics.
    save_and_print_file_inventory()

    outputs = {}
    outputs.update(run_subset("all_subjects", "Table2_model_performance_all_subjects"))
    outputs.update(run_subset("controls_only", "TableS2_model_performance_controls_only"))

    # Save raw analytic outputs.
    for name, df in outputs.items():
        path = OUTDIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print("[DONE]", path)

    # Save publication-ready Table 2 and Supplementary Table S2.
    write_publishable_tables(outputs)

    readme = OUTDIR / "Table2_model_performance_README.md"
    readme.write_text(
        "# Final Table 2 model performance\n\n"
        "Primary manuscript Table 2 uses all available subjects. "
        "Supplementary Table S2 uses controls/cognitively normal subjects only.\n\n"
        "Input priority per cohort/model:\n"
        "Primary all-subject Table 2: full-cohort predictions first, then metadata-enriched "
        "predictions, then validation fallback.\n\n"
        "Controls-only Supplementary Table S2: full-cohort predictions first, with control/status "
        "labels merged from metadata-enriched predictions when needed; metadata/validation files "
        "are fallback inputs.\n\n"
        "Performance metrics are computed from predicted brain age versus chronological age. "
        "If predicted age is unavailable but raw BAG is available, predicted age is reconstructed "
        "as chronological age + BAG. Bias-corrected cBAG columns are reported when available but "
        "are not used as predicted brain age for MAE/RMSE/R²/Pearson r.\n\n"
        "Publication ranking:\n"
        "- By-cohort tables rank the five BAG model specifications within each cohort from 1 to 5.\n"
        "- Global-rank tables rank every evaluated cohort × model row together across all cohorts and feature sets.\n"
        "- Pooled tables rank the five BAG model specifications overall from 1 to 5.\n"
        "- Ranking prioritizes lower MAE, then lower RMSE, higher R², and higher Pearson r.\n"
    )
    print("[DONE]", readme)

    global_table = outputs["Table2_model_performance_all_subjects_global_rank_all_evaluated"]
    print("\nPrimary Table 2 — all subjects, global rank of all evaluated models:")
    global_cols = [
        "global_rank_all_evaluated", "cohort", "feature_set", "n", "mae", "rmse", "r2", "pearson_r",
        "rank_global_mae", "rank_global_rmse", "rank_global_r2", "rank_global_pearson_r", "status",
    ]
    printable_global_cols = [c for c in global_cols if c in global_table.columns]
    if printable_global_cols and not global_table.empty:
        print(global_table[printable_global_cols].to_string(index=False))
    else:
        print("No scoreable all-subject global rows were found. Check diagnostic_all_attempts CSVs.")

    main_table = outputs["Table2_model_performance_all_subjects_overall"]
    print("\nPrimary Table 2 — all subjects, pooled:")
    cols = [
        "overall_model_rank", "feature_set", "n", "mae", "rmse", "r2", "pearson_r",
        "rank_mae", "rank_rmse", "rank_r2", "rank_pearson_r", "cohorts_included", "status",
    ]
    printable_cols = [c for c in cols if c in main_table.columns]
    if printable_cols and not main_table.empty:
        print(main_table[printable_cols].to_string(index=False))
    else:
        print("No scoreable all-subject models were found. Check diagnostic_all_attempts CSVs.")


if __name__ == "__main__":
    main()
