#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure8_fullcohort_longitudinal_delta_cBAG_SIGNIFICANT_STRICT_ENRICHED.py
===================================================

Strict enriched-input Figure 8 script with significant-panel selection.

Purpose
-------
Build Figure 8 using scan/session-level FULL-COHORT cBAG, not OOF-only tables.

This version is designed after fixing the HABS longitudinal pipeline:
    Graph builder -> Training -> Validation full-cohort

Default cohorts:
    ADNI: y0 -> y4
    HABS: y0 -> y2

Main design
-----------
For each cohort and feature set:
    1) Read full-cohort enriched validation input if available:
        validation_figures_full_cohort/subject_level_validation_input_enriched_for_Figure4.csv

    2) Fallback to non-enriched validation input:
        validation_figures_full_cohort/subject_level_validation_input.csv

    3) Deduplicate by scan/session ID:
        H4369_y0 and H4369_y2 are kept as separate rows.
        duplicate copies of H4369_y0 are collapsed to one row.

    4) Parse longitudinal subject and visit from scan IDs.

    5) Compute within-subject deltas:
        delta cBAG = follow-up cBAG - baseline cBAG
        delta endpoint = follow-up endpoint - baseline endpoint

    6) Test associations:
        delta cBAG vs delta biological/clinical endpoints.

Outputs
-------
<BASE_DIR>/Figure8_fullcohort_longitudinal_delta_cBAG_fixed/
    Figure8_fullcohort_longitudinal_delta_cBAG_fixed.png/pdf
    Supplementary_Figure8_fullcohort_longitudinal_delta_cBAG_fixed.png/pdf
    longitudinal_delta_subject_table.csv
    longitudinal_delta_association_stats.csv
    longitudinal_delta_dataset_summary.csv
    longitudinal_endpoint_column_selection.csv
    figure8_manifest.csv

Run examples
------------
From terminal:
    python Figure8_fullcohort_longitudinal_delta_cBAG_SIGNIFICANT_STRICT_ENRICHED.py --cohorts ADNI,HABS --feature-set full

If you want HABS only:
    python Figure8_fullcohort_longitudinal_delta_cBAG_SIGNIFICANT_STRICT_ENRICHED.py --cohorts HABS --feature-set full

If you want imaging_only:
    python Figure8_fullcohort_longitudinal_delta_cBAG_SIGNIFICANT_STRICT_ENRICHED.py --cohorts ADNI,HABS --feature-set imaging_only

Notes
-----
- This script does NOT use OOF-only validation tables.
- It uses full-cohort cBAG_global/cBAG so HABS MCI/dementia/full biological validation can be included.
- If enriched Figure 4 files are missing, the script falls back to validation input, but endpoint availability may be lower.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# Defaults
# =============================================================================

RESULTS_ROOT = Path("/mnt/newStor/paros/paros_WORK/ines/results")
BASE_DIR = (
    RESULTS_ROOT
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

DEFAULT_OUTDIR_NAME = "Figure8_fullcohort_longitudinal_delta_cBAG_SIGNIFICANT_STRICT_ENRICHED"
DEFAULT_FEATURE_SET = "full"
DEFAULT_COHORTS = ["ADNI", "HABS"]

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

# Prefer full-cohort global cBAG first.
CBAG_PRIORITY = [
    "cBAG_global_raw_clean",
    "cBAG_global",
    "cBAG_raw_clean",
    "cBAG",
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_foldwise_raw_clean",
    "cBAG_foldwise",
    "BAG_raw_clean",
    "BAG",
    "BAG_raw",
]

ID_PRIORITY = [
    "Subject_ID",
    "graph_id",
    "connectome_key",
    "runno",
    "DWI",
    "subject_match",
    "regional_id",
    "subject_id",
    "match_id",
    "PTID",
    "ptid",
    "RID",
    "ID",
    "MRI_Exam",
    "Subject",
]

SENTINEL_VALUES = {
    -999999, -888888, -777777,
    -99999, -88888, -77777,
    -9999, -8888, -7777,
    -999, -888, -777,
    999, 888, 777,
    9999, 8888, 7777,
    99999, 88888, 77777,
    999999, 888888, 777777,
}


# Curated endpoint candidates.
# Exact names are tried first; conservative fallback is allowed.
ENDPOINTS: Dict[str, Dict[str, object]] = {
    "Global cognition": {
        "domain": "Clinical",
        "candidates": [
            "Global_Cognition_Composite_raw_clean",
            "Global_Cognition_Composite",
            "cognition_composite_raw_clean",
            "cognition_composite",
            "Memory_Composite_raw_clean",
            "Memory_Composite",
            "MOCA_total_corrected_raw_clean",
            "MOCA_total_corrected",
            "MOCA_total_raw_clean",
            "MOCA_total",
            "MMSE_total_raw_clean",
            "MMSE_total",
            "ADAS_total_raw_clean",
            "ADAS_total",
            "CDRSB_raw_clean",
            "CDRSB",
            "Animal_Total_raw_clean",
            "Animal_Total",
        ],
        "prefer_keywords": ["global_cognition", "cognition", "memory", "moca", "mmse", "animal", "adas", "cdr"],
        "avoid_keywords": ["diagnosis", "status", "group", "dx", "apoe", "risk"],
        "value_range": None,
    },
    "Hippocampal volume": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hc_volume_mm3_raw_clean",
            "Hc_volume_mm3",
            "Hc_volume_pct_brain_raw_clean",
            "Hc_volume_pct_brain",
            "Hippocampus_Total_pct_raw_clean",
            "Hippocampus_Total_pct",
            "Hippocampus_Total_raw_clean",
            "Hippocampus_Total",
            "HC_volume_raw_clean",
            "HC_volume",
            "Hippocampus_volume_raw_clean",
            "Hippocampus_volume",
            "Hippocampal_volume_raw_clean",
            "Hippocampal_volume",
            "Left_Hippocampus_volume",
            "Right_Hippocampus_volume",
        ],
        "prefer_keywords": ["hc_volume", "hippocampus", "hippocampal", "volume"],
        "avoid_keywords": ["fa", "rd", "adc", "cluster", "path", "efficiency", "diagnosis", "status"],
        "value_range": None,
    },
    "Hippocampal volume pct brain": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hc_volume_pct_brain_raw_clean",
            "Hc_volume_pct_brain",
            "Hippocampus_Total_pct_raw_clean",
            "Hippocampus_Total_pct",
        ],
        "prefer_keywords": ["hc_volume_pct", "pct_brain", "hippocampus", "volume"],
        "avoid_keywords": ["fa", "rd", "adc", "cluster", "path", "efficiency", "diagnosis", "status"],
        "value_range": None,
    },
    "Hippocampal FA": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hc_FA_raw_clean",
            "Hc_FA",
            "Hippocampus_FA_Mean_raw_clean",
            "Hippocampus_FA_Mean",
            "Hippocampus_FA_Total_raw_clean",
            "Hippocampus_FA_Total",
            "Hippocampal_FA_raw_clean",
            "Hippocampal_FA",
            "Left_Hippocampus_FA",
            "Right_Hippocampus_FA",
        ],
        "prefer_keywords": ["hc_fa", "hippocampus", "hippocampal", "fa"],
        "avoid_keywords": ["volume", "vol", "rd", "adc", "diagnosis", "status"],
        "value_range": (0.0, 1.0),
    },
    "Hippocampal RD": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hc_RD_raw_clean",
            "Hc_RD",
            "Hippocampus_RD_Mean_raw_clean",
            "Hippocampus_RD_Mean",
            "Hippocampal_RD_raw_clean",
            "Hippocampal_RD",
        ],
        "prefer_keywords": ["hc_rd", "hippocampus", "hippocampal", "rd"],
        "avoid_keywords": ["volume", "fa", "diagnosis", "status"],
        "value_range": None,
    },
    "Hippocampal AD": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hc_AD_raw_clean",
            "Hc_AD",
            "Hippocampus_AD_Mean_raw_clean",
            "Hippocampus_AD_Mean",
            "Hippocampal_AD_raw_clean",
            "Hippocampal_AD",
        ],
        "prefer_keywords": ["hc_ad", "hippocampus", "hippocampal", "ad"],
        "avoid_keywords": ["volume", "fa", "diagnosis", "status"],
        "value_range": None,
    },
    "Hippocampal ADC": {
        "domain": "Neuroimaging",
        "candidates": [
            "Hc_ADC_raw_clean",
            "Hc_ADC",
            "Hippocampus_ADC_Mean_raw_clean",
            "Hippocampus_ADC_Mean",
            "Hippocampal_ADC_raw_clean",
            "Hippocampal_ADC",
        ],
        "prefer_keywords": ["hc_adc", "hippocampus", "hippocampal", "adc"],
        "avoid_keywords": ["volume", "fa", "diagnosis", "status"],
        "value_range": None,
    },
    "Whole-brain volume": {
        "domain": "Neuroimaging",
        "candidates": [
            "Total_Brain_volume_raw_clean",
            "Total_Brain_volume",
            "TotalBrainVolume_raw_clean",
            "TotalBrainVolume",
            "Brain_Volume_raw_clean",
            "Brain_Volume",
            "Volume_mean_raw_clean",
            "Volume_mean",
            "Volume_median_raw_clean",
            "Volume_median",
            "GM_volume_raw_clean",
            "GM_volume",
            "WM_volume_raw_clean",
            "WM_volume",
        ],
        "prefer_keywords": ["total_brain", "brain_volume", "volume"],
        "avoid_keywords": ["hipp", "hc_", "fa", "rd", "adc", "diagnosis", "status"],
        "value_range": None,
    },
    "Global FA": {
        "domain": "Neuroimaging",
        "candidates": [
            "Total_Brain_FA_raw_clean",
            "Total_Brain_FA",
            "FA_mean_raw_clean",
            "FA_mean",
            "FA_median_raw_clean",
            "FA_median",
            "Global_FA_raw_clean",
            "Global_FA",
            "Mean_FA",
            "Median_FA",
        ],
        "prefer_keywords": ["total_brain_fa", "global_fa", "fa_mean", "fa_median"],
        "avoid_keywords": ["hipp", "hc_", "volume", "diagnosis", "status"],
        "value_range": (0.0, 1.0),
    },
    "Graph clustering": {
        "domain": "Network",
        "candidates": [
            "Total_graph_clustering_coeff_raw_clean",
            "Total_graph_clustering_coeff",
            "Hc_clustering_coeff_raw_clean",
            "Hc_clustering_coeff",
            "Clustering_Coeff_raw_clean",
            "Clustering_Coeff",
            "Clustering_Coefficient",
            "clustering_coefficient",
        ],
        "prefer_keywords": ["clustering"],
        "avoid_keywords": ["cluster_id", "diagnosis", "status"],
        "value_range": None,
    },
    "Graph path length": {
        "domain": "Network",
        "candidates": [
            "Total_graph_path_length_raw_clean",
            "Total_graph_path_length",
            "Hc_path_length_raw_clean",
            "Hc_path_length",
            "Path_Length_raw_clean",
            "Path_Length",
            "Characteristic_Path_Length",
            "characteristic_path_length",
        ],
        "prefer_keywords": ["path_length", "path"],
        "avoid_keywords": ["diagnosis", "status"],
        "value_range": None,
    },
    "Global efficiency": {
        "domain": "Network",
        "candidates": [
            "Global_Efficiency_raw_clean",
            "Global_Efficiency",
            "global_efficiency",
        ],
        "prefer_keywords": ["global_efficiency", "efficiency"],
        "avoid_keywords": ["local", "diagnosis", "status"],
        "value_range": None,
    },
    "Local efficiency": {
        "domain": "Network",
        "candidates": [
            "Local_Efficiency_raw_clean",
            "Local_Efficiency",
            "local_efficiency",
        ],
        "prefer_keywords": ["local_efficiency"],
        "avoid_keywords": ["global", "diagnosis", "status"],
        "value_range": None,
    },
    "BMI": {
        "domain": "Vascular/metabolic",
        "candidates": [
            "BMI_raw_clean",
            "BMI",
            "bmi_raw_clean",
            "bmi",
            "OM_BMI_raw_clean",
            "OM_BMI",
            "BMI_calculated_raw_clean",
            "BMI_calculated",
            "PHC_BMI_raw_clean",
            "PHC_BMI",
        ],
        "prefer_keywords": ["bmi"],
        "avoid_keywords": ["zscore", "status", "binary", "diagnosis", "dx"],
        "value_range": (10.0, 80.0),
    },
    "Systolic BP": {
        "domain": "Vascular/metabolic",
        "candidates": [
            "bp_sys_raw_clean",
            "bp_sys",
            "OM_BP1_SYS_raw_clean",
            "OM_BP1_SYS",
            "OM_BP2_SYS_raw_clean",
            "OM_BP2_SYS",
            "VSBPSYS_raw_clean",
            "VSBPSYS",
        ],
        "prefer_keywords": ["bp_sys", "systolic", "bpsys"],
        "avoid_keywords": ["status", "diagnosis"],
        "value_range": (60.0, 260.0),
    },
    "Diastolic BP": {
        "domain": "Vascular/metabolic",
        "candidates": [
            "bp_dia_raw_clean",
            "bp_dia",
            "OM_BP1_DIA_raw_clean",
            "OM_BP1_DIA",
            "OM_BP2_DIA_raw_clean",
            "OM_BP2_DIA",
            "VSBPDIA_raw_clean",
            "VSBPDIA",
        ],
        "prefer_keywords": ["bp_dia", "diastolic", "bpdia"],
        "avoid_keywords": ["status", "diagnosis"],
        "value_range": (30.0, 160.0),
    },
    "Amyloid / Aβ": {
        "domain": "Biomarker",
        "candidates": [
            "amyloid_42_raw_clean",
            "amyloid_42",
            "Abeta42_raw_clean",
            "Abeta42",
            "ABETA42_raw_clean",
            "ABETA42",
            "amyloid_40_raw_clean",
            "amyloid_40",
            "Centiloid_raw_clean",
            "Centiloid",
        ],
        "prefer_keywords": ["amyloid", "abeta", "centiloid"],
        "avoid_keywords": ["status", "positive", "binary", "diagnosis", "dx"],
        "value_range": None,
    },
    "pTau": {
        "domain": "Biomarker",
        "candidates": [
            "ptau217_raw_clean",
            "PTAU217_raw_clean",
            "pTau217_raw_clean",
            "ptau181_raw_clean",
            "PTAU181_raw_clean",
            "pTau181_raw_clean",
            "PTAU_raw_clean",
            "pTau_raw_clean",
            "ptau217",
            "PTAU217",
            "pTau217",
            "PTAU",
        ],
        "prefer_keywords": ["ptau", "tau"],
        "avoid_keywords": ["status", "positive", "binary", "diagnosis", "dx"],
        "value_range": None,
    },
    "GFAP": {
        "domain": "Biomarker",
        "candidates": [
            "GFAP_raw_clean",
            "gfap_raw_clean",
            "PLASMA_GFAP_raw_clean",
            "GFAP",
            "gfap",
            "PLASMA_GFAP",
        ],
        "prefer_keywords": ["gfap"],
        "avoid_keywords": ["status", "positive", "binary", "diagnosis", "dx"],
        "value_range": None,
    },
    "NfL": {
        "domain": "Biomarker",
        "candidates": [
            "NFL_raw_clean",
            "NfL_raw_clean",
            "nfl_raw_clean",
            "PLASMA_NFL_raw_clean",
            "NFL",
            "NfL",
            "nfl",
        ],
        "prefer_keywords": ["nfl", "neurofilament"],
        "avoid_keywords": ["status", "positive", "binary", "diagnosis", "dx"],
        "value_range": None,
    },
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fixed full-cohort Figure 8 longitudinal delta-cBAG figure.")
    p.add_argument("--results-root", default=str(RESULTS_ROOT))
    p.add_argument("--base-dir", default=str(BASE_DIR))
    p.add_argument("--outdir", default=None)
    p.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    p.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    p.add_argument("--formats", default="png,pdf")
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--min-n", type=int, default=12)
    p.add_argument("--corr-method", default="pearson", choices=["pearson", "spearman"])
    p.add_argument("--max-endpoints-main", type=int, default=8)
    p.add_argument("--max-endpoints-supp", type=int, default=30)
    p.add_argument("--input-kind", default="enriched", choices=["enriched"], help="Strict mode: read only subject_level_validation_input_enriched_for_Figure4.csv. No fallback.")
    p.add_argument("--main-alpha", type=float, default=0.05, help="Alpha threshold for selecting significant main Figure 8 panels.")
    p.add_argument("--main-title", default="Figure 8. Longitudinal full-cohort cBAG and within-person biological change")
    p.add_argument("--supp-title", default="Supplementary Figure 8. Extended longitudinal full-cohort cBAG associations")
    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def clean_numeric(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").copy()
    for val in SENTINEL_VALUES:
        x = x.mask(x == val, np.nan)
    return x.replace([np.inf, -np.inf], np.nan)


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def parse_longitudinal_id(x) -> Tuple[str, Optional[float]]:
    s = str(x).strip()
    s = s.replace("_Y", "_y")
    s = re.sub(r"\.csv$", "", s, flags=re.I)
    s = re.sub(r"_conn_plain$", "", s, flags=re.I)
    m = re.match(r"^(.+?)_y(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), float(m.group(2))
    m = re.search(r"([A-Za-z]+\d+)_y(\d+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), float(m.group(2))
    return s.upper(), None


def normalize_scan_key(x) -> str:
    return (
        str(x)
        .strip()
        .replace("_Y", "_y")
        .lower()
    )


def nice_label(x: str) -> str:
    mapping = {
        "ADNI": "ADNI",
        "HABS": "HABS",
        "cBAG": "cBAG",
        "delta_cbag": "ΔcBAG",
        "Global cognition": "Cognition",
        "Hippocampal volume": "Hippocampal volume",
        "Hippocampal volume pct brain": "Hippocampal volume (% brain)",
        "Hippocampal FA": "Hippocampal FA",
        "Graph clustering": "Graph clustering",
        "Graph path length": "Graph path length",
        "Global efficiency": "Global efficiency",
        "Local efficiency": "Local efficiency",
        "Whole-brain volume": "Whole-brain volume",
        "Global FA": "Global FA",
        "BMI": "BMI",
        "Systolic BP": "Systolic BP",
        "Diastolic BP": "Diastolic BP",
    }
    if x in mapping:
        return mapping[x]
    out = str(x).replace("_raw_clean", "").replace("_", " ")
    out = out.replace("cBAG global", "cBAG")
    return re.sub(r"\s+", " ", out).strip()


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")


def save_all(fig: plt.Figure, outbase: Path, formats: Sequence[str], dpi: int) -> List[str]:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        path = outbase.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
    plt.close(fig)
    return saved


def fisher_ci_for_r(r: float, n: int) -> Tuple[float, float]:
    if pd.isna(r) or n <= 3:
        return np.nan, np.nan
    rr = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(rr)
    se = 1.0 / math.sqrt(n - 3)
    lo = np.tanh(z - 1.96 * se)
    hi = np.tanh(z + 1.96 * se)
    return float(lo), float(hi)


def corr_xy(x: pd.Series, y: pd.Series, method: str = "pearson") -> Dict[str, float]:
    xx = clean_numeric(x)
    yy = clean_numeric(y)
    ok = xx.notna() & yy.notna()
    n = int(ok.sum())
    if n < 3 or xx[ok].nunique() < 2 or yy[ok].nunique() < 2:
        return {"n": n, "r": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if method == "spearman":
        r, p = spearmanr(xx[ok], yy[ok])
    else:
        r, p = pearsonr(xx[ok], yy[ok])
    lo, hi = fisher_ci_for_r(float(r), n)
    return {"n": n, "r": float(r), "p": float(p), "ci_low": lo, "ci_high": hi}


def candidate_tokens(candidates: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for c in candidates:
        tokens.extend([t for t in re.split(r"[_\s/\-]+", str(c).lower()) if len(t) >= 3])
    return sorted(set(tokens), key=len, reverse=True)


def choose_endpoint_column(
    df: pd.DataFrame,
    endpoint: str,
    spec: Dict[str, object],
    min_n_visits: int = 12,
) -> Tuple[Optional[str], str, int]:
    candidates = list(spec.get("candidates", []))  # type: ignore[arg-type]
    prefer = [str(x).lower() for x in spec.get("prefer_keywords", [])]  # type: ignore[union-attr]
    avoid = [str(x).lower() for x in spec.get("avoid_keywords", [])]  # type: ignore[union-attr]
    value_range = spec.get("value_range", None)

    def usable(col: str) -> int:
        if col not in df.columns:
            return 0
        low = col.lower()
        if any(bad in low for bad in avoid):
            return 0
        x = clean_numeric(df[col])
        if value_range is not None:
            lo, hi = value_range  # type: ignore[misc]
            x = x.where((x >= lo) & (x <= hi))
        return int(x.notna().sum())

    expanded: List[str] = []
    for c in candidates:
        expanded.extend([c, f"{c}_raw_clean"])

    for col in expanded:
        n = usable(col)
        if n >= min_n_visits:
            return col, "exact", n

    # conservative fallback
    tokens = candidate_tokens(candidates)
    hits: List[Tuple[int, int, str]] = []
    for col in df.columns:
        low = col.lower()
        if any(bad in low for bad in avoid):
            continue
        n = usable(col)
        if n < min_n_visits:
            continue
        token_hit = any(tok in low for tok in tokens)
        prefer_score = sum(tok in low for tok in prefer)
        if token_hit or prefer_score > 0:
            hits.append((2 * prefer_score + int(token_hit), n, col))

    if hits:
        hits.sort(key=lambda x: (-x[0], -x[1], x[2].lower()))
        return hits[0][2], "fallback_keyword", hits[0][1]

    best_col, best_n = None, 0
    for col in expanded:
        n = usable(col)
        if n > best_n:
            best_col, best_n = col, n
    if best_col:
        return best_col, "insufficient_n", best_n

    return None, "missing", 0


def cohort_validation_dir(results_root: Path, cohort: str, feature_set: str) -> Path:
    return (
        results_root
        / RESULTS_DIR_MAP[cohort]
        / f"ablation_{feature_set}"
        / "validation_figures_full_cohort"
    )


def choose_input_path(results_root: Path, cohort: str, feature_set: str, input_kind: str) -> Tuple[Optional[Path], str]:
    """
    Strict input policy for Figure 8.

    This script reads ONLY the Figure 4 enriched full-cohort validation file:
        validation_figures_full_cohort/subject_level_validation_input_enriched_for_Figure4.csv

    It does not fall back to subject_level_validation_input.csv, because fallback can silently mix
    enriched and non-enriched inputs and produce misleading panels.
    """
    vdir = cohort_validation_dir(results_root, cohort, feature_set)
    enriched = vdir / "subject_level_validation_input_enriched_for_Figure4.csv"

    if enriched.exists():
        return enriched, "enriched"

    raise FileNotFoundError(
        f"Required enriched Figure 4 input is missing for {cohort} {feature_set}:
"
        f"{enriched}

"
        "Run Figure4_enrich_fullcohort_allcohorts_HABSfixed.py first."
    )



def deduplicate_by_scan(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_scan_key_norm"] = out[id_col].map(normalize_scan_key)

    priority_cols = [
        "cBAG_global",
        "cBAG",
        "BAG",
        "Predicted_Age_BiasCorrected",
        "Predicted_Age_RAW",
        "Hc_volume_mm3",
        "Hc_volume_pct_brain",
        "Hc_FA",
        "Total_Brain_volume",
        "Total_Brain_FA",
        "OM_BMI",
        "BMI",
        "bmi",
    ]

    out["_dedup_score"] = 0
    for c in priority_cols:
        if c in out.columns:
            out["_dedup_score"] += out[c].notna().astype(int)

    out = (
        out.sort_values(["_scan_key_norm", "_dedup_score"], ascending=[True, False])
        .drop_duplicates(subset=["_scan_key_norm"], keep="first")
        .drop(columns=["_scan_key_norm", "_dedup_score"], errors="ignore")
        .reset_index(drop=True)
    )
    return out


def load_fullcohort_longitudinal_tables(
    results_root: Path,
    cohorts: Sequence[str],
    feature_set: str,
    input_kind: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summary_rows = []

    for cohort in cohorts:
        path, source_kind = choose_input_path(results_root, cohort, feature_set, input_kind=input_kind)
        row = {
            "cohort": cohort,
            "feature_set": feature_set,
            "input_kind_requested": input_kind,
            "input_kind_used": source_kind,
            "path": str(path) if path else "",
            "exists": bool(path and path.exists()),
            "rows_before_dedup": 0,
            "rows_after_dedup": 0,
            "unique_subjects": 0,
            "subjects_with_ge2_visits": 0,
            "longitudinal_rows": 0,
            "id_col": None,
            "cbag_col": None,
            "duplicate_scan_ids_removed": 0,
            "status": "missing_file",
        }

        if path is None or not path.exists():
            summary_rows.append(row)
            continue

        df = pd.read_csv(path, low_memory=False)
        id_col = first_existing(df, ID_PRIORITY)
        cbag_col = first_existing(df, CBAG_PRIORITY)

        row["rows_before_dedup"] = len(df)
        row["id_col"] = id_col
        row["cbag_col"] = cbag_col

        if id_col is None:
            row["status"] = "missing_id_col"
            summary_rows.append(row)
            continue
        if cbag_col is None:
            row["status"] = "missing_cbag_col"
            summary_rows.append(row)
            continue

        n_before = len(df)
        df = deduplicate_by_scan(df, id_col)
        n_after = len(df)
        row["rows_after_dedup"] = n_after
        row["duplicate_scan_ids_removed"] = n_before - n_after

        parsed = df[id_col].apply(parse_longitudinal_id)
        df = df.copy()
        df["cohort"] = cohort
        df["feature_set"] = feature_set
        df["_source_path"] = str(path)
        df["_input_kind_used"] = source_kind
        df["_id_col"] = id_col
        df["_cbag_col"] = cbag_col
        df["_longitudinal_subject"] = parsed.apply(lambda z: z[0])
        df["_visit"] = parsed.apply(lambda z: z[1])
        df["_cbag"] = clean_numeric(df[cbag_col])

        visit_ok = df["_visit"].notna()
        cbag_ok = df["_cbag"].notna()
        visits = df.loc[visit_ok, :].groupby("_longitudinal_subject")["_visit"].nunique()
        repeated = visits[visits >= 2].index

        row["unique_subjects"] = int(df.loc[visit_ok, "_longitudinal_subject"].nunique())
        row["subjects_with_ge2_visits"] = int(len(repeated))
        row["longitudinal_rows"] = int(df["_longitudinal_subject"].isin(repeated).sum())
        row["status"] = "ok"

        summary_rows.append(row)
        frames.append(df.loc[visit_ok & cbag_ok].copy())

    if frames:
        out = pd.concat(frames, ignore_index=True, sort=False)
    else:
        out = pd.DataFrame()

    return out, pd.DataFrame(summary_rows)


def build_delta_table(long_df: pd.DataFrame, endpoint_cols: Dict[str, str]) -> pd.DataFrame:
    rows = []

    needed_cols = ["_cbag"] + [c for c in endpoint_cols.values() if c is not None]
    for col in needed_cols:
        if col in long_df.columns:
            long_df[f"_num_{col}"] = clean_numeric(long_df[col])

    for (cohort, subject), g in long_df.groupby(["cohort", "_longitudinal_subject"], sort=False):
        g = g.dropna(subset=["_visit"]).copy()
        if len(g) < 2:
            continue
        g = g.sort_values("_visit")
        first = g.iloc[0]
        last = g.iloc[-1]

        row = {
            "cohort": cohort,
            "longitudinal_subject": subject,
            "baseline_visit": first["_visit"],
            "followup_visit": last["_visit"],
            "delta_visit": last["_visit"] - first["_visit"],
            "n_visits": int(len(g)),
            "baseline_cbag": first["_cbag"],
            "followup_cbag": last["_cbag"],
            "delta_cbag": last["_cbag"] - first["_cbag"],
            "cbag_col": first["_cbag_col"],
            "input_kind_used": first["_input_kind_used"],
        }

        for endpoint, col in endpoint_cols.items():
            if col is None or col not in long_df.columns:
                row[f"{endpoint}__column"] = col
                row[f"baseline__{endpoint}"] = np.nan
                row[f"followup__{endpoint}"] = np.nan
                row[f"delta__{endpoint}"] = np.nan
                continue

            num_col = f"_num_{col}"
            row[f"{endpoint}__column"] = col
            row[f"baseline__{endpoint}"] = first.get(num_col, np.nan)
            row[f"followup__{endpoint}"] = last.get(num_col, np.nan)
            row[f"delta__{endpoint}"] = last.get(num_col, np.nan) - first.get(num_col, np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_delta_associations(delta_df: pd.DataFrame, endpoint_cols: Dict[str, str], min_n: int, method: str) -> pd.DataFrame:
    rows = []

    for endpoint, col in endpoint_cols.items():
        delta_col = f"delta__{endpoint}"
        spec = ENDPOINTS[endpoint]
        domain = str(spec.get("domain", ""))

        for cohort in sorted(delta_df["cohort"].dropna().unique()):
            sub = delta_df[delta_df["cohort"] == cohort].copy()
            if delta_col not in sub.columns:
                rows.append({
                    "endpoint": endpoint,
                    "domain": domain,
                    "cohort": cohort,
                    "column": col,
                    "delta_column": delta_col,
                    "n": 0,
                    "r": np.nan,
                    "p": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "status": "missing_delta_column",
                })
                continue
            stats = corr_xy(sub["delta_cbag"], sub[delta_col], method=method)
            status = "ok" if stats["n"] >= min_n and np.isfinite(stats["r"]) else "insufficient_n_or_variance"
            rows.append({
                "endpoint": endpoint,
                "domain": domain,
                "cohort": cohort,
                "column": col,
                "delta_column": delta_col,
                **stats,
                "status": status,
            })

        if delta_col in delta_df.columns:
            stats = corr_xy(delta_df["delta_cbag"], delta_df[delta_col], method=method)
            status = "ok" if stats["n"] >= min_n and np.isfinite(stats["r"]) else "insufficient_n_or_variance"
        else:
            stats = {"n": 0, "r": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            status = "missing_delta_column"

        rows.append({
            "endpoint": endpoint,
            "domain": domain,
            "cohort": "Pooled",
            "column": col,
            "delta_column": delta_col,
            **stats,
            "status": status,
        })

    out = pd.DataFrame(rows)
    out["abs_r"] = out["r"].abs()
    return out


def significant_endpoint_rank(stats: pd.DataFrame, alpha: float, min_n: int) -> pd.DataFrame:
    """
    Rank endpoints for main Figure 8 panels.

    Priority:
      1. Pooled significant endpoints, p < alpha.
      2. Cohort-specific significant endpoints, p < alpha.
      3. Non-significant but valid endpoints, ranked by p then |r|.

    This keeps the selection transparent: the chosen endpoints are still written to
    longitudinal_delta_association_stats.csv and figure8_selected_main_panels.csv.
    """
    s = stats.copy()
    s = s[(s["status"] == "ok") & (s["n"] >= min_n) & s["p"].notna() & s["r"].notna()].copy()
    if s.empty:
        return s

    s["is_pooled"] = s["cohort"].eq("Pooled")
    s["is_significant"] = s["p"] < alpha
    s["abs_r"] = s["r"].abs()

    # rank class: pooled significant, cohort significant, pooled non-significant, cohort non-significant
    s["rank_class"] = np.select(
        [
            s["is_pooled"] & s["is_significant"],
            (~s["is_pooled"]) & s["is_significant"],
            s["is_pooled"] & (~s["is_significant"]),
        ],
        [0, 1, 2],
        default=3,
    )

    s = s.sort_values(["rank_class", "p", "abs_r", "n"], ascending=[True, True, False, False])
    return s


def endpoint_has_complete_pairs(delta_df: pd.DataFrame, endpoint: str) -> bool:
    col = f"delta__{endpoint}"
    if col not in delta_df.columns:
        return False
    x = clean_numeric(delta_df["delta_cbag"])
    y = clean_numeric(delta_df[col])
    ok = x.notna() & y.notna()
    return bool(ok.sum() >= 3 and x[ok].nunique() >= 2 and y[ok].nunique() >= 2)



def select_main_endpoints(stats: pd.DataFrame, max_endpoints: int, alpha: float = 0.05, min_n: int = 12) -> List[str]:
    ranked = significant_endpoint_rank(stats, alpha=alpha, min_n=min_n)

    endpoints: List[str] = []
    for e in ranked["endpoint"].tolist() if not ranked.empty else []:
        if e not in endpoints:
            endpoints.append(e)
        if len(endpoints) >= max_endpoints:
            break

    return endpoints[:max_endpoints]



# =============================================================================
# Plotting
# =============================================================================

def scatter_delta(ax: plt.Axes, df: pd.DataFrame, endpoint: str, stats: pd.DataFrame, title: Optional[str] = None) -> None:
    delta_col = f"delta__{endpoint}"
    if delta_col not in df.columns:
        ax.text(0.5, 0.5, f"No {endpoint} delta", ha="center", va="center")
        ax.axis("off")
        return

    plotted = False
    for cohort in sorted(df["cohort"].dropna().unique()):
        sub = df[df["cohort"] == cohort]
        x = clean_numeric(sub["delta_cbag"])
        y = clean_numeric(sub[delta_col])
        ok = x.notna() & y.notna()
        if ok.sum() == 0:
            continue
        ax.scatter(x[ok], y[ok], s=24, alpha=0.72, edgecolor="black", linewidth=0.25, label=cohort)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, f"No complete pairs for {endpoint}", ha="center", va="center")
        ax.axis("off")
        return

    # Use a truthful stats label:
    # - "Pooled" only if more than one cohort contributes complete pairs.
    # - Otherwise label the single contributing cohort, e.g. ADNI or HABS.
    contributing = []
    for cohort in sorted(df["cohort"].dropna().unique()):
        sub = df[df["cohort"] == cohort]
        x_sub = clean_numeric(sub["delta_cbag"])
        y_sub = clean_numeric(sub[delta_col])
        ok_sub = x_sub.notna() & y_sub.notna()
        if ok_sub.sum() >= 3 and x_sub[ok_sub].nunique() >= 2 and y_sub[ok_sub].nunique() >= 2:
            contributing.append(cohort)

    label_cohort = "Pooled" if len(contributing) > 1 else (contributing[0] if contributing else "Pooled")
    label_row = stats[(stats["endpoint"] == endpoint) & (stats["cohort"] == label_cohort)]

    # Fallback if the label row is unexpectedly unavailable.
    if label_row.empty:
        label_row = stats[(stats["endpoint"] == endpoint) & (stats["cohort"] == "Pooled")]

    label = ""
    if not label_row.empty and pd.notna(label_row.iloc[0]["r"]):
        r = label_row.iloc[0]["r"]
        p = label_row.iloc[0]["p"]
        n = int(label_row.iloc[0]["n"])
        label = f"{label_cohort} n={n}, r={r:.2f}, p={p:.2g}"

    # Trendline pooled
    x_all = clean_numeric(df["delta_cbag"])
    y_all = clean_numeric(df[delta_col])
    ok = x_all.notna() & y_all.notna()
    if ok.sum() >= 3 and x_all[ok].nunique() >= 2:
        coef = np.polyfit(x_all[ok], y_all[ok], 1)
        xx = np.linspace(float(x_all[ok].min()), float(x_all[ok].max()), 100)
        ax.plot(xx, coef[0] * xx + coef[1], linestyle="--", linewidth=1.2)

    ax.axvline(0, linewidth=0.8, alpha=0.6)
    ax.axhline(0, linewidth=0.8, alpha=0.6)
    ax.set_xlabel("ΔcBAG")
    ax.set_ylabel(f"Δ{nice_label(endpoint)}")
    ax.set_title(title or f"ΔcBAG vs Δ{nice_label(endpoint)}\n{label}", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, frameon=False)


def plot_delta_cbag_distribution(ax: plt.Axes, df: pd.DataFrame) -> None:
    cohorts = sorted(df["cohort"].dropna().unique())
    data = [clean_numeric(df.loc[df["cohort"] == c, "delta_cbag"]).dropna().values for c in cohorts]
    data = [x for x in data if len(x) > 0]
    labels = [c for c in cohorts if clean_numeric(df.loc[df["cohort"] == c, "delta_cbag"]).dropna().shape[0] > 0]

    if not data:
        ax.text(0.5, 0.5, "No ΔcBAG values", ha="center", va="center")
        ax.axis("off")
        return

    ax.boxplot(data, labels=labels, showfliers=False)
    for i, vals in enumerate(data, start=1):
        rng = np.random.default_rng(1000 + i)
        jitter = rng.normal(i, 0.045, size=len(vals))
        ax.scatter(jitter, vals, s=12, alpha=0.35, edgecolor="none")
        ax.text(i, ax.get_ylim()[0], f"n={len(vals)}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, linewidth=0.9)
    ax.set_ylabel("ΔcBAG")
    ax.set_title("Within-person ΔcBAG by cohort")
    ax.grid(True, axis="y", alpha=0.25)


def plot_forest(ax: plt.Axes, stats: pd.DataFrame, endpoints: Sequence[str]) -> None:
    rows = stats[(stats["endpoint"].isin(endpoints)) & (stats["cohort"].isin(["ADNI", "HABS", "Pooled"]))].copy()
    if rows.empty:
        ax.text(0.5, 0.5, "No longitudinal associations", ha="center", va="center")
        ax.axis("off")
        return

    y_positions = np.arange(len(endpoints))
    offsets = {"ADNI": -0.22, "HABS": 0.0, "Pooled": 0.22}
    labels_used = set()

    for i, endpoint in enumerate(endpoints):
        for cohort in ["ADNI", "HABS", "Pooled"]:
            row = rows[(rows["endpoint"] == endpoint) & (rows["cohort"] == cohort)]
            if row.empty:
                continue
            r = row.iloc[0]["r"]
            n = int(row.iloc[0]["n"])
            lo = row.iloc[0]["ci_low"]
            hi = row.iloc[0]["ci_high"]
            y = i + offsets.get(cohort, 0.0)
            if pd.isna(r):
                ax.scatter([0], [y], marker="x", s=30, color="lightgray", label=None)
                continue
            if pd.notna(lo) and pd.notna(hi):
                ax.plot([lo, hi], [y, y], linewidth=1.0, alpha=0.65)
            marker = "D" if cohort == "Pooled" else "o"
            label = cohort if cohort not in labels_used else None
            ax.scatter([r], [y], s=45 + min(n, 400) * 0.18, marker=marker, alpha=0.8,
                       edgecolor="black", linewidth=0.3, label=label)
            labels_used.add(cohort)

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([nice_label(e) for e in endpoints], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Correlation r: ΔcBAG vs Δendpoint")
    ax.set_xlim(-1, 1)
    ax.set_title("Longitudinal Δ association summary")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)


def plot_availability(ax: plt.Axes, stats: pd.DataFrame, endpoints: Sequence[str]) -> None:
    cohorts = ["ADNI", "HABS", "Pooled"]
    mat = pd.DataFrame(index=endpoints, columns=cohorts, dtype=float)
    for e in endpoints:
        for c in cohorts:
            row = stats[(stats["endpoint"] == e) & (stats["cohort"] == c)]
            mat.loc[e, c] = row.iloc[0]["n"] if not row.empty else 0

    arr = mat.fillna(0).to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts)
    ax.set_yticks(range(len(endpoints)))
    ax.set_yticklabels([nice_label(e) for e in endpoints], fontsize=7)
    for i in range(len(endpoints)):
        for j in range(len(cohorts)):
            val = arr[i, j]
            ax.text(j, i, f"{int(val)}" if val > 0 else "", ha="center", va="center", fontsize=7)
    ax.set_title("Complete longitudinal pairs")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("N")


def plot_r_heatmap(ax: plt.Axes, stats: pd.DataFrame, endpoints: Sequence[str]) -> None:
    cohorts = ["ADNI", "HABS", "Pooled"]
    mat = pd.DataFrame(index=endpoints, columns=cohorts, dtype=float)
    for e in endpoints:
        for c in cohorts:
            row = stats[(stats["endpoint"] == e) & (stats["cohort"] == c)]
            mat.loc[e, c] = row.iloc[0]["r"] if not row.empty else np.nan

    arr = mat.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts)
    ax.set_yticks(range(len(endpoints)))
    ax.set_yticklabels([nice_label(e) for e in endpoints], fontsize=7)
    for i in range(len(endpoints)):
        for j in range(len(cohorts)):
            val = arr[i, j]
            ax.text(j, i, f"{val:.2f}" if np.isfinite(val) else "NA", ha="center", va="center", fontsize=7)
    ax.set_title("Longitudinal Δ association r")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("r")


def plot_delta_endpoint_distributions(ax: plt.Axes, df: pd.DataFrame, endpoints: Sequence[str]) -> None:
    selected = list(endpoints)[:6]
    if not selected:
        ax.text(0.5, 0.5, "No endpoints", ha="center", va="center")
        ax.axis("off")
        return

    labels, data = [], []
    for e in selected:
        col = f"delta__{e}"
        if col not in df.columns:
            continue
        vals = clean_numeric(df[col]).dropna().values
        if len(vals) > 1 and np.nanstd(vals) > 0:
            # z-score endpoint deltas so variables with large physical units
            # such as whole-brain volume do not flatten all other boxes.
            vals = (vals - np.nanmean(vals)) / np.nanstd(vals)
            labels.append(nice_label(e))
            data.append(vals)

    if not data:
        ax.text(0.5, 0.5, "No endpoint deltas", ha="center", va="center")
        ax.axis("off")
        return

    ax.boxplot(data, labels=labels, showfliers=False, vert=False)
    ax.axvline(0, linewidth=0.9)
    ax.set_title("Endpoint Δ distributions (z-scored)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", alpha=0.25)


def make_figure8(delta_df: pd.DataFrame, stats: pd.DataFrame, main_endpoints: Sequence[str],
                 outdir: Path, formats: Sequence[str], dpi: int, title: str) -> List[str]:
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.5))
    ax = axes.ravel()

    plot_delta_cbag_distribution(ax[0], delta_df)
    panel_label(ax[0], "A")

    # Main panels B-E: use the selected endpoint ranking, which prioritizes
    # significant associations instead of fixed preferred endpoints.
    selected_available = [
        e for e in main_endpoints
        if f"delta__{e}" in delta_df.columns and endpoint_has_complete_pairs(delta_df, e)
    ]

    endpoint_for_panel = selected_available[:4]

    while len(endpoint_for_panel) < 4:
        endpoint_for_panel.append("__missing__")

    for i, e in enumerate(endpoint_for_panel, start=1):
        if e == "__missing__":
            ax[i].text(0.5, 0.5, "No additional usable significant endpoint", ha="center", va="center")
            ax[i].axis("off")
        else:
            scatter_delta(ax[i], delta_df, e, stats)
        panel_label(ax[i], chr(ord("A") + i))

    plot_forest(ax[5], stats, main_endpoints)
    panel_label(ax[5], "F")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_all(fig, outdir / "Figure8_fullcohort_longitudinal_delta_cBAG_SIGNIFICANT_STRICT_ENRICHED", formats, dpi)


def make_supp_figure8(delta_df: pd.DataFrame, stats: pd.DataFrame, endpoints: Sequence[str],
                      outdir: Path, formats: Sequence[str], dpi: int, title: str) -> List[str]:
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.0))
    ax = axes.ravel()

    plot_availability(ax[0], stats, endpoints)
    panel_label(ax[0], "A")

    plot_r_heatmap(ax[1], stats, endpoints)
    panel_label(ax[1], "B")

    plot_forest(ax[2], stats, endpoints)
    ax[2].set_title("All available longitudinal Δ associations")
    panel_label(ax[2], "C")

    plot_delta_endpoint_distributions(ax[3], delta_df, endpoints)
    panel_label(ax[3], "D")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_all(fig, outdir / "Supplementary_Figure8_fullcohort_longitudinal_delta_cBAG_SIGNIFICANT_STRICT_ENRICHED", formats, dpi)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else base_dir / DEFAULT_OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    formats = [f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip()]

    print("=" * 100)
    print("Figure 8 fixed full-cohort longitudinal delta-cBAG")
    print("=" * 100)
    print(f"RESULTS_ROOT: {results_root}")
    print(f"BASE_DIR:     {base_dir}")
    print(f"OUTDIR:       {outdir}")
    print(f"COHORTS:      {', '.join(cohorts)}")
    print(f"FEATURE_SET:  {args.feature_set}")
    print(f"INPUT_KIND:   {args.input_kind}  # strict: enriched file only, no fallback")
    print(f"MAIN_ALPHA:   {args.main_alpha}")
    print(f"MIN_N:        {args.min_n}")
    print(f"CORR_METHOD:  {args.corr_method}")

    long_df, dataset_summary = load_fullcohort_longitudinal_tables(
        results_root=results_root,
        cohorts=cohorts,
        feature_set=args.feature_set,
        input_kind=args.input_kind,
    )
    dataset_summary.to_csv(outdir / "longitudinal_delta_dataset_summary.csv", index=False)

    print("\nDataset summary:")
    print(dataset_summary.to_string(index=False))

    if long_df.empty:
        raise SystemExit("No longitudinal full-cohort rows found.")

    # Choose endpoint columns using the visit-level table.
    endpoint_cols: Dict[str, str] = {}
    endpoint_selection_rows = []
    min_visit_n = max(args.min_n * 2, 10)

    for endpoint, spec in ENDPOINTS.items():
        col, status, n_visit = choose_endpoint_column(long_df, endpoint, spec, min_n_visits=min_visit_n)
        endpoint_cols[endpoint] = col  # type: ignore[assignment]
        endpoint_selection_rows.append({
            "endpoint": endpoint,
            "domain": spec.get("domain", ""),
            "selected_column": col,
            "selection_status": status,
            "visit_level_n": n_visit,
        })

    endpoint_selection = pd.DataFrame(endpoint_selection_rows)
    endpoint_selection.to_csv(outdir / "longitudinal_endpoint_column_selection.csv", index=False)

    print("\nEndpoint column selection:")
    print(endpoint_selection.to_string(index=False))

    delta_df = build_delta_table(long_df, endpoint_cols)
    if delta_df.empty:
        raise SystemExit("No subject-level delta rows could be created.")

    delta_df.to_csv(outdir / "longitudinal_delta_subject_table.csv", index=False)

    stats = compute_delta_associations(delta_df, endpoint_cols, min_n=args.min_n, method=args.corr_method)
    stats.to_csv(outdir / "longitudinal_delta_association_stats.csv", index=False)

    main_endpoints = select_main_endpoints(
        stats,
        max_endpoints=args.max_endpoints_main,
        alpha=args.main_alpha,
        min_n=args.min_n,
    )
    supp_endpoints = select_main_endpoints(
        stats,
        max_endpoints=args.max_endpoints_supp,
        alpha=args.main_alpha,
        min_n=args.min_n,
    )

    selected_panels_path = outdir / "figure8_selected_main_panels.csv"
    ranked_for_output = significant_endpoint_rank(stats, alpha=args.main_alpha, min_n=args.min_n)
    if not ranked_for_output.empty:
        ranked_for_output.to_csv(selected_panels_path, index=False)
    else:
        pd.DataFrame().to_csv(selected_panels_path, index=False)

    print("\nDelta table summary:")
    print(delta_df.groupby("cohort").agg(
        n_subjects=("longitudinal_subject", "nunique"),
        n_rows=("longitudinal_subject", "size"),
        mean_delta_cbag=("delta_cbag", "mean"),
        sd_delta_cbag=("delta_cbag", "std"),
    ).reset_index().to_string(index=False))

    print("\nSelected main endpoints:")
    print(main_endpoints)

    print("\nAssociation stats:")
    with pd.option_context("display.max_rows", 120, "display.max_columns", 20):
        print(stats.sort_values(["cohort", "abs_r"], ascending=[True, False]).head(120).to_string(index=False))

    fig8_outputs = make_figure8(
        delta_df=delta_df,
        stats=stats,
        main_endpoints=main_endpoints,
        outdir=outdir,
        formats=formats,
        dpi=args.dpi,
        title=args.main_title,
    )

    supp_outputs = make_supp_figure8(
        delta_df=delta_df,
        stats=stats,
        endpoints=supp_endpoints,
        outdir=outdir,
        formats=formats,
        dpi=args.dpi,
        title=args.supp_title,
    )

    manifest = pd.DataFrame([
        {
            "figure": "Figure 8",
            "description": "Strict enriched-input longitudinal full-cohort delta-cBAG biological validation with significant-panel selection",
            "feature_set": args.feature_set,
            "cohorts": ",".join(cohorts),
            "input_kind": args.input_kind,
            "corr_method": args.corr_method,
            "min_n": args.min_n,
            "outputs": ";".join(fig8_outputs),
        },
        {
            "figure": "Supplementary Figure 8",
            "description": "Extended longitudinal full-cohort delta-cBAG endpoint availability and association summary",
            "feature_set": args.feature_set,
            "cohorts": ",".join(cohorts),
            "input_kind": args.input_kind,
            "corr_method": args.corr_method,
            "min_n": args.min_n,
            "outputs": ";".join(supp_outputs),
        },
    ])
    manifest.to_csv(outdir / "figure8_manifest.csv", index=False)

    print("\nSaved outputs:")
    for p in fig8_outputs + supp_outputs:
        print(p)
    print(f"\nSaved tables to: {outdir}")
    print("DONE")


if __name__ == "__main__":
    main()
