#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Figure 4 and Supplementary Figure S4A-E.

Figure 4:
    Main neuroimaging-association figure using the imaging_only model.

Supplementary Figure S4A-E:
    Feature-set-specific neuroimaging-association figures:
        S4A = imaging_only
        S4B = imaging_demographics
        S4C = imaging_biomarkers
        S4D = full
        S4E = full_no_cardiovascular

Each figure:
    rows = cohorts
    columns within each cohort block =
        1. cBAG vs hippocampal volume relative to brain
        2. cBAG vs hippocampal FA
        3. cBAG vs hippocampal clustering coefficient
        4. cBAG vs hippocampal path length
        5. cBAG vs total brain volume
        6. cBAG vs total brain FA
        7. cBAG vs total/global graph clustering coefficient
        8. cBAG vs total/global graph path length

Inputs:
    /mnt/newStor/paros/paros_WORK/ines/results/
        BrainAgePrediction<COHORT>_stratified_groupcv_targetnorm_bagbiascorr_oofglobal/
            ablation_<feature_set>/
                validation_figures_full_cohort/
                    subject_level_validation_input_enriched_for_Figure4.csv

Outputs:
    /mnt/newStor/paros/paros_WORK/ines/results/
        BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/
            Figure4_full_cohort_main_and_S4_supplement_neuroimaging_associations/
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr


# =============================================================================
# DEFAULT PATHS
# =============================================================================

DEFAULT_RESULTS_ROOT = Path("/mnt/newStor/paros/paros_WORK/ines/results")

DEFAULT_BASE_VALIDATION_DIR = (
    DEFAULT_RESULTS_ROOT
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

DEFAULT_OUTDIR = (
    DEFAULT_BASE_VALIDATION_DIR
    / "Figure4_full_cohort_main_and_S4_supplement_neuroimaging_associations"
)


# =============================================================================
# SETTINGS
# =============================================================================

COHORT_ORDER = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

COHORT_LABELS = {
    "ADNI": "ADNI",
    "ADRC": "ADRC",
    "HABS": "HABS",
    "AD_DECODE": "AD-DECODE",
}

COHORT_LETTERS = {
    "ADNI": "A",
    "ADRC": "B",
    "HABS": "C",
    "AD_DECODE": "D",
}


COHORT_COLORS = {
    "ADNI": "#1f77b4",
    "ADRC": "#ff7f0e",
    "HABS": "#2ca02c",
    "AD_DECODE": "#d62728",
}

COHORT_MARKERS = {
    "ADNI": "o",
    "ADRC": "s",
    "HABS": "^",
    "AD_DECODE": "D",
}

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

SUPP_PANEL_MAP = {
    "A": "imaging_only",
    "B": "imaging_demographics",
    "C": "imaging_biomarkers",
    "D": "full",
    "E": "full_no_cardiovascular",
}

FEATURE_LABELS = {
    "imaging_only": "Imaging only",
    "imaging_demographics": "Imaging + demographics",
    "imaging_biomarkers": "Imaging + biomarkers",
    "full": "Full model",
    "full_no_cardiovascular": "Full model without cardiovascular variables",
}

MAIN_FEATURE_SET = "imaging_only"
VALIDATION_DIR_NAME = os.environ.get("FIGURE4_VALIDATION_DIR", "validation_figures_full_cohort")

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

CBAG_PRIORITY = [
    "cBAG_full_cohort_raw_clean",
    "cBAG_full_cohort",
    "cBAG_full_raw_clean",
    "cBAG_full",
    "BAG_full_cohort_raw_clean",
    "BAG_full_cohort",
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_foldwise_raw_clean",
    "cBAG_foldwise",
    "cBAG_raw_clean",
    "cBAG",
    "cBAG_global_raw_clean",
    "cBAG_global",
    "BAG_raw_clean",
    "BAG",
    "BAG_raw",
]


# =============================================================================
# VARIABLE CANDIDATES
# =============================================================================

HIPPO_VOLUME_PRIORITY = [
    "Hc_volume_relative_to_brain",
    "HC_volume_relative_to_brain",
    "Hippocampus_volume_relative_to_brain",
    "hippocampus_volume_relative_to_brain",
    "Hippocampus_Total_relative_to_brain",
    "Hippocampus_Total_pct",
    "Hippocampus_Total_percent",
    "Hippocampus_Total_norm",
    "Hippocampus_Total_normalized",
    "Hippocampus_Total_ICV_norm",
    "Hippocampus_Total_ICV_normalized",
    "Hippocampus_pct",
    "hippocampus_pct",
    "HC_volume_pct",
    "hc_volume_pct",
    "HC_vol_pct",
    "hc_vol_pct",
    "Hippocampus_Total",
    "Hippocampus_volume",
    "hippocampus_volume",
    "Hippocampal_volume",
    "hippocampal_volume",
    "hippocampal_vol",
    "HC_volume",
    "hc_volume",
    "HC_vol",
    "hc_vol",
]

HIPPO_FA_PRIORITY = [
    "Hc_FA",
    "HC_FA",
    "Hc_Fa",
    "HC_Fa",
    "Hippocampus_FA_Mean",
    "Hippocampus_FA_Total",
    "Hippocampus_FA",
    "hippocampus_FA",
    "hippocampal_FA",
    "Hippocampal_FA",
    "Left_Hippocampus_FA",
    "Right_Hippocampus_FA",
    "hc_fa",
]

HIPPO_CLUSTERING_PRIORITY = [
    "Hc_clustering_coeff",
    "HC_clustering_coeff",
    "Hc_Clustering_Coeff",
    "HC_Clustering_Coeff",
    "Hippocampus_clustering_coeff",
    "Hippocampus_Clustering_Coeff",
    "hippocampal_clustering_coeff",
    "Hippocampal_Clustering_Coeff",
    "Hc_clustering_coefficient",
    "HC_clustering_coefficient",
    "Hippocampus_clustering_coefficient",
    "Hippocampal_clustering_coefficient",
]

HIPPO_PATH_PRIORITY = [
    "Hc_path_length",
    "HC_path_length",
    "Hc_Path_Length",
    "HC_Path_Length",
    "Hippocampus_path_length",
    "Hippocampus_Path_Length",
    "hippocampal_path_length",
    "Hippocampal_Path_Length",
    "Hc_characteristic_path_length",
    "HC_characteristic_path_length",
    "Hippocampus_characteristic_path_length",
    "Hippocampal_characteristic_path_length",
]

TOTAL_BRAIN_VOLUME_PRIORITY = [
    "Total_Brain_volume",
    "total_brain_volume",
    "Total_Brain_Volume",
    "Brain_volume_total",
    "brain_volume_total",
    "TBV",
    "TotalBrainVolume",
    "Total_Brain_volume_pct",
    "Total_Brain_volume_norm",
    "Total_Brain_volume_normalized",
    "Relative_Brain_Volume",
    "relative_brain_volume",
    "Normalized_Brain_Volume",
    "normalized_brain_volume",
    "ICV_normalized_volume",
    "Volume_mean",
    "Volume_median",
]

TOTAL_BRAIN_FA_PRIORITY = [
    "Total_Brain_FA",
    "total_brain_FA",
    "Brain_FA_total",
    "brain_FA_total",
    "Global_FA",
    "global_FA",
    "FA_mean",
    "Mean_FA",
    "mean_FA",
    "FA_median",
    "FA",
]

TOTAL_GRAPH_CLUSTERING_PRIORITY = [
    "Total_graph_clustering_coeff",
    "total_graph_clustering_coeff",
    "Global_graph_clustering_coeff",
    "global_graph_clustering_coeff",
    "Graph_clustering_coeff",
    "graph_clustering_coeff",
    "Clustering_Coeff",
    "clustering_coeff",
    "ClusteringCoefficient",
    "Global clustering coefficient",
    "global_clustering_coefficient",
    "clustering coefficient",
]

TOTAL_GRAPH_PATH_PRIORITY = [
    "Total_graph_path_length",
    "total_graph_path_length",
    "Global_graph_path_length",
    "global_graph_path_length",
    "Graph_path_length",
    "graph_path_length",
    "Path_Length",
    "path_length",
    "Characteristic path length",
    "characteristic_path_length",
    "PathLength",
    "path length",
]


FIGURE4_SLOTS = [
    {
        "slot": "hippocampal_volume_relative_to_brain",
        "title": "cBAG vs Hc volume\n(relative to brain)",
        "candidates": HIPPO_VOLUME_PRIORITY,
        "avoid": ["fa", "clustering", "cluster", "path", "graph"],
        "require_any": ["hippocampus", "hippocampal", "hc_"],
        "prefer": ["relative", "pct", "percent", "norm", "normalized", "icv"],
        "range": None,
    },
    {
        "slot": "hippocampal_fa",
        "title": "cBAG vs Hc FA",
        "candidates": HIPPO_FA_PRIORITY,
        "avoid": ["total_brain", "global", "whole", "graph", "clustering", "path"],
        "require_any": ["hippocampus", "hippocampal", "hc_"],
        "prefer": ["fa"],
        "range": (0.0, 1.0),
    },
    {
        "slot": "hippocampal_clustering",
        "title": "cBAG vs Hc clustering coeff",
        "candidates": HIPPO_CLUSTERING_PRIORITY,
        "avoid": ["total", "global", "brain", "whole"],
        "require_any": ["hippocampus", "hippocampal", "hc_"],
        "prefer": ["clustering", "cluster", "coeff"],
        "range": None,
    },
    {
        "slot": "hippocampal_path_length",
        "title": "cBAG vs Hc path length",
        "candidates": HIPPO_PATH_PRIORITY,
        "avoid": ["total", "global", "brain", "whole"],
        "require_any": ["hippocampus", "hippocampal", "hc_"],
        "prefer": ["path", "length"],
        "range": None,
    },
    {
        "slot": "total_brain_volume",
        "title": "cBAG vs total brain volume",
        "candidates": TOTAL_BRAIN_VOLUME_PRIORITY,
        "avoid": ["hippocampus", "hippocampal", "hc_", "fa", "clustering", "cluster", "path", "graph"],
        "require_any": [],
        "prefer": ["total", "brain", "volume", "tbv"],
        "range": None,
    },
    {
        "slot": "total_brain_fa",
        "title": "cBAG vs total brain FA",
        "candidates": TOTAL_BRAIN_FA_PRIORITY,
        "avoid": ["hippocampus", "hippocampal", "hc_", "clustering", "cluster", "path", "graph"],
        "require_any": [],
        "prefer": ["total", "brain", "global", "fa"],
        "range": (0.0, 1.0),
    },
    {
        "slot": "total_graph_clustering",
        "title": "cBAG vs total graph clustering coeff",
        "candidates": TOTAL_GRAPH_CLUSTERING_PRIORITY,
        "avoid": ["hippocampus", "hippocampal", "hc_"],
        "require_any": [],
        "prefer": ["total", "global", "graph", "clustering", "cluster", "coeff"],
        "range": None,
    },
    {
        "slot": "total_graph_path_length",
        "title": "cBAG vs total graph path length",
        "candidates": TOTAL_GRAPH_PATH_PRIORITY,
        "avoid": ["hippocampus", "hippocampal", "hc_"],
        "require_any": [],
        "prefer": ["total", "global", "graph", "path", "length"],
        "range": None,
    },
]


# Heatmap summary rows. These are intentionally concise and match the summary
# view requested for Figure 4: color = Pearson r, annotation = r/stars/n.
HEATMAP_SLOTS = [
    {
        "slot": "hippocampal_volume_relative_to_brain",
        "label": "Hippocampal volume",
        "candidates": ["Hc_volume_pct_brain", *HIPPO_VOLUME_PRIORITY],
        "avoid": ["fa", "clustering", "cluster", "path", "graph"],
        "require_any": ["hippocampus", "hippocampal", "hc_"],
        "prefer": ["pct", "relative", "percent", "norm", "normalized", "icv"],
        "range": None,
    },
    {
        "slot": "hippocampal_fa",
        "label": "Hippocampal FA",
        "candidates": HIPPO_FA_PRIORITY,
        "avoid": ["total_brain", "global", "whole", "graph", "clustering", "path"],
        "require_any": ["hippocampus", "hippocampal", "hc_"],
        "prefer": ["fa"],
        "range": (0.0, 1.0),
    },
    {
        "slot": "total_brain_volume",
        "label": "Total brain volume",
        "candidates": TOTAL_BRAIN_VOLUME_PRIORITY,
        "avoid": ["hippocampus", "hippocampal", "hc_", "fa", "clustering", "cluster", "path", "graph"],
        "require_any": [],
        "prefer": ["total", "brain", "volume", "tbv"],
        "range": None,
    },
    {
        "slot": "total_brain_fa",
        "label": "Total brain FA",
        "candidates": TOTAL_BRAIN_FA_PRIORITY,
        "avoid": ["hippocampus", "hippocampal", "hc_", "clustering", "cluster", "path", "graph"],
        "require_any": [],
        "prefer": ["total", "brain", "global", "fa"],
        "range": (0.0, 1.0),
    },
    {
        "slot": "total_graph_clustering",
        "label": "Graph clustering coefficient",
        "candidates": TOTAL_GRAPH_CLUSTERING_PRIORITY,
        "avoid": ["hippocampus", "hippocampal", "hc_"],
        "require_any": [],
        "prefer": ["total", "global", "graph", "clustering", "cluster", "coeff"],
        "range": None,
    },
    {
        "slot": "total_graph_path_length",
        "label": "Graph path length",
        "candidates": TOTAL_GRAPH_PATH_PRIORITY,
        "avoid": ["hippocampus", "hippocampal", "hc_"],
        "require_any": [],
        "prefer": ["total", "global", "graph", "path", "length"],
        "range": None,
    },
    {
        "slot": "global_efficiency",
        "label": "Global Efficiency",
        "candidates": [
            "Global_Efficiency",
            "Global_Efficiency_raw_clean",
            "global_efficiency",
            "global efficiency",
        ],
        "avoid": ["local"],
        "require_any": [],
        "prefer": ["global", "efficiency"],
        "range": None,
    },
    {
        "slot": "local_efficiency",
        "label": "Local Efficiency",
        "candidates": [
            "Local_Efficiency",
            "Local_Efficiency_raw_clean",
            "local_efficiency",
            "local efficiency",
        ],
        "avoid": ["global"],
        "require_any": [],
        "prefer": ["local", "efficiency"],
        "range": None,
    },
]


# Pooled cross-cohort scatter summary. This mirrors the requested compact
# six-panel view: color/marker = cohort, one pooled regression line per panel.
CROSS_COHORT_SLOTS = [
    {**HEATMAP_SLOTS[0], "title": "Hippocampal volume", "xlabel": "HC volume"},
    {**HEATMAP_SLOTS[1], "title": "Hippocampal FA", "xlabel": "HC FA"},
    {**HEATMAP_SLOTS[2], "title": "Total brain volume", "xlabel": "Brain volume"},
    {**HEATMAP_SLOTS[3], "title": "Total brain FA", "xlabel": "Brain FA"},
    {**HEATMAP_SLOTS[4], "title": "Graph clustering coefficient", "xlabel": "Clustering"},
    {**HEATMAP_SLOTS[5], "title": "Graph path length", "xlabel": "Path length"},
]


# =============================================================================
# HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 4 and Supplementary Figure S4A-E."
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing BrainAgePrediction... outputs.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="Output directory for Figure 4 and Supplementary Figure S4 outputs.",
    )
    parser.add_argument(
        "--main-feature-set",
        default=MAIN_FEATURE_SET,
        help="Feature set for main Figure 4.",
    )
    parser.add_argument(
        "--validation-dir-name",
        default=VALIDATION_DIR_NAME,
        help="Validation subdirectory to read from, e.g. validation_figures_full_cohort or validation_figures.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=8,
        help="Minimum complete cases required for a scatter panel.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="DPI for saved PNG/PDF outputs.",
    )
    return parser.parse_args()


def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    for val in SENTINEL_VALUES:
        out = out.mask(out == val, np.nan)
    return out


def apply_range(series: pd.Series, value_range: Optional[tuple[float, float]]) -> pd.Series:
    out = clean_numeric(series)
    if value_range is None:
        return out
    lo, hi = value_range
    return out.mask((out < lo) | (out > hi), np.nan)


def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def usable_n(df: pd.DataFrame, col: str, value_range: Optional[tuple[float, float]]) -> int:
    if col not in df.columns:
        return 0
    return int(apply_range(df[col], value_range).notna().sum())


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    exact = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    norm = {normalize_name(c): c for c in df.columns}

    for cand in candidates:
        if cand in exact:
            return exact[cand]
    for cand in candidates:
        if str(cand).lower() in lower:
            return lower[str(cand).lower()]
    for cand in candidates:
        nc = normalize_name(cand)
        if nc in norm:
            return norm[nc]
    return None


def find_best_numeric_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    min_n: int,
    avoid_keywords: Sequence[str],
    require_any_keywords: Sequence[str],
    prefer_keywords: Sequence[str],
    value_range: Optional[tuple[float, float]],
) -> Optional[str]:
    avoid = [x.lower() for x in avoid_keywords]
    require_any = [x.lower() for x in require_any_keywords]
    prefer = [x.lower() for x in prefer_keywords]

    expanded = []
    for c in candidates:
        expanded.append(f"{c}_raw_clean")
        expanded.append(c)

    def eligible(col: str) -> bool:
        low = str(col).lower()
        if any(x in low for x in avoid):
            return False
        if require_any and not any(x in low for x in require_any):
            return False
        return usable_n(df, col, value_range) >= min_n

    for col in expanded:
        if col in df.columns and eligible(col):
            return col

    tokens = []
    for c in candidates:
        tokens.extend([t for t in re.split(r"[_\s]+", str(c).lower()) if len(t) >= 3])
    tokens = sorted(set(tokens), key=len, reverse=True)

    fallback = []
    for col in df.columns:
        low = str(col).lower()
        if not any(t in low for t in tokens):
            continue
        if not eligible(col):
            continue
        score = sum(p in low for p in prefer)
        fallback.append((score, col))

    if not fallback:
        return None

    fallback.sort(key=lambda x: (-x[0], str(x[1]).lower()))
    return fallback[0][1]


def validation_input_path(results_root: Path, cohort: str, feature_set: str) -> Path:
    return (
        results_root
        / RESULTS_DIR_MAP[cohort]
        / f"ablation_{feature_set}"
        / VALIDATION_DIR_NAME
        / "subject_level_validation_input_enriched_for_Figure4.csv"
    )


def load_validation_table(
    results_root: Path,
    cohort: str,
    feature_set: str,
) -> tuple[Optional[pd.DataFrame], dict]:
    path = validation_input_path(results_root, cohort, feature_set)

    availability = {
        "cohort": cohort,
        "feature_set": feature_set,
        "source_path": str(path),
        "status": "",
        "n_rows": 0,
        "cbag_col": "",
    }

    if not path.exists():
        availability["status"] = "missing_file"
        return None, availability

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        availability["status"] = f"read_error: {exc}"
        return None, availability

    availability["n_rows"] = len(df)

    cbag_col = first_existing(df, CBAG_PRIORITY)
    if cbag_col is None:
        availability["status"] = "missing_cbag_column"
        return None, availability

    df = df.copy()
    df["cohort"] = cohort
    df["feature_set"] = feature_set
    df["_source_path"] = str(path)
    df["_cbag"] = clean_numeric(df[cbag_col])
    df["_cbag_col"] = cbag_col

    df = df.dropna(subset=["_cbag"]).copy()

    availability["status"] = "loaded"
    availability["cbag_col"] = cbag_col
    availability["n_rows_after_cbag_clean"] = len(df)

    return df, availability


def nice_label(col: Optional[str]) -> str:
    if col is None:
        return ""

    raw = str(col).replace("_raw_clean", "")

    mapping = {
        "cBAG_oof_global": "cBAG",
        "cBAG_foldwise": "cBAG",
        "cBAG": "cBAG",
        "BAG_raw": "BAG",
        "BAG": "BAG",
        "Hc_volume_relative_to_brain": "Hc volume relative to brain",
        "HC_volume_relative_to_brain": "Hc volume relative to brain",
        "Hippocampus_volume_relative_to_brain": "Hc volume relative to brain",
        "Hippocampus_Total_pct": "Hippocampal volume",
        "Hippocampus_Total": "Hippocampal volume",
        "Hippocampus_volume": "Hippocampal volume",
        "Hippocampal_volume": "Hippocampal volume",
        "Hc_FA": "Hc FA",
        "HC_FA": "Hc FA",
        "Hippocampus_FA_Mean": "Hippocampal FA",
        "Hippocampus_FA_Total": "Hippocampal FA",
        "Hippocampus_FA": "Hippocampal FA",
        "Hc_clustering_coeff": "Hc clustering coefficient",
        "HC_clustering_coeff": "Hc clustering coefficient",
        "Hippocampus_clustering_coeff": "Hc clustering coefficient",
        "Hc_path_length": "Hc path length",
        "HC_path_length": "Hc path length",
        "Hippocampus_path_length": "Hc path length",
        "Total_Brain_volume": "Total brain volume",
        "total_brain_volume": "Total brain volume",
        "TBV": "Total brain volume",
        "Volume_mean": "Mean volume",
        "Volume_median": "Median volume",
        "Total_Brain_FA": "Total brain FA",
        "FA_mean": "Mean FA",
        "FA_median": "Median FA",
        "Global_FA": "Global FA",
        "Clustering_Coeff": "Clustering coefficient",
        "Global clustering coefficient": "Clustering coefficient",
        "Path_Length": "Path length",
        "Characteristic path length": "Path length",
    }

    return mapping.get(raw, raw.replace("_", " "))


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3g}"


def empty_axis(ax: plt.Axes, title: str, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8, wrap=True)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scatter_with_fit(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: Optional[str],
    slot: str,
    title: str,
    min_n: int,
    value_range: Optional[tuple[float, float]],
) -> tuple[dict, pd.DataFrame]:
    if x_col is None:
        empty_axis(ax, title, "Variable not found")
        return {
            "slot": slot,
            "title": title,
            "status": "missing_variable",
            "x_col": "",
            "n": 0,
            "r": np.nan,
            "p": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }, pd.DataFrame()

    plot_df = pd.DataFrame({
        "x": apply_range(df[x_col], value_range),
        "y": clean_numeric(df["_cbag"]),
        "cohort": df["cohort"].values,
        "feature_set": df["feature_set"].values,
        "x_col": x_col,
        "x_label": nice_label(x_col),
        "slot": slot,
        "source_path": df["_source_path"].values,
        "cbag_col": df["_cbag_col"].values,
    }).replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])

    if len(plot_df) < min_n:
        empty_axis(ax, title, f"Insufficient data\nn={len(plot_df)}")
        return {
            "slot": slot,
            "title": title,
            "status": "insufficient_n",
            "x_col": x_col,
            "n": int(len(plot_df)),
            "r": np.nan,
            "p": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }, plot_df

    if plot_df["x"].nunique() < 2 or plot_df["y"].nunique() < 2:
        empty_axis(ax, title, "Constant variable")
        return {
            "slot": slot,
            "title": title,
            "status": "constant_variable",
            "x_col": x_col,
            "n": int(len(plot_df)),
            "r": np.nan,
            "p": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }, plot_df

    ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=14,
        alpha=0.72,
        edgecolors="black",
        linewidth=0.2,
    )

    lr = linregress(plot_df["x"], plot_df["y"])
    r, p = pearsonr(plot_df["x"], plot_df["y"])

    xx = np.linspace(plot_df["x"].min(), plot_df["x"].max(), 100)
    ax.plot(xx, lr.intercept + lr.slope * xx, linestyle="--", linewidth=1.1)

    ax.text(
        0.03,
        0.97,
        f"n={len(plot_df)}\nr={r:.2f}\np={format_p(p)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, linewidth=0.4),
    )

    ax.set_title(title, fontsize=8)
    ax.set_xlabel(nice_label(x_col), fontsize=7)
    ax.set_ylabel("cBAG", fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", labelsize=7)

    stats = {
        "slot": slot,
        "title": title,
        "status": "plotted",
        "x_col": x_col,
        "x_label": nice_label(x_col),
        "n": int(len(plot_df)),
        "r": float(r),
        "p": float(p),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "x_mean": float(plot_df["x"].mean()),
        "x_sd": float(plot_df["x"].std(ddof=1)),
        "y_mean": float(plot_df["y"].mean()),
        "y_sd": float(plot_df["y"].std(ddof=1)),
    }

    return stats, plot_df


def choose_slot_column(df: pd.DataFrame, spec: dict, min_n: int) -> Optional[str]:
    return find_best_numeric_col(
        df=df,
        candidates=spec["candidates"],
        min_n=min_n,
        avoid_keywords=spec["avoid"],
        require_any_keywords=spec["require_any"],
        prefer_keywords=spec["prefer"],
        value_range=spec["range"],
    )


def plot_one_figure(
    results_root: Path,
    outdir: Path,
    figure_name: str,
    feature_set: str,
    title: str,
    min_n: int,
    dpi: int,
) -> dict:
    availability_rows = []
    stats_rows = []
    source_rows = []

    data_by_cohort = {}

    for cohort in COHORT_ORDER:
        df, availability = load_validation_table(results_root, cohort, feature_set)
        availability_rows.append(availability)
        if df is not None and not df.empty:
            data_by_cohort[cohort] = df

    fig = plt.figure(figsize=(18, 16), dpi=dpi)
    outer = fig.add_gridspec(
        nrows=len(COHORT_ORDER),
        ncols=1,
        hspace=0.65,
    )

    for row_idx, cohort in enumerate(COHORT_ORDER):
        inner = outer[row_idx].subgridspec(2, 4, wspace=0.45, hspace=0.55)
        axes = [fig.add_subplot(inner[i, j]) for i in range(2) for j in range(4)]

        cohort_df = data_by_cohort.get(cohort, pd.DataFrame())

        for slot_idx, (ax, spec) in enumerate(zip(axes, FIGURE4_SLOTS)):
            if slot_idx == 0:
                ax.text(
                    -0.20,
                    1.28,
                    f"{COHORT_LETTERS[cohort]}. {COHORT_LABELS[cohort]}",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )

            if cohort_df.empty:
                empty_axis(ax, spec["title"], "No data")
                stats_rows.append({
                    "figure": figure_name,
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "slot": spec["slot"],
                    "title": spec["title"],
                    "status": "no_data",
                    "x_col": "",
                    "n": 0,
                })
                continue

            x_col = choose_slot_column(cohort_df, spec, min_n=min_n)

            stats, plot_df = scatter_with_fit(
                ax=ax,
                df=cohort_df,
                x_col=x_col,
                slot=spec["slot"],
                title=spec["title"],
                min_n=min_n,
                value_range=spec["range"],
            )

            stats.update({
                "figure": figure_name,
                "cohort": cohort,
                "feature_set": feature_set,
                "cbag_col": cohort_df["_cbag_col"].iloc[0],
                "source_path": cohort_df["_source_path"].iloc[0],
            })
            stats_rows.append(stats)

            if not plot_df.empty:
                plot_df = plot_df.copy()
                plot_df["figure"] = figure_name
                source_rows.append(plot_df)

    fig.suptitle(title, fontsize=16, y=0.995)
    outbase = outdir / figure_name

    png_path = outbase.with_suffix(".png")
    pdf_path = outbase.with_suffix(".pdf")
    stats_path = outdir / f"{figure_name}_stats.csv"
    source_path = outdir / f"{figure_name}_source_data.csv"
    availability_path = outdir / f"{figure_name}_input_availability.csv"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(source_path, index=False)
    else:
        pd.DataFrame().to_csv(source_path, index=False)

    pd.DataFrame(availability_rows).to_csv(availability_path, index=False)

    print("\nSaved:")
    print(png_path)
    print(pdf_path)
    print(stats_path)
    print(source_path)
    print(availability_path)

    return {
        "figure": figure_name,
        "feature_set": feature_set,
        "png": str(png_path),
        "pdf": str(pdf_path),
        "stats": str(stats_path),
        "source_data": str(source_path),
        "availability": str(availability_path),
    }


def significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compute_heatmap_stats(
    results_root: Path,
    feature_set: str,
    min_n: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Compute cohort x neuroimaging-variable correlations for heatmaps."""
    rows = []
    availability_rows = []

    for cohort in COHORT_ORDER:
        df, availability = load_validation_table(results_root, cohort, feature_set)
        availability_rows.append(availability)

        for spec in HEATMAP_SLOTS:
            base = {
                "cohort": cohort,
                "cohort_label": COHORT_LABELS[cohort],
                "feature_set": feature_set,
                "slot": spec["slot"],
                "label": spec["label"],
                "x_col": "",
                "n": 0,
                "r": np.nan,
                "p": np.nan,
                "status": "no_data",
                "source_path": availability.get("source_path", ""),
                "cbag_col": availability.get("cbag_col", ""),
            }

            if df is None or df.empty:
                rows.append(base)
                continue

            x_col = choose_slot_column(df, spec, min_n=min_n)
            if x_col is None:
                base["status"] = "missing_variable"
                rows.append(base)
                continue

            tmp = pd.DataFrame({
                "x": apply_range(df[x_col], spec["range"]),
                "y": clean_numeric(df["_cbag"]),
            }).replace([np.inf, -np.inf], np.nan).dropna()

            base["x_col"] = str(x_col)
            base["n"] = int(len(tmp))

            if len(tmp) < min_n:
                base["status"] = "insufficient_n"
                rows.append(base)
                continue

            if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
                base["status"] = "constant_variable"
                rows.append(base)
                continue

            r, p = pearsonr(tmp["x"], tmp["y"])
            base.update({
                "status": "plotted",
                "r": float(r),
                "p": float(p),
                "stars": significance_stars(float(p)),
            })
            rows.append(base)

    out = pd.DataFrame(rows)
    if "stars" not in out.columns:
        out["stars"] = ""
    out["stars"] = out["stars"].fillna("")
    return out, availability_rows


def plot_heatmap_figure(
    results_root: Path,
    outdir: Path,
    heatmap_name: str,
    feature_set: str,
    title: str,
    min_n: int,
    dpi: int,
) -> dict:
    """Save Pearson-r heatmap plus the underlying stats table."""
    stats, availability_rows = compute_heatmap_stats(
        results_root=results_root,
        feature_set=feature_set,
        min_n=min_n,
    )

    row_labels = [spec["label"] for spec in HEATMAP_SLOTS]
    row_slots = [spec["slot"] for spec in HEATMAP_SLOTS]
    col_labels = [COHORT_LABELS[c] for c in COHORT_ORDER]

    r_matrix = np.full((len(row_slots), len(COHORT_ORDER)), np.nan, dtype=float)
    annot = [["" for _ in COHORT_ORDER] for _ in row_slots]

    for i, slot in enumerate(row_slots):
        for j, cohort in enumerate(COHORT_ORDER):
            hit = stats[(stats["slot"].eq(slot)) & (stats["cohort"].eq(cohort))]
            if hit.empty:
                annot[i][j] = "NA"
                continue
            row = hit.iloc[0]
            if row.get("status") != "plotted" or not np.isfinite(row.get("r", np.nan)):
                n = int(row.get("n", 0) or 0)
                annot[i][j] = f"NA\nn={n}" if n else "NA"
                continue
            r = float(row["r"])
            n = int(row["n"])
            stars = str(row.get("stars", ""))
            r_matrix[i, j] = r
            annot[i][j] = f"{r:.2f}{stars}\nn={n}"

    fig_width = max(6.0, 1.35 * len(COHORT_ORDER) + 2.9)
    fig_height = max(5.0, 0.48 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    im = ax.imshow(
        r_matrix,
        cmap="coolwarm",
        vmin=-0.75,
        vmax=0.75,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(COHORT_ORDER)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=12, pad=12)

    for i in range(len(row_labels)):
        for j in range(len(COHORT_ORDER)):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=7)

    ax.set_xticks(np.arange(-0.5, len(COHORT_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Pearson r", rotation=90)

    fig.tight_layout()

    outbase = outdir / heatmap_name
    png_path = outbase.with_suffix(".png")
    pdf_path = outbase.with_suffix(".pdf")
    stats_path = outdir / f"{heatmap_name}_stats.csv"
    availability_path = outdir / f"{heatmap_name}_input_availability.csv"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    stats.to_csv(stats_path, index=False)
    pd.DataFrame(availability_rows).to_csv(availability_path, index=False)

    print("\nSaved heatmap:")
    print(png_path)
    print(pdf_path)
    print(stats_path)
    print(availability_path)

    return {
        "figure": heatmap_name,
        "feature_set": feature_set,
        "png": str(png_path),
        "pdf": str(pdf_path),
        "stats": str(stats_path),
        "source_data": "",
        "availability": str(availability_path),
    }



def load_cross_cohort_plot_data(
    results_root: Path,
    feature_set: str,
    spec: dict,
    min_n: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Load one variable across all cohorts for a pooled cross-cohort plot."""
    plot_parts = []
    rows = []

    for cohort in COHORT_ORDER:
        df, availability = load_validation_table(results_root, cohort, feature_set)
        base = {
            "cohort": cohort,
            "cohort_label": COHORT_LABELS[cohort],
            "feature_set": feature_set,
            "slot": spec["slot"],
            "label": spec.get("label", spec.get("title", spec["slot"])),
            "x_col": "",
            "n": 0,
            "r": np.nan,
            "p": np.nan,
            "status": "no_data",
            "source_path": availability.get("source_path", ""),
            "cbag_col": availability.get("cbag_col", ""),
        }

        if df is None or df.empty:
            rows.append(base)
            continue

        x_col = choose_slot_column(df, spec, min_n=min_n)
        if x_col is None:
            base["status"] = "missing_variable"
            rows.append(base)
            continue

        tmp = pd.DataFrame({
            "x": apply_range(df[x_col], spec["range"]),
            "y": clean_numeric(df["_cbag"]),
            "cohort": cohort,
            "cohort_label": COHORT_LABELS[cohort],
            "feature_set": feature_set,
            "slot": spec["slot"],
            "label": spec.get("label", spec.get("title", spec["slot"])),
            "x_col": str(x_col),
            "x_label": nice_label(x_col),
            "source_path": df["_source_path"].values,
            "cbag_col": df["_cbag_col"].values,
        }).replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])

        base["x_col"] = str(x_col)
        base["n"] = int(len(tmp))

        if len(tmp) < min_n:
            base["status"] = "insufficient_n"
            rows.append(base)
            continue

        if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
            base["status"] = "constant_variable"
            rows.append(base)
            continue

        r, p = pearsonr(tmp["x"], tmp["y"])
        base.update({
            "status": "plotted",
            "r": float(r),
            "p": float(p),
            "stars": significance_stars(float(p)),
        })
        rows.append(base)
        plot_parts.append(tmp)

    if plot_parts:
        plot_df = pd.concat(plot_parts, ignore_index=True)
    else:
        plot_df = pd.DataFrame(columns=["x", "y", "cohort", "feature_set", "slot"])

    return plot_df, rows


def plot_cross_cohort_figure(
    results_root: Path,
    outdir: Path,
    figure_name: str,
    feature_set: str,
    title: str,
    min_n: int,
    dpi: int,
) -> dict:
    """Save pooled cross-cohort scatter panels colored by cohort."""
    stats_rows = []
    source_rows = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=dpi)
    axes = axes.ravel()

    for ax, spec in zip(axes, CROSS_COHORT_SLOTS):
        plot_df, cohort_rows = load_cross_cohort_plot_data(
            results_root=results_root,
            feature_set=feature_set,
            spec=spec,
            min_n=min_n,
        )
        stats_rows.extend(cohort_rows)

        if plot_df.empty or len(plot_df) < min_n:
            empty_axis(ax, spec["title"], f"Insufficient pooled data\nn={len(plot_df)}")
            continue

        if plot_df["x"].nunique() < 2 or plot_df["y"].nunique() < 2:
            empty_axis(ax, spec["title"], "Constant pooled variable")
            continue

        for cohort in COHORT_ORDER:
            sub = plot_df[plot_df["cohort"].eq(cohort)]
            if sub.empty:
                continue
            ax.scatter(
                sub["x"],
                sub["y"],
                s=18,
                alpha=0.72,
                linewidth=0.25,
                edgecolors="white",
                color=COHORT_COLORS.get(cohort, None),
                marker=COHORT_MARKERS.get(cohort, "o"),
                label=cohort,
            )

        lr = linregress(plot_df["x"], plot_df["y"])
        r, p = pearsonr(plot_df["x"], plot_df["y"])
        xx = np.linspace(plot_df["x"].min(), plot_df["x"].max(), 100)
        ax.plot(xx, lr.intercept + lr.slope * xx, color="black", linewidth=1.5)

        ci_txt = ""
        try:
            # Fisher-z approximate 95% CI for Pearson r.
            n = len(plot_df)
            if n > 3 and abs(r) < 1:
                z = np.arctanh(r)
                se = 1 / np.sqrt(n - 3)
                lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
                ci_txt = f" [{lo:.2f}, {hi:.2f}]"
        except Exception:
            ci_txt = ""

        ax.text(
            0.04,
            0.94,
            f"pooled r={r:.2f}{ci_txt}\np={format_p(p)}\nn={len(plot_df)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, linewidth=0.3),
        )

        ax.set_title(spec["title"], fontsize=12)
        ax.set_xlabel(spec.get("xlabel", spec.get("label", spec["slot"])), fontsize=10)
        ax.set_ylabel("Bias-corrected cBAG", fontsize=10)
        ax.grid(alpha=0.20)
        ax.tick_params(axis="both", labelsize=9)

        pooled_row = {
            "figure": figure_name,
            "cohort": "POOLED",
            "cohort_label": "Pooled",
            "feature_set": feature_set,
            "slot": spec["slot"],
            "label": spec.get("label", spec.get("title", spec["slot"])),
            "x_col": ";".join(sorted(set(map(str, plot_df["x_col"].dropna())))),
            "n": int(len(plot_df)),
            "r": float(r),
            "p": float(p),
            "stars": significance_stars(float(p)),
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "status": "plotted",
        }
        stats_rows.append(pooled_row)

        src = plot_df.copy()
        src["figure"] = figure_name
        source_rows.append(src)

    # Single shared legend at the bottom.
    handles = []
    labels = []
    for cohort in COHORT_ORDER:
        handles.append(plt.Line2D(
            [0], [0],
            marker=COHORT_MARKERS.get(cohort, "o"),
            color="none",
            markerfacecolor=COHORT_COLORS.get(cohort, "gray"),
            markeredgecolor="white",
            markersize=7,
            linestyle="",
        ))
        labels.append(COHORT_LABELS[cohort])

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(COHORT_ORDER),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(title, fontsize=16, y=0.985)
    fig.tight_layout(rect=[0.02, 0.06, 1.0, 0.94])

    outbase = outdir / figure_name
    png_path = outbase.with_suffix(".png")
    pdf_path = outbase.with_suffix(".pdf")
    stats_path = outdir / f"{figure_name}_stats.csv"
    source_path = outdir / f"{figure_name}_source_data.csv"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(source_path, index=False)
    else:
        pd.DataFrame().to_csv(source_path, index=False)

    print("\nSaved cross-cohort figure:")
    print(png_path)
    print(pdf_path)
    print(stats_path)
    print(source_path)

    return {
        "figure": figure_name,
        "feature_set": feature_set,
        "png": str(png_path),
        "pdf": str(pdf_path),
        "stats": str(stats_path),
        "source_data": str(source_path),
        "availability": "",
    }

def main() -> None:
    args = parse_args()

    global VALIDATION_DIR_NAME
    VALIDATION_DIR_NAME = args.validation_dir_name

    results_root = Path(args.results_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    print("=" * 100)
    print("Building Figure 4 and Supplementary Figure S4A-E")
    print("=" * 100)
    print("Results root:", results_root)
    print("Output directory:", outdir)
    print("Main feature set:", args.main_feature_set)
    print("Validation directory:", VALIDATION_DIR_NAME)
    print("Minimum n:", args.min_n)

    main_result = plot_one_figure(
        results_root=results_root,
        outdir=outdir,
        figure_name="Figure4_cBAG_neuroimaging_associations_imaging_only",
        feature_set=args.main_feature_set,
        title="Figure 4. cBAG neuroimaging associations across cohorts",
        min_n=args.min_n,
        dpi=args.dpi,
    )
    main_result["status"] = "main_figure"
    manifest_rows.append(main_result)

    main_heatmap = plot_heatmap_figure(
        results_root=results_root,
        outdir=outdir,
        heatmap_name="Figure4_heatmap_cBAG_neuroimaging_associations_imaging_only",
        feature_set=args.main_feature_set,
        title="Figure 4 heatmap. cBAG neuroimaging associations",
        min_n=args.min_n,
        dpi=args.dpi,
    )
    main_heatmap["status"] = "main_heatmap"
    manifest_rows.append(main_heatmap)

    main_cross = plot_cross_cohort_figure(
        results_root=results_root,
        outdir=outdir,
        figure_name="Figure4_cross_cohort_cBAG_neuroimaging_associations_imaging_only",
        feature_set=args.main_feature_set,
        title=(
            "Figure 4. Neuroimaging associations with bias-corrected cBAG\n"
            "Primary model: imaging only"
        ),
        min_n=args.min_n,
        dpi=args.dpi,
    )
    main_cross["status"] = "main_cross_cohort_figure"
    manifest_rows.append(main_cross)

    for panel_letter, feature_set in SUPP_PANEL_MAP.items():
        result = plot_one_figure(
            results_root=results_root,
            outdir=outdir,
            figure_name=f"SupplementaryFigureS4{panel_letter}_cBAG_neuroimaging_associations_{feature_set}",
            feature_set=feature_set,
            title=(
                f"Supplementary Figure S4{panel_letter}. "
                f"cBAG neuroimaging associations: {FEATURE_LABELS[feature_set]}"
            ),
            min_n=args.min_n,
            dpi=args.dpi,
        )
        result["status"] = "supplementary_figure"
        manifest_rows.append(result)

        heatmap_result = plot_heatmap_figure(
            results_root=results_root,
            outdir=outdir,
            heatmap_name=f"SupplementaryFigureS4{panel_letter}_heatmap_cBAG_neuroimaging_associations_{feature_set}",
            feature_set=feature_set,
            title=(
                f"Supplementary Figure S4{panel_letter} heatmap. "
                f"cBAG neuroimaging associations: {FEATURE_LABELS[feature_set]}"
            ),
            min_n=args.min_n,
            dpi=args.dpi,
        )
        heatmap_result["status"] = "supplementary_heatmap"
        manifest_rows.append(heatmap_result)

        cross_result = plot_cross_cohort_figure(
            results_root=results_root,
            outdir=outdir,
            figure_name=f"SupplementaryFigureS4{panel_letter}_cross_cohort_cBAG_neuroimaging_associations_{feature_set}",
            feature_set=feature_set,
            title=(
                f"Supplementary Figure S4{panel_letter}. "
                f"Neuroimaging associations with bias-corrected cBAG\n"
                f"Model: {FEATURE_LABELS[feature_set]}"
            ),
            min_n=args.min_n,
            dpi=args.dpi,
        )
        cross_result["status"] = "supplementary_cross_cohort_figure"
        manifest_rows.append(cross_result)

    manifest = pd.DataFrame(manifest_rows)

    manifest_csv = outdir / "Figure4_SupplementaryFigureS4_manifest.csv"
    manifest_xlsx = outdir / "Figure4_SupplementaryFigureS4_manifest.xlsx"

    manifest.to_csv(manifest_csv, index=False)
    manifest.to_excel(manifest_xlsx, index=False)

    print("\nSaved manifest:")
    print(manifest_csv)
    print(manifest_xlsx)

    print("\nDone.")


if __name__ == "__main__":
    main()