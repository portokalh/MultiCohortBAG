#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure4_enrich_fullcohort_allcohorts_HABSfixed_GRAPH_METRICS_FIXED.py

Recompute and append canonical Figure 4 neuroimaging variables to each
subject_level_validation_input.csv.

Writes, for each cohort x feature set:
    subject_level_validation_input_enriched_for_Figure4.csv

Canonical appended columns:
    Hc_volume_mm3
    Hc_volume_pct_brain
    Hc_FA
    Hc_RD
    Hc_AD
    Hc_ADC
    Hc_clustering_coeff
    Hc_path_length
    Total_Brain_volume
    Total_Brain_FA
    Total_graph_clustering_coeff
    Total_graph_path_length
    Global_Efficiency
    Local_Efficiency

Atlas mapping:
    Regional ROI labels:
        Left hippocampus  = index 17
        Right hippocampus = index 53

    Connectome node numbers:
        Left hippocampus  = index2 6
        Right hippocampus = index2 14

    Python zero-based connectome positions:
        Left hippocampus  = 5
        Right hippocampus = 13
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
BASE = WORK / "ines"
DATA_ROOT = BASE / "data"
RESULTS_ROOT = BASE / "results"

ATLAS_PATH = DATA_ROOT / "atlas" / "IITmean_RPI_index.xlsx"

BASE_VALIDATION_DIR = (
    RESULTS_ROOT
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

OUTDIR = BASE_VALIDATION_DIR / "Figure4_enrichment_full_cohort_GRAPH_METRICS_FIXED"

VALIDATION_INPUT = "subject_level_validation_input.csv"
VALIDATION_DIR_NAME = os.environ.get("FIGURE4_VALIDATION_DIR", "validation_figures_full_cohort")
ENRICHED_INPUT = "subject_level_validation_input_enriched_for_Figure4.csv"


# =============================================================================
# COHORT / FEATURE SETTINGS
# =============================================================================

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

PREDICTION_DIRS = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}


# =============================================================================
# REGIONAL SOURCE CANDIDATES
# =============================================================================

REGIONAL_CONFIG = {
    "ADNI": {
        "fa_candidates": [
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_fa.txt",
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_fa.csv",
        ],
        "rd_candidates": [
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_rd.txt",
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_rd.csv",
        ],
        "ad_candidates": [
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_ad.txt",
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_ad.csv",
        ],
        "adc_candidates": [
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_adc.txt",
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_adc.csv",
        ],
        "relative_volume_candidates": [
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_BrainPct.csv",
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_volume_norm.txt",
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_volume_norm.csv",
        ],
        "absolute_volume_candidates": [
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_BrainAbs.csv",
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_volume.txt",
            DATA_ROOT / "Regional_stats" / "ADNI" / "ADNI_studywide_stats_for_volume.csv",
        ],
        "harmonized_dirs": [DATA_ROOT / "harmonization" / "ADNI"],
    },
    "ADRC": {
        "fa_candidates": [
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_fa.txt",
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_fa.csv",
        ],
        "rd_candidates": [
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_rd.txt",
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_rd.csv",
        ],
        "ad_candidates": [
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_ad.txt",
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_ad.csv",
        ],
        "adc_candidates": [
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_adc.txt",
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_adc.csv",
        ],
        "relative_volume_candidates": [
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_BrainPct.csv",
        ],
        "absolute_volume_candidates": [
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_BrainAbs.csv",
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_volume.txt",
            DATA_ROOT / "Regional_stats" / "ADRC" / "ADRC_studywide_stats_for_volume.csv",
        ],
        "harmonized_dirs": [DATA_ROOT / "harmonization" / "ADRC"],
    },
    "HABS": {
        "fa_candidates": [
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_fa.txt",
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_fa.csv",
        ],
        "rd_candidates": [
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_rd.txt",
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_rd.csv",
        ],
        "ad_candidates": [
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_ad.txt",
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_ad.csv",
        ],
        "adc_candidates": [
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_adc.txt",
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_adc.csv",
        ],
        "relative_volume_candidates": [
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_BrainPct.csv",
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_volume_norm.txt",
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_volume_norm.csv",
        ],
        "absolute_volume_candidates": [
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_BrainAbs.csv",
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_volume.txt",
            DATA_ROOT / "Regional_stats" / "HABS" / "HABS_studywide_stats_for_volume.csv",
        ],
        "harmonized_dirs": [DATA_ROOT / "harmonization" / "HABS"],
    },
    "AD_DECODE": {
        "fa_candidates": [
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_mrtrixfa.csv",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_fa.txt",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_fa.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_mrtrixfa.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_fa.txt",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_fa.csv",
        ],
        "rd_candidates": [
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_rd.txt",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_rd.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_rd.txt",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_rd.csv",
        ],
        "ad_candidates": [
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_ad.txt",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_ad.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_ad.txt",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_ad.csv",
        ],
        "adc_candidates": [
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_adc.txt",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_adc.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_adc.txt",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_adc.csv",
        ],
        "relative_volume_candidates": [
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_volume_norm.txt",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_volume_norm.csv",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_ICVnorm.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_volume_norm.txt",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_volume_norm.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_ICVnorm.csv",
        ],
        "absolute_volume_candidates": [
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_volume.csv",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_for_volume.txt",
            DATA_ROOT / "Regional_stats" / "ADDecode" / "AD_Decode_studywide_stats_ICVregressed_mm3.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_volume.csv",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_for_volume.txt",
            DATA_ROOT / "Regional_stats" / "AD_Decode" / "AD_Decode_studywide_stats_ICVregressed_mm3.csv",
        ],
        "harmonized_dirs": [
            DATA_ROOT / "harmonization" / "ADDecode",
            DATA_ROOT / "harmonization" / "AD_Decode",
            DATA_ROOT / "pre_harmonization" / "AD_DECODE" / "DWI",
        ],
    },
}


SENTINELS = {
    -999999, -99999, -9999, -999,
    -888888, -88888, -8888, -888,
    -777777, -77777, -7777, -777,
    999, 9999, 99999, 999999,
    888, 8888, 88888, 888888,
    777, 7777, 77777, 777777,
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and append Figure 4 neuroimaging variables."
    )
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--atlas", default=str(ATLAS_PATH))
    parser.add_argument("--validation-dir-name", default=VALIDATION_DIR_NAME,
                        help="Validation subdirectory to enrich, e.g. validation_figures_full_cohort or validation_figures.")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--overwrite-original", action="store_true")
    return parser.parse_args()


def first_existing_path(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def is_probably_subject_column(col: str) -> bool:
    """Return True for columns that look like subject/session measurements.

    This prevents metadata/helper columns such as HABS *_assignments from being
    interpreted as real subjects, and keeps ROI/structure/index columns out of
    the feature tables. It also accepts AD_DECODE subject columns that carry
    acquisition suffixes, such as S02877_master_T or S02877_T1.
    """
    c = str(col).strip()
    cl = c.lower()

    if not c or cl in {"nan", "none", "unnamed: 0"}:
        return False

    metadata_like = {
        "roi", "roi_num", "index", "index2", "index3", "structure",
        "hemisphere", "subject", "subject_id", "participant_id", "id",
        "runno", "file", "regional_id", "connectome_key",
    }
    if cl in metadata_like:
        return False

    # HABS files can contain helper assignment columns next to the real subject
    # columns, e.g. H4369_y0_assignments. These are not measurements.
    if "assignment" in cl or cl.endswith("_assignments"):
        return False

    # Clean subject/session IDs observed in these data.
    subject_patterns = [
        r"^[A-Za-z]\d{4,5}_y\d+$",  # R0072_y0, H4369_y2
        r"^[A-Za-z]\d{5}$",          # S00775, J01257
        r"^[A-Za-z]\d{4}$",          # D0007, H4369, R0072
        r"^ADRC\d{4}$",              # legacy ADRC label; normalized later
    ]
    if any(re.match(pat, c, flags=re.IGNORECASE) for pat in subject_patterns):
        return True

    # AD_DECODE columns often append acquisition/run suffixes to the subject ID.
    # Reuse subject_match_key so S02877_master_T*, S02877_temp_T*, S02877_T1*,
    # etc. still count as subject columns and merge under s02877.
    key = subject_match_key(c)
    return bool(re.match(r"^[a-z]\d{4,5}(_y\d+)?$", key, flags=re.IGNORECASE))

def select_metric_value_column(df: pd.DataFrame, metric_kind: str) -> Optional[str]:
    """Best-effort metric-value column selector for long-format tables.

    Some AD_DECODE regional files are not ROI x subject wide matrices. They may
    instead be long tables with columns like subject, index/ROI, and FA/RD/AD/ADC.
    This selector lets the same feature builder handle both layouts.
    """
    metric_aliases = {
        "fa": ["fa", "mrtrixfa", "mrtrix_fa", "fractional_anisotropy"],
        "rd": ["rd", "radial_diffusivity"],
        "ad": ["ad", "axial_diffusivity"],
        "adc": ["adc", "md", "mean_diffusivity"],
        "relative_volume": ["brainpct", "volume_norm", "icvnorm", "rel_volume", "relative_volume", "pct"],
        "absolute_volume": ["brainabs", "volume", "volume_mm3", "icvregressed_mm3", "abs_volume"],
    }

    aliases = metric_aliases.get(metric_kind, [])
    metadata_like = {
        "roi", "roi_num", "index", "index2", "index3", "structure",
        "hemisphere", "subject", "subject_id", "participant_id", "id",
        "runno", "file", "regional_id", "connectome_key",
    }

    lower_to_col = {str(c).strip().lower(): c for c in df.columns}

    # Exact normalized matches first.
    for alias in aliases:
        if alias.lower() in lower_to_col:
            return lower_to_col[alias.lower()]

    # Then substring matches.
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in metadata_like or "assignment" in cl:
            continue
        if any(alias.lower() in cl for alias in aliases):
            return c

    # Final fallback: if exactly one non-metadata numeric column exists, use it.
    numeric_candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in metadata_like or "assignment" in cl:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() > 0:
            numeric_candidates.append(c)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    return None

def read_table_auto(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not Path(path).exists():
        return None

    path = Path(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    # Try likely delimiters first, then validate.
    attempts = []

    if path.suffix.lower() == ".csv":
        attempts.extend([
            dict(sep=",", encoding="utf-8-sig"),
            dict(sep=",", encoding="cp1252"),
            dict(sep=None, engine="python", encoding="utf-8-sig"),
            dict(sep="\t", encoding="utf-8-sig"),
            dict(sep=None, engine="python", encoding="cp1252"),
            dict(sep="\t", encoding="cp1252"),
        ])
    else:
        attempts.extend([
            dict(sep="\t", encoding="utf-8-sig"),
            dict(sep=",", encoding="utf-8-sig"),
            dict(sep=None, engine="python", encoding="utf-8-sig"),
            dict(sep="\t", encoding="cp1252"),
            dict(sep=",", encoding="cp1252"),
            dict(sep=None, engine="python", encoding="cp1252"),
        ])

    last_error = None

    for kwargs in attempts:
        try:
            df = pd.read_csv(path, low_memory=False, **kwargs)
            df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

            # Accept only if it actually split into columns.
            if df.shape[1] > 1:
                return df

            # If it is one column but header contains commas/tabs, delimiter failed.
            only_col = str(df.columns[0])
            if "," in only_col or "\t" in only_col:
                continue

            return df

        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read table with valid delimiter: {path}. Last error: {last_error}")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return out


def clean_numeric(x) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    for v in SENTINELS:
        s = s.mask(s == v, np.nan)
    return s.replace([np.inf, -np.inf], np.nan)


def normalize_key(x) -> str:
    """
    General normalized key.

    Preserves ADNI/HABS visit suffixes such as _y0/_y2/_y4.
    Strips AD-DECODE acquisition suffixes such as _master_T/_temp_T/_T/_T1/_T2.
    """
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"\.csv$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_connectomics$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_conn_plain$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"conn_plain$", "", s, flags=re.IGNORECASE)

    # AD-DECODE-style suffixes only. Do not remove _y0/_y2/_y4.
    s = re.sub(r"_master_T.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_temp_T.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_T1.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_T2.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_T.*$", "", s, flags=re.IGNORECASE)

    return s.lower()


def compact_key(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_key(x))


def adrc_legacy_to_d_id(x) -> Optional[str]:
    """Map legacy ADRC#### IDs to D#### IDs used by validation/regional tables."""
    m = re.search(r"(?:^|[^A-Za-z0-9])ADRC(\d{4})(?:[^0-9]|$)", str(x), flags=re.IGNORECASE)
    if m:
        return f"D{m.group(1)}"
    return None


def subject_match_key(x) -> str:
    """
    Extract biologically meaningful subject/session key.

    Priority:
    1. Visit/session IDs, preserving ADNI/HABS visits:
        R0072_y0 -> r0072_y0
        H4369_y2 -> h4369_y2

    2. Generic one-letter + five digits:
        S00775 -> s00775
        J01257 -> j01257

    3. Generic one-letter + four digits:
        D0007 -> d0007
        H4369 -> h4369
        R0072 -> r0072

    This searches the full input string, so it works for filenames, columns,
    and full paths.
    """
    s = str(x).strip()

    # Legacy ADRC connectome files are named ADRC####, while the validation and
    # regional tables use D####. Normalize early so all merge keys agree.
    adrc_id = adrc_legacy_to_d_id(s)
    if adrc_id is not None:
        return adrc_id.lower()

    s = re.sub(r"\.csv$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_conn_plain$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"conn_plain$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_connectomics$", "", s, flags=re.IGNORECASE)

    # Preserve visit-specific ADNI/HABS IDs first.
    m = re.search(r"\b([A-Za-z]\d{4,5}_y\d+)\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Remove AD-DECODE acquisition suffixes before generic matching.
    s2 = re.sub(r"_master_T.*$", "", s, flags=re.IGNORECASE)
    s2 = re.sub(r"_temp_T.*$", "", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"_T1.*$", "", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"_T2.*$", "", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"_T.*$", "", s2, flags=re.IGNORECASE)

    # Generic one-letter + five digits, e.g. S00775, J01257.
    m = re.search(r"\b([A-Za-z]\d{5})\b", s2, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Generic one-letter + four digits, e.g. D0007, H4369, R0072.
    m = re.search(r"\b([A-Za-z]\d{4})\b", s2, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return normalize_key(s2)


def possible_id_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "connectome_key",
        "regional_id",
        "Subject_ID",
        "subject_id",
        "SUBJECT_ID",
        "Subject",
        "subject",
        "SUBJECT",
        "participant_id",
        "Participant_ID",
        "participant",
        "Participant",
        "subj",
        "Subj",
        "SUBJ",
        "scan_id",
        "Scan_ID",
        "session_id",
        "graph_id",
        "runno",
        "RID",
        "file",
        "File",
        "filename",
        "Filename",
        "ID",
        "id",
    ]
    return [c for c in candidates if c in df.columns]

def validation_path(results_root: Path, cohort: str, feature_set: str) -> Path:
    return (
        Path(results_root)
        / PREDICTION_DIRS[cohort]
        / f"ablation_{feature_set}"
        / VALIDATION_DIR_NAME
        / VALIDATION_INPUT
    )


def auto_connectome_workers(n_files: int) -> int:
    """Choose a conservative worker count for connectome graph metrics.

    This avoids the previous failure mode where large shared servers spawned
    hundreds of workers. The cap is intentionally conservative because each
    worker imports pandas/networkx and reads matrices from disk.
    """
    if n_files <= 1:
        return 1

    cpu_count = os.cpu_count() or 1
    return max(1, min(16, cpu_count - 1, n_files))


# =============================================================================
# ATLAS MAPPING
# =============================================================================

def hippocampus_mapping(atlas_path: Path):
    atlas = read_table_auto(atlas_path)
    if atlas is None:
        raise FileNotFoundError(atlas_path)

    atlas = clean_columns(atlas)

    required = {"index", "index2", "Structure"}
    missing = required.difference(set(atlas.columns))
    if missing:
        raise ValueError(f"Atlas missing required columns: {missing}")

    hip = atlas[
        atlas["Structure"].astype(str).str.contains("Hippocampus", case=False, na=False)
    ].copy()

    if hip.empty:
        raise ValueError("No hippocampus rows found in atlas.")

    hip["python_zero_based_position"] = pd.to_numeric(hip["index2"], errors="coerce") - 1

    roi_labels = (
        pd.to_numeric(hip["index"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    node_positions = (
        pd.to_numeric(hip["python_zero_based_position"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    roi_labels = sorted(set(roi_labels))
    node_positions = sorted(set(node_positions))

    if roi_labels != [17, 53]:
        print("WARNING: unexpected hippocampus ROI labels:", roi_labels)

    if node_positions != [5, 13]:
        print("WARNING: unexpected hippocampus node positions:", node_positions)

    return roi_labels, node_positions, hip


# =============================================================================
# REGIONAL FEATURES
# =============================================================================

def regional_wide_to_features(path: Optional[Path], hip_roi_labels: Sequence[int], metric_kind: str):
    """
    metric_kind:
        fa, rd, ad, adc, relative_volume, absolute_volume
    """
    df = read_table_auto(path)

    info = {
        "path": str(path) if path else "",
        "exists": df is not None,
        "status": "",
        "metric_kind": metric_kind,
        "n_roi_rows": 0,
        "n_subject_columns": 0,
        "roi_source_col": "",
    }

    if df is None:
        info["status"] = "missing_file"
        return pd.DataFrame(), info

    df = clean_columns(df)

    # AD-DECODE may have atlas ROI label in "index".
    # ADNI/ADRC/HABS generally use "ROI".
    if "index" in df.columns:
        roi_source_col = "index"
    elif "ROI" in df.columns:
        roi_source_col = "ROI"
    else:
        roi_source_col = df.columns[0]

    df["ROI_num"] = pd.to_numeric(df[roi_source_col], errors="coerce")
    df = df[df["ROI_num"].notna()].copy()
    df["ROI_num"] = df["ROI_num"].astype(int)

    metadata_cols = {
        "ROI",
        "ROI_num",
        "Index2",
        "index",
        "Structure",
        "structure",
        "Hemisphere",
        "hemisphere",
        "index3",
    }

    subject_cols = [
        c for c in df.columns
        if c not in metadata_cols and is_probably_subject_column(c)
    ]

    info["n_roi_rows"] = len(df)
    info["n_subject_columns"] = len(subject_cols)
    info["roi_source_col"] = roi_source_col

    # Exclude exterior/background/brain rows for ROI-wise means/sums.
    # Your files use ROI=-1 for Brain and ROI=0 for exterior/background.
    df_regional = df[~df["ROI_num"].isin([-1, 0])].copy()

    # Long-format fallback, mainly for AD_DECODE-style files.
    # Expected shape: one subject/id column + one ROI/index column + one metric value column.
    if not subject_cols:
        id_cols = possible_id_columns(df)
        value_col = select_metric_value_column(df, metric_kind)

        if id_cols and value_col is not None:
            id_col = id_cols[0]
            info["status"] = "ok_long_format"
            info["long_id_col"] = id_col
            info["long_value_col"] = str(value_col)
            rows = []

            brain_row_long = df[df["ROI_num"].eq(-1)].copy()

            for subj, g in df.groupby(id_col, dropna=True):
                g_regional = g[~g["ROI_num"].isin([-1, 0])].copy()
                g_hip = g_regional[g_regional["ROI_num"].isin(hip_roi_labels)].copy()

                vals_all = clean_numeric(g_regional[value_col])
                vals_hip = clean_numeric(g_hip[value_col]) if not g_hip.empty else pd.Series(dtype=float)

                row = {
                    "subject_source": subj,
                    "subject_norm": normalize_key(subj),
                    "subject_compact": compact_key(subj),
                    "subject_match": subject_match_key(subj),
                }

                if metric_kind == "fa":
                    row["Hc_FA"] = vals_hip.mean(skipna=True) if len(vals_hip) else np.nan
                    row["Total_Brain_FA"] = vals_all.mean(skipna=True)
                elif metric_kind == "rd":
                    row["Hc_RD"] = vals_hip.mean(skipna=True) if len(vals_hip) else np.nan
                elif metric_kind == "ad":
                    row["Hc_AD"] = vals_hip.mean(skipna=True) if len(vals_hip) else np.nan
                elif metric_kind == "adc":
                    row["Hc_ADC"] = vals_hip.mean(skipna=True) if len(vals_hip) else np.nan
                elif metric_kind == "relative_volume":
                    row["Hc_volume_pct_brain"] = vals_hip.sum(skipna=True) if len(vals_hip) else np.nan
                elif metric_kind == "absolute_volume":
                    row["Hc_volume_mm3"] = vals_hip.sum(skipna=True) if len(vals_hip) else np.nan
                    subj_brain = brain_row_long[brain_row_long[id_col].astype(str).eq(str(subj))]
                    if not subj_brain.empty:
                        row["Total_Brain_volume"] = clean_numeric(subj_brain[value_col]).iloc[0]
                    else:
                        row["Total_Brain_volume"] = vals_all.sum(skipna=True)

                rows.append(row)

            return pd.DataFrame(rows), info

        info["status"] = "no_subject_columns"
        info["long_value_col"] = str(value_col) if value_col is not None else ""
        return pd.DataFrame(), info

    hip = df_regional[df_regional["ROI_num"].isin(hip_roi_labels)].copy()
    if hip.empty:
        info["status"] = "hippocampus_rows_missing"
    else:
        info["status"] = "ok"

    # Brain row if available; used for total absolute brain volume.
    brain_row = df[df["ROI_num"].eq(-1)].copy()

    if brain_row.empty:
        for scol in ["Structure", "structure"]:
            if scol in df.columns:
                brain_row = df[df[scol].astype(str).str.lower().eq("brain")].copy()
                if not brain_row.empty:
                    break

    rows = []

    for subj in subject_cols:
        vals_all = clean_numeric(df_regional[subj])
        vals_hip = clean_numeric(hip[subj]) if not hip.empty else pd.Series(dtype=float)

        row = {
            "subject_source": subj,
            "subject_norm": normalize_key(subj),
            "subject_compact": compact_key(subj),
            "subject_match": subject_match_key(subj),
        }

        if metric_kind == "fa":
            row["Hc_FA"] = vals_hip.mean(skipna=True) if len(vals_hip) else np.nan
            row["Total_Brain_FA"] = vals_all.mean(skipna=True)

        elif metric_kind == "rd":
            row["Hc_RD"] = vals_hip.mean(skipna=True) if len(vals_hip) else np.nan

        elif metric_kind == "ad":
            row["Hc_AD"] = vals_hip.mean(skipna=True) if len(vals_hip) else np.nan

        elif metric_kind == "adc":
            row["Hc_ADC"] = vals_hip.mean(skipna=True) if len(vals_hip) else np.nan

        elif metric_kind == "relative_volume":
            row["Hc_volume_pct_brain"] = (
                vals_hip.sum(skipna=True) if len(vals_hip) else np.nan
            )

        elif metric_kind == "absolute_volume":
            row["Hc_volume_mm3"] = (
                vals_hip.sum(skipna=True) if len(vals_hip) else np.nan
            )

            if not brain_row.empty and subj in brain_row.columns:
                row["Total_Brain_volume"] = clean_numeric(brain_row[subj]).iloc[0]
            else:
                row["Total_Brain_volume"] = vals_all.sum(skipna=True)

        rows.append(row)

    return pd.DataFrame(rows), info


def first_non_missing_value(series: pd.Series):
    """Return the first non-missing value in original row order."""
    for value in series:
        try:
            if pd.notna(value):
                return value
        except Exception:
            if value is not None:
                return value
    return np.nan


def collapse_duplicate_subject_rows(df: pd.DataFrame, label: str = "features") -> pd.DataFrame:
    """Collapse duplicate subject_match rows, keeping one row per biological key.

    This is important for AD_DECODE regional stat files, where the same connectome
    root can appear many times as columns such as S02877_master_T,
    S02877_temp_T, S02877_master_T_1, and S02877. All of those normalize to the
    same subject_match (s02877). Keeping them as separate rows causes the output
    to explode to thousands of rows. We collapse them in input order and keep the
    first non-missing value for each column.
    """
    if df is None or df.empty or "subject_match" not in df.columns:
        return df

    out = df.copy()
    out["subject_match"] = out["subject_match"].astype(str)

    before_rows = len(out)
    before_keys = out["subject_match"].nunique(dropna=True)
    duplicate_rows = before_rows - before_keys

    if duplicate_rows <= 0:
        return out

    # Preserve the first appearance order of subject_match.
    order = (
        out[["subject_match"]]
        .drop_duplicates(subset=["subject_match"])
        .assign(_subject_order=lambda x: np.arange(len(x)))
    )

    grouped = (
        out.groupby("subject_match", sort=False, dropna=False)
        .agg(first_non_missing_value)
        .reset_index()
    )

    grouped = grouped.merge(order, on="subject_match", how="left")
    grouped = grouped.sort_values("_subject_order").drop(columns=["_subject_order"])

    print(
        f"Collapsed duplicate subject rows for {label}: "
        f"{before_rows} rows -> {len(grouped)} rows "
        f"({duplicate_rows} duplicate rows collapsed)."
    )

    return grouped


def merge_feature_tables(tables):
    """
    Merge feature tables by canonical subject/session key first.
    This fixes cases where volume files, diffusion files, and connectome files
    use slightly different labels.
    """
    tables = [t for t in tables if t is not None and not t.empty]
    if not tables:
        return pd.DataFrame()

    cleaned = []

    for t in tables:
        t = t.copy()

        if "subject_norm" not in t.columns:
            t["subject_norm"] = t["subject_source"].map(normalize_key)

        if "subject_compact" not in t.columns:
            t["subject_compact"] = t["subject_source"].map(compact_key)

        if "subject_match" not in t.columns:
            t["subject_match"] = t["subject_source"].map(subject_match_key)

        cleaned.append(t)

    out = cleaned[0]

    for t in cleaned[1:]:
        out = out.merge(
            t,
            on=["subject_match"],
            how="outer",
            suffixes=("", "_dup"),
        )

        for c in list(out.columns):
            if c.endswith("_dup"):
                base = c.replace("_dup", "")

                if base in out.columns:
                    out[base] = out[base].combine_first(out[c])
                    out = out.drop(columns=[c])
                else:
                    out = out.rename(columns={c: base})

    return out


def build_regional_features(cohort: str, hip_roi_labels: Sequence[int]):
    cfg = REGIONAL_CONFIG[cohort]

    specs = [
        ("fa", "regional_fa", "fa_candidates"),
        ("rd", "regional_rd", "rd_candidates"),
        ("ad", "regional_ad", "ad_candidates"),
        ("adc", "regional_adc", "adc_candidates"),
        ("relative_volume", "regional_relative_volume", "relative_volume_candidates"),
        ("absolute_volume", "regional_absolute_volume", "absolute_volume_candidates"),
    ]

    tables = []
    source_rows = []

    for metric_kind, source_type, key in specs:
        selected = first_existing_path(cfg.get(key, []))

        df_metric, info = regional_wide_to_features(
            selected,
            hip_roi_labels=hip_roi_labels,
            metric_kind=metric_kind,
        )

        info.update({
            "cohort": cohort,
            "source_type": source_type,
            "selected_path": str(selected) if selected else "",
            "all_candidates": " | ".join(map(str, cfg.get(key, []))),
        })

        if not df_metric.empty:
            before_rows = len(df_metric)
            before_keys = int(df_metric["subject_match"].nunique()) if "subject_match" in df_metric.columns else 0
            df_metric = collapse_duplicate_subject_rows(
                df_metric,
                label=f"{cohort} {metric_kind} regional",
            )
            info["n_feature_rows_before_dedup"] = int(before_rows)
            info["n_feature_rows_after_dedup"] = int(len(df_metric))
            info["n_feature_keys_before_dedup"] = int(before_keys)
            info["n_duplicate_subject_rows_collapsed"] = int(before_rows - len(df_metric))

        source_rows.append(info)

        if not df_metric.empty:
            tables.append(df_metric)

    features = merge_feature_tables(tables)
    features = collapse_duplicate_subject_rows(features, label=f"{cohort} regional combined")

    if not features.empty:
        features["cohort"] = cohort

    return features, source_rows


# =============================================================================
# CONNECTOME FEATURES
# =============================================================================



def is_real_connectome_csv(path: Path) -> bool:
    """Return False for helper/sidecar CSVs that are not adjacency matrices."""
    p = Path(path)
    text = "/".join(part.lower() for part in p.parts)
    name = p.name.lower()

    # HABS has files such as H4916_y2_assignments_conn_plain.csv under an
    # assignments directory. They match *conn_plain.csv but are not matrices.
    bad_tokens = [
        "assignment",
        "assignments",
        "label",
        "labels",
        "node",
        "nodes",
        "lookup",
        "manifest",
        "qc",
    ]
    if any(tok in text for tok in bad_tokens):
        return False

    if not (name.endswith("_conn_plain.csv") or "conn_plain" in name):
        return False

    return True
def discover_connectomes(cohort: str) -> list[Path]:
    cfg = REGIONAL_CONFIG[cohort]
    files = []

    for root in cfg["harmonized_dirs"]:
        root = Path(root)
        if not root.exists():
            continue

        files.extend(root.rglob("*_conn_plain.csv"))
        files.extend(root.rglob("*conn_plain*.csv"))

    unique = {}
    skipped_sidecars = 0
    for p in files:
        if not is_real_connectome_csv(p):
            skipped_sidecars += 1
            continue
        unique[str(p)] = p

    out = list(unique.values())
    if skipped_sidecars:
        print(f"Skipped {skipped_sidecars} connectome sidecar/helper CSV(s) for {cohort}.")

    return out

def extract_connectome_subject(path: Path, cohort: Optional[str] = None) -> str:
    """
    Extract subject/session key from connectome path.

    Filename/stem are searched before full path so ADRC parent folders such as
    ADRC0001 do not override true filenames such as D0007_conn_plain.csv.

    Supports:
        R0072_y0_conn_plain.csv
        H4369_y0_conn_plain.csv
        D0007_conn_plain.csv
        S00775_conn_plain.csv
        J01257_conn_plain.csv
        any [letter][4-5 digits]_conn_plain.csv
    """
    p = Path(path)

    # Legacy ADRC connectome files are named ADRC####_conn_plain.csv, but the
    # validation/regional subject IDs use D####. Convert before generic matching.
    # This is intentionally not gated on `cohort` so debug calls and worker calls
    # cannot accidentally fall through to ADRC####.
    for text_candidate in [p.name, p.stem, str(p)]:
        adrc_id = adrc_legacy_to_d_id(text_candidate)
        if adrc_id is not None:
            return adrc_id

    patterns = [
        r"\b([A-Za-z]\d{4,5}_y\d+)\b",  # ADNI/HABS visit IDs
        r"\b([A-Za-z]\d{5})\b",         # S00775, J01257
        r"\b([A-Za-z]\d{4})\b",         # D0007, H4369, R0072
    ]

    # Filename/stem first.
    for text_candidate in [p.name, p.stem]:
        for pat in patterns:
            m = re.search(pat, text_candidate, flags=re.IGNORECASE)
            if m:
                return m.group(1)

    # Full path fallback only if filename/stem did not work.
    full_text = str(p)
    for pat in patterns:
        m = re.search(pat, full_text, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    stem = p.name
    stem = re.sub(r"\.csv$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_conn_plain$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"conn_plain$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_assignments$", "", stem, flags=re.IGNORECASE)
    return stem


def threshold_connectome(arr, keep_percentile=95.0):
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)

    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Non-square connectome: {arr.shape}")

    mask = ~np.eye(arr.shape[0], dtype=bool)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        np.fill_diagonal(arr, 0.0)
        return arr

    threshold_value = np.percentile(vals, 100.0 - keep_percentile)

    out = np.where(arr >= threshold_value, arr, 0.0)
    np.fill_diagonal(out, 0.0)

    return out


def compute_hc_graph_metrics(
    connectome_path: Path,
    hip_nodes_zero_based: Sequence[int],
    threshold_percentile: float,
    min_weight: float,
):
    """
    Compute hippocampal and whole-connectome graph metrics from one DWI connectome.

    This version keeps the original Figure 4 hippocampal metrics:
        Hc_clustering_coeff
        Hc_path_length

    and adds the four global graph metrics used in the older graph code:
        Total_graph_clustering_coeff
        Total_graph_path_length
        Global_Efficiency
        Local_Efficiency

    Global metric logic follows the same design as 0_ClusteringCoeff_PathLength.py:
        - threshold matrix
        - log1p transform
        - weighted average clustering
        - weighted path length using distance = 1 / weight and largest connected component
        - NetworkX global_efficiency
        - NetworkX local_efficiency
    """
    raw = pd.read_csv(connectome_path, header=None)
    mat = raw.to_numpy(dtype=float)

    mat = threshold_connectome(
        mat,
        keep_percentile=threshold_percentile,
    )

    mat = np.log1p(mat)
    mat = np.where(np.isfinite(mat), mat, 0.0)
    mat[mat <= min_weight] = 0.0
    np.fill_diagonal(mat, 0.0)

    matrix = pd.DataFrame(mat)

    # -------------------------------------------------------------------------
    # Whole-connectome graph metrics, matching the older graph-code logic.
    # -------------------------------------------------------------------------
    G = nx.from_numpy_array(matrix.to_numpy())

    for u, v, d in G.edges(data=True):
        weight = float(matrix.iloc[u, v])
        d["weight"] = weight
        d["distance"] = 1.0 / weight if weight > 0 else float("inf")

    try:
        total_clustering = nx.average_clustering(G, weight="weight")
    except Exception:
        total_clustering = np.nan

    try:
        G_path = G
        if not nx.is_connected(G_path):
            G_path = G_path.subgraph(max(nx.connected_components(G_path), key=len)).copy()
        total_path = nx.average_shortest_path_length(G_path, weight="distance")
    except Exception:
        total_path = np.nan

    try:
        global_eff = nx.global_efficiency(G)
    except Exception:
        global_eff = np.nan

    try:
        local_eff = nx.local_efficiency(G)
    except Exception:
        local_eff = np.nan

    # -------------------------------------------------------------------------
    # Hippocampal graph metrics.
    # -------------------------------------------------------------------------
    valid_nodes = [n for n in hip_nodes_zero_based if n in G.nodes]

    if not valid_nodes:
        hc_clustering = np.nan
        hc_path = np.nan
        graph_status = "hip_nodes_not_in_graph"
    else:
        try:
            clustering = nx.clustering(G, nodes=valid_nodes, weight="weight")
            hc_clustering = np.nanmean(list(clustering.values()))
        except Exception:
            hc_clustering = np.nan

        path_values = []
        for node in valid_nodes:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, node, weight="distance")
                path_values.extend([
                    v for target, v in lengths.items()
                    if target != node and np.isfinite(v)
                ])
            except Exception:
                pass

        hc_path = np.nanmean(path_values) if path_values else np.nan
        graph_status = "ok"

    def finite_or_nan(x):
        return float(x) if np.isfinite(x) else np.nan

    return {
        "Hc_clustering_coeff": finite_or_nan(hc_clustering),
        "Hc_path_length": finite_or_nan(hc_path),
        "Total_graph_clustering_coeff": finite_or_nan(total_clustering),
        "Total_graph_path_length": finite_or_nan(total_path),
        "Global_Efficiency": finite_or_nan(global_eff),
        "Local_Efficiency": finite_or_nan(local_eff),
        "graph_status": graph_status,
    }


def compute_connectome_feature_row(args_tuple):
    """Worker function for one connectome file. Must stay top-level for multiprocessing."""
    p, cohort, hip_nodes_zero_based, threshold_percentile, min_weight = args_tuple
    p = Path(p)
    subject = extract_connectome_subject(p, cohort=cohort)

    try:
        metrics = compute_hc_graph_metrics(
            p,
            hip_nodes_zero_based=hip_nodes_zero_based,
            threshold_percentile=threshold_percentile,
            min_weight=min_weight,
        )
    except Exception as exc:
        metrics = {
            "Hc_clustering_coeff": np.nan,
            "Hc_path_length": np.nan,
            "graph_status": f"error: {exc}",
        }

    return {
        "subject_source": subject,
        "subject_norm": normalize_key(subject),
        "subject_compact": compact_key(subject),
        "subject_match": subject_match_key(subject),
        "cohort": cohort,
        "connectome_path": str(p),
        **metrics,
    }


def build_connectome_features(
    cohort: str,
    hip_nodes_zero_based: Sequence[int],
    threshold_percentile: float,
    min_weight: float,
):
    files = discover_connectomes(cohort)
    n_jobs = auto_connectome_workers(len(files))

    if cohort in ["ADRC", "HABS", "AD_DECODE"]:
        print(f"\n{cohort} connectome filename extraction check:")
        for pp in files[:10]:
            print(Path(pp).name, "->", extract_connectome_subject(pp, cohort=cohort), "|", pp)

    worker_args = [
        (str(p), cohort, tuple(hip_nodes_zero_based), threshold_percentile, min_weight)
        for p in files
    ]

    if n_jobs == 1 or len(worker_args) <= 1:
        rows = [compute_connectome_feature_row(a) for a in worker_args]
    else:
        print(f"Computing connectome metrics with {n_jobs} worker processes...")
        # chunksize reduces scheduling overhead when many small connectome files exist.
        chunksize = max(1, len(worker_args) // (n_jobs * 4))
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            rows = list(executor.map(
                compute_connectome_feature_row,
                worker_args,
                chunksize=chunksize,
            ))

    df_conn = pd.DataFrame(rows)
    before_conn_rows = len(df_conn)
    before_conn_keys = int(df_conn["subject_match"].nunique()) if (not df_conn.empty and "subject_match" in df_conn.columns) else 0
    df_conn = collapse_duplicate_subject_rows(df_conn, label=f"{cohort} connectomes")

    info = {
        "cohort": cohort,
        "n_connectome_files": len(files),
        "n_connectome_rows_before_dedup": int(before_conn_rows),
        "n_connectome_rows": int(len(df_conn)),
        "n_connectome_keys_before_dedup": int(before_conn_keys),
        "n_duplicate_connectome_rows_collapsed": int(before_conn_rows - len(df_conn)),
        "n_jobs": n_jobs,
    }

    return df_conn, info


# =============================================================================
# MERGE INTO VALIDATION FILES
# =============================================================================

def coalesce_existing_global_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mapping = {
        "Total_graph_clustering_coeff": [
            "Total_graph_clustering_coeff",
            "Clustering_Coeff_raw_clean",
            "Clustering_Coeff",
        ],
        "Total_graph_path_length": [
            "Total_graph_path_length",
            "Path_Length_raw_clean",
            "Path_Length",
        ],
        "Global_Efficiency": [
            "Global_Efficiency",
            "Global_Efficiency_raw_clean",
        ],
        "Local_Efficiency": [
            "Local_Efficiency",
            "Local_Efficiency_raw_clean",
        ],
    }

    for canonical, candidates in mapping.items():
        if canonical not in out.columns:
            out[canonical] = np.nan

        for c in candidates:
            if c in out.columns:
                out[canonical] = out[canonical].combine_first(clean_numeric(out[c]))

    return out


def attach_features(df_val: pd.DataFrame, features: pd.DataFrame):
    out = df_val.copy()

    if features.empty:
        return out, {
            "merge_status": "empty_feature_table",
            "merge_id_col": "",
            "merge_key_col": "",
            "n_matched": 0,
        }

    id_cols = possible_id_columns(out)

    if not id_cols:
        return out, {
            "merge_status": "no_id_columns",
            "merge_id_col": "",
            "merge_key_col": "",
            "n_matched": 0,
        }

    best = None
    best_n = -1

    for id_col in id_cols:
        tmp = pd.DataFrame({
            "_row_index": np.arange(len(out)),
            id_col: out[id_col],
            "subject_norm": out[id_col].map(normalize_key),
            "subject_compact": out[id_col].map(compact_key),
            "subject_match": out[id_col].map(subject_match_key),
        })

        for key_col in ["subject_match", "subject_norm", "subject_compact"]:
            f = features.drop_duplicates(subset=[key_col]).copy()

            merged = tmp.merge(
                f,
                on=key_col,
                how="left",
                suffixes=("", "_feat"),
            )

            score_cols = [
                "Hc_volume_mm3",
                "Hc_volume_pct_brain",
                "Hc_FA",
                "Hc_RD",
                "Hc_AD",
                "Hc_ADC",
                "Hc_clustering_coeff",
                "Hc_path_length",
                "Total_Brain_FA",
                "Total_graph_clustering_coeff",
                "Total_graph_path_length",
                "Global_Efficiency",
                "Local_Efficiency",
            ]

            n = 0
            for c in score_cols:
                if c in merged.columns:
                    n = max(
                        n,
                        int(pd.to_numeric(merged[c], errors="coerce").notna().sum()),
                    )

            if n > best_n:
                best_n = n
                best = (id_col, key_col, merged)

    if best is None:
        return out, {
            "merge_status": "no_valid_merge",
            "merge_id_col": "",
            "merge_key_col": "",
            "n_matched": 0,
        }

    id_col, key_col, merged = best
    merged = merged.sort_values("_row_index")

    canonical_cols = [
        "Hc_volume_mm3",
        "Hc_volume_pct_brain",
        "Hc_FA",
        "Hc_RD",
        "Hc_AD",
        "Hc_ADC",
        "Hc_clustering_coeff",
        "Hc_path_length",
        "Total_Brain_volume",
        "Total_Brain_FA",
        "Total_graph_clustering_coeff",
        "Total_graph_path_length",
        "Global_Efficiency",
        "Local_Efficiency",
    ]

    for c in canonical_cols:
        if c in merged.columns:
            out[c] = merged[c].values
        elif c not in out.columns:
            out[c] = np.nan

    out = coalesce_existing_global_metrics(out)

    return out, {
        "merge_status": "ok",
        "merge_id_col": id_col,
        "merge_key_col": key_col,
        "n_matched": int(best_n),
    }



def deduplicate_validation_scan_rows(df: pd.DataFrame, cohort: str = "", feature_set: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Keep one row per scan/session ID before Figure 4 enrichment.

    This is mainly needed for HABS after the longitudinal fix, but it is safe for
    all cohorts because it deduplicates only exact duplicate scan/session IDs.
    It preserves longitudinal pairs such as H4369_y0 and H4369_y2 as separate rows.
    It never deduplicates by subject-level match_id.
    """
    if df is None or df.empty:
        return df, {
            "dedup_status": "empty_input",
            "dedup_id_col": "",
            "n_rows_before_dedup": 0,
            "n_rows_after_dedup": 0,
            "n_duplicate_scan_rows_removed": 0,
        }

    scan_id_candidates = [
        "Subject_ID",
        "connectome_key",
        "graph_id",
        "runno",
        "DWI",
        "subject_match",
        "regional_id",
    ]

    id_col = None
    best_long_count = -1

    # Prefer columns that actually contain visit-level IDs like H4369_y0 or R0072_y4.
    for c in scan_id_candidates:
        if c not in df.columns:
            continue
        vals = df[c].astype(str).str.strip().str.replace("_Y", "_y", regex=False)
        long_count = int(vals.str.contains(r"[A-Za-z]+\d+_y\d+", case=False, regex=True, na=False).sum())
        if long_count > best_long_count:
            best_long_count = long_count
            id_col = c

    if id_col is None:
        return df, {
            "dedup_status": "no_scan_id_col",
            "dedup_id_col": "",
            "n_rows_before_dedup": len(df),
            "n_rows_after_dedup": len(df),
            "n_duplicate_scan_rows_removed": 0,
        }

    out = df.copy()
    out["_scan_key_norm_for_dedup"] = (
        out[id_col]
        .astype(str)
        .str.strip()
        .str.replace("_Y", "_y", regex=False)
        .str.lower()
    )

    n_before = len(out)
    n_unique = int(out["_scan_key_norm_for_dedup"].nunique(dropna=True))
    n_dup = int(out["_scan_key_norm_for_dedup"].duplicated().sum())

    if n_dup <= 0:
        out = out.drop(columns=["_scan_key_norm_for_dedup"], errors="ignore")
        return out, {
            "dedup_status": "no_duplicates",
            "dedup_id_col": id_col,
            "n_rows_before_dedup": n_before,
            "n_rows_after_dedup": len(out),
            "n_duplicate_scan_rows_removed": 0,
            "n_unique_scan_ids_before_dedup": n_unique,
        }

    priority_cols = [
        "cBAG_global", "cBAG", "BAG", "Predicted_Age_BiasCorrected", "Predicted_Age_RAW",
        "Hc_volume_mm3", "Hc_volume_pct_brain", "Hc_FA", "Hc_RD", "Hc_AD", "Hc_ADC",
        "Hc_clustering_coeff", "Hc_path_length", "Total_Brain_volume", "Total_Brain_FA",
        "Global_Efficiency", "Local_Efficiency", "OM_BMI", "BMI", "bmi",
    ]

    out["_dedup_score"] = 0
    for c in priority_cols:
        if c in out.columns:
            out["_dedup_score"] += out[c].notna().astype(int)

    out = (
        out.sort_values(["_scan_key_norm_for_dedup", "_dedup_score"], ascending=[True, False])
        .drop_duplicates(subset=["_scan_key_norm_for_dedup"], keep="first")
        .drop(columns=["_scan_key_norm_for_dedup", "_dedup_score"], errors="ignore")
        .reset_index(drop=True)
    )

    print(
        f"[Figure4 enrichment dedup] {cohort} {feature_set}: "
        f"{n_before} -> {len(out)} rows using {id_col}; "
        f"removed {n_before - len(out)} duplicate scan rows."
    )

    return out, {
        "dedup_status": "deduplicated",
        "dedup_id_col": id_col,
        "n_rows_before_dedup": n_before,
        "n_rows_after_dedup": len(out),
        "n_duplicate_scan_rows_removed": n_before - len(out),
        "n_unique_scan_ids_before_dedup": n_unique,
    }



def enrich_validation_file(
    path: Path,
    cohort: str,
    feature_set: str,
    features: pd.DataFrame,
    overwrite_original: bool = False,
):
    if not path.exists():
        return {
            "cohort": cohort,
            "feature_set": feature_set,
            "input_path": str(path),
            "status": "missing_validation_input",
        }

    df = pd.read_csv(path, low_memory=False)

    df, dedup_info = deduplicate_validation_scan_rows(
        df,
        cohort=cohort,
        feature_set=feature_set,
    )

    enriched, merge_info = attach_features(df, features)

    out_path = path.with_name(ENRICHED_INPUT)
    enriched.to_csv(out_path, index=False)

    if overwrite_original:
        backup = path.with_name(path.stem + "_before_Figure4_enrichment.csv")
        if not backup.exists():
            df.to_csv(backup, index=False)
        enriched.to_csv(path, index=False)

    canonical_cols = [
        "Hc_volume_mm3",
        "Hc_volume_pct_brain",
        "Hc_FA",
        "Hc_RD",
        "Hc_AD",
        "Hc_ADC",
        "Hc_clustering_coeff",
        "Hc_path_length",
        "Total_Brain_volume",
        "Total_Brain_FA",
        "Total_graph_clustering_coeff",
        "Total_graph_path_length",
        "Global_Efficiency",
        "Local_Efficiency",
    ]

    counts = {}
    for c in canonical_cols:
        counts[f"n_nonnull_{c}"] = (
            int(pd.to_numeric(enriched[c], errors="coerce").notna().sum())
            if c in enriched.columns else 0
        )

    return {
        "cohort": cohort,
        "feature_set": feature_set,
        "input_path": str(path),
        "output_path": str(out_path),
        "status": "written",
        "n_rows": len(enriched),
        **dedup_info,
        **merge_info,
        **counts,
    }



# =============================================================================
# OVERLAP QA
# =============================================================================

QA_OVERLAP_FILE = "Figure4_enrichment_regional_connectome_overlap_QA.csv"

def make_overlap_qa_row(
    cohort: str,
    regional_features: pd.DataFrame,
    connectome_features: pd.DataFrame,
    regional_sources: list[dict],
    connectome_info: dict,
) -> dict:
    """Summarize subject-key overlap between regional stats and connectomes."""
    metric_cols = [
        "Hc_volume_mm3",
        "Hc_volume_pct_brain",
        "Hc_FA",
        "Hc_RD",
        "Hc_AD",
        "Hc_ADC",
        "Total_Brain_volume",
        "Total_Brain_FA",
    ]

    r = regional_features.copy() if regional_features is not None else pd.DataFrame()
    c = connectome_features.copy() if connectome_features is not None else pd.DataFrame()

    r_keys = set(r["subject_match"].dropna().astype(str)) if (not r.empty and "subject_match" in r.columns) else set()
    c_keys = set(c["subject_match"].dropna().astype(str)) if (not c.empty and "subject_match" in c.columns) else set()

    matching = sorted(r_keys & c_keys)
    regional_only = sorted(r_keys - c_keys)
    connectome_only = sorted(c_keys - r_keys)

    row = {
        "cohort": cohort,
        "n_regional_rows": int(len(r)),
        "n_connectome_rows": int(len(c)),
        "n_regional_keys": int(len(r_keys)),
        "n_connectome_keys": int(len(c_keys)),
        "n_matching_keys": int(len(matching)),
        "n_regional_only": int(len(regional_only)),
        "n_connectome_only": int(len(connectome_only)),
        "examples_matching": ";".join(matching[:20]),
        "examples_regional_only": ";".join(regional_only[:20]),
        "examples_connectome_only": ";".join(connectome_only[:20]),
        "n_connectome_files": int(connectome_info.get("n_connectome_files", 0)) if connectome_info else 0,
        "n_connectome_rows_before_dedup": int(connectome_info.get("n_connectome_rows_before_dedup", 0)) if connectome_info else 0,
        "n_duplicate_connectome_rows_collapsed": int(connectome_info.get("n_duplicate_connectome_rows_collapsed", 0)) if connectome_info else 0,
        "n_jobs": int(connectome_info.get("n_jobs", 0)) if connectome_info else 0,
    }

    for m in metric_cols:
        if not r.empty and m in r.columns and "subject_match" in r.columns:
            keys_with_metric = set(
                r.loc[pd.to_numeric(r[m], errors="coerce").notna(), "subject_match"]
                .dropna()
                .astype(str)
            )
            row[f"n_regional_keys_with_{m}"] = int(len(keys_with_metric))
            row[f"n_matching_keys_with_{m}"] = int(len(keys_with_metric & c_keys))
        else:
            row[f"n_regional_keys_with_{m}"] = 0
            row[f"n_matching_keys_with_{m}"] = 0

    if regional_sources:
        for src in regional_sources:
            metric = src.get("metric_kind", "")
            if not metric:
                continue
            prefix = f"source_{metric}"
            row[f"{prefix}_status"] = src.get("status", "")
            row[f"{prefix}_subject_cols"] = src.get("n_subject_columns", 0)
            row[f"{prefix}_roi_rows"] = src.get("n_roi_rows", 0)
            row[f"{prefix}_rows_before_dedup"] = src.get("n_feature_rows_before_dedup", 0)
            row[f"{prefix}_rows_after_dedup"] = src.get("n_feature_rows_after_dedup", 0)
            row[f"{prefix}_duplicate_rows_collapsed"] = src.get("n_duplicate_subject_rows_collapsed", 0)
            row[f"{prefix}_path"] = src.get("selected_path", src.get("path", ""))

    return row


def write_overlap_qa_file(rows: list[dict], outdir: Path, label: str = "") -> Path:
    """Write QA overlap CSV and fail loudly if it was not created."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / QA_OVERLAP_FILE

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
    else:
        pd.DataFrame([{
            "cohort": "__initialized__",
            "n_regional_rows": 0,
            "n_connectome_rows": 0,
            "n_regional_keys": 0,
            "n_connectome_keys": 0,
            "n_matching_keys": 0,
            "n_regional_only": 0,
            "n_connectome_only": 0,
        }]).to_csv(out_path, index=False)

    if not out_path.exists():
        raise RuntimeError(f"QA overlap file was not created: {out_path}")

    suffix = f" ({label})" if label else ""
    print(f"QA overlap file written{suffix}: {out_path}")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    global VALIDATION_DIR_NAME
    VALIDATION_DIR_NAME = args.validation_dir_name

    results_root = Path(args.results_root)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Figure 4 enrichment")
    print("=" * 100)
    print("WORK:", WORK)
    print("DATA_ROOT:", DATA_ROOT)
    print("RESULTS_ROOT:", results_root)
    print("Validation directory:", VALIDATION_DIR_NAME)
    print("Atlas:", args.atlas)
    print("Output QC:", OUTDIR)
    print("Threshold percentile:", args.threshold_percentile)
    print("Minimum graph edge weight:", args.min_weight)
    print("Connectome worker processes: automatic, capped at 16")

    hip_roi_labels, hip_nodes_zero_based, hip_atlas = hippocampus_mapping(Path(args.atlas))

    print("\nHippocampus mapping:")
    print("Regional ROI labels:", hip_roi_labels)
    print("Connectome node positions, zero-based:", hip_nodes_zero_based)
    print(hip_atlas.to_string(index=False))

    all_features = []
    regional_source_rows = []
    connectome_source_rows = []
    validation_manifest_rows = []
    overlap_qa_rows = []
    qa_overlap_path = write_overlap_qa_file(overlap_qa_rows, OUTDIR, label="initialized")

    for cohort in COHORTS:
        print("\n" + "=" * 100)
        print("COHORT:", cohort)
        print("=" * 100)

        regional_features, regional_sources = build_regional_features(
            cohort,
            hip_roi_labels=hip_roi_labels,
        )

        regional_source_rows.extend(regional_sources)

        print("Regional source status:")
        for src in regional_sources:
            print(
                "  {metric}: {status} | rows={rows} | subject_cols={cols} | roi_col={roi} | value_col={value} | {path}".format(
                    metric=src.get("metric_kind", ""),
                    status=src.get("status", ""),
                    rows=src.get("n_roi_rows", 0),
                    cols=src.get("n_subject_columns", 0),
                    roi=src.get("roi_source_col", ""),
                    value=src.get("long_value_col", ""),
                    path=src.get("selected_path", ""),
                )
                + (
                    f" | dedup {src.get('n_feature_rows_before_dedup')}->{src.get('n_feature_rows_after_dedup')}"
                    if src.get("n_duplicate_subject_rows_collapsed", 0)
                    else ""
                )
            )

        connectome_features, connectome_info = build_connectome_features(
            cohort,
            hip_nodes_zero_based=hip_nodes_zero_based,
            threshold_percentile=args.threshold_percentile,
            min_weight=args.min_weight,
        )

        connectome_source_rows.append(connectome_info)

        # AD_DECODE regional stat files contain many extra J*/T* and duplicate
        # suffixed columns. For this Figure 4 enrichment, keep only regional
        # rows whose cleaned root is present in the real S##### connectome files.
        # This does NOT map J IDs to S IDs; it simply excludes unrelated regional
        # columns from the connectome-based feature table.
        if cohort == "AD_DECODE" and not regional_features.empty and not connectome_features.empty:
            allowed_connectome_keys = set(connectome_features["subject_match"].dropna().astype(str))
            before_rows = len(regional_features)
            before_keys = regional_features["subject_match"].nunique() if "subject_match" in regional_features.columns else 0
            regional_features = regional_features[
                regional_features["subject_match"].astype(str).isin(allowed_connectome_keys)
            ].copy()
            after_keys = regional_features["subject_match"].nunique() if "subject_match" in regional_features.columns else 0
            print(
                "AD_DECODE regional/connectome filter:",
                f"rows {before_rows}->{len(regional_features)}",
                f"keys {before_keys}->{after_keys}",
                "(kept only roots present in real connectome filenames)",
            )

        qa_row = make_overlap_qa_row(
            cohort=cohort,
            regional_features=regional_features,
            connectome_features=connectome_features,
            regional_sources=regional_sources,
            connectome_info=connectome_info,
        )
        overlap_qa_rows.append(qa_row)
        write_overlap_qa_file(overlap_qa_rows, OUTDIR, label=f"after_{cohort}")
        print(
            "QA regional/connectome subject overlap:",
            f"regional_keys={qa_row['n_regional_keys']}",
            f"connectome_keys={qa_row['n_connectome_keys']}",
            f"matching={qa_row['n_matching_keys']}",
            f"regional_only={qa_row['n_regional_only']}",
            f"connectome_only={qa_row['n_connectome_only']}",
        )

        features = merge_feature_tables([regional_features, connectome_features])

        if not features.empty:
            features["cohort"] = cohort
            all_features.append(features)

        print("Regional features:", regional_features.shape)
        print("Connectome features:", connectome_features.shape)
        print("Combined features:", features.shape)

        preview_cols = [
            "subject_source",
            "subject_match",
            "Hc_volume_mm3",
            "Hc_volume_pct_brain",
            "Hc_FA",
            "Hc_RD",
            "Hc_AD",
            "Hc_ADC",
            "Hc_clustering_coeff",
            "Hc_path_length",
            "Total_graph_clustering_coeff",
            "Total_graph_path_length",
            "Global_Efficiency",
            "Local_Efficiency",
            "Total_Brain_volume",
            "Total_Brain_FA",
        ]
        preview_cols = [c for c in preview_cols if c in features.columns]

        if preview_cols and not features.empty:
            print(features[preview_cols].head().to_string(index=False))

        for fs in FEATURE_SETS:
            p = validation_path(results_root, cohort, fs)

            row = enrich_validation_file(
                p,
                cohort=cohort,
                feature_set=fs,
                features=features,
                overwrite_original=args.overwrite_original,
            )

            validation_manifest_rows.append(row)

            print(
                fs,
                "|",
                row.get("status"),
                "| matched:",
                row.get("n_matched"),
                "| output:",
                row.get("output_path"),
            )

    if all_features:
        pd.concat(all_features, ignore_index=True, sort=False).to_csv(
            OUTDIR / "Figure4_recomputed_neuroimaging_features_by_subject.csv",
            index=False,
        )

    pd.DataFrame(regional_source_rows).to_csv(
        OUTDIR / "Figure4_enrichment_regional_source_manifest.csv",
        index=False,
    )

    pd.DataFrame(connectome_source_rows).to_csv(
        OUTDIR / "Figure4_enrichment_connectome_source_manifest.csv",
        index=False,
    )

    pd.DataFrame(validation_manifest_rows).to_csv(
        OUTDIR / "Figure4_enrichment_validation_manifest.csv",
        index=False,
    )

    pd.DataFrame([{
        "atlas_path": str(args.atlas),
        "hip_roi_labels": ",".join(map(str, hip_roi_labels)),
        "hip_nodes_zero_based": ",".join(map(str, hip_nodes_zero_based)),
        "threshold_percentile": args.threshold_percentile,
        "min_weight": args.min_weight,
    }]).to_csv(
        OUTDIR / "Figure4_enrichment_atlas_and_parameters.csv",
        index=False,
    )

    qa_overlap_path = write_overlap_qa_file(overlap_qa_rows, OUTDIR, label="final")

    print("\nSaved QC/manifests to:")
    print(OUTDIR)
    print("QA overlap file:")
    print(qa_overlap_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
