#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final unified CLI graph builder for BrainAge harmonized connectome graphs.

Cohorts supported: ADNI, ADRC, HABS, AD_DECODE.

Validated special handling:
  - ADNI reads the audited 316-session central metadata and uses exact AGE,
    PTGENDER/SEX_from_covars, DWI, DWI_subject_key, and DWI_visit_label.
  - HABS preserves visit-specific scan IDs for longitudinal sessions.
  - PyG Data objects are built in the parent process to avoid Linux
    multiprocessing ancdata/BrokenProcessPool failures.

Core workflow:
  1. Load raw metadata and connectome matrices.
  2. Harmonize cohort-specific metadata to canonical columns.
  3. Match metadata rows to connectomes and regional FA/volume tables.
  4. Threshold/log-transform connectomes and compute graph metrics.
  5. Build PyTorch Geometric Data objects for:
       - healthy-control training set
       - all-subject/full-cohort prediction set
  6. Save graph objects, aligned metadata, QC summaries, and timing logs.

Longitudinal cohorts use scan/session IDs (connectome_key, e.g. H4369_y0)
so visits remain separate. Single-visit cohorts use match_id/regional_id.
"""

import os
import re
import json
import time
import random
import warnings
import argparse
import multiprocessing as mp
from collections import OrderedDict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
ProcessPoolExecutor = ThreadPoolExecutor  # patched: avoid torch multiprocessing ancdata crash

import numpy as np
import pandas as pd
import networkx as nx
import torch

from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

warnings.filterwarnings("ignore")


# =========================================================
# REPRODUCIBILITY
# =========================================================
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


# =========================================================
# GLOBAL SETTINGS / CLI
# =========================================================
VALID_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]
VALID_FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build harmonized BrainAge graph datasets from connectomes, metadata, FA, and volume tables."
    )
    parser.add_argument(
        "--work",
        default=os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"),
        help="Project WORK root. Default: env WORK, else /mnt/newStor/paros/paros_WORK",
    )
    parser.add_argument(
        "--cohort",
        default="HABS",
        choices=VALID_COHORTS,
        help="Cohort to build when --all-cohorts is not used.",
    )
    parser.add_argument(
        "--all-cohorts",
        action="store_true",
        help="Build all cohorts sequentially.",
    )
    parser.add_argument(
        "--feature-set",
        default="imaging_only",
        choices=VALID_FEATURE_SETS,
        help="Feature set to build when --all-feature-sets is not used.",
    )
    parser.add_argument(
        "--all-feature-sets",
        action="store_true",
        help="Build all five feature sets sequentially.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers used for parallel sections. Default: 4.",
    )
    parser.add_argument(
        "--edge-percentile",
        type=float,
        default=50,
        help="Top edge percentile retained during thresholding. Default: 50.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete each selected cohort/feature-set output directory before rebuilding.",
    )
    return parser.parse_args()


ARGS = parse_args()
WORK = os.path.abspath(ARGS.work)

EDGE_PERCENTILE = ARGS.edge_percentile
N_ROIS_EXPECTED = 84
N_JOBS = ARGS.workers

HARMONIZED_COLUMNS = [
    "cohort",
    "subject_id",
    "regional_id",
    "connectome_key",
    "match_id",
    "age",
    "sex",
    "genotype",
    "group_status",
    "is_healthy_control",
    "NORMCOG",
    "bp_sys",
    "bp_dia",
    "pulse",
    "height_cm",
    "weight_kg",
    "bmi",
    "amyloid_40",
    "amyloid_42",
    "tau_total",
    "ptau",
    "ptau217",
    "nfl",
    "gfap",
]

PCA_COLUMNS = [f"PC{i}" for i in range(1, 11)]


# =========================================================
# SINGLE ENTRY SETTINGS FROM CLI
# =========================================================
COHORT_NAME = ARGS.cohort
RUN_ALL_COHORTS = ARGS.all_cohorts
FEATURE_SET = ARGS.feature_set
FEATURE_SETS_TO_RUN = VALID_FEATURE_SETS if ARGS.all_feature_sets else [FEATURE_SET]

# Feature-set variant for graph-level/global predictors.
# Node features and edge features are always imaging-derived.
# Options:
#   "imaging_only"              = node/edge imaging + global graph metrics only
#   "imaging_demographics"      = imaging + sex/APOE
#   "imaging_biomarkers"        = imaging + sex/APOE + biomarkers or PCA; no BMI, no cardiovascular
#   "full"                      = imaging + sex/APOE + BMI + cardiovascular + biomarkers/PCA
#   "full_no_cardiovascular"    = full model without systolic BP, diastolic BP, pulse


'''

full
full_no_cardiovascular
imaging_only
imaging_demographics
imaging_biomarkers

'''

# =========================================================
# COHORT CONFIG
# =========================================================
COHORT_CONFIGS = {
    "ADNI": {
        "cohort_name": "ADNI",
        "connectome_dir": os.path.join(WORK, "ines/data/harmonization/ADNI/connectomes/DWI/plain"),
        "metadata_path": os.path.join(WORK, "ines/data/harmonization/harmonized_metadata/ADNI_harmonized_metadata.csv"),
        "fa_path": os.path.join(WORK, "ines/data/Regional_stats/ADNI/ADNI_studywide_stats_for_fa.txt"),
        "vol_path": os.path.join(WORK, "ines/data/Regional_stats/ADNI/ADNI_studywide_stats_BrainPct.csv"),
        "pca_path": None,
        "fa_sep": "\t",
        "metadata_sheet": None,
        "feature_mode": "biomarkers",
        "longitudinal": True,
        "graph_subject_key": "connectome_key",
        "node_feature_key": "connectome_key",
    },
    "ADRC": {
        "cohort_name": "ADRC",
        "connectome_dir": os.path.join(WORK, "ines/data/harmonization/ADRC/connectomes/DWI/plain"),
        "metadata_path": os.path.join(WORK, "ines/data/harmonization/ADRC/metadata/ADRC_metadata.xlsx"),
        "fa_path": os.path.join(WORK, "ines/data/Regional_stats/ADRC/ADRC_studywide_stats_for_fa.txt"),
        "vol_path": os.path.join(WORK, "ines/data/Regional_stats/ADRC/ADRC_studywide_stats_BrainPct.csv"),
        "pca_path": None,
        "fa_sep": "\t",
        "metadata_sheet": None,
        "feature_mode": "biomarkers",
        "longitudinal": False,
        "graph_subject_key": "match_id",
        "node_feature_key": "regional_id",
    },
    "HABS": {
        "cohort_name": "HABS",
        "connectome_dir": os.path.join(WORK, "ines/data/harmonization/HABS/connectomes/DWI/plain"),
        "metadata_path": os.path.join(WORK, "ines/data/harmonization/HABS/metadata/HABS_metadata.xlsx"),
        "fa_path": os.path.join(WORK, "ines/data/Regional_stats/HABS/HABS_studywide_stats_for_fa.txt"),
        "vol_path": os.path.join(WORK, "ines/data/Regional_stats/HABS/HABS_studywide_stats_BrainPct.csv"),
        "pca_path": None,
        "fa_sep": "\t",
        "metadata_sheet": None,
        "feature_mode": "biomarkers",
        "longitudinal": True,
        "graph_subject_key": "connectome_key",
        "node_feature_key": "connectome_key",
    },
    "AD_DECODE": {
        "cohort_name": "AD_DECODE",
        "connectome_dir": os.path.join(WORK, "ines/data/harmonization/AD_DECODE/connectomes/DWI/plain"),
        "metadata_path": os.path.join(WORK, "ines/data/harmonization/AD_DECODE/metadata/AD_DECODE_metadata.xlsx"),
        "fa_path": os.path.join(WORK, "ines/data/Regional_stats/ADDecode/ADDecode_studywide_stats_for_fa.txt"),
        "vol_path": os.path.join(WORK, "ines/data/Regional_stats/ADDecode/ADDecode_studywide_stats_BrainPct.csv"),
        "pca_path":None,
        "fa_sep": "\t",
        "metadata_sheet": None,
        "feature_mode": "pca",
        "longitudinal": False,
        "graph_subject_key": "match_id",
        "node_feature_key": "regional_id",
    },
}


# =========================================================
# TIMING
# =========================================================
timings = OrderedDict()
script_t0 = None

def fmt_seconds(secs):
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.2f}"

@contextmanager
def timed_block(name):
    t0 = time.perf_counter()
    print(f"\n[TIMER START] {name}")
    try:
        yield
    finally:
        secs = time.perf_counter() - t0
        timings[name] = timings.get(name, 0.0) + secs
        print(f"[TIMER END]   {name}: {fmt_seconds(secs)}")


# =========================================================
# IO HELPERS
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def load_table_auto(path, sheet_name=None):
    if path.lower().endswith((".xlsx", ".xls")):
        if sheet_name is None:
            return pd.read_excel(path)
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if path.lower().endswith(".txt"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")

def build_output_paths(cohort_name):
    cohort_lc = cohort_name.lower()
    results_dir = ensure_dir(os.path.join(WORK, "ines/results/harmonized", cohort_name, "graphs", FEATURE_SET))

    return {
        "results_dir": results_dir,
        "cleaned_metadata_path": os.path.join(results_dir, f"{cohort_name}_metadata_cleaned_harmonized.csv"),

        "save_graphs_path": os.path.join(results_dir, f"graph_data_list_{cohort_lc}.pt"),

        # RAW aligned metadata for evaluation/validation
        "save_metadata_path_raw": os.path.join(results_dir, f"{cohort_lc}_metadata_aligned_raw.csv"),

        # model-ready aligned metadata for training
        "save_metadata_path": os.path.join(results_dir, f"{cohort_lc}_metadata_aligned.csv"),

        "save_summary_path": os.path.join(results_dir, f"{cohort_lc}_graph_build_summary.csv"),
        "losses_path": os.path.join(results_dir, f"{cohort_lc}_connectomes_lost_before_match_with_reason.csv"),

        "save_graphs_all_path": os.path.join(results_dir, f"graph_data_list_{cohort_lc}_all.pt"),

        # RAW aligned metadata for all-subject set
        "save_metadata_all_path_raw": os.path.join(results_dir, f"{cohort_lc}_metadata_all_aligned_raw.csv"),

        # model-ready aligned metadata for all-subject set
        "save_metadata_all_path": os.path.join(results_dir, f"{cohort_lc}_metadata_all_aligned.csv"),

        "save_summary_all_path": os.path.join(results_dir, f"{cohort_lc}_graph_build_summary_all.csv"),
        "encoding_info_json": os.path.join(results_dir, f"{cohort_lc}_feature_encoding_info.json"),
        "timing_summary_path": os.path.join(results_dir, f"{cohort_lc}_graph_build_timing_summary.txt"),
        "subject_timing_csv": os.path.join(results_dir, f"{cohort_lc}_graph_build_subject_timings.csv"),
    }


def parse_longitudinal_scan_id_for_qc(x):
    s = str(x).strip().replace("_Y", "_y")
    m = re.search(r"([A-Za-z]\d+)_y(\d+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), f"y{int(m.group(2))}"
    return s.upper(), ""


def summarize_longitudinal_ids_for_qc(ids, label):
    vals = pd.Series(list(ids), dtype="object").astype(str)
    parsed = vals.apply(parse_longitudinal_scan_id_for_qc)
    tmp = pd.DataFrame({
        "id": vals,
        "base": parsed.apply(lambda z: z[0]),
        "visit": parsed.apply(lambda z: z[1]),
    })
    tmp = tmp[tmp["visit"].isin(["y0", "y2", "y4"])]
    paired = tmp.groupby("base")["visit"].nunique() if len(tmp) else pd.Series(dtype=int)
    return {
        "label": label,
        "n_rows": int(len(vals)),
        "n_longitudinal_rows": int(len(tmp)),
        "visit_counts": json.dumps(tmp["visit"].value_counts().to_dict()),
        "n_unique_bases": int(tmp["base"].nunique()) if len(tmp) else 0,
        "n_paired_bases_ge2_visits": int((paired >= 2).sum()) if len(tmp) else 0,
        "n_duplicate_ids": int(vals.duplicated().sum()),
        "example_ids": json.dumps(vals.head(10).tolist()),
    }


def save_graphbuilder_longitudinal_qc(paths, cohort_name, rows):
    qc = pd.DataFrame(rows)
    out = os.path.join(paths["results_dir"], f"{cohort_name.lower()}_graphbuilder_longitudinal_QC.csv")
    qc.to_csv(out, index=False)
    print("\n=== LONGITUDINAL QC ===")
    print(qc.to_string(index=False))
    print("Saved longitudinal QC:", out)
    return qc


# =========================================================
# GENERIC CLEANING / MATCH HELPERS
# =========================================================
def as_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def safe_series(df, col):
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)

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

def clean_numeric_sentinels(series, nonnegative=False, binary=False, max_abs=None):
    s = pd.to_numeric(series, errors="coerce").copy()

    for val in SENTINEL_VALUES:
        s = s.mask(s == val, np.nan)

    if max_abs is not None:
        s = s.mask(np.abs(s) > max_abs, np.nan)

    if nonnegative:
        s = s.mask(s < 0, np.nan)

    if binary:
        s = s.where(s.isin([0, 1]), np.nan)

    return s

def clean_sex_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    mapping = {
        "1": "M", "2": "F",
        "1.0": "M", "2.0": "F",
        "MALE": "M", "FEMALE": "F",
        "M": "M", "F": "F",
    }
    return mapping.get(s, s)

def parse_genotype_string(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "")
    if s == "":
        return np.nan
    return s

def compose_genotype(a1, a2):
    if pd.isna(a1) or pd.isna(a2):
        return np.nan
    try:
        return f"{int(float(a1))}_{int(float(a2))}"
    except Exception:
        return np.nan

def safe_label_transform(series, encoder, unknown_token):
    known = set(encoder.classes_)
    s = series.astype(str).copy()
    s = s.where(s.isin(known), other=unknown_token)
    return encoder.transform(s)
def add_normcog_column(df):
    df = df.copy()

    if "NORMCOG" in df.columns:
        df["NORMCOG"] = pd.to_numeric(df["NORMCOG"], errors="coerce")

    if "group_status" in df.columns:
        healthy_labels = {"CN", "HC", "CONTROL", "NORMAL", "HEALTHY"}
        nonhealthy_labels = {
            "MCI", "AD", "DEMENTIA", "ATRISK", "AT_RISK",
            "IMPAIRED_NON_MCI", "PATIENT", "CASE", "DEMENTED"
        }

        def _map_group_status(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip().upper().replace(" ", "_")
            if s in healthy_labels:
                return 1
            if s in nonhealthy_labels:
                return 0
            return np.nan

        derived = df["group_status"].apply(_map_group_status)
    else:
        derived = pd.Series(np.nan, index=df.index)

    if "NORMCOG" in df.columns:
        df["NORMCOG"] = df["NORMCOG"].fillna(derived)
    else:
        df["NORMCOG"] = derived

    if "is_healthy_control" in df.columns:
        df["NORMCOG"] = df["NORMCOG"].fillna(
            pd.to_numeric(df["is_healthy_control"], errors="coerce")
        )

    df["NORMCOG"] = pd.to_numeric(df["NORMCOG"], errors="coerce")
    return df
def add_bmi_column(df, bmi_col=None, height_col=None, weight_col=None, out_col="bmi"):
    df = df.copy()

    bmi = safe_series(df, bmi_col) if bmi_col else pd.Series(np.nan, index=df.index)
    height = safe_series(df, height_col) if height_col else pd.Series(np.nan, index=df.index)
    weight = safe_series(df, weight_col) if weight_col else pd.Series(np.nan, index=df.index)

    bmi = as_numeric(bmi)
    height = as_numeric(height)
    weight = as_numeric(weight)

    computed_bmi = weight / ((height / 100.0) ** 2)
    computed_bmi = computed_bmi.where((height > 0) & (weight > 0), np.nan)

    df["height_cm"] = height
    df["weight_kg"] = weight
    df[out_col] = bmi.fillna(computed_bmi)

    return df

def normalize_apoe_to_x_x(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper().replace(" ", "")

    mapping = {
        "E2E2": "2_2",
        "E2E3": "2_3",
        "E2E4": "2_4",
        "E3E3": "3_3",
        "E3E4": "3_4",
        "E4E4": "4_4",
        "APOE22": "2_2",
        "APOE23": "2_3",
        "APOE24": "2_4",
        "APOE33": "3_3",
        "APOE34": "3_4",
        "APOE44": "4_4",
        "2/2": "2_2",
        "2/3": "2_3",
        "2/4": "2_4",
        "3/3": "3_3",
        "3/4": "3_4",
        "4/4": "4_4",
        "2_2": "2_2",
        "2_3": "2_3",
        "2_4": "2_4",
        "3_3": "3_3",
        "3_4": "3_4",
        "4_4": "4_4",
    }

    return mapping.get(s, np.nan)

def save_cleaned_metadata(df_raw, df_h, out_csv, cohort_name):
    keep_h = [c for c in HARMONIZED_COLUMNS if c in df_h.columns] + [c for c in PCA_COLUMNS if c in df_h.columns]

    if cohort_name == "ADNI":
        merge_keys = ["DWI"]
    elif cohort_name == "HABS":
        # HABS runno is normalized to the scan/session ID (Hxxxx_y#).
        # Save the harmonized session-level table directly to avoid losing
        # y0/y2 rows if raw metadata uses a different runno convention.
        df_h.to_csv(out_csv, index=False)
        return df_h.copy()
    elif cohort_name == "ADRC":
        merge_keys = ["PTID"]
    elif cohort_name == "AD_DECODE":
        merge_keys = ["MRI_Exam"]
    else:
        raise ValueError(f"Unsupported cohort: {cohort_name}")

    for k in merge_keys:
        if k not in df_raw.columns:
            raise ValueError(f"Raw metadata does not contain merge key: {k}")
        if k not in df_h.columns:
            raise ValueError(f"Harmonized metadata does not contain merge key: {k}")

    df_out = df_raw.merge(
        df_h[merge_keys + keep_h].drop_duplicates(subset=merge_keys),
        how="left",
        on=merge_keys
    )

    df_out.to_csv(out_csv, index=False)
    return df_out
def extract_4digit_match_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    groups = re.findall(r"(\d+)", s)
    if len(groups) == 0:
        return np.nan
    digits = "".join(groups)
    return digits[-4:].zfill(4)

def build_regional_id_from_match_id(match_id, cohort_name):
    if pd.isna(match_id):
        return np.nan

    match_id = str(match_id).zfill(4)

    if cohort_name == "ADRC":
        return f"D{match_id}"
    if cohort_name == "HABS":
        return f"H{match_id}"
    if cohort_name == "ADNI":
        return f"R{match_id}"
    if cohort_name == "AD_DECODE":
        return f"S0{match_id}"
    return np.nan

def add_metadata_match_id(df, cohort_name):
    df = df.copy()

    if cohort_name == "ADRC":
        if "PTID" not in df.columns:
            raise ValueError("ADRC metadata must contain PTID.")
        df["match_id"] = df["PTID"].apply(extract_4digit_match_id)

    elif cohort_name == "HABS":
        if "Subject" in df.columns:
            df["match_id"] = safe_series(df, "Subject").astype(str).str.strip().apply(extract_4digit_match_id)
        elif "runno" in df.columns:
            df["match_id"] = safe_series(df, "runno").astype(str).str.strip().apply(extract_4digit_match_id)
        else:
            raise ValueError("HABS metadata must contain Subject or runno.")

    elif cohort_name == "ADNI":
        if "PTID" not in df.columns:
            raise ValueError("ADNI metadata must contain PTID.")
        df["match_id"] = df["PTID"].apply(extract_4digit_match_id)

    elif cohort_name == "AD_DECODE":
        if "MRI_Exam" not in df.columns:
            raise ValueError("AD_DECODE metadata must contain MRI_Exam.")
        df["match_id"] = df["MRI_Exam"].apply(extract_4digit_match_id)

    else:
        raise ValueError(f"Unsupported cohort: {cohort_name}")

    return df


# =========================================================
# GROUP STATUS HELPERS
# =========================================================
def infer_group_status_adrc(row):
    normcog = row.get("NORMCOG", np.nan)
    demented = row.get("DEMENTED", np.nan)
    impnomci = row.get("IMPNOMCI", np.nan)

    if normcog == 1 and (pd.isna(demented) or demented != 1):
        return "CN"
    if demented == 1:
        return "Demented"
    if impnomci == 1:
        return "Impaired_non_MCI"
    return "Unknown"

def infer_group_status_habs(row):
    val = row.get("Dementia", np.nan)

    if pd.isna(val):
        return "Unknown"

    try:
        v = float(val)
        if v == 0:
            return "CN"
        if v == 1:
            return "Dementia"
    except Exception:
        pass

    s = str(val).strip().upper()
    if s in ["0", "0.0", "NO", "FALSE", "CN", "CONTROL", "NORMAL", "HC", "HEALTHY"]:
        return "CN"
    if s in ["1", "1.0", "YES", "TRUE", "DEMENTIA", "AD", "MCI", "PATIENT", "CASE"]:
        return "Dementia"

    return "Unknown"

def infer_group_status_ad_decode(row):
    risk = str(row.get("Risk", "")).strip()
    if risk != "":
        return risk

    risk_for_ad = row.get("risk_for_ad", np.nan)
    if pd.notna(risk_for_ad):
        return "CN" if float(risk_for_ad) == 0 else "AtRisk"

    return "Unknown"

def get_adni_group_status(df):
    group_col = "Research Group" if "Research Group" in df.columns else "DX_bl"
    return safe_series(df, group_col).astype(str).str.strip()


# =========================================================
# COHORT-SPECIFIC HARMONIZATION
# =========================================================
def harmonize_adni_metadata(df_raw):
    """
    Consume the final 316-session ADNI harmonized metadata directly.

    Required authoritative fields:
      DWI                 exact session/connectome key, e.g. R4288_y0
      DWI_subject_key     participant key, e.g. R4288
      DWI_visit_label     y0 or y4
      AGE                 audited exact session age
      PTGENDER or SEX_from_covars
    """
    df = df_raw.copy()

    required = [
        "PTID", "RID", "DWI", "DWI_subject_key",
        "DWI_visit_label", "AGE",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ADNI central metadata missing columns: {missing}")

    df["PTID"] = df["PTID"].astype(str).str.strip()
    df["RID"] = as_numeric(df["RID"]).astype("Int64")
    df["DWI"] = df["DWI"].astype(str).str.strip()
    df["DWI_subject_key"] = (
        df["DWI_subject_key"].astype(str).str.strip().str.upper()
    )
    df["DWI_visit_label"] = (
        df["DWI_visit_label"].astype(str).str.strip().str.lower()
    )

    bad_visit = ~df["DWI_visit_label"].isin(["y0", "y4"])
    if bad_visit.any():
        print(
            df.loc[
                bad_visit,
                ["DWI", "DWI_subject_key", "DWI_visit_label"],
            ].head(50).to_string(index=False)
        )
        raise RuntimeError("Unexpected ADNI DWI_visit_label values.")

    if df["DWI"].duplicated().any():
        print(
            df.loc[
                df["DWI"].duplicated(False),
                ["DWI", "PTID", "DWI_subject_key", "DWI_visit_label"],
            ].head(50).to_string(index=False)
        )
        raise RuntimeError("Duplicate ADNI DWI session keys.")

    df["cohort"] = "ADNI"
    df["subject_id"] = df["PTID"]
    df["match_id"] = (
        df["DWI_subject_key"].str.extract(r"R(\d+)", expand=False)
    )
    df["regional_id"] = df["DWI_subject_key"]
    df["connectome_key"] = df["DWI"]
    df["visit_clean"] = df["DWI_visit_label"]

    # Preserve original visit code for diagnostics.
    if "VISCODE_x" in df.columns:
        df["VISCODE"] = df["VISCODE_x"].astype(str).str.strip().str.lower()
    elif "VISCODE" in df.columns:
        df["VISCODE"] = df["VISCODE"].astype(str).str.strip().str.lower()
    else:
        df["VISCODE"] = df["visit_clean"]

    # Exact audited session age.
    df["age"] = as_numeric(df["AGE"])
    if df["age"].isna().any():
        raise RuntimeError(
            f"ADNI AGE missing in {int(df['age'].isna().sum())} rows."
        )

    for check_col in ["age_visit_corrected", "AGE_session_covars"]:
        if check_col in df.columns:
            check = as_numeric(df[check_col])
            disagree = (
                check.notna()
                & ((df["age"] - check).abs() > 1e-6)
            )
            n_disagree = int(disagree.sum())
            print(f"[ADNI AGE QA] AGE vs {check_col}: disagreements={n_disagree}")
            if n_disagree:
                print(
                    df.loc[
                        disagree,
                        ["DWI", "AGE", check_col],
                    ].head(50).to_string(index=False)
                )
                raise RuntimeError(
                    f"AGE disagrees with {check_col} in ADNI central metadata."
                )

    # Sex: exact repaired PTGENDER first, covars fallback.
    sex_raw = (
        df["PTGENDER"]
        if "PTGENDER" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    if "SEX_from_covars" in df.columns:
        sex_raw = sex_raw.fillna(df["SEX_from_covars"])

    df["sex"] = sex_raw.apply(clean_sex_value)
    invalid_sex = ~df["sex"].isin(["M", "F"])
    if invalid_sex.any():
        print(
            df.loc[
                invalid_sex,
                ["DWI", "PTID", "PTGENDER", "SEX_from_covars"]
                if "PTGENDER" in df.columns and "SEX_from_covars" in df.columns
                else ["DWI", "PTID"],
            ].head(50).to_string(index=False)
        )
        raise RuntimeError(
            f"ADNI sex invalid/missing in {int(invalid_sex.sum())} rows."
        )

    print("[ADNI SEX QA]")
    print(df["sex"].value_counts(dropna=False).to_string())

    # APOE genotype.
    if "genotype" in df.columns:
        df["genotype"] = df["genotype"].apply(parse_genotype_string)
    else:
        for c in ["APOE_A1", "APOE_A2"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = as_numeric(df[c])
        df["genotype"] = df.apply(
            lambda r: compose_genotype(r["APOE_A1"], r["APOE_A2"]),
            axis=1,
        )

    # Diagnosis/group retained for validation; not a model predictor.
    if "GROUP_broad_from_covars" in df.columns:
        df["group_status"] = (
            df["GROUP_broad_from_covars"].astype(str).str.strip().str.upper()
        )
    elif "GROUP_from_covars" in df.columns:
        raw_group = df["GROUP_from_covars"].astype(str).str.strip().str.upper()
        broad_map = {
            "CN": "CN", "SMC": "CN",
            "EMCI": "MCI", "MCI": "MCI", "LMCI": "MCI",
            "AD": "AD",
        }
        df["group_status"] = raw_group.map(broad_map).fillna(raw_group)
    else:
        df["group_status"] = get_adni_group_status(df)

    # Controls: use the harmonized cognition/control indicator when present.
    if "NORMCOG_01" in df.columns:
        df["is_healthy_control"] = (
            as_numeric(df["NORMCOG_01"]).fillna(0).eq(1)
        ).astype(int)
    elif "NORMCOG" in df.columns:
        df["is_healthy_control"] = (
            as_numeric(df["NORMCOG"]).fillna(0).eq(1)
        ).astype(int)
    else:
        df["is_healthy_control"] = (
            df["group_status"].eq("CN")
        ).astype(int)

    # Cardiovascular.
    df["bp_sys"] = as_numeric(
        safe_series(df, "VSBPSYS")
        if "VSBPSYS" in df.columns
        else safe_series(df, "vitals_VSBPSYS")
    )
    df["bp_dia"] = as_numeric(
        safe_series(df, "VSBPDIA")
        if "VSBPDIA" in df.columns
        else safe_series(df, "vitals_VSBPDIA")
    )
    df["pulse"] = as_numeric(
        safe_series(df, "VSPULSE")
        if "VSPULSE" in df.columns
        else safe_series(df, "vitals_VSPULSE")
    )

    # Biomarkers.
    df["amyloid_40"] = as_numeric(safe_series(df, "ABETA40"))
    df["amyloid_42"] = as_numeric(safe_series(df, "ABETA42"))
    df["tau_total"] = as_numeric(safe_series(df, "TAU"))
    df["ptau"] = as_numeric(safe_series(df, "PTAU"))
    df["ptau217"] = as_numeric(safe_series(df, "PLASMA_PTAU217"))

    nfl_col = next((c for c in ["NfL", "NFL", "nfl"] if c in df.columns), None)
    gfap_col = next((c for c in ["GFAP", "gfap"] if c in df.columns), None)
    df["nfl"] = (
        as_numeric(df[nfl_col])
        if nfl_col is not None
        else pd.Series(np.nan, index=df.index)
    )
    df["gfap"] = (
        as_numeric(df[gfap_col])
        if gfap_col is not None
        else pd.Series(np.nan, index=df.index)
    )

    # BMI.
    if "BMI" in df.columns:
        df["bmi"] = as_numeric(df["BMI"])
        df["height_cm"] = as_numeric(safe_series(df, "VSHEIGHT"))
        df["weight_kg"] = as_numeric(safe_series(df, "VSWEIGHT"))
    else:
        df = add_bmi_column(
            df,
            bmi_col=None,
            height_col="VSHEIGHT",
            weight_col="VSWEIGHT",
        )

    for c in PCA_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = as_numeric(df[c]).fillna(0.0)

    df = add_normcog_column(df)

    # Longitudinal age QA.
    wide = df.pivot_table(
        index="regional_id",
        columns="visit_clean",
        values="age",
        aggfunc="first",
    )
    if {"y0", "y4"}.issubset(wide.columns):
        paired = wide.dropna(subset=["y0", "y4"]).copy()
        paired["delta"] = paired["y4"] - paired["y0"]
        bad = paired[(paired["delta"] <= 0) | (paired["delta"] > 7)]
        print("[ADNI LONGITUDINAL AGE QA]")
        print("pairs:", len(paired))
        print(paired["delta"].value_counts().sort_index().to_string())
        if not bad.empty:
            print(bad.head(50).to_string())
            raise RuntimeError("Implausible ADNI longitudinal age intervals.")

    print(
        f"[ADNI CENTRAL QA] rows={len(df)} "
        f"subjects={df['regional_id'].nunique()} "
        f"sessions={df['connectome_key'].nunique()} "
        f"controls={int(df['is_healthy_control'].sum())}"
    )

    return df

def harmonize_adrc_metadata(df_raw):
    df = df_raw.copy()

    if "PTID" not in df.columns:
        raise ValueError("ADRC metadata must contain PTID.")

    df["PTID"] = df["PTID"].astype(str).str.strip().str.upper()
    df["VISIT_AGE"] = as_numeric(safe_series(df, "VISIT_AGE"))
    df["SUBJECT_SEX"] = safe_series(df, "SUBJECT_SEX").apply(clean_sex_value)

    numeric_cols = [
        "NORMCOG", "DEMENTED", "IMPNOMCI",
        "BPSYS_AVG", "BPDIA_AVG",
        "AB40", "AB42", "TTAU", "PTAU181", "NFL", "GFAP",
        "BMI", "HEIGHT", "WEIGHT"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = as_numeric(df[col])

    df["cohort"] = "ADRC"
    df["subject_id"] = df["PTID"]
    df["match_id"] = df["PTID"].apply(extract_4digit_match_id)
    df["connectome_key"] = df["match_id"]
    df["regional_id"] = df["match_id"].apply(lambda x: build_regional_id_from_match_id(x, "ADRC"))
    df["age"] = df["VISIT_AGE"]
    df["sex"] = df["SUBJECT_SEX"]
    df["genotype"] = safe_series(df, "APOE").apply(normalize_apoe_to_x_x)

    df["group_status"] = df.apply(infer_group_status_adrc, axis=1)
    df["is_healthy_control"] = (df["group_status"] == "CN").astype(int)

    df["bp_sys"] = as_numeric(safe_series(df, "BPSYS_AVG"))
    df["bp_dia"] = as_numeric(safe_series(df, "BPDIA_AVG"))
    df["pulse"] = np.nan

    df["amyloid_40"] = as_numeric(safe_series(df, "AB40"))
    df["amyloid_42"] = as_numeric(safe_series(df, "AB42"))
    df["tau_total"] = as_numeric(safe_series(df, "TTAU"))
    df["ptau"] = as_numeric(safe_series(df, "PTAU181"))
    df["ptau217"] = np.nan
    df["nfl"] = as_numeric(safe_series(df, "NFL"))
    df["gfap"] = as_numeric(safe_series(df, "GFAP"))

    df = add_bmi_column(df, bmi_col="BMI", height_col="HEIGHT", weight_col="WEIGHT")

    for c in PCA_COLUMNS:
        df[c] = 0.0

    df = add_normcog_column(df)
    return df

def normalize_habs_apoe(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper().replace(" ", "")
    mapping = {
        "E2E2": "2_2",
        "E2E3": "2_3",
        "E2E4": "2_4",
        "E3E3": "3_3",
        "E3E4": "3_4",
        "E4E4": "4_4",
    }
    return mapping.get(s, np.nan)


def normalize_habs_scan_id(x):
    """Return a clean scan/session key such as H4369_y0 or H4369_y2."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("_Y", "_y")
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"_conn_plain$", "", s, flags=re.IGNORECASE)
    m = re.search(r"(H\d+)_y(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}_y{int(m.group(2))}"
    return s


def choose_habs_scan_column(df):
    """Pick the HABS column with the most scan-level Hxxxx_y# identifiers."""
    preferred = [
        "DWI", "dwi", "connectome_key", "runno", "RunNo", "RUNNO",
        "scan_id", "Scan_ID", "Subject_ID", "subject_match"
    ]
    candidates = [c for c in preferred if c in df.columns]
    candidates += [
        c for c in df.columns
        if c not in candidates and any(k in str(c).lower() for k in ["dwi", "run", "scan", "connectome"])
    ]

    best_col = None
    best_n = -1
    best_unique = -1
    pat = re.compile(r"H\d+_y\d+", flags=re.IGNORECASE)
    for c in candidates:
        vals = df[c].map(normalize_habs_scan_id).astype(str)
        hit = vals.str.contains(pat, na=False)
        n = int(hit.sum())
        unique = int(vals[hit].nunique())
        if (n, unique) > (best_n, best_unique):
            best_col, best_n, best_unique = c, n, unique

    if best_col is None or best_n <= 0:
        raise ValueError(
            "Could not find a HABS scan/session column with Hxxxx_y# IDs. "
            f"Checked columns: {candidates[:20]}"
        )

    print(f"HABS scan/session column selected: {best_col} | rows_with_Hxxxx_y#: {best_n} | unique_scan_ids: {best_unique}")
    return best_col


def harmonize_habs_metadata(df_raw):
    df = df_raw.copy()

    # HABS must be kept at scan/session level for longitudinal analyses.
    # Prefer the column that actually contains Hxxxx_y0/Hxxxx_y2 IDs.
    scan_col = choose_habs_scan_column(df)
    df["habs_scan_source_col"] = scan_col
    df["runno"] = df[scan_col].map(normalize_habs_scan_id)

    if "Subject" in df.columns:
        df["Subject"] = safe_series(df, "Subject").astype(str).str.strip()
    else:
        df["Subject"] = df["runno"].astype(str).str.extract(r"H(\d+)_y\d+", expand=False)

    df["Year"] = as_numeric(safe_series(df, "Year"))

    df["cohort"] = "HABS"
    df["subject_id"] = df["runno"].astype(str).str.extract(r"H(\d+)_y\d+", expand=False)
    df["subject_id"] = df["subject_id"].fillna(df["Subject"].astype(str)).astype(str).str.zfill(4)
    df["match_id"] = df["subject_id"].apply(extract_4digit_match_id)
    df["connectome_key"] = df["runno"]
    df["regional_id"] = df["match_id"].apply(lambda x: build_regional_id_from_match_id(x, "HABS"))

    df["age"] = clean_numeric_sentinels(safe_series(df, "Age"), nonnegative=True, max_abs=150)
    df["sex"] = safe_series(df, "Sex").apply(clean_sex_value)
    df["genotype"] = safe_series(df, "APOE4_Genotype_x").apply(normalize_habs_apoe)

    df["group_status"] = df.apply(infer_group_status_habs, axis=1)
    df["is_healthy_control"] = (df["group_status"] == "CN").astype(int)

    # Clean source BP / pulse columns BEFORE averaging
    for c in ["OM_BP1_SYS", "OM_BP2_SYS"]:
        if c in df.columns:
            df[c] = clean_numeric_sentinels(df[c], nonnegative=True, max_abs=400)

    for c in ["OM_BP1_DIA", "OM_BP2_DIA"]:
        if c in df.columns:
            df[c] = clean_numeric_sentinels(df[c], nonnegative=True, max_abs=250)

    for c in ["OM_Pulse1", "OM_Pulse2"]:
        if c in df.columns:
            df[c] = clean_numeric_sentinels(df[c], nonnegative=True, max_abs=250)

    df["bp_sys"] = df[["OM_BP1_SYS", "OM_BP2_SYS"]].mean(axis=1)
    df["bp_dia"] = df[["OM_BP1_DIA", "OM_BP2_DIA"]].mean(axis=1)
    df["pulse"]  = df[["OM_Pulse1", "OM_Pulse2"]].mean(axis=1)

    df["height_cm"] = np.nan
    df["weight_kg"] = np.nan
    df["bmi"] = clean_numeric_sentinels(safe_series(df, "BMI"), nonnegative=True, max_abs=100)

    df["amyloid_40"] = clean_numeric_sentinels(safe_series(df, "r2_QTX_Plasma_Abeta40"), nonnegative=True, max_abs=1e6)
    df["amyloid_42"] = clean_numeric_sentinels(safe_series(df, "r2_QTX_Plasma_Abeta42"), nonnegative=True, max_abs=1e6)
    df["tau_total"] = clean_numeric_sentinels(safe_series(df, "r5_LUM(Pro)_Plasma_Total_Tau"), nonnegative=True, max_abs=1e6)
    df["ptau"] = clean_numeric_sentinels(safe_series(df, "r3_QTX_Plasma_pTau181"), nonnegative=True, max_abs=1e6)
    df["ptau217"] = clean_numeric_sentinels(safe_series(df, "pTau217"), nonnegative=True, max_abs=1e6)
    df["nfl"] = clean_numeric_sentinels(safe_series(df, "r2_QTX_Plasma_NfL"), nonnegative=True, max_abs=1e6)
    df["gfap"] = clean_numeric_sentinels(safe_series(df, "GFAP"), nonnegative=True, max_abs=1e6)

    for c in PCA_COLUMNS:
        df[c] = 0.0

    df = add_normcog_column(df)
    return df
def harmonize_ad_decode_metadata(df_raw, pca_df=None):
    df = df_raw.copy()

    if "MRI_Exam" not in df.columns:
        raise ValueError("AD_DECODE metadata must contain MRI_Exam.")

    df["cohort"] = "AD_DECODE"
    df["subject_id"] = safe_series(df, "MRI_Exam").astype(str).str.strip()
    df["match_id"] = df["subject_id"].apply(extract_4digit_match_id)
    df["connectome_key"] = df["match_id"]
    df["regional_id"] = df["match_id"].apply(lambda x: build_regional_id_from_match_id(x, "AD_DECODE"))

    df["age"] = as_numeric(safe_series(df, "age"))
    df["sex"] = safe_series(df, "sex").apply(clean_sex_value)
    df["genotype"] = safe_series(df, "genotype").apply(normalize_apoe_to_x_x)

    risk_vals = as_numeric(safe_series(df, "risk_for_ad"))

    df["group_status"] = df.apply(infer_group_status_ad_decode, axis=1)
    df.loc[risk_vals.isin([0, 1]), "group_status"] = "CN"
    
    df["is_healthy_control"] = risk_vals.isin([0, 1]).astype(int)
    df["bp_sys"] = as_numeric(safe_series(df, "Systolic"))
    df["bp_dia"] = as_numeric(safe_series(df, "Diastolic"))
    df["pulse"] = as_numeric(safe_series(df, "Pulse"))

    df["amyloid_40"] = np.nan
    df["amyloid_42"] = np.nan
    df["tau_total"] = np.nan
    df["ptau"] = np.nan
    df["ptau217"] = np.nan
    df["nfl"] = np.nan
    df["gfap"] = np.nan

    df = add_bmi_column(df, bmi_col="BMI", height_col="Height", weight_col="Weight")

    for c in PCA_COLUMNS:
        if c in df.columns:
            df[c] = as_numeric(df[c]).fillna(0.0)
        else:
            df[c] = 0.0

    if pca_df is not None and len(pca_df) > 0:
        pca_df = pca_df.copy()
        if "ID" not in pca_df.columns:
            raise ValueError("AD_DECODE PCA file must contain an 'ID' column.")
        pca_df["ID"] = pca_df["ID"].astype(str).str.strip()
        pca_df["match_id"] = pca_df["ID"].apply(extract_4digit_match_id)

        for c in PCA_COLUMNS:
            if c not in pca_df.columns:
                pca_df[c] = 0.0

        df = df.merge(
            pca_df[["match_id"] + PCA_COLUMNS],
            how="left",
            on="match_id",
            suffixes=("", "_pca")
        )

        for c in PCA_COLUMNS:
            src = f"{c}_pca" if f"{c}_pca" in df.columns else c
            df[c] = as_numeric(df[src]).fillna(0.0)

        drop_cols = [f"{c}_pca" for c in PCA_COLUMNS if f"{c}_pca" in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

    df = add_normcog_column(df)
    return df

def harmonize_metadata(df_raw, cohort_name, pca_df=None):
    if cohort_name == "ADNI":
        return harmonize_adni_metadata(df_raw)
    if cohort_name == "ADRC":
        return harmonize_adrc_metadata(df_raw)
    if cohort_name == "HABS":
        return harmonize_habs_metadata(df_raw)
    if cohort_name == "AD_DECODE":
        return harmonize_ad_decode_metadata(df_raw, pca_df=pca_df)
    raise ValueError(f"Unsupported cohort: {cohort_name}")


# =========================================================
# CONNECTOME HELPERS
# =========================================================
def extract_connectome_identifiers(filename, cohort_name):
    stem = filename.replace(".csv", "")
    stem = re.sub(r"_conn_plain$", "", stem, flags=re.IGNORECASE)
    s = stem.strip()

    if cohort_name == "ADNI":
        m = re.match(r"^(R\d+_y\d+)$", s, flags=re.IGNORECASE)
        if not m:
            return None, None
        connectome_key = m.group(1)
        connectome_key = re.sub(r"_Y", "_y", connectome_key)
        match_id = extract_4digit_match_id(connectome_key)
        return match_id, connectome_key

    elif cohort_name == "HABS":
        m = re.match(r"^(H\d+_y\d+)$", s, flags=re.IGNORECASE)
        if not m:
            return None, None
        connectome_key = m.group(1)
        connectome_key = re.sub(r"_Y", "_y", connectome_key)
        match_id = extract_4digit_match_id(connectome_key)
        return match_id, connectome_key

    elif cohort_name == "ADRC":
        s_up = s.upper()
        m = re.search(r"D(\d+)", s_up)
        if not m:
            return None, None
        match_id = m.group(1)[-4:].zfill(4)
        return match_id, s_up

    elif cohort_name == "AD_DECODE":
        s_up = s.upper()
        m = re.search(r"S(\d+)", s_up)
        if not m:
            return None, None
        match_id = m.group(1)[-4:].zfill(4)
        return match_id, s_up

    return None, None

def load_connectomes_for_cohort(cohort_name, connectome_dir, subject_key_col):
    connectomes = {}
    connectome_key_lookup = {}
    bad_files = []

    if not os.path.isdir(connectome_dir):
        raise ValueError(f"Connectome directory does not exist: {connectome_dir}")

    for root, _, files in os.walk(connectome_dir):
        for fn in files:
            if not fn.endswith("_conn_plain.csv"):
                continue

            file_path = os.path.join(root, fn)

            try:
                match_id, connectome_key = extract_connectome_identifiers(fn, cohort_name)
                if match_id is None:
                    bad_files.append((fn, "Could not parse connectome identifiers"))
                    continue

                df_conn = pd.read_csv(file_path, header=None)

                if subject_key_col == "connectome_key":
                    store_key = connectome_key
                else:
                    store_key = match_id

                connectomes[store_key] = df_conn
                connectome_key_lookup[store_key] = connectome_key

            except Exception as e:
                bad_files.append((fn, str(e)))

    print(f"Total connectomes loaded: {len(connectomes)}")
    if bad_files:
        print(f"Bad files: {len(bad_files)}")
        print("First bad files:", bad_files[:10])

    if len(connectomes) == 0:
        raise ValueError(f"No connectomes found for cohort {cohort_name}.")

    first_key = next(iter(connectomes.keys()))
    n_rois = connectomes[first_key].shape[0]
    print("Detected n_rois:", n_rois)
    if n_rois != N_ROIS_EXPECTED:
        print(f"Warning: expected {N_ROIS_EXPECTED} ROIs but detected {n_rois}")

    print("Example connectome store keys:", list(connectomes.keys())[:10])
    print("Example full connectome keys:", list(connectome_key_lookup.values())[:10])

    return connectomes, connectome_key_lookup, n_rois


# =========================================================
# GRAPH / METRIC HELPERS
# =========================================================
def threshold_connectome(matrix_values, percentile=75):
    matrix_np = np.asarray(matrix_values, dtype=np.float32)
    matrix_np = 0.5 * (matrix_np + matrix_np.T)
    np.fill_diagonal(matrix_np, 0.0)

    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if len(values) == 0:
        return np.zeros_like(matrix_np)

    threshold_value = np.percentile(values, 100 - percentile)
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0.0)
    np.fill_diagonal(thresholded_np, 0.0)
    return thresholded_np

def compute_nodewise_clustering_from_np(matrix_np):
    G = nx.from_numpy_array(matrix_np)
    for u, v, d in G.edges(data=True):
        d["weight"] = float(matrix_np[u, v])
    clust = nx.clustering(G, weight="weight")
    vals = [clust[i] for i in range(len(clust))]
    return np.asarray(vals, dtype=np.float32).reshape(-1, 1)

def compute_clustering_coefficient_from_np(matrix_np):
    G = nx.from_numpy_array(matrix_np)
    for u, v, d in G.edges(data=True):
        d["weight"] = float(matrix_np[u, v])
    return float(nx.average_clustering(G, weight="weight"))

def compute_path_length_from_np(matrix_np):
    G = nx.from_numpy_array(matrix_np)

    zero_edges = [(u, v) for u, v, _ in G.edges(data=True) if matrix_np[u, v] <= 0]
    G.remove_edges_from(zero_edges)

    for u, v, d in G.edges(data=True):
        w = float(matrix_np[u, v])
        d["distance"] = 1.0 / w if w > 0 else float("inf")

    if G.number_of_edges() == 0:
        return float("nan")

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    try:
        return float(nx.average_shortest_path_length(G, weight="distance"))
    except Exception:
        return float("nan")

def compute_global_efficiency_from_np(matrix_np):
    G = nx.from_numpy_array(matrix_np)
    return float(nx.global_efficiency(G))

def compute_local_efficiency_from_np(matrix_np):
    G = nx.from_numpy_array(matrix_np)
    return float(nx.local_efficiency(G))

def build_node_feature_dict(df_fa_t, df_vol_t):
    multimodal = {}
    common_subjects = sorted(set(df_fa_t.index).intersection(df_vol_t.index))
    for subj in common_subjects:
        fa = torch.tensor(df_fa_t.loc[subj].values, dtype=torch.float32)
        vol = torch.tensor(df_vol_t.loc[subj].values, dtype=torch.float32)
        multimodal[subj] = torch.stack([fa, vol], dim=1)
    return multimodal

def normalize_multimodal_nodewise(feature_dict):
    if len(feature_dict) == 0:
        raise ValueError("feature_dict is empty before normalization.")

    all_features = torch.stack(list(feature_dict.values()))
    means = all_features.mean(dim=0)
    stds = all_features.std(dim=0) + 1e-8

    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

def get_global_feature_columns(feature_set, feature_mode):
    """
    Return the ordered list of graph-level/global columns used for each model.

    Diagnosis, cognitive status, and clinical outcomes are NOT model inputs.
    They should be used later as validation/association outcomes.
    """

    demographics = [
        "sex_encoded",
        "genotype_encoded",
    ]

    graph_metrics = [
        "Clustering_Coeff",
        "Path_Length",
        "Global_Efficiency",
        "Local_Efficiency",
    ]

    body = [
        "bmi",
    ]

    cardiovascular = [
        "bp_sys",
        "bp_dia",
        "pulse",
    ]

    biomarkers = [
        "amyloid_40",
        "amyloid_42",
        "tau_total",
        "ptau",
        "ptau217",
        "nfl",
        "gfap",
    ]

    pca_features = PCA_COLUMNS

    if feature_set == "imaging_only":
        return graph_metrics

    if feature_set == "imaging_demographics":
        return demographics + graph_metrics

    if feature_set == "imaging_biomarkers":
        if feature_mode == "pca":
            return demographics + graph_metrics + pca_features
        return demographics + graph_metrics + biomarkers

    if feature_set == "full":
        if feature_mode == "pca":
            return demographics + graph_metrics + body + cardiovascular + pca_features
        return demographics + graph_metrics + body + cardiovascular + biomarkers

    if feature_set == "full_no_cardiovascular":
        if feature_mode == "pca":
            return demographics + graph_metrics + body + pca_features
        return demographics + graph_metrics + body + biomarkers

    raise ValueError(
        f"Unknown FEATURE_SET='{feature_set}'. Valid options are: "
        "imaging_only, imaging_demographics, imaging_biomarkers, full, full_no_cardiovascular"
    )


def build_global_feature_tensor(row, feature_mode, feature_set):
    """
    Build graph-level/global feature tensor using FEATURE_SET.
    Missing values are encoded as 0.0 after cohort-specific scaling/imputation.
    """
    cols = get_global_feature_columns(feature_set, feature_mode)

    vals = []
    for c in cols:
        val = row[c] if c in row.index else np.nan
        vals.append(float(val) if pd.notna(val) else 0.0)

    return torch.tensor(vals, dtype=torch.float32)


# =========================================================
# PARALLEL WORKERS
# =========================================================
def process_connectome_subject(args):
    subject, matrix_values, edge_percentile, stage = args
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    thresholded_np = threshold_connectome(matrix_values, percentile=edge_percentile)
    threshold_secs = time.perf_counter() - t1

    t2 = time.perf_counter()
    log_np = np.log1p(thresholded_np).astype(np.float32)
    log_secs = time.perf_counter() - t2

    t3 = time.perf_counter()
    clustering = compute_clustering_coefficient_from_np(log_np)
    path_length = compute_path_length_from_np(log_np)
    global_efficiency = compute_global_efficiency_from_np(log_np)
    local_efficiency = compute_local_efficiency_from_np(log_np)
    metric_secs = time.perf_counter() - t3

    total_secs = time.perf_counter() - t0

    return {
        "subject": subject,
        "stage": stage,
        "log_matrix_values": log_np,
        "clustering": clustering,
        "path_length": path_length,
        "global_efficiency": global_efficiency,
        "local_efficiency": local_efficiency,
        "threshold_secs": threshold_secs,
        "log_secs": log_secs,
        "metric_secs": metric_secs,
        "total_secs": total_secs,
    }

def build_graph_object_worker(args):
    subject, matrix_values, node_feature_values, age_value, global_feature_values, stage = args
    t0 = time.perf_counter()

    mat = np.asarray(matrix_values, dtype=np.float32)
    rows, cols = np.where(np.triu(mat, k=1) > 0)

    if len(rows) == 0:
        return {
            "subject": subject,
            "stage": stage,
            "ok": False,
            "error": "No nonzero edges after thresholding",
            "build_secs": time.perf_counter() - t0,
        }

    weights = mat[rows, cols]

    edge_index = torch.tensor(
        np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows]),
        ]),
        dtype=torch.long
    )

    edge_attr = torch.tensor(
        np.concatenate([weights, weights]),
        dtype=torch.float32
    ).view(-1, 1)

    node_features = torch.tensor(node_feature_values, dtype=torch.float32)
    age = torch.tensor([float(age_value)], dtype=torch.float32)
    global_feat = torch.tensor(global_feature_values, dtype=torch.float32).unsqueeze(0)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=age,
        age_chrono=age.clone(),
        global_features=global_feat
    )
    # Store the scan/session key under multiple names so downstream training,
    # validation, and Figure 8 can recover Hxxxx_y0/Hxxxx_y2 reliably.
    data.subject_id = subject
    data.graph_id = subject
    data.connectome_key = subject
    data.runno = subject

    return {
        "subject": subject,
        "stage": stage,
        "ok": True,
        "data": data,
        "build_secs": time.perf_counter() - t0,
    }


# =========================================================
# PARALLEL RUNNERS
# =========================================================
def parallel_process_connectomes(connectome_dict, edge_percentile, stage):
    results = {}
    metric_dict = {}
    timing_rows = []

    jobs = [
        (subject, matrix.to_numpy(dtype=np.float32), edge_percentile, stage)
        for subject, matrix in connectome_dict.items()
    ]

    print(f"Parallel processing {len(jobs)} connectomes for {stage} with N_JOBS={N_JOBS}")
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futures = [ex.submit(process_connectome_subject, job) for job in jobs]

        for i, fut in enumerate(as_completed(futures), 1):
            out = fut.result()
            subject = out["subject"]

            results[subject] = pd.DataFrame(out["log_matrix_values"])
            metric_dict[subject] = {
                "Clustering_Coeff": out["clustering"],
                "Path_Length": out["path_length"],
                "Global_Efficiency": out["global_efficiency"],
                "Local_Efficiency": out["local_efficiency"],
            }

            timing_rows.append({
                "subject": subject,
                "stage": stage,
                "threshold_secs": out["threshold_secs"],
                "log_secs": out["log_secs"],
                "metric_secs": out["metric_secs"],
                "total_secs": out["total_secs"],
            })

            if i % 25 == 0 or i == len(jobs):
                print(f"  completed {i}/{len(jobs)}")

    return results, metric_dict, pd.DataFrame(timing_rows)

def parallel_build_graphs(
    subjects,
    log_connectomes,
    node_features_dict,
    df_features,
    feature_mode,
    feature_set,
    stage,
    graph_subject_key_col,
    node_feature_key_col
):
    """
    Build PyG Data objects in the parent process.

    Connectome thresholding and graph-metric computation remain parallel.
    PyTorch/PyG objects are intentionally NOT returned through a
    ProcessPoolExecutor because Linux file-descriptor/shared-memory transfer
    can fail with:
        RuntimeError: received 0 items of ancdata
        BrokenProcessPool

    This parent-process build is fast relative to NetworkX metric computation
    and avoids multiprocessing serialization failures.
    """
    graph_data_list = []
    aligned_subjects = []
    timing_rows = []

    subject_values = set(df_features[graph_subject_key_col].astype(str).values)

    jobs = []

    for subject in subjects:
        if subject not in log_connectomes:
            continue
        if subject not in subject_values:
            continue

        row = df_features[
            df_features[graph_subject_key_col].astype(str) == str(subject)
        ]
        if row.empty:
            continue
        row = row.iloc[0]

        node_key = str(row[node_feature_key_col])
        if node_key not in node_features_dict:
            continue

        matrix_np = log_connectomes[subject].to_numpy(dtype=np.float32)

        node_base = node_features_dict[node_key].numpy()
        nodewise_clust = compute_nodewise_clustering_from_np(matrix_np)
        node_features = np.concatenate(
            [
                node_base[:, 0:1],
                node_base[:, 1:2],
                nodewise_clust,
            ],
            axis=1,
        )

        global_feat = build_global_feature_tensor(
            row,
            feature_mode,
            feature_set,
        ).numpy()

        jobs.append(
            (
                subject,
                matrix_np,
                node_features,
                float(row["age"]),
                global_feat,
                stage,
            )
        )

    print(
        f"Parent-process graph build for {stage}: "
        f"{len(jobs)} subjects"
    )

    for i, job in enumerate(jobs, 1):
        out = build_graph_object_worker(job)

        timing_rows.append({
            "subject": out["subject"],
            "stage": out["stage"],
            "build_secs": out["build_secs"],
            "ok": out["ok"],
        })

        if out["ok"]:
            graph_data_list.append(out["data"])
            aligned_subjects.append(out["subject"])

        if i % 25 == 0 or i == len(jobs):
            print(f"  completed {i}/{len(jobs)}")

    return (
        graph_data_list,
        aligned_subjects,
        pd.DataFrame(timing_rows),
    )


# =========================================================
# NODE FEATURE LOADING
# =========================================================

def load_raw_node_features(config, valid_subjects_all, n_rois, node_feature_key_name="subject_key"):
    df_fa = pd.read_csv(config["fa_path"], sep=config["fa_sep"])
    if "ROI" in df_fa.columns:
        try:
            df_fa["ROI_numeric"] = pd.to_numeric(df_fa["ROI"], errors="coerce")
            if len(df_fa) > 0 and pd.isna(df_fa["ROI_numeric"].iloc[0]):
                df_fa = df_fa.iloc[1:].copy()
            df_fa = df_fa[df_fa["ROI"].astype(str) != "0"].copy().reset_index(drop=True)
            df_fa = df_fa.drop(columns=["ROI_numeric"], errors="ignore")
        except Exception:
            pass

    subject_cols_fa = [col for col in df_fa.columns if str(col).strip() in valid_subjects_all]
    df_fa_t = df_fa[subject_cols_fa].transpose()
    df_fa_t.columns = [f"ROI_{i+1}" for i in range(df_fa_t.shape[1])]
    df_fa_t.index = df_fa_t.index.astype(str).str.strip()
    df_fa_t.index.name = node_feature_key_name
    df_fa_t = df_fa_t.astype(float)

    df_vol = pd.read_csv(config["vol_path"])
    if "ROI" in df_vol.columns:
        df_vol = df_vol[df_vol["ROI"].astype(str) != "-1"].copy().reset_index(drop=True)

    subject_cols_vol = [col for col in df_vol.columns if str(col).strip() in valid_subjects_all]
    df_vol_t = df_vol[subject_cols_vol].transpose()
    df_vol_t.columns = [f"ROI_{i+1}" for i in range(df_vol_t.shape[1])]
    df_vol_t.index = df_vol_t.index.astype(str).str.strip()
    df_vol_t.index.name = node_feature_key_name
    df_vol_t = df_vol_t.astype(float)

    if df_fa_t.shape[1] != n_rois and df_fa_t.shape[1] > 0:
        raise ValueError(f"FA node count ({df_fa_t.shape[1]}) does not match connectome node count ({n_rois}).")
    if df_vol_t.shape[1] != n_rois and df_vol_t.shape[1] > 0:
        raise ValueError(f"Volume node count ({df_vol_t.shape[1]}) does not match connectome node count ({n_rois}).")

    raw_node_features_dict = build_node_feature_dict(df_fa_t, df_vol_t)

    print("Subjects with FA+Volume:", len(raw_node_features_dict))
    if len(raw_node_features_dict) == 0:
        raise ValueError("No subjects with both FA and Volume found.")

    normalized_node_features_dict = normalize_multimodal_nodewise(raw_node_features_dict)
    return normalized_node_features_dict


# =========================================================
# MAIN COHORT BUILDER
# =========================================================
def build_graphs_for_cohort(cohort_name):
    global timings, script_t0
    timings = OrderedDict()
    script_t0 = time.perf_counter()

    config = COHORT_CONFIGS[cohort_name]
    paths = build_output_paths(cohort_name)
    subject_timing_frames = []

    graph_subject_key_col = config["graph_subject_key"]
    node_feature_key_col = config["node_feature_key"]

    print("\n" + "=" * 80)
    print(f"BUILDING COHORT: {cohort_name}")
    print("=" * 80)
    print("Saving outputs to:", paths["results_dir"])
    print("feature_set:", FEATURE_SET)
    print("global_feature_columns:", get_global_feature_columns(FEATURE_SET, config["feature_mode"]))
    print("graph_subject_key_col:", graph_subject_key_col)
    print("node_feature_key_col:", node_feature_key_col)

    with timed_block("load_connectomes"):
        connectomes, connectome_key_lookup, n_rois = load_connectomes_for_cohort(
            cohort_name,
            config["connectome_dir"],
            graph_subject_key_col
        )

    with timed_block("load_and_harmonize_metadata"):
        df_raw = load_table_auto(config["metadata_path"], sheet_name=config["metadata_sheet"])
        print("Raw metadata shape:", df_raw.shape)

        pca_df = None
        if config["feature_mode"] == "pca" and config["pca_path"] is not None and os.path.exists(config["pca_path"]):
            pca_df = load_table_auto(config["pca_path"])
            print("Loaded PCA file:", config["pca_path"], "| shape:", pca_df.shape)
        elif config["feature_mode"] == "pca":
            print("No PCA file found. PCA columns will be filled with zeros.")

        df_h = harmonize_metadata(df_raw, cohort_name, pca_df=pca_df)
        df_h = add_metadata_match_id(df_h, cohort_name)

        required_h_cols = ["match_id", "connectome_key", "regional_id", "age", "sex", "genotype", "is_healthy_control"]
        missing = [c for c in required_h_cols if c not in df_h.columns]
        if missing:
            raise ValueError(f"Missing harmonized columns: {missing}")

        df_h = df_h.dropna(subset=["match_id", "connectome_key", "age"]).copy()

        if graph_subject_key_col == "connectome_key":
            # Longitudinal cohorts use scan/session IDs here. This keeps
            # Hxxxx_y0 and Hxxxx_y2 as separate rows while removing true
            # duplicate scan rows only.
            df_h = df_h.drop_duplicates(subset=["connectome_key"]).copy()
        else:
            df_h = df_h.drop_duplicates(subset=["match_id"]).copy()

        if cohort_name == "HABS":
            print("\nHABS harmonized metadata longitudinal check after connectome_key de-duplication:")
            print(pd.DataFrame([summarize_longitudinal_ids_for_qc(df_h["connectome_key"], "df_h:connectome_key")]).to_string(index=False))

        save_cleaned_metadata(df_raw, df_h, paths["cleaned_metadata_path"], cohort_name)
        print("Saved cleaned harmonized metadata:", paths["cleaned_metadata_path"])

    with timed_block("match_metadata_to_connectomes"):
        df_h[graph_subject_key_col] = df_h[graph_subject_key_col].astype(str)
        df_matched = df_h[df_h[graph_subject_key_col].isin(connectomes.keys())].copy()
        df_matched["connectome_full_key"] = df_matched[graph_subject_key_col].map(connectome_key_lookup)

        df_controls = df_matched[df_matched["is_healthy_control"] == 1].copy()
        df_all_subjects = df_matched.copy()

        print("Matched connectomes with metadata:", df_matched.shape[0])
        print("Healthy matched subjects:", df_controls.shape[0])
        print("All matched subjects:", df_all_subjects.shape[0])

        if df_controls.shape[0] == 0:
            raise ValueError(f"[{cohort_name}] No healthy matched subjects found.")
        if df_all_subjects.shape[0] == 0:
            raise ValueError(f"[{cohort_name}] No all-subject matches found.")

    with timed_block("save_connectome_losses"):
        all_connectome_keys = set(connectomes.keys())
        metadata_keys = set(df_h[graph_subject_key_col].astype(str))
        lost_keys = sorted(all_connectome_keys - metadata_keys)

        df_lost = pd.DataFrame({graph_subject_key_col: lost_keys})
        df_lost["connectome_full_key"] = df_lost[graph_subject_key_col].map(connectome_key_lookup)
        df_lost["reason"] = "connectome exists but no metadata match"
        df_lost.to_csv(paths["losses_path"], index=False)

        print("Lost connectomes before metadata match:", len(df_lost))
        print("Saved to:", paths["losses_path"])

    with timed_block("select_matched_connectomes"):
        matched_connectomes_healthy = {
            str(row[graph_subject_key_col]): connectomes[str(row[graph_subject_key_col])]
            for _, row in df_controls.iterrows()
        }

        matched_connectomes_all = {
            str(row[graph_subject_key_col]): connectomes[str(row[graph_subject_key_col])]
            for _, row in df_all_subjects.iterrows()
        }

        print("Healthy connectomes selected:", len(matched_connectomes_healthy))
        print("All connectomes selected:", len(matched_connectomes_all))

    with timed_block("load_raw_node_features"):
        valid_node_subjects_all = set(df_all_subjects[node_feature_key_col].astype(str))

        normalized_node_features_dict = load_raw_node_features(
            config,
            valid_node_subjects_all,
            n_rois,
            node_feature_key_name=node_feature_key_col
        )

    with timed_block("parallel_threshold_log_metrics_healthy"):
        log_thresholded_connectomes_healthy, metric_dict_healthy, timing_healthy_proc = parallel_process_connectomes(
            matched_connectomes_healthy,
            EDGE_PERCENTILE,
            "healthy_process"
        )
        subject_timing_frames.append(timing_healthy_proc)

    with timed_block("parallel_threshold_log_metrics_all"):
        log_thresholded_connectomes_all, metric_dict_all, timing_all_proc = parallel_process_connectomes(
            matched_connectomes_all,
            EDGE_PERCENTILE,
            "all_process"
        )
        subject_timing_frames.append(timing_all_proc)

    with timed_block("attach_graph_metrics"):
        metric_cols = ["Clustering_Coeff", "Path_Length", "Global_Efficiency", "Local_Efficiency"]

        df_controls_gm = df_controls.reset_index(drop=True).copy()
        for c in metric_cols:
            df_controls_gm[c] = np.nan

        for subject, metrics in metric_dict_healthy.items():
            df_controls_gm.loc[
                df_controls_gm[graph_subject_key_col].astype(str) == str(subject),
                metric_cols
            ] = [
                metrics["Clustering_Coeff"],
                metrics["Path_Length"],
                metrics["Global_Efficiency"],
                metrics["Local_Efficiency"],
            ]

        df_all_gm = df_all_subjects.reset_index(drop=True).copy()
        for c in metric_cols:
            df_all_gm[c] = np.nan

        for subject, metrics in metric_dict_all.items():
            df_all_gm.loc[
                df_all_gm[graph_subject_key_col].astype(str) == str(subject),
                metric_cols
            ] = [
                metrics["Clustering_Coeff"],
                metrics["Path_Length"],
                metrics["Global_Efficiency"],
                metrics["Local_Efficiency"],
            ]

    with timed_block("process_global_features"):
        df_controls_gm = df_controls_gm.dropna(subset=["age", "sex", node_feature_key_col] + metric_cols).reset_index(drop=True)
        df_all_gm = df_all_gm.dropna(subset=["age", "sex", node_feature_key_col] + metric_cols).reset_index(drop=True)

        sex_values_train = df_controls_gm["sex"].astype(str).copy()
        geno_values_train = df_controls_gm["genotype"].astype(str).copy()

        if "UNK" not in set(sex_values_train):
            sex_values_train = pd.concat([sex_values_train, pd.Series(["UNK"])], ignore_index=True)
        if "GENO_UNKNOWN" not in set(geno_values_train):
            geno_values_train = pd.concat([geno_values_train, pd.Series(["GENO_UNKNOWN"])], ignore_index=True)

        le_sex = LabelEncoder()
        le_sex.fit(sex_values_train)

        le_geno = LabelEncoder()
        le_geno.fit(geno_values_train)

        df_controls_gm["sex_encoded"] = safe_label_transform(df_controls_gm["sex"], le_sex, "UNK")
        df_all_gm["sex_encoded"] = safe_label_transform(df_all_gm["sex"], le_sex, "UNK")

        df_controls_gm["genotype_encoded"] = safe_label_transform(
            df_controls_gm["genotype"].fillna("GENO_UNKNOWN"), le_geno, "GENO_UNKNOWN"
        )
        df_all_gm["genotype_encoded"] = safe_label_transform(
            df_all_gm["genotype"].fillna("GENO_UNKNOWN"), le_geno, "GENO_UNKNOWN"
        )

        scale_cols_shared = ["Clustering_Coeff", "Path_Length", "Global_Efficiency", "Local_Efficiency", "bmi"]
        for c in scale_cols_shared:
            if c not in df_controls_gm.columns:
                df_controls_gm[c] = np.nan
            if c not in df_all_gm.columns:
                df_all_gm[c] = np.nan

        shared_mean = df_controls_gm[scale_cols_shared].mean()
        shared_std = df_controls_gm[scale_cols_shared].std().replace(0, 1e-8)

        df_controls_gm[scale_cols_shared] = (df_controls_gm[scale_cols_shared] - shared_mean) / shared_std
        df_all_gm[scale_cols_shared] = (df_all_gm[scale_cols_shared] - shared_mean) / shared_std

        if config["feature_mode"] == "pca":
            for c in PCA_COLUMNS:
                df_controls_gm[c] = as_numeric(safe_series(df_controls_gm, c)).fillna(0.0)
                df_all_gm[c] = as_numeric(safe_series(df_all_gm, c)).fillna(0.0)

            pca_mean = df_controls_gm[PCA_COLUMNS].mean()
            pca_std = df_controls_gm[PCA_COLUMNS].std().replace(0, 1e-8)

            df_controls_gm[PCA_COLUMNS] = (df_controls_gm[PCA_COLUMNS] - pca_mean) / pca_std
            df_all_gm[PCA_COLUMNS] = (df_all_gm[PCA_COLUMNS] - pca_mean) / pca_std

        else:
            biomarker_cols = [
                "bp_sys", "bp_dia", "pulse",
                "amyloid_40", "amyloid_42",
                "tau_total", "ptau", "ptau217",
                "nfl", "gfap",
            ]

            for c in biomarker_cols:
                df_controls_gm[c] = as_numeric(safe_series(df_controls_gm, c))
                df_all_gm[c] = as_numeric(safe_series(df_all_gm, c))

            bio_mean = df_controls_gm[biomarker_cols].mean()
            bio_std = df_controls_gm[biomarker_cols].std().replace(0, 1e-8)
            
            df_controls_gm[biomarker_cols] = (df_controls_gm[biomarker_cols] - bio_mean) / bio_std
            df_all_gm[biomarker_cols] = (df_all_gm[biomarker_cols] - bio_mean) / bio_std
            
            # optional: for model compatibility only
            df_controls_gm[biomarker_cols] = df_controls_gm[biomarker_cols].fillna(0.0)
            df_all_gm[biomarker_cols] = df_all_gm[biomarker_cols].fillna(0.0)

            bio_mean = df_controls_gm[biomarker_cols].mean()
            bio_std = df_controls_gm[biomarker_cols].std().replace(0, 1e-8)

            df_controls_gm[biomarker_cols] = (df_controls_gm[biomarker_cols] - bio_mean) / bio_std
            df_all_gm[biomarker_cols] = (df_all_gm[biomarker_cols] - bio_mean) / bio_std

        encoding_info = {
            "cohort": cohort_name,
            "feature_mode": config["feature_mode"],
            "feature_set": FEATURE_SET,
            "global_feature_columns": get_global_feature_columns(FEATURE_SET, config["feature_mode"]),
            "graph_subject_key": graph_subject_key_col,
            "node_feature_key": node_feature_key_col,
            "sex_classes_train": [str(x) for x in le_sex.classes_.tolist()],
            "genotype_classes_train": [str(x) for x in le_geno.classes_.tolist()],
        }
        with open(paths["encoding_info_json"], "w") as f:
            json.dump(encoding_info, f, indent=2)

    with timed_block("parallel_build_healthy_graph_objects"):
        healthy_subjects = list(df_controls_gm[graph_subject_key_col].astype(str))
        graph_data_list, aligned_subjects_healthy, timing_build_healthy = parallel_build_graphs(
            healthy_subjects,
            log_thresholded_connectomes_healthy,
            normalized_node_features_dict,
            df_controls_gm,
            config["feature_mode"],
            FEATURE_SET,
            "build_healthy",
            graph_subject_key_col,
            node_feature_key_col
        )
        subject_timing_frames.append(timing_build_healthy)

        if len(graph_data_list) == 0:
            raise ValueError(f"[{cohort_name}] No healthy graph objects were built.")

        aligned_metadata = (
            df_controls_gm[df_controls_gm[graph_subject_key_col].isin(aligned_subjects_healthy)]
            .copy()
            .reset_index(drop=True)
        )

        for data in graph_data_list:
            row = aligned_metadata[aligned_metadata[graph_subject_key_col] == data.subject_id]
            if not row.empty:
                data.regional_id = row.iloc[0]["regional_id"]
                data.match_id = row.iloc[0].get("match_id", None)
                data.connectome_full_key = row.iloc[0].get("connectome_full_key", None)
                if cohort_name == "HABS":
                    scan_key = str(row.iloc[0][graph_subject_key_col])
                    data.subject_id = scan_key
                    data.graph_id = scan_key
                    data.connectome_key = scan_key
                    data.runno = scan_key

        first_healthy = graph_data_list[0]
        print("\nExample HEALTHY Data object:")
        print("subject_id:", first_healthy.subject_id)
        print("regional_id:", getattr(first_healthy, "regional_id", None))
        print("connectome_full_key:", getattr(first_healthy, "connectome_full_key", None))
        print("x:", first_healthy.x.shape)
        print("edge_index:", first_healthy.edge_index.shape)
        print("edge_attr:", first_healthy.edge_attr.shape)
        print("global_features:", first_healthy.global_features.shape)
        print("y:", first_healthy.y.item())

    with timed_block("parallel_build_all_graph_objects"):
        all_subjects = list(df_all_gm[graph_subject_key_col].astype(str))
        graph_data_list_all, aligned_subjects_all, timing_build_all = parallel_build_graphs(
            all_subjects,
            log_thresholded_connectomes_all,
            normalized_node_features_dict,
            df_all_gm,
            config["feature_mode"],
            FEATURE_SET,
            "build_all",
            graph_subject_key_col,
            node_feature_key_col
        )
        subject_timing_frames.append(timing_build_all)

        if len(graph_data_list_all) == 0:
            raise ValueError(f"[{cohort_name}] No all-subject graph objects were built.")

        aligned_metadata_all = (
            df_all_gm[df_all_gm[graph_subject_key_col].isin(aligned_subjects_all)]
            .copy()
            .reset_index(drop=True)
        )

        for data in graph_data_list_all:
            row = aligned_metadata_all[aligned_metadata_all[graph_subject_key_col] == data.subject_id]
            if not row.empty:
                data.regional_id = row.iloc[0]["regional_id"]
                data.match_id = row.iloc[0].get("match_id", None)
                data.connectome_full_key = row.iloc[0].get("connectome_full_key", None)
                if cohort_name == "HABS":
                    scan_key = str(row.iloc[0][graph_subject_key_col])
                    data.subject_id = scan_key
                    data.graph_id = scan_key
                    data.connectome_key = scan_key
                    data.runno = scan_key

        first_all = graph_data_list_all[0]
        print("\nExample ALL Data object:")
        print("subject_id:", first_all.subject_id)
        print("regional_id:", getattr(first_all, "regional_id", None))
        print("connectome_full_key:", getattr(first_all, "connectome_full_key", None))
        print("x:", first_all.x.shape)
        print("edge_index:", first_all.edge_index.shape)
        print("edge_attr:", first_all.edge_attr.shape)
        print("global_features:", first_all.global_features.shape)
        print("y:", first_all.y.item())

    with timed_block("save_outputs"):
    # -----------------------------
    # HEALTHY SET
    # -----------------------------
        torch.save(graph_data_list, paths["save_graphs_path"])
    
        aligned_metadata_raw = (
            df_controls[df_controls[graph_subject_key_col].isin(aligned_subjects_healthy)]
            .copy()
            .reset_index(drop=True)
        )
        aligned_metadata_raw.to_csv(paths["save_metadata_path_raw"], index=False)
    
        aligned_metadata.to_csv(paths["save_metadata_path"], index=False)
    
        summary_df = pd.DataFrame([{
            "cohort": cohort_name,
            "feature_set": FEATURE_SET,
            "global_feature_columns": json.dumps(get_global_feature_columns(FEATURE_SET, config["feature_mode"])),
            "graph_subject_key": graph_subject_key_col,
            "node_feature_key": node_feature_key_col,
            "n_connectomes_loaded": len(connectomes),
            "n_metadata_rows": len(df_h),
            "n_matched_rows": len(df_matched),
            "n_healthy_rows": len(df_controls),
            "n_healthy_rows_after_feature_filter": len(df_controls_gm),
            "n_graph_objects_saved": len(graph_data_list),
            "n_rois": n_rois,
            "node_feature_dim": 3,
            "global_feature_dim": int(graph_data_list[0].global_features.shape[1]),
            "graph_path": paths["save_graphs_path"],
            "metadata_path_raw": paths["save_metadata_path_raw"],
            "metadata_path_model_ready": paths["save_metadata_path"],
        }])
        summary_df.to_csv(paths["save_summary_path"], index=False)
    
        # -----------------------------
        # ALL-SUBJECT SET
        # -----------------------------
        torch.save(graph_data_list_all, paths["save_graphs_all_path"])
    
        aligned_metadata_all_raw = (
            df_all_subjects[df_all_subjects[graph_subject_key_col].isin(aligned_subjects_all)]
            .copy()
            .reset_index(drop=True)
        )
        aligned_metadata_all_raw.to_csv(paths["save_metadata_all_path_raw"], index=False)
    
        aligned_metadata_all.to_csv(paths["save_metadata_all_path"], index=False)
    
        summary_all_df = pd.DataFrame([{
            "cohort": cohort_name,
            "feature_set": FEATURE_SET,
            "global_feature_columns": json.dumps(get_global_feature_columns(FEATURE_SET, config["feature_mode"])),
            "graph_subject_key": graph_subject_key_col,
            "node_feature_key": node_feature_key_col,
            "n_connectomes_loaded": len(connectomes),
            "n_metadata_rows": len(df_h),
            "n_matched_rows": len(df_matched),
            "n_all_rows_after_feature_filter": len(df_all_gm),
            "n_graph_objects_saved_all": len(graph_data_list_all),
            "n_rois": n_rois,
            "node_feature_dim": 3,
            "global_feature_dim": int(graph_data_list_all[0].global_features.shape[1]),
            "graph_path_all": paths["save_graphs_all_path"],
            "metadata_path_all_raw": paths["save_metadata_all_path_raw"],
            "metadata_path_all_model_ready": paths["save_metadata_all_path"],
        }])
        summary_all_df.to_csv(paths["save_summary_all_path"], index=False)

        if cohort_name == "HABS":
            qc_rows = []
            qc_rows.append(summarize_longitudinal_ids_for_qc(df_h["connectome_key"], "df_h:connectome_key"))
            qc_rows.append(summarize_longitudinal_ids_for_qc(df_controls[graph_subject_key_col], "df_controls:connectome_key"))
            qc_rows.append(summarize_longitudinal_ids_for_qc(df_all_subjects[graph_subject_key_col], "df_all_subjects:connectome_key"))
            qc_rows.append(summarize_longitudinal_ids_for_qc(aligned_subjects_healthy, "healthy_graphs:aligned_subjects"))
            qc_rows.append(summarize_longitudinal_ids_for_qc(aligned_subjects_all, "all_graphs:aligned_subjects"))
            qc_rows.append(summarize_longitudinal_ids_for_qc(aligned_metadata_all[graph_subject_key_col], "aligned_metadata_all:connectome_key"))
            save_graphbuilder_longitudinal_qc(paths, cohort_name, qc_rows)
    
        print("\n=== SAVED CLEANED METADATA ===")
        print(paths["cleaned_metadata_path"])
    
        print("\n=== SAVED HEALTHY ===")
        print(paths["save_graphs_path"])
        print("RAW aligned:", paths["save_metadata_path_raw"])
        print("MODEL ready:", paths["save_metadata_path"])
        print(paths["save_summary_path"])
    
        print("\n=== SAVED ALL ===")
        print(paths["save_graphs_all_path"])
        print("RAW aligned:", paths["save_metadata_all_path_raw"])
        print("MODEL ready:", paths["save_metadata_all_path"])
        print(paths["save_summary_all_path"])

    with timed_block("save_timing_outputs"):
        all_subject_timing_df = pd.concat(subject_timing_frames, ignore_index=True)
        all_subject_timing_df.to_csv(paths["subject_timing_csv"], index=False)

        total_script_secs = time.perf_counter() - script_t0
        timings["total_script"] = total_script_secs

        with open(paths["timing_summary_path"], "w") as f:
            f.write("GRAPH BUILD TIMING SUMMARY\n")
            for name, secs in timings.items():
                f.write(f"{name}={secs:.3f}\n")
                f.write(f"{name}_hms={fmt_seconds(secs)}\n")

            f.write("\nPER-SUBJECT TIMING SUMMARY\n")
            numeric_cols = [c for c in all_subject_timing_df.columns if c.endswith("_secs")]
            for c in numeric_cols:
                vals = pd.to_numeric(all_subject_timing_df[c], errors="coerce").dropna()
                if len(vals) > 0:
                    f.write(f"{c}_mean={vals.mean():.6f}\n")
                    f.write(f"{c}_std={vals.std():.6f}\n")
                    f.write(f"{c}_median={vals.median():.6f}\n")
                    f.write(f"{c}_max={vals.max():.6f}\n")

            f.write(f"\nN_JOBS={N_JOBS}\n")
            f.write(f"EDGE_PERCENTILE={EDGE_PERCENTILE}\n")
            f.write(f"FEATURE_SET={FEATURE_SET}\n")

        print("Saved subject timing CSV:", paths["subject_timing_csv"])
        print("Saved timing summary:", paths["timing_summary_path"])

    print(f"\nDone: {cohort_name}")


# =========================================================
# MAIN
# =========================================================
def main():
    global FEATURE_SET

    if RUN_ALL_COHORTS:
        cohorts_to_run = VALID_COHORTS
    else:
        if COHORT_NAME not in COHORT_CONFIGS:
            raise ValueError(
                f"COHORT_NAME must be one of: {list(COHORT_CONFIGS.keys())}"
            )
        cohorts_to_run = [COHORT_NAME]

    print("=" * 100)
    print("UNIFIED GRAPH BUILDER")
    print("=" * 100)
    print("WORK:", WORK)
    print("Cohorts to run:", cohorts_to_run)
    print("Feature sets to run:", FEATURE_SETS_TO_RUN)
    print("N_JOBS:", N_JOBS)
    print("EDGE_PERCENTILE:", EDGE_PERCENTILE)
    print("Overwrite:", ARGS.overwrite)

    for cohort_name in cohorts_to_run:
        for feature_set in FEATURE_SETS_TO_RUN:
            FEATURE_SET = feature_set

            if ARGS.overwrite:
                import shutil

                out_dir = os.path.join(
                    WORK,
                    "ines/results/harmonized",
                    cohort_name,
                    "graphs",
                    FEATURE_SET,
                )
                if os.path.isdir(out_dir):
                    print()
                    print("Removing existing output directory:", out_dir)
                    shutil.rmtree(out_dir)

            print()
            print("#" * 100)
            print(f"RUN: cohort={cohort_name} feature_set={FEATURE_SET}")
            print("#" * 100)

            build_graphs_for_cohort(cohort_name)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
