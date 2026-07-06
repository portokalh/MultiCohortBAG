#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure7_SHAP_subject_cluster_enriched_risk_profiles.py

Cross-cohort SHAP-defined subject clusters + enriched risk profiles.

Purpose
-------
Uses existing per-subject SHAP files from Figure7_SHAP.
Does NOT recompute SHAP.

Main analysis:
  1. Load subject-level SHAP profiles across ADNI, ADRC, HABS, AD-DECODE.
  2. Build cross-cohort subject x SHAP-feature matrix.
  3. Cluster subjects across cohorts together.
  4. Merge current enriched prediction metadata:
       subject_level_validation_input_enriched_for_Figure4.csv
  5. Test whether SHAP-defined clusters differ in:
       cBAG, APOE4, sex, diagnosis/cognitive status,
       cognition, amyloid/tau/GFAP/NfL, hippocampal metrics,
       graph metrics, BMI/BP/diabetes.
  6. Save figures and tables.

Recommended:
  python Figure7_SHAP_subject_cluster_enriched_risk_profiles.py \
    --model imaging_only \
    --kinds node,global \
    --n-clusters 4

Primary output:
  .../Figure7_SHAP/cross_cohort_aggregated/imaging_only/
      subject_shap_clusters_enriched_risk_profiles/
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency, fisher_exact, kruskal
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Paths
# =============================================================================

DEFAULT_ROOT = Path(
    "/mnt/newStor/paros/paros_WORK/ines/results/"
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

DEFAULT_SHAP_ROOT = DEFAULT_ROOT / "Figure7_SHAP"

PRED_ROOT = Path("/mnt/newStor/paros/paros_WORK/ines/results")

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

COHORT_CONFIG = {
    "ADNI": {
        "slug": "adni",
        "pred_dir": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    },
    "ADRC": {
        "slug": "adrc",
        "pred_dir": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    },
    "HABS": {
        "slug": "habs",
        "pred_dir": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    },
    "AD_DECODE": {
        "slug": "addecode",
        "pred_dir": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    },
}

KIND_SUBDIR = {
    "global": "global_feature_shap",
    "node": "node_feature_shap",
    "edge": "edge_shap",
}

KIND_PATTERN = {
    "global": "global_feature_shap_subject_*.csv",
    "node": "node_feature_shap_subject_*.csv",
    "edge": "edge_shap_subject_*.csv",
}

KIND_PREFIX = {
    "global": "global_feature_shap_subject_",
    "node": "node_feature_shap_subject_",
    "edge": "edge_shap_subject_",
}


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_name(x) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def normalize_key(x, cohort: Optional[str] = None) -> Optional[str]:
    """
    Normalize SHAP filenames and enriched metadata session IDs.

    Handles:
      H7282_y0, H7282_y2
      R0908_y0, R4288_y4
      ADRC0097, D0097
      S00001
    """
    if pd.isna(x):
        return None

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "missing"}:
        return None

    s = Path(s).stem
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"_conn_plain$", "", s, flags=re.I)
    s = re.sub(r"conn_plain$", "", s, flags=re.I)
    s = re.sub(r"_connectome.*$", "", s, flags=re.I)

    # HABS / ADNI session keys
    m = re.search(r"\b([RH]\d{4,5})[_-]?(y\d+)\b", s, flags=re.I)
    if m:
        return f"{m.group(1).upper()}_{m.group(2).lower()}"

    # Already H####_y# / R####_y#
    m = re.search(r"\b([RH]\d{4,5}_y\d+)\b", s, flags=re.I)
    if m:
        return m.group(1).upper().replace("_Y", "_y")

    # ADRC
    m = re.search(r"ADRC[_-]?0*(\d{3,5})", s, flags=re.I)
    if m:
        return f"D{int(m.group(1)):04d}"

    m = re.search(r"\bD0*(\d{3,5})\b", s, flags=re.I)
    if m:
        return f"D{int(m.group(1)):04d}"

    # AD-DECODE
    m = re.search(r"\bS0*(\d{1,5})\b", s, flags=re.I)
    if m:
        return f"S{int(m.group(1)):05d}"

    if cohort == "AD_DECODE" and re.fullmatch(r"\d+(\.0)?", s):
        return f"S{int(float(s)):05d}"

    # plain H/R/D/S with no visit
    m = re.search(r"\b([RH]\d{4,5})\b", s, flags=re.I)
    if m:
        return m.group(1).upper()

    return s.upper()


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.apply(pd.to_numeric, errors="coerce").astype(float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0).replace(0, np.nan)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def infer_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def fdr_bh(pvals: pd.Series) -> np.ndarray:
    p = pd.to_numeric(pvals, errors="coerce").fillna(1.0)
    return multipletests(p, method="fdr_bh")[1]


# =============================================================================
# SHAP loading
# =============================================================================

def make_node_label(df: pd.DataFrame) -> Optional[pd.Series]:
    if "node_feature_label" in df.columns:
        return df["node_feature_label"].astype(str)
    if {"node_label", "feature_name"}.issubset(df.columns):
        return df["node_label"].astype(str) + " | " + df["feature_name"].astype(str)
    if {"Structure", "feature_name"}.issubset(df.columns):
        return df["Structure"].astype(str) + " | " + df["feature_name"].astype(str)
    if {"structure", "feature_name"}.issubset(df.columns):
        return df["structure"].astype(str) + " | " + df["feature_name"].astype(str)
    return None


def make_edge_label(df: pd.DataFrame) -> Optional[pd.Series]:
    if "edge_feature_label" in df.columns:
        return df["edge_feature_label"].astype(str)
    if "Edge" in df.columns:
        return df["Edge"].astype(str)
    if {"Structure_i", "Structure_j"}.issubset(df.columns):
        return df["Structure_i"].astype(str) + " -- " + df["Structure_j"].astype(str)
    if {"structure_i", "structure_j"}.issubset(df.columns):
        return df["structure_i"].astype(str) + " -- " + df["structure_j"].astype(str)
    if {"Node_i", "Node_j"}.issubset(df.columns):
        return df["Node_i"].astype(str) + " -- " + df["Node_j"].astype(str)
    return None


def read_one_subject_shap(path: Path, cohort: str, model: str, kind: str, value_col: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty or "SHAP_val" not in df.columns:
        return None

    df = df.copy()
    df["SHAP_val"] = pd.to_numeric(df["SHAP_val"], errors="coerce")
    if "abs_SHAP" not in df.columns:
        df["abs_SHAP"] = df["SHAP_val"].abs()
    else:
        df["abs_SHAP"] = pd.to_numeric(df["abs_SHAP"], errors="coerce")

    if value_col not in df.columns:
        value_col = "SHAP_val"

    if kind == "global":
        if "feature_name" not in df.columns:
            return None
        df["feature_label"] = df["feature_name"].astype(str)
    elif kind == "node":
        lab = make_node_label(df)
        if lab is None:
            return None
        df["feature_label"] = lab
    elif kind == "edge":
        lab = make_edge_label(df)
        if lab is None:
            return None
        df["feature_label"] = lab
    else:
        raise ValueError(kind)

    subject_raw = path.stem.replace(KIND_PREFIX[kind], "")
    session_key = normalize_key(subject_raw, cohort=cohort)

    out = pd.DataFrame({
        "cohort": cohort,
        "cohort_slug": COHORT_CONFIG[cohort]["slug"],
        "model": model,
        "kind": kind,
        "subject_raw_from_shap": subject_raw,
        "session_key": session_key,
        "subject_uid": cohort + "__" + str(session_key),
        "feature_label": df["feature_label"].astype(str),
        "shap_value": pd.to_numeric(df[value_col], errors="coerce"),
        "abs_shap": pd.to_numeric(df["abs_SHAP"], errors="coerce"),
        "source_shap_file": str(path),
    }).dropna(subset=["feature_label", "shap_value"])

    return out


def load_shap_long(shap_root: Path, model: str, kind: str, value_col: str, cohorts: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    logs = []

    for cohort in cohorts:
        slug = COHORT_CONFIG[cohort]["slug"]
        d = shap_root / slug / model / KIND_SUBDIR[kind]
        files = sorted(d.glob(KIND_PATTERN[kind])) if d.exists() else []

        loaded = 0
        skipped = 0
        for f in files:
            tmp = read_one_subject_shap(f, cohort, model, kind, value_col)
            if tmp is None or tmp.empty:
                skipped += 1
                continue
            frames.append(tmp)
            loaded += 1

        logs.append({
            "cohort": cohort,
            "kind": kind,
            "model": model,
            "input_dir": str(d),
            "n_files": len(files),
            "n_loaded": loaded,
            "n_skipped": skipped,
        })

    if not frames:
        return pd.DataFrame(), pd.DataFrame(logs)

    return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(logs)


def rank_features(long_df: pd.DataFrame, min_cohorts: int) -> pd.DataFrame:
    by = (
        long_df.groupby(["cohort", "feature_label"], as_index=False)
        .agg(
            mean_abs_shap=("abs_shap", "mean"),
            mean_signed_shap=("shap_value", "mean"),
            n_subjects=("subject_uid", "nunique"),
        )
    )

    ranked = (
        by.groupby("feature_label", as_index=False)
        .agg(
            mean_abs_shap_across_cohorts=("mean_abs_shap", "mean"),
            sd_abs_shap_across_cohorts=("mean_abs_shap", "std"),
            mean_signed_shap_across_cohorts=("mean_signed_shap", "mean"),
            n_cohorts=("cohort", "nunique"),
            total_n_subjects=("n_subjects", "sum"),
            cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .sort_values(["n_cohorts", "mean_abs_shap_across_cohorts"], ascending=[False, False])
        .reset_index(drop=True)
    )

    ranked = ranked[ranked["n_cohorts"] >= min_cohorts].reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def build_matrix(long_df: pd.DataFrame, ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    keep = ranked.head(top_n)["feature_label"].astype(str).tolist()
    tmp = long_df[long_df["feature_label"].isin(keep)].copy()
    mat = tmp.pivot_table(
        index="subject_uid",
        columns="feature_label",
        values="shap_value",
        aggfunc="mean",
    )
    mat = mat.reindex(columns=keep).fillna(0.0)
    return mat


# =============================================================================
# Enriched metadata loading
# =============================================================================

def enriched_file_for(cohort: str, model: str) -> Path:
    pred_dir = COHORT_CONFIG[cohort]["pred_dir"]
    return (
        PRED_ROOT / pred_dir / f"ablation_{model}" /
        "validation_figures_full_cohort" /
        "subject_level_validation_input_enriched_for_Figure4.csv"
    )


def find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def candidate_session_keys(row: pd.Series, cohort: str, df_cols: List[str]) -> List[str]:
    keys = []

    direct_cols = [
        "CONNECTOME_KEY_USED_FOR_INTERSECTION",
        "CONNECTOME_KEY_CLEAN",
        "table1_session_key",
        "connectome_key",
        "connectome_full_key",
        "graph_id",
        "DWI_subject_key",
        "DWI",
        "DWI_key",
        "MRI_Exam",
        "MRI_Exam_fixed",
        "runno",
        "subject_session",
        "session_key",
    ]

    for c in direct_cols:
        if c in df_cols:
            k = normalize_key(row.get(c), cohort=cohort)
            if k:
                keys.append(k)

    subj_cols = [
        "subject", "Subject", "subject_id", "Subject_ID", "participant_id",
        "PTID", "RID", "rid", "ID", "id", "DWI_subject_key"
    ]
    visit_cols = [
        "visit", "Visit", "VISCODE", "VISCODE2", "session", "Session",
        "DWI_visit_label", "visit_label"
    ]

    s_col = find_first_col(pd.DataFrame(columns=df_cols), subj_cols)
    v_col = find_first_col(pd.DataFrame(columns=df_cols), visit_cols)

    if s_col is not None:
        s = clean_str(row.get(s_col))
        s_norm = normalize_key(s, cohort=cohort)
        if s_norm:
            keys.append(s_norm)

        if v_col is not None:
            v = clean_str(row.get(v_col)).lower()
            if v:
                v = v.replace("year", "y").replace(" ", "")
                if re.fullmatch(r"\d+", v):
                    v = "y" + v
                if not v.startswith("y") and re.fullmatch(r"m\d+", v):
                    pass
                keys.append(normalize_key(f"{s}_{v}", cohort=cohort) or f"{s}_{v}")

    # Deduplicate while preserving order.
    out = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


def add_metadata_keys(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)

    key_rows = []
    for idx, row in df.iterrows():
        keys = candidate_session_keys(row, cohort, cols)
        if not keys:
            keys = [None]
        for k in keys:
            key_rows.append((idx, k))

    key_df = pd.DataFrame(key_rows, columns=["_row_idx", "session_key"])
    keyed = key_df.merge(
        df.reset_index().rename(columns={"index": "_row_idx"}),
        on="_row_idx",
        how="left",
    )
    keyed["cohort"] = cohort
    keyed["subject_uid"] = cohort + "__" + keyed["session_key"].astype(str)
    keyed = keyed[keyed["session_key"].notna()].copy()
    keyed = keyed.drop_duplicates(["cohort", "session_key"])
    return keyed


def derive_binary_apoe4(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "APOE4_carriage", "APOE4_strict_34_44_carrier", "APOE4_carrier",
        "APOE4", "apoe4", "APOE4_dosage", "genotype", "APOE", "APOE_genotype"
    ]
    col = find_first_col(df, candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index)

    s = df[col].astype(str).str.lower().str.replace(" ", "", regex=False)
    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[s.isin(["1", "1.0", "true", "yes", "carrier", "positive"])] = "Carrier"
    out[s.isin(["0", "0.0", "false", "no", "noncarrier", "non-carrier", "negative"])] = "Non-carrier"
    out[s.str.contains("4", na=False)] = "Carrier"
    out[s.isin(["2_2", "2_3", "3_3", "22", "23", "33", "2/2", "2/3", "3/3"])] = "Non-carrier"
    return out


def derive_sex(df: pd.DataFrame) -> pd.Series:
    col = find_first_col(df, ["sex", "Sex", "SEX", "PTGENDER", "gender", "Gender"])
    if col is None:
        return pd.Series(np.nan, index=df.index)

    s = df[col].astype(str).str.lower().str.strip()
    out = pd.Series(df[col].astype(str), index=df.index, dtype="object")
    out[s.isin(["m", "male", "1", "1.0"])] = "Male"
    out[s.isin(["f", "female", "0", "0.0", "2", "2.0"])] = "Female"
    return out


def derive_cognitive_status(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="object")

    dx_col = find_first_col(df, [
        "DX_Label_harmonized", "diagnosis", "Diagnosis", "DX", "dx",
        "cognitive_status", "Cognitive_Status", "ResearchGroup", "Group"
    ])

    if dx_col:
        s = df[dx_col].astype(str).str.lower()
        out[s.str.contains("normal|control|cn|norm", na=False)] = "CN"
        out[s.str.contains("mci", na=False)] = "MCI"
        out[s.str.contains("ad|dement|alzheimer", na=False)] = "AD"

    for col, label in [("NORMCOG_01", "CN"), ("MCI_01", "MCI"), ("AD_01", "AD"), ("DEMENTIA_01", "AD")]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            out[x.eq(1)] = label

    return out


def derive_ad_status(df: pd.DataFrame) -> pd.Series:
    cog = derive_cognitive_status(df)
    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[cog.isin(["CN", "MCI"])] = "Non-AD"
    out[cog.eq("AD")] = "AD"

    for col in ["AD_01", "DEMENTIA_01"]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            out[x.eq(1)] = "AD"
            out[x.eq(0) & out.isna()] = "Non-AD"

    return out


def load_enriched_metadata(model: str, cohorts: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    logs = []

    for cohort in cohorts:
        f = enriched_file_for(cohort, model)
        if not f.exists():
            logs.append({"cohort": cohort, "file": str(f), "status": "missing"})
            continue

        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            logs.append({"cohort": cohort, "file": str(f), "status": f"read_failed: {e}"})
            continue

        keyed = add_metadata_keys(df, cohort)
        keyed["metadata_file"] = str(f)
        keyed["apoe4_carriage_derived"] = derive_binary_apoe4(keyed)
        keyed["sex_derived"] = derive_sex(keyed)
        keyed["cognitive_status_derived"] = derive_cognitive_status(keyed)
        keyed["ad_status_derived"] = derive_ad_status(keyed)

        frames.append(keyed)

        logs.append({
            "cohort": cohort,
            "file": str(f),
            "status": "loaded",
            "rows_raw": len(df),
            "rows_keyed": len(keyed),
            "unique_session_keys": keyed["session_key"].nunique(),
            "n_columns": df.shape[1],
        })

    if not frames:
        return pd.DataFrame(), pd.DataFrame(logs)

    meta = pd.concat(frames, ignore_index=True, sort=False)
    meta = meta.drop_duplicates(["cohort", "session_key"])
    return meta, pd.DataFrame(logs)


# =============================================================================
# Clustering
# =============================================================================

def cluster_subject_matrix(mat_scaled: pd.DataFrame, n_clusters: int, linkage_method: str, metric: str):
    if mat_scaled.shape[0] < 3:
        raise ValueError("Need at least 3 subjects for clustering.")

    if linkage_method == "ward" and metric != "euclidean":
        warnings.warn("Ward requires Euclidean distance. Switching metric to euclidean.")
        metric = "euclidean"

    d = pdist(mat_scaled.values, metric=metric)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    Z = linkage(d, method=linkage_method)

    order = leaves_list(Z)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters = pd.DataFrame({
        "subject_uid": mat_scaled.index,
        "shap_cluster": labels.astype(int),
    })
    return clusters, Z, order


# =============================================================================
# Risk variables
# =============================================================================

def select_risk_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    always = [
        "cohort", "session_key", "subject_uid", "shap_cluster",
        "apoe4_carriage_derived", "sex_derived",
        "cognitive_status_derived", "ad_status_derived",
    ]

    patterns = {
        "continuous": [
            "cBAG", "BAG",
            "age",
            "Global_Cognition", "Memory", "Executive", "cognition", "cognitive",
            "amyloid", "abeta", "centiloid", "tau", "ptau", "gfap", "nfl",
            "Hc_", "hippocamp", "entorhinal",
            "Global_Efficiency", "Local_Efficiency", "Clustering", "Path",
            "BMI", "systolic", "diastolic", "blood", "glucose", "diabetes",
            "Total_Brain", "FA", "MD", "RD", "AD",
        ],
        "categorical": [
            "APOE", "genotype", "sex", "diagnosis", "DX", "status",
            "amyloid_status", "tau_status", "positivity",
        ],
    }

    continuous = []
    categorical = []

    for c in df.columns:
        cl = c.lower()

        if c in always:
            continue

        if any(p.lower() in cl for p in patterns["continuous"]):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 20 and s.nunique(dropna=True) > 3:
                continuous.append(c)

        if any(p.lower() in cl for p in patterns["categorical"]):
            if df[c].nunique(dropna=True) >= 2 and df[c].nunique(dropna=True) <= 12:
                categorical.append(c)

    # Add derived categorical variables first.
    categorical = [
        "apoe4_carriage_derived",
        "sex_derived",
        "cognitive_status_derived",
        "ad_status_derived",
    ] + categorical

    continuous = list(dict.fromkeys(continuous))
    categorical = list(dict.fromkeys([c for c in categorical if c in df.columns]))

    return continuous, categorical


def kruskal_by_cluster(df: pd.DataFrame, variables: List[str], group_col: str = "shap_cluster") -> pd.DataFrame:
    rows = []

    for v in variables:
        groups = []
        ns = {}
        for cl, g in df.groupby(group_col):
            x = pd.to_numeric(g[v], errors="coerce").dropna()
            ns[str(cl)] = len(x)
            if len(x) >= 5:
                groups.append(x.values)

        if len(groups) < 2:
            continue

        try:
            stat, p = kruskal(*groups)
        except Exception:
            continue

        rows.append({
            "variable": v,
            "test": "kruskal",
            "stat": stat,
            "p": p,
            "n_total": int(pd.to_numeric(df[v], errors="coerce").notna().sum()),
            "cluster_ns": str(ns),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"])
    return out


def categorical_by_cluster(df: pd.DataFrame, variables: List[str], group_col: str = "shap_cluster") -> Tuple[pd.DataFrame, pd.DataFrame]:
    stat_rows = []
    count_frames = []

    for v in variables:
        tmp = df[[group_col, v]].copy()
        tmp[v] = tmp[v].astype("object")
        tmp = tmp.dropna()
        tmp = tmp[~tmp[v].astype(str).str.lower().isin(["nan", "none", "missing", "<na>", ""])]

        if tmp[group_col].nunique() < 2 or tmp[v].nunique() < 2:
            continue

        tab = pd.crosstab(tmp[group_col], tmp[v])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue

        try:
            chi2, p, dof, expected = chi2_contingency(tab)
        except Exception:
            continue

        cramers = np.sqrt((chi2 / tab.to_numpy().sum()) / max(min(tab.shape[0] - 1, tab.shape[1] - 1), 1))

        fisher_p = np.nan
        odds = np.nan
        if tab.shape == (2, 2):
            try:
                odds, fisher_p = fisher_exact(tab.to_numpy())
            except Exception:
                pass

        stat_rows.append({
            "variable": v,
            "test": "chi_square",
            "chi2": chi2,
            "p": p,
            "dof": dof,
            "cramers_v": cramers,
            "fisher_p_if_2x2": fisher_p,
            "odds_ratio_if_2x2": odds,
            "n_total": int(tab.to_numpy().sum()),
            "levels": ";".join(map(str, tab.columns)),
        })

        ct = tab.reset_index()
        ct.insert(0, "variable", v)
        count_frames.append(ct)

    stats = pd.DataFrame(stat_rows)
    if not stats.empty:
        stats["q_fdr"] = fdr_bh(stats["p"])

    counts = pd.concat(count_frames, ignore_index=True, sort=False) if count_frames else pd.DataFrame()
    return stats, counts


def summarize_cluster_profiles(df: pd.DataFrame, continuous: List[str], categorical: List[str]) -> pd.DataFrame:
    rows = []

    for cl, g in df.groupby("shap_cluster"):
        row = {
            "shap_cluster": cl,
            "n": len(g),
            "cohort_counts": dict(g["cohort"].value_counts()),
        }

        for v in continuous:
            x = pd.to_numeric(g[v], errors="coerce")
            if x.notna().sum() >= 3:
                row[f"{v}__mean"] = x.mean()
                row[f"{v}__sd"] = x.std()
                row[f"{v}__median"] = x.median()
                row[f"{v}__n"] = x.notna().sum()

        for v in categorical:
            if v in g.columns:
                vc = g[v].dropna().astype(str).value_counts(normalize=True)
                for level, prop in vc.items():
                    row[f"{v}__prop_{safe_name(level)}"] = prop

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmap(mat_scaled: pd.DataFrame, order: np.ndarray, clusters: pd.DataFrame, out_png: Path, title: str):
    ordered = mat_scaled.iloc[order]
    row_clusters = clusters.set_index("subject_uid").loc[ordered.index, "shap_cluster"]

    fig_h = max(7, min(24, 0.025 * ordered.shape[0] + 5))
    fig_w = max(8, min(24, 0.16 * ordered.shape[1] + 5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(ordered.values, aspect="auto", interpolation="nearest", vmin=-2.5, vmax=2.5)
    ax.set_title(title)
    ax.set_xlabel("Top SHAP features")
    ax.set_ylabel("Subjects ordered by SHAP-profile clustering")
    ax.set_xticks(np.arange(ordered.shape[1]))
    ax.set_xticklabels(ordered.columns, rotation=90, fontsize=5)
    ax.set_yticks([])

    # cluster separator lines
    prev = row_clusters.iloc[0]
    for i, cl in enumerate(row_clusters.iloc[1:], start=1):
        if cl != prev:
            ax.axhline(i - 0.5, linewidth=0.8)
            prev = cl

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Column-standardized signed SHAP")

    fig.tight_layout()
    fig.savefig(out_png, dpi=350)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_composition(df: pd.DataFrame, variable: str, out_png: Path, title: str):
    tmp = df[["shap_cluster", variable]].dropna().copy()
    tmp = tmp[~tmp[variable].astype(str).str.lower().isin(["nan", "none", "missing", ""])]
    if tmp.empty or tmp["shap_cluster"].nunique() < 2 or tmp[variable].nunique() < 2:
        return

    tab = pd.crosstab(tmp["shap_cluster"], tmp[variable], normalize="index")

    ax = tab.plot(kind="bar", stacked=True, figsize=(7, 4))
    ax.set_title(title)
    ax.set_xlabel("SHAP cluster")
    ax.set_ylabel("Proportion")
    ax.legend(title=variable, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=350, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_top_stats(cont_stats: pd.DataFrame, cat_stats: pd.DataFrame, out_png: Path, title: str):
    rows = []

    if not cont_stats.empty:
        tmp = cont_stats.copy()
        tmp["minus_log10_q"] = -np.log10(tmp["q_fdr"].clip(lower=1e-300))
        tmp["label"] = tmp["variable"]
        tmp["kind"] = "continuous"
        rows.append(tmp[["label", "minus_log10_q", "kind"]])

    if not cat_stats.empty:
        tmp = cat_stats.copy()
        tmp["minus_log10_q"] = -np.log10(tmp["q_fdr"].clip(lower=1e-300))
        tmp["label"] = tmp["variable"]
        tmp["kind"] = "categorical"
        rows.append(tmp[["label", "minus_log10_q", "kind"]])

    if not rows:
        return

    plot_df = pd.concat(rows, ignore_index=True).sort_values("minus_log10_q", ascending=False).head(25)
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(plot_df))))
    ax.barh(plot_df["label"], plot_df["minus_log10_q"])
    ax.axvline(-np.log10(0.05), linestyle="--", linewidth=0.8)
    ax.set_xlabel("-log10(FDR q)")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================

def run_one_kind(args, kind: str) -> List[str]:
    outdir = ensure_dir(
        args.shap_root / "cross_cohort_aggregated" / args.model /
        "subject_shap_clusters_enriched_risk_profiles"
    )

    prefix = f"Figure7_{args.model}_{kind}_SHAP_enriched_clusters_k{args.n_clusters}"
    outputs = []

    long_df, load_log = load_shap_long(
        shap_root=args.shap_root,
        model=args.model,
        kind=kind,
        value_col=args.value_col,
        cohorts=args.cohorts,
    )
    load_log_file = outdir / f"{prefix}_shap_load_log.csv"
    load_log.to_csv(load_log_file, index=False)
    outputs.append(str(load_log_file))

    if long_df.empty:
        print(f"[WARN] No SHAP data loaded for kind={kind}")
        return outputs

    long_file = outdir / f"{prefix}_subject_level_shap_long.csv"
    long_df.to_csv(long_file, index=False)
    outputs.append(str(long_file))

    ranked = rank_features(long_df, min_cohorts=args.min_cohorts)
    ranked_file = outdir / f"{prefix}_ranked_features.csv"
    ranked.to_csv(ranked_file, index=False)
    outputs.append(str(ranked_file))

    top_n = {
        "global": args.top_n_global,
        "node": args.top_n_node,
        "edge": args.top_n_edge,
    }[kind]

    mat_raw = build_matrix(long_df, ranked, top_n=top_n)
    mat_scaled = zscore_columns(mat_raw)

    raw_file = outdir / f"{prefix}_matrix_raw.csv"
    scaled_file = outdir / f"{prefix}_matrix_scaled.csv"
    mat_raw.to_csv(raw_file)
    mat_scaled.to_csv(scaled_file)
    outputs.extend([str(raw_file), str(scaled_file)])

    clusters, Z, order = cluster_subject_matrix(
        mat_scaled,
        n_clusters=args.n_clusters,
        linkage_method=args.linkage_method,
        metric=args.distance_metric,
    )

    # Subject metadata from SHAP filenames.
    shap_subject_meta = (
        long_df[["subject_uid", "cohort", "cohort_slug", "session_key", "subject_raw_from_shap"]]
        .drop_duplicates("subject_uid")
    )

    clusters = clusters.merge(shap_subject_meta, on="subject_uid", how="left")

    # Enriched metadata.
    meta, meta_log = load_enriched_metadata(args.model, args.cohorts)
    meta_log_file = outdir / f"{prefix}_enriched_metadata_load_log.csv"
    meta_log.to_csv(meta_log_file, index=False)
    outputs.append(str(meta_log_file))

    if not meta.empty:
        merged = clusters.merge(
            meta,
            on=["cohort", "session_key", "subject_uid"],
            how="left",
            suffixes=("", "_meta"),
        )
    else:
        merged = clusters.copy()

    merged_file = outdir / f"{prefix}_cluster_assignments_with_enriched_metadata.csv"
    merged.to_csv(merged_file, index=False)
    outputs.append(str(merged_file))

    cluster_file = outdir / f"{prefix}_cluster_assignments.csv"
    clusters.to_csv(cluster_file, index=False)
    outputs.append(str(cluster_file))

    # QC merge summary.
    qc = pd.DataFrame([{
        "kind": kind,
        "model": args.model,
        "n_shap_subjects": clusters["subject_uid"].nunique(),
        "n_enriched_metadata_rows": len(meta),
        "n_merged_rows": len(merged),
        "n_with_metadata_file": merged["metadata_file"].notna().sum() if "metadata_file" in merged.columns else 0,
        "metadata_match_rate": (
            merged["metadata_file"].notna().mean()
            if "metadata_file" in merged.columns and len(merged) else np.nan
        ),
    }])
    qc_file = outdir / f"{prefix}_merge_qc.csv"
    qc.to_csv(qc_file, index=False)
    outputs.append(str(qc_file))

    # Cluster ordered matrices.
    row_order = pd.DataFrame({
        "row_order": np.arange(len(order)),
        "subject_uid": mat_scaled.index[order],
    }).merge(clusters, on="subject_uid", how="left")

    row_order_file = outdir / f"{prefix}_row_order.csv"
    row_order.to_csv(row_order_file, index=False)
    outputs.append(str(row_order_file))

    mat_raw.iloc[order].to_csv(outdir / f"{prefix}_clustered_matrix_raw.csv")
    mat_scaled.iloc[order].to_csv(outdir / f"{prefix}_clustered_matrix_scaled.csv")

    # Risk-profile stats.
    continuous, categorical = select_risk_columns(merged)

    profile = summarize_cluster_profiles(merged, continuous, categorical)
    cont_stats = kruskal_by_cluster(merged, continuous)
    cat_stats, cat_counts = categorical_by_cluster(merged, categorical)

    profile_file = outdir / f"{prefix}_cluster_risk_profile_table.csv"
    cont_file = outdir / f"{prefix}_continuous_cluster_tests.csv"
    cat_file = outdir / f"{prefix}_categorical_cluster_tests.csv"
    counts_file = outdir / f"{prefix}_categorical_cluster_counts.csv"

    profile.to_csv(profile_file, index=False)
    cont_stats.to_csv(cont_file, index=False)
    cat_stats.to_csv(cat_file, index=False)
    cat_counts.to_csv(counts_file, index=False)

    outputs.extend([str(profile_file), str(cont_file), str(cat_file), str(counts_file)])

    # Figures.
    plot_heatmap(
        mat_scaled,
        order,
        clusters,
        outdir / f"{prefix}_heatmap.png",
        f"Cross-cohort {kind}-SHAP subject clusters ({args.model}, k={args.n_clusters})",
    )
    outputs.append(str(outdir / f"{prefix}_heatmap.png"))

    for v, ttl in [
        ("ad_status_derived", "AD-status composition by SHAP cluster"),
        ("cognitive_status_derived", "Cognitive-status composition by SHAP cluster"),
        ("apoe4_carriage_derived", "APOE4 composition by SHAP cluster"),
        ("sex_derived", "Sex composition by SHAP cluster"),
        ("cohort", "Cohort composition by SHAP cluster"),
    ]:
        if v in merged.columns:
            plot_composition(
                merged,
                v,
                outdir / f"{prefix}_{v}_composition.png",
                ttl,
            )
            outputs.append(str(outdir / f"{prefix}_{v}_composition.png"))

    plot_top_stats(
        cont_stats,
        cat_stats,
        outdir / f"{prefix}_top_cluster_enrichments.png",
        f"Top enriched phenotypes across {kind}-SHAP clusters",
    )
    outputs.append(str(outdir / f"{prefix}_top_cluster_enrichments.png"))

    manifest = pd.DataFrame({"output": outputs})
    manifest_file = outdir / f"{prefix}_manifest.csv"
    manifest.to_csv(manifest_file, index=False)
    outputs.append(str(manifest_file))

    print(f"[DONE] kind={kind}")
    print(f"  Output dir: {outdir}")
    print(f"  Merge match rate: {qc.loc[0, 'metadata_match_rate']}")
    print(f"  Continuous variables tested: {len(continuous)}")
    print(f"  Categorical variables tested: {len(categorical)}")

    return outputs


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--shap-root", type=Path, default=DEFAULT_SHAP_ROOT)
    p.add_argument("--model", default="imaging_only",
                   choices=[
                       "imaging_only",
                       "imaging_demographics",
                       "imaging_biomarkers",
                       "full",
                       "full_no_cardiovascular",
                   ])
    p.add_argument("--kinds", default="node,global", help="Comma list: node,global,edge")
    p.add_argument("--cohorts", default="ADNI,ADRC,HABS,AD_DECODE")

    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--top-n-global", type=int, default=35)
    p.add_argument("--top-n-node", type=int, default=50)
    p.add_argument("--top-n-edge", type=int, default=50)
    p.add_argument("--min-cohorts", type=int, default=2)

    p.add_argument("--value-col", default="SHAP_val", choices=["SHAP_val", "abs_SHAP"])
    p.add_argument("--linkage-method", default="ward")
    p.add_argument("--distance-metric", default="euclidean")

    return p.parse_args()


def main():
    args = parse_args()
    args.kinds = [x.strip().lower() for x in args.kinds.split(",") if x.strip()]
    args.cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]

    bad = [c for c in args.cohorts if c not in COHORT_CONFIG]
    if bad:
        raise SystemExit(f"Unknown cohorts: {bad}")

    badk = [k for k in args.kinds if k not in KIND_SUBDIR]
    if badk:
        raise SystemExit(f"Unknown SHAP kinds: {badk}")

    print("=" * 100)
    print("Figure7 SHAP subject clusters + enriched risk profiles")
    print("=" * 100)
    print("SHAP root:", args.shap_root)
    print("Model:", args.model)
    print("Kinds:", ",".join(args.kinds))
    print("Cohorts:", ",".join(args.cohorts))
    print("n_clusters:", args.n_clusters)
    print("=" * 100)

    all_outputs = []
    for kind in args.kinds:
        all_outputs.extend(run_one_kind(args, kind))

    outdir = ensure_dir(
        args.shap_root / "cross_cohort_aggregated" / args.model /
        "subject_shap_clusters_enriched_risk_profiles"
    )
    pd.DataFrame({"output": all_outputs}).to_csv(
        outdir / f"Figure7_{args.model}_all_enriched_cluster_outputs_manifest.csv",
        index=False,
    )

    print("\nAll done.")
    print("Primary output dir:")
    print(outdir)


if __name__ == "__main__":
    main()

