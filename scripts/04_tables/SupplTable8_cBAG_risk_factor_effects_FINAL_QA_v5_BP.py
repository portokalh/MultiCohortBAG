#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SupplTable8_cBAG_risk_factor_effects.py

Targeted cross-sectional risk-factor association analysis for cBAG.

Purpose
-------
Quantify whether cBAG is associated with established demographic, genetic,
vascular, and metabolic risk factors.

This is intended to produce:
  Supplementary Table S8. Associations between cBAG and demographic, genetic,
  and metabolic risk factors.

Models
------
Binary targets:
    target ~ z(cBAG) + z(age) + sex when appropriate

    Effect size:
      OR per 1 SD higher cBAG
      95% CI
      p-value
      AUC for the model with cBAG + covariates
      AUC for covariates only
      delta AUC

Continuous targets:
    z(target) ~ z(cBAG) + z(age) + sex when appropriate

    Effect size:
      standardized beta for cBAG
      95% CI
      partial r
      p-value

FDR
---
BH-FDR within cohort × risk_factor_family × model_type.

Inputs
------
- cBAG-enriched validation files, one per cohort.
- Optional harmonized metadata directory to add missing risk-factor columns.
  Merge uses Table1-style session keys.

Recommended run
---------------
cd /mnt/newStor/paros/paros_WORK/ines/code/BAG_Stability052627

ADNI_CBAG=/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal/ablation_imaging_only/validation_figures_full_cohort/subject_level_validation_input_enriched_for_Figure4.csv
ADRC_CBAG=/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal/ablation_imaging_only/validation_figures_full_cohort/subject_level_validation_input_enriched_for_Figure4.csv
HABS_CBAG=/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal/ablation_imaging_only/validation_figures_full_cohort/subject_level_validation_input_enriched_for_Figure4.csv
ADDECODE_CBAG=/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionAD_DECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal/ablation_imaging_only/validation_figures_full_cohort/subject_level_validation_input_enriched_for_Figure4.csv

OUT=/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/SupplTable8_cBAG_risk_factor_effects

python SupplTable8_cBAG_risk_factor_effects.py \
  --metadata-dir /mnt/newStor/paros/paros_WORK/ines/data/harmonization/harmonized_metadata \
  --cohorts ADNI,ADRC,HABS,AD_DECODE \
  --cbag-column cBAG_raw_clean \
  --cbag-files ADNI=$ADNI_CBAG,ADRC=$ADRC_CBAG,HABS=$HABS_CBAG,AD_DECODE=$ADDECODE_CBAG \
  --min-n 30 \
  --min-binary-class-n 10 \
  --supplement-number S8 \
  --outdir "$OUT"
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None


WORK = Path("/mnt/newStor/paros/paros_WORK")
DEFAULT_METADATA_DIR = WORK / "ines/data/harmonization/harmonized_metadata"
DEFAULT_OUTDIR = (
    WORK
    / "ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "SupplTable8_cBAG_risk_factor_effects"
)

MISSING_SENTINELS = {
    -777777, -77777, -7777, -777,
    -888888, -88888, -8888, -888,
    -999999, -99999, -9999, -999,
    777777, 77777, 7777, 777,
    888888, 88888, 8888, 888,
    999999, 99999, 9999, 999,
}

EXCLUDE_COL_RE = re.compile(
    r"subject|session|visit|date|file|path|scanner|site|cohort|fold|split|"
    r"target|source|unnamed|connectome|dwi_key|graph_id|matrix|csv",
    re.I,
)


# =============================================================================
# Generic helpers
# =============================================================================

def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


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
        if normalize_name(c) in norm:
            return norm[normalize_name(c)]
    return None


def clean_key_component(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})
    )


def clean_numeric(s: pd.Series, mask_extreme: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.mask(x.isin(MISSING_SENTINELS))
    if mask_extreme:
        x = x.mask(x <= -100000)
        x = x.mask(x >= 1000000)
    return x.replace([np.inf, -np.inf], np.nan)


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean(skipna=True)) / sd


def fdr_bh(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    if ok.sum() == 0:
        return q
    if multipletests is not None:
        q.loc[ok] = multipletests(p.loc[ok].values, method="fdr_bh")[1]
        return q
    vals = p.loc[ok].values
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(vals)
    qrank = ranked * n / (np.arange(n) + 1)
    qrank = np.minimum.accumulate(qrank[::-1])[::-1]
    qrank = np.clip(qrank, 0, 1)
    back = np.empty_like(qrank)
    back[order] = qrank
    q.loc[ok] = back
    return q


def parse_cbag_files_arg(s: str) -> Dict[str, Path]:
    out = {}
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad --cbag-files entry: {part}. Use COHORT=/path/file.csv")
        cohort, path = part.split("=", 1)
        out[cohort.strip()] = Path(path.strip())
    return out


# =============================================================================
# Key construction and merge
# =============================================================================

def normalize_adni_subject(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.search(r"R(\d+)", s, flags=re.I)
    if m:
        return f"R{int(m.group(1)):04d}"
    if re.fullmatch(r"\d+(?:\.0)?", s):
        return f"R{int(float(s)):04d}"
    return np.nan


def normalize_adni_visit(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    mapping = {
        "sc": "y0", "bl": "y0", "baseline": "y0", "init": "y0",
        "m00": "y0", "m0": "y0", "0": "y0", "0.0": "y0", "y0": "y0",
        "m06": "y0.5", "m6": "y0.5", "6": "y0.5", "6.0": "y0.5", "y0.5": "y0.5",
        "m12": "y1", "12": "y1", "12.0": "y1", "y1": "y1",
        "m24": "y2", "24": "y2", "24.0": "y2", "y2": "y2",
        "m36": "y3", "36": "y3", "36.0": "y3", "y3": "y3",
        "m48": "y4", "48": "y4", "48.0": "y4", "y4": "y4",
        "m60": "y5", "60": "y5", "60.0": "y5", "y5": "y5",
    }
    return mapping.get(s, np.nan)


def build_session_keys(df: pd.DataFrame, cohort: str) -> Tuple[pd.DataFrame, str]:
    df = df.copy()

    if cohort == "ADNI":
        sc = first_existing(df, ["DWI_subject_key"])
        vc = first_existing(df, ["DWI_key", "DWI"])
        if sc is not None and vc is not None:
            df["_subject_key"] = clean_key_component(df[sc])
            df["_session_key"] = clean_key_component(df[vc])
            return df, f"ADNI:{sc}+{vc}"

        raw_s = first_existing(df, ["Subject", "RID", "PTID"])
        raw_v = first_existing(df, ["Visit", "VISCODE", "VISCODE2"])
        if raw_s is not None and raw_v is not None:
            subj = df[raw_s].map(normalize_adni_subject)
            visit = df[raw_v].map(normalize_adni_visit)
            df["_subject_key"] = subj
            df["_session_key"] = (subj.astype(object) + "_" + visit.astype(object)).where(subj.notna() & visit.notna(), np.nan)
            return df, f"ADNI_FALLBACK:{raw_s}+{raw_v}"

    if cohort == "HABS":
        sc = first_existing(df, ["DWI_subject", "Subject", "Med_ID", "subject_id"])
        vc = first_existing(
            df,
            [
                "DWI", "runno", "DWI_key",
                "CONNECTOME_KEY_USED_FOR_INTERSECTION", "CONNECTOME_KEY_CLEAN",
                "connectome_key", "connectome_full_key", "graph_id",
            ],
        )
        df["_subject_key"] = clean_key_component(df[sc]) if sc is not None else np.nan
        df["_session_key"] = clean_key_component(df[vc]) if vc is not None else np.nan
        return df, f"HABS:{sc}+{vc or 'NO_SESSION_KEY'}"

    # Generic rule for ADRC / AD_DECODE etc.
    sc = first_existing(df, ["DWI_subject_key", "DWI_subject", "Subject", "subject_id", "ID", "RID", "PTID"])
    vc = first_existing(df, ["DWI_key", "DWI", "Visit", "VISCODE", "VISCODE2", "session", "visit"])
    if sc is not None and vc is not None:
        df["_subject_key"] = clean_key_component(df[sc])
        df["_session_key"] = clean_key_component(df[vc])
        return df, f"{cohort}:{sc}+{vc}"
    if sc is not None:
        df["_subject_key"] = clean_key_component(df[sc])
        df["_session_key"] = df["_subject_key"].copy()
        return df, f"{cohort}:{sc}+subject"
    df["_subject_key"] = pd.Series([f"{cohort}_{i}" for i in range(len(df))], index=df.index)
    df["_session_key"] = df["_subject_key"].copy()
    return df, f"{cohort}:row_index"


def load_and_merge_cohort(metadata_dir: Path, cohort: str, cbag_file: Path) -> Tuple[pd.DataFrame, dict]:
    cbag = read_table(cbag_file)
    cbag, cbag_rule = build_session_keys(cbag, cohort)

    meta_file = metadata_dir / f"{cohort}_harmonized_metadata.csv"
    if meta_file.exists():
        meta = read_table(meta_file)
        meta, meta_rule = build_session_keys(meta, cohort)

        # Keep metadata columns absent from cbag to avoid overwriting useful enriched columns.
        meta_cols = ["_subject_key", "_session_key"] + [c for c in meta.columns if c not in cbag.columns and not str(c).startswith("_")]
        meta_small = meta[meta_cols].drop_duplicates(["_subject_key", "_session_key"], keep="first")
        df = cbag.merge(meta_small, on=["_subject_key", "_session_key"], how="left", validate="m:1")
    else:
        meta = pd.DataFrame()
        meta_rule = "metadata_missing"
        df = cbag.copy()

    df["cohort"] = cohort

    summary = {
        "cohort": cohort,
        "cbag_file": str(cbag_file),
        "metadata_file": str(meta_file) if meta_file.exists() else "",
        "cbag_key_rule": cbag_rule,
        "metadata_key_rule": meta_rule,
        "cbag_rows": len(cbag),
        "cbag_subjects": int(cbag["_subject_key"].nunique(dropna=True)),
        "cbag_sessions": int(cbag["_session_key"].nunique(dropna=True)),
        "metadata_rows": len(meta) if not meta.empty else 0,
        "merged_rows": len(df),
        "merged_subjects": int(df["_subject_key"].nunique(dropna=True)),
    }
    return df, summary



# =============================================================================
# Derived cognitive status endpoints and QA helpers
# =============================================================================

def derive_cognitive_status_ordinal(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Derive:
      1) ordinal cognitive status: 0 = CN/NORM/control, 1 = MCI, 2 = AD/dementia
      2) binary cognitive impairment: CN vs MCI/AD

    The function is deliberately permissive in column matching because cohort
    files use different diagnosis/status labels.
    """
    df = df.copy()

    candidates = [
        "dx_sum", "DX_sum", "DX_SUM", "meta__dx_sum", "meta__DX_sum", "meta__DX_SUM",
        "DX_Label_harmonized", "meta__DX_Label_harmonized",
        "_cognitive_status", "cognitive_status", "Cognitive_status", "cognitive_status_label",
        "cog_status", "CogStatus", "clinical_status", "Clinical_status",
        "diagnosis", "Diagnosis", "diagnosis_label",
        "DX", "dx", "DX_bl", "DX_bl2", "DXCHANGE", "DXCURREN",
        "Group", "group", "Dx", "final_dx", "diagnostic_group",
    ]
    col = first_existing(df, candidates)

    # If no single diagnosis label exists, try one-hot Table1-style columns.
    onehot_norm = first_existing(df, ["NORMCOG_01", "meta__NORMCOG_01"],) if True else None
    onehot_mci = first_existing(df, ["MCI_01", "meta__MCI_01"],) if True else None
    onehot_ad = first_existing(df, ["AD_01", "DEMENTIA_01", "meta__AD_01", "meta__DEMENTIA_01"],) if True else None

    qa = {
        "source_column": col or "",
        "n_source_nonmissing": 0,
        "n_ordinal_nonmissing": 0,
        "n_CN_0": 0,
        "n_MCI_1": 0,
        "n_AD_2": 0,
        "n_binary_nonmissing": 0,
        "n_binary_CN_0": 0,
        "n_binary_impaired_1": 0,
        "created": False,
    }

    out = pd.Series(np.nan, index=df.index, dtype=float)

    if col is not None:
        raw = df[col]
        qa["n_source_nonmissing"] = int(raw.notna().sum())

        x = raw.astype(str).str.strip().str.lower()

        # Explicit string labels.
        normal_pat = r"(?:^|[^a-z])cn(?:[^a-z]|$)|normal|norm|control|cognitively normal|unimpaired|healthy"
        mci_pat = r"(?:^|[^a-z])mci(?:[^a-z]|$)|mild cognitive impairment|mild_impair"
        ad_pat = r"(?:^|[^a-z])ad(?:[^a-z]|$)|dementia|alz|alzheimer"

        out.loc[x.str.contains(normal_pat, regex=True, na=False)] = 0
        out.loc[x.str.contains(mci_pat, regex=True, na=False)] = 1
        out.loc[x.str.contains(ad_pat, regex=True, na=False)] = 2

        # Numeric fallback if already coded as 0/1/2. Avoid DXCHANGE-style values >2.
        xn = pd.to_numeric(raw, errors="coerce")
        valid_numeric = xn.isin([0, 1, 2])
        out.loc[out.isna() & valid_numeric] = xn.loc[out.isna() & valid_numeric]

        # Common labels.
        out.loc[x.eq("norm")] = 0
        out.loc[x.eq("mci")] = 1
        out.loc[x.eq("ad")] = 2

    # One-hot fallback/augmentation.
    # Priority: AD/dementia > MCI > normal.
    if onehot_norm is not None:
        v = pd.to_numeric(df[onehot_norm], errors="coerce")
        out.loc[out.isna() & v.eq(1)] = 0
        if not qa["source_column"]:
            qa["source_column"] = onehot_norm
    if onehot_mci is not None:
        v = pd.to_numeric(df[onehot_mci], errors="coerce")
        out.loc[v.eq(1)] = 1
        if "source_column" in qa and onehot_mci not in qa["source_column"]:
            qa["source_column"] = (qa["source_column"] + ";" + onehot_mci).strip(";")
    if onehot_ad is not None:
        v = pd.to_numeric(df[onehot_ad], errors="coerce")
        out.loc[v.eq(1)] = 2
        if "source_column" in qa and onehot_ad not in qa["source_column"]:
            qa["source_column"] = (qa["source_column"] + ";" + onehot_ad).strip(";")

    if out.notna().sum() >= 10 and out.nunique(dropna=True) >= 2:
        ord_col = "derived_cognitive_status_ordinal_0CN_1MCI_2AD"
        bin_col = "derived_cognitive_impairment_binary_CN_vs_MCIAD"
        df[ord_col] = out
        df[bin_col] = (out > 0).astype(float).where(out.notna(), np.nan)

        qa["n_ordinal_nonmissing"] = int(df[ord_col].notna().sum())
        qa["n_CN_0"] = int((df[ord_col] == 0).sum())
        qa["n_MCI_1"] = int((df[ord_col] == 1).sum())
        qa["n_AD_2"] = int((df[ord_col] == 2).sum())
        qa["n_binary_nonmissing"] = int(df[bin_col].notna().sum())
        qa["n_binary_CN_0"] = int((df[bin_col] == 0).sum())
        qa["n_binary_impaired_1"] = int((df[bin_col] == 1).sum())
        qa["created"] = True

    return df, qa



def _clean_bp_numeric(s: pd.Series, kind: str) -> pd.Series:
    """
    Physiologic cleaner for BP/pulse variables.
    kind: sbp, dbp, pulse_rate, pulse_pressure, map
    """
    x = clean_numeric(s, mask_extreme=False)
    if kind == "sbp":
        return x.mask((x < 70) | (x > 260))
    if kind == "dbp":
        return x.mask((x < 30) | (x > 160))
    if kind == "pulse_rate":
        return x.mask((x < 30) | (x > 220))
    if kind == "pulse_pressure":
        return x.mask((x < 15) | (x > 140))
    if kind == "map":
        return x.mask((x < 40) | (x > 180))
    return x


def _find_best_bp_col(df: pd.DataFrame, candidates: Sequence[str], kind: str) -> Optional[str]:
    """
    Pick the highest-priority BP column with usable physiologic values.
    Preference is explicit cleaned/average HABS columns before generic SBP/DBP.
    """
    for cand in candidates:
        col = first_existing(df, [cand])
        if col is None:
            continue
        x = _clean_bp_numeric(df[col], kind=kind)
        if x.notna().sum() >= 30 and x.nunique(dropna=True) >= 5:
            return col
    return None


def derive_primary_bp_features(df: pd.DataFrame, cohort: str) -> Tuple[pd.DataFrame, dict]:
    """
    Create primary BP variables using best-quality columns and physiologic cleaning.

    HABS contains generic SBP/DBP/Pulse columns with extreme negative coded values.
    Prefer bp_sys/bp_dia or OM_BP averaged measurements, then derive pulse pressure
    and MAP from the matched primary SBP/DBP.
    """
    df = df.copy()

    sbp_candidates = [
        "bp_sys_raw_clean", "bp_sys", "clinical__bp_sys",
        "OM_BP1_SYS", "OM_BP2_SYS",
        "SBP_raw_clean", "SBP",
        "VSBPSYS_raw_clean", "VSBPSYS", "Systolic_raw_clean", "Systolic",
    ]
    dbp_candidates = [
        "bp_dia_raw_clean", "bp_dia", "clinical__bp_dia",
        "OM_BP1_DIA", "OM_BP2_DIA",
        "DBP_raw_clean", "DBP",
        "VSBPDIA_raw_clean", "VSBPDIA", "Diastolic_raw_clean", "Diastolic",
    ]
    pulse_candidates = [
        "pulse_raw_clean", "pulse", "OM_Pulse1_raw_clean", "OM_Pulse1",
        "OM_Pulse2_raw_clean", "OM_Pulse2", "Pulse_raw_clean", "Pulse",
    ]

    sbp_col = _find_best_bp_col(df, sbp_candidates, "sbp")
    dbp_col = _find_best_bp_col(df, dbp_candidates, "dbp")
    pulse_col = _find_best_bp_col(df, pulse_candidates, "pulse_rate")

    qa = {
        "cohort": cohort,
        "primary_sbp_col": sbp_col or "",
        "primary_dbp_col": dbp_col or "",
        "primary_pulse_col": pulse_col or "",
        "n_primary_sbp": 0,
        "n_primary_dbp": 0,
        "n_primary_pulse": 0,
        "n_derived_pulse_pressure": 0,
        "n_derived_map": 0,
    }

    if sbp_col is not None:
        df["derived_primary_SBP_phys_clean"] = _clean_bp_numeric(df[sbp_col], "sbp")
        qa["n_primary_sbp"] = int(df["derived_primary_SBP_phys_clean"].notna().sum())

    if dbp_col is not None:
        df["derived_primary_DBP_phys_clean"] = _clean_bp_numeric(df[dbp_col], "dbp")
        qa["n_primary_dbp"] = int(df["derived_primary_DBP_phys_clean"].notna().sum())

    if pulse_col is not None:
        df["derived_primary_pulse_rate_phys_clean"] = _clean_bp_numeric(df[pulse_col], "pulse_rate")
        qa["n_primary_pulse"] = int(df["derived_primary_pulse_rate_phys_clean"].notna().sum())

    if "derived_primary_SBP_phys_clean" in df.columns and "derived_primary_DBP_phys_clean" in df.columns:
        pp = df["derived_primary_SBP_phys_clean"] - df["derived_primary_DBP_phys_clean"]
        mapv = df["derived_primary_DBP_phys_clean"] + pp / 3.0
        df["derived_primary_pulse_pressure_phys_clean"] = _clean_bp_numeric(pp, "pulse_pressure")
        df["derived_primary_MAP_phys_clean"] = _clean_bp_numeric(mapv, "map")
        qa["n_derived_pulse_pressure"] = int(df["derived_primary_pulse_pressure_phys_clean"].notna().sum())
        qa["n_derived_map"] = int(df["derived_primary_MAP_phys_clean"].notna().sum())

    return df, qa


def qa_column_snapshot(df: pd.DataFrame, cbag_col: str, age_col: Optional[str], sex_col: Optional[str]) -> pd.DataFrame:
    """
    Lightweight QA inventory of key risk-factor-like columns.
    """
    rows = []
    for c in df.columns:
        info = classify_risk_factor_col(c)
        if info is None and c not in [cbag_col, age_col, sex_col]:
            continue

        xnum = clean_numeric(df[c]) if c in df.columns else pd.Series(dtype=float)
        rows.append({
            "column": c,
            "is_cbag": c == cbag_col,
            "is_age": c == age_col,
            "is_sex": c == sex_col,
            "classified_family": info.get("risk_factor_family", "") if info else "",
            "classified_label": info.get("semantic_label", "") if info else "",
            "classified_model_type": info.get("model_type", "") if info else "",
            "n_nonmissing_raw": int(df[c].notna().sum()),
            "n_numeric_nonmissing": int(xnum.notna().sum()),
            "n_unique_raw": int(df[c].nunique(dropna=True)),
            "numeric_mean": float(xnum.mean(skipna=True)) if xnum.notna().sum() else np.nan,
            "numeric_sd": float(xnum.std(skipna=True)) if xnum.notna().sum() else np.nan,
            "example_values": "; ".join(map(str, list(pd.Series(df[c].dropna().unique()).head(8)))),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Target detection
# =============================================================================

def infer_age_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, ["age_for_table1", "age", "Age", "AGE", "VISIT_AGE", "DWI_visit_age", "DWI_age"])


def infer_sex_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, ["sex_label", "sex", "Sex", "SEX", "PTGENDER", "SUBJECT_SEX", "gender", "ID_Gender"])


def encode_sex(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[x.isin(["m", "male", "1", "1.0", "man"])] = 1.0
    out.loc[x.isin(["f", "female", "2", "2.0", "0", "woman"])] = 0.0
    if out.notna().sum() < max(5, len(s) * 0.1):
        codes, _ = pd.factorize(s.astype(str), sort=True)
        out = pd.Series(codes, index=s.index).replace(-1, np.nan).astype(float)
    return out


def encode_binary_general(s: pd.Series) -> pd.Series:
    xnum = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if xnum.notna().sum() >= 10:
        vals = set(pd.Series(xnum.dropna().unique()).round(6).tolist())
        if vals.issubset({0, 1, 0.0, 1.0}):
            return xnum.astype(float)
        if vals.issubset({0, 1, 2, 0.0, 1.0, 2.0}):
            return (xnum > 0).astype(float).where(xnum.notna(), np.nan)

    x = s.astype(str).str.strip().str.lower()
    pos = ["1", "1.0", "yes", "y", "true", "carrier", "positive", "pos", "present", "impaired", "mci", "ad", "case"]
    neg = ["0", "0.0", "no", "n", "false", "noncarrier", "non-carrier", "negative", "neg", "absent", "normal", "cn", "control"]
    out.loc[x.isin(pos)] = 1.0
    out.loc[x.isin(neg)] = 0.0
    return out


def encode_apoe4(s: pd.Series) -> pd.Series:
    out = encode_binary_general(s)
    xs = s.astype(str).str.strip().str.lower()
    has4 = xs.str.contains(r"(?:^|[^0-9])4(?:[^0-9]|$)|34|43|44|2/4|3/4|4/4|e4", regex=True, na=False)
    genotype_like = xs.str.contains(r"2|3|4|e", regex=True, na=False)
    out.loc[out.isna() & genotype_like & has4] = 1.0
    out.loc[out.isna() & genotype_like & ~has4] = 0.0
    return out



def is_false_bmi_column(col: str) -> bool:
    c = normalize_name(col)
    return bool(re.search(
        r"education|educ|age|sex|gender|race|ethnic|diagnosis|dx|visit|"
        r"cardiovascularrisk|cvrscore|hypertension|heart|stroke|diabetes|glucose|"
        r"weight|vsweight|waist|height",
        c,
    ))


def is_true_bmi_column(col: str) -> bool:
    c = normalize_name(col)
    if is_false_bmi_column(c):
        return False
    return bool(re.search(
        r"(^|_)bmi($|_)|body_mass_index|bodymassindex|om_bmi|phc_bmi",
        c,
    ))

def classify_risk_factor_col(col: str) -> Optional[dict]:
    raw = str(col)
    c = normalize_name(raw)

    if EXCLUDE_COL_RE.search(c):
        return None

    # Exclude imaging/cognition/fluid biomarkers that are not risk factors.
    if re.search(r"hippocamp|brain|global_eff|local_eff|clustering|graph|fa$|rd$|ad$|adc|ptau|tau|abeta|gfap|nfl|moca|mmse|adas|cdr|trail", c):
        return None

    # Binary / categorical risk factors.
    if re.search(r"apoe.*4|apoe4|e4_carrier|apoe4_carriage|apoe4_carrier", c):
        return {
            "risk_factor_family": "genetic",
            "semantic_label": "APOE4 carriage",
            "model_type": "binary",
            "encoder": "apoe4",
        }

    if re.fullmatch(r"sex|gender|ptgender|subject_sex|sex_label|id_gender", c):
        return {
            "risk_factor_family": "demographic",
            "semantic_label": "Sex",
            "model_type": "binary",
            "encoder": "sex",
        }

    if c in {"dx_sum", "dxsum"}:
        return {
            "risk_factor_family": "clinical_status",
            "semantic_label": "Cognitive status ordinal / dx_sum",
            "model_type": "continuous",
            "encoder": "numeric",
        }

    if re.search(r"cognitive_status|diagnosis|dx_status|dx_group|clinical_status", c):
        return {
            "risk_factor_family": "clinical_status",
            "semantic_label": "Cognitive impairment",
            "model_type": "binary",
            "encoder": "impairment",
        }

    if re.search(r"type_?2|t2d|diabetes|diabetic|dm2|imh_diabetes|phc_diabetes", c):
        if re.search(r"age|duration|year", c):
            return None
        return {
            "risk_factor_family": "metabolic_diabetes",
            "semantic_label": "Type 2 diabetes / diabetes status",
            "model_type": "binary",
            "encoder": "binary",
        }

    # Continuous risk markers.
    # True observed BMI only. Do not allow variables from a vrf_bmi namespace
    # such as PHC_Education or Age_CardiovascularRisk to be treated as BMI.
    if is_true_bmi_column(raw):
        return {
            "risk_factor_family": "metabolic_adiposity",
            "semantic_label": "BMI",
            "model_type": "continuous",
            "encoder": "numeric",
        }

    continuous_rules = [
        ("metabolic_diabetes", "HbA1c", r"hba1c|hemoglobin_a1c|a1c"),
        ("metabolic_diabetes", "Glucose", r"glucose|glu($|_)|fasting_glucose"),
        ("metabolic_diabetes", "Insulin", r"insulin|fasting_insulin"),
        ("metabolic_diabetes", "HOMA-IR", r"homa|homa_ir|homa2"),
        ("lipids", "Cholesterol/HDL ratio", r"cholhdl|chol_hdl|cholesterol_hdl|chol.?hdl.?ratio"),
        ("lipids", "Triglycerides", r"triglyceride|triglycerides|(^|_)tg($|_)"),
        ("lipids", "HDL cholesterol", r"(^|_)hdl($|_)|hdlchol|hdl_chol"),
        ("lipids", "LDL cholesterol", r"(^|_)ldl($|_)|ldlchol|ldl_chol"),
        ("lipids", "Total cholesterol", r"total.?chol|choltotal|cholesterol|(^|_)chol($|_)"),
        ("vascular", "Systolic BP", r"systolic|sbp"),
        ("vascular", "Diastolic BP", r"diastolic|dbp"),
        ("vascular", "Pulse pressure", r"pulse_pressure|ppressure|pulsepress"),
        ("vascular", "Mean arterial pressure", r"mean_arterial|map_bp|(^|_)map($|_)"),
    ]

    for fam, label, pat in continuous_rules:
        if re.search(pat, c):
            # avoid binary high cholesterol age etc unless no better target
            if re.search(r"age|onset|year", c):
                return None
            return {
                "risk_factor_family": fam,
                "semantic_label": label,
                "model_type": "continuous",
                "encoder": "numeric",
            }

    return None


def build_inventory(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    seen = set()

    # Force derived cognitive status endpoints if present.
    ord_col = "derived_cognitive_status_ordinal_0CN_1MCI_2AD"
    bin_col = "derived_cognitive_impairment_binary_CN_vs_MCIAD"

    # Force primary BP endpoints if present. These are preferred over generic
    # SBP/DBP/Pulse columns because they use physiologic cleaning and cohort-
    # specific primary-column selection.
    forced_bp = [
        ("derived_primary_SBP_phys_clean", "vascular", "Systolic BP"),
        ("derived_primary_DBP_phys_clean", "vascular", "Diastolic BP"),
        ("derived_primary_pulse_pressure_phys_clean", "vascular", "Pulse pressure"),
        ("derived_primary_MAP_phys_clean", "vascular", "Mean arterial pressure"),
        ("derived_primary_pulse_rate_phys_clean", "vascular", "Pulse rate"),
    ]
    for bp_col, fam, label in forced_bp:
        if bp_col in df.columns:
            x = clean_numeric(df[bp_col], mask_extreme=False)
            if int(x.notna().sum()) >= min_n and int(x.nunique(dropna=True)) >= 4:
                rows.append({
                    "column": bp_col,
                    "risk_factor_family": fam,
                    "semantic_label": label,
                    "model_type": "continuous",
                    "encoder": "numeric",
                    "n_nonmissing": int(x.notna().sum()),
                })
                seen.add(bp_col)

    if ord_col in df.columns:
        x = clean_numeric(df[ord_col], mask_extreme=False)
        if int(x.notna().sum()) >= min_n and int(x.nunique(dropna=True)) >= 2:
            rows.append({
                "column": ord_col,
                "risk_factor_family": "clinical_status",
                "semantic_label": "Cognitive status ordinal, 0=CN, 1=MCI, 2=AD",
                "model_type": "continuous",
                "encoder": "numeric",
                "n_nonmissing": int(x.notna().sum()),
            })
            seen.add(ord_col)

    if bin_col in df.columns:
        x = encode_binary_general(df[bin_col])
        if int(x.notna().sum()) >= min_n and min(int((x == 0).sum()), int((x == 1).sum())) >= 5:
            rows.append({
                "column": bin_col,
                "risk_factor_family": "clinical_status",
                "semantic_label": "Cognitive impairment, CN vs MCI/AD",
                "model_type": "binary",
                "encoder": "binary",
                "n_nonmissing": int(x.notna().sum()),
            })
            seen.add(bin_col)

    # Prioritized exact/fuzzy columns first.
    for col in df.columns:
        info = classify_risk_factor_col(col)
        if info is None:
            continue
        if col in seen:
            continue
        seen.add(col)

        if info["model_type"] == "continuous":
            x = clean_numeric(df[col])
            n = int(x.notna().sum())
            nunique = int(x.nunique(dropna=True))
            if n < min_n or nunique < 4:
                continue
        else:
            if info["encoder"] == "sex":
                x = encode_sex(df[col])
            elif info["encoder"] == "apoe4":
                x = encode_apoe4(df[col])
            elif info["encoder"] == "impairment":
                x = encode_impairment(df[col])
            else:
                x = encode_binary_general(df[col])
            n = int(x.notna().sum())
            n1 = int((x == 1).sum())
            n0 = int((x == 0).sum())
            if n < min_n or min(n0, n1) < 5:
                continue

        rows.append({
            "column": col,
            **info,
            "n_nonmissing": n,
        })

    inv = pd.DataFrame(rows)

    # Remove redundant aliases by choosing best column per semantic label.
    if inv.empty:
        return inv
    inv["_priority"] = inv["column"].map(column_priority)
    inv = inv.sort_values(["risk_factor_family", "semantic_label", "_priority", "n_nonmissing"], ascending=[True, True, True, False])
    # Keep top 2 per semantic label to allow BMI/glucose alternatives, but not explode duplicates.
    inv = inv.groupby(["risk_factor_family", "semantic_label", "model_type"], group_keys=False).head(2).copy()
    inv = inv.drop(columns="_priority")
    return inv.reset_index(drop=True)


def column_priority(col: str) -> int:
    c = normalize_name(col)
    priority_tokens = [
        "derived_primary_sbp_phys_clean",
        "derived_primary_dbp_phys_clean",
        "derived_primary_pulse_pressure_phys_clean",
        "derived_primary_map_phys_clean",
        "derived_primary_pulse_rate_phys_clean",
        "bp_sys_raw_clean", "bp_dia_raw_clean", "pulse_raw_clean",
        "bp_sys", "bp_dia", "pulse",
        "om_bp1_sys", "om_bp1_dia", "om_pulse1",
        "om_bp2_sys", "om_bp2_dia", "om_pulse2",
        "raw_clean", "clinical__", "om_bmi", "bw_", "apoe4_carrier", "apoe4_carriage",
        "bmi", "hba1c", "glucose", "insulin", "hdl", "ldl", "triglycerides",
    ]
    for i, tok in enumerate(priority_tokens):
        if tok in c:
            return i
    if "z_clean" in c:
        return 50
    return 20


def encode_impairment(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    x = s.astype(str).str.strip().str.lower()
    # Binary if explicit.
    out.loc[x.isin(["1", "1.0", "impaired", "mci", "ad", "dementia", "ci", "case"])] = 1.0
    out.loc[x.isin(["0", "0.0", "normal", "cn", "control", "cognitively normal", "norm"])] = 0.0
    # Numeric fallback: >0 as impaired, but only if values are small categories.
    xn = pd.to_numeric(s, errors="coerce")
    if out.notna().sum() < 10 and xn.notna().sum() >= 10:
        vals = set(pd.Series(xn.dropna().unique()).round(6).tolist())
        if vals.issubset({0, 1, 2, 3, 0.0, 1.0, 2.0, 3.0}):
            out = (xn > 0).astype(float).where(xn.notna(), np.nan)
    return out


# =============================================================================
# Models
# =============================================================================

def fit_ols(y: pd.Series, X: pd.DataFrame, term: str) -> dict:
    data = pd.concat([y.rename("y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n <= X.shape[1] + 3:
        return {"n_model": n, "fit_error": f"insufficient_complete_cases:n={n},p={X.shape[1]}"}

    yv = data["y"].astype(float).to_numpy()
    Xm = data[X.columns].astype(float).to_numpy()
    Xm = np.column_stack([np.ones(n), Xm])
    names = ["Intercept"] + list(X.columns)

    try:
        beta = np.linalg.lstsq(Xm, yv, rcond=None)[0]
        resid = yv - Xm @ beta
        rank = int(np.linalg.matrix_rank(Xm))
        df = n - rank
        rss = float(np.sum(resid ** 2))
        tss = float(np.sum((yv - np.mean(yv)) ** 2))
        sigma2 = rss / df
        cov = sigma2 * np.linalg.pinv(Xm.T @ Xm)
        se = np.sqrt(np.maximum(np.diag(cov), 0))

        idx = names.index(term)
        t = float(beta[idx] / se[idx])
        p = float(2 * stats.t.sf(abs(t), df=df))
        partial_r = float(np.sign(t) * math.sqrt((t * t) / (t * t + df)))
        return {
            "n_model": n,
            "beta": float(beta[idx]),
            "se": float(se[idx]),
            "ci_low": float(beta[idx] - 1.96 * se[idx]),
            "ci_high": float(beta[idx] + 1.96 * se[idx]),
            "t": t,
            "p": p,
            "partial_r": partial_r,
            "r2": float(1 - rss / tss) if tss > 0 else np.nan,
            "model_rank": rank,
            "model_df": df,
        }
    except Exception as e:
        return {"n_model": n, "fit_error": str(e)}


def fit_logistic_irls(y: pd.Series, X: pd.DataFrame, term: str, max_iter: int = 100) -> dict:
    """
    Minimal logistic regression with IRLS and pseudo-inverse.

    Returns OR for the term, Wald p, and AUC for full/covariate model.
    """
    data = pd.concat([y.rename("y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n <= X.shape[1] + 5:
        return {"n_model": n, "fit_error": f"insufficient_complete_cases:n={n},p={X.shape[1]}"}
    if data["y"].nunique() < 2:
        return {"n_model": n, "fit_error": "one_class_after_cleaning"}

    yv = data["y"].astype(float).to_numpy()
    Xm0 = data[X.columns].astype(float).to_numpy()
    Xm = np.column_stack([np.ones(n), Xm0])
    names = ["Intercept"] + list(X.columns)

    beta = np.zeros(Xm.shape[1], dtype=float)
    try:
        for _ in range(max_iter):
            eta = np.clip(Xm @ beta, -30, 30)
            mu = 1 / (1 + np.exp(-eta))
            W = np.clip(mu * (1 - mu), 1e-6, None)
            z = eta + (yv - mu) / W
            XTWX = Xm.T @ (Xm * W[:, None])
            XTWz = Xm.T @ (W * z)
            beta_new = np.linalg.pinv(XTWX) @ XTWz
            if np.max(np.abs(beta_new - beta)) < 1e-7:
                beta = beta_new
                break
            beta = beta_new

        eta = np.clip(Xm @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        cov = np.linalg.pinv(Xm.T @ (Xm * W[:, None]))
        se = np.sqrt(np.maximum(np.diag(cov), 0))

        idx = names.index(term)
        zval = float(beta[idx] / se[idx])
        p = float(2 * stats.norm.sf(abs(zval)))
        or_val = float(np.exp(beta[idx]))
        or_low = float(np.exp(beta[idx] - 1.96 * se[idx]))
        or_high = float(np.exp(beta[idx] + 1.96 * se[idx]))

        auc_full = np.nan
        auc_covariates = np.nan
        delta_auc = np.nan
        if roc_auc_score is not None:
            try:
                auc_full = float(roc_auc_score(yv, mu))
                # covariates-only model by dropping term.
                cov_cols = [c for c in X.columns if c != term]
                if cov_cols:
                    fit_cov = fit_logistic_pred(data["y"], data[cov_cols])
                    auc_covariates = fit_cov.get("auc", np.nan)
                    if pd.notna(auc_covariates):
                        delta_auc = auc_full - auc_covariates
                else:
                    auc_covariates = 0.5
                    delta_auc = auc_full - auc_covariates
            except Exception:
                pass

        return {
            "n_model": n,
            "beta_logit": float(beta[idx]),
            "se_logit": float(se[idx]),
            "z": zval,
            "p": p,
            "or_per_1sd_cBAG": or_val,
            "or_ci_low": or_low,
            "or_ci_high": or_high,
            "auc_full": auc_full,
            "auc_covariates": auc_covariates,
            "delta_auc": delta_auc,
            "model_rank": int(np.linalg.matrix_rank(Xm)),
        }
    except Exception as e:
        return {"n_model": n, "fit_error": str(e)}


def fit_logistic_pred(y: pd.Series, X: pd.DataFrame, max_iter: int = 100) -> dict:
    data = pd.concat([y.rename("y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n <= X.shape[1] + 5 or data["y"].nunique() < 2 or roc_auc_score is None:
        return {"auc": np.nan}

    yv = data["y"].astype(float).to_numpy()
    Xm = data[X.columns].astype(float).to_numpy()
    Xm = np.column_stack([np.ones(n), Xm])
    beta = np.zeros(Xm.shape[1], dtype=float)
    try:
        for _ in range(max_iter):
            eta = np.clip(Xm @ beta, -30, 30)
            mu = 1 / (1 + np.exp(-eta))
            W = np.clip(mu * (1 - mu), 1e-6, None)
            z = eta + (yv - mu) / W
            beta_new = np.linalg.pinv(Xm.T @ (Xm * W[:, None])) @ (Xm.T @ (W * z))
            if np.max(np.abs(beta_new - beta)) < 1e-7:
                beta = beta_new
                break
            beta = beta_new
        mu = 1 / (1 + np.exp(-np.clip(Xm @ beta, -30, 30)))
        return {"auc": float(roc_auc_score(yv, mu))}
    except Exception:
        return {"auc": np.nan}


def build_covariates(df: pd.DataFrame, target_col: str, sex_col: Optional[str], age_col: Optional[str], include_sex: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["cBAG_z"] = zscore(df["_cbag_clean"])

    if age_col is not None:
        age_z = zscore(clean_numeric(df[age_col], mask_extreme=False))
        if age_z.notna().sum() >= 20 and age_z.nunique(dropna=True) >= 4:
            X["age_z"] = age_z

    if include_sex and sex_col is not None and sex_col != target_col:
        sx = encode_sex(df[sex_col])
        if sx.notna().sum() >= 20 and sx.nunique(dropna=True) >= 2:
            X["sex_code"] = sx

    return X


def run_target_model(df: pd.DataFrame, inv_row: pd.Series, cbag_col: str, age_col: Optional[str], sex_col: Optional[str], min_n: int, min_binary_class_n: int) -> dict:
    col = inv_row["column"]
    model_type = inv_row["model_type"]
    encoder = inv_row["encoder"]

    base = {
        "cohort": df["cohort"].iloc[0],
        "risk_factor_family": inv_row["risk_factor_family"],
        "semantic_label": inv_row["semantic_label"],
        "column": col,
        "model_type": model_type,
        "encoder": encoder,
        "age_col": age_col or "",
        "sex_col": sex_col or "",
    }

    if cbag_col not in df.columns:
        base["status"] = f"missing_cbag_column:{cbag_col}"
        return base
    df = df.copy()
    df["_cbag_clean"] = clean_numeric(df[cbag_col], mask_extreme=False)

    # Avoid cBAG as its own outcome.
    if col == cbag_col:
        base["status"] = "skip_cbag_self"
        return base

    if model_type == "continuous":
        yraw = clean_numeric(df[col])

        # Physiologic cleaning for BP/pulse variables.
        label_norm = normalize_name(inv_row.get("semantic_label", ""))
        col_norm = normalize_name(col)
        if "systolic" in label_norm or re.search(r"(^|_)sbp($|_)|bp_sys|systolic", col_norm):
            yraw = _clean_bp_numeric(df[col], "sbp")
        elif "diastolic" in label_norm or re.search(r"(^|_)dbp($|_)|bp_dia|diastolic", col_norm):
            yraw = _clean_bp_numeric(df[col], "dbp")
        elif "pulse_pressure" in label_norm or "pulse_pressure" in col_norm:
            yraw = _clean_bp_numeric(df[col], "pulse_pressure")
        elif "mean_arterial" in label_norm or col_norm.endswith("_map_phys_clean") or re.search(r"(^|_)map($|_)", col_norm):
            yraw = _clean_bp_numeric(df[col], "map")
        elif label_norm == "pulse_rate" or re.search(r"(^|_)pulse($|_)|om_pulse", col_norm):
            yraw = _clean_bp_numeric(df[col], "pulse_rate")
        n = int(pd.concat([df["_cbag_clean"], yraw], axis=1).dropna().shape[0])
        base.update({
            "n_nonmissing": int(yraw.notna().sum()),
            "n_cbag_overlap": n,
            "target_mean": float(yraw.mean(skipna=True)) if yraw.notna().sum() else np.nan,
            "target_sd": float(yraw.std(skipna=True)) if yraw.notna().sum() else np.nan,
        })
        min_unique = 2 if str(col).startswith("derived_cognitive_status_ordinal") or normalize_name(col) in {"dx_sum", "dxsum"} else 4
        if n < min_n or yraw.nunique(dropna=True) < min_unique:
            base["status"] = f"too_few_or_low_variance_continuous<{min_n}"
            return base

        X = build_covariates(df, target_col=col, sex_col=sex_col, age_col=age_col, include_sex=True)
        y = zscore(yraw)
        fit = fit_ols(y, X, "cBAG_z")
        base.update({
            "n_model": fit.get("n_model", np.nan),
            "beta_cBAG_std": fit.get("beta", np.nan),
            "se_cBAG_std": fit.get("se", np.nan),
            "beta_ci_low": fit.get("ci_low", np.nan),
            "beta_ci_high": fit.get("ci_high", np.nan),
            "t_cBAG": fit.get("t", np.nan),
            "p_cBAG": fit.get("p", np.nan),
            "partial_r_cBAG": fit.get("partial_r", np.nan),
            "model_r2": fit.get("r2", np.nan),
            "covariates": ";".join(X.columns),
            "status": "ok" if pd.notna(fit.get("p", np.nan)) else "fit_failed",
        })
        if "fit_error" in fit:
            base["fit_error"] = fit["fit_error"]
        return base

    # Binary targets.
    if encoder == "sex":
        y = encode_sex(df[col])
        include_sex_cov = False
    elif encoder == "apoe4":
        y = encode_apoe4(df[col])
        include_sex_cov = True
    elif encoder == "impairment":
        y = encode_impairment(df[col])
        include_sex_cov = True
    else:
        y = encode_binary_general(df[col])
        include_sex_cov = True

    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    n = int(pd.concat([df["_cbag_clean"], y], axis=1).dropna().shape[0])
    base.update({
        "n_nonmissing": int(y.notna().sum()),
        "n_cbag_overlap": n,
        "n_cases": n1,
        "n_controls": n0,
        "case_fraction": float(n1 / (n1 + n0)) if (n1 + n0) else np.nan,
    })

    if n < min_n or min(n0, n1) < min_binary_class_n:
        base["status"] = f"too_few_binary_cases_or_controls<{min_binary_class_n}"
        return base

    X = build_covariates(df, target_col=col, sex_col=sex_col, age_col=age_col, include_sex=include_sex_cov)
    fit = fit_logistic_irls(y, X, "cBAG_z")
    base.update({
        "n_model": fit.get("n_model", np.nan),
        "beta_logit_cBAG": fit.get("beta_logit", np.nan),
        "se_logit_cBAG": fit.get("se_logit", np.nan),
        "z_cBAG": fit.get("z", np.nan),
        "p_cBAG": fit.get("p", np.nan),
        "or_per_1sd_cBAG": fit.get("or_per_1sd_cBAG", np.nan),
        "or_ci_low": fit.get("or_ci_low", np.nan),
        "or_ci_high": fit.get("or_ci_high", np.nan),
        "auc_full": fit.get("auc_full", np.nan),
        "auc_covariates": fit.get("auc_covariates", np.nan),
        "delta_auc": fit.get("delta_auc", np.nan),
        "covariates": ";".join(X.columns),
        "status": "ok" if pd.notna(fit.get("p", np.nan)) else "fit_failed",
    })
    if "fit_error" in fit:
        base["fit_error"] = fit["fit_error"]
    return base


# =============================================================================
# Supplement export
# =============================================================================

def export_supplementary_tables(results: pd.DataFrame, selected: pd.DataFrame, outdir: Path, supplement_number: str) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"SupplementaryTable_{supplement_number}_cBAG_risk_factor_effects"
    files = {
        "selected_csv": outdir / f"{prefix}_selected.csv",
        "selected_xlsx": outdir / f"{prefix}_selected.xlsx",
        "full_csv": outdir / f"{prefix}_full_screen.csv",
        "full_xlsx": outdir / f"{prefix}_full_screen.xlsx",
    }
    selected.to_csv(files["selected_csv"], index=False)
    results.to_csv(files["full_csv"], index=False)
    try:
        with pd.ExcelWriter(files["selected_xlsx"], engine="openpyxl") as writer:
            selected.to_excel(writer, index=False, sheet_name="selected")
        with pd.ExcelWriter(files["full_xlsx"], engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="full_screen")
    except Exception as e:
        print(f"[WARN] Could not write XLSX tables: {e}")
    return files


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--cohorts", default="ADNI,ADRC,HABS,AD_DECODE")
    ap.add_argument("--cbag-files", required=True)
    ap.add_argument("--cbag-column", default="cBAG_raw_clean")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-binary-class-n", type=int, default=10)
    ap.add_argument("--supplement-number", default="S8")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    cbag_files = parse_cbag_files_arg(args.cbag_files)
    cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]

    all_rows = []
    inv_rows = []
    merge_rows = []
    cog_qa_rows = []
    bp_qa_rows = []
    column_qa_rows = []

    for cohort in cohorts:
        if cohort not in cbag_files:
            print(f"[WARN] Missing cBAG file argument for {cohort}; skipping")
            merge_rows.append({
                "cohort": cohort,
                "status": "missing_cbag_file_argument",
            })
            continue

        if not cbag_files[cohort].exists():
            print(f"[WARN] cBAG file does not exist for {cohort}; skipping: {cbag_files[cohort]}")
            merge_rows.append({
                "cohort": cohort,
                "cbag_file": str(cbag_files[cohort]),
                "status": "cbag_file_not_found",
            })
            continue

        print("\n" + "=" * 100)
        print(f"COHORT: {cohort}")
        print("=" * 100)

        df, merge_summary = load_and_merge_cohort(args.metadata_dir, cohort, cbag_files[cohort])

        # Derive ordinal/binary cognitive status endpoints before target inventory.
        df, cog_qa = derive_cognitive_status_ordinal(df)
        cog_qa["cohort"] = cohort

        # Derive primary BP features before target inventory.
        df, bp_qa = derive_primary_bp_features(df, cohort)
        bp_qa_rows.append(bp_qa)

        merge_summary.update({
            "cognitive_status_source_column": cog_qa.get("source_column", ""),
            "cognitive_status_created": cog_qa.get("created", False),
            "cognitive_status_n_CN": cog_qa.get("n_CN_0", 0),
            "cognitive_status_n_MCI": cog_qa.get("n_MCI_1", 0),
            "cognitive_status_n_AD": cog_qa.get("n_AD_2", 0),
            "primary_sbp_col": bp_qa.get("primary_sbp_col", ""),
            "primary_dbp_col": bp_qa.get("primary_dbp_col", ""),
            "primary_pulse_col": bp_qa.get("primary_pulse_col", ""),
        })
        merge_rows.append(merge_summary)

        cbag_col = args.cbag_column
        if cbag_col not in df.columns:
            alt = first_existing(df, ["cBAG_raw_clean", "cBAG", "cBAG_z_clean", "BAG_raw_clean"])
            if alt is None:
                print(f"[WARN] No cBAG column for {cohort}; skipping")
                continue
            cbag_col = alt

        age_col = infer_age_col(df)
        sex_col = infer_sex_col(df)

        cog_qa_rows.append(cog_qa)
        snapshot = qa_column_snapshot(df, cbag_col=cbag_col, age_col=age_col, sex_col=sex_col)
        snapshot.insert(0, "cohort", cohort)
        column_qa_rows.append(snapshot)

        inv = build_inventory(df, min_n=args.min_n)
        inv["cohort"] = cohort
        inv["age_col"] = age_col or ""
        inv["sex_col"] = sex_col or ""
        inv_rows.append(inv)

        print(f"Rows: {len(df)}")
        print(f"cBAG column: {cbag_col}")
        print(f"Age column: {age_col or 'not found'}")
        print(f"Sex column: {sex_col or 'not found'}")
        print(f"Risk factor targets: {len(inv)}")
        if not inv.empty:
            print(inv.groupby(["risk_factor_family", "model_type"]).size().reset_index(name="n").to_string(index=False))

        for _, row in inv.iterrows():
            res = run_target_model(
                df=df,
                inv_row=row,
                cbag_col=cbag_col,
                age_col=age_col,
                sex_col=sex_col,
                min_n=args.min_n,
                min_binary_class_n=args.min_binary_class_n,
            )
            all_rows.append(res)

    results = pd.DataFrame(all_rows)
    inventory = pd.concat(inv_rows, ignore_index=True, sort=False) if inv_rows else pd.DataFrame()
    merge_df = pd.DataFrame(merge_rows)
    cognitive_status_qa = pd.DataFrame(cog_qa_rows)
    bp_primary_qa = pd.DataFrame(bp_qa_rows)
    column_qa = pd.concat(column_qa_rows, ignore_index=True, sort=False) if column_qa_rows else pd.DataFrame()

    if not results.empty:
        results["p_cBAG"] = pd.to_numeric(results["p_cBAG"], errors="coerce")
        results["q_cBAG_within_family"] = np.nan
        for key, idx in results.groupby(["cohort", "risk_factor_family", "model_type"], dropna=False).groups.items():
            results.loc[idx, "q_cBAG_within_family"] = fdr_bh(results.loc[idx, "p_cBAG"])
        results["is_nominal_0_05"] = results["p_cBAG"] < 0.05
        results["is_fdr_0_05"] = results["q_cBAG_within_family"] < 0.05

        # sort for readability
        sort_abs = pd.to_numeric(results.get("partial_r_cBAG", pd.Series(np.nan, index=results.index)), errors="coerce").abs()
        sort_abs = sort_abs.fillna(pd.to_numeric(results.get("delta_auc", pd.Series(np.nan, index=results.index)), errors="coerce").abs())
        results["_abs_effect"] = sort_abs
        results = results.sort_values(
            ["is_fdr_0_05", "cohort", "risk_factor_family", "q_cBAG_within_family", "p_cBAG", "_abs_effect"],
            ascending=[False, True, True, True, True, False],
        ).drop(columns="_abs_effect", errors="ignore")

    selected = results[results["status"].eq("ok")].copy() if not results.empty and "status" in results.columns else pd.DataFrame()
    if not selected.empty:
        selected["_priority"] = selected["column"].map(column_priority)
        selected = selected.sort_values(
            ["cohort", "risk_factor_family", "semantic_label", "q_cBAG_within_family", "p_cBAG", "_priority"],
            ascending=[True, True, True, True, True, True],
        )
        selected = selected.drop_duplicates(["cohort", "risk_factor_family", "semantic_label", "model_type"], keep="first")
        selected = selected.sort_values(
            ["is_fdr_0_05", "cohort", "risk_factor_family", "q_cBAG_within_family", "p_cBAG"],
            ascending=[False, True, True, True, True],
        ).drop(columns="_priority", errors="ignore")

    summary = pd.DataFrame()
    if not results.empty:
        summary = (
            results.groupby(["cohort", "risk_factor_family", "model_type", "status"], dropna=False)
            .size()
            .reset_index(name="n_tests")
        )

    # Save core outputs.
    files = {
        "merge": args.outdir / "SupplTable8_cBAG_risk_factor_merge_summary.csv",
        "inventory": args.outdir / "SupplTable8_cBAG_risk_factor_inventory.csv",
        "all": args.outdir / "SupplTable8_cBAG_risk_factor_effects_all.csv",
        "selected": args.outdir / "SupplTable8_cBAG_risk_factor_effects_selected.csv",
        "summary": args.outdir / "SupplTable8_cBAG_risk_factor_summary.csv",
        "cognitive_status_qa": args.outdir / "SupplTable8_cognitive_status_QA.csv",
        "bp_primary_qa": args.outdir / "SupplTable8_BP_primary_column_QA.csv",
        "column_qa": args.outdir / "SupplTable8_risk_factor_column_QA.csv",
        "readme": args.outdir / "SupplTable8_cBAG_risk_factor_README.txt",
    }
    merge_df.to_csv(files["merge"], index=False)
    inventory.to_csv(files["inventory"], index=False)
    results.to_csv(files["all"], index=False)
    selected.to_csv(files["selected"], index=False)
    summary.to_csv(files["summary"], index=False)
    cognitive_status_qa.to_csv(files["cognitive_status_qa"], index=False)
    bp_primary_qa.to_csv(files["bp_primary_qa"], index=False)
    column_qa.to_csv(files["column_qa"], index=False)

    supp_files = export_supplementary_tables(
        results=results,
        selected=selected,
        outdir=args.outdir / "supplementary",
        supplement_number=args.supplement_number,
    )

    readme = (
        "Supplementary Table S8: cBAG risk-factor effect sizes\n"
        "=====================================================\n\n"
        "Purpose:\n"
        "  Targeted cross-sectional association analysis testing whether cBAG is associated\n"
        "  with demographic, genetic, clinical, vascular, and metabolic risk factors.\n\n"
        "Binary targets:\n"
        "  logistic model: target ~ z(cBAG) + z(age) + sex where appropriate.\n"
        "  Effect size: OR per 1 SD higher cBAG, 95% CI, p, q, AUC, delta AUC.\n\n"
        "Continuous targets:\n"
        "  linear model: z(target) ~ z(cBAG) + z(age) + sex.\n"
        "  Effect size: standardized beta, 95% CI, partial r, p, q.\n\n"
        "FDR:\n"
        "  BH-FDR within cohort x risk factor family x model type.\n\n"
        "Primary output:\n"
        f"  {files['selected']}\n\n"
        "QA outputs:\n"
        f"  {files['cognitive_status_qa']}\n"
        f"  {files['bp_primary_qa']}\n"
        f"  {files['column_qa']}\n\n"
        "Supplementary table exports:\n"
        f"  {supp_files['selected_csv']}\n"
        f"  {supp_files['selected_xlsx']}\n"
        f"  {supp_files['full_csv']}\n"
        f"  {supp_files['full_xlsx']}\n"
    )
    files["readme"].write_text(readme)

    print("\n[OK] Wrote:")
    for p in files.values():
        print(p)
    print("\n[OK] Supplementary table exports:")
    for p in supp_files.values():
        print(p)

    print("\nSummary:")
    print(summary.to_string(index=False) if not summary.empty else "EMPTY")

    print("\nCognitive status QA:")
    print(cognitive_status_qa.to_string(index=False) if not cognitive_status_qa.empty else "EMPTY")

    print("\nPrimary BP QA:")
    print(bp_primary_qa.to_string(index=False) if not bp_primary_qa.empty else "EMPTY")

    print("\nTop selected risk-factor associations:")
    if selected.empty:
        print("EMPTY")
    else:
        show = [c for c in [
            "cohort", "risk_factor_family", "semantic_label", "column", "model_type",
            "n_model", "n_cases", "n_controls", "beta_cBAG_std", "partial_r_cBAG",
            "or_per_1sd_cBAG", "or_ci_low", "or_ci_high",
            "auc_full", "delta_auc", "p_cBAG", "q_cBAG_within_family", "is_fdr_0_05",
        ] if c in selected.columns]
        print(selected[show].head(100).to_string(index=False))


if __name__ == "__main__":
    main()
