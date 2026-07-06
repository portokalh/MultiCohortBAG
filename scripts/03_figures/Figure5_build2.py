#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Figure 5 from FINAL harmonized metadata + cBAG validation outputs.

Key design choices:
  - Uses the final harmonized metadata files as the source of diagnosis/APOE/cognition:
        $WORK/ines/data/harmonization/harmonized_metadata/<COHORT>_harmonized_metadata.csv
  - Uses enriched validation files when available:
        subject_level_validation_input_enriched_for_Figure4.csv
    and falls back to:
        subject_level_validation_input.csv
  - Patched for full-cohort validation folders by default:
        validation_figures_full_cohort/
  - Forces metadata merge by connectome/session key first. No row-order fallback.
  - Preserves all validation rows with cBAG, and reports whether metadata/status were recovered.
  - Uses meta__NORMCOG_01/meta__MCI_01/meta__AD_01/meta__DX_Label_harmonized for point colors.
  - Screens numeric metadata/neuroimaging variables against bias-corrected cBAG.
  - Creates a main Figure 5 for imaging_only and supplementary figures S5A-E for all models.

Run:
  python Figure5_CN_MCI_AD_fullcohort_patched.py \
    --validation-dir-name validation_figures_full_cohort
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Sequence
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# SETTINGS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
BASE_DIR = WORK / "ines"
RESULTS_ROOT = BASE_DIR / "results"
HARMONIZED_DIR = BASE_DIR / "data" / "harmonization" / "harmonized_metadata"

DATE_TAG = os.environ.get("FIGURE5_DATE_TAG", datetime.now().strftime("%Y%m%d"))
BIOVALIDATION_ROOT = RESULTS_ROOT / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
OUTDIR = BIOVALIDATION_ROOT / f"Figure5_{DATE_TAG}"
MERGED_OUTDIR = OUTDIR / "merged_tables"
FIGURE_OUTDIR = OUTDIR / "figures"
QA_OUTDIR = OUTDIR / "qa"
AUC_OUTDIR = OUTDIR / "auc"

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]
COHORT_LABELS = {"ADNI": "ADNI", "ADRC": "ADRC", "HABS": "HABS", "AD_DECODE": "AD-DECODE"}

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]
MAIN_FEATURE_SET = "imaging_only"
MODEL_LETTERS = {
    "imaging_only": "A",
    "imaging_demographics": "B",
    "imaging_biomarkers": "C",
    "full": "D",
    "full_no_cardiovascular": "E",
}
MODEL_LABELS = {
    "imaging_only": "imaging-only model",
    "imaging_demographics": "imaging + demographics model",
    "imaging_biomarkers": "imaging + biomarkers model",
    "full": "full model",
    "full_no_cardiovascular": "full model excluding cardiovascular variables",
}

PREDICTION_DIRS = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

# Folder inside each ablation_<feature_set> directory that contains the
# validation input files. Default patched to full-cohort outputs.
VALIDATION_DIR_NAME = os.environ.get("FIGURE5_VALIDATION_DIR_NAME", "validation_figures_full_cohort")

CBAG_PRIORITY = [
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_bias_corrected",
    "cBAG_BiasCorrected",
    "cBAG_foldwise_raw_clean",
    "cBAG_foldwise",
    "cBAG_raw_clean",
    "cBAG",
]

MIN_N = 30
MIN_UNIQUE = 5
FDR_THRESHOLD = 0.05
FIGURE_FORMATS = ["png", "pdf"]
DPI = 450
ALLOW_NONFDR_FALLBACK_FOR_MAIN = False
MIN_AUC_N = 8
MIN_AUC_CLASS_N = 3

SENTINELS = {
    -999999, -888888, -777777,
    -99999, -88888, -77777,
    -9999, -8888, -7777,
    -999, -888, -777,
    999, 888, 777,
    9999, 8888, 7777,
    99999, 88888, 77777,
    999999, 888888, 777777,
}

# Curated biological families for variable screening.
CURATED_LABELS = {
    "memory": "Memory",
    "executive_function": "Executive function",
    "processing_speed": "Processing speed",
    "language": "Language",
    "visuospatial": "Visuospatial",
    "global_cognition_screening": "MoCA / MMSE / CDR",
    "tau_ptau": "Tau / pTau",
    "amyloid_abeta": "Amyloid / Aβ",
    "nfl": "NfL",
    "gfap": "GFAP",
    "apoe": "APOE",
    "hippocampus": "Hippocampus",
    "brain_volume": "Brain volume",
    "fa_diffusion": "FA / diffusion",
    "graph_clustering": "Graph clustering",
    "graph_path_length": "Graph path length",
    "global_efficiency": "Global efficiency",
    "local_efficiency": "Local efficiency",
    "cardiovascular": "Cardiovascular",
    "depression_anxiety": "Depression/anxiety",
}
CURATED_FAMILIES = list(CURATED_LABELS)

MAIN_COLUMNS = {
    "Cognition": [
        "memory", "executive_function", "processing_speed", "language",
        "visuospatial", "global_cognition_screening",
    ],
    "Fluid / genetic biomarker": ["tau_ptau", "amyloid_abeta", "nfl", "gfap", "apoe"],
    "Brain network / structure": [
        "global_efficiency", "local_efficiency", "graph_clustering", "graph_path_length",
        "hippocampus", "brain_volume", "fa_diffusion",
    ],
    "Clinical / vascular": ["cardiovascular", "depression_anxiety"],
}

# Broad testing domains used for domain-wise FDR.
# These are intentionally aligned with the Figure 5 conceptual columns.
FDR_DOMAIN_BY_FAMILY = {
    fam: domain
    for domain, families in MAIN_COLUMNS.items()
    for fam in families
}
UNCORRECTED_ALPHA = 0.05

# Remove height and weight from association screening, but keep BMI/body mass index.
# IMPORTANT:
# normalize_name() converts names such as "vitals_VSHEIGHT" or "VSWeight"
# to lowercase underscore-style strings. The patterns below intentionally catch
# embedded height/weight strings, including cohort-specific vital-sign fields,
# while explicitly preserving BMI/body_mass_index.
HEIGHT_WEIGHT_EXCLUDE_PATTERNS = [
    # Any embedded height-like variable:
    # height, heigh, VSHEIGHT, VSHEIGH, VSHEI, vitals_VSHEIGHT, stature, standingheight, etc.
    r"height",
    r"heigh",
    r"vshei",
    r"stature",

    # Any embedded weight-like variable:
    # weight, weigh, VSWEIGHT, VSWEIG, VSWT, body_weight, bodyweight, etc.
    r"weight",
    r"weigh",
    r"vswei",
    r"vswt",
    r"body_weight",
    r"bodyweight",

    # Common abbreviated anthropometric weight fields.
    # Keep this conservative so we do not accidentally exclude unrelated columns.
    r"(^|_)wt($|_)",
    r"(^|_)kg($|_)",
    r"kilogram",
]
BMI_KEEP_PATTERNS = [
    # Keep any column containing BMI anywhere after normalize_name().
    # Examples kept: BMI, meta__BMI, vitals_BMI, subject_bmi_value.
    r"bmi",

    # Also keep explicit body-mass-index spellings.
    r"body_mass_index",
    r"bodymassindex",
]

CURATED_RULES = [
    ("global_efficiency", r"global[_\s]*efficiency"),
    ("local_efficiency", r"local[_\s]*efficiency"),
    ("graph_clustering", r"cluster|clustering|coeff|transitivity|modular"),
    ("graph_path_length", r"path[_\s]*length|shortest[_\s]*path|characteristic[_\s]*path|pathlength"),
    ("hippocampus", r"hippo|hippocampus|\bhc[_\s]"),
    ("brain_volume", r"brain[_\s]*volume|total[_\s]*brain|\btbv\b|\bicv\b|volume|volumetric|cortical[_\s]*thickness|thickness"),
    ("fa_diffusion", r"\bfa\b|fractional[_\s]*anisotropy|diffusion|dti|mean[_\s]*diffus|\bmd\b|\brd\b|\bad\b|\badc\b"),
    ("tau_ptau", r"ptau|p[_\s\-]*tau|tau|total[_\s]*tau|totaltau"),
    ("amyloid_abeta", r"abeta|aβ|amyloid|a_beta|ab42|ab40"),
    ("nfl", r"\bnfl\b|nfl_|nefl|neurofilament"),
    ("gfap", r"gfap"),
    ("apoe", r"apoe|e4|genotype"),
    ("memory", r"memory|sevlt|ravlt|avlt|logical|lm1|lm2|delayed|recall|bentd|story"),
    ("executive_function", r"executive|trail[_\s]*b|trailb|bckwds|backward|abstraction|set[_\s]*shift"),
    ("processing_speed", r"processing[_\s]*speed|trail[_\s]*a|traila|digit[_\s]*symbol|digitsymbol|symbol[_\s]*substitution|ufov"),
    ("language", r"language|fluency|fas|animals|animal|naming|wat|word[_\s]*accent|verbal[_\s]*flu"),
    ("visuospatial", r"visuospatial|benson|figure|copy|construction"),
    ("global_cognition_screening", r"moca|mocatots|mmse|cdr|cdrsb|cdglobal|adas|cognition|cognitive|global[_\s]*cog"),
    ("cardiovascular", r"blood[_\s]*pressure|\bsbp\b|\bdbp\b|pulse|chol|hdl|ldl|triglycer|glucose|hba1c|diabetes|hypertension|\bbmi\b|body[_\s]*mass[_\s]*index|insulin|homa|egfr|creatinine|vascular|vitals"),
    ("depression_anxiety", r"depress|anxiety|\bgds\b|pswq|worry"),
]

EXCLUDE_TOKENS = [
    "cbag", "bag", "brainage", "brain_age", "predictedage", "predicted_age",
    "prediction", "biascorrect", "bias_correct", "correctedage", "chronological",
    "fold", "split", "train", "test", "val", "oof", "rmse", "mae", "r2",
    "auc", "target", "label", "index", "unnamed", "path", "file", "filename",
    "metadata", "connectome_key", "connectome_full_key", "graph_path", "source_path",
    "merge", "status_source", "subject_key", "session_key",
]
DEMOGRAPHIC_TOKENS = ["age", "sex", "gender", "educ", "education", "site", "scanner", "race", "ethnic"]
INCLUDE_DEMOGRAPHICS_IN_SCREEN = False


# =============================================================================
# PATHS / KEYS
# =============================================================================

def validation_path(cohort: str, feature_set: str) -> Path:
    """Return validation input path for the selected validation output folder.

    For full-cohort Figure 5, use validation_figures_full_cohort.
    Prefer the enriched Figure 4 file because it contains neuroimaging variables
    used by Figure 5 screening; fall back to the raw validation input.
    """
    vdir = RESULTS_ROOT / PREDICTION_DIRS[cohort] / f"ablation_{feature_set}" / VALIDATION_DIR_NAME
    enriched = vdir / "subject_level_validation_input_enriched_for_Figure4.csv"
    raw = vdir / "subject_level_validation_input.csv"
    return enriched if enriched.exists() else raw


def metadata_path(cohort: str) -> Path:
    return HARMONIZED_DIR / f"{cohort}_harmonized_metadata.csv"


def normalize_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def clean_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").copy()
    for val in SENTINELS:
        out = out.mask(out == val, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


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


def normalize_connectome_key(x: object, cohort: Optional[str] = None) -> Optional[str]:
    """Normalize validation and harmonized metadata IDs to the connectome/session key."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return None

    # Operate on filename stem if path-like.
    s = Path(s).stem
    s = re.sub(r"^\._", "", s)
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"_conn_plain$", "", s, flags=re.I)
    s = re.sub(r"conn_plain$", "", s, flags=re.I)
    s = re.sub(r"_connectomics$", "", s, flags=re.I)

    # ADRC legacy filenames/labels -> D####.
    m = re.search(r"ADRC\s*0*(\d{4})", s, flags=re.I)
    if m:
        return f"d{int(m.group(1)):04d}"

    # ADNI/HABS visit keys.
    m = re.search(r"\b([RH]\d{4,5}_y\d+)\b", s, flags=re.I)
    if m:
        return m.group(1).lower()

    # AD-DECODE Sxxxxx. Strip suffixes such as S02877_master_T_1 first via regex.
    m = re.search(r"\b(S\d{5})\b", s, flags=re.I)
    if m:
        return m.group(1).lower()

    # AD-DECODE numeric MRI exam only when explicitly indicated by cohort.
    if cohort == "AD_DECODE" and re.fullmatch(r"\d+(?:\.0)?", s):
        return f"s{int(float(s)):05d}"

    # ADRC D####.
    m = re.search(r"\b(D\d{4})\b", s, flags=re.I)
    if m:
        return m.group(1).lower()

    # Generic one-letter + five/four digits.
    m = re.search(r"\b([A-Za-z]\d{5})\b", s, flags=re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b([A-Za-z]\d{4})\b", s, flags=re.I)
    if m:
        return m.group(1).lower()

    # Strip AD-DECODE acquisition suffixes and retry.
    s2 = re.sub(r"_master_T.*$", "", s, flags=re.I)
    s2 = re.sub(r"_temp_T.*$", "", s2, flags=re.I)
    s2 = re.sub(r"_T\d?.*$", "", s2, flags=re.I)
    if s2 != s:
        return normalize_connectome_key(s2, cohort=cohort)

    return normalize_name(s) or None


VAL_KEY_CANDIDATES = [
    "connectome_key", "connectome_full_key", "CONNECTOME_KEY_CLEAN", "subject_match",
    "subject_source", "subject_id", "Subject_ID", "Subject", "participant_id",
    "DWI", "DWI_key", "runno", "graph_id", "MRI_Exam", "MRI_Exam_fixed",
    "PTID", "RID", "ID", "id", "match_id",
]
META_KEY_CANDIDATES = [
    "CONNECTOME_KEY_USED_FOR_INTERSECTION", "CONNECTOME_KEY_CLEAN", "table1_session_key",
    "DWI", "DWI_key", "connectome_key", "connectome_full_key", "graph_id", "runno",
    "MRI_Exam", "MRI_Exam_fixed", "PTID", "RID", "subject_id", "Subject", "ID", "id", "match_id",
]


def choose_forced_connectome_merge(val: pd.DataFrame, meta: pd.DataFrame, cohort: str) -> tuple[str, str, int]:
    """Pick the validation/meta key pair with largest normalized connectome-key overlap."""
    val_cols = [c for c in VAL_KEY_CANDIDATES if c in val.columns]
    meta_cols = [c for c in META_KEY_CANDIDATES if c in meta.columns]

    # Include any plausible key columns as fallback, but keep explicit candidates first.
    def plausible(cols):
        toks = ["connectome", "dwi", "subject", "participant", "runno", "graph", "mri", "ptid", "rid", "id", "match"]
        return [c for c in cols if any(t in str(c).lower() for t in toks)]
    val_cols += [c for c in plausible(val.columns) if c not in val_cols]
    meta_cols += [c for c in plausible(meta.columns) if c not in meta_cols]

    best = ("", "", 0)
    for vc in val_cols:
        vkeys = set(val[vc].map(lambda x: normalize_connectome_key(x, cohort)).dropna())
        if not vkeys:
            continue
        for mc in meta_cols:
            mkeys = set(meta[mc].map(lambda x: normalize_connectome_key(x, cohort)).dropna())
            if not mkeys:
                continue
            overlap = len(vkeys & mkeys)
            if overlap > best[2]:
                best = (vc, mc, overlap)
    return best


def prefix_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    rename = {c: f"meta__{c}" for c in out.columns if not str(c).startswith("meta__")}
    return out.rename(columns=rename)


# =============================================================================
# MERGE + QA
# =============================================================================

def load_and_merge(cohort: str, feature_set: str) -> tuple[pd.DataFrame, dict]:
    vpath = validation_path(cohort, feature_set)
    mpath = metadata_path(cohort)

    if not vpath.exists():
        raise FileNotFoundError(f"Missing validation file: {vpath}")
    if not mpath.exists():
        raise FileNotFoundError(f"Missing harmonized metadata file: {mpath}")

    val = pd.read_csv(vpath, low_memory=False)
    meta = pd.read_csv(mpath, low_memory=False)

    cbag_col = first_existing(val, CBAG_PRIORITY)
    if cbag_col is None:
        raise ValueError(f"No bias-corrected cBAG column found in {vpath}")

    vc, mc, overlap = choose_forced_connectome_merge(val, meta, cohort)
    if not vc or not mc or overlap == 0:
        raise RuntimeError(
            f"Could not find connectome-key overlap for {cohort} {feature_set}. "
            f"Validation={vpath}; metadata={mpath}"
        )

    val = val.copy()
    meta = meta.copy()
    val["_merge_key"] = val[vc].map(lambda x: normalize_connectome_key(x, cohort))
    meta["_merge_key"] = meta[mc].map(lambda x: normalize_connectome_key(x, cohort))

    # Deduplicate metadata by merge key, prioritizing known diagnosis/cognition/APOE.
    meta["_has_dx"] = ~meta.get("DX_Label_harmonized", pd.Series("Unknown", index=meta.index)).fillna("Unknown").astype(str).eq("Unknown")
    meta["_has_cog"] = meta.get("Global_Cognition_Composite", pd.Series(np.nan, index=meta.index)).notna()
    meta["_has_apoe"] = meta.get("APOE_genotype_harmonized", pd.Series(np.nan, index=meta.index)).notna()
    meta["_orig_order"] = np.arange(len(meta))
    meta_dedup = (
        meta.dropna(subset=["_merge_key"])
        .sort_values(["_merge_key", "_has_dx", "_has_cog", "_has_apoe", "_orig_order"], ascending=[True, False, False, False, True])
        .drop_duplicates("_merge_key", keep="first")
        .drop(columns=["_has_dx", "_has_cog", "_has_apoe", "_orig_order"], errors="ignore")
    )

    val_pref = val.copy()
    val_pref["cohort"] = cohort
    val_pref["feature_set"] = feature_set
    val_pref["_cbag"] = clean_numeric(val_pref[cbag_col])
    val_pref["_cbag_col"] = cbag_col
    val_pref["_validation_path"] = str(vpath)
    val_pref["_metadata_path"] = str(mpath)
    val_pref["_validation_key_col"] = vc
    val_pref["_metadata_key_col"] = mc

    meta_pref = prefix_metadata(meta_dedup)
    meta_pref = meta_pref.rename(columns={"meta___merge_key": "_merge_key"})

    merged = val_pref.merge(meta_pref, on="_merge_key", how="left", validate="m:1")

    # Derive a consistent status column for coloring/QC.
    merged["_cognitive_status"] = derive_cognitive_status(merged)

    matched = int(merged["meta__DX_Label_harmonized"].notna().sum()) if "meta__DX_Label_harmonized" in merged.columns else int(merged.filter(like="meta__").notna().any(axis=1).sum())
    with_cbag = merged["_cbag"].notna()
    status_counts_all = merged["_cognitive_status"].value_counts(dropna=False).to_dict()
    status_counts_cbag = merged.loc[with_cbag, "_cognitive_status"].value_counts(dropna=False).to_dict()

    qa = {
        "cohort": cohort,
        "feature_set": feature_set,
        "validation_path": str(vpath),
        "metadata_path": str(mpath),
        "validation_rows": len(val),
        "metadata_rows": len(meta),
        "metadata_rows_dedup": len(meta_dedup),
        "validation_key_col": vc,
        "metadata_key_col": mc,
        "key_overlap_unique": overlap,
        "merged_rows": len(merged),
        "matched_metadata_rows": matched,
        "unmatched_validation_rows": int(len(merged) - matched),
        "cbag_col": cbag_col,
        "n_cbag_nonnull": int(with_cbag.sum()),
        "status_counts_all": status_counts_all,
        "status_counts_cbag_nonnull": status_counts_cbag,
        "n_normal_cbag": int((merged.loc[with_cbag, "_cognitive_status"] == "Cognitively normal").sum()),
        "n_impaired_cbag": int((merged.loc[with_cbag, "_cognitive_status"] == "Cognitively impaired").sum()),
        "n_unknown_status_cbag": int((merged.loc[with_cbag, "_cognitive_status"] == "Unknown status").sum()),
    }
    return merged, qa


def derive_cognitive_status(df: pd.DataFrame) -> pd.Series:
    """Return normal/impaired/unknown using unprefixed or meta-prefixed harmonized diagnosis columns."""
    idx = df.index
    status = pd.Series("Unknown status", index=idx, dtype="object")

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    norm_col = col("NORMCOG_01", "meta__NORMCOG_01")
    mci_col = col("MCI_01", "meta__MCI_01")
    ad_col = col("AD_01", "meta__AD_01")
    dem_col = col("DEMENTIA_01", "meta__DEMENTIA_01")
    dx_col = col("DX_Label_harmonized", "meta__DX_Label_harmonized")
    comp_col = col("cognitive_impairment_composite", "meta__cognitive_impairment_composite")

    if norm_col is not None:
        x = pd.to_numeric(df[norm_col], errors="coerce")
        status.loc[x.eq(1)] = "Cognitively normal"
        status.loc[x.eq(0)] = "Cognitively impaired"

    for c in [mci_col, ad_col, dem_col]:
        if c is not None:
            x = pd.to_numeric(df[c], errors="coerce")
            status.loc[x.eq(1)] = "Cognitively impaired"

    if dx_col is not None:
        dx = df[dx_col].astype(str).str.lower()
        known = ~dx.isin(["nan", "none", "unknown", ""])
        status.loc[known & dx.str.contains("normal|normcog|control|cn", regex=True, na=False)] = "Cognitively normal"
        status.loc[known & dx.str.contains("mci|ad|dement|impaired", regex=True, na=False)] = "Cognitively impaired"

    if comp_col is not None:
        x = pd.to_numeric(df[comp_col], errors="coerce")
        status.loc[status.eq("Unknown status") & x.eq(0)] = "Cognitively normal"
        status.loc[status.eq("Unknown status") & x.eq(1)] = "Cognitively impaired"

    return status


# =============================================================================
# SCREENING
# =============================================================================

def fdr_bh(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    p_ok = p[ok]
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_ok = np.empty_like(q_ranked)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


def is_bmi_col(col: str) -> bool:
    low = normalize_name(col)
    return any(re.search(pat, low, flags=re.I) for pat in BMI_KEEP_PATTERNS)


def is_height_weight_col(col: str) -> bool:
    """Exclude height/weight variables from screening while explicitly preserving BMI."""
    low = normalize_name(col)
    if is_bmi_col(low):
        return False
    return any(re.search(pat, low, flags=re.I) for pat in HEIGHT_WEIGHT_EXCLUDE_PATTERNS)


def is_excluded_col(col: str) -> bool:
    low = normalize_name(col)
    if any(tok in low for tok in EXCLUDE_TOKENS):
        return True
    if is_height_weight_col(col):
        return True
    if not INCLUDE_DEMOGRAPHICS_IN_SCREEN and any(tok in low for tok in DEMOGRAPHIC_TOKENS):
        return True
    return False


def testing_domain_from_family(fam: Optional[str]) -> str:
    if fam in FDR_DOMAIN_BY_FAMILY:
        return FDR_DOMAIN_BY_FAMILY[fam]
    return "Uncurated / other"


def curated_family(var: str) -> Optional[str]:
    low = normalize_name(var)
    if re.search(r"sars|covid|spike|nucleocapsid|rbd|\bpc\d+\b|id_rna|idrna|runno|subject", low, flags=re.I):
        return None
    for fam, pat in CURATED_RULES:
        if re.search(pat, low, flags=re.I):
            return fam
    return None


def scan_numeric_vars(df: pd.DataFrame, cohort: str, feature_set: str) -> pd.DataFrame:
    rows = []
    y = clean_numeric(df["_cbag"])

    for col in df.columns:
        if col.startswith("_") or is_excluded_col(col):
            continue
        x = clean_numeric(df[col])
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(tmp) < MIN_N or tmp["x"].nunique() < MIN_UNIQUE or tmp["y"].nunique() < 2:
            continue
        try:
            r, p = stats.pearsonr(tmp["x"], tmp["y"])
            lr = stats.linregress(tmp["x"], tmp["y"])
        except Exception:
            continue
        fam = curated_family(col)
        rows.append({
            "feature_set": feature_set,
            "cohort": cohort,
            "variable": col,
            "variable_norm": normalize_name(col),
            "curated_family": fam,
            "curated_family_label": CURATED_LABELS.get(fam, "") if fam else "",
            "testing_domain": testing_domain_from_family(fam),
            "is_bmi_like": bool(is_bmi_col(col)),
            "is_height_weight_like": bool(is_height_weight_col(col)),
            "n": int(len(tmp)),
            "n_unique": int(tmp["x"].nunique()),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "abs_pearson_r": float(abs(r)),
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "x_mean": float(tmp["x"].mean()),
            "x_sd": float(tmp["x"].std(ddof=1)),
            "x_min": float(tmp["x"].min()),
            "x_max": float(tmp["x"].max()),
            "cbag_col": df["_cbag_col"].iloc[0] if "_cbag_col" in df.columns and len(df) else "",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_uncorrected_lt_0_05"] = pd.to_numeric(out["pearson_p"], errors="coerce") < UNCORRECTED_ALPHA
        out["fdr_q_within_cohort"] = fdr_bh(out["pearson_p"].values)
    return out


def add_fdr_columns(assoc_all: pd.DataFrame) -> pd.DataFrame:
    """Add overall, within-cohort, within-domain, and within-family FDR columns."""
    if assoc_all.empty:
        return assoc_all
    df = assoc_all.copy()
    df["pearson_p"] = pd.to_numeric(df["pearson_p"], errors="coerce")
    df["p_uncorrected_lt_0_05"] = df["pearson_p"] < UNCORRECTED_ALPHA

    # FDR across every tested association in this feature set.
    df["fdr_q_all_tests_feature_set"] = fdr_bh(df["pearson_p"].values)

    # FDR within each cohort across all tested variables.
    df["fdr_q_within_cohort"] = np.nan
    for cohort, idx in df.groupby("cohort").groups.items():
        df.loc[idx, "fdr_q_within_cohort"] = fdr_bh(df.loc[idx, "pearson_p"].values)

    # FDR within broad biological/testing domain inside each cohort.
    df["fdr_q_within_cohort_domain"] = np.nan
    for (_, _), idx in df.groupby(["cohort", "testing_domain"], dropna=False).groups.items():
        df.loc[idx, "fdr_q_within_cohort_domain"] = fdr_bh(df.loc[idx, "pearson_p"].values)

    # Optional narrower FDR inside curated family within each cohort.
    df["fdr_q_within_cohort_family"] = np.nan
    fam_df = df[df["curated_family"].notna()].copy()
    for (_, _), idx in fam_df.groupby(["cohort", "curated_family"], dropna=False).groups.items():
        df.loc[idx, "fdr_q_within_cohort_family"] = fdr_bh(df.loc[idx, "pearson_p"].values)

    df["sig_fdr_all_tests_feature_set"] = pd.to_numeric(df["fdr_q_all_tests_feature_set"], errors="coerce") < FDR_THRESHOLD
    df["sig_fdr_within_cohort"] = pd.to_numeric(df["fdr_q_within_cohort"], errors="coerce") < FDR_THRESHOLD
    df["sig_fdr_within_cohort_domain"] = pd.to_numeric(df["fdr_q_within_cohort_domain"], errors="coerce") < FDR_THRESHOLD
    df["sig_fdr_within_cohort_family"] = pd.to_numeric(df["fdr_q_within_cohort_family"], errors="coerce") < FDR_THRESHOLD
    return df


def summarize_test_counts(assoc_all: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count tested columns and significant associations by cohort and by cohort/domain."""
    base_cols = [
        "feature_set", "cohort", "testing_domain", "n_tested",
        "n_uncorrected_p_lt_0_05",
        "n_fdr_all_tests_feature_set_lt_0_05",
        "n_fdr_within_cohort_lt_0_05",
        "n_fdr_within_cohort_domain_lt_0_05",
        "n_fdr_within_cohort_family_lt_0_05",
    ]
    if assoc_all.empty:
        empty_domain = pd.DataFrame(columns=base_cols)
        empty_cohort = pd.DataFrame(columns=[c for c in base_cols if c != "testing_domain"])
        return empty_domain, empty_cohort

    df = assoc_all.copy()
    for c in [
        "p_uncorrected_lt_0_05",
        "sig_fdr_all_tests_feature_set",
        "sig_fdr_within_cohort",
        "sig_fdr_within_cohort_domain",
        "sig_fdr_within_cohort_family",
    ]:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)

    def agg(g):
        return pd.Series({
            "n_tested": int(len(g)),
            "n_uncorrected_p_lt_0_05": int(g["p_uncorrected_lt_0_05"].sum()),
            "n_fdr_all_tests_feature_set_lt_0_05": int(g["sig_fdr_all_tests_feature_set"].sum()),
            "n_fdr_within_cohort_lt_0_05": int(g["sig_fdr_within_cohort"].sum()),
            "n_fdr_within_cohort_domain_lt_0_05": int(g["sig_fdr_within_cohort_domain"].sum()),
            "n_fdr_within_cohort_family_lt_0_05": int(g["sig_fdr_within_cohort_family"].sum()),
        })

    by_domain = df.groupby(["feature_set", "cohort", "testing_domain"], dropna=False).apply(agg).reset_index()
    by_cohort = df.groupby(["feature_set", "cohort"], dropna=False).apply(agg).reset_index()
    return by_domain, by_cohort


def save_candidate_variable_inventory(merged: pd.DataFrame, cohort: str, feature_set: str) -> pd.DataFrame:
    """Save a column-level audit showing what was excluded and what was testable."""
    rows = []
    y = clean_numeric(merged["_cbag"]) if "_cbag" in merged.columns else pd.Series(np.nan, index=merged.index)
    for col in merged.columns:
        if col.startswith("_"):
            continue
        fam = curated_family(col)
        excluded = is_excluded_col(col)
        x = clean_numeric(merged[col])
        complete = pd.DataFrame({"x": x, "y": y}).dropna()
        rows.append({
            "feature_set": feature_set,
            "cohort": cohort,
            "variable": col,
            "variable_norm": normalize_name(col),
            "curated_family": fam,
            "testing_domain": testing_domain_from_family(fam),
            "excluded_from_screen": bool(excluded),
            "is_bmi_like": bool(is_bmi_col(col)),
            "is_height_weight_like": bool(is_height_weight_col(col)),
            "is_tested": bool((not excluded) and len(complete) >= MIN_N and complete["x"].nunique() >= MIN_UNIQUE and complete["y"].nunique() >= 2),
            "n_complete_with_cbag": int(len(complete)),
            "n_unique": int(complete["x"].nunique()) if not complete.empty else 0,
        })
    return pd.DataFrame(rows)


def best_fdr_by_cohort_family(assoc: pd.DataFrame) -> pd.DataFrame:
    if assoc.empty:
        return assoc
    df = assoc[assoc["curated_family"].notna()].copy()
    q_col = "fdr_q_within_cohort_domain" if "fdr_q_within_cohort_domain" in df.columns else "fdr_q_within_cohort"
    df = df[pd.to_numeric(df[q_col], errors="coerce") < FDR_THRESHOLD]
    if df.empty:
        return df
    best = (
        df.sort_values(["cohort", "curated_family", "abs_pearson_r", "pearson_p"], ascending=[True, True, False, True])
        .groupby(["cohort", "curated_family"], as_index=False)
        .head(1)
        .copy()
    )
    best["r2"] = best["pearson_r"] ** 2
    return best


def select_main_associations(assoc: pd.DataFrame) -> pd.DataFrame:
    selected = []
    q_col = "fdr_q_within_cohort_domain" if "fdr_q_within_cohort_domain" in assoc.columns else "fdr_q_within_cohort"
    fdr = assoc[assoc["curated_family"].notna() & (pd.to_numeric(assoc[q_col], errors="coerce") < FDR_THRESHOLD)].copy()
    for cohort in COHORTS:
        for category, families in MAIN_COLUMNS.items():
            sub = fdr[(fdr["cohort"].eq(cohort)) & (fdr["curated_family"].isin(families))].copy()
            source = "FDR"
            if sub.empty and ALLOW_NONFDR_FALLBACK_FOR_MAIN:
                sub = assoc[(assoc["cohort"].eq(cohort)) & (assoc["curated_family"].isin(families))].copy()
                source = "non-FDR fallback"
            if sub.empty:
                continue
            row = sub.assign(_q=pd.to_numeric(sub[q_col], errors="coerce")).sort_values(
                ["_q", "pearson_p", "abs_pearson_r"],
                ascending=[True, True, False]
            ).iloc[0].copy()
            row = row.drop(labels=["_q"], errors="ignore")
            row["main_category"] = category
            row["selected_from"] = source
            row["r2"] = row["pearson_r"] ** 2
            selected.append(row)
    return pd.DataFrame(selected)


def run_screening_for_feature_set(feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    assoc_frames = []
    qa_rows = []
    unmatched_frames = []
    inventory_frames = []
    input_path_rows = []

    for cohort in COHORTS:
        print(f"\n[INFO] {feature_set} | {cohort}")
        merged, qa = load_and_merge(cohort, feature_set)
        qa_rows.append(qa)
        input_path_rows.append({
            "feature_set": feature_set,
            "cohort": cohort,
            "validation_path": qa.get("validation_path", ""),
            "metadata_path": qa.get("metadata_path", ""),
            "validation_key_col": qa.get("validation_key_col", ""),
            "metadata_key_col": qa.get("metadata_key_col", ""),
            "cbag_col": qa.get("cbag_col", ""),
        })

        merged_path = MERGED_OUTDIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv"
        merged.to_csv(merged_path, index=False)

        inv = save_candidate_variable_inventory(merged, cohort, feature_set)
        inventory_frames.append(inv)

        unmatched = merged[merged.filter(like="meta__").notna().any(axis=1).eq(False)].copy()
        if not unmatched.empty:
            unmatched["cohort"] = cohort
            unmatched["feature_set"] = feature_set
            unmatched_frames.append(unmatched)

        assoc = scan_numeric_vars(merged, cohort, feature_set)
        if not assoc.empty:
            assoc_frames.append(assoc)
        print(
            f"  validation rows={qa['validation_rows']} | matched metadata={qa['matched_metadata_rows']} | "
            f"cBAG n={qa['n_cbag_nonnull']} | status(cBAG)={qa['status_counts_cbag_nonnull']} | tested={len(assoc)}"
        )

    assoc_all = pd.concat(assoc_frames, ignore_index=True, sort=False) if assoc_frames else pd.DataFrame()
    if not assoc_all.empty:
        assoc_all = add_fdr_columns(assoc_all)
        assoc_all = assoc_all.sort_values(["cohort", "testing_domain", "abs_pearson_r", "pearson_p"], ascending=[True, True, False, True])

    qa_df = pd.DataFrame(qa_rows)
    assoc_all.to_csv(OUTDIR / f"metadata_variable_associations_{feature_set}.csv", index=False)
    qa_df.to_csv(QA_OUTDIR / f"Figure5_merge_QA_{feature_set}.csv", index=False)

    if inventory_frames:
        pd.concat(inventory_frames, ignore_index=True, sort=False).to_csv(
            QA_OUTDIR / f"Figure5_candidate_variable_inventory_{feature_set}.csv", index=False
        )
    pd.DataFrame(input_path_rows).to_csv(
        QA_OUTDIR / f"Figure5_input_paths_validation_and_metadata_{feature_set}.csv", index=False
    )

    by_domain, by_cohort = summarize_test_counts(assoc_all, feature_set)
    by_domain.to_csv(QA_OUTDIR / f"Figure5_association_test_counts_by_domain_{feature_set}.csv", index=False)
    by_cohort.to_csv(QA_OUTDIR / f"Figure5_association_test_counts_by_cohort_{feature_set}.csv", index=False)

    if unmatched_frames:
        pd.concat(unmatched_frames, ignore_index=True, sort=False).to_csv(
            QA_OUTDIR / f"Figure5_unmatched_validation_rows_{feature_set}.csv", index=False
        )

    pd.DataFrame(best_fdr_by_cohort_family(assoc_all)).to_csv(
        OUTDIR / f"curated_family_fdr_significant_{feature_set}.csv", index=False
    )

    print(f"\n[COUNT SUMMARY] {feature_set} by cohort")
    if not by_cohort.empty:
        print(by_cohort.to_string(index=False))
    print(f"[INFO] Saved counts: {QA_OUTDIR / f'Figure5_association_test_counts_by_cohort_{feature_set}.csv'}")
    print(f"[INFO] Saved domain counts: {QA_OUTDIR / f'Figure5_association_test_counts_by_domain_{feature_set}.csv'}")
    return assoc_all, qa_df


# =============================================================================
# PLOTTING
# =============================================================================

def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3g}"


def q_text(q: float) -> str:
    if not np.isfinite(q):
        return "q=NA"
    if q < 1e-4:
        return "q<1e-4"
    if q < 0.001:
        return "q<0.001"
    return f"q={q:.3g}"


def shorten(x: object, n: int = 28) -> str:
    s = str(x).replace("meta__", "").replace("_raw_clean", "")
    return s if len(s) <= n else s[: n - 1] + "…"


def regression_ci(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lr = stats.linregress(x, y)
    y_grid = lr.intercept + lr.slope * x_grid
    n = len(x)
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    y_hat = lr.intercept + lr.slope * x
    resid = y - y_hat
    dof = max(n - 2, 1)
    mse = np.sum(resid ** 2) / dof
    if sxx <= 0:
        se = np.full_like(x_grid, np.nan)
    else:
        se = np.sqrt(mse * (1 / n + (x_grid - x_mean) ** 2 / sxx))
    tcrit = stats.t.ppf(0.975, dof)
    return y_grid, y_grid - tcrit * se, y_grid + tcrit * se


def plot_association_panel(ax: plt.Axes, merged: pd.DataFrame, row: pd.Series, title_prefix: str) -> dict:
    var = row["variable"]
    if var not in merged.columns:
        ax.text(0.5, 0.5, "Variable not found", ha="center", va="center", fontsize=8)
        ax.axis("off")
        return {"status": "missing_variable"}

    tmp = pd.DataFrame({
        "x": clean_numeric(merged[var]),
        "y": clean_numeric(merged["_cbag"]),
        "status": merged["_cognitive_status"].fillna("Unknown status"),
    }).dropna(subset=["x", "y"])

    if len(tmp) < MIN_N or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        ax.text(0.5, 0.5, f"Insufficient data\nn={len(tmp)}", ha="center", va="center", fontsize=8)
        ax.axis("off")
        return {"status": "insufficient", "n": len(tmp)}

    colors = {
        "Cognitively normal": "#009E73",
        "Cognitively impaired": "#CC79A7",
        "Unknown status": "#BDBDBD",
    }
    for lab in ["Unknown status", "Cognitively normal", "Cognitively impaired"]:
        sub = tmp[tmp["status"].eq(lab)]
        if sub.empty:
            continue
        ax.scatter(sub["x"], sub["y"], s=16, alpha=0.70, label=lab, color=colors[lab], edgecolors="white", linewidth=0.2)

    r, p = stats.pearsonr(tmp["x"], tmp["y"])
    xgrid = np.linspace(tmp["x"].min(), tmp["x"].max(), 100)
    yhat, lo, hi = regression_ci(tmp["x"].to_numpy(float), tmp["y"].to_numpy(float), xgrid)
    ax.plot(xgrid, yhat, color="black", linewidth=1.2)
    ax.fill_between(xgrid, lo, hi, color="black", alpha=0.12)
    ax.axhline(0, color="0.65", linestyle="--", linewidth=0.7)
    ax.grid(True, alpha=0.20)

    q = pd.to_numeric(row.get("fdr_q_within_cohort_domain", row.get("fdr_q_within_cohort", np.nan)), errors="coerce")
    ax.set_title(
        f"{title_prefix}\n{CURATED_LABELS.get(row.get('curated_family'), row.get('curated_family', ''))} — {shorten(var, 22)}\n"
        f"n={len(tmp)}, r={r:.2f}, R²={r*r:.2f}, {p_text(p)}, {q_text(q)}",
        fontsize=7,
    )
    ax.set_xlabel(shorten(var, 34), fontsize=7)
    ax.set_ylabel("bias-corrected cBAG", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(fontsize=5.8, frameon=False, loc="best")
    return {"status": "plotted", "n": len(tmp), "r": float(r), "p": float(p)}


def make_main_figure(feature_set: str, selected: pd.DataFrame) -> None:
    categories = list(MAIN_COLUMNS)
    fig, axes = plt.subplots(len(COHORTS), len(categories), figsize=(17, 10.5), squeeze=False)

    for i, cohort in enumerate(COHORTS):
        for j, category in enumerate(categories):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(category, fontsize=10, fontweight="bold", pad=22)
            if j == 0:
                ax.text(-0.35, 0.5, COHORT_LABELS[cohort], transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=11, fontweight="bold")
            sub = selected[(selected["cohort"].eq(cohort)) & (selected["main_category"].eq(category))]
            if sub.empty:
                ax.text(0.5, 0.5, "No FDR-significant\nassociation", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue
            merged = pd.read_csv(MERGED_OUTDIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv", low_memory=False)
            if "_cognitive_status" not in merged.columns:
                merged["_cognitive_status"] = derive_cognitive_status(merged)
            row = sub.sort_values(["abs_pearson_r", "pearson_p"], ascending=[False, True]).iloc[0]
            plot_association_panel(ax, merged, row, f"{cohort}: {category}")

    fig.suptitle(f"Figure 5. Biological validation of bias-corrected cBAG ({MODEL_LABELS[feature_set]})", fontsize=14, y=0.99)
    fig.text(0.5, 0.012,
             "Each panel shows the strongest FDR-significant association within the indicated biological category and cohort. "
             "Points are colored by harmonized cognitive status from final harmonized metadata. Lines show linear regression fits with 95% confidence intervals.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.95])
    stem = FIGURE_OUTDIR / f"Figure5_Main_BiologicalValidation_{feature_set}"
    for fmt in FIGURE_FORMATS:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def matrix_from_best(best: pd.DataFrame, families: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mat = pd.DataFrame(index=families, columns=COHORTS, dtype=float)
    ann = pd.DataFrame("", index=families, columns=COHORTS)
    if best.empty:
        return mat, ann
    for _, row in best.iterrows():
        fam = row.get("curated_family")
        cohort = row.get("cohort")
        if fam not in mat.index or cohort not in mat.columns:
            continue
        r = float(row["pearson_r"])
        q = pd.to_numeric(row.get("fdr_q_within_cohort_domain", row.get("fdr_q_within_cohort", np.nan)), errors="coerce")
        n = int(row.get("n", 0))
        star = "***" if pd.notna(q) and q < 0.001 else "**" if pd.notna(q) and q < 0.01 else "*" if pd.notna(q) and q < 0.05 else ""
        mat.loc[fam, cohort] = r
        ann.loc[fam, cohort] = f"{r:.2f}{star}\nn={n}\n{shorten(row.get('variable'), 14)}"
    return mat, ann


def make_heatmap_figure(feature_set: str, assoc: pd.DataFrame) -> None:
    best = best_fdr_by_cohort_family(assoc)
    families = [f for f in CURATED_FAMILIES if f in set(best.get("curated_family", []))]
    if not families:
        families = CURATED_FAMILIES
    mat, ann = matrix_from_best(best, families)
    data = mat.to_numpy(dtype=float)

    fig_h = max(4.5, 0.55 * len(mat.index) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    vmax = max(0.05, np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0)
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title(f"Supplementary Figure S5{MODEL_LETTERS.get(feature_set, '')}. Curated metadata-wide cBAG associations\n{MODEL_LABELS[feature_set]}", fontsize=12, pad=12)
    ax.set_xticks(np.arange(len(COHORTS)))
    ax.set_xticklabels([COHORT_LABELS[c] for c in COHORTS], fontsize=9)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([CURATED_LABELS.get(f, f) for f in mat.index], fontsize=8)
    for i, fam in enumerate(mat.index):
        for j, cohort in enumerate(mat.columns):
            txt = ann.loc[fam, cohort]
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=6)
    ax.set_xticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Signed Pearson r", fontsize=9)
    fig.tight_layout()
    stem = FIGURE_OUTDIR / f"SupplementaryFigureS5{MODEL_LETTERS.get(feature_set, '')}_CuratedMetadataHeatmap_{feature_set}"
    for fmt in FIGURE_FORMATS:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)



# =============================================================================
# COMPLETE / NO-EMPTY-CELL SUPPLEMENTARY FIGURES
# =============================================================================

def q_sort_column(df: pd.DataFrame) -> str:
    """Primary q-value used to rank associations for figure selection."""
    for c in ["fdr_q_within_cohort_domain", "fdr_q_within_cohort", "fdr_q_all_tests_feature_set"]:
        if c in df.columns:
            return c
    return "pearson_p"


def select_domain_associations_complete(assoc: pd.DataFrame) -> pd.DataFrame:
    """
    Select one association for every cohort x main biological domain, regardless of significance.

    Selection rule:
      1) restrict to variables in that domain / families
      2) sort by q-value ascending
      3) tie-break by p-value ascending
      4) tie-break by absolute Pearson r descending

    This creates the complete supplementary figure/heatmap with no intentionally empty cells
    as long as at least one variable was tested in that domain.
    """
    if assoc.empty:
        return pd.DataFrame()

    q_col = q_sort_column(assoc)
    selected = []

    for cohort in COHORTS:
        for category, families in MAIN_COLUMNS.items():
            sub = assoc[
                (assoc["cohort"].eq(cohort)) &
                (
                    assoc["curated_family"].isin(families) |
                    assoc["testing_domain"].eq(category)
                )
            ].copy()

            if sub.empty:
                # Leave a record so the supplementary table documents why a cell could not be filled.
                selected.append({
                    "feature_set": assoc["feature_set"].iloc[0] if "feature_set" in assoc.columns and len(assoc) else "",
                    "cohort": cohort,
                    "main_category": category,
                    "selection_status": "no_tested_variable_in_domain",
                    "selection_q_column": q_col,
                })
                continue

            sub["_q_for_sort"] = pd.to_numeric(sub[q_col], errors="coerce")
            sub["_p_for_sort"] = pd.to_numeric(sub["pearson_p"], errors="coerce")
            sub["_abs_r_for_sort"] = pd.to_numeric(sub["abs_pearson_r"], errors="coerce")

            row = sub.sort_values(
                ["_q_for_sort", "_p_for_sort", "_abs_r_for_sort"],
                ascending=[True, True, False]
            ).iloc[0].copy()

            row = row.drop(labels=["_q_for_sort", "_p_for_sort", "_abs_r_for_sort"], errors="ignore")
            row["main_category"] = category
            row["selection_status"] = (
                "FDR_significant_domain"
                if pd.to_numeric(row.get("fdr_q_within_cohort_domain", np.nan), errors="coerce") < FDR_THRESHOLD
                else "not_FDR_significant_domain"
            )
            row["selection_q_column"] = q_col
            row["r2"] = pd.to_numeric(row.get("pearson_r", np.nan), errors="coerce") ** 2
            selected.append(row)

    out = pd.DataFrame(selected)
    return out


def make_complete_main_figure(feature_set: str, selected_complete: pd.DataFrame) -> None:
    """
    Supplementary scatter-panel version with the same 4 columns as main Figure 5,
    but filled with the best q-ranked association even when it is not FDR-significant.
    """
    categories = list(MAIN_COLUMNS)
    fig, axes = plt.subplots(len(COHORTS), len(categories), figsize=(17, 10.5), squeeze=False)

    for i, cohort in enumerate(COHORTS):
        for j, category in enumerate(categories):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(category, fontsize=10, fontweight="bold", pad=22)
            if j == 0:
                ax.text(
                    -0.35, 0.5, COHORT_LABELS[cohort],
                    transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=11, fontweight="bold"
                )

            sub = selected_complete[
                selected_complete["cohort"].eq(cohort) &
                selected_complete["main_category"].eq(category)
            ].copy()

            if sub.empty or sub.iloc[0].get("selection_status") == "no_tested_variable_in_domain":
                ax.text(0.5, 0.5, "No tested variable\nin this domain", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue

            merged = pd.read_csv(MERGED_OUTDIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv", low_memory=False)
            if "_cognitive_status" not in merged.columns:
                merged["_cognitive_status"] = derive_cognitive_status(merged)

            row = sub.iloc[0]
            plot_association_panel(ax, merged, row, f"{cohort}: {category}")

            # Add a small status tag so non-significant panels are not overinterpreted.
            status = str(row.get("selection_status", ""))
            q = pd.to_numeric(row.get("fdr_q_within_cohort_domain", np.nan), errors="coerce")
            tag = "domain FDR" if status == "FDR_significant_domain" else "not domain-FDR significant"
            ax.text(
                0.02, 0.02,
                tag if not np.isfinite(q) else f"{tag}\nq_domain={q:.3g}",
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=6,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.80, boxstyle="round,pad=0.2")
            )

    fig.suptitle(
        f"Supplementary Figure S5{MODEL_LETTERS.get(feature_set, '')}-complete. "
        f"Top q-ranked cBAG association per domain ({MODEL_LABELS[feature_set]})",
        fontsize=14,
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        "Each panel shows the association with the smallest domain-level FDR q-value within that cohort and biological domain. "
        "Panels are shown regardless of whether q<0.05; significance status is marked inside each panel.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.95])

    stem = FIGURE_OUTDIR / f"SupplementaryFigureS5{MODEL_LETTERS.get(feature_set, '')}_CompleteDomainPanels_{feature_set}"
    for fmt in FIGURE_FORMATS:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_complete_domain_heatmap(feature_set: str, selected_complete: pd.DataFrame) -> None:
    """
    Supplementary heatmap with one filled cell per cohort x main biological domain when possible.
    Color is signed Pearson r. Cells are selected by smallest domain-level FDR q-value.
    """
    categories = list(MAIN_COLUMNS)
    mat = pd.DataFrame(index=categories, columns=COHORTS, dtype=float)
    ann = pd.DataFrame("", index=categories, columns=COHORTS)

    if not selected_complete.empty:
        for _, row in selected_complete.iterrows():
            cohort = row.get("cohort")
            category = row.get("main_category")
            if cohort not in COHORTS or category not in categories:
                continue
            if row.get("selection_status") == "no_tested_variable_in_domain":
                ann.loc[category, cohort] = "No tested\nvariable"
                continue

            r = pd.to_numeric(row.get("pearson_r", np.nan), errors="coerce")
            p = pd.to_numeric(row.get("pearson_p", np.nan), errors="coerce")
            q = pd.to_numeric(row.get("fdr_q_within_cohort_domain", np.nan), errors="coerce")
            n = int(row.get("n", 0)) if pd.notna(row.get("n", np.nan)) else 0
            var = shorten(row.get("variable", ""), 18)

            mat.loc[category, cohort] = r

            if np.isfinite(q):
                star = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"
                qtxt = "q<1e-4" if q < 1e-4 else f"q={q:.3g}"
            else:
                star = "ns"
                qtxt = "q=NA"

            ptxt = "p<1e-4" if np.isfinite(p) and p < 1e-4 else f"p={p:.3g}" if np.isfinite(p) else "p=NA"
            ann.loc[category, cohort] = f"{r:.2f} {star}\nn={n}\n{qtxt}\n{var}"

    data = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    vmax = max(0.05, np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0)
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")

    ax.set_title(
        f"Supplementary Figure S5{MODEL_LETTERS.get(feature_set, '')}-complete heatmap. "
        f"Top q-ranked association per domain\n{MODEL_LABELS[feature_set]}",
        fontsize=12,
        pad=12,
    )
    ax.set_xticks(np.arange(len(COHORTS)))
    ax.set_xticklabels([COHORT_LABELS[c] for c in COHORTS], fontsize=9)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)

    for i, category in enumerate(categories):
        for j, cohort in enumerate(COHORTS):
            txt = ann.loc[category, cohort]
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=6)

    ax.set_xticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Signed Pearson r", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Each cell shows the tested variable with the smallest domain-level FDR q-value in that cohort/domain. "
        "Stars indicate q_domain: *<0.05, **<0.01, ***<0.001; ns = not significant.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    stem = FIGURE_OUTDIR / f"SupplementaryFigureS5{MODEL_LETTERS.get(feature_set, '')}_CompleteDomainHeatmap_{feature_set}"
    for fmt in FIGURE_FORMATS:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)





# =============================================================================
# ROC / AUC ANALYSES
# =============================================================================

AUC_TARGETS = [
    {
        "target": "APOE4_carriage",
        "label": "APOE4 carriage",
        "positive_label": "APOE4 carrier",
        "negative_label": "non-carrier",
    },
    {
        "target": "sex_female",
        "label": "Sex",
        "positive_label": "female",
        "negative_label": "male",
    },
    {
        "target": "cognitive_impairment",
        "label": "Cognitive impairment",
        "positive_label": "MCI/AD/impaired",
        "negative_label": "cognitively normal",
    },
    {
        "target": "AD_status",
        "label": "AD/dementia status",
        "positive_label": "AD/dementia",
        "negative_label": "not AD/dementia",
    },
]


def _first_col_any(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    """Case/normalization tolerant column finder used by AUC target derivation."""
    return first_existing(df, names)


def derive_apoe4_binary(df: pd.DataFrame) -> pd.Series:
    """APOE4 target. Prefer strict 3/4-or-4/4 if available, then any-carriage."""
    col = _first_col_any(df, [
        "meta__APOE4_strict_34_44_carrier", "APOE4_strict_34_44_carrier",
        "meta__APOE4_carriage", "APOE4_carriage",
        "meta__APOE4_carrier", "APOE4_carrier",
        "meta__APOE4_Positivity", "APOE4_Positivity",
    ])
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if col is not None:
        x = pd.to_numeric(df[col], errors="coerce")
        out.loc[x.eq(0)] = 0
        out.loc[x.gt(0)] = 1
        return out

    geno_col = _first_col_any(df, [
        "meta__APOE_genotype_harmonized", "APOE_genotype_harmonized",
        "meta__APOE_Label_for_table1", "APOE_Label_for_table1",
        "meta__APOE", "APOE", "meta__genotype", "genotype",
    ])
    if geno_col is not None:
        txt = df[geno_col].astype(str).str.upper().str.replace(" ", "", regex=False)
        known = ~txt.isin(["", "NAN", "NONE", "<NA>"])
        out.loc[known & txt.str.contains("4", na=False)] = 1
        out.loc[known & ~txt.str.contains("4", na=False)] = 0
    return out


def derive_sex_female_binary(df: pd.DataFrame) -> pd.Series:
    """Female vs male target. Female=1, male=0."""
    col = _first_col_any(df, [
        "meta__sex_label_for_table1", "sex_label_for_table1",
        "meta__sex_label", "sex_label",
        "meta__sex", "sex", "meta__Sex", "Sex", "meta__SEX", "SEX",
        "meta__gender", "gender", "meta__Gender", "Gender",
        "meta__PTGENDER", "PTGENDER", "meta__PTSEX", "PTSEX",
    ])
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if col is None:
        return out

    raw = df[col]
    num = pd.to_numeric(raw, errors="coerce")
    # Common codings vary, so use text first, then numeric fallback.
    txt = raw.astype(str).str.strip().str.lower()
    out.loc[txt.isin(["f", "female", "woman", "women"])] = 1
    out.loc[txt.isin(["m", "male", "man", "men"])] = 0
    # If still unknown, treat 2 as female and 1 as male, common in several datasets.
    out.loc[out.isna() & num.eq(2)] = 1
    out.loc[out.isna() & num.eq(1)] = 0
    return out


def derive_cognitive_impairment_binary(df: pd.DataFrame) -> pd.Series:
    """Cognitive impairment target: MCI or AD/dementia = 1, normal = 0."""
    out = pd.Series(np.nan, index=df.index, dtype=float)

    norm_col = _first_col_any(df, ["meta__NORMCOG_01", "NORMCOG_01"])
    mci_col = _first_col_any(df, ["meta__MCI_01", "MCI_01"])
    ad_col = _first_col_any(df, ["meta__AD_01", "AD_01"])
    dem_col = _first_col_any(df, ["meta__DEMENTIA_01", "DEMENTIA_01"])
    dx_col = _first_col_any(df, ["meta__DX_Label_harmonized", "DX_Label_harmonized"])
    comp_col = _first_col_any(df, ["meta__cognitive_impairment_composite", "cognitive_impairment_composite"])

    if norm_col is not None:
        x = pd.to_numeric(df[norm_col], errors="coerce")
        out.loc[x.eq(1)] = 0
        out.loc[x.eq(0)] = 1

    for c in [mci_col, ad_col, dem_col]:
        if c is not None:
            x = pd.to_numeric(df[c], errors="coerce")
            out.loc[x.eq(1)] = 1

    if dx_col is not None:
        dx = df[dx_col].astype(str).str.lower()
        known = ~dx.isin(["", "nan", "none", "unknown", "<na>"])
        out.loc[known & dx.str.contains("normal|normcog|control|cn", regex=True, na=False)] = 0
        out.loc[known & dx.str.contains("mci|ad|dement|impaired", regex=True, na=False)] = 1

    if comp_col is not None:
        x = pd.to_numeric(df[comp_col], errors="coerce")
        out.loc[out.isna() & x.eq(0)] = 0
        out.loc[out.isna() & x.eq(1)] = 1

    return out


def derive_ad_status_binary(df: pd.DataFrame) -> pd.Series:
    """AD/dementia target: AD/dementia = 1, non-AD known status = 0."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    ad_col = _first_col_any(df, ["meta__AD_01", "AD_01"])
    dem_col = _first_col_any(df, ["meta__DEMENTIA_01", "DEMENTIA_01"])
    norm_col = _first_col_any(df, ["meta__NORMCOG_01", "NORMCOG_01"])
    mci_col = _first_col_any(df, ["meta__MCI_01", "MCI_01"])
    dx_col = _first_col_any(df, ["meta__DX_Label_harmonized", "DX_Label_harmonized"])

    for c in [norm_col, mci_col]:
        if c is not None:
            x = pd.to_numeric(df[c], errors="coerce")
            out.loc[x.eq(1)] = 0

    for c in [ad_col, dem_col]:
        if c is not None:
            x = pd.to_numeric(df[c], errors="coerce")
            out.loc[x.eq(0) & out.isna()] = 0
            out.loc[x.eq(1)] = 1

    if dx_col is not None:
        dx = df[dx_col].astype(str).str.lower()
        known = ~dx.isin(["", "nan", "none", "unknown", "<na>"])
        out.loc[known & dx.str.contains("normal|normcog|control|cn|mci|impaired", regex=True, na=False)] = 0
        out.loc[known & dx.str.contains("ad|dement", regex=True, na=False)] = 1

    return out


def derive_auc_target(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "APOE4_carriage":
        return derive_apoe4_binary(df)
    if target == "sex_female":
        return derive_sex_female_binary(df)
    if target == "cognitive_impairment":
        return derive_cognitive_impairment_binary(df)
    if target == "AD_status":
        return derive_ad_status_binary(df)
    return pd.Series(np.nan, index=df.index, dtype=float)


def roc_curve_manual(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return FPR, TPR, and AUC for binary labels using only numpy."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return np.array([]), np.array([]), np.nan

    # Step curve at every unique threshold.
    distinct = np.r_[True, s[1:] != s[:-1]]
    idx = np.where(distinct)[0]
    tps = np.cumsum(y == 1)[idx]
    fps = np.cumsum(y == 0)[idx]
    tpr = np.r_[0, tps / pos, 1]
    fpr = np.r_[0, fps / neg, 1]
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def compute_auc_for_target(merged: pd.DataFrame, target_spec: dict, cohort: str, feature_set: str) -> tuple[dict, pd.DataFrame]:
    target = target_spec["target"]
    y = derive_auc_target(merged, target)
    score = clean_numeric(merged["_cbag"])
    tmp = pd.DataFrame({"y": y, "score": score}).replace([np.inf, -np.inf], np.nan).dropna()

    row = {
        "feature_set": feature_set,
        "cohort": cohort,
        "target": target,
        "target_label": target_spec["label"],
        "positive_label": target_spec["positive_label"],
        "negative_label": target_spec["negative_label"],
        "n": int(len(tmp)),
        "n_negative": int((tmp["y"] == 0).sum()) if not tmp.empty else 0,
        "n_positive": int((tmp["y"] == 1).sum()) if not tmp.empty else 0,
        "auc_raw": np.nan,
        "auc_oriented": np.nan,
        "score_flipped": False,
        "status": "not_computed",
    }

    if len(tmp) < MIN_AUC_N:
        row["status"] = "insufficient_n"
        return row, pd.DataFrame()
    if row["n_negative"] < MIN_AUC_CLASS_N or row["n_positive"] < MIN_AUC_CLASS_N:
        row["status"] = "insufficient_class_n"
        return row, pd.DataFrame()

    fpr, tpr, auc_raw = roc_curve_manual(tmp["y"].to_numpy(int), tmp["score"].to_numpy(float))
    if not np.isfinite(auc_raw):
        row["status"] = "auc_failed"
        return row, pd.DataFrame()

    score_for_plot = tmp["score"].to_numpy(float)
    flipped = auc_raw < 0.5
    auc_oriented = 1.0 - auc_raw if flipped else auc_raw
    if flipped:
        fpr, tpr, _ = roc_curve_manual(tmp["y"].to_numpy(int), -score_for_plot)

    row.update({
        "auc_raw": float(auc_raw),
        "auc_oriented": float(auc_oriented),
        "score_flipped": bool(flipped),
        "status": "plotted",
    })
    curve = pd.DataFrame({
        "feature_set": feature_set,
        "cohort": cohort,
        "target": target,
        "target_label": target_spec["label"],
        "fpr": fpr,
        "tpr": tpr,
        "auc_oriented": auc_oriented,
        "score_flipped": flipped,
    })
    return row, curve


def plot_auc_panel(ax: plt.Axes, merged: pd.DataFrame, target_spec: dict, cohort: str, feature_set: str) -> tuple[dict, pd.DataFrame]:
    row, curve = compute_auc_for_target(merged, target_spec, cohort, feature_set)
    label = target_spec["label"]
    if curve.empty:
        ax.text(0.5, 0.5, f"{row['status']}\nn={row['n']}\npos={row['n_positive']} neg={row['n_negative']}",
                ha="center", va="center", fontsize=8)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return row, curve

    ax.plot(curve["fpr"], curve["tpr"], linewidth=1.5)
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=0.9, color="0.5")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.22)
    ax.set_title(f"{label}\nAUC={row['auc_oriented']:.2f}, n={row['n']}", fontsize=8)
    ax.set_xlabel("False positive rate", fontsize=7)
    ax.set_ylabel("True positive rate", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    return row, curve


def make_auc_figure(feature_set: str) -> pd.DataFrame:
    """Create ROC/AUC panels for APOE4, sex, cognitive impairment, and AD status."""
    stat_rows = []
    curve_rows = []

    fig, axes = plt.subplots(len(COHORTS), len(AUC_TARGETS), figsize=(14, 10), squeeze=False)
    for i, cohort in enumerate(COHORTS):
        merged_path = MERGED_OUTDIR / f"merged_metadata_screening_{feature_set}_{cohort}.csv"
        if not merged_path.exists():
            for j, target_spec in enumerate(AUC_TARGETS):
                ax = axes[i, j]
                ax.text(0.5, 0.5, "Merged table missing", ha="center", va="center", fontsize=8)
                ax.axis("off")
            continue

        merged = pd.read_csv(merged_path, low_memory=False)
        for j, target_spec in enumerate(AUC_TARGETS):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(target_spec["label"], fontsize=10, fontweight="bold", pad=18)
            if j == 0:
                ax.text(-0.35, 0.5, COHORT_LABELS[cohort], transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=11, fontweight="bold")
            row, curve = plot_auc_panel(ax, merged, target_spec, cohort, feature_set)
            stat_rows.append(row)
            if not curve.empty:
                curve_rows.append(curve)

    fig.suptitle(
        f"cBAG ROC/AUC validation targets ({MODEL_LABELS[feature_set]})",
        fontsize=14,
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        "AUCs use bias-corrected cBAG as the score. If raw AUC was <0.5, the score direction was flipped and this is recorded in the CSV.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.95])

    stem = AUC_OUTDIR / f"Figure5_AUC_ROC_{feature_set}"
    for fmt in FIGURE_FORMATS:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    stats_df = pd.DataFrame(stat_rows)
    stats_path = AUC_OUTDIR / f"Figure5_AUC_summary_{feature_set}.csv"
    stats_df.to_csv(stats_path, index=False)
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True, sort=False).to_csv(AUC_OUTDIR / f"Figure5_AUC_ROC_curves_{feature_set}.csv", index=False)
    else:
        pd.DataFrame().to_csv(AUC_OUTDIR / f"Figure5_AUC_ROC_curves_{feature_set}.csv", index=False)

    print(f"[INFO] Saved AUC figure/statistics for {feature_set}: {stem}.png/.pdf")
    return stats_df


# =============================================================================
# PREFLIGHT DIAGNOSIS / MERGE QA
# =============================================================================

def _diagnosis_count_dict(df: pd.DataFrame, prefix: str = "") -> dict:
    """Count normal/MCI/AD/unknown using harmonized diagnosis columns."""
    def getcol(name: str):
        for c in [prefix + name, name, f"meta__{name}"]:
            if c in df.columns:
                return c
        return None

    norm_col = getcol("NORMCOG_01")
    mci_col = getcol("MCI_01")
    ad_col = getcol("AD_01")
    dem_col = getcol("DEMENTIA_01")
    dx_col = getcol("DX_Label_harmonized")

    out = {"n_rows": int(len(df)), "n_normal": 0, "n_mci": 0, "n_ad_dementia": 0, "n_impaired_total": 0, "n_unknown": 0}

    if norm_col is not None:
        norm = pd.to_numeric(df[norm_col], errors="coerce")
        out["n_normal"] = int(norm.eq(1).sum())
    if mci_col is not None:
        mci = pd.to_numeric(df[mci_col], errors="coerce")
        out["n_mci"] = int(mci.eq(1).sum())
    if ad_col is not None or dem_col is not None:
        ad = pd.to_numeric(df[ad_col], errors="coerce") if ad_col is not None else pd.Series(0, index=df.index)
        dem = pd.to_numeric(df[dem_col], errors="coerce") if dem_col is not None else pd.Series(0, index=df.index)
        out["n_ad_dementia"] = int((ad.eq(1) | dem.eq(1)).sum())

    out["n_impaired_total"] = int(out["n_mci"] + out["n_ad_dementia"])

    if dx_col is not None:
        dx = df[dx_col].fillna("Unknown").astype(str).str.lower()
        out["n_unknown"] = int(dx.isin(["unknown", "nan", "none", ""]).sum())
    return out


def run_preflight_diagnosis_qa() -> pd.DataFrame:
    """Compare harmonized metadata diagnosis counts to validation+cBAG merge counts."""
    rows = []
    print("\n" + "=" * 100)
    print("PREFLIGHT: MCI/AD RECOVERY FROM HARMONIZED METADATA")
    print("=" * 100)

    for fs in FEATURE_SETS:
        for cohort in COHORTS:
            mpath = metadata_path(cohort)
            vpath = validation_path(cohort, fs)
            base = {"cohort": cohort, "feature_set": fs, "metadata_path": str(mpath), "validation_path": str(vpath)}

            if not mpath.exists():
                rows.append({**base, "status": "missing_metadata"})
                print(f"[WARN] {fs} | {cohort}: missing metadata {mpath}")
                continue
            if not vpath.exists():
                rows.append({**base, "status": "missing_validation"})
                print(f"[WARN] {fs} | {cohort}: missing validation {vpath}")
                continue

            meta = pd.read_csv(mpath, low_memory=False)
            meta_counts = _diagnosis_count_dict(meta)

            try:
                merged, qa = load_and_merge(cohort, fs)
                with_cbag = merged["_cbag"].notna()
                merged_cbag = merged.loc[with_cbag].copy()
                merged_counts = {
                    "n_cbag_rows": int(with_cbag.sum()),
                    "n_cbag_normal": int((merged_cbag["_cognitive_status"] == "Cognitively normal").sum()),
                    "n_cbag_impaired": int((merged_cbag["_cognitive_status"] == "Cognitively impaired").sum()),
                    "n_cbag_unknown": int((merged_cbag["_cognitive_status"] == "Unknown status").sum()),
                }
                status = "ok"
                warning = ""
                if meta_counts["n_impaired_total"] > 0 and merged_counts["n_cbag_impaired"] == 0:
                    warning = "WARNING_metadata_has_MCI_AD_but_cBAG_merge_has_no_impaired"
                elif merged_counts["n_cbag_unknown"] > 0:
                    warning = "WARNING_some_cBAG_rows_missing_status"

                row = {
                    **base,
                    "status": status,
                    "metadata_rows": meta_counts["n_rows"],
                    "metadata_normal": meta_counts["n_normal"],
                    "metadata_mci": meta_counts["n_mci"],
                    "metadata_ad_dementia": meta_counts["n_ad_dementia"],
                    "metadata_impaired_total": meta_counts["n_impaired_total"],
                    **merged_counts,
                    "matched_metadata_rows": qa.get("matched_metadata_rows", np.nan),
                    "unmatched_validation_rows": qa.get("unmatched_validation_rows", np.nan),
                    "validation_key_col": qa.get("validation_key_col", ""),
                    "metadata_key_col": qa.get("metadata_key_col", ""),
                    "warning": warning,
                }
                rows.append(row)

                if warning or fs == MAIN_FEATURE_SET:
                    print(
                        f"{fs:28s} | {cohort:9s} | "
                        f"metadata N/MCI/AD={meta_counts['n_normal']}/{meta_counts['n_mci']}/{meta_counts['n_ad_dementia']} | "
                        f"cBAG normal/impaired/unknown={merged_counts['n_cbag_normal']}/{merged_counts['n_cbag_impaired']}/{merged_counts['n_cbag_unknown']} | "
                        f"matched={qa.get('matched_metadata_rows', 0)} | {warning}"
                    )

            except Exception as exc:
                rows.append({**base, "status": f"merge_error: {exc}", **{f"metadata_{k}": v for k, v in meta_counts.items()}})
                print(f"[ERROR] {fs} | {cohort}: {exc}")

    out = pd.DataFrame(rows)
    QA_OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = QA_OUTDIR / "Figure5_preflight_MCI_AD_recovery_all_models.csv"
    out.to_csv(out_path, index=False)
    print("\nPreflight MCI/AD recovery QA written to:")
    print(out_path)

    bad = out[out.get("warning", "").astype(str).str.len().gt(0)] if "warning" in out.columns else pd.DataFrame()
    if not bad.empty:
        print("\n[IMPORTANT] Potential missing MCI/AD recovery detected:")
        show = [c for c in ["feature_set", "cohort", "metadata_mci", "metadata_ad_dementia", "n_cbag_normal", "n_cbag_impaired", "n_cbag_unknown", "warning"] if c in bad.columns]
        print(bad[show].to_string(index=False))
    return out

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Figure 5 from harmonized metadata and full-cohort cBAG validation outputs."
    )
    parser.add_argument(
        "--validation-dir-name",
        default=VALIDATION_DIR_NAME,
        help=(
            "Validation output folder inside each ablation directory. "
            "Use validation_figures_full_cohort for the full-cohort Figure 5. "
            "Use validation_figures for the older OOF version."
        ),
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    global VALIDATION_DIR_NAME
    args = parse_args()
    VALIDATION_DIR_NAME = args.validation_dir_name

    for d in [OUTDIR, MERGED_OUTDIR, FIGURE_OUTDIR, QA_OUTDIR, AUC_OUTDIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FIGURE 5 FROM FINAL HARMONIZED METADATA")
    print("=" * 100)
    print("Harmonized metadata:", HARMONIZED_DIR)
    print("Validation root:", RESULTS_ROOT)
    print("Validation directory name:", VALIDATION_DIR_NAME)
    print("Biological validation root:", BIOVALIDATION_ROOT)
    print("Output:", OUTDIR)
    print("Height/weight exclusion: ON — excludes columns containing height/heigh/VSHEI/weight/weigh/VSWEI/VSWT; keeps any column containing BMI and body_mass_index.")
    print("Merge policy: validation rows left-joined to final harmonized metadata by connectome/session key; no row-order fallback.")

    preflight_df = run_preflight_diagnosis_qa()

    all_qa = []
    manifest_rows = []
    all_auc = []
    all_count_domain_frames = []
    all_count_cohort_frames = []
    all_inventory_frames = []
    all_input_path_frames = []

    for fs in FEATURE_SETS:
        assoc, qa = run_screening_for_feature_set(fs)
        all_qa.append(qa)
        count_domain_path = QA_OUTDIR / f"Figure5_association_test_counts_by_domain_{fs}.csv"
        count_cohort_path = QA_OUTDIR / f"Figure5_association_test_counts_by_cohort_{fs}.csv"
        inventory_path = QA_OUTDIR / f"Figure5_candidate_variable_inventory_{fs}.csv"
        input_paths_path = QA_OUTDIR / f"Figure5_input_paths_validation_and_metadata_{fs}.csv"
        if count_domain_path.exists():
            all_count_domain_frames.append(pd.read_csv(count_domain_path))
        if count_cohort_path.exists():
            all_count_cohort_frames.append(pd.read_csv(count_cohort_path))
        if inventory_path.exists():
            all_inventory_frames.append(pd.read_csv(inventory_path))
        if input_paths_path.exists():
            all_input_path_frames.append(pd.read_csv(input_paths_path))
        selected = select_main_associations(assoc)
        selected.to_csv(FIGURE_OUTDIR / f"Figure5_SelectedAssociations_{fs}.csv", index=False)
        make_heatmap_figure(fs, assoc)

        # Complete supplementary outputs: one q-ranked association per cohort x domain, regardless of significance.
        selected_complete = select_domain_associations_complete(assoc)
        selected_complete.to_csv(
            FIGURE_OUTDIR / f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_CompleteDomainSelectedAssociations_{fs}.csv",
            index=False
        )
        make_complete_main_figure(fs, selected_complete)
        make_complete_domain_heatmap(fs, selected_complete)

        auc_df = make_auc_figure(fs)
        if not auc_df.empty:
            all_auc.append(auc_df)
        manifest_rows.append({"figure": f"Figure5_AUC_ROC_{fs}", "feature_set": fs, "path_stem": str(AUC_OUTDIR / f"Figure5_AUC_ROC_{fs}")})

        if fs == MAIN_FEATURE_SET:
            make_main_figure(fs, selected)
            manifest_rows.append({"figure": "Figure5_Main", "feature_set": fs, "path_stem": str(FIGURE_OUTDIR / f"Figure5_Main_BiologicalValidation_{fs}")})
        else:
            # Also make model-specific main-style panels as supplementary checks.
            make_main_figure(fs, selected)
            manifest_rows.append({"figure": f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_main_style", "feature_set": fs, "path_stem": str(FIGURE_OUTDIR / f"Figure5_Main_BiologicalValidation_{fs}")})

        manifest_rows.append({"figure": f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_heatmap", "feature_set": fs, "path_stem": str(FIGURE_OUTDIR / f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_CuratedMetadataHeatmap_{fs}")})
        manifest_rows.append({"figure": f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_complete_domain_panels", "feature_set": fs, "path_stem": str(FIGURE_OUTDIR / f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_CompleteDomainPanels_{fs}")})
        manifest_rows.append({"figure": f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_complete_domain_heatmap", "feature_set": fs, "path_stem": str(FIGURE_OUTDIR / f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_CompleteDomainHeatmap_{fs}")})
        manifest_rows.append({"figure": f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_complete_domain_selected_table", "feature_set": fs, "path_stem": str(FIGURE_OUTDIR / f"SupplementaryFigureS5{MODEL_LETTERS.get(fs, '')}_CompleteDomainSelectedAssociations_{fs}.csv")})

    all_qa_df = pd.concat(all_qa, ignore_index=True, sort=False) if all_qa else pd.DataFrame()
    all_qa_df.to_csv(QA_OUTDIR / "Figure5_merge_QA_all_models.csv", index=False)
    if all_count_domain_frames:
        pd.concat(all_count_domain_frames, ignore_index=True, sort=False).to_csv(
            QA_OUTDIR / "Figure5_association_test_counts_by_domain_all_models.csv", index=False
        )
    if all_count_cohort_frames:
        pd.concat(all_count_cohort_frames, ignore_index=True, sort=False).to_csv(
            QA_OUTDIR / "Figure5_association_test_counts_by_cohort_all_models.csv", index=False
        )
    if all_inventory_frames:
        pd.concat(all_inventory_frames, ignore_index=True, sort=False).to_csv(
            QA_OUTDIR / "Figure5_candidate_variable_inventory_all_models_all_cohorts.csv", index=False
        )
    if all_input_path_frames:
        pd.concat(all_input_path_frames, ignore_index=True, sort=False).to_csv(
            QA_OUTDIR / "Figure5_input_paths_validation_and_metadata.csv", index=False
        )
    if all_auc:
        pd.concat(all_auc, ignore_index=True, sort=False).to_csv(AUC_OUTDIR / "Figure5_AUC_summary_all_models.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(FIGURE_OUTDIR / "Figure5_manifest.csv", index=False)

    print("\n[DONE]")
    print("Main outputs:")
    print(FIGURE_OUTDIR)
    print("QA:")
    print(QA_OUTDIR / "Figure5_merge_QA_all_models.csv")
    print("Association test counts by cohort:")
    print(QA_OUTDIR / "Figure5_association_test_counts_by_cohort_all_models.csv")
    print("Association test counts by domain:")
    print(QA_OUTDIR / "Figure5_association_test_counts_by_domain_all_models.csv")
    print("Candidate variable inventory:")
    print(QA_OUTDIR / "Figure5_candidate_variable_inventory_all_models_all_cohorts.csv")
    print("Input paths used:")
    print(QA_OUTDIR / "Figure5_input_paths_validation_and_metadata.csv")
    print("AUC summary:")
    print(AUC_OUTDIR / "Figure5_AUC_summary_all_models.csv")
    print("\nInspect status recovery with:")
    print(f"column -s, -t {QA_OUTDIR / 'Figure5_merge_QA_all_models.csv'} | less -S")


if __name__ == "__main__":
    main()
