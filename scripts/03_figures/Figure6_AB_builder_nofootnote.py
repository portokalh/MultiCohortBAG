# FINAL PATCH NOTES:
# - DEDICATED_OBSERVED_BMI_RESOLVER: all generic BMI candidates are replaced by one
#   cleaned observed kg/m^2 BMI candidate per cohort; plausible range is 10-80 kg/m^2.
# - BMI_PLAUSIBLE_RANGE_CLEANING: observed BMI candidates are cleaned to 10-80 kg/m^2;
#   z-scored/standardized BMI and sentinels such as -9999 are excluded from BMI association/QA.
# - HABS_BMI_OBSERVED_PRIORITY: HABS BMI now prioritizes observed kg/m^2 columns
#   (OM_BMI_clinical/OM_BMI) before z-scored or standardized BMI copies.
# - BMI_QA_HEAD: writes and prints selected BMI candidate summaries and first paired cBAG-BMI rows for all cohorts.
# - COMPACT_PANEL_A_LABELS: main heatmap cells show only r and FDR stars/NS;
#   variable names, q-values, p-values, and N remain in source tables and supplementary plots.
# - AD-DECODE transcriptomic PCs are retained in full/supplementary exploratory
#   transcriptomic-PC domain outputs, so PC17 appears in the supplementary
#   exploratory FDR-domain heatmap under "Transcriptomic PCs".
# - The same PC rows are suppressed only from the main selected
#   "Exploratory molecular" cell to avoid duplicating PC17 next to the curated
#   molecular/transcriptomic column.
# - AD-DECODE BMI selection prioritizes metadata/observed BMI.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure6build_FINAL_twoCog_twoFluid_directionality.py

Final patched Figure 6 builder with two-tier cognition, two-tier molecular validation, ADRC BPAVG columns, molecular de-duplication, and directionality panels.

Inputs
------
Existing Figure 6 directory, for example:
  /mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_biological_validation_20260615

Expected input files:
  merged_tables/merged_metadata_screening_<FEATURE_SET>_<COHORT>.csv
  metadata_variable_associations_<FEATURE_SET>.csv
  auc/Figure6_AUC_summary_<FEATURE_SET>.csv   [optional; AUCs are recomputed if needed]

Main outputs
------------
  Figure6A_multidomain_top_hit_FDRfirst_NS_<FEATURE_SET>.png/pdf
  Figure6B_AUC_APOE4_CogImpairment_Sex_<FEATURE_SET>.png/pdf
  Figure6_FINAL_combined_<FEATURE_SET>.png/pdf

Selection rule
--------------
For each cohort x domain:
  1. Select the FDR-significant candidate with largest |r| if any exists.
  2. Otherwise select the top candidate by |r| and mark the cell NS.

Main domains
------------
  - Cognition
  - Fluid biomarkers / transcriptomic PCs
  - BMI
  - Metabolic / lipids / inflammation
  - Cardiovascular

Important patch
---------------
Hippocampus / HC / brain-volume / graph-network variables are not classified as
vascular/cardiovascular. They are written to separate imaging/hippocampus QA.

AUCs
----
AUCs are recomputed directly from merged tables for:
  - APOE4
  - Cognitive impairment
  - Sex

Usage
-----
OUT=/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure6_biological_validation_20260615

python Figure6build.py \
  --figure6-dir "$OUT" \
  --feature-set imaging_only \
  --fluid-main-min-n 20 \
  --supp-top-k 25
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats


COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]
COHORT_LABELS = {"ADNI": "ADNI", "ADRC": "ADRC", "HABS": "HABS", "AD_DECODE": "AD-DECODE"}

MAIN_DOMAINS = [
    "Cognition composites",
    "Cognition tests",
    "Curated fluid / transcriptomic",
    "Exploratory molecular",
    "Imaging / hippocampus",
    "BMI",
    "Cardiovascular",
]

QA_DOMAINS = [
    "Imaging / hippocampus / network QA",
]

CBAG_PRIORITY = [
    "cBAG_raw_clean",
    "_cbag",
    "cBAG_oof_global_raw_clean",
    "cBAG_oof_global",
    "cBAG_bias_corrected",
    "cBAG_BiasCorrected",
    "cBAG_foldwise_raw_clean",
    "cBAG_foldwise",
    "cBAG",
]

COG_COLS = {
    # Five prespecified domain composites for primary cognition summary.
    # Global cognition is not included here because the Figure 6 cognition
    # composite column is intended to represent five comparable domains.
    "Memory": "Memory_Composite",
    "Executive": "Executive_Function_Composite",
    "Processing speed": "Processing_Speed_Composite",
    "Language": "Language_Composite",
    "Visuospatial": "Visuospatial_Composite",
}

BAD_COGNITIVE_COL_RE = re.compile(
    r"n_domains|n_tests|n_available|availability|available|count|num_|number|"
    r"diagnosis|dx_|status|subject|visit|age|sex|education|apoe",
    re.I,
)

COGNITIVE_GATE_RE = re.compile(
    r"cog|memory|recall|recognition|executive|speed|language|visuo|spatial|"
    r"trail|trails|tmt|stroop|fluency|naming|clock|figure|digit|symbol|"
    r"avlt|ravlt|cvlt|bnt|mmse|moca|cdr|adas|attention",
    re.I,
)

COGNITIVE_TEST_DOMAIN_PATTERNS = {
    "Memory": re.compile(
        r"memory|mem_|recall|recognition|delayed|immediate|logical|story|word.*list|"
        r"ravglt|ravlt|avlt|cvlt|lm_|wms|cdr.*memory|cdmemory",
        re.I,
    ),
    "Executive": re.compile(
        r"executive|trail.*b|trails.*b|tmt.?b|switch|stroop|inhibit|"
        r"set.?shift|digit.*back|working.*memory|letter.*fluency|phonemic|fascat|fas",
        re.I,
    ),
    "Processing speed": re.compile(
        r"processing.*speed|speed|trail.*a|trails.*a|tmt.?a|digit.*symbol|"
        r"coding|symbol|reaction|psychomotor",
        re.I,
    ),
    "Language": re.compile(
        r"language|naming|boston|bnt|vocab|vocabulary|semantic|animal|"
        r"fluency|category|word.*fluency|ffluency",
        re.I,
    ),
    "Visuospatial": re.compile(
        r"visuospatial|visuo|spatial|clock|figure|copy|rey|benson|block|"
        r"construction|cube|draw",
        re.I,
    ),
}

AGE_ALIASES = [
    "age", "Age", "AGE", "age_at_scan", "Age_at_scan", "age_years",
    "chronological_age", "Chronological_Age", "scan_age", "AgeAtMRI",
    "age_at_mri", "Age_bl", "AGE_BL",
]
SEX_ALIASES = [
    "sex", "Sex", "SEX", "gender", "Gender", "GENDER",
    "PTGENDER", "biological_sex", "Biological_Sex",
]
EDU_ALIASES = [
    "education", "Education", "EDUCATION", "edu", "Edu", "EDU",
    "education_years", "years_education", "PTEDUCAT", "Educ", "educ",
]

SENTINELS = [
    -999999, -888888, -777777, -99999, -88888, -77777,
    -9999, -8888, -7777, -999, -888, -777,
    999, 888, 777, 9999, 8888, 7777, 99999, 88888, 77777,
]


# =============================================================================
# Basic utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Build final patched Figure 6.")
    p.add_argument("--figure6-dir", required=True)
    p.add_argument("--feature-set", default="imaging_only")
    p.add_argument("--outdir-name", default="Figure6_final_patched_build")
    p.add_argument("--cbag-column", default="auto")
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--min-n", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--supp-top-k", type=int, default=25,
                   help="Number of supplementary exploratory hits to retain per cohort/category.")
    p.add_argument("--fluid-main-min-n", type=int, default=50,
                   help="For the main fluid cell, prefer FDR hits with at least this N if available.")
    p.add_argument("--max-transcriptomic-pc", type=int, default=20,
                   help="Highest AD-DECODE transcriptomic PC to include in the curated main fluid scan.")
    p.add_argument("--skip-volcano-plots", action="store_true",
                   help="Skip supplementary volcano plots.")
    p.add_argument("--save-component-figures", action="store_true",
                   help="Keep standalone Figure 6A and Figure 6B outputs even after the combined figure is generated.")
    p.add_argument("--show-panel-a-agreement-footnote", action="store_true",
                   help="Draw the cross-cohort FDR agreement footnote under Figure 6A. Default: omit it for manuscript readability.")
    p.add_argument("--volcano-alpha", type=float, default=0.05,
                   help="Nominal p-value threshold line for volcano plots.")
    p.add_argument("--volcano-label-top", type=int, default=8,
                   help="Maximum number of labels to draw per volcano plot.")
    p.add_argument("--domain-heatmap-topk-list", default="10,30",
                   help="Comma-separated top-k values for domain-level hit heatmaps, e.g. 10,30.")
    p.add_argument("--cognitive-mode", choices=["resid", "raw"], default="resid")
    p.add_argument("--cognitive-scope", choices=["all", "normal_only"], default="all")
    p.add_argument("--force-onfly-resid", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--auc-bootstrap-n", type=int, default=5000,
                   help="Number of stratified bootstrap replicates for Figure 6B AUC confidence intervals.")
    p.add_argument("--auc-bootstrap-seed", type=int, default=42,
                   help="Random seed for stratified bootstrap AUC confidence intervals.")
    p.add_argument("--skip-bmi-histograms", action="store_true",
                   help="Skip per-cohort observed BMI histogram QA plots.")
    return p.parse_args()


def norm(x) -> str:
    s = "" if pd.isna(x) else str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def short(x, n=18) -> str:
    s = "" if pd.isna(x) else str(x)
    s = re.sub(r"^(meta__|biomarkers__|biomarkers_csv__)", "", s)
    s = s.replace("_z_clean", "").replace("_raw_clean", "")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def fmt(x) -> str:
    if pd.isna(x):
        return "NA"
    x = float(x)
    return f"{x:.1e}" if abs(x) < 1e-4 else f"{x:.3f}"


def sig_label(q, alpha=0.05) -> str:
    if pd.isna(q):
        return "NS"
    q = float(q)
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < alpha:
        return "*"
    return "NS"


def find_col(df: pd.DataFrame, candidates, contains=None):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    nmap = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in nmap:
            return nmap[norm(c)]
    if contains:
        for c in df.columns:
            lc = str(c).lower()
            if all(x.lower() in lc for x in contains):
                return c
    return None


def clean_numeric(s):
    out = pd.to_numeric(s, errors="coerce")
    for val in SENTINELS:
        out = out.mask(out == val, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


# PATCH_BMI_PLAUSIBLE_RANGE
def clean_bmi_numeric(s, min_bmi=10.0, max_bmi=80.0):
    """
    Clean observed BMI values. Keeps plausible kg/m^2 values and removes
    sentinel/standardized/out-of-range values (e.g., -9999, -0.07, z-scores).
    """
    x = clean_numeric(s)
    return x.where((x >= min_bmi) & (x <= max_bmi))


def bmi_column_is_observed_kgm2(s, min_n=10):
    """
    Heuristic to decide whether a candidate BMI column looks like observed kg/m^2,
    not a z-scored/standardized copy.
    """
    x_raw = clean_numeric(s)
    x = clean_bmi_numeric(s)
    n = int(x.notna().sum())
    if n < min_n:
        return False
    med = float(x.median(skipna=True))
    sd = float(x.std(skipna=True))
    uniq = int(x.nunique(dropna=True))
    # Observed BMI should have adult-like median and enough spread.
    return (uniq >= 3) and (15 <= med <= 45) and (sd > 1.0)



def bh_fdr(pvals):
    p = pd.to_numeric(pvals, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    if ok.sum() == 0:
        return q
    vals = p.loc[ok].to_numpy(float)
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    qrank = ranked * m / (np.arange(m) + 1)
    qrank = np.minimum.accumulate(qrank[::-1])[::-1]
    qrank = np.minimum(qrank, 1.0)
    out = np.empty_like(qrank)
    out[order] = qrank
    q.loc[ok] = out
    return q


def corr_assoc(x, y):
    x = clean_numeric(x)
    y = clean_numeric(y)
    z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    out = {
        "n": int(len(z)),
        "n_unique_x": int(z["x"].nunique()) if len(z) else 0,
        "n_unique_y": int(z["y"].nunique()) if len(z) else 0,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
    }
    if len(z) < 3 or out["n_unique_x"] < 3 or out["n_unique_y"] < 3:
        return out
    r, p = stats.pearsonr(z["x"], z["y"])
    slope, intercept = np.polyfit(z["x"], z["y"], 1)
    out.update({
        "pearson_r": float(r),
        "pearson_p": float(p),
        "slope": float(slope),
        "intercept": float(intercept),
    })
    return out


def get_cbag_col(df: pd.DataFrame, requested="auto") -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise KeyError(f"Requested cBAG column not found: {requested}")
        return requested
    c = find_col(df, CBAG_PRIORITY)
    if c is None:
        raise KeyError("Could not identify cBAG column")
    return c


def read_merged(fig6_dir: Path, feature_set: str, cohort: str) -> pd.DataFrame:
    f = fig6_dir / "merged_tables" / f"merged_metadata_screening_{feature_set}_{cohort}.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    return pd.read_csv(f, low_memory=False)


# =============================================================================
# Cognitive composite candidates
# =============================================================================

def cognitive_status(df):
    if "_cognitive_status" in df.columns:
        return df["_cognitive_status"].fillna("Unknown status")
    out = pd.Series("Unknown status", index=df.index, dtype="object")
    dx = find_col(df, ["meta__DX_Label_harmonized", "DX_Label_harmonized"])
    normc = find_col(df, ["meta__NORMCOG_01", "NORMCOG_01"])
    mcic = find_col(df, ["meta__MCI_01", "MCI_01"])
    adc = find_col(df, ["meta__AD_01", "AD_01"])
    demc = find_col(df, ["meta__DEMENTIA_01", "DEMENTIA_01"])
    if normc:
        x = pd.to_numeric(df[normc], errors="coerce")
        out.loc[x.eq(1)] = "Cognitively normal"
        out.loc[x.eq(0)] = "Cognitively impaired"
    for c in [mcic, adc, demc]:
        if c:
            x = pd.to_numeric(df[c], errors="coerce")
            out.loc[x.eq(1)] = "Cognitively impaired"
    if dx:
        s = df[dx].astype(str).str.lower()
        out.loc[s.str.contains("normal|control|cn", regex=True, na=False)] = "Cognitively normal"
        out.loc[s.str.contains("mci|ad|dement|impaired", regex=True, na=False)] = "Cognitively impaired"
    return out


def auto_covariates(df):
    out = []
    for group in [AGE_ALIASES, SEX_ALIASES, EDU_ALIASES]:
        c = find_col(df, group)
        if c and c not in out:
            out.append(c)
    return out


def design_matrix(df, covs):
    parts = [pd.Series(1.0, index=df.index, name="Intercept")]
    for c in covs:
        raw = df[c]
        num = pd.to_numeric(raw, errors="coerce")
        if num.notna().sum() >= max(10, int(0.5 * len(df))):
            parts.append(num.rename(c))
        else:
            d = pd.get_dummies(raw.astype("string").fillna("Missing"), prefix=c, drop_first=True, dtype=float)
            if d.shape[1] > 0:
                parts.append(d)
    return pd.concat(parts, axis=1)


def residualize(y, df, covs):
    y = clean_numeric(y)
    if not covs:
        return y - y.mean(skipna=True), "demeaned", int(y.notna().sum())
    X = design_matrix(df, covs)
    work = pd.concat([y.rename("_y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if len(work) < max(10, X.shape[1] + 3):
        return out, "resid_failed_insufficient_complete_cases", int(len(work))
    yv = work["_y"].to_numpy(float)
    Xm = work.drop(columns=["_y"]).to_numpy(float)
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    out.loc[work.index] = yv - Xm @ beta
    return out, "onfly_resid_" + "+".join(covs), int(len(work))


def composite_column_preferences(base, mode):
    """Column preference for cognitive composites."""
    if mode == "raw":
        return [
            base + "_raw_clean", "meta__" + base + "_raw_clean",
            base, "meta__" + base,
            base + "_z_clean", "meta__" + base + "_z_clean",
            base + "_resid", "meta__" + base + "_resid",
        ]
    return [
        base + "_resid", "meta__" + base + "_resid",
        base + "_z_clean", "meta__" + base + "_z_clean",
        base + "_raw_clean", "meta__" + base + "_raw_clean",
        base, "meta__" + base,
    ]


def is_cognitive_composite_column(col):
    """True for the five prespecified cognitive domain composites."""
    cn = norm(col)
    if BAD_COGNITIVE_COL_RE.search(cn):
        return False
    composite_terms = [norm(v) for v in COG_COLS.values()]
    return any(term in cn for term in composite_terms)


def classify_cognitive_test_column(col):
    """Classify non-composite individual cognitive test columns into 5 domains."""
    c = str(col)
    cn = norm(c)

    if BAD_COGNITIVE_COL_RE.search(cn):
        return None
    if is_cognitive_composite_column(c):
        return None
    if not COGNITIVE_GATE_RE.search(cn):
        return None

    hits = []
    for dom, pat in COGNITIVE_TEST_DOMAIN_PATTERNS.items():
        if pat.search(c) or pat.search(cn):
            hits.append(dom)

    if not hits:
        return None

    # Resolve common fluency ambiguity.
    if "Language" in hits and "Executive" in hits:
        if re.search(r"animal|semantic|category|ffluency", cn, re.I):
            return "Language"
        if re.search(r"letter|phonemic|fas", cn, re.I):
            return "Executive"

    return hits[0]


def cognitive_variant_priority(col):
    """Lower is preferred for duplicate cognitive encodings."""
    c = str(col).lower()
    if c.endswith("_resid") or "_resid__onfly" in c:
        return 0
    if c.endswith("_z_clean") or c.endswith("_z"):
        return 1
    if c.endswith("_raw_clean"):
        return 2
    return 3


def canonical_cognitive_key(col):
    """Collapse meta/raw/z/resid duplicate versions of the same cognitive measure."""
    s = str(col)
    s = re.sub(r"^(meta__|clinical__|neurobat__|cognitive__)", "", s)
    s = re.sub(r"(_resid__onfly|_resid|_z_clean|_raw_clean|_clean|_z)$", "", s)
    s = re.sub(r"_[xy]$", "", s)
    return norm(s)


def _finalize_cognitive_rows(rows, fdr_group_cols, feature_set, outpath):
    out = pd.DataFrame(rows)
    if out.empty:
        out.to_csv(outpath, index=False)
        return out
    out["pearson_r"] = pd.to_numeric(out["pearson_r"], errors="coerce")
    out["pearson_p"] = pd.to_numeric(out["pearson_p"], errors="coerce")
    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    out["fdr_q"] = np.nan
    valid = out["tested"].fillna(False) & out["pearson_p"].notna()
    if valid.any():
        for _, idx in out.loc[valid].groupby(fdr_group_cols).groups.items():
            out.loc[idx, "fdr_q"] = bh_fdr(out.loc[idx, "pearson_p"])
    out["is_fdr"] = pd.to_numeric(out["fdr_q"], errors="coerce") < 0.05
    out["abs_r"] = out["pearson_r"].abs()
    out["candidate_status"] = np.where(out["is_fdr"], "FDR", "NS")
    out.to_csv(outpath, index=False)
    return out


def compute_cognitive_candidates(fig6_dir, feature_set, args, outdir):
    """
    Two-tier cognition resolver for Figure 6.

    Main Figure 6 receives two separate cognition columns:
      1. Cognition composites: five prespecified domain composites.
      2. Cognition tests: exploratory individual cognitive tests, excluding composites.

    Duplicate/availability/count/metadata columns are excluded. For duplicate
    raw/z/residualized encodings of the same test, one preferred version is kept.
    """
    comp_rows, test_rows, inv_rows = [], [], []

    for cohort in COHORTS:
        df0 = read_merged(fig6_dir, feature_set, cohort)
        cbag = get_cbag_col(df0, args.cbag_column)

        # ------------------------------------------------------------------
        # Tier 1: five prespecified cognitive domain composites.
        # ------------------------------------------------------------------
        for subdomain, raw_col in COG_COLS.items():
            df = df0.copy()
            selected_col = None
            source = ""
            covstr = ""
            n_resid = 0

            preferred = composite_column_preferences(raw_col, args.cognitive_mode)
            preferred = [c for c in preferred if not BAD_COGNITIVE_COL_RE.search(norm(c))]
            selected_col = find_col(df, preferred)

            if selected_col:
                source = "existing_column"
                if selected_col.endswith("_resid"):
                    source = "existing_resid"
                elif selected_col.endswith("_z_clean"):
                    source = "existing_z_clean"
                elif selected_col.endswith("_raw_clean") or selected_col.endswith(raw_col):
                    source = "existing_raw_or_clean"
            elif args.cognitive_mode == "resid":
                raw_existing = find_col(df, [raw_col, "meta__" + raw_col, raw_col + "_raw_clean", "meta__" + raw_col + "_raw_clean"])
                if raw_existing and not BAD_COGNITIVE_COL_RE.search(norm(raw_existing)):
                    covs = auto_covariates(df)
                    selected_col = raw_col + "_resid__onfly"
                    df[selected_col], source, n_resid = residualize(df[raw_existing], df, covs)
                    covstr = ",".join(covs)
                else:
                    source = "missing_composite"
            else:
                source = "missing_composite"

            inv_rows.append({
                "cohort": cohort,
                "tier": "Cognition composites",
                "subdomain": subdomain,
                "raw_column": raw_col,
                "selected_column": selected_col,
                "endpoint_source": source,
                "resid_covariates": covstr,
                "n_residualization_complete_cases": n_resid,
                "n_nonmissing_selected": int(clean_numeric(df[selected_col]).notna().sum()) if selected_col else 0,
            })

            if not selected_col:
                comp_rows.append({
                    "source": "cognitive_composite_resolver",
                    "cohort": cohort,
                    "main_domain": "Cognition composites",
                    "subdomain": subdomain,
                    "variable": "",
                    "clean": "",
                    "domain": "",
                    "family": "Cognitive composite",
                    "family_label": subdomain,
                    "display_label": subdomain,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "fdr_q": np.nan,
                    "n": 0,
                    "tested": False,
                    "is_fdr": False,
                    "candidate_status": "missing",
                    "endpoint_source": source,
                    "resid_covariates": covstr,
                    "cognitive_tier": "composite",
                    "cognitive_key": raw_col,
                    "variant_priority": np.nan,
                })
                continue

            dscope = df.copy()
            if args.cognitive_scope == "normal_only":
                dscope["_cognitive_status_tmp"] = cognitive_status(dscope)
                dscope = dscope[dscope["_cognitive_status_tmp"].eq("Cognitively normal")].copy()

            a = corr_assoc(dscope[selected_col], dscope[cbag])
            tested = bool(a["n"] >= args.min_n and a["n_unique_x"] >= 3 and a["n_unique_y"] >= 3)
            comp_rows.append({
                "source": "cognitive_composite_resolver",
                "cohort": cohort,
                "main_domain": "Cognition composites",
                "subdomain": subdomain,
                "variable": selected_col,
                "clean": norm(selected_col),
                "domain": "",
                "family": "Cognitive composite",
                "family_label": subdomain,
                "display_label": subdomain,
                "pearson_r": a["pearson_r"],
                "pearson_p": a["pearson_p"],
                "fdr_q": np.nan,
                "n": a["n"],
                "tested": tested,
                "is_fdr": False,
                "candidate_status": "tested" if tested else "not_tested",
                "endpoint_source": source,
                "resid_covariates": covstr,
                "slope": a["slope"],
                "intercept": a["intercept"],
                "cognitive_tier": "composite",
                "cognitive_key": canonical_cognitive_key(selected_col),
                "variant_priority": cognitive_variant_priority(selected_col),
            })

        # ------------------------------------------------------------------
        # Tier 2: exploratory individual cognitive tests.
        # ------------------------------------------------------------------
        raw_test_rows = []
        for col in df0.columns:
            subdomain = classify_cognitive_test_column(col)
            if subdomain is None or str(col) == cbag:
                continue
            x = clean_numeric(df0[col])
            a = corr_assoc(df0[col], df0[cbag])
            tested = bool(a["n"] >= args.min_n and a["n_unique_x"] >= 3 and a["n_unique_y"] >= 3)
            key = canonical_cognitive_key(col)
            raw_test_rows.append({
                "source": "cognitive_test_resolver",
                "cohort": cohort,
                "main_domain": "Cognition tests",
                "subdomain": subdomain,
                "variable": str(col),
                "clean": norm(col),
                "domain": "",
                "family": "Individual cognitive test",
                "family_label": subdomain,
                "display_label": short(col, 22),
                "pearson_r": a["pearson_r"],
                "pearson_p": a["pearson_p"],
                "fdr_q": np.nan,
                "n": a["n"],
                "n_unique_x": a["n_unique_x"],
                "tested": tested,
                "is_fdr": False,
                "candidate_status": "tested" if tested else "not_tested",
                "endpoint_source": "individual_test_column",
                "resid_covariates": "",
                "slope": a["slope"],
                "intercept": a["intercept"],
                "cognitive_tier": "individual_test",
                "cognitive_key": key,
                "variant_priority": cognitive_variant_priority(col),
                "classification_reason": "cognitive_test_pattern_5domain",
            })
            inv_rows.append({
                "cohort": cohort,
                "tier": "Cognition tests",
                "subdomain": subdomain,
                "raw_column": str(col),
                "selected_column": str(col),
                "endpoint_source": "individual_test_column",
                "resid_covariates": "",
                "n_residualization_complete_cases": 0,
                "n_nonmissing_selected": int(x.notna().sum()),
                "n_unique_selected": int(x.nunique(dropna=True)),
                "examples": " | ".join(df0[col].dropna().astype(str).head(5).tolist()),
            })

        # De-duplicate individual tests within cohort/key before FDR.
        if raw_test_rows:
            tmp = pd.DataFrame(raw_test_rows)
            tmp["_bad_p"] = pd.to_numeric(tmp["pearson_p"], errors="coerce").isna()
            tmp["_abs_r"] = pd.to_numeric(tmp["pearson_r"], errors="coerce").abs()
            tmp["_n"] = pd.to_numeric(tmp["n"], errors="coerce")
            tmp = tmp.sort_values(
                ["cognitive_key", "_bad_p", "variant_priority", "_n", "_abs_r"],
                ascending=[True, True, True, False, False],
            )
            tmp = tmp.groupby("cognitive_key", as_index=False, group_keys=False).head(1)
            test_rows.extend(tmp.drop(columns=["_bad_p", "_abs_r", "_n"]).to_dict("records"))

    comp = _finalize_cognitive_rows(
        comp_rows,
        ["cohort", "main_domain"],
        feature_set,
        outdir / f"QA_cognitive_composite_candidates_{feature_set}.csv",
    )
    tests = _finalize_cognitive_rows(
        test_rows,
        ["cohort", "main_domain"],
        feature_set,
        outdir / f"QA_cognitive_individual_test_candidates_{feature_set}.csv",
    )

    # Extra FDR within each cognitive test subdomain for supplementary reporting.
    if not tests.empty:
        tests["fdr_q_within_cohort_subdomain"] = np.nan
        valid = tests["tested"].fillna(False) & pd.to_numeric(tests["pearson_p"], errors="coerce").notna()
        for _, idx in tests.loc[valid].groupby(["cohort", "subdomain"]).groups.items():
            tests.loc[idx, "fdr_q_within_cohort_subdomain"] = bh_fdr(tests.loc[idx, "pearson_p"])
        tests.to_csv(outdir / f"QA_cognitive_individual_test_candidates_{feature_set}.csv", index=False)

        top_subdomain_rows = []
        for (cohort, subdomain), g in tests.groupby(["cohort", "subdomain"]):
            g = g[pd.to_numeric(g["pearson_r"], errors="coerce").notna()].copy()
            if g.empty:
                continue
            fdr = g[g["fdr_q_within_cohort_subdomain"] < args.fdr_alpha].copy()
            if not fdr.empty:
                pick = fdr.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
                pick["subdomain_selection_rule"] = "subdomain_FDR_then_max_abs_r"
            else:
                pick = g.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
                pick["subdomain_selection_rule"] = "subdomain_NS_fallback_max_abs_r"
            top_subdomain_rows.append(pick)
        pd.DataFrame(top_subdomain_rows).to_csv(
            outdir / f"Supplementary_cognitive_tests_top_per_5domain_{feature_set}.csv",
            index=False,
        )

    pd.DataFrame(inv_rows).to_csv(outdir / f"QA_cognitive_column_inventory_{feature_set}.csv", index=False)
    cog = pd.concat([comp, tests], ignore_index=True, sort=False) if not tests.empty else comp
    cog.to_csv(outdir / f"QA_cognitive_two_tier_candidates_{feature_set}.csv", index=False)
    return cog


# =============================================================================
# Metadata association table standardization and classification
# =============================================================================

def infer_assoc_columns(df, q_arg="auto"):
    c = {}
    c["cohort"] = find_col(df, ["cohort", "Cohort"])
    c["variable"] = find_col(df, ["variable", "endpoint", "column", "original_variable", "variable_original", "source_column"])
    c["clean"] = find_col(df, ["variable_clean", "clean_variable", "endpoint_clean", "column_clean", "variable_norm"])
    c["domain"] = find_col(df, ["testing_domain", "domain", "Domain"])
    c["family"] = find_col(df, ["curated_family", "family", "variable_family"])
    c["family_label"] = find_col(df, ["curated_family_label", "family_label", "label", "variable_label", "display_label"])
    c["r"] = find_col(df, ["pearson_r", "r", "correlation", "rho"])
    c["p"] = find_col(df, ["pearson_p", "p_value", "p", "pval"])
    c["n"] = find_col(df, ["n_complete", "n", "n_used", "N"])
    if q_arg != "auto":
        c["q"] = q_arg if q_arg in df.columns else None
    else:
        c["q"] = find_col(df, [
            "fdr_q_within_cohort_domain",
            "q_within_cohort_domain",
            "fdr_within_cohort_domain",
            "q_domain",
            "fdr_q_within_cohort_family",
            "fdr_q_within_cohort",
            "fdr_q_all_tests_feature_set",
            "fdr_q",
            "q_value",
        ])
    return c


def classify_row(row):
    """
    Fixed curated classification.

    Key correction:
      hippocampus / HC / graph / connectome / brain volume are NOT vascular.
    """
    text_raw = " ".join(str(row.get(k, "")) for k in [
        "variable", "clean", "domain", "family", "family_label", "display_label"
    ])
    text = norm(text_raw)

    # Imaging/hippocampus QA domain first, to avoid HC/hippocampus entering vascular.
    if re.search(
        r"hippocamp|hippo|\bhc\b|hc_fa|hc_md|hc_rd|brain_volume|total_brain|tbv|icv|"
        r"clustering|cluster|path_length|pathlength|global_eff|local_eff|efficiency|"
        r"graph|connectome|network|fractional_anisotropy|\bfa\b|\bmd\b|\brd\b",
        text,
        re.I,
    ):
        return "Imaging / hippocampus", "pattern_imaging_hippocampus_network"

    # BMI is its own main-panel domain.
    # This captures BMI itself and ADNI vrf_bmi_* fields.
    if re.search(
        r"(^|_)bmi($|_)|body_mass_index|body.?mass.?index|vrf_bmi|obes|adipos|weight",
        text,
        re.I,
    ):
        return "BMI", "pattern_bmi_adiposity"

    # Metabolic / lipids / inflammation, excluding BMI/weight because those are handled above.
    if re.search(
        r"diab|diabetes|glucose|fasting_glucose|hba1c|\ba1c\b|hemoglobin_a1c|"
        r"insulin|homa|triglycer|cholesterol|\bhdl\b|\bldl\b|lipid|"
        r"metabolic|metformin|hypogly|dyslip|crp|tnf|interleukin|\bil\b|"
        r"il_?\d+|cytokine|chemokine|mcp|ifn|interferon|inflamm|immune|"
        r"sicam|svcam|platelet|wbc|rbc|hemoglobin|creatinine|albumin",
        text,
        re.I,
    ):
        return "Metabolic / lipids / inflammation", "pattern_metabolic_lipids_inflammation"

    # Fluid biomarkers / transcriptomic PCs.
    if re.search(
        r"ab40|ab42|abeta|a_beta|amyloid|ptau|p_tau|ttau|t_tau|tau|nfl|nf_l|gfap|"
        r"\bpc\d+\b|transcriptomic|rna|gene|expression",
        text,
        re.I,
    ):
        return "Fluid biomarkers / transcriptomic PCs", "pattern_fluid_transcriptomic"

    # Vascular/cardiovascular. No generic "hip" token here.
    if re.search(
        r"vascular|cardio|heart|\bbp\b|blood_pressure|systolic|diastolic|\bsbp\b|\bdbp\b|bpsys|bpdia|"
        r"hypertension|hyperten|pulse|stroke|infarct|ischemi|ischaemi|vit.*shiel",
        text,
        re.I,
    ):
        return "Cardiovascular", "pattern_vascular_cardiovascular"

    dom = str(row.get("domain", ""))
    if re.search(r"vascular|cardio", dom, re.I):
        return "Cardiovascular", "fallback_existing_vascular"
    return None, "unclassified"


def load_standardized_metadata_associations(fig6_dir, feature_set, args, outdir):
    f = fig6_dir / f"metadata_variable_associations_{feature_set}.csv"
    if not f.exists():
        raise FileNotFoundError(f)

    raw = pd.read_csv(f, low_memory=False)
    cols = infer_assoc_columns(raw)
    missing = [k for k in ["cohort", "variable", "domain", "r", "p"] if not cols.get(k)]
    if missing:
        raise ValueError(f"Could not infer association columns {missing}. Available columns: {list(raw.columns)}")

    out = pd.DataFrame()
    out["source"] = "metadata_screen"
    out["cohort"] = raw[cols["cohort"]].astype(str)
    out["variable"] = raw[cols["variable"]].astype(str)
    out["clean"] = raw[cols["clean"]].astype(str) if cols.get("clean") else out["variable"].map(norm)
    out["domain"] = raw[cols["domain"]].astype(str)
    out["family"] = raw[cols["family"]].astype(str) if cols.get("family") else ""
    out["family_label"] = raw[cols["family_label"]].astype(str) if cols.get("family_label") else out["family"]
    out["display_label"] = out["family_label"].where(
        out["family_label"].astype(str).ne("") & out["family_label"].astype(str).ne("nan"),
        out["variable"],
    ).map(short)
    out["pearson_r"] = pd.to_numeric(raw[cols["r"]], errors="coerce")
    out["pearson_p"] = pd.to_numeric(raw[cols["p"]], errors="coerce")
    out["n"] = pd.to_numeric(raw[cols["n"]], errors="coerce") if cols.get("n") else np.nan

    if cols.get("q") and cols["q"] in raw.columns:
        out["fdr_q"] = pd.to_numeric(raw[cols["q"]], errors="coerce")
        out["fdr_source"] = cols["q"]
    else:
        out["fdr_q"] = bh_fdr(out["pearson_p"])
        out["fdr_source"] = "fallback_BH_all_rows"

    cls = out.apply(classify_row, axis=1, result_type="expand")
    out["main_domain"] = cls[0]
    out["classification_reason"] = cls[1]
    out["tested"] = True
    out["is_fdr"] = pd.to_numeric(out["fdr_q"], errors="coerce") < args.fdr_alpha
    out["abs_r"] = pd.to_numeric(out["pearson_r"], errors="coerce").abs()
    out["candidate_status"] = np.where(out["is_fdr"], "FDR", "NS")

    out.to_csv(outdir / f"QA_metadata_associations_standardized_classified_{feature_set}.csv", index=False)
    return out



# =============================================================================
# Direct merged-table scans, curated fluid markers, and supplementary discovery
# =============================================================================

DIRECT_EXCLUDE = re.compile(
    r"source_columns|filepath|filename|path|merge_key|subject_key|session_key|"
    r"label|status|diagnosis|cohort|feature_set|index|unnamed|cbag|brainage|"
    r"predicted|chronological|auc|target",
    re.I,
)

AB40_RE = re.compile(r"abeta40|abeta_40|a_beta_40|ab40|\babeta40\b|\bab40\b", re.I)
AB42_RE = re.compile(r"abeta42|abeta_42|a_beta_42|ab42|\babeta42\b|\bab42\b", re.I)
PTAU_RE = re.compile(r"ptau|p_tau|p.?tau181|p.?tau217|p.?tau231", re.I)
TTAU_RE = re.compile(r"ttau|t_tau|total.?tau|total_tau|\btau\b", re.I)
GFAP_RE = re.compile(r"gfap", re.I)
NFL_RE = re.compile(r"\bnfl\b|nf_l|neurofilament", re.I)
PC_Z_RE = re.compile(r"^PC([1-9][0-9]?)_z_clean$", re.I)

BIOMARKER_FALSE_POSITIVE_RE = re.compile(
    r"genetic_|bcvol|pbmcvol|rnavol|apvolume|source_columns|filepath|filename|path",
    re.I,
)


def clean_domain_numeric(s, colname="", domain=""):
    """
    Numeric cleaner with domain-specific physiological filters.

    - BMI/adiposity: masks impossible BMI/weight values.
    - Vascular: masks impossible raw pulse/SBP/DBP values.
    - Metabolic/lipids/inflammation: masks negative values for raw biomarkers
      where negative concentrations are not meaningful.
    """
    x = clean_numeric(s)
    cn = str(colname).lower()
    is_z = cn.endswith("_z_clean") or "_z_" in cn

    if domain == "BMI":
        if not is_z:
            if re.search(r"(^|_)bmi($|_)|body.?mass.?index", cn):
                x = x.mask((x <= 0) | (x < 10) | (x > 80), np.nan)
            if re.search(r"weight|vsweight|om_weight", cn):
                x = x.mask((x <= 0) | (x < 20) | (x > 400), np.nan)

    if domain == "Cardiovascular":
        if not is_z:
            if re.search(r"(^|_)pulse(\d|_|$)|(^|_)om_pulse", cn):
                x = x.mask((x < 30) | (x > 220), np.nan)
            if re.search(r"systolic|(^|_)sbp($|_)|bpsys|bp_sys", cn):
                x = x.mask((x < 70) | (x > 260), np.nan)
            if re.search(r"diastolic|(^|_)dbp($|_)|bpdia|bp_dia", cn):
                x = x.mask((x < 30) | (x > 160), np.nan)

    if domain == "Metabolic / lipids / inflammation":
        if not is_z:
            if re.search(
                r"glucose|hba1c|a1c|insulin|homa|triglycer|cholesterol|hdl|ldl|"
                r"lipid|crp|tnf|interleukin|(^|_)il_?\d|mcp|ifn|cytokine|"
                r"sicam|svcam|albumin|creatinine|platelet|wbc|rbc|hemoglobin",
                cn,
            ):
                x = x.mask(x < 0, np.nan)
    return x


def classify_merged_domain(col):
    """Direct merged-table domain classification for non-cognition and non-fluid main domains."""
    c = str(col)
    t = norm(c)

    if re.search(
        r"hippocamp|hippo|\bhc\b|hc_fa|hc_md|hc_rd|brain_volume|total_brain|tbv|icv|"
        r"clustering|cluster|path_length|pathlength|global_eff|local_eff|efficiency|"
        r"graph|connectome|network|fractional_anisotropy|\bfa\b|\bmd\b|\brd\b",
        t,
        re.I,
    ):
        return "Imaging / hippocampus", "direct_imaging_hippocampus_network"

    # Observed BMI only. Allow true BMI fields, including PHC BMI,
    # but reject vrf_bmi namespace variables that encode age/risk factors.
    if (
        re.search(r"(^|_)bmi($|_)|body_mass_index|body.?mass.?index|obes|adipos", t, re.I)
        or re.search(r"(^|_)vrf_bmi_phc_bmi($|_|raw|z)", t, re.I)
        or re.search(r"(^|_)meta_vrf_bmi_phc_bmi($|_|raw|z)", t, re.I)
    ) and not re.search(
        r"age_cardiovascularrisk|cardiovascularrisk|cvrscore|hypertension|diabetes|heart|"
        r"diagnosis|sex|education|race|ethnicity|visit|weight|vsweight",
        t,
        re.I,
    ):
        return "BMI", "direct_bmi_observed_priority"

    if re.search(
        r"diab|diabetes|glucose|fasting_glucose|hba1c|\ba1c\b|hemoglobin_a1c|"
        r"insulin|homa|triglycer|cholesterol|\bhdl\b|\bldl\b|lipid|"
        r"metabolic|metformin|hypogly|dyslip|crp|tnf|interleukin|\bil\b|"
        r"il_?\d+|cytokine|chemokine|mcp|ifn|interferon|inflamm|immune|"
        r"sicam|svcam|platelet|wbc|rbc|hemoglobin|creatinine|albumin|"
        r"msd_serum|qtx_plasma",
        t,
        re.I,
    ):
        return "Metabolic / lipids / inflammation", "direct_metabolic_lipids_inflammation"

    # Avoid LUMIPULSE being read as "pulse". Main cardiovascular candidates are
    # restricted to systolic BP, diastolic BP, and pulse / pulse pressure.
    if re.search(
        r"\bbp\b|blood_pressure|systolic|diastolic|\bsbp\b|\bdbp\b|bpsys|bpdia|"
        r"(^|_)pulse(\d|_|$)|(^|_)om_pulse|pulse_pressure|vspulse|vsbpsys|vsbpdia",
        t,
        re.I,
    ):
        return "Cardiovascular", "direct_cardiovascular_vitals"

    return None, "unclassified"



# PATCH_ADDECODE_COMPUTED_BMI_FROM_HEIGHT_WEIGHT
def _candidate_numeric_cols(df, include_re, reject_re=None):
    """Return plausible numeric columns whose names match include_re and not reject_re."""
    out = []
    for c in df.columns:
        s = str(c)
        if not re.search(include_re, s, re.I):
            continue
        if reject_re and re.search(reject_re, s, re.I):
            continue
        x = clean_numeric(df[c])
        if int(x.notna().sum()) >= 10 and int(x.nunique(dropna=True)) >= 3:
            out.append(c)
    return out


def _unit_converted_weight_kg(x):
    """Return plausible kg Series from a raw weight-like column."""
    x = clean_numeric(x)
    med = x.median(skipna=True)
    candidates = []
    if pd.notna(med):
        # Already kg: most adult kg values fall roughly 35-200.
        if 25 <= med <= 220:
            candidates.append(("kg", x))
        # Pounds: adult pounds typically 80-500.
        if 80 <= med <= 500:
            candidates.append(("lb_to_kg", x / 2.2046226218))
    return candidates


def _unit_converted_height_m(x):
    """Return plausible meter Series from a raw height-like column."""
    x = clean_numeric(x)
    med = x.median(skipna=True)
    candidates = []
    if pd.notna(med):
        # Meters.
        if 1.2 <= med <= 2.2:
            candidates.append(("m", x))
        # Centimeters.
        if 120 <= med <= 220:
            candidates.append(("cm_to_m", x / 100.0))
        # Inches.
        if 48 <= med <= 85:
            candidates.append(("in_to_m", x * 0.0254))
    return candidates


def compute_addecode_bmi_from_height_weight(df):
    """
    Compute BMI = weight_kg / height_m^2 for AD_DECODE if plausible height and weight
    columns are present. Chooses the pair/unit conversion with largest valid N and
    plausible BMI distribution.
    """
    weight_cols = _candidate_numeric_cols(
        df,
        r"(^|[^A-Za-z0-9])(weight|wt|body_weight|bodyweight|kg|lbs?|pounds?)([^A-Za-z0-9]|$)|(^|_)weight(_|$)",
        r"brain|edge|matrix|volume|weighted|weighting|birth|sample|tube|plate|batch"
    )
    height_cols = _candidate_numeric_cols(
        df,
        r"(^|[^A-Za-z0-9])(height|ht|body_height|bodyheight|stature|cm|inch|inches)([^A-Za-z0-9]|$)|(^|_)height(_|$)",
        r"brain|edge|matrix|volume|sample|tube|plate|batch"
    )

    best = None
    audit = []
    for wc in weight_cols:
        for hc in height_cols:
            for wu, wkg in _unit_converted_weight_kg(df[wc]):
                for hu, hm in _unit_converted_height_m(df[hc]):
                    bmi = wkg / (hm ** 2)
                    bmi = bmi.where((bmi >= 10) & (bmi <= 80))
                    n = int(bmi.notna().sum())
                    uniq = int(bmi.nunique(dropna=True))
                    med = float(bmi.median(skipna=True)) if n else np.nan
                    sd = float(bmi.std(skipna=True)) if n else np.nan
                    ok = n >= 10 and uniq >= 3 and pd.notna(med) and 15 <= med <= 45 and pd.notna(sd) and sd > 1
                    audit.append({
                        "weight_col": wc, "height_col": hc, "weight_unit": wu, "height_unit": hu,
                        "n": n, "n_unique": uniq, "median_bmi": med, "sd_bmi": sd, "plausible": ok
                    })
                    if not ok:
                        continue
                    score = (n, uniq, -abs(med - 27))
                    if best is None or score > best[0]:
                        best = (score, bmi, audit[-1])
    if best is None:
        return None, pd.DataFrame(audit)
    return best[1], pd.DataFrame(audit)


def direct_merged_domain_scan(fig6_dir, feature_set, args, outdir):
    """Scan merged metadata for BMI/metabolic, vascular, and imaging/network variables."""
    rows, inv = [], []
    for cohort in COHORTS:
        df = read_merged(fig6_dir, feature_set, cohort)
        cbag = get_cbag_col(df, args.cbag_column)
        y = clean_numeric(df[cbag])
        # For AD_DECODE, compute observed BMI from height and weight when possible.
        # This avoids selecting standardized/lowercase BMI encodings as the displayed BMI variable.
        if cohort == "AD_DECODE":
            bmi_hw, bmi_audit = compute_addecode_bmi_from_height_weight(df)
            try:
                bmi_audit.to_csv(outdir / f"QA_AD_DECODE_BMI_from_height_weight_audit_{feature_set}.csv", index=False)
            except Exception:
                pass
            if bmi_hw is not None:
                a_bmi = corr_assoc(bmi_hw, y)
                inv.append({
                    "cohort": cohort,
                    "column": "BMI_from_height_weight",
                    "main_domain": "BMI",
                    "classification_reason": "computed_from_height_weight",
                    "n_numeric": int(bmi_hw.notna().sum()),
                    "n_unique": int(bmi_hw.nunique(dropna=True)),
                })
                if a_bmi["n"] >= args.min_n and a_bmi["n_unique_x"] >= 3 and a_bmi["n_unique_y"] >= 3:
                    rows.append({
                        "source": "computed_bmi_from_height_weight",
                        "cohort": cohort,
                        "main_domain": "BMI",
                        "subdomain": "",
                        "variable": "BMI_from_height_weight",
                        "clean": "bmi_from_height_weight",
                        "domain": "",
                        "family": "",
                        "family_label": "",
                        "display_label": "BMI",
                        "pearson_r": a_bmi["pearson_r"],
                        "pearson_p": a_bmi["pearson_p"],
                        "fdr_q": np.nan,
                        "n": a_bmi["n"],
                        "tested": True,
                        "is_fdr": False,
                        "candidate_status": "tested",
                        "classification_reason": "computed_from_height_weight",
                        "slope": a_bmi["slope"],
                        "intercept": a_bmi["intercept"],
                    })


        for col in df.columns:
            if str(col) == cbag or DIRECT_EXCLUDE.search(str(col)):
                continue
            dom, reason = classify_merged_domain(col)
            if dom is None:
                continue

            x = clean_domain_numeric(df[col], colname=col, domain=dom)
            a = corr_assoc(x, y)
            inv.append({
                "cohort": cohort,
                "column": col,
                "main_domain": dom,
                "classification_reason": reason,
                "n_numeric": int(x.notna().sum()),
                "n_unique": int(x.nunique(dropna=True)),
            })
            if not (a["n"] >= args.min_n and a["n_unique_x"] >= 3 and a["n_unique_y"] >= 3):
                continue
            rows.append({
                "source": "direct_merged_scan",
                "cohort": cohort,
                "main_domain": dom,
                "subdomain": "",
                "variable": str(col),
                "clean": norm(col),
                "domain": "",
                "family": "",
                "family_label": "",
                "display_label": short(col, 22),
                "pearson_r": a["pearson_r"],
                "pearson_p": a["pearson_p"],
                "fdr_q": np.nan,
                "n": a["n"],
                "tested": True,
                "is_fdr": False,
                "candidate_status": "tested",
                "classification_reason": reason,
                "slope": a["slope"],
                "intercept": a["intercept"],
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        for (_, _), idx in out.groupby(["cohort", "main_domain"]).groups.items():
            out.loc[idx, "fdr_q"] = bh_fdr(out.loc[idx, "pearson_p"])
        out["is_fdr"] = pd.to_numeric(out["fdr_q"], errors="coerce") < args.fdr_alpha
        out["abs_r"] = pd.to_numeric(out["pearson_r"], errors="coerce").abs()
        out["candidate_status"] = np.where(out["is_fdr"], "FDR", "NS")
    pd.DataFrame(inv).to_csv(outdir / f"QA_direct_merged_column_inventory_{feature_set}.csv", index=False)
    out.to_csv(outdir / f"QA_direct_merged_candidates_{feature_set}.csv", index=False)
    return out


def fluid_base_marker(col):
    s = str(col)
    s = re.sub(r"^meta__", "", s)
    s = re.sub(r"^biomarkers__", "", s)
    s = re.sub(r"^biomarkers_csv__", "", s)
    s = re.sub(r"_raw_clean$", "", s)
    s = re.sub(r"_z_clean$", "", s)
    s = re.sub(r"_clean$", "", s)
    return s


def fluid_variant_priority(col):
    c = str(col)
    score = 0
    if c.startswith("meta__"):
        score += 10
    if c.startswith("biomarkers_csv__"):
        score += 5
    if c.endswith("_raw_clean"):
        score += 1
    if c.endswith("_z_clean"):
        score += 3
    return score


def molecular_variant_priority(col):
    """Priority used to collapse duplicate raw/z/meta molecular columns before FDR.

    Higher is preferred. This prevents the same biological assay from being
    counted multiple times when both raw, z-clean, and meta__ copies exist.
    """
    c = str(col)
    score = 0
    # Prefer standardized numeric columns for single markers.
    if c.endswith("_z_clean"):
        score += 40
    elif c.endswith("_raw_clean"):
        score += 30
    else:
        score += 20
    # Prefer biomarker-source columns over duplicated meta__ aliases.
    if c.startswith("meta__"):
        score -= 10
    if c.startswith("biomarkers_csv__") or c.startswith("biomarkers__"):
        score += 5
    return score


def molecular_dedup_key(row):
    family = str(row.get("marker_family", row.get("subdomain", "")))
    base = str(row.get("base_marker", row.get("variable", "")))
    base = re.sub(r"^meta__", "", base)
    base = re.sub(r"^biomarkers__", "", base)
    base = re.sub(r"^biomarkers_csv__", "", base)
    base = re.sub(r"_raw_clean$|_z_clean$|_clean$", "", base)
    return norm(f"{family}__{base}")


def deduplicate_molecular_variants(out, outdir=None, feature_set=None, label="molecular"):
    """Collapse duplicated molecular rows before FDR correction.

    Multiple pTau/GFAP/NfL/Aβ entries often arise from raw, z-clean, and
    meta__ aliases of the same assay. FDR is therefore computed after this
    de-duplication step, using one representative row per cohort and biological
    marker key.
    """
    if out is None or out.empty:
        return out
    x = out.copy()
    x["dedup_marker_key"] = x.apply(molecular_dedup_key, axis=1)
    x["dedup_variant_priority"] = x["variable"].map(molecular_variant_priority)
    x["abs_r"] = pd.to_numeric(x["pearson_r"], errors="coerce").abs()
    x["n"] = pd.to_numeric(x["n"], errors="coerce")
    x["pearson_p"] = pd.to_numeric(x["pearson_p"], errors="coerce")
    pre = x.copy()
    x = x.sort_values(
        ["cohort", "dedup_marker_key", "dedup_variant_priority", "n", "pearson_p", "abs_r"],
        ascending=[True, True, False, False, True, False],
    )
    x = x.groupby(["cohort", "dedup_marker_key"], as_index=False, group_keys=False).head(1).copy()
    x["dedup_n_removed_same_marker"] = 0
    counts = pre.groupby(["cohort", "dedup_marker_key"]).size().rename("n_variants_before_dedup").reset_index()
    x = x.merge(counts, on=["cohort", "dedup_marker_key"], how="left")
    x["dedup_n_removed_same_marker"] = x["n_variants_before_dedup"].fillna(1).astype(int) - 1
    if outdir is not None and feature_set is not None:
        pre.to_csv(Path(outdir) / f"QA_{label}_candidates_pre_dedup_{feature_set}.csv", index=False)
        counts.to_csv(Path(outdir) / f"QA_{label}_dedup_variant_counts_{feature_set}.csv", index=False)
    return x


def curated_marker_rows(df, y, cohort, pat, family, label, args):
    rows = []
    for col in df.columns:
        c = str(col)
        if DIRECT_EXCLUDE.search(c) or BIOMARKER_FALSE_POSITIVE_RE.search(c):
            continue
        if not pat.search(c):
            continue
        if family == "Total tau" and PTAU_RE.search(c):
            continue
        a = corr_assoc(df[col], y)
        if not (a["n"] >= args.min_n and a["n_unique_x"] >= 3 and a["n_unique_y"] >= 3):
            continue
        rows.append({
            "source": "curated_fluid_resolver",
            "cohort": cohort,
            "main_domain": "Curated fluid / transcriptomic",
            "subdomain": family,
            "variable": c,
            "clean": norm(c),
            "domain": "",
            "family": "Curated fluid marker",
            "family_label": family,
            "display_label": label,
            "pearson_r": a["pearson_r"],
            "pearson_p": a["pearson_p"],
            "fdr_q": np.nan,
            "n": a["n"],
            "tested": True,
            "is_fdr": False,
            "candidate_status": "tested",
            "classification_reason": "curated_fluid_marker",
            "marker_family": family,
            "base_marker": fluid_base_marker(c),
            "variant_priority": fluid_variant_priority(c),
            "slope": a["slope"],
            "intercept": a["intercept"],
        })
    return rows


def curated_amyloid_ratio_rows(df, y, cohort, args):
    rows = []
    ab40 = [c for c in df.columns if (not DIRECT_EXCLUDE.search(str(c))) and (not BIOMARKER_FALSE_POSITIVE_RE.search(str(c))) and AB40_RE.search(str(c))]
    ab42 = [c for c in df.columns if (not DIRECT_EXCLUDE.search(str(c))) and (not BIOMARKER_FALSE_POSITIVE_RE.search(str(c))) and AB42_RE.search(str(c))]
    for c42 in ab42:
        for c40 in ab40:
            # Aβ42/Aβ40 is a biological ratio and should be computed on raw-scale
            # values. Do not mix z-clean and raw values, and do not ratio two
            # standardized variables. Raw-clean aliases are acceptable.
            if "_z_clean" in str(c42).lower() or "_z_clean" in str(c40).lower():
                continue
            x42 = clean_numeric(df[c42])
            x40 = clean_numeric(df[c40])
            ratio = x42 / x40.replace(0, np.nan)
            a = corr_assoc(ratio, y)
            if not (a["n"] >= args.min_n and a["n_unique_x"] >= 3 and a["n_unique_y"] >= 3):
                continue
            b42, b40 = fluid_base_marker(c42), fluid_base_marker(c40)
            source_bonus = -5 if re.sub("42", "", b42) == re.sub("40", "", b40) else 0
            rows.append({
                "source": "curated_fluid_resolver",
                "cohort": cohort,
                "main_domain": "Curated fluid / transcriptomic",
                "subdomain": "Amyloid ratio",
                "variable": f"{c42} / {c40}",
                "clean": f"{norm(c42)}_over_{norm(c40)}",
                "domain": "",
                "family": "Curated fluid marker",
                "family_label": "Amyloid ratio",
                "display_label": "Aβ42/Aβ40",
                "pearson_r": a["pearson_r"],
                "pearson_p": a["pearson_p"],
                "fdr_q": np.nan,
                "n": a["n"],
                "tested": True,
                "is_fdr": False,
                "candidate_status": "tested",
                "classification_reason": "curated_computed_amyloid_ratio",
                "marker_family": "Amyloid ratio",
                "base_marker": f"{b42} / {b40}",
                "variant_priority": fluid_variant_priority(c42) + fluid_variant_priority(c40) + source_bonus,
                "slope": a["slope"],
                "intercept": a["intercept"],
            })
    return rows


def curated_fluid_scan(fig6_dir, feature_set, args, outdir):
    """Curated fluid panel for main Figure 6: Aβ42/Aβ40, pTau, total tau, GFAP, NfL; AD-DECODE PC1-PC{max_transcriptomic_pc}."""
    rows = []
    for cohort in COHORTS:
        df = read_merged(fig6_dir, feature_set, cohort)
        cbag = get_cbag_col(df, args.cbag_column)
        y = clean_numeric(df[cbag])

        if cohort == "AD_DECODE":
            max_pc = int(getattr(args, "max_transcriptomic_pc", 20))
            for col in df.columns:
                mpc = PC_Z_RE.fullmatch(str(col))
                if not mpc:
                    continue
                pc_num = int(mpc.group(1))
                if pc_num > max_pc:
                    continue
                a = corr_assoc(df[col], y)
                if not (a["n"] >= args.min_n and a["n_unique_x"] >= 3 and a["n_unique_y"] >= 3):
                    continue
                label = str(col).replace("_z_clean", "")
                rows.append({
                    "source": "curated_fluid_resolver",
                    "cohort": cohort,
                    "main_domain": "Curated fluid / transcriptomic",
                    "subdomain": "Transcriptomic PC",
                    "variable": str(col),
                    "clean": norm(col),
                    "domain": "",
                    "family": "Transcriptomic PC",
                    "family_label": "Transcriptomic PC",
                    "display_label": label,
                    "pearson_r": a["pearson_r"],
                    "pearson_p": a["pearson_p"],
                    "fdr_q": np.nan,
                    "n": a["n"],
                    "tested": True,
                    "is_fdr": False,
                    "candidate_status": "tested",
                    "classification_reason": f"curated_AD_DECODE_PC1_{max_pc}_z_clean",
                    "marker_family": "Transcriptomic PC",
                    "base_marker": label,
                    "variant_priority": 0,
                    "slope": a["slope"],
                    "intercept": a["intercept"],
                })
            continue

        rows.extend(curated_amyloid_ratio_rows(df, y, cohort, args))
        rows.extend(curated_marker_rows(df, y, cohort, PTAU_RE, "pTau", "pTau", args))
        rows.extend(curated_marker_rows(df, y, cohort, TTAU_RE, "Total tau", "Total tau", args))
        rows.extend(curated_marker_rows(df, y, cohort, GFAP_RE, "GFAP", "GFAP", args))
        rows.extend(curated_marker_rows(df, y, cohort, NFL_RE, "NfL", "NfL", args))

    out = pd.DataFrame(rows)
    if not out.empty:
        out = deduplicate_molecular_variants(out, outdir, feature_set, label="fluid_curated")
        out["abs_r"] = pd.to_numeric(out["pearson_r"], errors="coerce").abs()
        for cohort, idx in out.groupby("cohort").groups.items():
            out.loc[idx, "fdr_q"] = bh_fdr(out.loc[idx, "pearson_p"])
        out["is_fdr"] = pd.to_numeric(out["fdr_q"], errors="coerce") < args.fdr_alpha
        out["candidate_status"] = np.where(out["is_fdr"], "FDR", "NS")
    out.to_csv(outdir / f"QA_fluid_curated_candidates_used_by_Figure6_{feature_set}.csv", index=False)
    if not out.empty:
        counts = out.groupby(["cohort", "marker_family"]).agg(
            n_candidate_columns=("variable", "nunique"),
            n_fdr=("is_fdr", "sum"),
            max_n=("n", "max"),
            min_p=("pearson_p", "min"),
            min_q=("fdr_q", "min"),
        ).reset_index()
    else:
        counts = pd.DataFrame()
    counts.to_csv(outdir / f"QA_fluid_curated_counts_used_by_Figure6_{feature_set}.csv", index=False)
    return out



# Curated diabetes/metabolic/lipid/inflammation variables from harmonized metadata audit.
# These exact names override broad regex classification, preventing triglycerides from
# being placed in glycemia and insulin/HOMA from being placed in inflammation.
CURATED_METABOLIC_CATEGORY_BY_COHORT = {
    "ADNI": {
        "vrf_bmi_PHC_Diabetes": "diabetes_glucose_insulin",
        "vrf_bmi_PHC_Diabetes_raw_clean": "diabetes_glucose_insulin",
        "meta__vrf_bmi_PHC_Diabetes": "diabetes_glucose_insulin",
    },
    "ADRC": {
        "DIABETES": "diabetes_glucose_insulin",
        "DIABTYPE": "diabetes_glucose_insulin",
    },
    "HABS": {
        # Diabetes / glycemia
        "clinical__CDX_Diabetes": "diabetes_glucose_insulin",
        "CDX_Diabetes_x": "diabetes_glucose_insulin",
        "CDX_Diabetes_y": "diabetes_glucose_insulin",
        "IMH_Diabetes": "diabetes_glucose_insulin",
        "DSQ_Treatment_Insulin": "diabetes_glucose_insulin",
        "clinical__BW_HBA1c": "diabetes_glucose_insulin",
        "BW_HBA1c_x": "diabetes_glucose_insulin",
        "BW_HBA1c_y": "diabetes_glucose_insulin",
        "clinical__BW_Glucose": "diabetes_glucose_insulin",
        "BW_Glucose_x": "diabetes_glucose_insulin",
        "BW_Glucose_y": "diabetes_glucose_insulin",
        # Insulin resistance
        "r3_MSD_Plasma_Insulin": "diabetes_glucose_insulin",
        "biomarkers__r3_HOMA_IR": "diabetes_glucose_insulin",
        "biomarkers_csv__r3_HOMA_IR": "diabetes_glucose_insulin",
        "HOMA_IR_x": "diabetes_glucose_insulin",
        "HOMA_IR_y": "diabetes_glucose_insulin",
        # Lipids
        "clinical__BW_Triglycerides": "lipids_cholesterol",
        "BW_Triglycerides_x": "lipids_cholesterol",
        "BW_Triglycerides_y": "lipids_cholesterol",
        "clinical__BW_HDLChol": "lipids_cholesterol",
        "BW_HDLChol_x": "lipids_cholesterol",
        "BW_HDLChol_y": "lipids_cholesterol",
        "clinical__BW_LDLchol": "lipids_cholesterol",
        "BW_LDLchol_x": "lipids_cholesterol",
        "BW_LDLchol_y": "lipids_cholesterol",
        "clinical__BW_CholTotal": "lipids_cholesterol",
        "BW_CholTotal_x": "lipids_cholesterol",
        "BW_CholTotal_y": "lipids_cholesterol",
        "BW_Nonhdl": "lipids_cholesterol",
        "BW_Cholhdlcratio": "lipids_cholesterol",
        "clinical__CDX_Dyslipidemia": "lipids_cholesterol",
        "IMH_HighCholesterol": "lipids_cholesterol",
        # Inflammation
        "r3_QTX_Plasma_IL_6": "inflammation_immune",
        "r3_QTX_Plasma_TNFalpha": "inflammation_immune",
        "r3_QTX_Plasma_IL_10": "inflammation_immune",
        "r3_QTX_Plasma_IL_5": "inflammation_immune",
        "r3_MSD_Serum_CRP": "inflammation_immune",
        "r3_MSD_Serum_TNFalpha": "inflammation_immune",
        "r3_MSD_Serum_IL_6": "inflammation_immune",
    },
}

def curated_metabolic_category(cohort, col):
    """Exact-name classification for audited diabetes/metabolic variables."""
    c = str(col)
    mapping = CURATED_METABOLIC_CATEGORY_BY_COHORT.get(str(cohort), {})
    if c in mapping:
        return mapping[c]
    # Also accept meta__ prefixed copies of curated variables when present.
    if c.startswith("meta__") and c[6:] in mapping:
        return mapping[c[6:]]
    return None


SUPP_PATTERNS = {
    # Fluid / biomarker subfamilies
    "biomarker_abeta": re.compile(r"ab40|ab42|abeta|a_beta|amyloid", re.I),
    "biomarker_tau_ptau": re.compile(r"ptau|p.?tau|ttau|t.?tau|total.?tau|\btau\b", re.I),
    "biomarker_gfap": re.compile(r"gfap", re.I),
    "biomarker_nfl": re.compile(r"\bnfl\b|nf_l|neurofilament", re.I),

    # Exploratory clinical / metabolic biology
    "diabetes_glucose_insulin": re.compile(
        r"diab|diabetes|(^|_)glucose($|_)|fasting_glucose|blood.?glucose|"
        r"hba1c|\ba1c\b|hemoglobin_a1c|insulin|homa|metformin|hypogly",
        re.I,
    ),
    "lipids_cholesterol": re.compile(
        r"triglycer|cholesterol|\bchol\b|\bhdl\b|\bldl\b|nonhdl|cholhdl|"
        r"lipid|dyslip|highcholesterol",
        re.I,
    ),
    "inflammation_immune": re.compile(
        r"crp|tnf|tnfalpha|interleukin|\bil\b|il_?\d+|cytokine|chemokine|"
        r"mcp|ifn|interferon|inflamm|immune|sicam|svcam|icam|vcam|"
        r"platelet|wbc|rbc|hemoglobin|haemoglobin|neutrophil|lymphocyte|"
        r"monocyte|albumin|globulin|complement",
        re.I,
    ),
    "lifestyle_smoking": re.compile(
        r"smok|smoking|smoker|cigarette|pack.?year|tobacco|alcohol|exercise|"
        r"physical_activity|diet",
        re.I,
    ),
    "extended_cardiovascular_risk": re.compile(
        r"hypertension|hyperten|stroke|heart|cardio|vascular|myocardial|infarct|"
        r"ischemi|ischaemi|blood.?pressure|systolic|diastolic|\bsbp\b|\bdbp\b|bpsys|bpdia|"
        r"(^|_)pulse(\d|_|$)|(^|_)om_pulse|pulse_pressure",
        re.I,
    ),

    # Other broad discovery buckets
    "cognition": re.compile(
        r"memory|executive|processing.?speed|language|visuospatial|global.?cog|"
        r"moca|mmse|cdr|adas|trail|fluency|recall|recognition|attention",
        re.I,
    ),
    "imaging_hippocampus_network": re.compile(
        r"hippocamp|hippo|\bhc\b|brain.?volume|\btbv\b|\bicv\b|"
        r"\bfa\b|\bmd\b|\brd\b|clustering|path.?length|graph|network|"
        r"connectome|efficiency",
        re.I,
    ),
}


SUPP_CATEGORY_LABELS = {
    "biomarker_abeta": "Aβ / amyloid",
    "biomarker_tau_ptau": "Tau / pTau",
    "biomarker_gfap": "GFAP",
    "biomarker_nfl": "NfL",
    "transcriptomic_PC_AD_DECODE": "Transcriptomic PCs",
    "diabetes_glucose_insulin": "Diabetes / glucose",
    "lipids_cholesterol": "Lipids / cholesterol",
    "inflammation_immune": "Inflammation / immune",
    "lifestyle_smoking": "Lifestyle / smoking",
    "extended_cardiovascular_risk": "Extended CV risk",
    "cognition": "Cognition",
    "imaging_hippocampus_network": "Imaging / network",
}

SUPP_FIGURE_CATEGORIES = [
    "diabetes_glucose_insulin",
    "lipids_cholesterol",
    "inflammation_immune",
    "lifestyle_smoking",
    "extended_cardiovascular_risk",
    "biomarker_abeta",
    "biomarker_tau_ptau",
    "biomarker_gfap",
    "biomarker_nfl",
    "transcriptomic_PC_AD_DECODE",
]


def infer_supp_category(cohort, col):
    c = str(col)
    if cohort == "AD_DECODE" and PC_Z_RE.fullmatch(c):
        return "transcriptomic_PC_AD_DECODE"
    if BIOMARKER_FALSE_POSITIVE_RE.search(c):
        return None

    # Biomarkers first so LUMIPULSE never becomes vascular due "pulse".
    for cat in ["biomarker_abeta", "biomarker_tau_ptau", "biomarker_gfap", "biomarker_nfl"]:
        if SUPP_PATTERNS[cat].search(c):
            return cat

    # Exact curated diabetes/metabolic/lipid/inflammatory variables from
    # harmonized metadata audit. This prevents regex artifacts.
    curated_cat = curated_metabolic_category(cohort, c)
    if curated_cat is not None:
        return curated_cat

    # Then clinically interpretable exploratory families.
    for cat in [
        "diabetes_glucose_insulin",
        "lipids_cholesterol",
        "inflammation_immune",
        "lifestyle_smoking",
        "extended_cardiovascular_risk",
        "cognition",
        "imaging_hippocampus_network",
    ]:
        if SUPP_PATTERNS[cat].search(c):
            return cat
    return None



MOLECULAR_TECHNICAL_EXCLUDE_RE = re.compile(
    r"sample|plate|batch|run|assay|tube|aliquot|well|barcode|file|path|"
    r"source_columns|missing|availability|available|rnavol|pbmcvol|bcvol|apvolume|"
    r"processing|qc|quality|collection|draw|freeze|thaw|storage|hemoly|hemolysis",
    re.I,
)

EXPLORATORY_MOLECULAR_CATEGORIES = {
    "biomarker_abeta", "biomarker_tau_ptau", "biomarker_gfap",
    "biomarker_nfl", "transcriptomic_PC_AD_DECODE",
}


def exploratory_molecular_scan(fig6_dir, feature_set, args, outdir):
    """
    Exploratory molecular screen for Figure 6.

    This tier is separate from the curated AD/neurodegeneration panel. It scans
    all eligible amyloid/tau/GFAP/NfL biomarker columns and AD-DECODE
    transcriptomic PCs, removes technical/sample-processing columns, applies FDR
    within cohort across the exploratory molecular family, and provides one top
    hit for the main heatmap by the standard FDR-first rule.
    """
    rows, inv = [], []
    for cohort in COHORTS:
        df = read_merged(fig6_dir, feature_set, cohort)
        cbag = get_cbag_col(df, args.cbag_column)
        y = clean_numeric(df[cbag])

        for col in df.columns:
            c = str(col)
            if c == cbag or DIRECT_EXCLUDE.search(c):
                continue
            if BIOMARKER_FALSE_POSITIVE_RE.search(c) or MOLECULAR_TECHNICAL_EXCLUDE_RE.search(c):
                continue

            cat = infer_supp_category(cohort, c)
            if cat not in EXPLORATORY_MOLECULAR_CATEGORIES:
                continue

            # For transcriptomics, keep the same max-PC guard used by the curated
            # AD-DECODE molecular panel unless the user raises --max-transcriptomic-pc.
            if cat == "transcriptomic_PC_AD_DECODE":
                mpc = PC_Z_RE.fullmatch(c)
                if not mpc:
                    continue
                if int(mpc.group(1)) > int(getattr(args, "max_transcriptomic_pc", 20)):
                    continue

            x = clean_numeric(df[c])
            inv.append({
                "cohort": cohort,
                "column": c,
                "molecular_category": cat,
                "molecular_category_label": SUPP_CATEGORY_LABELS.get(cat, cat),
                "n_numeric": int(x.notna().sum()),
                "n_unique": int(x.nunique(dropna=True)),
                "examples": " | ".join(df[c].dropna().astype(str).head(5).tolist()),
            })

            a = corr_assoc(x, y)
            if not (a["n"] >= args.min_n and a["n_unique_x"] >= 3 and a["n_unique_y"] >= 3):
                continue

            label = str(c).replace("_z_clean", "")
            if cat == "biomarker_abeta":
                family_label = "Aβ / amyloid"
            elif cat == "biomarker_tau_ptau":
                family_label = "Tau / pTau"
            elif cat == "biomarker_gfap":
                family_label = "GFAP"
            elif cat == "biomarker_nfl":
                family_label = "NfL"
            elif cat == "transcriptomic_PC_AD_DECODE":
                family_label = "Transcriptomic PC"
            else:
                family_label = SUPP_CATEGORY_LABELS.get(cat, cat)

            rows.append({
                "source": "exploratory_molecular_scan",
                "cohort": cohort,
                "main_domain": "Exploratory molecular",
                "subdomain": family_label,
                "variable": c,
                "clean": norm(c),
                "domain": "",
                "family": "Exploratory molecular",
                "family_label": family_label,
                "display_label": short(label, 22),
                "pearson_r": a["pearson_r"],
                "pearson_p": a["pearson_p"],
                "fdr_q": np.nan,
                "n": a["n"],
                "tested": True,
                "is_fdr": False,
                "candidate_status": "tested",
                "classification_reason": "exploratory_molecular_non_curated_screen",
                "marker_family": family_label,
                "base_marker": fluid_base_marker(c),
                "variant_priority": fluid_variant_priority(c),
                "slope": a["slope"],
                "intercept": a["intercept"],
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = deduplicate_molecular_variants(out, outdir, feature_set, label="exploratory_molecular")
        out["abs_r"] = pd.to_numeric(out["pearson_r"], errors="coerce").abs()
        for cohort, idx in out.groupby("cohort").groups.items():
            out.loc[idx, "fdr_q"] = bh_fdr(out.loc[idx, "pearson_p"])
        out["is_fdr"] = pd.to_numeric(out["fdr_q"], errors="coerce") < args.fdr_alpha
        out["candidate_status"] = np.where(out["is_fdr"], "FDR", "NS")

    pd.DataFrame(inv).to_csv(outdir / f"QA_exploratory_molecular_column_inventory_{feature_set}.csv", index=False)
    out.to_csv(outdir / f"QA_exploratory_molecular_candidates_{feature_set}.csv", index=False)

    if not out.empty:
        top_sub = []
        for (cohort, subdomain), g in out.groupby(["cohort", "subdomain"], dropna=False):
            fdr = g[g["is_fdr"]].copy()
            if not fdr.empty:
                pick = fdr.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
                pick["selection_rule"] = "FDR_then_max_abs_r_within_subfamily"
            else:
                pick = g.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
                pick["selection_rule"] = "NS_fallback_max_abs_r_within_subfamily"
            top_sub.append(pick)
        pd.DataFrame(top_sub).to_csv(
            outdir / f"Supplementary_exploratory_molecular_top_per_subfamily_{feature_set}.csv",
            index=False,
        )

    return out


def write_supplementary_discovery_tables(fig6_dir, feature_set, args, outdir):
    """Broad direct merged-table top-hit tables for supplementary results."""
    rows = []
    for cohort in COHORTS:
        df = read_merged(fig6_dir, feature_set, cohort)
        cbag = get_cbag_col(df, args.cbag_column)
        y = clean_numeric(df[cbag])
        for col in df.columns:
            if str(col) == cbag or DIRECT_EXCLUDE.search(str(col)):
                continue
            cat = infer_supp_category(cohort, col)
            if cat is None:
                continue
            dom, _ = classify_merged_domain(col)
            if cat.startswith("biomarker") or cat == "transcriptomic_PC_AD_DECODE":
                dom = "Fluid biomarkers / transcriptomic PCs"
            if cat == "cognition":
                dom = "Cognition tests"
            x = clean_domain_numeric(df[col], colname=col, domain=dom or "")
            a = corr_assoc(x, y)
            if not (a["n"] >= args.min_n and a["n_unique_x"] >= 3 and a["n_unique_y"] >= 3):
                continue
            rows.append({
                "source": "supplementary_direct_merged_scan",
                "cohort": cohort,
                "supp_category": cat,
                "supp_category_label": SUPP_CATEGORY_LABELS.get(cat, cat),
                "main_domain": dom,
                "variable": str(col),
                "clean": norm(col),
                "display_label": short(col, 28),
                "pearson_r": a["pearson_r"],
                "pearson_p": a["pearson_p"],
                "n": a["n"],
                "n_unique_x": a["n_unique_x"],
                "classification_reason": "supplementary_direct_pattern",
            })

    broad = pd.DataFrame(rows)
    if broad.empty:
        broad.to_csv(outdir / f"Supplementary_all_individual_column_candidates_{feature_set}.csv", index=False)
        return broad

    broad["abs_r"] = pd.to_numeric(broad["pearson_r"], errors="coerce").abs()
    broad["supp_fdr_q_within_cohort_category"] = np.nan
    for (_, _), idx in broad.groupby(["cohort", "supp_category"]).groups.items():
        broad.loc[idx, "supp_fdr_q_within_cohort_category"] = bh_fdr(broad.loc[idx, "pearson_p"])
    broad["supp_is_fdr_within_category"] = broad["supp_fdr_q_within_cohort_category"] < args.fdr_alpha

    broad = broad.sort_values(
        ["cohort", "supp_category", "supp_is_fdr_within_category", "abs_r", "n"],
        ascending=[True, True, False, False, False],
    )
    broad.to_csv(outdir / f"Supplementary_all_individual_column_candidates_{feature_set}.csv", index=False)

    counts = broad.groupby(["cohort", "supp_category", "supp_category_label"]).agg(
        n_candidates=("variable", "nunique"),
        n_fdr=("supp_is_fdr_within_category", "sum"),
        max_n=("n", "max"),
        median_n=("n", "median"),
        min_p=("pearson_p", "min"),
        min_q=("supp_fdr_q_within_cohort_category", "min"),
    ).reset_index().sort_values(["cohort", "supp_category"])
    counts.to_csv(outdir / f"Supplementary_counts_by_cohort_category_{feature_set}.csv", index=False)

    top = broad.groupby(["cohort", "supp_category"], group_keys=False).head(args.supp_top_k)
    top.to_csv(outdir / f"Supplementary_top{args.supp_top_k}_individual_columns_by_category_{feature_set}.csv", index=False)

    selected_rows = []
    for (_, _), g in broad.groupby(["cohort", "supp_category"], dropna=False):
        fdr = g[g["supp_is_fdr_within_category"]]
        if not fdr.empty:
            pick = fdr.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
            pick["supp_selection_rule"] = "FDR_first_max_abs_r"
        else:
            pick = g.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
            pick["supp_selection_rule"] = "NS_fallback_max_abs_r"
        selected_rows.append(pick)
    sel = pd.DataFrame(selected_rows)
    sel.to_csv(outdir / f"Supplementary_selected_top_individual_hit_by_category_{feature_set}.csv", index=False)

    # FDR-only top hits for the supplementary exploratory heatmap.
    fdr_selected_rows = []
    for (_, _), g in broad.groupby(["cohort", "supp_category"], dropna=False):
        fdr = g[g["supp_is_fdr_within_category"]].copy()
        if fdr.empty:
            continue
        pick = fdr.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
        pick["supp_selection_rule"] = "FDR_only_max_abs_r"
        fdr_selected_rows.append(pick)
    fdr_sel = pd.DataFrame(fdr_selected_rows)
    fdr_sel.to_csv(outdir / f"Supplementary_selected_FDR_only_top_hit_by_category_{feature_set}.csv", index=False)

    biomarker_cats = [
        "biomarker_abeta", "biomarker_tau_ptau", "biomarker_gfap",
        "biomarker_nfl", "transcriptomic_PC_AD_DECODE",
    ]
    broad[broad["supp_category"].isin(biomarker_cats)].to_csv(
        outdir / f"Supplementary_biomarker_individual_columns_Abeta_tau_GFAP_NfL_PC_{feature_set}.csv",
        index=False,
    )
    sel[sel["supp_category"].isin(biomarker_cats)].to_csv(
        outdir / f"Supplementary_selected_biomarker_top_hits_Abeta_tau_GFAP_NfL_PC_{feature_set}.csv",
        index=False,
    )
    return broad


# =============================================================================
# Candidate selection
# =============================================================================

def combine_candidates(*frames):
    keep = [
        "source", "cohort", "main_domain", "subdomain", "variable", "clean",
        "domain", "family", "family_label", "display_label", "pearson_r",
        "pearson_p", "fdr_q", "n", "tested", "is_fdr", "abs_r",
        "candidate_status", "classification_reason", "endpoint_source",
        "resid_covariates", "slope", "intercept", "marker_family", "base_marker",
        "variant_priority",
    ]
    clean_frames = []
    for frame in frames:
        if frame is None or len(frame) == 0:
            continue
        f = frame.copy()
        for c in keep:
            if c not in f.columns:
                f[c] = np.nan if c in ["pearson_r", "pearson_p", "fdr_q", "n", "abs_r", "slope", "intercept"] else ""
        clean_frames.append(f[keep])
    if not clean_frames:
        return pd.DataFrame(columns=keep)
    out = pd.concat(clean_frames, ignore_index=True)
    out["pearson_r"] = pd.to_numeric(out["pearson_r"], errors="coerce")
    out["pearson_p"] = pd.to_numeric(out["pearson_p"], errors="coerce")
    out["fdr_q"] = pd.to_numeric(out["fdr_q"], errors="coerce")
    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    out["abs_r"] = out["pearson_r"].abs()
    out["is_fdr"] = out["fdr_q"] < 0.05
    out["candidate_status"] = np.where(out["is_fdr"], "FDR", "NS")
    return out



def _row_text_for_selection(df):
    cols = ["variable", "clean", "display_label", "family", "family_label", "domain", "classification_reason"]
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series("", index=df.index)
    return df[existing].astype(str).agg(" ".join, axis=1).map(norm)


def _is_bmi_like_text(s):
    return s.str.contains(r"(^|_)bmi($|_)|body_mass_index|body.?mass.?index|vrf_bmi", regex=True, na=False)


def _is_vascular_preferred_text(s):
    # Main cardiovascular cell is restricted to the three vital-sign families:
    # systolic BP, diastolic BP, and pulse. Other vascular-risk variables remain
    # in supplementary discovery tables.
    preferred = _is_cardio_vital_text(s)
    excluded = s.str.contains(
        r"(?:^|_)bmi(?:$|_)|vrf_bmi|body_mass|weight|waist|waist_hip|whr|hip_circ|"
        r"smok|pack_year|age_at|(^|_)age($|_)|hypertension|stroke|heart|cardiac|infarct|ischemi|ischaemi",
        regex=True,
        na=False,
    )
    return preferred & ~excluded



def _is_fluid_tau_text(s):
    # Prefer tau/pTau if it is FDR-significant, because it is directly AD-related
    # and avoids replacing a true tau hit with an NS amyloid-ratio fallback.
    return s.str.contains(
        r"ptau|p_tau|p.?tau|ttau|t_tau|total.?tau|total_tau|(^|_)tau($|_)",
        regex=True,
        na=False,
    )


def _is_cardio_vital_text(s):
    # Main cardiovascular cell: restrict to the three vital-sign families requested:
    # systolic BP, diastolic BP, and pulse. Other vascular-risk variables remain
    # in supplementary discovery tables.
    return s.str.contains(
        r"systolic|diastolic|(?:^|_)sbp(?:$|_)|(?:^|_)dbp(?:$|_)|blood_pressure|"
        r"(?:^|_)bp(?:$|_)|bpsys|bpdia|(?:^|_)pulse(?:\d|_|$)|(?:^|_)om_pulse|pulse_pressure|vspulse|vsbpsys|vsbpdia",
        regex=True,
        na=False,
    )



BMI_PRIORITY_BY_COHORT = {
    # ADNI: prefer true PHC BMI because it has higher N, then direct/calculated BMI.
    # Do not use Age_CardiovascularRisk, CVRScore, Weight, VSWEIGHT, or other vrf_bmi PHC risk fields.
    "ADNI": [
        "vrf_bmi_PHC_BMI_raw_clean",
        "vrf_bmi_PHC_BMI",
        "meta__vrf_bmi_PHC_BMI",
        "vrf_bmi_PHC_BMI_z_clean",
        "BMI_raw_clean",
        "BMI_calculated_raw_clean",
        "BMI",
        "BMI_calculated",
        "meta__BMI",
        "meta__BMI_calculated",
    ],
    "ADRC": [
        "BMI_original",
        "BMI_raw_clean",
        "BMI",
        "bmi",
        "bmi_raw_clean",
    ],
    "HABS": [
        # Prefer observed kg/m^2 BMI values; z-scored copies are fallback only.
        "OM_BMI_clinical",
        "OM_BMI_clinical_raw_clean",
        "OM_BMI",
        "OM_BMI_raw_clean",
        "BMI",
        "BMI_raw_clean",
        "bmi",
        "bmi_raw_clean",
        "OM_BMI_clinical_z_clean",
        "OM_BMI_clinical_raw_clean",
        "OM_BMI_clinical",
        "BMI_raw_clean",
        "BMI",
        "bmi",
    ],
    # AD_DECODE: use actual BMI only. If absent, leave the BMI cell blank;
    # do not use Weight.
    "AD_DECODE": [
        # Prefer observed BMI carried from harmonized metadata; recomputed H/W BMI is fallback.
        "meta__BMI",
        "BMI",
        "BMI_raw_clean",
        "BMI_from_height_weight",
        "BMI",
        "body_mass_index",
        "BodyMassIndex",
        "body mass index",
    ],
}


BMI_REJECT_RE = re.compile(
    r"age_cardiovascularrisk|cardiovascularrisk|cvrscore|hypertension|diabetes|heart|"
    r"diagnosis|sex|education|race|ethnicity|visit|weight|vsweight",
    re.I,
)


def _pick_priority_bmi_candidate(g, cohort):
    """Return the first valid observed-BMI candidate using a cohort-specific priority list."""
    if g is None or g.empty:
        return None
    gg = g.copy()
    for c in ["variable", "display_label", "clean"]:
        if c not in gg.columns:
            gg[c] = ""
    gg["_var"] = gg["variable"].astype(str)
    gg["_norm_var"] = gg["_var"].map(norm)
    gg["_text"] = (
        gg["variable"].astype(str) + " " +
        gg["display_label"].astype(str) + " " +
        gg["clean"].astype(str)
    )

    # Keep true BMI labels and explicitly reject non-BMI fields in the same namespace.
    observed_bmi_mask = (
        gg["_text"].str.contains(r"(?:^|[^A-Za-z0-9])BMI(?:[^A-Za-z0-9]|$)|body.?mass.?index|body_mass_index", regex=True, case=False, na=False)
        | gg["_var"].str.contains(r"(?:^|_)vrf_bmi_PHC_BMI(?:$|_|raw|z)", regex=True, case=False, na=False)
        | gg["_var"].str.contains(r"(?:^|_)meta__vrf_bmi_PHC_BMI(?:$|_|raw|z)", regex=True, case=False, na=False)
    )
    observed_bmi_mask = observed_bmi_mask & ~gg["_text"].str.contains(BMI_REJECT_RE, regex=True, na=False)
    gg = gg.loc[observed_bmi_mask].copy()
    gg = gg[pd.to_numeric(gg["pearson_r"], errors="coerce").notna()].copy()
    if gg.empty:
        return None

    priorities = BMI_PRIORITY_BY_COHORT.get(str(cohort), [])
    norm_priorities = [norm(x) for x in priorities]

    for p_raw, p_norm in zip(priorities, norm_priorities):
        # First prefer an exact variable-name match. This prevents lowercase z-scored
        # duplicates such as bmi_raw_clean from being chosen when BMI_raw_clean exists.
        hit = gg[gg["_var"] == p_raw].copy()
        if hit.empty:
            hit = gg[gg["_norm_var"] == p_norm].copy()
        if not hit.empty:
            hit = hit.sort_values(["n"], ascending=[False])
            pick = hit.iloc[0].copy()
            pick["selected_rule"] = f"BMI_priority_observed_no_domain_FDR:{p_raw}"
            return pick

    # If no priority string matched, use observed BMI with largest N, not strongest |r|.
    # This prevents correlation-driven selection among redundant BMI encodings.
    hit = gg.sort_values(["n"], ascending=[False]).iloc[0].copy()
    hit["selected_rule"] = "BMI_observed_fallback_largest_N_no_domain_FDR"
    return hit



# PATCH_DEDICATED_OBSERVED_BMI_RESOLVER
OBSERVED_BMI_PRIORITY_BY_COHORT = {
    "ADNI": [
        "vrf_bmi_PHC_BMI_raw_clean", "vrf_bmi_PHC_BMI", "meta__vrf_bmi_PHC_BMI",
        "BMI_raw_clean", "BMI", "BMI_calculated_raw_clean", "BMI_calculated", "meta__BMI",
    ],
    "ADRC": [
        "BMI_original", "BMI_raw_clean", "BMI", "bmi", "bmi_raw_clean", "meta__BMI",
    ],
    "HABS": [
        "OM_BMI_clinical", "OM_BMI_clinical_raw_clean",
        "OM_BMI", "OM_BMI_raw_clean",
        "bmi", "bmi_raw_clean",
        "BMI", "BMI_raw_clean",
        "meta__BMI",
        # z-scored BMI intentionally excluded from observed-BMI resolver
    ],
    "AD_DECODE": [
        "meta__BMI", "BMI", "BMI_raw_clean", "BMI_from_height_weight",
        # lowercase bmi intentionally excluded unless no observed candidate exists
    ],
}


def observed_bmi_resolver(fig6_dir, feature_set, args, outdir):
    """
    Build one observed-BMI candidate per cohort from merged tables using plausible kg/m^2 values only.
    This overrides generic BMI candidates so Figure 6 never selects z-scored BMI or sentinel values.
    """
    rows = []
    qa = []
    for cohort in COHORTS:
        f = fig6_dir / "merged_tables" / f"merged_metadata_screening_{feature_set}_{cohort}.csv"
        if not f.exists():
            qa.append({"cohort": cohort, "status": "missing_merged_table", "path": str(f)})
            continue
        df = pd.read_csv(f, low_memory=False)

        cbag_col = get_cbag_col(df)
        if not cbag_col:
            # fallback
            cbag_candidates = [c for c in ["cBAG", "cBAG_global", "cBAG_withPCA", "cBAG_withoutPCA"] if c in df.columns]
            cbag_candidates += [c for c in df.columns if "cbag" in str(c).lower()]
            cbag_candidates = list(dict.fromkeys(cbag_candidates))
            cbag_col = cbag_candidates[0] if cbag_candidates else None
        if not cbag_col:
            qa.append({"cohort": cohort, "status": "missing_cbag", "path": str(f)})
            continue

        y = clean_numeric(df[cbag_col])
        picked = None
        candidates = []

        for var in OBSERVED_BMI_PRIORITY_BY_COHORT.get(cohort, []):
            if var == "BMI_from_height_weight":
                if cohort != "AD_DECODE":
                    continue
                x, audit = compute_addecode_bmi_from_height_weight(df)
                if x is None:
                    continue
                x = clean_bmi_numeric(x)
                source_reason = "computed_from_height_weight"
            else:
                if var not in df.columns:
                    continue
                x = clean_bmi_numeric(df[var])
                source_reason = "observed_metadata_bmi"

            a = corr_assoc(x, y)
            n_nonmiss = int(x.notna().sum())
            n_unique = int(x.nunique(dropna=True))
            bmi_min = float(x.min(skipna=True)) if x.notna().any() else np.nan
            bmi_max = float(x.max(skipna=True)) if x.notna().any() else np.nan
            bmi_mean = float(x.mean(skipna=True)) if x.notna().any() else np.nan
            bmi_sd = float(x.std(skipna=True)) if x.notna().any() else np.nan

            ok = (
                a["n"] >= args.min_n and
                a["n_unique_x"] >= 3 and
                a["n_unique_y"] >= 3 and
                n_unique >= 3 and
                pd.notna(bmi_mean) and
                10 <= bmi_min <= 80 and
                10 <= bmi_max <= 80 and
                pd.notna(bmi_sd) and bmi_sd > 1
            )
            candidates.append({
                "cohort": cohort, "variable": var, "ok": ok, "source_reason": source_reason,
                "n_nonmiss": n_nonmiss, "n_pair": a["n"], "n_unique": n_unique,
                "bmi_min": bmi_min, "bmi_max": bmi_max, "bmi_mean": bmi_mean, "bmi_sd": bmi_sd,
                "r": a["pearson_r"], "p": a["pearson_p"], "cbag_col": cbag_col,
            })
            if ok and picked is None:
                picked = (var, x, a, source_reason)

        if picked is None:
            qa.extend(candidates if candidates else [{"cohort": cohort, "status": "no_plausible_observed_bmi", "path": str(f)}])
            continue

        var, x, a, reason = picked
        qa.extend(candidates)
        label = "BMI" if var in ["BMI", "BMI_raw_clean", "meta__BMI", "BMI_from_height_weight"] else re.sub(r"_raw_clean$|^meta__", "", var)

        rows.append({
            "source": "observed_bmi_resolver",
            "cohort": cohort,
            "main_domain": "BMI",
            "subdomain": "",
            "variable": var,
            "clean": norm(var),
            "domain": "Observed BMI",
            "family": "BMI",
            "family_label": "BMI",
            "display_label": label.replace("_", " "),
            "pearson_r": a["pearson_r"],
            "pearson_p": a["pearson_p"],
            "fdr_q": a["pearson_p"],  # BMI is a single selected phenotype, not domain-screen FDR
            "n": a["n"],
            "tested": True,
            "is_fdr": False,
            "abs_r": abs(a["pearson_r"]) if pd.notna(a["pearson_r"]) else np.nan,
            "candidate_status": "NS",
            "classification_reason": reason,
            "endpoint_source": "observed_BMI_kgm2_clean_10_80",
            "resid_covariates": "",
            "slope": a["slope"],
            "intercept": a["intercept"],
            "marker_family": "",
            "base_marker": "",
            "variant_priority": 0,
            "selected_rule": f"observed_BMI_priority_clean_10_80:{var}",
            "cell_sig_label": "NS",
        })

    pd.DataFrame(qa).to_csv(outdir / f"QA_observed_BMI_resolver_candidates_{feature_set}.csv", index=False)
    return pd.DataFrame(rows)


def select_top_fdr_first(candidates, domains, alpha=0.05, fluid_main_min_n=50):
    rows = []
    for cohort in COHORTS:
        for dom in domains:
            g = candidates[(candidates["cohort"] == cohort) & (candidates["main_domain"] == dom)].copy()
            g = g[pd.to_numeric(g["pearson_r"], errors="coerce").notna()]
            if g.empty:
                continue
            # BMI: priority-based observed BMI only. Do not use within-domain FDR
            # because BMI columns are redundant encodings of one phenotype. Do not
            # select by strongest |r|; select by predefined cohort-specific BMI
            # priority, then largest N. AD_DECODE is left blank if no BMI exists.
            if dom == "BMI":
                pick = _pick_priority_bmi_candidate(g, cohort)
                if pick is None:
                    continue
                pick["fdr_q"] = pick["pearson_p"]  # display p-value in q/p field for BMI only
                pick["cell_sig_label"] = sig_label(pick["pearson_p"], alpha)
                rows.append(pick)
                continue

            # Cardiovascular: restrict main cell to systolic BP, diastolic BP,
            # and pulse / pulse pressure only. Other vascular-risk variables stay
            # in supplementary discovery tables.
            if dom == "Cardiovascular":
                txtsel = _row_text_for_selection(g)
                preferred = _is_vascular_preferred_text(txtsel)
                if preferred.any():
                    g = g.loc[preferred].copy()

            fdr = g[g["fdr_q"] < alpha].copy()
            if dom == "Curated fluid / transcriptomic" and not fdr.empty:
                robust = fdr[pd.to_numeric(fdr["n"], errors="coerce") >= fluid_main_min_n].copy()
                if not robust.empty:
                    fdr = robust
                # If tau/pTau is FDR-significant, prefer it over non-tau hits.
                # This specifically protects ADRC tau/pTau from being displaced by
                # an amyloid-ratio fallback.
                ft = _row_text_for_selection(fdr)
                tau_fdr = fdr.loc[_is_fluid_tau_text(ft)].copy()
                if not tau_fdr.empty:
                    fdr = tau_fdr
            if not fdr.empty:
                pick = fdr.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
                pick["selected_rule"] = "FDR_then_max_abs_r"
                if dom == "Curated fluid / transcriptomic":
                    pick["selected_rule"] = f"FDR_then_max_abs_r_prefer_N_ge_{fluid_main_min_n}"
                pick["cell_sig_label"] = sig_label(pick["fdr_q"], alpha)
            else:
                pick = g.sort_values(["abs_r", "n"], ascending=[False, False]).iloc[0].copy()
                pick["selected_rule"] = "NS_fallback_max_abs_r"
                pick["cell_sig_label"] = "NS"
            rows.append(pick)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def top_k_by_domain(candidates, domains, k=20):
    rows = []
    for cohort in COHORTS:
        for dom in domains:
            g = candidates[(candidates["cohort"] == cohort) & (candidates["main_domain"] == dom)].copy()
            g = g[pd.to_numeric(g["pearson_r"], errors="coerce").notna()]
            if g.empty:
                continue
            g = g.sort_values(["is_fdr", "abs_r", "n"], ascending=[False, False, False])
            rows.append(g.head(k))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def agreement_summary(selected, domains):
    rows = []
    for dom in domains:
        g = selected[(selected["main_domain"] == dom) & (selected["cell_sig_label"] != "NS")].copy()
        if g.empty:
            rows.append({
                "main_domain": dom,
                "n_fdr_cohorts": 0,
                "n_possible_cohorts": len(COHORTS),
                "coverage": 0,
                "n_positive": 0,
                "n_negative": 0,
                "majority_direction": "none",
                "agreement_fraction": np.nan,
                "median_r_fdr_only": np.nan,
            })
            continue
        signs = np.sign(g["pearson_r"].astype(float))
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())
        if n_pos >= n_neg:
            maj = "positive"
            agree = n_pos / len(g)
        else:
            maj = "negative"
            agree = n_neg / len(g)
        rows.append({
            "main_domain": dom,
            "n_fdr_cohorts": int(len(g)),
            "n_possible_cohorts": len(COHORTS),
            "coverage": len(g) / len(COHORTS),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "majority_direction": maj,
            "agreement_fraction": agree,
            "median_r_fdr_only": float(g["pearson_r"].median()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# BMI and domain QA
# =============================================================================

def bmi_inventory(fig6_dir, feature_set, args, outdir):
    pat = re.compile(r"(^|[^a-z])bmi([^a-z]|$)|body.?mass.?index|body_mass_index|obes", re.I)
    inv_rows, assoc_rows = [], []
    for cohort in COHORTS:
        df = read_merged(fig6_dir, feature_set, cohort)
        cbag = get_cbag_col(df, args.cbag_column)
        for col in df.columns:
            if pat.search(str(col)):
                x = clean_numeric(df[col])
                a = corr_assoc(x, df[cbag])
                inv_rows.append({
                    "cohort": cohort,
                    "column": col,
                    "n_total": len(df),
                    "n_nonmissing_numeric": int(x.notna().sum()),
                    "mean": float(x.mean()) if x.notna().any() else np.nan,
                    "sd": float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan,
                    "min": float(x.min()) if x.notna().any() else np.nan,
                    "max": float(x.max()) if x.notna().any() else np.nan,
                })
                assoc_rows.append({
                    "cohort": cohort,
                    "column": col,
                    **a,
                    "abs_r": abs(a["pearson_r"]) if pd.notna(a["pearson_r"]) else np.nan,
                })
    inv = pd.DataFrame(inv_rows)
    assoc = pd.DataFrame(assoc_rows)
    if not inv.empty:
        inv = inv.sort_values(["cohort", "n_nonmissing_numeric"], ascending=[True, False])
    if not assoc.empty:
        assoc = assoc.sort_values(["cohort", "abs_r"], ascending=[True, False])
    inv.to_csv(outdir / f"QA_BMI_column_inventory_{feature_set}.csv", index=False)
    assoc.to_csv(outdir / f"QA_BMI_direct_cBAG_associations_{feature_set}.csv", index=False)
    return inv, assoc



def plot_selected_bmi_histograms(fig6_dir, feature_set, args, outdir, selected_main=None):
    """Save per-cohort observed BMI histograms as Figure 6 QA/sanity checks."""
    if getattr(args, "skip_bmi_histograms", False):
        return pd.DataFrame()

    hist_dir = outdir / "QA_BMI_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    selected_bmi = pd.DataFrame()
    if selected_main is not None and len(selected_main) > 0:
        selected_bmi = selected_main[selected_main["main_domain"].astype(str).eq("BMI")].copy()

    for cohort in COHORTS:
        try:
            df = read_merged(fig6_dir, feature_set, cohort)
        except Exception as exc:
            rows.append({"cohort": cohort, "status": f"read_failed:{exc}"})
            continue

        var = None
        source_reason = "selected_main_BMI"
        g = selected_bmi[selected_bmi["cohort"].astype(str).eq(cohort)] if not selected_bmi.empty else pd.DataFrame()
        if not g.empty:
            var = str(g.iloc[0].get("variable", ""))

        if var == "BMI_from_height_weight":
            bmi, audit = compute_addecode_bmi_from_height_weight(df)
            bmi = clean_bmi_numeric(bmi) if bmi is not None else None
            source_reason = "computed_from_height_weight_selected"
        elif var and var in df.columns:
            bmi = clean_bmi_numeric(df[var])
        else:
            # Fallback: use observed-BMI resolver candidate order.
            bmi = None
            source_reason = "fallback_observed_BMI_priority"
            for cand in OBSERVED_BMI_PRIORITY_BY_COHORT.get(cohort, []):
                if cand == "BMI_from_height_weight":
                    b, _ = compute_addecode_bmi_from_height_weight(df)
                    b = clean_bmi_numeric(b) if b is not None else None
                elif cand in df.columns:
                    b = clean_bmi_numeric(df[cand])
                else:
                    continue
                if b is not None and b.notna().sum() >= args.min_n and b.nunique(dropna=True) >= 3:
                    var, bmi = cand, b
                    break

        if bmi is None:
            rows.append({
                "cohort": cohort,
                "bmi_column": var,
                "source_reason": source_reason,
                "status": "no_plausible_BMI",
                "n": 0,
            })
            continue

        vals = pd.to_numeric(bmi, errors="coerce").dropna()
        if vals.empty:
            rows.append({
                "cohort": cohort,
                "bmi_column": var,
                "source_reason": source_reason,
                "status": "no_nonmissing_BMI_after_cleaning",
                "n": 0,
            })
            continue

        safe_cohort = COHORT_LABELS.get(cohort, cohort).replace("-", "_")
        png = hist_dir / f"Figure6_BMI_histogram_{safe_cohort}_{feature_set}.png"
        pdf = hist_dir / f"Figure6_BMI_histogram_{safe_cohort}_{feature_set}.pdf"

        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        ax.hist(vals, bins=25, edgecolor="black", linewidth=0.4)
        ax.axvline(vals.mean(), linestyle="--", linewidth=1.2, label=f"Mean={vals.mean():.1f}")
        ax.axvline(vals.median(), linestyle=":", linewidth=1.2, label=f"Median={vals.median():.1f}")
        ax.set_title(f"{COHORT_LABELS.get(cohort, cohort)} observed BMI QA")
        ax.set_xlabel("BMI (kg/m²)")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8)
        ax.text(
            0.98, 0.95, f"column: {var}\nn={len(vals)}",
            transform=ax.transAxes, va="top", ha="right", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.70, edgecolor="0.8", pad=1.5),
        )
        fig.tight_layout()
        fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

        rows.append({
            "cohort": cohort,
            "bmi_column": var,
            "source_reason": source_reason,
            "status": "ok",
            "n": int(vals.shape[0]),
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            "median": float(vals.median()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "plot_png": str(png),
            "plot_pdf": str(pdf),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / f"QA_BMI_histogram_summary_{feature_set}.csv", index=False)
    return summary

def write_domain_qa(candidates, outdir, feature_set, top_k):
    domains = MAIN_DOMAINS + QA_DOMAINS
    for dom in domains:
        safe = norm(dom)
        tab = top_k_by_domain(candidates, [dom], top_k)
        tab.to_csv(outdir / f"QA_top{top_k}_{safe}_{feature_set}.csv", index=False)

    # Extra targeted QA for specific biomarkers.
    targets = {
        "abeta": r"ab40|ab42|abeta|amyloid",
        "tau_ptau": r"ptau|p_tau|ttau|t_tau|\btau\b",
        "gfap": r"gfap",
        "nfl": r"\bnfl\b|nf_l|neurofilament",
        "transcriptomic_pc": r"^PC(?:[1-9]|[12][0-9]|30)_z_clean$|transcriptomic",
        "diabetes_glucose_insulin": r"diab|glucose|hba1c|\ba1c\b|insulin|homa|bmi|triglycer|cholesterol|hdl|ldl",
    }
    for name, pat in targets.items():
        mask = candidates.apply(
            lambda r: bool(re.search(pat, " ".join(str(r.get(k, "")) for k in [
                "variable", "clean", "display_label", "family", "family_label", "domain"
            ]).lower(), re.I)),
            axis=1,
        )
        tab = candidates[mask].copy()
        if not tab.empty:
            tab = tab.sort_values(["cohort", "is_fdr", "abs_r", "n"], ascending=[True, False, False, False])
        tab.to_csv(outdir / f"QA_targeted_{name}_{feature_set}.csv", index=False)


# =============================================================================
# AUC computation directly from merged tables
# =============================================================================

def derive_apoe4(df):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    c = find_col(df, [
        "meta__APOE4_strict_34_44_carrier", "APOE4_strict_34_44_carrier",
        "meta__APOE4_carriage", "APOE4_carriage",
        "meta__APOE4_carrier", "APOE4_carrier",
        "meta__APOE4_Positivity", "APOE4_Positivity",
    ])
    if c:
        x = pd.to_numeric(df[c], errors="coerce")
        out.loc[x.eq(0)] = 0
        out.loc[x.gt(0)] = 1
        return out
    g = find_col(df, [
        "meta__APOE_genotype_harmonized", "APOE_genotype_harmonized",
        "meta__APOE_Label_for_table1", "APOE_Label_for_table1",
        "meta__APOE", "APOE",
    ])
    if g:
        txt = df[g].astype(str).str.upper().str.replace(" ", "", regex=False)
        known = ~txt.isin(["", "NAN", "NONE", "<NA>"])
        out.loc[known & txt.str.contains("4", na=False)] = 1
        out.loc[known & ~txt.str.contains("4", na=False)] = 0
    return out


def derive_sex_female(df):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    c = find_col(df, [
        "meta__sex_label_for_table1", "sex_label_for_table1",
        "meta__sex_label", "sex_label",
        "meta__sex", "sex", "meta__Sex", "Sex", "meta__SEX", "SEX",
        "meta__gender", "gender", "meta__Gender", "Gender",
        "meta__PTGENDER", "PTGENDER",
    ])
    if not c:
        return out
    raw = df[c]
    num = pd.to_numeric(raw, errors="coerce")
    txt = raw.astype(str).str.strip().str.lower()
    out.loc[txt.isin(["f", "female", "woman", "women"])] = 1
    out.loc[txt.isin(["m", "male", "man", "men"])] = 0
    out.loc[out.isna() & num.eq(2)] = 1
    out.loc[out.isna() & num.eq(1)] = 0
    return out


def derive_cog_impairment(df):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    status = cognitive_status(df)
    out.loc[status.eq("Cognitively normal")] = 0
    out.loc[status.eq("Cognitively impaired")] = 1
    return out


def roc_auc_manual(y, score):
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(y) == 0:
        return np.nan, np.array([]), np.array([])
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return np.nan, np.array([]), np.array([])

    order = np.argsort(-s)
    y = y[order]
    s = s[order]
    distinct = np.r_[True, s[1:] != s[:-1]]
    idx = np.where(distinct)[0]
    tps = np.cumsum(y == 1)[idx]
    fps = np.cumsum(y == 0)[idx]
    tpr = np.r_[0, tps / pos, 1]
    fpr = np.r_[0, fps / neg, 1]
    auc = float(np.trapz(tpr, fpr))
    return auc, fpr, tpr



def _stratified_bootstrap_oriented_auc(y, score, n_boot=5000, seed=42):
    """Stratified nonparametric bootstrap for oriented ROC AUC."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return {
            "auc_boot_mean": np.nan,
            "auc_boot_sd": np.nan,
            "auc_ci_low": np.nan,
            "auc_ci_high": np.nan,
            "auc_p_vs_0p5": np.nan,
            "n_boot_valid": 0,
        }
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([b_pos, b_neg])
        auc_b, _, _ = roc_auc_manual(y[idx], score[idx])
        if pd.isna(auc_b):
            continue
        vals.append(max(float(auc_b), 1.0 - float(auc_b)))
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return {
            "auc_boot_mean": np.nan,
            "auc_boot_sd": np.nan,
            "auc_ci_low": np.nan,
            "auc_ci_high": np.nan,
            "auc_p_vs_0p5": np.nan,
            "n_boot_valid": 0,
        }
    return {
        "auc_boot_mean": float(np.mean(vals)),
        "auc_boot_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "auc_ci_low": float(np.quantile(vals, 0.025)),
        "auc_ci_high": float(np.quantile(vals, 0.975)),
        "auc_p_vs_0p5": float((np.sum(vals <= 0.5) + 1) / (len(vals) + 1)),
        "n_boot_valid": int(len(vals)),
    }


def compute_auc_table(fig6_dir, feature_set, args, outdir):
    """Recompute Figure 6B AUCs and add stratified bootstrap 95% CIs."""
    rows, curves = [], []
    n_boot = int(getattr(args, "auc_bootstrap_n", 5000))
    seed0 = int(getattr(args, "auc_bootstrap_seed", 42))

    for cohort_i, cohort in enumerate(COHORTS):
        df = read_merged(fig6_dir, feature_set, cohort)
        cbag = get_cbag_col(df, args.cbag_column)
        score = clean_numeric(df[cbag])

        target_data = [
            ("APOE4", derive_apoe4(df), "APOE4 carrier", "non-carrier"),
            ("Cognitive impairment", derive_cog_impairment(df), "impaired", "normal"),
            ("Sex", derive_sex_female(df), "female", "male"),
        ]

        for target_i, (target, y, pos_label, neg_label) in enumerate(target_data):
            tmp = pd.DataFrame({"y": y, "score": score}).dropna()
            n = len(tmp)
            n_pos = int((tmp["y"] == 1).sum()) if n else 0
            n_neg = int((tmp["y"] == 0).sum()) if n else 0
            row = {
                "feature_set": feature_set,
                "cohort": cohort,
                "target": target,
                "positive_label": pos_label,
                "negative_label": neg_label,
                "n": n,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "auc_raw": np.nan,
                "auc_oriented": np.nan,
                "score_flipped": False,
                "auc_boot_mean": np.nan,
                "auc_boot_sd": np.nan,
                "auc_ci_low": np.nan,
                "auc_ci_high": np.nan,
                "auc_p_vs_0p5": np.nan,
                "n_boot": n_boot,
                "n_boot_valid": 0,
                "status": "not_computed",
            }
            if n < 8 or n_pos < 3 or n_neg < 3:
                row["status"] = "insufficient_class_n"
                rows.append(row)
                continue

            yv = tmp["y"].astype(int).to_numpy()
            sv = tmp["score"].astype(float).to_numpy()
            auc, fpr, tpr = roc_auc_manual(yv, sv)
            if pd.isna(auc):
                row["status"] = "auc_failed"
                rows.append(row)
                continue

            flipped = auc < 0.5
            if flipped:
                auc2, fpr, tpr = roc_auc_manual(yv, -sv)
                auc_oriented = auc2
            else:
                auc_oriented = auc

            boot = _stratified_bootstrap_oriented_auc(
                yv, sv, n_boot=n_boot, seed=seed0 + 100 * cohort_i + target_i
            )
            row.update({
                "auc_raw": float(auc),
                "auc_oriented": float(auc_oriented),
                "score_flipped": bool(flipped),
                **boot,
                "status": "computed",
            })
            rows.append(row)
            curves.append(pd.DataFrame({
                "feature_set": feature_set,
                "cohort": cohort,
                "target": target,
                "fpr": fpr,
                "tpr": tpr,
                "auc_oriented": auc_oriented,
                "auc_ci_low": boot["auc_ci_low"],
                "auc_ci_high": boot["auc_ci_high"],
                "score_flipped": flipped,
            }))

    auc_df = pd.DataFrame(rows)
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    auc_df.to_csv(outdir / f"Figure6_AUC_recomputed_APOE4_CogImpairment_Sex_{feature_set}.csv", index=False)
    auc_df.to_csv(outdir / f"Figure6B_AUC_bootstrap_summary_{feature_set}.csv", index=False)
    curve_df.to_csv(outdir / f"Figure6_AUC_recomputed_ROC_curves_{feature_set}.csv", index=False)
    return auc_df, curve_df


# =============================================================================
# Plotting
# =============================================================================

def draw_main_heatmap(selected, agreement, outbase, args):
    mat = np.full((len(COHORTS), len(MAIN_DOMAINS)), np.nan)
    labels = [["No data" for _ in MAIN_DOMAINS] for _ in COHORTS]

    for i, cohort in enumerate(COHORTS):
        for j, dom in enumerate(MAIN_DOMAINS):
            g = selected[(selected["cohort"] == cohort) & (selected["main_domain"] == dom)]
            if g.empty:
                continue
            r = g.iloc[0]
            rv = float(r["pearson_r"])
            q = r["fdr_q"]
            p = r.get("pearson_p", np.nan)
            n = int(r["n"]) if pd.notna(r["n"]) else 0
            stat_label = "p" if dom == "BMI" else "q"
            stat_value = p if dom == "BMI" else q
            sig = r.get("cell_sig_label", sig_label(stat_value, args.fdr_alpha))
            mat[i, j] = rv
            # Clean main-panel labels: full q/p and N are kept in the source table.
            labels[i][j] = f"r={rv:.2f}\n{sig}"

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-0.6, vmax=0.6)
    ax.set_xticks(np.arange(len(MAIN_DOMAINS)))
    ax.set_xticklabels(MAIN_DOMAINS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(COHORTS)))
    ax.set_yticklabels([COHORT_LABELS[c] for c in COHORTS], fontweight="bold")

    for i in range(len(COHORTS)):
        for j in range(len(MAIN_DOMAINS)):
            color = "white" if pd.notna(mat[i, j]) and abs(mat[i, j]) > 0.32 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color=color)

    ax.set_title(
        "Figure 6A. Multidomain cBAG validation\n"
        "Top cohort-domain association; cell text shows r and FDR significance",
        fontsize=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Signed Pearson r")

    if getattr(args, "show_panel_a_agreement_footnote", False) and not agreement.empty:
        bits = []
        for _, r in agreement.iterrows():
            if int(r["n_fdr_cohorts"]) == 0:
                bits.append(f"{r['main_domain']}: 0/4 FDR")
            else:
                bits.append(
                    f"{r['main_domain']}: {int(r['n_fdr_cohorts'])}/4 FDR, "
                    f"agreement={r['agreement_fraction']:.2f} ({r['majority_direction']})"
                )
        fig.text(0.02, -0.03, "Cross-cohort FDR agreement: " + " | ".join(bits), fontsize=8)

    plt.tight_layout()
    fig.savefig(str(outbase) + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def draw_qa_heatmap(selected, domain, outbase, title, args):
    rows, vals, labels = [], [], []
    for cohort in COHORTS:
        g = selected[(selected["cohort"] == cohort) & (selected["main_domain"] == domain)]
        rows.append(COHORT_LABELS[cohort])
        if g.empty:
            vals.append(np.nan)
            labels.append("No data")
        else:
            r = g.iloc[0]
            vals.append(float(r["pearson_r"]))
            sig = r.get("cell_sig_label", sig_label(r["fdr_q"], args.fdr_alpha))
            labels.append(
                f"{short(r.get('display_label', r.get('variable','')), 18)}\n"
                f"r={float(r['pearson_r']):.2f}\nq={fmt(r['fdr_q'])}\n"
                f"N={int(r['n']) if pd.notna(r['n']) else 0} {sig}"
            )

    mat = np.array(vals, dtype=float).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(6, max(3.2, 0.8 * len(COHORTS) + 1.4)))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["Top hit"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontweight="bold")
    for i, lab in enumerate(labels):
        color = "white" if pd.notna(mat[i, 0]) and abs(mat[i, 0]) > 0.32 else "black"
        ax.text(0, i, lab, ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.09, pad=0.04).set_label("Signed Pearson r")
    plt.tight_layout()
    fig.savefig(str(outbase) + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)



def draw_auc_heatmap(auc_df, outbase, args):
    targets = ["APOE4", "Cognitive impairment", "Sex"]
    mat = np.full((len(COHORTS), len(targets)), np.nan)
    labels = [["NA" for _ in targets] for _ in COHORTS]

    for i, cohort in enumerate(COHORTS):
        for j, target in enumerate(targets):
            g = auc_df[(auc_df["cohort"] == cohort) & (auc_df["target"] == target)]
            if g.empty:
                continue
            r = g.iloc[0]
            if r["status"] != "computed" or pd.isna(r["auc_oriented"]):
                labels[i][j] = f"{r['status']}\nN={int(r['n'])}"
            else:
                mat[i, j] = float(r["auc_oriented"])
                lo = pd.to_numeric(pd.Series([r.get("auc_ci_low", np.nan)]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([r.get("auc_ci_high", np.nan)]), errors="coerce").iloc[0]
                if pd.notna(lo) and pd.notna(hi):
                    labels[i][j] = f"{float(r['auc_oriented']):.2f}\n95% CI {lo:.2f}-{hi:.2f}\nN={int(r['n'])}"
                else:
                    labels[i][j] = f"{float(r['auc_oriented']):.2f}\nN={int(r['n'])}"

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.5, vmax=1.0, alpha=0.72)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(COHORTS)))
    ax.set_yticklabels([COHORT_LABELS[c] for c in COHORTS], fontweight="bold")
    for i in range(len(COHORTS)):
        for j in range(len(targets)):
            color = "white" if pd.notna(mat[i, j]) and mat[i, j] > 0.88 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color=color, fontweight="bold")
    ax.set_title("Figure 6B. cBAG discrimination targets\nAUC with stratified bootstrap 95% confidence intervals")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Oriented AUC")
    plt.tight_layout()
    fig.savefig(str(outbase) + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)



def draw_auc_roc_grid(curves, outbase, args):
    targets = ["APOE4", "Cognitive impairment", "Sex"]
    fig, axes = plt.subplots(len(COHORTS), len(targets), figsize=(11, 10), squeeze=False)
    for i, cohort in enumerate(COHORTS):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            g = curves[(curves["cohort"] == cohort) & (curves["target"] == target)]
            if g.empty:
                ax.text(0.5, 0.5, "No ROC", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.plot(g["fpr"], g["tpr"], lw=1.5)
                ax.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
                auc = float(g["auc_oriented"].iloc[0])
                lo = pd.to_numeric(pd.Series([g.get("auc_ci_low", pd.Series([np.nan])).iloc[0]]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([g.get("auc_ci_high", pd.Series([np.nan])).iloc[0]]), errors="coerce").iloc[0]
                flip = bool(g["score_flipped"].iloc[0])
                ci_txt = f" [{lo:.2f}-{hi:.2f}]" if pd.notna(lo) and pd.notna(hi) else ""
                ax.set_title(f"{COHORT_LABELS[cohort]} | {target}\nAUC={auc:.2f}{ci_txt}{' flipped' if flip else ''}", fontsize=8)
                ax.set_xlabel("FPR", fontsize=7)
                ax.set_ylabel("TPR", fontsize=7)
                ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(f"{target}\n" + ax.get_title(), fontsize=8)
    fig.suptitle("Figure 6B supplement. ROC curves from cBAG with bootstrap CIs", fontsize=14)
    plt.tight_layout()
    fig.savefig(str(outbase) + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)



def draw_supplementary_fdr_top_hits_heatmap(broad, outbase, args):
    """
    Supplementary exploratory heatmap.

    Displays only FDR-significant top hits within each cohort x supplementary
    category. Cells without an FDR hit are left blank/marked No FDR.
    """
    if broad is None or len(broad) == 0:
        return None
    df = broad.copy()
    if "supp_is_fdr_within_category" not in df.columns:
        return None
    df = df[df["supp_category"].isin(SUPP_FIGURE_CATEGORIES)].copy()
    if df.empty:
        return None

    fig_cats = [c for c in SUPP_FIGURE_CATEGORIES if c in set(df["supp_category"])]
    if not fig_cats:
        return None

    mat = np.full((len(COHORTS), len(fig_cats)), np.nan)
    labels = [["No FDR" for _ in fig_cats] for _ in COHORTS]
    selected_rows = []

    for i, cohort in enumerate(COHORTS):
        for j, cat in enumerate(fig_cats):
            g = df[(df["cohort"] == cohort) & (df["supp_category"] == cat)].copy()
            g = g[g["supp_is_fdr_within_category"].fillna(False)]
            if g.empty:
                continue
            g = g.sort_values(["abs_r", "n"], ascending=[False, False])
            r = g.iloc[0].copy()
            selected_rows.append(r)
            rv = float(r["pearson_r"])
            q = r["supp_fdr_q_within_cohort_category"]
            n = int(r["n"]) if pd.notna(r["n"]) else 0
            mat[i, j] = rv
            labels[i][j] = (
                f"{short(r.get('display_label', r.get('variable', '')), 18)}\n"
                f"r={rv:.2f}\nq={fmt(q)}\nN={n}"
            )

    source = pd.DataFrame(selected_rows)
    source.to_csv(str(outbase) + "_source_data.csv", index=False)

    fig_w = max(13, 1.25 * len(fig_cats) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-0.6, vmax=0.6)

    ax.set_xticks(np.arange(len(fig_cats)))
    ax.set_xticklabels([SUPP_CATEGORY_LABELS.get(c, c) for c in fig_cats], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(COHORTS)))
    ax.set_yticklabels([COHORT_LABELS[c] for c in COHORTS], fontweight="bold")

    for i in range(len(COHORTS)):
        for j in range(len(fig_cats)):
            if pd.isna(mat[i, j]):
                ax.text(j, i, "No FDR", ha="center", va="center", fontsize=7, color="0.35")
            else:
                color = "white" if abs(mat[i, j]) > 0.32 else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7, color=color)

    ax.set_title(
        "Supplementary exploratory domain hits\n"
        "Strongest FDR-significant association per cohort-category; blank cells indicate no FDR hit",
        fontsize=13,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("Signed Pearson r")

    plt.tight_layout()
    fig.savefig(str(outbase) + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return source




def write_all_fdr_hits_by_category(broad, outdir, feature_set):
    """Save all FDR-significant supplementary hits overall and one CSV per category."""
    if broad is None or len(broad) == 0:
        return pd.DataFrame()
    df = broad.copy()
    if "supp_is_fdr_within_category" not in df.columns:
        return pd.DataFrame()
    sig = df[df["supp_is_fdr_within_category"].fillna(False)].copy()
    sig.to_csv(outdir / f"Supplementary_all_FDR_hits_by_category_{feature_set}.csv", index=False)
    subdir = outdir / f"Supplementary_all_FDR_hits_by_category_{feature_set}"
    subdir.mkdir(parents=True, exist_ok=True)
    for cat, g in sig.groupby("supp_category", dropna=False):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cat)).strip("_")
        g.sort_values(["cohort", "abs_r", "n"], ascending=[True, False, False]).to_csv(subdir / f"{name}.csv", index=False)
    return sig


def draw_selected_supplementary_heatmap(broad, categories, outbase, title, args):
    """Draw a targeted FDR-only supplementary heatmap for a chosen set of categories."""
    if broad is None or len(broad) == 0:
        return pd.DataFrame()
    df = broad.copy()
    df = df[df["supp_category"].isin(categories)].copy()
    if df.empty:
        return pd.DataFrame()

    fig_cats = [c for c in categories if c in set(df["supp_category"])]
    if not fig_cats:
        return pd.DataFrame()

    mat = np.full((len(COHORTS), len(fig_cats)), np.nan)
    labels = [["No FDR" for _ in fig_cats] for _ in COHORTS]
    selected_rows = []

    for i, cohort in enumerate(COHORTS):
        for j, cat in enumerate(fig_cats):
            g = df[(df["cohort"] == cohort) & (df["supp_category"] == cat)].copy()
            g = g[g["supp_is_fdr_within_category"].fillna(False)].copy()
            if g.empty:
                continue
            g = g.sort_values(["abs_r", "n"], ascending=[False, False])
            r = g.iloc[0].copy()
            selected_rows.append(r)
            rv = float(r["pearson_r"])
            q = pd.to_numeric(r["supp_fdr_q_within_cohort_category"], errors="coerce")
            n = int(r["n"]) if pd.notna(r["n"]) else 0
            mat[i, j] = rv
            labels[i][j] = (
                f"{short(r.get('display_label', r.get('variable', '')), 18)}\n"
                f"r={rv:.2f}\nq={fmt(q)}\nN={n}"
            )

    source = pd.DataFrame(selected_rows)
    source.to_csv(str(outbase) + "_source_data.csv", index=False)

    fig_w = max(9.5, 1.55 * len(fig_cats) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-0.6, vmax=0.6)
    ax.set_xticks(np.arange(len(fig_cats)))
    ax.set_xticklabels([SUPP_CATEGORY_LABELS.get(c, c) for c in fig_cats], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(COHORTS)))
    ax.set_yticklabels([COHORT_LABELS[c] for c in COHORTS], fontweight="bold")

    for i in range(len(COHORTS)):
        for j in range(len(fig_cats)):
            if pd.isna(mat[i, j]):
                ax.text(j, i, "No FDR", ha="center", va="center", fontsize=8, color="0.35")
            else:
                color = "white" if abs(mat[i, j]) > 0.32 else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.5, color=color)

    ax.set_title(title, fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("Signed Pearson r")
    plt.tight_layout()
    fig.savefig(str(outbase) + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return source


def _volcano_safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x[:120] if x else "NA"



def volcano_point_label(row, domain="", max_len=28):
    """
    Human-readable volcano labels. For imaging/hippocampus, expose the actual
    metric family (FA/RD/MD/hippocampus/brain volume/efficiency/path length)
    rather than relying only on generic display_label text.
    """
    var = str(row.get("variable", ""))
    disp = str(row.get("display_label", var))
    text = " ".join([var, disp]).lower()
    clean = norm(" ".join([var, disp]))

    if str(domain).lower().startswith("imaging"):
        prefix = ""
        if re.search(r"hippocamp|hippo|(^|_)hc($|_)", text) or re.search(r"hippocamp|hippo|(^|_)hc($|_)", clean):
            prefix = "Hc "
        elif re.search(r"brain.?volume|total.?brain|tbv", text) or re.search(r"brain_volume|total_brain|tbv", clean):
            prefix = "Brain "

        if re.search(r"(^|[_\\W])fa([_\\W]|$)|fractional.?anisotropy|hc_fa", text) or re.search(r"(^|_)fa($|_)|fractional_anisotropy|hc_fa", clean):
            return short(prefix + "FA", max_len)
        if re.search(r"(^|[_\\W])rd([_\\W]|$)|radial.?diffus|hc_rd", text) or re.search(r"(^|_)rd($|_)|radial_diffus|hc_rd", clean):
            return short(prefix + "RD", max_len)
        if re.search(r"(^|[_\\W])md([_\\W]|$)|mean.?diffus|hc_md", text) or re.search(r"(^|_)md($|_)|mean_diffus|hc_md", clean):
            return short(prefix + "MD", max_len)
        if re.search(r"hippocamp|hippo|(^|_)hc($|_)", text) or re.search(r"hippocamp|hippo|(^|_)hc($|_)", clean):
            return short("Hippocampus", max_len)
        if re.search(r"brain.?volume|total.?brain|tbv", text) or re.search(r"brain_volume|total_brain|tbv", clean):
            return short("Brain volume", max_len)
        if re.search(r"global.?eff", text) or "global_eff" in clean:
            return short("Global efficiency", max_len)
        if re.search(r"local.?eff", text) or "local_eff" in clean:
            return short("Local efficiency", max_len)
        if re.search(r"path.?length|pathlength", text) or "path_length" in clean:
            return short("Path length", max_len)
        if re.search(r"clustering|cluster", text) or "clustering" in clean:
            return short("Clustering", max_len)

    return short(disp if disp and disp.lower() != "nan" else var, max_len)


def _prepare_volcano_table(candidates, supp_broad):
    frames = []

    if candidates is not None and len(candidates) > 0:
        c = candidates.copy()
        c["volcano_source"] = "main_curated_candidates"
        c["volcano_category"] = c.get("main_domain", "")
        c["volcano_category_label"] = c["volcano_category"]
        if "marker_family" in c.columns:
            is_fluid = c["main_domain"].eq("Fluid biomarkers / transcriptomic PCs")
            fam = c["marker_family"].fillna("").astype(str)
            c.loc[is_fluid & fam.ne(""), "volcano_category"] = "fluid_" + fam.str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
            c.loc[is_fluid & fam.ne(""), "volcano_category_label"] = "Fluid: " + fam
        c["volcano_q"] = pd.to_numeric(c.get("fdr_q", np.nan), errors="coerce")
        c["volcano_p"] = pd.to_numeric(c.get("pearson_p", np.nan), errors="coerce")
        c["volcano_r"] = pd.to_numeric(c.get("pearson_r", np.nan), errors="coerce")
        c["volcano_n"] = pd.to_numeric(c.get("n", np.nan), errors="coerce")
        c["volcano_is_fdr"] = c["volcano_q"] < 0.05
        keep = ["volcano_source", "cohort", "volcano_category", "volcano_category_label", "main_domain", "variable", "display_label", "volcano_r", "volcano_p", "volcano_q", "volcano_n", "volcano_is_fdr"]
        frames.append(c[[k for k in keep if k in c.columns]].copy())

    if supp_broad is not None and len(supp_broad) > 0:
        s = supp_broad.copy()
        s["volcano_source"] = "supplementary_direct_scan"
        s["volcano_category"] = s.get("supp_category", "")
        s["volcano_category_label"] = s.get("supp_category_label", s["volcano_category"])
        s["volcano_q"] = pd.to_numeric(s.get("supp_fdr_q_within_cohort_category", np.nan), errors="coerce")
        s["volcano_p"] = pd.to_numeric(s.get("pearson_p", np.nan), errors="coerce")
        s["volcano_r"] = pd.to_numeric(s.get("pearson_r", np.nan), errors="coerce")
        s["volcano_n"] = pd.to_numeric(s.get("n", np.nan), errors="coerce")
        s["volcano_is_fdr"] = s.get("supp_is_fdr_within_category", False).fillna(False).astype(bool)
        keep = ["volcano_source", "cohort", "volcano_category", "volcano_category_label", "main_domain", "variable", "display_label", "volcano_r", "volcano_p", "volcano_q", "volcano_n", "volcano_is_fdr"]
        frames.append(s[[k for k in keep if k in s.columns]].copy())

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out[pd.to_numeric(out["volcano_r"], errors="coerce").notna()].copy()
    out = out[pd.to_numeric(out["volcano_p"], errors="coerce").notna()].copy()
    out = out[out["volcano_p"] > 0].copy()
    out["neg_log10_p"] = -np.log10(out["volcano_p"].clip(lower=1e-300))
    out["abs_r"] = out["volcano_r"].abs()
    out["volcano_category_label"] = out["volcano_category_label"].fillna(out["volcano_category"]).astype(str)
    out["display_label"] = out["display_label"].fillna(out["variable"]).astype(str)
    return out


def _get_domain_plot_tables(candidates, supp_broad):
    tables = {}
    main_domains = [
        "Imaging / hippocampus",
        "Cognition composites",
        "Cognition tests",
        "Curated fluid / transcriptomic",
        "Exploratory molecular",
    ]
    for dom in main_domains:
        if candidates is None or len(candidates) == 0:
            tables[dom] = pd.DataFrame()
            continue
        d = candidates[candidates["main_domain"].eq(dom)].copy()
        if d.empty:
            tables[dom] = d
            continue
        d["plot_domain"] = dom
        d["plot_q"] = pd.to_numeric(d.get("fdr_q", np.nan), errors="coerce")
        d["plot_p"] = pd.to_numeric(d.get("pearson_p", np.nan), errors="coerce")
        d["plot_r"] = pd.to_numeric(d.get("pearson_r", np.nan), errors="coerce")
        d["plot_n"] = pd.to_numeric(d.get("n", np.nan), errors="coerce")
        d["plot_is_fdr"] = d["plot_q"] < 0.05
        d["display_label"] = d.get("display_label", d.get("variable", "")).fillna(d.get("variable", "")).astype(str)
        d["source_kind"] = "main_curated_candidates"
        tables[dom] = d

    metabolic_cats = {"diabetes_glucose_insulin", "lipids_cholesterol", "inflammation_immune"}
    if supp_broad is None or len(supp_broad) == 0:
        tables["Diabetes / metabolic"] = pd.DataFrame()
    else:
        d = supp_broad[supp_broad["supp_category"].isin(metabolic_cats)].copy()
        if d.empty:
            tables["Diabetes / metabolic"] = d
        else:
            d["plot_domain"] = "Diabetes / metabolic"
            d["plot_q"] = pd.to_numeric(d.get("supp_fdr_q_within_cohort_category", np.nan), errors="coerce")
            d["plot_p"] = pd.to_numeric(d.get("pearson_p", np.nan), errors="coerce")
            d["plot_r"] = pd.to_numeric(d.get("pearson_r", np.nan), errors="coerce")
            d["plot_n"] = pd.to_numeric(d.get("n", np.nan), errors="coerce")
            d["plot_is_fdr"] = d.get("supp_is_fdr_within_category", False).fillna(False).astype(bool)
            d["display_label"] = d.get("display_label", d.get("variable", "")).fillna(d.get("variable", "")).astype(str)
            d["source_kind"] = "supplementary_direct_scan"
            tables["Diabetes / metabolic"] = d
    return tables


def draw_volcano_plots(candidates, supp_broad, outdir, feature_set, args):
    if getattr(args, "skip_volcano_plots", False):
        return pd.DataFrame()

    volcano_dir = outdir / "Supplementary_volcano_plots"
    volcano_dir.mkdir(parents=True, exist_ok=True)

    tables = _get_domain_plot_tables(candidates, supp_broad)
    alpha = float(getattr(args, 'volcano_alpha', 0.05))
    p_lines = [(0.05, '--', '0.60'), (0.01, ':', '0.45'), (0.001, '-.', '0.30')]
    if alpha not in [0.05, 0.01, 0.001]:
        p_lines = [(alpha, '--', '0.60')] + p_lines

    all_stats = []
    manifest_rows = []
    source_rows = []

    domain_order = [
        "Imaging / hippocampus",
        "Cognition composites",
        "Cognition tests",
        "Curated fluid / transcriptomic",
        "Exploratory molecular",
        "Diabetes / metabolic",
    ]

    for dom in domain_order:
        df = tables.get(dom, pd.DataFrame()).copy()
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0), squeeze=False)
        any_panel = False
        dom_source_rows = []
        ymax_global = 1.6
        if not df.empty:
            df = df[pd.to_numeric(df['plot_r'], errors='coerce').notna()].copy()
            df = df[pd.to_numeric(df['plot_p'], errors='coerce').notna()].copy()
            df = df[df['plot_p'] > 0].copy()
            if not df.empty:
                df['neg_log10_p'] = -np.log10(df['plot_p'].clip(lower=1e-300))
                df['abs_r'] = df['plot_r'].abs()
                ymax_global = max(ymax_global, float(df['neg_log10_p'].max()))
        ymax_global *= 1.08

        for idx, cohort in enumerate(COHORTS):
            ax = axes[idx // 2, idx % 2]
            g = df[df['cohort'].eq(cohort)].copy() if not df.empty else pd.DataFrame()
            if g.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=11)
                ax.set_title(COHORT_LABELS.get(cohort, cohort))
                ax.set_xlabel('Pearson r')
                ax.set_ylabel('-log10(p)')
                ax.set_xlim(-0.65, 0.65)
                ax.set_ylim(0, ymax_global)
                for pthr, ls, c in p_lines:
                    y = -np.log10(pthr)
                    ax.axhline(y, linestyle=ls, linewidth=0.9, color=c)
                all_stats.append({
                    'cohort': cohort, 'domain': dom, 'n_tested': 0, 'n_nominal_p_lt_0_05': 0, 'n_fdr_lt_0_05': 0,
                    'top_variable_by_p': np.nan, 'top_display_label_by_p': np.nan, 'top_r_by_p': np.nan,
                    'top_p_by_p': np.nan, 'top_q_by_p': np.nan,
                })
                continue

            any_panel = True
            g = g.sort_values(['plot_p', 'abs_r', 'plot_n'], ascending=[True, False, False])
            ns = g[~g['plot_is_fdr'].fillna(False)].copy()
            fs = g[g['plot_is_fdr'].fillna(False)].copy()
            if not ns.empty:
                ax.scatter(ns['plot_r'], ns['neg_log10_p'], s=24, alpha=0.45, color='0.72', edgecolors='none')
            if not fs.empty:
                ax.scatter(fs['plot_r'], fs['neg_log10_p'], s=38, alpha=0.92, color='tab:red', edgecolors='black', linewidths=0.25)
            ax.axvline(0, linestyle='--', linewidth=0.8, color='0.45')
            for pthr, ls, c in p_lines:
                y = -np.log10(pthr)
                ax.axhline(y, linestyle=ls, linewidth=0.9, color=c)
            ax.set_xlim(-0.65, 0.65)
            ax.set_ylim(0, ymax_global)
            ax.set_xlabel('Pearson r')
            ax.set_ylabel('-log10(p)')
            ax.set_title(COHORT_LABELS.get(cohort, cohort))
            for pthr, _, c in p_lines:
                y = -np.log10(pthr)
                ax.text(0.985, min(y / ymax_global + 0.01, 0.97), f'p={pthr:g}', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=7, color=c)

            top = g.iloc[0]
            all_stats.append({
                'cohort': cohort, 'domain': dom, 'n_tested': int(len(g)),
                'n_nominal_p_lt_0_05': int((g['plot_p'] < 0.05).sum()),
                'n_fdr_lt_0_05': int(g['plot_is_fdr'].sum()),
                'top_variable_by_p': top.get('variable', np.nan),
                'top_display_label_by_p': top.get('display_label', np.nan),
                'top_r_by_p': float(top.get('plot_r', np.nan)),
                'top_p_by_p': float(top.get('plot_p', np.nan)),
                'top_q_by_p': float(top.get('plot_q', np.nan)) if pd.notna(top.get('plot_q', np.nan)) else np.nan,
            })
            dom_source_rows.append(g.assign(domain=dom, cohort_label=COHORT_LABELS.get(cohort, cohort)))
            source_rows.append(g.assign(domain=dom, cohort_label=COHORT_LABELS.get(cohort, cohort)))
            # Label every statistically significant point on the volcano plot.
            # Priority order: FDR-significant points; if none exist in a panel,
            # fall back to all nominally significant points at the requested alpha.
            if not fs.empty:
                lab = fs.sort_values(['plot_p', 'abs_r', 'plot_n'], ascending=[True, False, False]).copy()
            else:
                lab = g[g['plot_p'] < alpha].sort_values(['plot_p', 'abs_r', 'plot_n'], ascending=[True, False, False]).copy()
                if lab.empty:
                    lab = g.sort_values(['plot_p', 'abs_r'], ascending=[True, False]).head(min(5, int(getattr(args, 'volcano_label_top', 8))))

            # Use small alternating offsets to reduce label overlap while ensuring
            # that every significant point remains labeled.
            offsets = [
                (4, 4), (4, 10), (4, -10), (8, 0), (-4, 4), (-4, 10), (-4, -10),
                (8, 10), (8, -10), (-8, 0), (0, 12), (0, -12), (12, 4), (-12, 4)
            ]
            for i_lab, (_, r) in enumerate(lab.iterrows()):
                dx, dy = offsets[i_lab % len(offsets)]
                ax.annotate(
                    volcano_point_label(r, dom, max_len=28),
                    xy=(r['plot_r'], r['neg_log10_p']),
                    xytext=(dx, dy),
                    textcoords='offset points',
                    fontsize=7.5,
                    fontweight='bold' if bool(r.get('plot_is_fdr', False)) else 'normal',
                    ha='left' if dx >= 0 else 'right',
                    va='bottom' if dy >= 0 else 'top',
                    arrowprops=dict(arrowstyle='-', lw=0.4, color='0.45', shrinkA=0, shrinkB=0),
                )

        plt.suptitle(f'Supplementary volcano plots — {dom}', fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        safe = _volcano_safe_name(dom)
        png = volcano_dir / f'SupplementaryVolcano_{safe}_{feature_set}.png'
        pdf = volcano_dir / f'SupplementaryVolcano_{safe}_{feature_set}.pdf'
        fig.savefig(png, dpi=args.dpi, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)
        dom_csv = volcano_dir / f'SupplementaryVolcano_{safe}_{feature_set}_source_data.csv'
        if dom_source_rows:
            pd.concat(dom_source_rows, ignore_index=True, sort=False).to_csv(dom_csv, index=False)
        else:
            pd.DataFrame().to_csv(dom_csv, index=False)
        manifest_rows.append({'domain': dom, 'png': str(png), 'pdf': str(pdf), 'csv': str(dom_csv), 'has_any_data': bool(any_panel)})

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(volcano_dir / f'Supplementary_volcano_plot_manifest_{feature_set}.csv', index=False)
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(volcano_dir / f'Supplementary_volcano_stats_by_cohort_domain_{feature_set}.csv', index=False)
    if not stats_df.empty:
        stats_df.pivot(index='cohort', columns='domain', values='n_fdr_lt_0_05').to_csv(
            volcano_dir / f'Supplementary_volcano_FDRcount_matrix_{feature_set}.csv'
        )
    if source_rows:
        pd.concat(source_rows, ignore_index=True, sort=False).to_csv(
            volcano_dir / f'Supplementary_volcano_all_source_data_{feature_set}.csv', index=False
        )
    return manifest


def _parse_topk_list(spec):
    out = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            k = int(part)
        except Exception:
            continue
        if k > 0 and k not in out:
            out.append(k)
    return out or [10, 30]



def canonical_heatmap_label(variable, display_label, domain=""):
    """Collapse duplicate source-column labels for domain heatmaps."""
    var = str(variable)
    lab = str(display_label) if pd.notna(display_label) else var
    # Remove common prefixes/suffixes and x/y duplicate suffixes.
    lab = re.sub(r"^(meta__|clinical__|biomarkers__|biomarkers_csv__)", "", lab)
    lab = re.sub(r"(_raw_clean|_z_clean|_clean)$", "", lab)
    lab = re.sub(r"_[xy]$", "", lab)
    lab = lab.replace("_", " ")
    lab = re.sub(r"\s+", " ", lab).strip()

    v = norm(var)
    l = norm(lab)
    d = str(domain).lower()

    if "diabetes" in d or "metabolic" in d:
        # Collapse exact duplicated HABS CBC/inflammatory/lipid variables.
        if "bw_platelet" in v or "platelet" in l:
            return "Platelets"
        if "bw_monocyte" in v or "monocyte" in l:
            return "Monocytes"
        if "tnfalpha" in v or "tnf_alpha" in v or "tnfalpha" in l:
            if "serum" in v:
                return "Serum TNFα"
            if "plasma" in v:
                return "Plasma TNFα"
            return "TNFα"
        if "highcholesterolage" in v or "highcholesterol_age" in v:
            return "Age at high cholesterol"
        if "cdx_diabetes" in v or l == "cdx_diabetes" or l == "clinical_cdx_diabetes":
            return "Diabetes diagnosis"
        if "diabtype" in v:
            return "Diabetes type"
        if "hba1c" in v or "a1c" in l:
            return "HbA1c"
        if "glucose" in v:
            return "Glucose"
        if "insulin" in v:
            return "Insulin"
        if "homa_ir" in v:
            return "HOMA-IR"
        if "triglycer" in v:
            return "Triglycerides"
        if "hdlchol" in v or re.search(r"\bhdl\b", l):
            return "HDL cholesterol"
        if "ldlchol" in v or re.search(r"\bldl\b", l):
            return "LDL cholesterol"
        if "choltotal" in v:
            return "Total cholesterol"
        if "nonhdl" in v:
            return "Non-HDL cholesterol"
        if "cholhdlcratio" in v:
            return "Cholesterol/HDL ratio"
        if "crp" in v:
            return "CRP"
        if re.search(r"il_?6|il6", v):
            return "IL-6"
        if re.search(r"il_?10|il10", v):
            return "IL-10"
        if re.search(r"il_?5|il5", v):
            return "IL-5"

    return short(lab if lab else var, 36)


def canonical_heatmap_key(variable, display_label, domain=""):
    """Stable key used to deduplicate rows in top-k heatmaps."""
    return norm(canonical_heatmap_label(variable, display_label, domain))


def _select_top_hits_union(df, top_k, domain=""):
    if df is None or len(df) == 0:
        return []
    needed = ['cohort', 'variable', 'display_label', 'plot_r', 'plot_p', 'plot_q', 'plot_n', 'plot_is_fdr']
    x = df.copy()
    for col in needed:
        if col not in x.columns:
            x[col] = np.nan
    x['abs_r'] = pd.to_numeric(x['plot_r'], errors='coerce').abs()
    x['plot_p'] = pd.to_numeric(x['plot_p'], errors='coerce')
    x['plot_n'] = pd.to_numeric(x['plot_n'], errors='coerce')
    x['heatmap_label'] = x.apply(lambda r: canonical_heatmap_label(r.get('variable', ''), r.get('display_label', ''), domain), axis=1)
    x['heatmap_key'] = x.apply(lambda r: canonical_heatmap_key(r.get('variable', ''), r.get('display_label', ''), domain), axis=1)
    x = x[x['plot_p'].notna()].copy()
    chosen = []
    for cohort in COHORTS:
        g = x[x['cohort'].eq(cohort)].copy()
        if g.empty:
            continue
        # Collapse duplicates within cohort/key first; keep the best available representative.
        g['fdr_rank'] = (~g['plot_is_fdr'].fillna(False)).astype(int)
        g = g.sort_values(['heatmap_key', 'fdr_rank', 'plot_p', 'abs_r', 'plot_n'],
                          ascending=[True, True, True, False, False])
        g = g.groupby('heatmap_key', as_index=False, group_keys=False).head(1)
        g = g.sort_values(['fdr_rank', 'plot_p', 'abs_r', 'plot_n'], ascending=[True, True, False, False])
        chosen.append(g.head(top_k))
    if not chosen:
        return []
    sel = pd.concat(chosen, ignore_index=True, sort=False)
    summary = (sel.groupby(['heatmap_key', 'heatmap_label'], dropna=False)
                 .agg(best_p=('plot_p', 'min'), best_abs_r=('abs_r', 'max'), n_cohorts=('cohort', 'nunique'))
                 .reset_index()
                 .sort_values(['best_p', 'best_abs_r', 'n_cohorts'], ascending=[True, False, False]))
    return list(summary['heatmap_key'])

def draw_domain_hit_heatmaps(candidates, supp_broad, outdir, feature_set, args):
    heat_dir = outdir / 'Supplementary_domain_hit_heatmaps'
    heat_dir.mkdir(parents=True, exist_ok=True)
    tables = _get_domain_plot_tables(candidates, supp_broad)
    topk_list = _parse_topk_list(getattr(args, 'domain_heatmap_topk_list', '10,30'))
    manifest = []
    domain_order = [
        'Imaging / hippocampus',
        'Cognition composites',
        'Cognition tests',
        'Curated fluid / transcriptomic',
        'Exploratory molecular',
        'Diabetes / metabolic',
    ]
    for dom in domain_order:
        df = tables.get(dom, pd.DataFrame()).copy()
        if not df.empty:
            df = df[pd.to_numeric(df['plot_r'], errors='coerce').notna()].copy()
            df = df[pd.to_numeric(df['plot_p'], errors='coerce').notna()].copy()
        for top_k in topk_list:
            selected_keys = _select_top_hits_union(df, top_k, domain=dom)
            if not df.empty:
                df['heatmap_label'] = df.apply(lambda r: canonical_heatmap_label(r.get('variable', ''), r.get('display_label', ''), dom), axis=1)
                df['heatmap_key'] = df.apply(lambda r: canonical_heatmap_key(r.get('variable', ''), r.get('display_label', ''), dom), axis=1)
            if selected_keys:
                plot_df = df[df['heatmap_key'].isin(selected_keys)].copy()
            else:
                plot_df = pd.DataFrame(columns=list(df.columns) if not df.empty else [])
            if not plot_df.empty:
                plot_df['abs_r'] = pd.to_numeric(plot_df['plot_r'], errors='coerce').abs()
                # One row per canonical label/key, not one row per duplicate source column.
                ordering = (plot_df.groupby(['heatmap_key', 'heatmap_label'], dropna=False)
                                 .agg(best_p=('plot_p', 'min'), best_abs_r=('abs_r', 'max'))
                                 .reset_index()
                                 .sort_values(['best_p', 'best_abs_r'], ascending=[True, False]))
                row_keys = list(ordering[['heatmap_key', 'heatmap_label']].itertuples(index=False, name=None))
            else:
                row_keys = []

            mat = np.full((len(row_keys), len(COHORTS)), np.nan)
            ann = [["" for _ in COHORTS] for __ in row_keys]
            for i, (hkey, dlab) in enumerate(row_keys):
                for j, cohort in enumerate(COHORTS):
                    g = plot_df[(plot_df['heatmap_key'].eq(hkey)) & (plot_df['cohort'].eq(cohort))].copy()
                    if g.empty:
                        ann[i][j] = '—'
                        continue
                    g = g.sort_values(['plot_p', 'abs_r', 'plot_n'], ascending=[True, False, False])
                    r = g.iloc[0]
                    rv = float(r['plot_r'])
                    mat[i, j] = rv
                    if bool(r.get('plot_is_fdr', False)):
                        sig = '**'
                    elif pd.notna(r.get('plot_p', np.nan)) and float(r['plot_p']) < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'
                    ann[i][j] = f"r={rv:.2f}\n{sig}"

            fig_h = max(4.8, 0.42 * max(1, len(row_keys)) + 1.8)
            fig, ax = plt.subplots(figsize=(8.2, fig_h))
            if len(row_keys) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.axis('off')
            else:
                im = ax.imshow(mat, aspect='auto', cmap='coolwarm', vmin=-0.6, vmax=0.6)
                ax.set_xticks(np.arange(len(COHORTS)))
                ax.set_xticklabels([COHORT_LABELS[c] for c in COHORTS], rotation=20, ha='right')
                ax.set_yticks(np.arange(len(row_keys)))
                ax.set_yticklabels([short(lbl, 28) for _, lbl in row_keys], fontsize=8)
                for i in range(len(row_keys)):
                    for j in range(len(COHORTS)):
                        color = 'white' if pd.notna(mat[i, j]) and abs(mat[i, j]) > 0.32 else 'black'
                        ax.text(j, i, ann[i][j], ha='center', va='center', fontsize=7, color=color)
                cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
                cbar.set_label('Signed Pearson r')
            ax.set_title(f'Supplementary domain heatmap — {dom} (top {top_k} hits per cohort, union across cohorts)')
            plt.tight_layout()
            safe = _volcano_safe_name(dom)
            png = heat_dir / f'SupplementaryHeatmap_{safe}_top{top_k}_{feature_set}.png'
            pdf = heat_dir / f'SupplementaryHeatmap_{safe}_top{top_k}_{feature_set}.pdf'
            csv = heat_dir / f'SupplementaryHeatmap_{safe}_top{top_k}_{feature_set}.csv'
            src = heat_dir / f'SupplementaryHeatmap_{safe}_top{top_k}_{feature_set}_source_data.csv'
            fig.savefig(png, dpi=args.dpi, bbox_inches='tight')
            fig.savefig(pdf, bbox_inches='tight')
            plt.close(fig)
            matrix_rows = []
            for i, (hkey, dlab) in enumerate(row_keys):
                row = {'heatmap_key': hkey, 'display_label': dlab, 'domain': dom}
                for j, cohort in enumerate(COHORTS):
                    row[f'{cohort}_r'] = mat[i, j]
                    row[f'{cohort}_annotation'] = ann[i][j]
                matrix_rows.append(row)
            pd.DataFrame(matrix_rows).to_csv(csv, index=False)
            plot_df.to_csv(src, index=False)
            manifest.append({'domain': dom, 'top_k': top_k, 'png': str(png), 'pdf': str(pdf), 'csv': str(csv), 'source_data': str(src)})
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(heat_dir / f'Supplementary_domain_hit_heatmap_manifest_{feature_set}.csv', index=False)
    return manifest_df


# =============================================================================
# Supplementary directionality scatter grid
# =============================================================================

def selected_variable_vector(df, variable):
    """Return x vector for selected variable, including computed Aβ ratios."""
    variable = str(variable)
    if " / " in variable:
        parts = [p.strip() for p in variable.split(" / ")]
        if len(parts) == 2 and parts[0] in df.columns and parts[1] in df.columns:
            num = clean_numeric(df[parts[0]])
            den = clean_numeric(df[parts[1]]).replace(0, np.nan)
            return num / den, variable
    if variable in df.columns:
        return clean_numeric(df[variable]), variable
    # Try normalized column-name match.
    nmap = {norm(c): c for c in df.columns}
    if norm(variable) in nmap:
        c = nmap[norm(variable)]
        return clean_numeric(df[c]), c
    return pd.Series(np.nan, index=df.index, dtype=float), variable


def scatter_status_colors(df):
    status = cognitive_status(df)
    colors = pd.Series("0.70", index=df.index, dtype="object")
    colors.loc[status.eq("Cognitively normal")] = "#1b9e77"
    colors.loc[status.eq("Cognitively impaired")] = "#cc79a7"
    return colors, status


def draw_directionality_scatter_grid(fig6_dir, feature_set, selected, outbase, args):
    """Directionality companion figure using the selected top-hit table."""
    if selected is None or selected.empty:
        return pd.DataFrame()
    domains = [d for d in MAIN_DOMAINS if d in set(selected["main_domain"].astype(str))]
    if not domains:
        return pd.DataFrame()

    fig_w = max(18, 3.15 * len(domains))
    fig_h = max(10, 2.45 * len(COHORTS))
    fig, axes = plt.subplots(len(COHORTS), len(domains), figsize=(fig_w, fig_h), squeeze=False)
    source_rows = []

    for i, cohort in enumerate(COHORTS):
        try:
            df = read_merged(fig6_dir, feature_set, cohort)
            cbag = get_cbag_col(df, args.cbag_column)
            y = clean_numeric(df[cbag])
            colors, status = scatter_status_colors(df)
        except Exception as exc:
            df = None
            y = None
            colors = None
            status = None

        for j, dom in enumerate(domains):
            ax = axes[i, j]
            ax.set_title(f"{COHORT_LABELS.get(cohort, cohort)}: {dom}", fontsize=8)
            g = selected[(selected["cohort"].astype(str).eq(cohort)) & (selected["main_domain"].astype(str).eq(dom))].copy()
            if g.empty or df is None:
                ax.text(0.5, 0.5, "No selected variable", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            row = g.iloc[0]
            var = row.get("variable", "")
            x, used_var = selected_variable_vector(df, var)
            z = pd.concat([x.rename("x"), y.rename("cBAG")], axis=1).dropna()
            if len(z) < 3 or z["x"].nunique() < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            c = colors.loc[z.index] if colors is not None else "0.55"
            ax.scatter(z["x"], z["cBAG"], c=c, s=10, alpha=0.75, edgecolors="none")
            try:
                slope, intercept = np.polyfit(z["x"], z["cBAG"], 1)
                xs = np.linspace(float(z["x"].min()), float(z["x"].max()), 100)
                ax.plot(xs, slope * xs + intercept, color="black", lw=1.1)
            except Exception:
                slope, intercept = np.nan, np.nan

            r = pd.to_numeric(pd.Series([row.get("pearson_r", np.nan)]), errors="coerce").iloc[0]
            p = pd.to_numeric(pd.Series([row.get("pearson_p", np.nan)]), errors="coerce").iloc[0]
            q = pd.to_numeric(pd.Series([row.get("fdr_q", np.nan)]), errors="coerce").iloc[0]
            n = int(row.get("n", len(z))) if pd.notna(row.get("n", np.nan)) else len(z)
            sig = row.get("cell_sig_label", sig_label(q, args.fdr_alpha))
            lab = row.get("display_label", used_var)
            ax.set_xlabel(short(lab, 24), fontsize=7)
            ax.set_ylabel("cBAG", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.text(
                0.02, 0.98,
                f"N={n}\nr={r:.2f}\np={fmt(p)}\nq={fmt(q)} {sig}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                bbox=dict(facecolor="white", alpha=0.70, edgecolor="0.8", pad=1.5),
            )
            source_rows.append({
                "cohort": cohort, "main_domain": dom, "variable": var, "used_variable": used_var,
                "display_label": lab, "n": n, "pearson_r": r, "pearson_p": p, "fdr_q": q,
                "cell_sig_label": sig, "slope_plotted": slope, "intercept_plotted": intercept,
            })

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Cognitively normal', markerfacecolor='#1b9e77', markersize=5),
        plt.Line2D([0], [0], marker='o', color='w', label='Cognitively impaired', markerfacecolor='#cc79a7', markersize=5),
        plt.Line2D([0], [0], marker='o', color='w', label='Unknown status', markerfacecolor='0.70', markersize=5),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Supplementary Figure. Directionality of selected cBAG associations by cohort and domain",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(str(outbase) + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    source = pd.DataFrame(source_rows)
    source.to_csv(str(outbase) + "_source_data.csv", index=False)
    return source


def draw_combined(main_png, auc_png, outbase, args):
    try:
        img1 = plt.imread(main_png)
        img2 = plt.imread(auc_png)
    except Exception:
        return
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 1, height_ratios=[1.5, 1.0], hspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax1.axis("off")
    ax2.axis("off")
    fig.suptitle("Figure 6. Multidomain validation of imaging-only cBAG", fontsize=16)
    fig.savefig(str(outbase) + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    fig6_dir = Path(args.figure6_dir)
    outdir = fig6_dir / args.outdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FIGURE6BUILD: final patched multidomain Figure 6")
    print("=" * 100)
    print("Input :", fig6_dir)
    print("Output:", outdir)
    print("Feature set:", args.feature_set)

    print("\n[1/10] Cognitive composite candidates")
    cog = compute_cognitive_candidates(fig6_dir, args.feature_set, args, outdir)

    print("[2/10] Metadata association candidates")
    meta = load_standardized_metadata_associations(fig6_dir, args.feature_set, args, outdir)

    print("[3/10] Direct merged-table domain scan")
    direct = direct_merged_domain_scan(fig6_dir, args.feature_set, args, outdir)

    print("[4/10] Curated fluid biomarker / transcriptomic PC scan")
    fluid = curated_fluid_scan(fig6_dir, args.feature_set, args, outdir)

    print("[4b/10] Exploratory molecular biomarker / transcriptomic scan")
    molecular = exploratory_molecular_scan(fig6_dir, args.feature_set, args, outdir)

    print("[5/10] Combining and selecting candidates")
    # Main figure uses two-tier curated cognition and curated fluid/transcriptomic candidates.
    # Remove broad cognition/fluid hits from metadata/direct scans before combining.
    resolved_domains = ["Cognition", "Cognition composites", "Cognition tests", "Fluid biomarkers / transcriptomic PCs", "Curated fluid / transcriptomic", "Exploratory molecular"]
    meta_main = meta[~meta["main_domain"].isin(resolved_domains)].copy()
    direct_main = direct[~direct["main_domain"].isin(resolved_domains)].copy() if not direct.empty else direct

    candidates = combine_candidates(cog, meta_main, direct_main, fluid, molecular)

    # PATCH_USE_DEDICATED_OBSERVED_BMI:
    # Remove all generic BMI candidates and replace with one cleaned observed kg/m^2 BMI candidate per cohort.
    bmi_obs = observed_bmi_resolver(fig6_dir, args.feature_set, args, outdir)
    if not bmi_obs.empty:
        candidates = candidates[~candidates["main_domain"].astype(str).eq("BMI")].copy()
        candidates = pd.concat([candidates, bmi_obs], ignore_index=True, sort=False)

    candidates = candidates[candidates["main_domain"].isin(MAIN_DOMAINS + QA_DOMAINS)].copy()
    candidates.to_csv(outdir / f"Figure6_all_candidates_classified_{args.feature_set}.csv", index=False)

    selected_main = select_top_fdr_first(
        candidates, MAIN_DOMAINS, args.fdr_alpha, fluid_main_min_n=args.fluid_main_min_n
    )
    # PATCH_DROP_DUP_ADDECODE_EXPLORATORY_PC_SELECTED:
    # If an AD-DECODE transcriptomic PC appears in both curated and exploratory
    # molecular selections, keep it only in the curated molecular column.
    if not selected_main.empty:
        dup_mask = (
            selected_main["cohort"].astype(str).eq("AD_DECODE")
            & selected_main["main_domain"].astype(str).eq("Exploratory molecular")
            & selected_main["variable"].astype(str).str.match(r"PC\d+_z_clean", na=False)
        )
        selected_main = selected_main.loc[~dup_mask].copy()
    selected_main.to_csv(outdir / f"Figure6_selected_top_per_cell_FDRfirst_NS_{args.feature_set}.csv", index=False)

    # BMI QA: print/write first paired cBAG-BMI values for the selected BMI candidate in each cohort.
    # This is intentionally run after selected_main is finalized, so it audits the exact BMI variable
    # used in the main Figure 6 heatmap.
    try:
        bmi_qa_rows = []
        bmi_head_rows = []
        selected_bmi = selected_main[selected_main["main_domain"].astype(str).eq("BMI")].copy()
        for _, rr in selected_bmi.iterrows():
            cohort = str(rr.get("cohort", ""))
            var = str(rr.get("variable", ""))
            merged_path = fig6_dir / "merged_tables" / f"merged_metadata_screening_{args.feature_set}_{cohort}.csv"
            if not merged_path.exists():
                bmi_qa_rows.append({
                    "cohort": cohort, "selected_variable": var, "status": "missing_merged_table",
                    "merged_path": str(merged_path)
                })
                continue

            mdf = pd.read_csv(merged_path, low_memory=False)

            # Recreate computed BMI if that selected candidate was used.
            if var == "BMI_from_height_weight":
                bmi_series, bmi_audit = compute_addecode_bmi_from_height_weight(mdf)
                if bmi_series is None:
                    bmi_qa_rows.append({
                        "cohort": cohort, "selected_variable": var, "status": "computed_bmi_failed",
                        "merged_path": str(merged_path)
                    })
                    continue
                x = clean_bmi_numeric(bmi_series)
            elif var in mdf.columns:
                x = clean_bmi_numeric(mdf[var])
            else:
                bmi_qa_rows.append({
                    "cohort": cohort, "selected_variable": var, "status": "selected_variable_not_in_merged_table",
                    "merged_path": str(merged_path)
                })
                continue

            # Use the same cBAG preference as the main loader if available.
            cbag_candidates = [c for c in ["cBAG", "cBAG_global", "cBAG_withPCA", "cBAG_withoutPCA"] if c in mdf.columns]
            cbag_candidates += [c for c in mdf.columns if "cbag" in str(c).lower()]
            cbag_candidates = list(dict.fromkeys(cbag_candidates))
            cbag_col = cbag_candidates[0] if cbag_candidates else None
            y = clean_numeric(mdf[cbag_col]) if cbag_col else pd.Series(np.nan, index=mdf.index)

            tmp = pd.DataFrame({
                "cohort": cohort,
                "selected_variable": var,
                "display_label": rr.get("display_label", ""),
                "cBAG_column": cbag_col,
                "cBAG": y,
                "BMI_value": x,
            }).dropna(subset=["cBAG", "BMI_value"])

            bmi_qa_rows.append({
                "cohort": cohort,
                "selected_variable": var,
                "display_label": rr.get("display_label", ""),
                "cBAG_column": cbag_col,
                "n_all_nonmissing_BMI": int(x.notna().sum()),
                "n_paired_cBAG_BMI": int(len(tmp)),
                "n_unique_BMI": int(x.nunique(dropna=True)),
                "BMI_min": float(x.min(skipna=True)) if x.notna().any() else np.nan,
                "BMI_max": float(x.max(skipna=True)) if x.notna().any() else np.nan,
                "BMI_mean": float(x.mean(skipna=True)) if x.notna().any() else np.nan,
                "BMI_sd": float(x.std(skipna=True)) if x.notna().any() else np.nan,
                "selected_r": rr.get("pearson_r", np.nan),
                "selected_p": rr.get("pearson_p", np.nan),
                "selected_q": rr.get("fdr_q", np.nan),
                "status": "ok",
                "merged_path": str(merged_path),
            })

            head = tmp.head(12).copy()
            head.insert(0, "row_number_in_head", range(1, len(head) + 1))
            bmi_head_rows.append(head)

        bmi_qa = pd.DataFrame(bmi_qa_rows)
        bmi_qa.to_csv(outdir / f"QA_BMI_selected_top_candidates_summary_{args.feature_set}.csv", index=False)

        if bmi_head_rows:
            bmi_head = pd.concat(bmi_head_rows, ignore_index=True)
        else:
            bmi_head = pd.DataFrame()
        bmi_head.to_csv(outdir / f"QA_BMI_selected_top_candidates_head_{args.feature_set}.csv", index=False)

        print("\nBMI QA: selected top BMI candidates")
        if not bmi_qa.empty:
            print(bmi_qa.to_string(index=False))
        print("\nBMI QA: head of paired cBAG and selected BMI values")
        if not bmi_head.empty:
            print(bmi_head.to_string(index=False))
        else:
            print("No paired BMI head rows available.")
    except Exception as e:
        print(f"[WARN] BMI QA failed: {e}")

    selected_qa = select_top_fdr_first(candidates, QA_DOMAINS, args.fdr_alpha)
    selected_qa.to_csv(outdir / f"QA_selected_top_imaging_hippocampus_network_{args.feature_set}.csv", index=False)

    selected_fluid = select_top_fdr_first(
        candidates, ["Curated fluid / transcriptomic"], args.fdr_alpha,
        fluid_main_min_n=args.fluid_main_min_n,
    )
    selected_fluid.to_csv(outdir / f"QA_selected_curated_fluid_transcriptomic_{args.feature_set}.csv", index=False)

    selected_molecular = select_top_fdr_first(
        candidates, ["Exploratory molecular"], args.fdr_alpha,
        fluid_main_min_n=args.fluid_main_min_n,
    )
    selected_molecular.to_csv(outdir / f"QA_selected_exploratory_molecular_{args.feature_set}.csv", index=False)

    top_main = top_k_by_domain(candidates, MAIN_DOMAINS, args.top_k)
    top_main.to_csv(outdir / f"QA_top{args.top_k}_main_domains_{args.feature_set}.csv", index=False)

    top_qa = top_k_by_domain(candidates, QA_DOMAINS, args.top_k)
    top_qa.to_csv(outdir / f"QA_top{args.top_k}_imaging_hippocampus_network_{args.feature_set}.csv", index=False)

    agreement = agreement_summary(selected_main, MAIN_DOMAINS)
    agreement.to_csv(outdir / f"Figure6_cross_cohort_FDR_agreement_{args.feature_set}.csv", index=False)

    print("[6/10] BMI / metabolic QA and targeted QA")
    bmi_inventory(fig6_dir, args.feature_set, args, outdir)
    bmi_hist = plot_selected_bmi_histograms(fig6_dir, args.feature_set, args, outdir, selected_main)
    if not bmi_hist.empty:
        print("BMI histogram QA saved:", outdir / f"QA_BMI_histogram_summary_{args.feature_set}.csv")
    write_domain_qa(candidates, outdir, args.feature_set, args.top_k)

    print("[7/10] Supplementary broad individual-column discovery")
    supp_broad = write_supplementary_discovery_tables(fig6_dir, args.feature_set, args, outdir)

    print("[8/10] Recomputing AUCs")
    auc_df, curves = compute_auc_table(fig6_dir, args.feature_set, args, outdir)

    print("[9/11] Drawing main and QA heatmaps")
    main_base = outdir / f"Figure6A_multidomain_top_hit_FDRfirst_NS_{args.feature_set}"
    draw_main_heatmap(selected_main, agreement, main_base, args)

    auc_base = outdir / f"Figure6B_AUC_APOE4_CogImpairment_Sex_{args.feature_set}"
    draw_auc_heatmap(auc_df, auc_base, args)

    if not selected_qa.empty:
        draw_qa_heatmap(
            selected_qa,
            "Imaging / hippocampus / network QA",
            outdir / f"QA_imaging_hippocampus_network_top_hit_{args.feature_set}",
            "QA. Imaging / hippocampal / network biological-validation hits",
            args,
        )

    if not selected_fluid.empty:
        draw_qa_heatmap(
            selected_fluid,
            "Curated fluid / transcriptomic",
            outdir / f"QA_curated_fluid_transcriptomic_top_hit_{args.feature_set}",
            "QA. Curated fluid biomarkers / transcriptomic PCs",
            args,
        )

    if not selected_molecular.empty:
        draw_qa_heatmap(
            selected_molecular,
            "Exploratory molecular",
            outdir / f"QA_exploratory_molecular_top_hit_{args.feature_set}",
            "QA. Exploratory non-curated molecular biomarkers",
            args,
        )

    if not curves.empty:
        draw_auc_roc_grid(curves, outdir / f"Figure6B_ROC_grid_APOE4_CogImpairment_Sex_{args.feature_set}", args)

    draw_supplementary_fdr_top_hits_heatmap(
        supp_broad,
        outdir / f"SupplementaryFigure_exploratory_FDR_top_hits_by_domain_{args.feature_set}",
        args,
    )

    write_all_fdr_hits_by_category(supp_broad, outdir, args.feature_set)

    draw_selected_supplementary_heatmap(
        supp_broad,
        ["diabetes_glucose_insulin", "lipids_cholesterol", "inflammation_immune"],
        outdir / f"SupplementaryFigure_diabetes_lipids_inflammation_FDR_top_hits_{args.feature_set}",
        "Supplementary exploratory metabolic / lipid / inflammatory hits\nStrongest FDR-significant association per cohort-category",
        args,
    )
    draw_selected_supplementary_heatmap(
        supp_broad,
        ["extended_cardiovascular_risk", "lifestyle_smoking"],
        outdir / f"SupplementaryFigure_cardiovascular_risk_lifestyle_FDR_top_hits_{args.feature_set}",
        "Supplementary exploratory cardiovascular-risk / lifestyle hits\nStrongest FDR-significant association per cohort-category",
        args,
    )
    draw_selected_supplementary_heatmap(
        supp_broad,
        ["biomarker_abeta", "biomarker_tau_ptau", "biomarker_gfap", "biomarker_nfl", "transcriptomic_PC_AD_DECODE"],
        outdir / f"SupplementaryFigure_biomarker_subfamilies_FDR_top_hits_{args.feature_set}",
        "Supplementary biomarker subfamily hits\nStrongest FDR-significant association per cohort-category",
        args,
    )

    draw_volcano_plots(candidates, supp_broad, outdir, args.feature_set, args)

    draw_domain_hit_heatmaps(candidates, supp_broad, outdir, args.feature_set, args)

    draw_directionality_scatter_grid(
        fig6_dir, args.feature_set, selected_main,
        outdir / f"SupplementaryFigure_directionality_top_hits_current_domains_{args.feature_set}",
        args,
    )

    print("[10/11] Domain volcano panels, hit heatmaps, and directionality scatter grid saved")

    print("[11/11] Combined figure and manifest")
    combined_base = outdir / f"Figure6_FINAL_combined_{args.feature_set}"
    draw_combined(Path(str(main_base) + ".png"), Path(str(auc_base) + ".png"), combined_base, args)

    if not getattr(args, "save_component_figures", False):
        for _base in [main_base, auc_base]:
            for _ext in [".png", ".pdf"]:
                _f = Path(str(_base) + _ext)
                if _f.exists():
                    try:
                        _f.unlink()
                    except Exception:
                        pass

    for ext in ["png", "pdf"]:
        f = fig6_dir / "auc" / f"Figure6_AUC_ROC_{args.feature_set}.{ext}"
        if f.exists():
            shutil.copy2(f, outdir / f"Reference_existing_ROC_panel_{f.name}")

    manifest = []
    for f in sorted(outdir.glob("*")):
        if f.is_file():
            manifest.append({"file": f.name, "path": str(f), "size_bytes": f.stat().st_size})
    pd.DataFrame(manifest).to_csv(outdir / "Figure6build_manifest.csv", index=False)

    print("\n[DONE]")
    print("Output directory:")
    print(outdir)
    print("\nMain outputs:")
    print(outdir / f"Figure6A_multidomain_top_hit_FDRfirst_NS_{args.feature_set}.png")
    print(outdir / f"Figure6B_AUC_APOE4_CogImpairment_Sex_{args.feature_set}.png")
    print(outdir / f"Figure6_FINAL_combined_{args.feature_set}.png")
    print("\nKey QA:")
    print(outdir / f"Figure6_selected_top_per_cell_FDRfirst_NS_{args.feature_set}.csv")
    print(outdir / f"QA_cognitive_two_tier_candidates_{args.feature_set}.csv")
    print(outdir / f"QA_cognitive_composite_candidates_{args.feature_set}.csv")
    print(outdir / f"QA_cognitive_individual_test_candidates_{args.feature_set}.csv")
    print(outdir / f"Supplementary_cognitive_tests_top_per_5domain_{args.feature_set}.csv")
    print(outdir / f"QA_fluid_curated_candidates_used_by_Figure6_{args.feature_set}.csv")
    print(outdir / f"QA_exploratory_molecular_candidates_{args.feature_set}.csv")
    print(outdir / f"Supplementary_exploratory_molecular_top_per_subfamily_{args.feature_set}.csv")
    print(outdir / f"Supplementary_selected_biomarker_top_hits_Abeta_tau_GFAP_NfL_PC_{args.feature_set}.csv")
    print(outdir / f"Supplementary_top{args.supp_top_k}_individual_columns_by_category_{args.feature_set}.csv")
    print(outdir / f"Supplementary_counts_by_cohort_category_{args.feature_set}.csv")
    print(outdir / f"Supplementary_selected_FDR_only_top_hit_by_category_{args.feature_set}.csv")
    print(outdir / f"Supplementary_all_FDR_hits_by_category_{args.feature_set}.csv")
    print(outdir / f"SupplementaryFigure_exploratory_FDR_top_hits_by_domain_{args.feature_set}.png")
    print(outdir / f"SupplementaryFigure_diabetes_lipids_inflammation_FDR_top_hits_{args.feature_set}.png")
    print(outdir / f"SupplementaryFigure_cardiovascular_risk_lifestyle_FDR_top_hits_{args.feature_set}.png")
    print(outdir / f"SupplementaryFigure_biomarker_subfamilies_FDR_top_hits_{args.feature_set}.png")
    print(outdir / "Supplementary_volcano_plots")
    print(outdir / "Supplementary_domain_hit_heatmaps")
    print(outdir / f"QA_top{args.top_k}_main_domains_{args.feature_set}.csv")
    print(outdir / f"QA_targeted_tau_ptau_{args.feature_set}.csv")
    print(outdir / f"QA_targeted_gfap_{args.feature_set}.csv")
    print(outdir / f"QA_targeted_nfl_{args.feature_set}.csv")
    print(outdir / f"QA_targeted_abeta_{args.feature_set}.csv")
    print(outdir / f"QA_targeted_diabetes_metabolic_lipids_{args.feature_set}.csv")
    print(outdir / f"QA_targeted_inflammation_immune_{args.feature_set}.csv")
    print(outdir / f"QA_BMI_column_inventory_{args.feature_set}.csv")
    print(outdir / f"QA_BMI_histogram_summary_{args.feature_set}.csv")
    print(outdir / "QA_BMI_histograms")
    print(outdir / f"Figure6_AUC_recomputed_APOE4_CogImpairment_Sex_{args.feature_set}.csv")
    print(outdir / f"Figure6B_AUC_bootstrap_summary_{args.feature_set}.csv")


if __name__ == "__main__":
    main()
