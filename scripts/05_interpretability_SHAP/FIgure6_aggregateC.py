#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 MASTER v3
==================

Build cross-cohort SHAP result tables and PNG figures for:
  - global features
  - node-feature contributors
  - top edges
  - edge-derived regions

Then assemble:
  - Figure 6 (main model): global + edges + regions + nodes
  - Supplementary Figure 6 (all models): global + edges + regions + nodes

DEFAULT USAGE
-------------
Just run:

    python figure6_master_v3_complete_cross_cohort_shap.py

No parameters are required.

DEFAULT INPUT ROOT
------------------
/mnt/newStor/paros/paros_WORK/ines/results/
BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/
Figure6_SHAP

EXPECTED RAW INPUT FOLDERS
--------------------------
<out-base>/<cohort>/<model>/global_feature_shap/
<out-base>/<cohort>/<model>/node_feature_shap/
<out-base>/<cohort>/<model>/edge_shap/

The script will use summary CSVs when they already exist, otherwise it will
rebuild summaries from subject-level CSVs.

OUTPUTS
-------
Per model:
  <out-base>/cross_cohort_aggregated/<model>/

with CSVs and PNGs for:
  - global contributors
  - node-feature contributors
  - top edges
  - edge-derived regions
  - global beeswarm-style plot
  - node-feature beeswarm-style plot for Supplementary Figure 6C

Final figures:
  <out-base>/cross_cohort_aggregated/final_figures/
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Defaults
# =============================================================================
DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_OUT_BASE = (
    Path(DEFAULT_WORK)
    / "ines/results"
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Figure6_SHAP"
)

COHORT_CONFIG = {
    "ADNI": {"cohort_slug": "adni", "display": "ADNI"},
    "ADRC": {"cohort_slug": "adrc", "display": "ADRC"},
    "HABS": {"cohort_slug": "habs", "display": "HABS"},
    "AD_DECODE": {"cohort_slug": "addecode", "display": "AD-DECODE"},
}
DEFAULT_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

PREFERRED_MODEL_ORDER = [
    "full",
    "imaging_only",
    "clinical_only",
    "non_imaging_only",
    "multimodal",
    "graph_only",
]

MODEL_LABELS = {
    "full": "Full model",
    "imaging_only": "Imaging-only model",
    "clinical_only": "Clinical-only model",
    "non_imaging_only": "Non-imaging model",
    "multimodal": "Multimodal model",
    "graph_only": "Graph-only model",
}

DEFAULT_KEYWORDS = [
    "APOE", "APOE4", "genotype", "BMI", "blood pressure", "systolic",
    "diastolic", "hypertension", "SBP", "DBP",
]


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build complete cross-cohort Figure 6 SHAP outputs with globals, nodes, edges, and regions."
    )
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE))
    p.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    p.add_argument("--models", "--feature-sets", dest="models", default=None)
    p.add_argument("--main-model", default="full")
    p.add_argument("--top-n-global", type=int, default=30)
    p.add_argument("--top-n-node", type=int, default=25)
    p.add_argument("--top-n-edge", type=int, default=25)
    p.add_argument("--top-n-region", type=int, default=25)
    p.add_argument("--supp-top-n-global", type=int, default=10)
    p.add_argument("--supp-top-n-node", type=int, default=10)
    p.add_argument("--supp-top-n-region", type=int, default=10)
    p.add_argument("--sort-by-support-first", action="store_true")
    p.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS))
    p.add_argument("--dpi", type=int, default=350)
    p.add_argument("--skip-beeswarm", action="store_true")
    p.add_argument("--skip-final-assembly", action="store_true")
    return p.parse_args()


# =============================================================================
# General helpers
# =============================================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sem(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def wrap_label(x: str, width: int = 34) -> str:
    x = str(x).replace(" | ", "\n")
    if len(x) <= width:
        return x
    return "\n".join(textwrap.wrap(x, width=width, break_long_words=False))


def discover_models(out_base: Path, cohorts: List[str]) -> List[str]:
    found = set()
    for cohort in cohorts:
        cfg = COHORT_CONFIG.get(cohort)
        if cfg is None:
            continue
        cohort_dir = out_base / cfg["cohort_slug"]
        if not cohort_dir.exists():
            continue
        for child in cohort_dir.iterdir():
            if not child.is_dir():
                continue
            if (
                (child / "global_feature_shap").exists()
                or (child / "node_feature_shap").exists()
                or (child / "edge_shap").exists()
            ):
                found.add(child.name)

    ordered = [x for x in PREFERRED_MODEL_ORDER if x in found]
    ordered.extend(sorted(x for x in found if x not in set(ordered)))
    return ordered


def cohort_model_dir(out_base: Path, cohort: str, model: str, subdir: str) -> Path:
    return out_base / COHORT_CONFIG[cohort]["cohort_slug"] / model / subdir


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_barplot(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    outpath: Path,
    top_n: int,
    support_col: Optional[str] = None,
    label_width: int = 36,
    dpi: int = 350,
) -> Optional[str]:
    if df is None or df.empty or label_col not in df.columns or value_col not in df.columns:
        return None

    plot_df = df.dropna(subset=[label_col, value_col]).head(top_n).copy()
    if plot_df.empty:
        return None

    plot_df = plot_df.iloc[::-1]
    labels = [wrap_label(x, width=label_width) for x in plot_df[label_col]]

    fig_h = max(8, 0.34 * len(plot_df))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    bars = ax.barh(labels, plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    xmax = float(np.nanmax(plot_df[value_col])) if len(plot_df) else 0.0
    if support_col and support_col in plot_df.columns and plot_df[support_col].notna().any():
        for bar, support in zip(bars, plot_df[support_col]):
            if pd.isna(support):
                continue
            ax.text(
                bar.get_width() + max(xmax * 0.01, 1e-6),
                bar.get_y() + bar.get_height() / 2,
                f"n={int(support)}",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(outpath)


# =============================================================================
# GLOBAL
# =============================================================================
def load_global_subjects(out_base: Path, cohort: str, model: str) -> Optional[pd.DataFrame]:
    gdir = cohort_model_dir(out_base, cohort, model, "global_feature_shap")
    files = sorted(gdir.glob("global_feature_shap_subject_*.csv"))
    frames = []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or not {"feature_name", "SHAP_val"}.issubset(df.columns):
            continue
        df = df.copy()
        if "abs_SHAP" not in df.columns:
            df["abs_SHAP"] = pd.to_numeric(df["SHAP_val"], errors="coerce").abs()
        df["subject"] = f.stem.replace("global_feature_shap_subject_", "")
        df["cohort"] = cohort
        df["feature_set"] = model
        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False) if frames else None


def load_or_build_global_summary(out_base: Path, cohort: str, model: str) -> Optional[pd.DataFrame]:
    gdir = cohort_model_dir(out_base, cohort, model, "global_feature_shap")
    summary_path = gdir / "global_feature_shap_summary_all_subjects.csv"

    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = None
        if df is not None and not df.empty and "feature_name" in df.columns:
            df = df.copy()
            df.insert(0, "cohort", cohort)
            df.insert(1, "feature_set", model)
            return df

    long_df = load_global_subjects(out_base, cohort, model)
    if long_df is None or long_df.empty:
        return None

    long_df = coerce_numeric(long_df, ["SHAP_val", "abs_SHAP", "feature_value"])
    summary = (
        long_df.groupby("feature_name", as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            sd_abs_SHAP=("abs_SHAP", "std"),
            sem_abs_SHAP=("abs_SHAP", sem),
            mean_SHAP=("SHAP_val", "mean"),
            sd_SHAP=("SHAP_val", "std"),
            sem_SHAP=("SHAP_val", sem),
            median_SHAP=("SHAP_val", "median"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("mean_abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "cohort", cohort)
    summary.insert(1, "feature_set", model)
    return summary


def make_global_beeswarm(
    out_base: Path,
    cohorts: List[str],
    model: str,
    ranked: pd.DataFrame,
    outdir: Path,
    top_n: int,
    dpi: int,
) -> Optional[str]:
    if ranked is None or ranked.empty:
        return None

    keep = ranked.head(top_n)["feature_name"].astype(str).tolist()
    frames = []
    for cohort in cohorts:
        df = load_global_subjects(out_base, cohort, model)
        if df is not None and not df.empty:
            df = df[df["feature_name"].astype(str).isin(keep)].copy()
            if not df.empty:
                frames.append(df)

    if not frames:
        return None

    long_df = pd.concat(frames, ignore_index=True, sort=False)
    long_df = coerce_numeric(long_df, ["SHAP_val", "feature_value"])
    long_df = long_df.dropna(subset=["feature_name", "SHAP_val"])
    if long_df.empty:
        return None

    order = keep[::-1]
    y_map = {f: i for i, f in enumerate(order)}
    rng = np.random.default_rng(12345)
    long_df["y"] = long_df["feature_name"].astype(str).map(y_map) + rng.normal(0, 0.08, size=len(long_df))

    fig_h = max(7, 0.42 * len(order))
    fig, ax = plt.subplots(figsize=(11, fig_h))

    cvals = long_df["feature_value"] if "feature_value" in long_df.columns else pd.Series(dtype=float)
    if len(cvals) > 0 and cvals.notna().any():
        sc = ax.scatter(long_df["SHAP_val"], long_df["y"], c=cvals, s=9, alpha=0.65)
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Feature value", fontsize=9)
    else:
        ax.scatter(long_df["SHAP_val"], long_df["y"], s=9, alpha=0.65)

    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([wrap_label(x, 38) for x in order], fontsize=8)
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Global feature")
    ax.set_title(f"Global SHAP beeswarm-style plot ({model})")
    fig.tight_layout()

    outpath = outdir / f"Supplementary_Figure6_cross_cohort_{model}_global_feature_SHAP_beeswarm.png"
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(outpath)


def aggregate_global(
    out_base: Path,
    cohorts: List[str],
    model: str,
    top_n: int,
    sort_by_support_first: bool,
    keywords: List[str],
    dpi: int,
    make_beeswarm_flag: bool,
) -> List[str]:
    outdir = ensure_dir(out_base / "cross_cohort_aggregated" / model)
    prefix = f"Supplementary_Figure6_cross_cohort_{model}_global_feature_SHAP_union"

    frames = []
    log_rows = []
    for cohort in cohorts:
        df = load_or_build_global_summary(out_base, cohort, model)
        if df is None or df.empty:
            log_rows.append({"cohort": cohort, "status": "missing", "n_features": 0})
        else:
            frames.append(df)
            log_rows.append({"cohort": cohort, "status": "loaded", "n_features": int(df['feature_name'].nunique())})

    load_log = pd.DataFrame(log_rows)
    load_log_path = outdir / f"{prefix}_input_load_log.csv"
    load_log.to_csv(load_log_path, index=False)

    if not frames:
        return [str(load_log_path)]

    by_cohort = pd.concat(frames, ignore_index=True, sort=False)
    by_cohort = coerce_numeric(by_cohort, ["mean_abs_SHAP", "mean_SHAP", "n_subjects"])
    by_cohort_path = outdir / f"{prefix}_by_cohort.csv"
    by_cohort.to_csv(by_cohort_path, index=False)

    ranked = (
        by_cohort.dropna(subset=["feature_name", "mean_abs_SHAP"])
        .groupby("feature_name", as_index=False)
        .agg(
            mean_abs_SHAP_across_cohorts=("mean_abs_SHAP", "mean"),
            sd_abs_SHAP_across_cohorts=("mean_abs_SHAP", "std"),
            mean_signed_SHAP_across_cohorts=("mean_SHAP", "mean"),
            median_signed_SHAP_across_cohorts=("mean_SHAP", "median"),
            n_cohorts=("cohort", "nunique"),
            total_n_subjects=("n_subjects", "sum"),
            cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
        )
    )

    if sort_by_support_first:
        ranked = ranked.sort_values(["n_cohorts", "mean_abs_SHAP_across_cohorts"], ascending=[False, False])
    else:
        ranked = ranked.sort_values("mean_abs_SHAP_across_cohorts", ascending=False)

    ranked_path = outdir / f"{prefix}_ranked_all_features.csv"
    ranked.to_csv(ranked_path, index=False)

    top = ranked.head(top_n).copy()
    top_path = outdir / f"{prefix}_top{top_n}.csv"
    top.to_csv(top_path, index=False)

    max_support_df = ranked[ranked["n_cohorts"] == ranked["n_cohorts"].max()].copy() if len(ranked) else ranked.copy()
    max_support_path = outdir / f"{prefix}_max_cohort_support_features.csv"
    max_support_df.to_csv(max_support_path, index=False)

    kw = [k.strip().lower() for k in keywords if k.strip()]
    keyword_subset = ranked[
        ranked["feature_name"].astype(str).str.lower().apply(lambda s: any(k in s for k in kw))
    ].copy() if kw else ranked.iloc[0:0].copy()
    keyword_subset_path = outdir / f"{prefix}_keyword_subset.csv"
    keyword_subset.to_csv(keyword_subset_path, index=False)

    pngs = []
    p = save_barplot(
        ranked,
        label_col="feature_name",
        value_col="mean_abs_SHAP_across_cohorts",
        title=f"Top cross-cohort global SHAP contributors ({model})",
        xlabel="Mean |SHAP| across cohorts",
        outpath=outdir / f"{prefix}_top{top_n}_barplot.png",
        top_n=top_n,
        support_col="n_cohorts",
        label_width=40,
        dpi=dpi,
    )
    if p:
        pngs.append(p)

    if make_beeswarm_flag:
        b = make_global_beeswarm(out_base, cohorts, model, ranked, outdir, top_n=min(20, top_n), dpi=dpi)
        if b:
            pngs.append(b)

    manifest_path = outdir / f"{prefix}_manifest.csv"
    outputs = [
        str(by_cohort_path),
        str(ranked_path),
        str(top_path),
        str(max_support_path),
        str(keyword_subset_path),
        str(load_log_path),
        *pngs,
    ]
    pd.DataFrame([{"model": model, "outputs": ";".join(outputs)}]).to_csv(manifest_path, index=False)
    outputs.append(str(manifest_path))
    return outputs


# =============================================================================
# NODE
# =============================================================================
def make_node_feature_label(df: pd.DataFrame) -> Optional[pd.Series]:
    if "node_feature_label" in df.columns:
        return df["node_feature_label"].astype(str)
    if {"node_label", "feature_name"}.issubset(df.columns):
        return df["node_label"].astype(str) + " | " + df["feature_name"].astype(str)
    if {"Structure", "feature_name"}.issubset(df.columns):
        return df["Structure"].astype(str) + " | " + df["feature_name"].astype(str)
    if {"structure", "feature_name"}.issubset(df.columns):
        return df["structure"].astype(str) + " | " + df["feature_name"].astype(str)
    return None


def load_or_build_node_summary(out_base: Path, cohort: str, model: str) -> Optional[pd.DataFrame]:
    ndir = cohort_model_dir(out_base, cohort, model, "node_feature_shap")
    summary_path = ndir / "node_feature_shap_summary_all_subjects.csv"

    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = None
        if df is not None and not df.empty:
            lab = make_node_feature_label(df)
            if lab is None:
                return None
            df = df.copy()
            df["node_feature_label"] = lab
            df.insert(0, "cohort", cohort)
            df.insert(1, "feature_set", model)
            return df

    files = sorted(ndir.glob("node_feature_shap_subject_*.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or "SHAP_val" not in df.columns:
            continue
        lab = make_node_feature_label(df)
        if lab is None:
            continue
        df = df.copy()
        df["node_feature_label"] = lab
        if "abs_SHAP" not in df.columns:
            df["abs_SHAP"] = pd.to_numeric(df["SHAP_val"], errors="coerce").abs()
        df["subject"] = f.stem.replace("node_feature_shap_subject_", "")
        frames.append(df)

    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = coerce_numeric(all_df, ["SHAP_val", "abs_SHAP"])

    summary = (
        all_df.groupby("node_feature_label", as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            sd_abs_SHAP=("abs_SHAP", "std"),
            sem_abs_SHAP=("abs_SHAP", sem),
            mean_SHAP=("SHAP_val", "mean"),
            sd_SHAP=("SHAP_val", "std"),
            sem_SHAP=("SHAP_val", sem),
            median_SHAP=("SHAP_val", "median"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("mean_abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "cohort", cohort)
    summary.insert(1, "feature_set", model)
    return summary


def load_node_subjects(out_base: Path, cohort: str, model: str) -> Optional[pd.DataFrame]:
    ndir = cohort_model_dir(out_base, cohort, model, "node_feature_shap")
    files = sorted(ndir.glob("node_feature_shap_subject_*.csv"))
    frames = []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        if df.empty or "SHAP_val" not in df.columns:
            continue

        lab = make_node_feature_label(df)
        if lab is None:
            continue

        df = df.copy()
        df["node_feature_label"] = lab
        if "abs_SHAP" not in df.columns:
            df["abs_SHAP"] = pd.to_numeric(df["SHAP_val"], errors="coerce").abs()

        df["subject"] = f.stem.replace("node_feature_shap_subject_", "")
        df["cohort"] = cohort
        df["feature_set"] = model
        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False) if frames else None


def make_node_beeswarm(
    out_base: Path,
    cohorts: List[str],
    model: str,
    ranked: pd.DataFrame,
    outdir: Path,
    top_n: int,
    dpi: int,
) -> Optional[str]:
    """
    Supplementary Figure 6C candidate:
    subject-level cross-cohort node-feature SHAP beeswarm for top node features.

    This does not recompute SHAP. It only reads existing
    node_feature_shap_subject_*.csv files.
    """
    if ranked is None or ranked.empty or "node_feature_label" not in ranked.columns:
        return None

    keep = ranked.head(top_n)["node_feature_label"].astype(str).tolist()
    frames = []

    for cohort in cohorts:
        df = load_node_subjects(out_base, cohort, model)
        if df is not None and not df.empty:
            df = df[df["node_feature_label"].astype(str).isin(keep)].copy()
            if not df.empty:
                frames.append(df)

    if not frames:
        return None

    long_df = pd.concat(frames, ignore_index=True, sort=False)
    long_df = coerce_numeric(long_df, ["SHAP_val", "feature_value"])
    long_df = long_df.dropna(subset=["node_feature_label", "SHAP_val"])
    if long_df.empty:
        return None

    order = keep[::-1]
    y_map = {f: i for i, f in enumerate(order)}
    rng = np.random.default_rng(12345)
    long_df["y"] = long_df["node_feature_label"].astype(str).map(y_map) + rng.normal(0, 0.08, size=len(long_df))

    fig_h = max(8, 0.50 * len(order))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    cvals = long_df["feature_value"] if "feature_value" in long_df.columns else pd.Series(dtype=float)
    if len(cvals) > 0 and cvals.notna().any():
        sc = ax.scatter(long_df["SHAP_val"], long_df["y"], c=cvals, s=8, alpha=0.60)
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Feature value", fontsize=9)
    else:
        ax.scatter(long_df["SHAP_val"], long_df["y"], s=8, alpha=0.60)

    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([wrap_label(x, 46) for x in order], fontsize=8)
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Node-feature contributor")
    ax.set_title(f"Supplementary Figure 6C. Node-feature SHAP beeswarm ({model})")
    fig.tight_layout()

    outpath = outdir / f"Supplementary_Figure6_cross_cohort_{model}_node_feature_SHAP_beeswarm.png"
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(outpath)


def aggregate_node(out_base: Path, cohorts: List[str], model: str, top_n: int, dpi: int) -> List[str]:
    outdir_model = ensure_dir(out_base / "cross_cohort_aggregated" / model)
    outdir_flat = ensure_dir(out_base / "cross_cohort_aggregated")
    prefix = f"Supplementary_Figure6_cross_cohort_{model}_node_feature_SHAP"

    frames = []
    log_rows = []
    for cohort in cohorts:
        df = load_or_build_node_summary(out_base, cohort, model)
        if df is None or df.empty:
            log_rows.append({"cohort": cohort, "status": "missing", "n_node_features": 0})
        else:
            frames.append(df)
            log_rows.append({"cohort": cohort, "status": "loaded", "n_node_features": int(len(df))})

    load_log_path = outdir_model / f"{prefix}_input_load_log.csv"
    pd.DataFrame(log_rows).to_csv(load_log_path, index=False)

    if not frames:
        return [str(load_log_path)]

    by_cohort = pd.concat(frames, ignore_index=True, sort=False)
    by_cohort = coerce_numeric(by_cohort, ["mean_abs_SHAP", "mean_SHAP", "n_subjects"])
    by_cohort_path = outdir_model / f"{prefix}_summary_by_cohort.csv"
    by_cohort.to_csv(by_cohort_path, index=False)

    ranked = (
        by_cohort.dropna(subset=["node_feature_label", "mean_abs_SHAP"])
        .groupby("node_feature_label", as_index=False)
        .agg(
            mean_abs_SHAP_across_cohorts=("mean_abs_SHAP", "mean"),
            sd_abs_SHAP_across_cohorts=("mean_abs_SHAP", "std"),
            mean_signed_SHAP_across_cohorts=("mean_SHAP", "mean"),
            median_signed_SHAP_across_cohorts=("mean_SHAP", "median"),
            n_cohorts=("cohort", "nunique"),
            total_n_subjects=("n_subjects", "sum"),
            cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .sort_values(["n_cohorts", "mean_abs_SHAP_across_cohorts"], ascending=[False, False])
        .reset_index(drop=True)
    )

    collapsed_model_path = outdir_model / f"{prefix}_summary_collapsed.csv"
    collapsed_flat_path = outdir_flat / f"{prefix}_summary_collapsed.csv"
    ranked.to_csv(collapsed_model_path, index=False)
    ranked.to_csv(collapsed_flat_path, index=False)

    top = ranked.head(top_n).copy()
    top_path = outdir_model / f"{prefix}_top{top_n}.csv"
    top.to_csv(top_path, index=False)

    barplot_path = save_barplot(
        ranked,
        label_col="node_feature_label",
        value_col="mean_abs_SHAP_across_cohorts",
        title=f"Top cross-cohort node-feature SHAP contributors ({model})",
        xlabel="Mean |SHAP| across cohorts",
        outpath=outdir_model / f"{prefix}_top{top_n}_barplot.png",
        top_n=top_n,
        support_col="n_cohorts",
        label_width=44,
        dpi=dpi,
    )

    node_beeswarm_path = make_node_beeswarm(
        out_base=out_base,
        cohorts=cohorts,
        model=model,
        ranked=ranked,
        outdir=outdir_model,
        top_n=min(15, top_n),
        dpi=dpi,
    )

    manifest_path = outdir_model / f"{prefix}_manifest.csv"
    outputs = [
        str(by_cohort_path),
        str(collapsed_model_path),
        str(collapsed_flat_path),
        str(top_path),
        str(load_log_path),
    ]
    if barplot_path:
        outputs.append(barplot_path)
    if node_beeswarm_path:
        outputs.append(node_beeswarm_path)
    pd.DataFrame([{"model": model, "outputs": ";".join(outputs)}]).to_csv(manifest_path, index=False)
    outputs.append(str(manifest_path))
    return outputs


# =============================================================================
# EDGE + REGION
# =============================================================================
def make_edge_label(df: pd.DataFrame) -> Optional[pd.Series]:
    if "Edge" in df.columns:
        return df["Edge"].astype(str)
    if {"Structure_i", "Structure_j"}.issubset(df.columns):
        return df["Structure_i"].astype(str) + " -- " + df["Structure_j"].astype(str)
    if {"structure_i", "structure_j"}.issubset(df.columns):
        return df["structure_i"].astype(str) + " -- " + df["structure_j"].astype(str)
    if {"Node_i", "Node_j"}.issubset(df.columns):
        return df["Node_i"].astype(str) + " -- " + df["Node_j"].astype(str)
    return None


def get_region_cols(df: pd.DataFrame) -> List[str]:
    pairs = []
    if {"Structure_i", "Structure_j"}.issubset(df.columns):
        pairs = [("Structure_i", "Structure_j")]
    elif {"structure_i", "structure_j"}.issubset(df.columns):
        pairs = [("structure_i", "structure_j")]
    return pairs


def load_or_build_edge_summary(out_base: Path, cohort: str, model: str) -> Optional[pd.DataFrame]:
    edir = cohort_model_dir(out_base, cohort, model, "edge_shap")
    summary_path = edir / "edge_shap_summary_all_subjects.csv"

    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = None
        if df is not None and not df.empty:
            edge_label = make_edge_label(df)
            if edge_label is None:
                return None
            df = df.copy()
            df["Edge"] = edge_label
            df.insert(0, "cohort", cohort)
            df.insert(1, "feature_set", model)
            return df

    files = sorted(edir.glob("edge_shap_subject_*.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or "SHAP_val" not in df.columns:
            continue
        edge_label = make_edge_label(df)
        if edge_label is None:
            continue
        df = df.copy()
        df["Edge"] = edge_label
        if "abs_SHAP" not in df.columns:
            df["abs_SHAP"] = pd.to_numeric(df["SHAP_val"], errors="coerce").abs()
        df["subject"] = f.stem.replace("edge_shap_subject_", "")
        frames.append(df)

    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = coerce_numeric(all_df, ["SHAP_val", "abs_SHAP"])

    group_cols = ["Edge"]
    for c in ["Structure_i", "Structure_j", "structure_i", "structure_j", "Node_i", "Node_j", "edge_feature_name"]:
        if c in all_df.columns and c not in group_cols:
            group_cols.append(c)

    summary = (
        all_df.groupby(group_cols, as_index=False)
        .agg(
            mean_abs_SHAP=("abs_SHAP", "mean"),
            sd_abs_SHAP=("abs_SHAP", "std"),
            sem_abs_SHAP=("abs_SHAP", sem),
            mean_SHAP=("SHAP_val", "mean"),
            sd_SHAP=("SHAP_val", "std"),
            sem_SHAP=("SHAP_val", sem),
            median_SHAP=("SHAP_val", "median"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("mean_abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "cohort", cohort)
    summary.insert(1, "feature_set", model)
    return summary


def aggregate_edge_and_regions(
    out_base: Path,
    cohorts: List[str],
    model: str,
    top_n_edge: int,
    top_n_region: int,
    dpi: int,
) -> List[str]:
    outdir_model = ensure_dir(out_base / "cross_cohort_aggregated" / model)
    outdir_flat = ensure_dir(out_base / "cross_cohort_aggregated")
    edge_prefix = f"Supplementary_Figure6_cross_cohort_{model}_edge_SHAP"
    region_prefix = f"Supplementary_Figure6_cross_cohort_{model}_edge_derived_region_SHAP"

    frames = []
    log_rows = []
    for cohort in cohorts:
        df = load_or_build_edge_summary(out_base, cohort, model)
        if df is None or df.empty:
            log_rows.append({"cohort": cohort, "status": "missing", "n_edges": 0})
        else:
            frames.append(df)
            log_rows.append({"cohort": cohort, "status": "loaded", "n_edges": int(len(df))})

    load_log_path = outdir_model / f"{edge_prefix}_input_load_log.csv"
    pd.DataFrame(log_rows).to_csv(load_log_path, index=False)

    if not frames:
        return [str(load_log_path)]

    by_cohort = pd.concat(frames, ignore_index=True, sort=False)
    by_cohort = coerce_numeric(by_cohort, ["mean_abs_SHAP", "mean_SHAP", "n_subjects"])
    by_cohort_path = outdir_model / f"{edge_prefix}_summary_by_cohort.csv"
    by_cohort.to_csv(by_cohort_path, index=False)

    edge_ranked = (
        by_cohort.dropna(subset=["Edge", "mean_abs_SHAP"])
        .groupby("Edge", as_index=False)
        .agg(
            mean_abs_SHAP_across_cohorts=("mean_abs_SHAP", "mean"),
            sd_abs_SHAP_across_cohorts=("mean_abs_SHAP", "std"),
            mean_signed_SHAP_across_cohorts=("mean_SHAP", "mean"),
            median_signed_SHAP_across_cohorts=("mean_SHAP", "median"),
            n_cohorts=("cohort", "nunique"),
            total_n_subjects=("n_subjects", "sum"),
            cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .sort_values(["n_cohorts", "mean_abs_SHAP_across_cohorts"], ascending=[False, False])
        .reset_index(drop=True)
    )

    edge_ranked_model_path = outdir_model / f"{edge_prefix}_ranked.csv"
    edge_ranked_flat_path = outdir_flat / f"{edge_prefix}_ranked.csv"
    edge_ranked.to_csv(edge_ranked_model_path, index=False)
    edge_ranked.to_csv(edge_ranked_flat_path, index=False)

    edge_top = edge_ranked.head(top_n_edge).copy()
    edge_top_path = outdir_model / f"{edge_prefix}_top{top_n_edge}.csv"
    edge_top.to_csv(edge_top_path, index=False)

    edge_barplot_path = save_barplot(
        edge_ranked,
        label_col="Edge",
        value_col="mean_abs_SHAP_across_cohorts",
        title=f"Top cross-cohort edge SHAP contributors ({model})",
        xlabel="Mean |SHAP| across cohorts",
        outpath=outdir_model / f"{edge_prefix}_top{top_n_edge}_barplot.png",
        top_n=top_n_edge,
        support_col="n_cohorts",
        label_width=42,
        dpi=dpi,
    )

    # derive region rankings from incident edges
    region_rows = []
    for _, row in by_cohort.iterrows():
        if pd.isna(row.get("mean_abs_SHAP")):
            continue

        if "Structure_i" in row and "Structure_j" in row:
            pairs = [("Structure_i", "Structure_j")]
        elif "structure_i" in row and "structure_j" in row:
            pairs = [("structure_i", "structure_j")]
        else:
            pairs = []

        for ri, rj in pairs:
            for reg_col in [ri, rj]:
                reg = row.get(reg_col)
                if pd.isna(reg):
                    continue
                region_rows.append(
                    {
                        "cohort": row["cohort"],
                        "feature_set": model,
                        "region": str(reg),
                        "incident_mean_abs_SHAP": row["mean_abs_SHAP"],
                        "incident_mean_signed_SHAP": row.get("mean_SHAP", np.nan),
                        "n_subjects": row.get("n_subjects", np.nan),
                    }
                )

    outputs = [
        str(by_cohort_path),
        str(edge_ranked_model_path),
        str(edge_ranked_flat_path),
        str(edge_top_path),
        str(load_log_path),
    ]
    if edge_barplot_path:
        outputs.append(edge_barplot_path)

    if region_rows:
        region_long = pd.DataFrame(region_rows)
        region_long_path = outdir_model / f"{region_prefix}_long.csv"
        region_long.to_csv(region_long_path, index=False)

        region_ranked = (
            region_long.groupby("region", as_index=False)
            .agg(
                mean_incident_abs_SHAP_across_cohorts=("incident_mean_abs_SHAP", "mean"),
                sd_incident_abs_SHAP_across_cohorts=("incident_mean_abs_SHAP", "std"),
                mean_incident_signed_SHAP_across_cohorts=("incident_mean_signed_SHAP", "mean"),
                n_cohorts=("cohort", "nunique"),
                cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
            )
            .sort_values(["n_cohorts", "mean_incident_abs_SHAP_across_cohorts"], ascending=[False, False])
            .reset_index(drop=True)
        )

        region_ranked_model_path = outdir_model / f"{region_prefix}_ranked.csv"
        region_ranked_flat_path = outdir_flat / f"{region_prefix}_ranked.csv"
        region_ranked.to_csv(region_ranked_model_path, index=False)
        region_ranked.to_csv(region_ranked_flat_path, index=False)

        region_top = region_ranked.head(top_n_region).copy()
        region_top_path = outdir_model / f"{region_prefix}_top{top_n_region}.csv"
        region_top.to_csv(region_top_path, index=False)

        region_barplot_path = save_barplot(
            region_ranked,
            label_col="region",
            value_col="mean_incident_abs_SHAP_across_cohorts",
            title=f"Top cross-cohort edge-derived regions ({model})",
            xlabel="Mean incident edge |SHAP| across cohorts",
            outpath=outdir_model / f"{region_prefix}_top{top_n_region}_barplot.png",
            top_n=top_n_region,
            support_col="n_cohorts",
            label_width=34,
            dpi=dpi,
        )

        outputs.extend([
            str(region_long_path),
            str(region_ranked_model_path),
            str(region_ranked_flat_path),
            str(region_top_path),
        ])
        if region_barplot_path:
            outputs.append(region_barplot_path)

    manifest_path = outdir_model / f"{edge_prefix}_and_{region_prefix}_manifest.csv"
    pd.DataFrame([{"model": model, "outputs": ";".join(outputs)}]).to_csv(manifest_path, index=False)
    outputs.append(str(manifest_path))
    return outputs


# =============================================================================
# FINAL FIGURE ASSEMBLY
# =============================================================================
def load_global_ranked(out_base: Path, model: str) -> Optional[pd.DataFrame]:
    path = out_base / "cross_cohort_aggregated" / model / f"Supplementary_Figure6_cross_cohort_{model}_global_feature_SHAP_union_ranked_all_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "feature_name" not in df.columns or "mean_abs_SHAP_across_cohorts" not in df.columns:
        return None
    out = df.copy()
    out["label"] = out["feature_name"].astype(str)
    out["value"] = pd.to_numeric(out["mean_abs_SHAP_across_cohorts"], errors="coerce")
    out["support"] = pd.to_numeric(out.get("n_cohorts"), errors="coerce")
    out["source_csv"] = str(path)
    return out.dropna(subset=["value"]).sort_values("value", ascending=False)


def load_region_ranked(out_base: Path, model: str) -> Optional[pd.DataFrame]:
    candidates = [
        out_base / "cross_cohort_aggregated" / model / f"Supplementary_Figure6_cross_cohort_{model}_edge_derived_region_SHAP_ranked.csv",
        out_base / "cross_cohort_aggregated" / f"Supplementary_Figure6_cross_cohort_{model}_edge_derived_region_SHAP_ranked.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path)
    if df.empty or "region" not in df.columns or "mean_incident_abs_SHAP_across_cohorts" not in df.columns:
        return None
    out = df.copy()
    out["label"] = out["region"].astype(str)
    out["value"] = pd.to_numeric(out["mean_incident_abs_SHAP_across_cohorts"], errors="coerce")
    out["support"] = pd.to_numeric(out.get("n_cohorts"), errors="coerce")
    out["source_csv"] = str(path)
    return out.dropna(subset=["value"]).sort_values("value", ascending=False)


def load_node_ranked(out_base: Path, model: str) -> Optional[pd.DataFrame]:
    candidates = [
        out_base / "cross_cohort_aggregated" / model / f"Supplementary_Figure6_cross_cohort_{model}_node_feature_SHAP_summary_collapsed.csv",
        out_base / "cross_cohort_aggregated" / f"Supplementary_Figure6_cross_cohort_{model}_node_feature_SHAP_summary_collapsed.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path)
    if df.empty or "node_feature_label" not in df.columns:
        return None
    value_col = "mean_abs_SHAP_across_cohorts"
    if value_col not in df.columns:
        return None
    out = df.copy()
    out["label"] = out["node_feature_label"].astype(str)
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out["support"] = pd.to_numeric(out.get("n_cohorts"), errors="coerce")
    out["source_csv"] = str(path)
    return out.dropna(subset=["value"]).sort_values("value", ascending=False)


def load_edge_ranked(out_base: Path, model: str) -> Optional[pd.DataFrame]:
    candidates = [
        out_base / "cross_cohort_aggregated" / model / f"Supplementary_Figure6_cross_cohort_{model}_edge_SHAP_ranked.csv",
        out_base / "cross_cohort_aggregated" / f"Supplementary_Figure6_cross_cohort_{model}_edge_SHAP_ranked.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path)
    if df.empty or "Edge" not in df.columns or "mean_abs_SHAP_across_cohorts" not in df.columns:
        return None
    out = df.copy()
    out["label"] = out["Edge"].astype(str)
    out["value"] = pd.to_numeric(out["mean_abs_SHAP_across_cohorts"], errors="coerce")
    out["support"] = pd.to_numeric(out.get("n_cohorts"), errors="coerce")
    out["source_csv"] = str(path)
    return out.dropna(subset=["value"]).sort_values("value", ascending=False)


def plot_panel(ax, df: Optional[pd.DataFrame], title: str, top_n: int, xlabel: str, label_width: int = 32):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No aggregate CSV found", ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        return

    top = df.head(top_n).copy().iloc[::-1]
    labels = [wrap_label(x, width=label_width) for x in top["label"]]
    bars = ax.barh(labels, top["value"])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    xmax = float(np.nanmax(top["value"])) if len(top) else 0.0
    if "support" in top.columns and top["support"].notna().any():
        for bar, support in zip(bars, top["support"]):
            if pd.isna(support):
                continue
            ax.text(
                bar.get_width() + max(xmax * 0.015, 1e-6),
                bar.get_y() + bar.get_height() / 2,
                f"n={int(support)}",
                va="center",
                fontsize=7,
            )


def save_main_figure(
    out_base: Path,
    model: str,
    final_dir: Path,
    dpi: int,
    top_global: int,
    top_edge: int,
    top_region: int,
    top_node: int,
) -> str:
    g = load_global_ranked(out_base, model)
    e = load_edge_ranked(out_base, model)
    r = load_region_ranked(out_base, model)
    n = load_node_ranked(out_base, model)

    fig, axes = plt.subplots(1, 4, figsize=(25, 7.5))
    plot_panel(axes[0], g, "A. Top global contributors", top_global, "Mean |SHAP| across cohorts", 34)
    plot_panel(axes[1], e, "B. Top edges", top_edge, "Mean |SHAP| across cohorts", 34)
    plot_panel(axes[2], r, "C. Top edge-derived regions", top_region, "Mean incident edge |SHAP|", 28)
    plot_panel(axes[3], n, "D. Top node-feature contributors", top_node, "Mean |SHAP| across cohorts", 38)
    fig.suptitle(
        f"Figure 6. Cross-cohort SHAP contributors to predicted brain age ({model_label(model)})",
        fontsize=16,
        y=1.02,
    )
    fig.tight_layout()
    outpath = final_dir / f"Figure6_combined_{model}_model_WITH_EDGES.png"
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(outpath)


def save_supplement_figure(
    out_base: Path,
    models: List[str],
    final_dir: Path,
    dpi: int,
    top_global: int,
    top_edge: int,
    top_region: int,
    top_node: int,
) -> str:
    fig_h = max(5.5 * len(models), 6)
    fig, axes = plt.subplots(len(models), 4, figsize=(25, fig_h), squeeze=False)

    for i, model in enumerate(models):
        g = load_global_ranked(out_base, model)
        e = load_edge_ranked(out_base, model)
        r = load_region_ranked(out_base, model)
        n = load_node_ranked(out_base, model)
        plot_panel(axes[i, 0], g, f"{model_label(model)}\nGlobal features", top_global, "Mean |SHAP|", 26)
        plot_panel(axes[i, 1], e, f"{model_label(model)}\nTop edges", top_edge, "Mean |SHAP|", 26)
        plot_panel(axes[i, 2], r, f"{model_label(model)}\nEdge-derived regions", top_region, "Mean incident edge |SHAP|", 22)
        plot_panel(axes[i, 3], n, f"{model_label(model)}\nNode-feature contributors", top_node, "Mean |SHAP|", 26)

    fig.suptitle("Supplementary Figure 6. Cross-cohort SHAP contributors by model", fontsize=16, y=1.005)
    fig.tight_layout()
    outpath = final_dir / "Supplementary_Figure6_model_comparison_global_edges_regions_nodes.png"
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(outpath)


def save_per_model_panels(
    out_base: Path,
    models: List[str],
    final_dir: Path,
    dpi: int,
    top_global: int,
    top_edge: int,
    top_region: int,
    top_node: int,
) -> List[str]:
    panel_dir = ensure_dir(final_dir / "per_model_panels")
    outputs = []
    for model in models:
        outputs.append(
            save_main_figure(
                out_base=out_base,
                model=model,
                final_dir=panel_dir,
                dpi=dpi,
                top_global=top_global,
                top_edge=top_edge,
                top_region=top_region,
                top_node=top_node,
            )
        )
    return outputs


def save_source_manifest(out_base: Path, models: List[str], final_dir: Path) -> str:
    rows = []
    for model in models:
        for kind, loader in [
            ("global", load_global_ranked),
            ("edge", load_edge_ranked),
            ("region", load_region_ranked),
            ("node", load_node_ranked),
        ]:
            df = loader(out_base, model)
            if df is None or df.empty:
                rows.append({"model": model, "kind": kind, "status": "missing", "source_csv": "", "n_rows": 0})
            else:
                rows.append({
                    "model": model,
                    "kind": kind,
                    "status": "loaded",
                    "source_csv": df["source_csv"].iloc[0],
                    "n_rows": int(len(df)),
                })

    out = final_dir / "Figure6_final_assembly_source_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)



def copy_beeswarm_to_final_figures(
    out_base: Path,
    model: str,
    final_dir: Path,
) -> List[str]:
    """
    Create manuscript-friendly Supplementary Figure 6B/6C filenames
    in final_figures/ from the per-model beeswarm outputs.
    """
    outputs = []

    global_src = (
        out_base
        / "cross_cohort_aggregated"
        / model
        / f"Supplementary_Figure6_cross_cohort_{model}_global_feature_SHAP_beeswarm.png"
    )
    if global_src.exists():
        global_dst = final_dir / f"Supplementary_Figure6B_global_feature_SHAP_beeswarm_{model}_model.png"
        global_dst.write_bytes(global_src.read_bytes())
        outputs.append(str(global_dst))

    node_src = (
        out_base
        / "cross_cohort_aggregated"
        / model
        / f"Supplementary_Figure6_cross_cohort_{model}_node_feature_SHAP_beeswarm.png"
    )
    if node_src.exists():
        node_dst = final_dir / f"Supplementary_Figure6C_node_feature_SHAP_beeswarm_{model}_model.png"
        node_dst.write_bytes(node_src.read_bytes())
        outputs.append(str(node_dst))

    return outputs


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    out_base = Path(args.out_base).expanduser().resolve()
    cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]
    keywords = [x.strip() for x in args.keywords.split(",") if x.strip()]

    if args.models:
        models = [x.strip() for x in args.models.split(",") if x.strip()]
    else:
        models = discover_models(out_base, cohorts)

    if not models:
        raise SystemExit(f"No models found under {out_base}")

    print("\n" + "=" * 100)
    print("FIGURE 6 MASTER v3: COMPLETE CROSS-COHORT SHAP BUILD")
    print("=" * 100)
    print(f"OUT_BASE: {out_base}")
    print(f"Cohorts:  {', '.join(cohorts)}")
    print(f"Models:   {', '.join(models)}")
    print("=" * 100)

    aggregate_outputs = []

    for model in models:
        print(f"\n--- Processing model: {model} ---")
        aggregate_outputs.extend(
            aggregate_global(
                out_base=out_base,
                cohorts=cohorts,
                model=model,
                top_n=args.top_n_global,
                sort_by_support_first=bool(args.sort_by_support_first),
                keywords=keywords,
                dpi=args.dpi,
                make_beeswarm_flag=not args.skip_beeswarm,
            )
        )
        aggregate_outputs.extend(
            aggregate_node(
                out_base=out_base,
                cohorts=cohorts,
                model=model,
                top_n=args.top_n_node,
                dpi=args.dpi,
            )
        )
        aggregate_outputs.extend(
            aggregate_edge_and_regions(
                out_base=out_base,
                cohorts=cohorts,
                model=model,
                top_n_edge=args.top_n_edge,
                top_n_region=args.top_n_region,
                dpi=args.dpi,
            )
        )

    final_outputs = []
    if not args.skip_final_assembly:
        final_dir = ensure_dir(out_base / "cross_cohort_aggregated" / "final_figures")
        main_model = args.main_model if args.main_model in models else models[0]

        final_outputs.extend(
            save_per_model_panels(
                out_base=out_base,
                models=models,
                final_dir=final_dir,
                dpi=args.dpi,
                top_global=args.top_n_global,
                top_edge=args.top_n_edge,
                top_region=args.top_n_region,
                top_node=args.top_n_node,
            )
        )
        final_outputs.append(
            save_main_figure(
                out_base=out_base,
                model=main_model,
                final_dir=final_dir,
                dpi=args.dpi,
                top_global=args.top_n_global,
                top_edge=args.top_n_edge,
                top_region=args.top_n_region,
                top_node=args.top_n_node,
            )
        )
        final_outputs.append(
            save_supplement_figure(
                out_base=out_base,
                models=models,
                final_dir=final_dir,
                dpi=args.dpi,
                top_global=args.supp_top_n_global,
                top_edge=min(args.top_n_edge, 10),
                top_region=args.supp_top_n_region,
                top_node=args.supp_top_n_node,
            )
        )
        final_outputs.append(save_source_manifest(out_base, models, final_dir))

        # Manuscript-friendly beeswarm copies:
        # Supplementary Figure 6B = global beeswarm
        # Supplementary Figure 6C = node-feature beeswarm
        beeswarm_models = []
        for candidate in [main_model, "imaging_only"]:
            if candidate in models and candidate not in beeswarm_models:
                beeswarm_models.append(candidate)
        for beeswarm_model in beeswarm_models:
            final_outputs.extend(copy_beeswarm_to_final_figures(out_base, beeswarm_model, final_dir))

        final_manifest = final_dir / "Figure6_final_outputs_manifest.csv"
        pd.DataFrame([{
            "out_base": str(out_base),
            "main_model": main_model,
            "models": ",".join(models),
            "outputs": ";".join(final_outputs),
        }]).to_csv(final_manifest, index=False)
        final_outputs.append(str(final_manifest))

    master_manifest = ensure_dir(out_base / "cross_cohort_aggregated") / "Figure6_MASTER_v3_workflow_manifest.csv"
    pd.DataFrame([{
        "out_base": str(out_base),
        "cohorts": ",".join(cohorts),
        "models": ",".join(models),
        "aggregate_outputs": ";".join(aggregate_outputs),
        "final_outputs": ";".join(final_outputs),
    }]).to_csv(master_manifest, index=False)

    print("\nFinished Figure 6 MASTER v3.")
    print(f"Master manifest: {master_manifest}")
    if final_outputs:
        print("\nKey final outputs:")
        for p in final_outputs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
