#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 7 raw + scaled SHAP sensitivity for all feature-set models
=================================================================

Purpose
-------
Post-hoc SHAP analysis only. Does NOT recompute SHAP.

For each model/feature set, this script computes:

1. Raw cross-cohort SHAP rankings
2. Within-cohort scaled SHAP rankings
3. Main 4-panel scaled figure
4. Raw figure
5. Raw-vs-scaled sensitivity figure
6. Cross-model biomarker/non-imaging feature summary table

Scaling
-------
For each cohort and SHAP class separately:

    scaled_abs_SHAP = mean_abs_SHAP / sum(mean_abs_SHAP over features in that cohort)

Models
------
Default:
  imaging_only
  imaging_demographics
  imaging_biomarkers
  full
  full_no_cardiovascular

Outputs
-------
Figure7_SHAP/paper_ready_all_models/<model>/
Figure7_SHAP/paper_ready_all_models/cross_model/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUT = Path(
    "/mnt/newStor/paros/paros_WORK/ines/results/"
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/"
    "Figure7_SHAP"
)

DEFAULT_MODELS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

N_COHORTS_REQUIRED = 4
EDGE_MIN_TOTAL_N = 100

TOP_N_GLOBAL = 20
TOP_N_NODE = 20
TOP_N_REGION = 20
TOP_N_EDGE = 15

FORMATS = ["png", "pdf"]
DPI = 450


BIOMARKER_PATTERNS = [
    "apoe", "apoe4", "amyloid", "abeta", "aβ", "tau", "ptau", "p-tau",
    "nfl", "neurofilament", "gfap", "csf", "plasma", "pet",
    "centiloid", "braak", "cdr", "mmse", "moca", "memory",
    "executive", "language", "attention", "cognition", "cognitive",
    "bmi", "diabetes", "hba1c", "glucose", "insulin",
    "hypertension", "vascular", "cardio", "cholesterol",
    "ldl", "hdl", "triglyceride", "blood pressure", "bp",
    "sex", "gender", "education", "age",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-base", default=str(DEFAULT_OUT))
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--top-n-node", type=int, default=TOP_N_NODE)
    p.add_argument("--top-n-region", type=int, default=TOP_N_REGION)
    p.add_argument("--top-n-edge", type=int, default=TOP_N_EDGE)
    p.add_argument("--edge-min-total-n", type=int, default=EDGE_MIN_TOTAL_N)
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--formats", default="png,pdf")
    return p.parse_args()


def find_existing(paths, required=True, label="file"):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(
            f"Could not find required {label}. Tried:\n" +
            "\n".join(str(p) for p in paths)
        )
    return None


def wrap_label(x, width=32):
    x = str(x)
    x = x.replace(" | ", "\n")
    x = x.replace(" -- ", " — ")
    return "\n".join(textwrap.wrap(x, width=width, break_long_words=False))


def clean_cohort(x):
    x = str(x)
    if x.upper() in {"ADDECODE", "AD-DECODE"}:
        return "AD_DECODE"
    return x


def infer_feature_label(df, kind):
    df = df.copy()

    if "cohort" in df.columns:
        df["cohort"] = df["cohort"].map(clean_cohort)

    if "feature_label" in df.columns:
        df["feature_label"] = df["feature_label"].astype(str)
        return df

    if kind == "global":
        if "feature_name" in df.columns:
            df["feature_label"] = df["feature_name"].astype(str)
            return df

    if kind == "node":
        if "node_feature_label" in df.columns:
            df["feature_label"] = df["node_feature_label"].astype(str)
            return df
        if {"node_label", "feature_name"}.issubset(df.columns):
            df["feature_label"] = df["node_label"].astype(str) + " | " + df["feature_name"].astype(str)
            return df
        if {"Structure", "feature_name"}.issubset(df.columns):
            df["feature_label"] = df["Structure"].astype(str) + " | " + df["feature_name"].astype(str)
            return df

    if kind == "edge":
        if "Edge" in df.columns:
            df["feature_label"] = df["Edge"].astype(str)
            return df
        if {"Structure_i", "Structure_j"}.issubset(df.columns):
            df["feature_label"] = df["Structure_i"].astype(str) + " -- " + df["Structure_j"].astype(str)
            return df
        if {"structure_i", "structure_j"}.issubset(df.columns):
            df["feature_label"] = df["structure_i"].astype(str) + " -- " + df["structure_j"].astype(str)
            return df
        if {"Node_i", "Node_j"}.issubset(df.columns):
            df["feature_label"] = df["Node_i"].astype(str) + " -- " + df["Node_j"].astype(str)
            return df

    raise ValueError(f"Could not infer feature_label for kind={kind}. Columns: {list(df.columns)}")


def normalize_by_cohort(df, value_col="mean_abs_SHAP"):
    df = df.copy()
    for c in ["cohort", "feature_label", value_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}. Columns: {list(df.columns)}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["cohort", "feature_label", value_col])

    totals = (
        df.groupby("cohort", as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "cohort_total_abs_SHAP"})
    )
    df = df.merge(totals, on="cohort", how="left")
    df["scaled_abs_SHAP"] = df[value_col] / df["cohort_total_abs_SHAP"].replace(0, np.nan)
    return df


def aggregate_raw_scaled(by, kind, min_total_n=None):
    by = infer_feature_label(by, kind)
    by = normalize_by_cohort(by, "mean_abs_SHAP")

    if "mean_SHAP" in by.columns:
        by["mean_SHAP"] = pd.to_numeric(by["mean_SHAP"], errors="coerce")
    else:
        by["mean_SHAP"] = np.nan

    if "n_subjects" in by.columns:
        by["n_subjects"] = pd.to_numeric(by["n_subjects"], errors="coerce")
    else:
        by["n_subjects"] = np.nan

    agg = (
        by.groupby("feature_label", as_index=False)
        .agg(
            mean_abs_SHAP_across_cohorts=("mean_abs_SHAP", "mean"),
            median_abs_SHAP_across_cohorts=("mean_abs_SHAP", "median"),
            sd_abs_SHAP_across_cohorts=("mean_abs_SHAP", "std"),
            mean_scaled_abs_SHAP_across_cohorts=("scaled_abs_SHAP", "mean"),
            median_scaled_abs_SHAP_across_cohorts=("scaled_abs_SHAP", "median"),
            sd_scaled_abs_SHAP_across_cohorts=("scaled_abs_SHAP", "std"),
            mean_signed_SHAP_across_cohorts=("mean_SHAP", "mean"),
            median_signed_SHAP_across_cohorts=("mean_SHAP", "median"),
            n_cohorts=("cohort", "nunique"),
            total_n_subjects=("n_subjects", "sum"),
            cohorts=("cohort", lambda x: ",".join(sorted(set(map(str, x))))),
        )
    )

    raw_wide = (
        by.pivot_table(index="feature_label", columns="cohort", values="mean_abs_SHAP", aggfunc="mean")
        .rename(columns={c: f"{c}_raw_mean_abs_SHAP" for c in by["cohort"].dropna().unique()})
        .reset_index()
    )
    scaled_wide = (
        by.pivot_table(index="feature_label", columns="cohort", values="scaled_abs_SHAP", aggfunc="mean")
        .rename(columns={c: f"{c}_scaled_abs_SHAP" for c in by["cohort"].dropna().unique()})
        .reset_index()
    )

    agg = agg.merge(raw_wide, on="feature_label", how="left")
    agg = agg.merge(scaled_wide, on="feature_label", how="left")

    agg = agg[agg["n_cohorts"] == N_COHORTS_REQUIRED].copy()

    if min_total_n is not None:
        agg = agg[agg["total_n_subjects"] >= min_total_n].copy()

    agg["rank_raw"] = agg["mean_abs_SHAP_across_cohorts"].rank(ascending=False, method="min")
    agg["rank_scaled"] = agg["mean_scaled_abs_SHAP_across_cohorts"].rank(ascending=False, method="min")
    return agg.reset_index(drop=True)


def split_edge_label(edge):
    edge = str(edge)
    if " -- " in edge:
        a, b = edge.split(" -- ", 1)
    elif " — " in edge:
        a, b = edge.split(" — ", 1)
    elif "--" in edge:
        a, b = edge.split("--", 1)
    else:
        return None
    return a.strip(), b.strip()


def make_edge_region_from_edge_bycohort(edge_by):
    edge_by = infer_feature_label(edge_by, "edge")

    rows = []
    for _, r in edge_by.iterrows():
        parts = split_edge_label(r["feature_label"])
        if parts is None:
            continue

        val = pd.to_numeric(r.get("mean_abs_SHAP", np.nan), errors="coerce")
        signed = pd.to_numeric(r.get("mean_SHAP", np.nan), errors="coerce")
        n_sub = pd.to_numeric(r.get("n_subjects", np.nan), errors="coerce")

        if not np.isfinite(val):
            continue

        for node in parts:
            rows.append({
                "cohort": clean_cohort(r["cohort"]),
                "feature_label": node,
                "mean_abs_SHAP": val,
                "mean_SHAP": signed,
                "n_subjects": n_sub,
            })

    reg = pd.DataFrame(rows)
    if reg.empty:
        raise ValueError("Could not derive edge-region rows.")

    reg = (
        reg.groupby(["cohort", "feature_label"], as_index=False)
        .agg(
            mean_abs_SHAP=("mean_abs_SHAP", "sum"),
            mean_SHAP=("mean_SHAP", "sum"),
            n_subjects=("n_subjects", "sum"),
        )
    )
    return reg


def classify_feature(feature_label):
    s = str(feature_label).lower()
    hits = [p for p in BIOMARKER_PATTERNS if p in s]
    if not hits:
        return "imaging_or_graph"
    if any(p in s for p in ["abeta", "amyloid", "aβ", "tau", "ptau", "p-tau", "nfl", "gfap", "csf", "plasma", "centiloid"]):
        return "ad_molecular_biomarker"
    if any(p in s for p in ["apoe", "apoe4"]):
        return "genetic_risk"
    if any(p in s for p in ["memory", "executive", "language", "attention", "cognition", "cognitive", "cdr", "mmse", "moca"]):
        return "cognitive_clinical"
    if any(p in s for p in ["bmi", "diabetes", "hba1c", "glucose", "insulin", "hypertension", "vascular", "cardio", "cholesterol", "ldl", "hdl", "triglyceride", "bp"]):
        return "vascular_metabolic"
    if any(p in s for p in ["age", "sex", "gender", "education"]):
        return "demographic"
    return "other_nonimaging"


def plot_table_to_bar_input(df, sort_col, top_n, width):
    out = (
        df.sort_values(sort_col, ascending=False)
        .head(top_n)
        .copy()
    )
    out["plot_label"] = out["feature_label"].map(lambda x: wrap_label(x, width))
    return out


def barh_panel(ax, df, valcol, panel_letter, title, xlabel, note_mode=None):
    plot = df.iloc[::-1].copy()
    ax.barh(plot["plot_label"], plot[valcol])
    ax.set_title(title, fontsize=11, pad=7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=8)
    ax.text(-0.12, 1.04, panel_letter, transform=ax.transAxes, fontsize=15, fontweight="bold")

    xmax = float(plot[valcol].max()) if len(plot) else 1.0
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0

    for y, (_, r) in enumerate(plot.iterrows()):
        if note_mode == "edge":
            n = int(r["n_cohorts"]) if pd.notna(r.get("n_cohorts")) else 4
            N = int(r["total_n_subjects"]) if pd.notna(r.get("total_n_subjects")) else -1
            txt = f"n={n}, N={N}" if N >= 0 else f"n={n}"
        else:
            n = int(r["n_cohorts"]) if pd.notna(r.get("n_cohorts")) else 4
            txt = f"n={n}"
        ax.text(r[valcol] + xmax * 0.01, y, txt, va="center", fontsize=6)

    ax.set_xlim(0, xmax * 1.22)
    ax.grid(axis="x", alpha=0.15)


def make_figure(model, outdir, global_tbl, node_tbl, region_tbl, edge_tbl, mode, top_n_node, top_n_region, top_n_edge, formats, dpi):
    if mode == "RAW":
        valcol = "mean_abs_SHAP_across_cohorts"
        subtitle = "Raw mean absolute SHAP"
        xlabel = "Mean |SHAP| across cohorts"
        region_xlabel = "Mean incident edge |SHAP| across cohorts"
    elif mode == "SCALED":
        valcol = "mean_scaled_abs_SHAP_across_cohorts"
        subtitle = "Within-cohort normalized SHAP"
        xlabel = "Mean within-cohort scaled |SHAP|"
        region_xlabel = "Mean within-cohort scaled incident edge |SHAP|"
    else:
        raise ValueError(mode)

    g = plot_table_to_bar_input(global_tbl, valcol, min(20, len(global_tbl)), 26)
    n = plot_table_to_bar_input(node_tbl, valcol, top_n_node, 30)
    r = plot_table_to_bar_input(region_tbl, valcol, top_n_region, 28)
    e = plot_table_to_bar_input(edge_tbl, valcol, top_n_edge, 32)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.30)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    barh_panel(axA, g, valcol, "A", "Global contributors present in all 4 cohorts", xlabel)
    barh_panel(axB, n, valcol, "B", "Node-feature contributors present in all 4 cohorts", xlabel)
    barh_panel(axC, r, valcol, "C", "Edge-derived regional contributors present in all 4 cohorts", region_xlabel)
    barh_panel(axD, e, valcol, "D", "Exact edge contributors present in all 4 cohorts", xlabel, note_mode="edge")

    fig.suptitle(
        f"Figure 7 SHAP contributors: {model}\n{subtitle}",
        fontsize=16,
        y=0.995,
    )

    fig.text(
        0.02,
        0.01,
        "Exact-edge panel restricted to n_cohorts = 4 and total N ≥ 100. "
        "Scaled values are normalized within cohort and SHAP feature class before averaging across cohorts.",
        fontsize=9,
        ha="left",
    )

    stem = outdir / f"Figure7_{model}_barplots_common4_{mode}"
    for ext in formats:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print("Saved:", out)
    plt.close(fig)


def make_feature_table_for_biomarkers(model, global_tbl, top_n=50):
    df = global_tbl.copy()
    df["model"] = model
    df["feature_category"] = df["feature_label"].map(classify_feature)
    df = df.sort_values("mean_scaled_abs_SHAP_across_cohorts", ascending=False).copy()
    df["rank_scaled_overall"] = np.arange(1, len(df) + 1)
    return df.head(top_n)


def run_model(out_base, model, args, formats):
    common = out_base / "common4_consensus" / model
    model_out = out_base / "paper_ready_all_models" / model
    model_out.mkdir(parents=True, exist_ok=True)

    global_by_candidates = [
        common / f"common4_{model}_global_SHAP_by_cohort.csv",
        common / f"Figure7_{model}_global_SHAP_common4_by_cohort.csv",
    ]
    node_by_candidates = [
        common / f"common4_{model}_node_SHAP_by_cohort.csv",
        common / f"Figure7_{model}_node_SHAP_common4_by_cohort.csv",
    ]
    edge_by_candidates = [
        common / f"common4_{model}_edge_SHAP_by_cohort.csv",
        common / f"Figure7_{model}_edge_SHAP_common4_by_cohort.csv",
    ]

    global_by_path = find_existing(global_by_candidates, True, f"{model} global by-cohort")
    node_by_path = find_existing(node_by_candidates, True, f"{model} node by-cohort")
    edge_by_path = find_existing(edge_by_candidates, True, f"{model} edge by-cohort")

    print("\n" + "=" * 100)
    print("MODEL:", model)
    print("global:", global_by_path)
    print("node:  ", node_by_path)
    print("edge:  ", edge_by_path)
    print("=" * 100)

    global_by = pd.read_csv(global_by_path)
    node_by = pd.read_csv(node_by_path)
    edge_by = pd.read_csv(edge_by_path)

    global_tbl = aggregate_raw_scaled(global_by, "global")
    node_tbl = aggregate_raw_scaled(node_by, "node")
    edge_tbl = aggregate_raw_scaled(edge_by, "edge", min_total_n=args.edge_min_total_n)

    region_by = make_edge_region_from_edge_bycohort(edge_by)
    region_tbl = aggregate_raw_scaled(region_by, "global")

    global_tbl = global_tbl.sort_values("mean_scaled_abs_SHAP_across_cohorts", ascending=False)
    node_tbl = node_tbl.sort_values("mean_scaled_abs_SHAP_across_cohorts", ascending=False)
    region_tbl = region_tbl.sort_values("mean_scaled_abs_SHAP_across_cohorts", ascending=False)
    edge_tbl = edge_tbl.sort_values("mean_scaled_abs_SHAP_across_cohorts", ascending=False)

    global_out = model_out / f"Figure7_{model}_global_raw_scaled_common4.csv"
    node_out = model_out / f"Figure7_{model}_node_raw_scaled_common4.csv"
    region_out = model_out / f"Figure7_{model}_edge_region_raw_scaled_common4.csv"
    edge_out = model_out / f"Figure7_{model}_exact_edges_raw_scaled_common4_minN{args.edge_min_total_n}.csv"

    global_tbl.to_csv(global_out, index=False)
    node_tbl.to_csv(node_out, index=False)
    region_tbl.to_csv(region_out, index=False)
    edge_tbl.to_csv(edge_out, index=False)

    print("Saved table:", global_out, "rows=", len(global_tbl))
    print("Saved table:", node_out, "rows=", len(node_tbl))
    print("Saved table:", region_out, "rows=", len(region_tbl))
    print("Saved table:", edge_out, "rows=", len(edge_tbl))

    make_figure(
        model, model_out, global_tbl, node_tbl, region_tbl, edge_tbl,
        mode="SCALED",
        top_n_node=args.top_n_node,
        top_n_region=args.top_n_region,
        top_n_edge=args.top_n_edge,
        formats=formats,
        dpi=args.dpi,
    )

    make_figure(
        model, model_out, global_tbl, node_tbl, region_tbl, edge_tbl,
        mode="RAW",
        top_n_node=args.top_n_node,
        top_n_region=args.top_n_region,
        top_n_edge=args.top_n_edge,
        formats=formats,
        dpi=args.dpi,
    )

    biomarker_top = make_feature_table_for_biomarkers(model, global_tbl, top_n=100)
    biomarker_out = model_out / f"Figure7_{model}_global_top100_feature_categories.csv"
    biomarker_top.to_csv(biomarker_out, index=False)
    print("Saved biomarker/global category table:", biomarker_out)

    return {
        "model": model,
        "global_tbl": global_tbl,
        "node_tbl": node_tbl,
        "region_tbl": region_tbl,
        "edge_tbl": edge_tbl,
        "biomarker_top": biomarker_top,
        "model_out": model_out,
    }


def make_cross_model_summary(results, out_base):
    cross = out_base / "paper_ready_all_models" / "cross_model"
    cross.mkdir(parents=True, exist_ok=True)

    rows = []
    biomarker_rows = []

    for res in results:
        model = res["model"]
        for kind, tbl in [
            ("global", res["global_tbl"]),
            ("node", res["node_tbl"]),
            ("edge_region", res["region_tbl"]),
            ("exact_edge", res["edge_tbl"]),
        ]:
            if tbl.empty:
                continue

            top = tbl.sort_values("mean_scaled_abs_SHAP_across_cohorts", ascending=False).head(25).copy()
            for i, r in top.iterrows():
                rows.append({
                    "model": model,
                    "kind": kind,
                    "rank_scaled": len(rows) + 1,
                    "feature_label": r["feature_label"],
                    "mean_scaled_abs_SHAP_across_cohorts": r["mean_scaled_abs_SHAP_across_cohorts"],
                    "mean_abs_SHAP_across_cohorts": r["mean_abs_SHAP_across_cohorts"],
                    "n_cohorts": r["n_cohorts"],
                    "total_n_subjects": r["total_n_subjects"],
                    "feature_category": classify_feature(r["feature_label"]),
                })

        g = res["global_tbl"].copy()
        g["model"] = model
        g["feature_category"] = g["feature_label"].map(classify_feature)
        g["rank_scaled"] = g["mean_scaled_abs_SHAP_across_cohorts"].rank(ascending=False, method="min")
        biomarker_rows.append(g)

    summary = pd.DataFrame(rows)
    summary_out = cross / "Figure7_cross_model_top25_by_kind_scaled.csv"
    summary.to_csv(summary_out, index=False)
    print("Saved:", summary_out)

    biomarker = pd.concat(biomarker_rows, ignore_index=True) if biomarker_rows else pd.DataFrame()
    biomarker_out = cross / "Figure7_cross_model_global_features_biomarker_screen.csv"
    biomarker.to_csv(biomarker_out, index=False)
    print("Saved:", biomarker_out)

    # Biomarker/non-imaging subset.
    if not biomarker.empty:
        nonimaging = biomarker[biomarker["feature_category"] != "imaging_or_graph"].copy()
        nonimaging = nonimaging.sort_values(["model", "rank_scaled"])
        nonimaging_out = cross / "Figure7_cross_model_nonimaging_biomarker_candidates.csv"
        nonimaging.to_csv(nonimaging_out, index=False)
        print("Saved:", nonimaging_out)

    # Compact plot: top global features per model by scaled SHAP.
    plot_rows = []
    for res in results:
        model = res["model"]
        top = res["global_tbl"].sort_values("mean_scaled_abs_SHAP_across_cohorts", ascending=False).head(10)
        for _, r in top.iterrows():
            plot_rows.append({
                "model": model,
                "feature_label": r["feature_label"],
                "value": r["mean_scaled_abs_SHAP_across_cohorts"],
                "category": classify_feature(r["feature_label"]),
            })
    plot_df = pd.DataFrame(plot_rows)
    plot_out = cross / "Figure7_cross_model_top_global_scaled.csv"
    plot_df.to_csv(plot_out, index=False)
    print("Saved:", plot_out)


def main():
    args = parse_args()
    out_base = Path(args.out_base)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    formats = [x.strip().lower().lstrip(".") for x in args.formats.split(",") if x.strip()]

    print("=" * 100)
    print("Figure7 SHAP raw/scaled sensitivity for all models")
    print("OUT_BASE:", out_base)
    print("MODELS:", models)
    print("=" * 100)

    results = []
    for model in models:
        try:
            results.append(run_model(out_base, model, args, formats))
        except Exception as e:
            warnings.warn(f"FAILED model={model}: {e}")

    if results:
        make_cross_model_summary(results, out_base)

    print("\nDone.")


if __name__ == "__main__":
    main()
