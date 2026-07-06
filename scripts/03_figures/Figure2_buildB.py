#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:55:25 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Figure 2 and Supplementary Figure 2.

Figure 2:
    OOF_RAW model performance across cohorts and feature sets:
    MAE, RMSE, R², Pearson r with bootstrap 95% CIs.

Supplementary Figure 2:
    OOF_BIAS_CORRECTED diagnostic performance across cohorts and feature sets:
    MAE, RMSE, R², Pearson r with bootstrap 95% CIs.

Input:
    combined_bootstrap_metric_summary.csv

Output directory:
    BASE_DIR/Figure2/

Output:
    Figure2_OOF_RAW_performance_bootstrap_CI_2x2.png/pdf/csv
    SupplementaryFigure2_OOF_BIAS_CORRECTED_performance_bootstrap_CI_2x2.png/pdf/csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================
BASE_DIR = (
    "/mnt/newStor/paros/paros_WORK/ines/results/"
    "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
)

FIGURE_DIR = os.path.join(BASE_DIR, "Figure2")

BOOTSTRAP_PATH = os.path.join(BASE_DIR, "combined_bootstrap_metric_summary.csv")


# =========================
# SETTINGS
# =========================
COHORT_ORDER = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

FEATURE_ORDER = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full_no_cardiovascular",
    "full",
]

METRICS = ["MAE", "RMSE", "R2", "r"]

METRIC_LABELS = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "R2": "R²",
    "r": "Pearson r",
}

PANEL_LABELS = {
    "MAE": "A",
    "RMSE": "B",
    "R2": "C",
    "r": "D",
}

FIGURE_CONFIGS = [
    {
        "evaluation": "OOF_RAW",
        "title": "Out-of-fold raw brain-age prediction performance across cohorts and feature sets",
        "basename": "Figure2_OOF_RAW_performance_bootstrap_CI_2x2",
    },
    {
        "evaluation": "OOF_BIAS_CORRECTED",
        "title": "Fold-wise bias-corrected brain-age prediction performance across cohorts and feature sets",
        "basename": "SupplementaryFigure2A_OOF_BIAS_CORRECTED_performance_bootstrap_CI_2x2",
    },
    {
        "evaluation": "OOF_GLOBAL_BAG_RESIDUALIZED",
        "title": "OOF-global BAG-residualized brain-age prediction performance across cohorts and feature sets",
        "basename": "SupplementaryFigure2B_OOF_GLOBAL_BAG_RESIDUALIZED_performance_bootstrap_CI_2x2",
    },
]


# =========================
# HELPERS
# =========================
def load_bootstrap_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing bootstrap summary file: {path}")

    df = pd.read_csv(path)

    required = {
        "cohort",
        "feature_set",
        "evaluation",
        "metric",
        "point_estimate",
        "ci_low",
        "ci_high",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Bootstrap file is missing required columns: {missing}")

    for col in ["point_estimate", "ci_low", "ci_high"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def prepare_figure_data(df, evaluation):
    out = df[
        (df["evaluation"].astype(str) == evaluation)
        & (df["metric"].astype(str).isin(METRICS))
    ].copy()

    if out.empty:
        available = sorted(df["evaluation"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No rows found for evaluation='{evaluation}'. "
            f"Available evaluations: {available}"
        )

    out["cohort"] = pd.Categorical(
        out["cohort"],
        categories=COHORT_ORDER,
        ordered=True,
    )

    out["feature_set"] = pd.Categorical(
        out["feature_set"],
        categories=FEATURE_ORDER,
        ordered=True,
    )

    out["metric"] = pd.Categorical(
        out["metric"],
        categories=METRICS,
        ordered=True,
    )

    out = out.sort_values(["metric", "cohort", "feature_set"]).copy()

    return out


def get_axis_limits(metric, data):
    vals = pd.to_numeric(data["point_estimate"], errors="coerce")
    lows = pd.to_numeric(data["ci_low"], errors="coerce")
    highs = pd.to_numeric(data["ci_high"], errors="coerce")

    finite_vals = (
        pd.concat([vals, lows, highs], ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if finite_vals.empty:
        return None

    ymin = finite_vals.min()
    ymax = finite_vals.max()
    pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.1

    if metric in ["MAE", "RMSE"]:
        return max(0, ymin - pad), ymax + pad

    if metric == "R2":
        lower = min(0, ymin - pad)
        upper = ymax + pad
        return lower, upper

    if metric == "r":
        lower = min(0, ymin - pad)
        upper = min(1.0, ymax + pad)
        return lower, upper

    return ymin - pad, ymax + pad


def plot_metric(ax, data, metric, evaluation):
    sub = data[data["metric"].astype(str) == metric].copy()

    x = np.arange(len(COHORT_ORDER))
    width = 0.15
    offsets = np.linspace(-2, 2, len(FEATURE_ORDER)) * width

    for i, fs in enumerate(FEATURE_ORDER):
        fs_df = sub[sub["feature_set"].astype(str) == fs].copy()

        y_vals = []
        yerr_low = []
        yerr_high = []

        for cohort in COHORT_ORDER:
            row = fs_df[fs_df["cohort"].astype(str) == cohort]

            if row.empty:
                y_vals.append(np.nan)
                yerr_low.append(0.0)
                yerr_high.append(0.0)
                continue

            point = float(row["point_estimate"].iloc[0])
            ci_low = float(row["ci_low"].iloc[0])
            ci_high = float(row["ci_high"].iloc[0])

            y_vals.append(point)

            if np.isfinite(point) and np.isfinite(ci_low) and np.isfinite(ci_high):
                yerr_low.append(max(0.0, point - ci_low))
                yerr_high.append(max(0.0, ci_high - point))
            else:
                yerr_low.append(0.0)
                yerr_high.append(0.0)

        y_vals = np.array(y_vals, dtype=float)
        yerr = np.array([yerr_low, yerr_high], dtype=float)

        ax.bar(
            x + offsets[i],
            y_vals,
            width=width,
            label=fs,
            yerr=yerr,
            capsize=4,
            edgecolor="black",
            linewidth=0.3,
            error_kw={
                "elinewidth": 1.1,
                "capthick": 1.1,
            },
        )

    ax.set_xticks(x)
    ax.set_xticklabels(COHORT_ORDER)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(
        f"{PANEL_LABELS[metric]}. {METRIC_LABELS[metric]} "
        f"({evaluation}, bootstrap 95% CI)"
    )
    ax.grid(axis="y", alpha=0.3)

    if metric == "R2":
        ax.axhline(0, linestyle="--", linewidth=1)

    ylim = get_axis_limits(metric, sub)
    if ylim is not None:
        ax.set_ylim(*ylim)


def make_2x2_figure(df, evaluation, title, basename):
    fig_df = prepare_figure_data(df, evaluation)

    out_png = os.path.join(FIGURE_DIR, f"{basename}.png")
    out_pdf = os.path.join(FIGURE_DIR, f"{basename}.pdf")
    out_csv = os.path.join(FIGURE_DIR, f"{basename}_source_data.csv")

    fig_df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)

    plot_metric(axes[0, 0], fig_df, "MAE", evaluation)
    plot_metric(axes[0, 1], fig_df, "RMSE", evaluation)
    plot_metric(axes[1, 0], fig_df, "R2", evaluation)
    plot_metric(axes[1, 1], fig_df, "r", evaluation)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(FEATURE_ORDER),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        title,
        fontsize=18,
        y=1.06,
    )

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(out_png)
    print(out_pdf)
    print(out_csv)

    return out_png, out_pdf, out_csv


# =========================
# MAIN
# =========================
def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)

    df = load_bootstrap_table(BOOTSTRAP_PATH)

    print("Loaded bootstrap table:")
    print(BOOTSTRAP_PATH)
    print("Shape:", df.shape)
    print("\nAvailable evaluations:")
    print(sorted(df["evaluation"].dropna().astype(str).unique().tolist()))

    print("\nSaving Figure 2 outputs to:")
    print(FIGURE_DIR)

    for cfg in FIGURE_CONFIGS:
        make_2x2_figure(
            df=df,
            evaluation=cfg["evaluation"],
            title=cfg["title"],
            basename=cfg["basename"],
        )


if __name__ == "__main__":
    main()