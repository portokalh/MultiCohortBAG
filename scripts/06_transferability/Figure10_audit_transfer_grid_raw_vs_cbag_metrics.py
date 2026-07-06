#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit and fix cross-cohort transfer-grid metrics.

Why this exists
---------------
The first transfer-grid script computed MAE/RMSE/R2/r using
pred_bias_corrected_global. That column is useful for cBAG/residual analyses,
but it is age-informed because BAG residualization uses chronological age.

For external predictive transfer, report RAW predicted-age metrics:
    pred_raw vs age_true

For biological residual stability, report:
    cBAG_global = pred_bias_corrected_global - age_true
    mean_abs_cBAG
    cBAG_age_slope
    cBAG_age_r

Default usage
-------------
python audit_transfer_grid_raw_vs_cbag_metrics.py

Default input:
    /mnt/newStor/paros/paros_WORK/ines/results/
    BrainAgeTransferValidation_TrainTestGrid_OOFGlobal/imaging_only/
        transfer_grid_predictions_all.csv

Outputs:
    transfer_grid_metrics_RAW_vs_cBAG.csv
    heatmap_MAE_raw.png
    heatmap_RMSE_raw.png
    heatmap_R2_raw.png
    heatmap_r_raw.png
    heatmap_mean_abs_cBAG.png
    heatmap_cBAG_age_slope_fixed.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_GRID_DIR = (
    Path(DEFAULT_WORK)
    / "ines/results"
    / "BrainAgeTransferValidation_TrainTestGrid_OOFGlobal"
    / "imaging_only"
)

DEFAULT_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", default=str(DEFAULT_GRID_DIR))
    p.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def safe_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def safe_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0:
        return np.nan, np.nan, np.nan
    lr = linregress(x, y)
    return float(lr.slope), float(lr.intercept), float(lr.pvalue)


def metric_row(g):
    y = pd.to_numeric(g["age_true"], errors="coerce").to_numpy(float)
    pred_raw = pd.to_numeric(g["pred_raw"], errors="coerce").to_numpy(float)

    if "pred_bias_corrected_global" in g.columns:
        pred_bc = pd.to_numeric(g["pred_bias_corrected_global"], errors="coerce").to_numpy(float)
    else:
        pred_bc = np.full_like(y, np.nan)

    if "cBAG_global" in g.columns:
        cbag = pd.to_numeric(g["cBAG_global"], errors="coerce").to_numpy(float)
    else:
        cbag = pred_bc - y

    raw_mask = np.isfinite(y) & np.isfinite(pred_raw)
    cbag_mask = np.isfinite(y) & np.isfinite(cbag)

    out = {"n": int(raw_mask.sum())}

    if raw_mask.sum() >= 3:
        yy = y[raw_mask]
        pp = pred_raw[raw_mask]
        out.update({
            "MAE_raw": float(mean_absolute_error(yy, pp)),
            "RMSE_raw": float(np.sqrt(mean_squared_error(yy, pp))),
            "R2_raw": float(r2_score(yy, pp)),
            "r_raw": safe_r(yy, pp),
            "age_mean": float(np.mean(yy)),
            "age_sd": float(np.std(yy, ddof=1)),
            "pred_raw_mean": float(np.mean(pp)),
            "pred_raw_sd": float(np.std(pp, ddof=1)),
        })
    else:
        out.update({
            "MAE_raw": np.nan,
            "RMSE_raw": np.nan,
            "R2_raw": np.nan,
            "r_raw": np.nan,
            "age_mean": np.nan,
            "age_sd": np.nan,
            "pred_raw_mean": np.nan,
            "pred_raw_sd": np.nan,
        })

    if np.isfinite(pred_bc).sum() >= 3:
        bc_mask = np.isfinite(y) & np.isfinite(pred_bc)
        yy = y[bc_mask]
        pb = pred_bc[bc_mask]
        out.update({
            "MAE_age_informed_corrected_DO_NOT_USE_AS_TRANSFER_PERFORMANCE": float(mean_absolute_error(yy, pb)),
            "RMSE_age_informed_corrected_DO_NOT_USE_AS_TRANSFER_PERFORMANCE": float(np.sqrt(mean_squared_error(yy, pb))),
            "R2_age_informed_corrected_DO_NOT_USE_AS_TRANSFER_PERFORMANCE": float(r2_score(yy, pb)),
            "r_age_informed_corrected_DO_NOT_USE_AS_TRANSFER_PERFORMANCE": safe_r(yy, pb),
        })

    if cbag_mask.sum() >= 3:
        yy = y[cbag_mask]
        cb = cbag[cbag_mask]
        slope, intercept, pval = safe_slope(yy, cb)
        out.update({
            "mean_cBAG": float(np.mean(cb)),
            "mean_abs_cBAG": float(np.mean(np.abs(cb))),
            "sd_cBAG": float(np.std(cb, ddof=1)),
            "cBAG_age_r": safe_r(yy, cb),
            "cBAG_age_slope": slope,
            "cBAG_age_intercept": intercept,
            "cBAG_age_slope_p": pval,
        })
    else:
        out.update({
            "mean_cBAG": np.nan,
            "mean_abs_cBAG": np.nan,
            "sd_cBAG": np.nan,
            "cBAG_age_r": np.nan,
            "cBAG_age_slope": np.nan,
            "cBAG_age_intercept": np.nan,
            "cBAG_age_slope_p": np.nan,
        })

    return out


def save_heatmap(df, metric, cohorts, out_png, title, fmt="{:.2f}"):
    grid = pd.DataFrame(index=cohorts, columns=cohorts, dtype=float)
    labels = pd.DataFrame(index=cohorts, columns=cohorts, dtype=object)

    for _, row in df.iterrows():
        tr, te = row["train_cohort"], row["test_cohort"]
        val = row.get(metric, np.nan)
        grid.loc[tr, te] = val
        suffix = "OOF" if tr == te else "EXT"
        labels.loc[tr, te] = "NA" if pd.isna(val) else f"{fmt.format(val)}\n{suffix}"

    arr = grid.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(cohorts)))
    ax.set_yticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts, rotation=35, ha="right")
    ax.set_yticklabels(cohorts)
    ax.set_xlabel("Test cohort")
    ax.set_ylabel("Train cohort")
    ax.set_title(title)

    for i in range(len(cohorts)):
        for j in range(len(cohorts)):
            ax.text(j, i, labels.iloc[i, j], ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    grid_dir = Path(args.grid_dir).expanduser().resolve()
    cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]

    pred_path = grid_dir / "transfer_grid_predictions_all.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    df = pd.read_csv(pred_path)
    needed = {"train_cohort", "test_cohort", "age_true", "pred_raw"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {pred_path}: {missing}")

    rows = []
    for (train, test), g in df.groupby(["train_cohort", "test_cohort"], sort=False):
        row = {
            "train_cohort": train,
            "test_cohort": test,
            "feature_set": g["feature_set"].iloc[0] if "feature_set" in g.columns else grid_dir.name,
            "cell_type": "diagonal_oof" if train == test else "external",
        }
        row.update(metric_row(g))
        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = grid_dir / "transfer_grid_metrics_RAW_vs_cBAG.csv"
    out.to_csv(out_csv, index=False)

    heatmaps = [
        ("MAE_raw", "RAW predicted-age MAE"),
        ("RMSE_raw", "RAW predicted-age RMSE"),
        ("R2_raw", "RAW predicted-age R²"),
        ("r_raw", "RAW predicted-age Pearson r"),
        ("mean_abs_cBAG", "Mean absolute cBAG"),
        ("cBAG_age_slope", "cBAG-age slope"),
    ]

    for metric, title in heatmaps:
        save_heatmap(
            out,
            metric=metric,
            cohorts=cohorts,
            out_png=grid_dir / f"heatmap_{metric}.png",
            title=f"{grid_dir.name}: {title}",
        )

    print("\nSaved audited transfer metrics:")
    print(f"  {out_csv}")
    for metric, _ in heatmaps:
        print(f"  {grid_dir / f'heatmap_{metric}.png'}")

    print("\nRecommended manuscript use:")
    print("  Use MAE_raw / RMSE_raw / R2_raw / r_raw for external age-prediction transfer.")
    print("  Use mean_abs_cBAG and cBAG_age_slope for residual/biological-validation stability.")
    print("  Do NOT present age-informed corrected MAE/RMSE as predictive transfer performance.")


if __name__ == "__main__":
    main()
