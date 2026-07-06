#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability/imaging_only")

PRED = BASE / "transfer_grid_predictions_all.csv"
RAW = BASE / "transfer_grid_metrics_RAW_vs_cBAG.csv"

pred = pd.read_csv(PRED)
raw = pd.read_csv(RAW)

pred["age_true"] = pd.to_numeric(pred["age_true"], errors="coerce")
pred["pred_raw"] = pd.to_numeric(pred["pred_raw"], errors="coerce")
pred["raw_error"] = pred["pred_raw"] - pred["age_true"]
pred["abs_raw_error"] = pred["raw_error"].abs()

rows = []
for (tr, te), g in pred.groupby(["train_cohort", "test_cohort"]):
    rows.append({
        "train_cohort": tr,
        "test_cohort": te,
        "median_abs_error_raw": g["abs_raw_error"].median(),
        "mean_error_raw": g["raw_error"].mean(),
        "median_error_raw": g["raw_error"].median(),
        "pred_raw_mean": g["pred_raw"].mean(),
        "pred_raw_sd": g["pred_raw"].std(),
        "pred_raw_min": g["pred_raw"].min(),
        "pred_raw_max": g["pred_raw"].max(),
        "n_pred_lt0": int((g["pred_raw"] < 0).sum()),
        "n_pred_gt120": int((g["pred_raw"] > 120).sum()),
        "n_abs_error_gt50": int((g["abs_raw_error"] > 50).sum()),
        "pct_abs_error_gt50": 100 * float((g["abs_raw_error"] > 50).mean()),
    })

robust = pd.DataFrame(rows)

merged = raw.merge(robust, on=["train_cohort", "test_cohort"], how="left")

out = BASE / "transfer_grid_metrics_RAW_vs_cBAG_WITH_ROBUST_QC.csv"
merged.to_csv(out, index=False)

summary = (
    merged.assign(is_external=merged["train_cohort"] != merged["test_cohort"])
    .groupby("is_external")
    .agg(
        n_cells=("train_cohort", "size"),
        median_MAE_raw=("MAE_raw", "median"),
        median_MdAE_raw=("median_abs_error_raw", "median"),
        median_RMSE_raw=("RMSE_raw", "median"),
        median_r_raw=("r_raw", "median"),
        median_R2_raw=("R2_raw", "median"),
        total_pred_lt0=("n_pred_lt0", "sum"),
        total_pred_gt120=("n_pred_gt120", "sum"),
        total_abs_error_gt50=("n_abs_error_gt50", "sum"),
    )
    .reset_index()
)
summary["group"] = summary["is_external"].map({False: "diagonal_oof", True: "external_transfer"})
summary = summary.drop(columns=["is_external"])

out2 = BASE / "Figure10_robust_transfer_summary.csv"
summary.to_csv(out2, index=False)

print("\nMerged robust QC:")
print(merged[[
    "train_cohort", "test_cohort", "cell_type", "n",
    "MAE_raw", "median_abs_error_raw", "RMSE_raw", "r_raw", "R2_raw",
    "n_pred_lt0", "n_pred_gt120", "n_abs_error_gt50", "pct_abs_error_gt50",
    "pred_raw_min", "pred_raw_max"
]].round(3).to_string(index=False))

print("\nSummary:")
print(summary.round(3).to_string(index=False))

print("\nSaved:")
print(out)
print(out2)
