#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability/imaging_only")
PRED = BASE / "transfer_grid_predictions_all.csv"

df = pd.read_csv(PRED)

df["age_true"] = pd.to_numeric(df["age_true"], errors="coerce")
df["pred_raw"] = pd.to_numeric(df["pred_raw"], errors="coerce")
df["raw_error"] = df["pred_raw"] - df["age_true"]
df["abs_raw_error"] = df["raw_error"].abs()
df["is_external"] = df["train_cohort"] != df["test_cohort"]

# Extreme absolute prediction values or extreme errors
extreme = df[
    (df["is_external"]) &
    (
        (df["pred_raw"] < 0) |
        (df["pred_raw"] > 120) |
        (df["abs_raw_error"] > 50)
    )
].copy()

extreme = extreme.sort_values(
    ["abs_raw_error", "train_cohort", "test_cohort"],
    ascending=[False, True, True]
)

out1 = BASE / "Figure10_extreme_transfer_predictions.csv"
extreme.to_csv(out1, index=False)

# Cell-level outlier counts
cell = (
    df[df["is_external"]]
    .groupby(["train_cohort", "test_cohort"])
    .agg(
        n=("graph_id", "size"),
        n_pred_lt0=("pred_raw", lambda x: int((x < 0).sum())),
        n_pred_gt120=("pred_raw", lambda x: int((x > 120).sum())),
        n_abs_error_gt50=("abs_raw_error", lambda x: int((x > 50).sum())),
        max_pred=("pred_raw", "max"),
        min_pred=("pred_raw", "min"),
        max_abs_error=("abs_raw_error", "max"),
        median_abs_error=("abs_raw_error", "median"),
    )
    .reset_index()
)

out2 = BASE / "Figure10_extreme_transfer_prediction_counts_by_cell.csv"
cell.to_csv(out2, index=False)

print("\nExtreme predictions:")
print(extreme[[
    "graph_id", "train_cohort", "test_cohort",
    "age_true", "pred_raw", "raw_error", "abs_raw_error",
    "pred_bias_corrected_global", "cBAG_global"
]].head(50).round(3).to_string(index=False))

print("\nCell-level outlier counts:")
print(cell.round(3).to_string(index=False))

print("\nSaved:")
print(out1)
print(out2)
