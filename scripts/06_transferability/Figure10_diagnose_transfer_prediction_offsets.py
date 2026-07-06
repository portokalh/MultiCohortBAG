#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability/imaging_only")
PRED = BASE / "transfer_grid_predictions_all.csv"

df = pd.read_csv(PRED)

rows = []
for (tr, te), g in df.groupby(["train_cohort", "test_cohort"]):
    age = pd.to_numeric(g["age_true"], errors="coerce")
    pred = pd.to_numeric(g["pred_raw"], errors="coerce")
    cbag = pd.to_numeric(g["cBAG_global"], errors="coerce")

    rows.append({
        "train_cohort": tr,
        "test_cohort": te,
        "n": len(g),
        "age_mean": age.mean(),
        "age_sd": age.std(),
        "age_min": age.min(),
        "age_max": age.max(),
        "pred_raw_mean": pred.mean(),
        "pred_raw_sd": pred.std(),
        "pred_raw_min": pred.min(),
        "pred_raw_max": pred.max(),
        "mean_pred_minus_age": (pred - age).mean(),
        "median_abs_pred_minus_age": (pred - age).abs().median(),
        "mean_cBAG": cbag.mean(),
        "mean_abs_cBAG": cbag.abs().mean(),
    })

out = pd.DataFrame(rows)
out = out.sort_values(["train_cohort", "test_cohort"])

out_path = BASE / "Figure10_prediction_offset_diagnostics.csv"
out.to_csv(out_path, index=False)

print(out.round(3).to_string(index=False))
print("\nSaved:", out_path)
