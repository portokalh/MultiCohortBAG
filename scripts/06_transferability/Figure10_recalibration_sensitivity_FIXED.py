#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

BASE = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability/imaging_only")
PRED = BASE / "transfer_grid_predictions_all.csv"
COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

def safe_r(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    y = y[m]
    pred = pred[m]
    if len(y) < 3 or np.nanstd(y) == 0 or np.nanstd(pred) == 0:
        return np.nan
    return float(pearsonr(y, pred)[0])

def calc_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    y = y[m]
    pred = pred[m]
    err = pred - y

    if len(y) == 0:
        return {
            "n": 0, "MAE": np.nan, "MdAE": np.nan, "RMSE": np.nan,
            "R2": np.nan, "r": np.nan, "mean_error": np.nan,
            "n_pred_lt0": np.nan, "n_pred_gt120": np.nan,
            "n_abs_error_gt50": np.nan,
        }

    return {
        "n": int(len(y)),
        "MAE": float(mean_absolute_error(y, pred)),
        "MdAE": float(np.median(np.abs(err))),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
        "r": safe_r(y, pred),
        "mean_error": float(np.mean(err)),
        "n_pred_lt0": int((pred < 0).sum()),
        "n_pred_gt120": int((pred > 120).sum()),
        "n_abs_error_gt50": int((np.abs(err) > 50).sum()),
    }

def intercept_recal(y, pred):
    return pred + (np.nanmean(y) - np.nanmean(pred))

def linear_recal(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)

    out = np.full_like(pred, np.nan, dtype=float)
    if m.sum() < 3 or np.nanstd(pred[m]) == 0:
        return out, np.nan, np.nan

    model = LinearRegression()
    model.fit(pred[m].reshape(-1, 1), y[m])
    out[m] = model.predict(pred[m].reshape(-1, 1))

    return out, float(model.intercept_), float(model.coef_[0])

def make_grid(df, mode, metric):
    sub = df[df["mode"] == mode].copy()
    grid = pd.DataFrame(index=COHORTS, columns=COHORTS, dtype=float)

    for _, row in sub.iterrows():
        tr = row["train_cohort"]
        te = row["test_cohort"]
        if tr in COHORTS and te in COHORTS:
            grid.loc[tr, te] = pd.to_numeric(row[metric], errors="coerce")

    return grid

def fmt_val(val, metric, vmin, vmax):
    if pd.isna(val):
        return "NA"
    val = float(val)
    if metric == "R2" and val < vmin:
        return f"<{vmin:g}"
    if metric in ["MAE", "RMSE", "MdAE"]:
        return f"{val:.1f}"
    return f"{val:.2f}"

def save_heatmap(df, mode, metric, out_png, title, cmap, limits):
    grid = make_grid(df, mode, metric)
    arr = grid.to_numpy(dtype=float)

    vmin, vmax = limits
    display_arr = np.clip(arr, vmin, vmax)

    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=300)
    im = ax.imshow(display_arr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.88)

    ax.set_xticks(range(len(COHORTS)))
    ax.set_yticks(range(len(COHORTS)))
    ax.set_xticklabels(COHORTS, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(COHORTS, fontsize=9)

    ax.set_xlabel("Test cohort", fontsize=10)
    ax.set_ylabel("Train cohort", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    ax.set_xticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(COHORTS)):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))

    for i, tr in enumerate(COHORTS):
        for j, te in enumerate(COHORTS):
            val = grid.loc[tr, te]
            tag = "OOF" if tr == te else "EXT"
            ax.text(
                j, i, f"{fmt_val(val, metric, vmin, vmax)}\n{tag}",
                ha="center", va="center",
                fontsize=7.8,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.60, pad=1.2),
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def assemble_panel():
    import matplotlib.image as mpimg

    panels = [
        ("A", "Raw MAE", "heatmap_recal_sens_raw_MAE.png"),
        ("B", "Intercept-recalibrated MAE", "heatmap_recal_sens_intercept_MAE.png"),
        ("C", "Linear-recalibrated MAE", "heatmap_recal_sens_linear_MAE.png"),
        ("D", "Raw R²", "heatmap_recal_sens_raw_R2.png"),
        ("E", "Intercept-recalibrated R²", "heatmap_recal_sens_intercept_R2.png"),
        ("F", "Linear-recalibrated R²", "heatmap_recal_sens_linear_R2.png"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=300)
    axes = axes.ravel()

    for ax, (letter, title, fname) in zip(axes, panels):
        p = BASE / fname
        ax.axis("off")
        if not p.exists():
            ax.text(0.5, 0.5, f"Missing:\n{fname}", ha="center", va="center", color="red")
        else:
            ax.imshow(mpimg.imread(p))
        ax.set_title(f"{letter}. {title}", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Supplementary Figure. Post-hoc recalibration sensitivity of cross-cohort transfer",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = BASE / "Figure10_recalibration_sensitivity_panel_FIXED.png"
    out_pdf = BASE / "Figure10_recalibration_sensitivity_panel_FIXED.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved panel:")
    print(out_png)
    print(out_pdf)

def main():
    pred = pd.read_csv(PRED)
    pred["age_true"] = pd.to_numeric(pred["age_true"], errors="coerce")
    pred["pred_raw"] = pd.to_numeric(pred["pred_raw"], errors="coerce")

    rows = []

    for (train, test), g in pred.groupby(["train_cohort", "test_cohort"], sort=False):
        y = g["age_true"].to_numpy(float)
        p = g["pred_raw"].to_numpy(float)

        p_intercept = intercept_recal(y, p)
        p_linear, alpha, beta = linear_recal(y, p)

        for mode, pvec, a, b in [
            ("raw", p, 0.0, 1.0),
            ("intercept", p_intercept, float(np.nanmean(y) - np.nanmean(p)), 1.0),
            ("linear", p_linear, alpha, beta),
        ]:
            row = {
                "train_cohort": train,
                "test_cohort": test,
                "cell_type": "diagonal_oof" if train == test else "external",
                "mode": mode,
                "recal_alpha": a,
                "recal_beta": b,
            }
            row.update(calc_metrics(y, pvec))
            rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = BASE / "transfer_grid_metrics_RECALIBRATION_SENSITIVITY_FIXED.csv"
    out.to_csv(out_csv, index=False)

    summary_rows = []
    for mode in ["raw", "intercept", "linear"]:
        sub0 = out[out["mode"] == mode]
        for group_name, sub in [
            ("diagonal_oof", sub0[sub0["train_cohort"] == sub0["test_cohort"]]),
            ("external_transfer", sub0[sub0["train_cohort"] != sub0["test_cohort"]]),
        ]:
            summary_rows.append({
                "mode": mode,
                "group": group_name,
                "n_cells": len(sub),
                "median_MAE": sub["MAE"].median(),
                "median_MdAE": sub["MdAE"].median(),
                "median_RMSE": sub["RMSE"].median(),
                "median_r": sub["r"].median(),
                "median_R2": sub["R2"].median(),
                "total_pred_lt0": int(sub["n_pred_lt0"].sum()),
                "total_pred_gt120": int(sub["n_pred_gt120"].sum()),
                "total_abs_error_gt50": int(sub["n_abs_error_gt50"].sum()),
            })

    summary = pd.DataFrame(summary_rows)
    summary_csv = BASE / "Figure10_recalibration_summary_FIXED.csv"
    summary.to_csv(summary_csv, index=False)

    print("Recalibration summary:")
    print(summary.round(3).to_string(index=False))

    limits = {
        "MAE": (0, 85),
        "RMSE": (0, 85),
        "R2": (-20, 1.0),
        "r": (-0.05, 0.65),
    }

    cmap = {
        "MAE": "YlOrRd",
        "RMSE": "YlOrRd",
        "R2": "RdBu_r",
        "r": "YlGnBu",
    }

    for mode in ["raw", "intercept", "linear"]:
        save_heatmap(out, mode, "MAE", BASE / f"heatmap_recal_sens_{mode}_MAE.png", f"{mode}: MAE", cmap["MAE"], limits["MAE"])
        save_heatmap(out, mode, "R2",  BASE / f"heatmap_recal_sens_{mode}_R2.png",  f"{mode}: R²",  cmap["R2"],  limits["R2"])

    assemble_panel()

    print("Saved tables:")
    print(out_csv)
    print(summary_csv)

if __name__ == "__main__":
    main()
