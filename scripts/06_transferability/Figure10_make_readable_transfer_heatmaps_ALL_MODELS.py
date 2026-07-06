#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability_ALL_MODELS")
COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

METRICS = [
    ("MAE_raw", "Raw MAE", "YlOrRd", "{:.1f}", True),
    ("RMSE_raw", "Raw RMSE", "YlOrRd", "{:.1f}", True),
    ("r_raw", "Raw Pearson r", "YlGnBu", "{:.2f}", False),
    ("R2_raw", "Raw R²", "RdBu_r", "{:.2f}", False),
    ("mean_abs_cBAG", "Mean absolute cBAG", "YlOrRd", "{:.1f}", True),
    ("cBAG_age_slope", "cBAG–age slope", "RdBu_r", "{:.2f}", False),
]

def robust_limits(values, metric):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0, 1

    if metric == "R2_raw":
        # Extreme failed-transfer cells otherwise dominate the scale.
        lo = max(np.nanpercentile(v, 10), -20)
        hi = min(np.nanpercentile(v, 90), 0.5)
        if lo >= hi:
            lo, hi = np.nanmin(v), np.nanmax(v)
        return lo, hi

    if metric == "cBAG_age_slope":
        lim = np.nanpercentile(np.abs(v), 90)
        lim = max(lim, 0.05)
        return -lim, lim

    lo, hi = np.nanpercentile(v, [5, 95])
    if lo == hi:
        lo, hi = np.nanmin(v), np.nanmax(v)
    if lo == hi:
        hi = lo + 1
    return lo, hi

def make_grid(df, metric):
    grid = pd.DataFrame(index=COHORTS, columns=COHORTS, dtype=float)
    for _, row in df.iterrows():
        tr = row["train_cohort"]
        te = row["test_cohort"]
        if tr in COHORTS and te in COHORTS:
            grid.loc[tr, te] = pd.to_numeric(row[metric], errors="coerce")
    return grid

def save_heatmap(df, metric, title, cmap, fmt, out_png, out_pdf):
    grid = make_grid(df, metric)
    arr = grid.to_numpy(dtype=float)
    vmin, vmax = robust_limits(arr.ravel(), metric)

    fig, ax = plt.subplots(figsize=(5.3, 4.8), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.82)

    ax.set_xticks(range(len(COHORTS)))
    ax.set_yticks(range(len(COHORTS)))
    ax.set_xticklabels(COHORTS, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(COHORTS, fontsize=10)

    ax.set_xlabel("Test cohort", fontsize=11)
    ax.set_ylabel("Train cohort", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)

    ax.set_xticks(np.arange(-.5, len(COHORTS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(COHORTS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(COHORTS)):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
            )
        )

    for i, tr in enumerate(COHORTS):
        for j, te in enumerate(COHORTS):
            val = grid.loc[tr, te]
            if pd.isna(val):
                label = "NA"
            else:
                label = fmt.format(float(val))
            ax.text(
                j, i, label,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.2),
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)

def main():
    generated = []

    for d in sorted(BASE.iterdir()):
        if not d.is_dir() or d.name == "logs":
            continue

        metrics_path = d / "transfer_grid_metrics_RAW_vs_cBAG.csv"
        if not metrics_path.exists():
            continue

        df = pd.read_csv(metrics_path)
        outdir = d / "readable_heatmaps"
        outdir.mkdir(exist_ok=True)

        for metric, title, cmap, fmt, _lower in METRICS:
            if metric not in df.columns:
                continue
            safe = metric.replace("²", "2").replace("–", "_").replace("-", "_")
            out_png = outdir / f"readable_heatmap_{safe}.png"
            out_pdf = outdir / f"readable_heatmap_{safe}.pdf"
            save_heatmap(
                df=df,
                metric=metric,
                title=f"{d.name}: {title}",
                cmap=cmap,
                fmt=fmt,
                out_png=out_png,
                out_pdf=out_pdf,
            )
            generated.append(str(out_png))

    print("Generated readable heatmaps:", len(generated))
    for p in generated:
        print(p)

if __name__ == "__main__":
    main()
