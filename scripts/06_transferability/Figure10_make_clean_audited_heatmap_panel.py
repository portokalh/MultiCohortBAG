#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability/imaging_only")
CSV = BASE / "transfer_grid_metrics_RAW_vs_cBAG.csv"

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

PANELS = [
    ("A", "Raw MAE", "MAE_raw", "Years", "YlOrRd", "{:.1f}", (0, 85)),
    ("B", "Raw RMSE", "RMSE_raw", "Years", "YlOrRd", "{:.1f}", (0, 85)),
    ("C", "Raw Pearson r", "r_raw", "r", "YlGnBu", "{:.2f}", (-0.05, 0.60)),
    ("D", "Raw R²", "R2_raw", "R²", "RdBu_r", "{:.2f}", (-20, 0.5)),
    ("E", "Mean |cBAG|", "mean_abs_cBAG", "Years", "YlOrRd", "{:.1f}", (0, 70)),
    ("F", "cBAG-age slope", "cBAG_age_slope", "Slope", "RdBu_r", "{:.2f}", (-0.55, 0.25)),
]

def make_grid(df, metric):
    grid = pd.DataFrame(index=COHORTS, columns=COHORTS, dtype=float)
    for _, row in df.iterrows():
        tr = row["train_cohort"]
        te = row["test_cohort"]
        if tr in COHORTS and te in COHORTS:
            grid.loc[tr, te] = pd.to_numeric(row[metric], errors="coerce")
    return grid

def format_value(val, fmt, metric, vmin, vmax):
    if not np.isfinite(val):
        return "NA"
    if metric == "R2_raw" and val < vmin:
        return f"<{vmin:g}"
    if metric == "R2_raw" and val > vmax:
        return f">{vmax:g}"
    return fmt.format(val)

def draw_heatmap(ax, df, letter, title, metric, cbar_label, cmap, fmt, limits):
    grid = make_grid(df, metric)
    arr = grid.to_numpy(dtype=float)

    vmin, vmax = limits
    display_arr = np.clip(arr, vmin, vmax)

    im = ax.imshow(display_arr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.88)

    ax.set_xticks(range(len(COHORTS)))
    ax.set_yticks(range(len(COHORTS)))
    ax.set_xticklabels(COHORTS, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(COHORTS, fontsize=9)

    ax.set_xlabel("Test cohort", fontsize=10)
    ax.set_ylabel("Train cohort", fontsize=10)
    ax.set_title(f"{letter}. {title}", fontsize=13, fontweight="bold", pad=8)

    ax.set_xticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
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
            label = format_value(float(val), fmt, metric, vmin, vmax) if pd.notna(val) else "NA"
            tag = "OOF" if tr == te else "EXT"
            ax.text(
                j,
                i,
                f"{label}\n{tag}",
                ha="center",
                va="center",
                fontsize=7.8,
                color="black",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.60, pad=1.2),
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

def main():
    df = pd.read_csv(CSV)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    axes = axes.ravel()

    for ax, panel in zip(axes, PANELS):
        draw_heatmap(ax, df, *panel)

    fig.suptitle(
        "Figure 10. Cross-cohort transferability of imaging-only brain-age models",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = BASE / "Figure10_imaging_only_audited_transfer_heatmaps_panel_CLEAN.png"
    out_pdf = BASE / "Figure10_imaging_only_audited_transfer_heatmaps_panel_CLEAN.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(out_png)
    print(out_pdf)

if __name__ == "__main__":
    main()
