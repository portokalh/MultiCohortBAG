#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability/imaging_only")

panels = [
    ("A", "Raw MAE", BASE / "heatmap_MAE_raw.png"),
    ("B", "Raw RMSE", BASE / "heatmap_RMSE_raw.png"),
    ("C", "Raw Pearson r", BASE / "heatmap_r_raw.png"),
    ("D", "Raw R²", BASE / "heatmap_R2_raw.png"),
    ("E", "Mean absolute cBAG", BASE / "heatmap_mean_abs_cBAG.png"),
    ("F", "cBAG–age slope", BASE / "heatmap_cBAG_age_slope.png"),
]

missing = [str(p) for _, _, p in panels if not p.exists()]
if missing:
    raise FileNotFoundError("Missing input heatmaps:\n" + "\n".join(missing))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
axes = axes.ravel()

for ax, (letter, title, path) in zip(axes, panels):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis("off")
    ax.text(
        0.01, 0.99, letter,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )
    ax.set_title(title, fontsize=12, pad=6)

fig.suptitle(
    "Figure 10. Cross-cohort transferability of imaging-only brain-age models",
    fontsize=16,
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.965])

out_png = BASE / "Figure10_cross_cohort_transferability_panel.png"
out_pdf = BASE / "Figure10_cross_cohort_transferability_panel.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(out_png)
print(out_pdf)
