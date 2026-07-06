#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability/imaging_only")

PANELS = [
    ("A", "Raw MAE", "heatmap_raw_MAE.png"),
    ("B", "Intercept-recalibrated MAE", "heatmap_intercept_recal_MAE.png"),
    ("C", "Linear-recalibrated MAE", "heatmap_linear_recal_MAE.png"),
    ("D", "Raw R²", "heatmap_raw_R2.png"),
    ("E", "Intercept-recalibrated R²", "heatmap_intercept_recal_R2.png"),
    ("F", "Linear-recalibrated R²", "heatmap_linear_recal_R2.png"),
]

fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=300)
axes = axes.ravel()

for ax, (letter, title, fname) in zip(axes, PANELS):
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

out_png = BASE / "Figure10_recalibration_sensitivity_panel.png"
out_pdf = BASE / "Figure10_recalibration_sensitivity_panel.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(out_png)
print(out_pdf)
