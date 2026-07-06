#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assemble paper-ready Figure 7 with SHAP panels plus top common-4 edge table.
"""

from pathlib import Path
import textwrap

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


OUT = Path("/mnt/newStor/paros/paros_WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure7_SHAP")

PAPER = OUT / "paper_ready"
COMMON = OUT / "common4_consensus" / "imaging_only"
FINAL = OUT / "cross_cohort_aggregated" / "final_figures"

GLOBAL_PLOT_CANDIDATES = [
    COMMON / "Figure7_imaging_only_global_SHAP_common4_top25_cohort_heatmap.png",
    COMMON / "common4_imaging_only_global_SHAP_top25_cohort_heatmap.png",
]

NODE_PLOT_CANDIDATES = [
    COMMON / "Figure7_imaging_only_node_SHAP_common4_top25_cohort_heatmap.png",
    COMMON / "common4_imaging_only_node_SHAP_top25_cohort_heatmap.png",
]

EDGE_PLOT_CANDIDATES = [
    PAPER / "Figure7_paper_ready_common4_edges_minN100_top25.png",
    COMMON / "Figure7_imaging_only_edge_SHAP_common4_top25_cohort_heatmap.png",
    COMMON / "common4_imaging_only_edge_SHAP_top25_cohort_heatmap.png",
]

REGION_PLOT_CANDIDATES = [
    OUT / "cross_cohort_aggregated" / "imaging_only" / "Supplementary_Figure6_cross_cohort_imaging_only_edge_derived_region_SHAP_top25_barplot.png",
    FINAL / "Figure6_combined_imaging_only_model_WITH_EDGES.png",
]

EDGE_TABLE = PAPER / "Figure7_paper_ready_common4_edges_minN100.csv"

OUT_PNG = PAPER / "Figure7_paper_ready_SHAP_with_common4_edge_table.png"
OUT_PDF = PAPER / "Figure7_paper_ready_SHAP_with_common4_edge_table.pdf"


def find_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def add_image_panel(ax, path, label, title=None):
    ax.axis("off")
    ax.text(-0.02, 1.02, label, transform=ax.transAxes, fontsize=16, fontweight="bold")

    if path is None or not Path(path).exists():
        ax.text(0.5, 0.5, f"Missing panel:\n{path}", ha="center", va="center", fontsize=10)
        return

    img = mpimg.imread(str(path))
    ax.imshow(img)

    if title:
        ax.set_title(title, fontsize=11, pad=6)


def wrap_text(x, width=42):
    return "\n".join(textwrap.wrap(str(x), width=width, break_long_words=False))


def add_edge_table(ax, table_csv, top_n=10):
    ax.axis("off")
    ax.text(-0.02, 1.02, "E", transform=ax.transAxes, fontsize=16, fontweight="bold")
    ax.set_title("Top common-4 exact edge SHAP contributors", fontsize=11, pad=6)

    if not table_csv.exists():
        ax.text(0.5, 0.5, f"Missing table:\n{table_csv}", ha="center", va="center", fontsize=10)
        return

    df = pd.read_csv(table_csv).head(top_n).copy()

    show = pd.DataFrame({
        "Edge": df["Edge"].map(lambda x: wrap_text(x, 44)),
        "Mean |SHAP|": df["mean_abs_SHAP_across_cohorts"].map(lambda x: f"{x:.4f}"),
        "Signed SHAP": df["mean_signed_SHAP_across_cohorts"].map(lambda x: f"{x:.4f}"),
        "N": df["total_n_subjects"].astype(int).astype(str),
    })

    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.64, 0.13, 0.13, 0.07],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_text_props(weight="bold")

    ax.text(
        0.0,
        -0.05,
        "Edges restricted to n_cohorts = 4 and total N ≥ 100; ranked by mean |SHAP| across cohorts.",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )


def main():
    PAPER.mkdir(parents=True, exist_ok=True)

    global_plot = find_existing(GLOBAL_PLOT_CANDIDATES)
    node_plot = find_existing(NODE_PLOT_CANDIDATES)
    edge_plot = find_existing(EDGE_PLOT_CANDIDATES)
    region_plot = find_existing(REGION_PLOT_CANDIDATES)

    print("Using panels:")
    print("  global:", global_plot)
    print("  node:  ", node_plot)
    print("  region:", region_plot)
    print("  edge:  ", edge_plot)
    print("  table: ", EDGE_TABLE)

    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.85],
        width_ratios=[1.0, 1.0],
        hspace=0.18,
        wspace=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    add_image_panel(ax_a, global_plot, "A", "Global SHAP contributors across cohorts")
    add_image_panel(ax_b, node_plot, "B", "Node-feature SHAP contributors across cohorts")
    add_image_panel(ax_c, region_plot, "C", "Edge-derived regional SHAP structure")
    add_image_panel(ax_d, edge_plot, "D", "Exact edge SHAP contributors present in all cohorts")
    add_edge_table(ax_e, EDGE_TABLE, top_n=10)

    fig.suptitle(
        "Figure 7. Cross-cohort SHAP contributors to predicted brain age",
        fontsize=18,
        y=0.995,
    )

    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_PNG)
    print("Saved:", OUT_PDF)


if __name__ == "__main__":
    main()
