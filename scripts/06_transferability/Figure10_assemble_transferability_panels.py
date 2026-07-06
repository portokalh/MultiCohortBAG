#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ORIGINAL_PANELS = [
    ("A", "MAE corrected", "heatmap_MAE.png"),
    ("B", "RMSE corrected", "heatmap_RMSE.png"),
    ("C", "Pearson r corrected", "heatmap_r.png"),
    ("D", "R² corrected", "heatmap_R2.png"),
    ("E", "cBAG-age slope", "heatmap_cBAG_age_slope.png"),
]

AUDITED_PANELS = [
    ("A", "Raw MAE", "heatmap_MAE_raw.png"),
    ("B", "Raw RMSE", "heatmap_RMSE_raw.png"),
    ("C", "Raw Pearson r", "heatmap_r_raw.png"),
    ("D", "Raw R²", "heatmap_R2_raw.png"),
    ("E", "Mean |cBAG|", "heatmap_mean_abs_cBAG.png"),
    ("F", "cBAG-age slope", "heatmap_cBAG_age_slope.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--prefix", default="Figure10")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def add_original_note(ax) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        "Original output note",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.36,
        "Original script outputs age-informed corrected metrics.\n"
        "Use the audited/raw panel for manuscript\n"
        "predictive transfer performance.",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )


def assemble_panel(
    input_dir: Path,
    panel_defs: list[tuple[str, str, str]],
    outstem: Path,
    suptitle: str,
    dpi: int,
    note_last: bool = False,
) -> tuple[Path, Path]:
    ncols = 3
    nslots = 6
    nrows = math.ceil(nslots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10), dpi=dpi)
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for idx, (letter, title, filename) in enumerate(panel_defs):
        ax = axes[idx]
        img_path = input_dir / filename

        if not img_path.exists():
            ax.text(
                0.5,
                0.5,
                f"Missing:\n{filename}",
                ha="center",
                va="center",
                fontsize=12,
                color="red",
                transform=ax.transAxes,
            )
        else:
            img = mpimg.imread(img_path)
            ax.imshow(img)

        ax.axis("off")
        ax.set_title(f"{letter}. {title}", fontsize=13, fontweight="bold", pad=8)

    if note_last and len(panel_defs) < len(axes):
        add_original_note(axes[len(panel_defs)])

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png = outstem.with_suffix(".png")
    pdf = outstem.with_suffix(".pdf")

    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    print("=" * 100)
    print("Figure 10 heatmap panel assembly")
    print("=" * 100)
    print(f"Input directory: {input_dir}")
    print(f"Prefix:          {args.prefix}")
    print(f"DPI:             {args.dpi}")

    original_png, original_pdf = assemble_panel(
        input_dir=input_dir,
        panel_defs=ORIGINAL_PANELS,
        outstem=input_dir / f"{args.prefix}_original_transfer_heatmaps_panel",
        suptitle="Figure 10. Cross-cohort transferability heatmaps: original corrected outputs",
        dpi=args.dpi,
        note_last=True,
    )

    audited_png, audited_pdf = assemble_panel(
        input_dir=input_dir,
        panel_defs=AUDITED_PANELS,
        outstem=input_dir / f"{args.prefix}_audited_transfer_heatmaps_panel",
        suptitle="Figure 10. Cross-cohort transferability heatmaps: audited manuscript metrics",
        dpi=args.dpi,
        note_last=False,
    )

    print("\nSaved assembled panels:")
    print(f"  ORIGINAL PNG: {original_png}")
    print(f"  ORIGINAL PDF: {original_pdf}")
    print(f"  AUDITED  PNG: {audited_png}")
    print(f"  AUDITED  PDF: {audited_pdf}")
    print("\nRecommended manuscript file:")
    print(f"  {audited_pdf}")


if __name__ == "__main__":
    main()
