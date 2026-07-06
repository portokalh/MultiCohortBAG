#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assemble_Figure6_ABCD_uniform_titles_final.py

Final 2x2 Figure 6 assembler with:
  - white-margin trimming
  - equal panel boxes
  - uniform panel letters/titles
  - no extra small footnote text added under panels
  - conservative zoom to avoid chopping labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--figure6-outdir", type=Path, required=True)
    p.add_argument("--feature-set", default="imaging_only")

    p.add_argument("--panel-a", type=Path, required=True)
    p.add_argument("--panel-b", type=Path, required=True)
    p.add_argument("--risk-plot", type=Path, required=True)
    p.add_argument("--longitudinal-plot", type=Path, required=True)

    p.add_argument("--out-prefix", default="Figure6_FINAL_ABCD_UNIFORM_FINAL")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--figure-title", default="Figure 6. Multidomain, risk-factor, and longitudinal validation of imaging-only cBAG")

    p.add_argument("--trim-threshold", type=int, default=245)
    p.add_argument("--trim-pad", type=int, default=24)

    p.add_argument("--box-width", type=int, default=2600)
    p.add_argument("--box-height", type=int, default=1780)

    p.add_argument("--zoom-a", type=float, default=1.00)
    p.add_argument("--zoom-b", type=float, default=0.96)
    p.add_argument("--zoom-c", type=float, default=0.90)
    p.add_argument("--zoom-d", type=float, default=0.90)

    p.add_argument("--title-band-frac", type=float, default=0.08)
    p.add_argument("--title-a", default="Multidomain validation")
    p.add_argument("--title-b", default="Discrimination targets")
    p.add_argument("--title-c", default="Risk-factor specificity")
    p.add_argument("--title-d", default="Longitudinal change")

    p.add_argument("--panel-label-size", type=int, default=30)
    p.add_argument("--panel-title-size", type=int, default=10)
    p.add_argument("--save-trimmed-panels", action="store_true")
    return p.parse_args()


def open_img(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    return img.convert("RGB")


def trim_white(img: Image.Image, threshold: int = 245, pad: int = 24) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    if not mask.any():
        return img

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img.width - 1, x1 + pad)
    y1 = min(img.height - 1, y1 + pad)

    return img.crop((x0, y0, x1 + 1, y1 + 1))


def fit_to_box(img: Image.Image, box_w: int, box_h: int, zoom: float = 1.0) -> Image.Image:
    scale = min(box_w / img.width, box_h / img.height) * zoom
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img2 = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (box_w, box_h), "white")
    if new_w <= box_w and new_h <= box_h:
        x = (box_w - new_w) // 2
        y = (box_h - new_h) // 2
        canvas.paste(img2, (x, y))
    else:
        left = max(0, (new_w - box_w) // 2)
        top = max(0, (new_h - box_h) // 2)
        crop = img2.crop((left, top, left + min(box_w, new_w), top + min(box_h, new_h)))
        x = max(0, (box_w - crop.width) // 2)
        y = max(0, (box_h - crop.height) // 2)
        canvas.paste(crop, (x, y))
    return canvas


def process(path: Path, threshold: int, pad: int, box_w: int, box_h: int, zoom: float) -> Image.Image:
    img = open_img(path)
    img = trim_white(img, threshold=threshold, pad=pad)
    img = fit_to_box(img, box_w, box_h, zoom=zoom)
    return img


def add_uniform_panel_decor(ax, letter: str, title: str, title_band_frac: float, panel_label_size: int, panel_title_size: int):
    band = Rectangle(
        (0, 1 - title_band_frac), 1, title_band_frac,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        zorder=3,
        alpha=0.99,
    )
    ax.add_patch(band)

    ax.text(
        0.005, 0.995, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=panel_label_size,
        fontweight="bold",
        zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.98, pad=2),
    )

    ax.text(
        0.52, 0.99 - 0.5 * title_band_frac, title,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=panel_title_size,
        fontweight="bold",
        zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.98, pad=1.5),
    )


def main():
    args = parse_args()
    args.figure6_outdir.mkdir(parents=True, exist_ok=True)

    panels = {
        "A": process(args.panel_a, args.trim_threshold, args.trim_pad, args.box_width, args.box_height, args.zoom_a),
        "B": process(args.panel_b, args.trim_threshold, args.trim_pad, args.box_width, args.box_height, args.zoom_b),
        "C": process(args.risk_plot, args.trim_threshold, args.trim_pad, args.box_width, args.box_height, args.zoom_c),
        "D": process(args.longitudinal_plot, args.trim_threshold, args.trim_pad, args.box_width, args.box_height, args.zoom_d),
    }

    if args.save_trimmed_panels:
        for k, img in panels.items():
            img.save(args.figure6_outdir / f"{args.out_prefix}_{args.feature_set}_panel{k}_box.png")

    fig, axes = plt.subplots(2, 2, figsize=(17.0, 12.0))
    mapping = [
        ("A", axes[0, 0], args.title_a),
        ("B", axes[0, 1], args.title_b),
        ("C", axes[1, 0], args.title_c),
        ("D", axes[1, 1], args.title_d),
    ]

    for letter, ax, title in mapping:
        ax.imshow(np.asarray(panels[letter]))
        ax.axis("off")
        add_uniform_panel_decor(
            ax=ax,
            letter=letter,
            title=title,
            title_band_frac=args.title_band_frac,
            panel_label_size=args.panel_label_size,
            panel_title_size=args.panel_title_size,
        )

    fig.suptitle(args.figure_title, fontsize=15.5, fontweight="bold", y=0.992)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.01, wspace=0.035, hspace=0.08)

    png = args.figure6_outdir / f"{args.out_prefix}_{args.feature_set}.png"
    pdf = args.figure6_outdir / f"{args.out_prefix}_{args.feature_set}.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("[DONE]", png)
    print("[DONE]", pdf)


if __name__ == "__main__":
    main()
