#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_Figure6C_risk_factor_specificity_forest_HARDCAP.py

Panel C forest plotter with a hard row cap.

Difference from previous version:
    --max-rows 20 means exactly at most 20 rows, even if more than 20 rows are FDR-significant.

Ranking:
    1. FDR-significant rows first
    2. risk-factor family order
    3. q-value
    4. p-value
    5. absolute effect size

Inputs:
    SupplTable8_cBAG_risk_factor_effects_selected.csv
or:
    supplementary/SupplementaryTable_S8_cBAG_risk_factor_effects_selected.csv

Outputs:
    <prefix>.png/pdf
    <prefix>_source_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILY_ORDER = [
    "demographic",
    "genetic",
    "clinical_status",
    "metabolic_adiposity",
    "metabolic_diabetes",
    "lipids",
    "vascular",
]

FAMILY_LABELS = {
    "demographic": "Demographic",
    "genetic": "Genetic",
    "clinical_status": "Clinical status",
    "metabolic_adiposity": "Adiposity",
    "metabolic_diabetes": "Diabetes / glycemia",
    "lipids": "Lipids",
    "vascular": "Vascular",
}

FAMILY_COLORS = {
    "demographic": "#4d4d4d",
    "genetic": "#756bb1",
    "clinical_status": "#2b8cbe",
    "metabolic_adiposity": "#238b45",
    "metabolic_diabetes": "#41ab5d",
    "lipids": "#fd8d3c",
    "vascular": "#de2d26",
}


def parse_args():
    p = argparse.ArgumentParser(description="Make hard-capped Figure 6C risk-factor forest plot.")
    p.add_argument("--risk-dir", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--selected-csv", type=Path, default=None)
    p.add_argument("--prefix", default="Figure6C_RiskFactorSpecificity_TOP20_HARDCAP_forest")
    p.add_argument("--max-rows", type=int, default=20)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default="Figure 6C. Risk-factor specificity of cBAG")
    p.add_argument("--include-ns", action="store_true",
                   help="Allow nonsignificant rows after significant rows. Default true-style behavior is top ranked regardless, but hard capped.")
    return p.parse_args()


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def p_to_stars(q):
    try:
        q = float(q)
    except Exception:
        return ""
    if np.isnan(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def load_selected(args):
    if args.selected_csv:
        if not args.selected_csv.exists():
            raise FileNotFoundError(args.selected_csv)
        return pd.read_csv(args.selected_csv, low_memory=False), args.selected_csv

    candidates = [
        args.risk_dir / "SupplTable8_cBAG_risk_factor_effects_selected.csv",
        args.risk_dir / "supplementary" / "SupplementaryTable_S8_cBAG_risk_factor_effects_selected.csv",
        args.risk_dir / "supplementary" / "SupplementaryTable_S8_cBAG_risk_factor_effects_selected_independent.csv",
    ]
    f = first_existing(candidates)
    if f is None:
        raise FileNotFoundError("Could not find selected risk-factor table in: " + str(args.risk_dir))
    return pd.read_csv(f, low_memory=False), f


def prepare(df, max_rows):
    x = df.copy()
    if "status" in x.columns:
        x = x[x["status"].astype(str).eq("ok")].copy()

    for c in [
        "p_cBAG", "q_cBAG_within_family", "beta_cBAG_std", "beta_ci_low", "beta_ci_high",
        "or_per_1sd_cBAG", "or_ci_low", "or_ci_high", "partial_r_cBAG", "delta_auc", "n_model",
    ]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    if "risk_factor_family" not in x.columns:
        x["risk_factor_family"] = "risk_factor"
    if "semantic_label" not in x.columns:
        x["semantic_label"] = x.get("column", "").astype(str)
    if "model_type" not in x.columns:
        x["model_type"] = np.where(x.get("or_per_1sd_cBAG", pd.Series(np.nan, index=x.index)).notna(), "binary", "continuous")

    rows = []
    for _, r in x.iterrows():
        model_type = str(r.get("model_type", "")).lower()
        fam = str(r.get("risk_factor_family", "risk_factor"))
        q = r.get("q_cBAG_within_family", np.nan)
        p = r.get("p_cBAG", np.nan)
        n = r.get("n_model", np.nan)

        if model_type == "binary":
            orv = r.get("or_per_1sd_cBAG", np.nan)
            lo = r.get("or_ci_low", np.nan)
            hi = r.get("or_ci_high", np.nan)
            if not np.isfinite(orv) or orv <= 0:
                continue
            effect = np.log(orv)
            ci_low = np.log(lo) if np.isfinite(lo) and lo > 0 else np.nan
            ci_high = np.log(hi) if np.isfinite(hi) and hi > 0 else np.nan
            display_effect = f"OR={orv:.2f}"
        else:
            effect = r.get("beta_cBAG_std", np.nan)
            ci_low = r.get("beta_ci_low", np.nan)
            ci_high = r.get("beta_ci_high", np.nan)
            if not np.isfinite(effect):
                continue
            display_effect = f"β={effect:.2f}"

        cohort = str(r.get("cohort", ""))
        label = f"{cohort} | {FAMILY_LABELS.get(fam, fam)} | {r.get('semantic_label','')}"

        rows.append({
            "cohort": cohort,
            "risk_factor_family": fam,
            "risk_factor_family_label": FAMILY_LABELS.get(fam, fam),
            "semantic_label": r.get("semantic_label", ""),
            "column": r.get("column", ""),
            "model_type": model_type,
            "effect": effect,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "display_effect": display_effect,
            "p_cBAG": p,
            "q_cBAG_within_family": q,
            "stars": p_to_stars(q),
            "n_model": n,
            "is_fdr_0_05": bool(q < 0.05) if np.isfinite(q) else False,
            "plot_label": label,
            "partial_r_cBAG": r.get("partial_r_cBAG", np.nan),
            "delta_auc": r.get("delta_auc", np.nan),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    out["_fam_rank"] = out["risk_factor_family"].map(fam_rank).fillna(999)
    out["_abs_effect"] = out["effect"].abs()
    out["_q_sort"] = pd.to_numeric(out["q_cBAG_within_family"], errors="coerce").fillna(999)
    out["_p_sort"] = pd.to_numeric(out["p_cBAG"], errors="coerce").fillna(999)

    out = out.sort_values(
        ["is_fdr_0_05", "_q_sort", "_p_sort", "_abs_effect", "_fam_rank"],
        ascending=[False, True, True, False, True],
    )

    # HARD CAP: at most max_rows, no exceptions.
    if max_rows and max_rows > 0:
        out = out.head(max_rows).copy()

    out = out.reset_index(drop=True)
    out["y"] = np.arange(len(out))[::-1]
    return out.drop(columns=["_fam_rank", "_abs_effect", "_q_sort", "_p_sort"], errors="ignore")


def draw(df, outdir, prefix, dpi, title):
    outdir.mkdir(parents=True, exist_ok=True)
    source = outdir / f"{prefix}_source_data.csv"
    df.to_csv(source, index=False)

    if df.empty:
        raise RuntimeError("No plottable risk-factor rows.")

    height = max(5.8, 0.38 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, height))
    ax.axvline(0, color="black", lw=1.0, alpha=0.8)
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    for fam, g in df.groupby("risk_factor_family", sort=False):
        color = FAMILY_COLORS.get(str(fam), "#737373")
        xerr = np.vstack([
            g["effect"] - g["ci_low"],
            g["ci_high"] - g["effect"],
        ])
        xerr = np.where(np.isfinite(xerr), xerr, 0)
        ax.errorbar(
            g["effect"], g["y"],
            xerr=xerr,
            fmt="o", ms=5.8, lw=1.25, capsize=2.5,
            color=color, ecolor=color,
            label=FAMILY_LABELS.get(str(fam), str(fam)),
            alpha=0.95,
        )

    labels = []
    for _, r in df.iterrows():
        q = r.get("q_cBAG_within_family", np.nan)
        qtxt = f"{q:.3g}" if np.isfinite(q) else "NA"
        n = r.get("n_model", np.nan)
        ntxt = str(int(n)) if np.isfinite(n) else "NA"
        star = f" {r['stars']}" if r.get("stars", "") else ""
        labels.append(f"{r['plot_label']}{star}\n{r['display_effect']}, N={ntxt}, q={qtxt}")

    ax.set_yticks(df["y"])
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("Effect of 1 SD higher cBAG: standardized β for continuous targets; log(OR) for binary targets")
    ax.set_title(f"{title} — top {len(df)} rows", fontsize=12, fontweight="bold")

    finite_los = df["ci_low"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_his = df["ci_high"].replace([np.inf, -np.inf], np.nan).dropna()
    xmin = min(float(finite_los.min()) if len(finite_los) else float(df["effect"].min()), -0.05)
    xmax = max(float(finite_his.max()) if len(finite_his) else float(df["effect"].max()), 0.05)
    pad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)

    handles, labs = ax.get_legend_handles_labels()
    by_lab = dict(zip(labs, handles))
    ax.legend(by_lab.values(), by_lab.keys(), loc="lower right", fontsize=8, frameon=True, title="Risk-factor family")

    fig.text(
        0.01, 0.01,
        "* q<0.05, ** q<0.01, *** q<0.001 within cohort × family × model type. Hard-capped to top-ranked rows.",
        fontsize=8,
        ha="left",
        va="bottom",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    png = outdir / f"{prefix}.png"
    pdf = outdir / f"{prefix}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Wrote:", png)
    print("[OK] Wrote:", pdf)
    print("[OK] Wrote:", source)
    print(f"[OK] Rows plotted: {len(df)}")


def main():
    args = parse_args()
    df, src = load_selected(args)
    outdir = args.outdir or (args.risk_dir / "figures")
    print("[INFO] Risk selected table:", src)
    plot_df = prepare(df, args.max_rows)
    draw(plot_df, outdir, args.prefix, args.dpi, args.title)


if __name__ == "__main__":
    main()
