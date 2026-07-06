#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5 longitudinal cBAG–neuroimaging associations.

Longitudinal companion to cross-sectional Figure 4:
within-person change in cBAG vs within-person change in neuroimaging/graph measures.

Main figure:
    imaging_only model, ADNI + HABS

Supplementary:
    all five feature sets, same layout

Inputs:
    subject_level_validation_input_enriched_for_Figure4.csv
    from validation_figures_full_cohort directories

Outputs:
    PNG/PDF figures
    source_data.csv
    stats.csv
    input_availability.csv
    manifest.csv/.xlsx
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIG
# =============================================================================

WORK = Path("/mnt/newStor/paros/paros_WORK")
RESULTS_ROOT = WORK / "ines" / "results"

OUTDIR = (
    RESULTS_ROOT
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Figure5_longitudinal_cBAG_neuroimaging_associations"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

COHORT_RESULT_DIRS = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
}

COHORTS = ["ADNI", "HABS"]

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

FEATURE_SET_LABELS = {
    "imaging_only": "Imaging only",
    "imaging_demographics": "Imaging + demographics",
    "imaging_biomarkers": "Imaging + biomarkers",
    "full": "Full model",
    "full_no_cardiovascular": "Full model without cardiovascular variables",
}

SUPP_LABELS = {
    "imaging_only": "S5A",
    "imaging_demographics": "S5B",
    "imaging_biomarkers": "S5C",
    "full": "S5D",
    "full_no_cardiovascular": "S5E",
}

# Pairing mode:
#   "first_last"  -> one pair per subject, earliest to latest
#   "consecutive" -> adjacent visit pairs per subject
PAIR_MODE = "first_last"

# Same biological variables used in Figure 4, but expressed as within-person deltas.
METRICS = [
    ("Hc_volume_pct_brain", "Δ Hippocampal volume / brain"),
    ("Hc_FA", "Δ Hippocampal FA"),
    ("Hc_clustering_coeff", "Δ Hippocampal clustering"),
    ("Hc_path_length", "Δ Hippocampal path length"),
    ("Total_Brain_volume", "Δ Total brain volume"),
    ("Total_Brain_FA", "Δ Total brain FA"),
    ("Total_graph_clustering_coeff", "Δ Global clustering"),
    ("Total_graph_path_length", "Δ Global path length"),
]

# Compact Figure 4-style cross-cohort longitudinal summary.
# These six variables match the cross-cohort aggregate view used for Figure 4.
CROSS_COHORT_METRICS = [
    ("Hc_volume_pct_brain", "Δ Hippocampal volume"),
    ("Hc_FA", "Δ Hippocampal FA"),
    ("Total_Brain_volume", "Δ Total brain volume"),
    ("Total_Brain_FA", "Δ Total brain FA"),
    ("Total_graph_clustering_coeff", "Δ Graph clustering coefficient"),
    ("Total_graph_path_length", "Δ Graph path length"),
]

COHORT_COLORS = {
    "ADNI": "#1f77b4",
    "HABS": "#2ca02c",
}

COHORT_MARKERS = {
    "ADNI": "o",
    "HABS": "^",
}

PARTICIPANT_CANDIDATES = [
    "PTID", "subject_id", "Subject", "Subject_ID", "ID", "participant_id",
    "RID", "subject", "subj", "participant",
]

VISIT_CANDIDATES = [
    "Visit", "visit", "VISCODE", "visit_label", "DWI_visit_label", "session", "timepoint",
]

AGE_CANDIDATES = [
    "age", "Age", "AGE", "age_true", "chronological_age", "Chronological_Age",
    "ChronologicalAge", "true_age", "y_true",
]

CBAG_CANDIDATES = [
    "cBAG",
    "cBAG_global",
    "bag_global_residualized",
    "BAG_global_residualized",
    "global_bag_residualized",
    "BAG_residualized_global",
    "brain_age_gap_global_residualized",
    "BAG_bias_corrected",
    "bag_bias_corrected",
]

# =============================================================================
# HELPERS
# =============================================================================

def enriched_input_path(cohort: str, feature_set: str) -> Path:
    return (
        RESULTS_ROOT
        / COHORT_RESULT_DIRS[cohort]
        / f"ablation_{feature_set}"
        / "validation_figures_full_cohort"
        / "subject_level_validation_input_enriched_for_Figure4.csv"
    )


def first_existing_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the candidate columns were found: {candidates}")
    return None


def clean_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def strip_visit_suffix(x: str) -> str:
    x = str(x).strip()
    # ADNI/HABS connectome keys are often R####_y# or H####_y#.
    x = re.sub(r"_y\d+$", "", x, flags=re.I)
    x = re.sub(r"_m\d+$", "", x, flags=re.I)
    x = re.sub(r"-y\d+$", "", x, flags=re.I)
    x = re.sub(r"-m\d+$", "", x, flags=re.I)
    return x


def extract_visit_order_from_text(x: str) -> float:
    x = str(x).strip().lower()
    patterns = [
        r"_y(\d+(?:\.\d+)?)$",
        r"-y(\d+(?:\.\d+)?)$",
        r"^y(\d+(?:\.\d+)?)$",
        r"_m(\d+(?:\.\d+)?)$",
        r"-m(\d+(?:\.\d+)?)$",
        r"^m(\d+(?:\.\d+)?)$",
        r"month\s*(\d+(?:\.\d+)?)",
        r"visit\s*(\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, x)
        if m:
            val = float(m.group(1))
            if "m" in pat or "month" in pat:
                return val / 12.0
            return val
    return np.nan


def derive_participant_id(df: pd.DataFrame, cohort: str) -> pd.Series:
    # Prefer explicit participant columns with repeated IDs.
    for c in PARTICIPANT_CANDIDATES:
        if c in df.columns:
            s = clean_string_series(df[c])
            if s.nunique(dropna=True) < len(s):
                return s.map(strip_visit_suffix)

    # Fallback: derive from visit-specific IDs.
    for c in ["graph_id", "Subject_ID", "subject_source", "match_id", "subject_match"]:
        if c in df.columns:
            return clean_string_series(df[c]).map(strip_visit_suffix)

    raise KeyError(
        f"Could not derive participant ID for {cohort}. Available columns include: "
        f"{list(df.columns)[:40]}"
    )


def derive_visit_order(df: pd.DataFrame) -> pd.Series:
    # Prefer chronological age if present; this gives real time between visits.
    age_col = first_existing_column(df, AGE_CANDIDATES, required=False)
    if age_col is not None:
        age = pd.to_numeric(df[age_col], errors="coerce")
        if age.notna().sum() > 0:
            return age

    # Then use visit labels or visit-specific graph IDs.
    for c in ["graph_id", "Subject_ID", "subject_source", "subject_match"] + VISIT_CANDIDATES:
        if c in df.columns:
            order = clean_string_series(df[c]).map(extract_visit_order_from_text)
            if pd.Series(order).notna().any():
                return pd.Series(order, index=df.index)

    raise KeyError("Could not derive visit order from age, graph_id, Subject_ID, or visit labels.")


def derive_cbag(df: pd.DataFrame) -> pd.Series:
    c = first_existing_column(df, CBAG_CANDIDATES, required=False)
    if c is not None:
        return pd.to_numeric(df[c], errors="coerce")

    # Flexible fallback: residualized BAG columns.
    for col in df.columns:
        low = col.lower()
        if "bag" in low and ("resid" in low or "correct" in low or "cbag" in low):
            return pd.to_numeric(df[col], errors="coerce")

    # Last fallback: raw predicted age minus age, if columns exist.
    pred_cols = [c for c in df.columns if c.lower() in {"y_pred", "pred_age", "predicted_age", "brain_age"}]
    age_col = first_existing_column(df, AGE_CANDIDATES, required=False)
    if pred_cols and age_col is not None:
        return pd.to_numeric(df[pred_cols[0]], errors="coerce") - pd.to_numeric(df[age_col], errors="coerce")

    raise KeyError(
        "Could not find cBAG/global residualized BAG column. "
        f"Tried: {CBAG_CANDIDATES}"
    )


def build_longitudinal_pairs(df: pd.DataFrame, cohort: str, feature_set: str) -> pd.DataFrame:
    df = df.copy()

    df["_participant_id"] = derive_participant_id(df, cohort)
    df["_visit_order"] = derive_visit_order(df)
    df["_cBAG"] = derive_cbag(df)

    keep_cols = ["_participant_id", "_visit_order", "_cBAG"]
    for metric_name, _ in METRICS:
        if metric_name in df.columns:
            keep_cols.append(metric_name)

    df = df[keep_cols].copy()
    df = df.dropna(subset=["_participant_id", "_visit_order", "_cBAG"])
    df = df.sort_values(["_participant_id", "_visit_order"])

    rows: list[dict] = []

    for pid, g in df.groupby("_participant_id", sort=False):
        g = g.sort_values("_visit_order").reset_index(drop=True)
        if len(g) < 2:
            continue

        if PAIR_MODE == "first_last":
            pair_indices = [(0, len(g) - 1)]
        elif PAIR_MODE == "consecutive":
            pair_indices = [(i, i + 1) for i in range(len(g) - 1)]
        else:
            raise ValueError(f"Unsupported PAIR_MODE: {PAIR_MODE}")

        for i, j in pair_indices:
            earlier = g.iloc[i]
            later = g.iloc[j]

            delta_time = later["_visit_order"] - earlier["_visit_order"]
            if pd.isna(delta_time) or delta_time <= 0:
                continue

            row = {
                "cohort": cohort,
                "feature_set": feature_set,
                "participant_id": pid,
                "earlier_visit_order": earlier["_visit_order"],
                "later_visit_order": later["_visit_order"],
                "delta_visit_order": delta_time,
                "delta_cBAG": later["_cBAG"] - earlier["_cBAG"],
                "annualized_delta_cBAG": (later["_cBAG"] - earlier["_cBAG"]) / delta_time,
            }

            for metric_name, _ in METRICS:
                delta_col = f"delta_{metric_name}"
                annual_col = f"annualized_delta_{metric_name}"
                if metric_name in g.columns and pd.notna(earlier[metric_name]) and pd.notna(later[metric_name]):
                    delta_val = later[metric_name] - earlier[metric_name]
                    row[delta_col] = delta_val
                    row[annual_col] = delta_val / delta_time
                else:
                    row[delta_col] = np.nan
                    row[annual_col] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def association_stats(df: pd.DataFrame, xcol: str, ycol: str = "delta_cBAG") -> dict:
    sub = df[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(sub)

    if n < 3 or sub[xcol].nunique() < 2 or sub[ycol].nunique() < 2:
        return {"n": n, "slope": np.nan, "intercept": np.nan, "r": np.nan, "p": np.nan}

    fit = linregress(sub[xcol], sub[ycol])
    return {"n": n, "slope": fit.slope, "intercept": fit.intercept, "r": fit.rvalue, "p": fit.pvalue}


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

# =============================================================================
# FIGURE BUILDERS
# =============================================================================

def build_scatter_figure(
    pairs_df: pd.DataFrame,
    feature_set: str,
    out_prefix: str,
    title: str,
    use_annualized: bool = False,
) -> dict:
    nrows = len(COHORTS)
    ncols = len(METRICS)

    ycol = "annualized_delta_cBAG" if use_annualized else "delta_cBAG"
    y_label = "Annualized ΔcBAG" if use_annualized else "ΔcBAG"
    x_prefix = "annualized_delta_" if use_annualized else "delta_"

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.7 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=16, y=1.02)

    stats_rows: list[dict] = []
    source_rows: list[dict] = []
    availability_rows: list[dict] = []

    for r_i, cohort in enumerate(COHORTS):
        cohort_df = pairs_df[pairs_df["cohort"] == cohort].copy()

        for c_i, (metric_col, metric_label) in enumerate(METRICS):
            ax = axes[r_i, c_i]
            xcol = f"{x_prefix}{metric_col}"

            sub = cohort_df[[xcol, ycol, "participant_id", "delta_visit_order"]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            availability_rows.append({
                "feature_set": feature_set,
                "cohort": cohort,
                "metric": metric_col,
                "metric_label": metric_label,
                "n_pairs": len(sub),
                "pair_mode": PAIR_MODE,
                "use_annualized": use_annualized,
            })

            for _, row in sub.iterrows():
                source_rows.append({
                    "feature_set": feature_set,
                    "cohort": cohort,
                    "metric": metric_col,
                    "metric_label": metric_label,
                    "participant_id": row["participant_id"],
                    "delta_visit_order": row["delta_visit_order"],
                    "x_delta_metric": row[xcol],
                    "y_delta_cBAG": row[ycol],
                    "pair_mode": PAIR_MODE,
                    "use_annualized": use_annualized,
                })

            ax.scatter(sub[xcol], sub[ycol], s=18, alpha=0.70)

            stat = association_stats(cohort_df, xcol=xcol, ycol=ycol)
            stat.update({
                "feature_set": feature_set,
                "cohort": cohort,
                "metric": metric_col,
                "metric_label": metric_label,
                "pair_mode": PAIR_MODE,
                "use_annualized": use_annualized,
                "x_column": xcol,
                "y_column": ycol,
            })
            stats_rows.append(stat)

            if len(sub) >= 3 and pd.notna(stat["slope"]):
                xx = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
                yy = stat["slope"] * xx + stat["intercept"]
                ax.plot(xx, yy, linewidth=1.5)

            if r_i == 0:
                ax.set_title(metric_label, fontsize=10)

            if c_i == 0:
                ax.set_ylabel(f"{cohort}\n{y_label}")
            else:
                ax.set_ylabel("")

            ax.set_xlabel(metric_label, fontsize=9)

            txt = f"n={stat['n']}"
            if pd.notna(stat["r"]):
                txt += f"\nr={stat['r']:.2f}{p_to_stars(stat['p'])}\np={stat['p']:.3g}"
            ax.text(
                0.04,
                0.96,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.15),
            )

    plt.tight_layout()

    png = OUTDIR / f"{out_prefix}.png"
    pdf = OUTDIR / f"{out_prefix}.pdf"
    stats_csv = OUTDIR / f"{out_prefix}_stats.csv"
    source_csv = OUTDIR / f"{out_prefix}_source_data.csv"
    avail_csv = OUTDIR / f"{out_prefix}_input_availability.csv"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
    pd.DataFrame(source_rows).to_csv(source_csv, index=False)
    pd.DataFrame(availability_rows).to_csv(avail_csv, index=False)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {stats_csv}")
    print(f"Saved: {source_csv}")
    print(f"Saved: {avail_csv}")

    return {
        "figure_prefix": out_prefix,
        "png": str(png),
        "pdf": str(pdf),
        "stats_csv": str(stats_csv),
        "source_csv": str(source_csv),
        "availability_csv": str(avail_csv),
    }


def build_heatmap(stats_df: pd.DataFrame, out_prefix: str, title: str) -> dict:
    heat = stats_df.pivot(index="cohort", columns="metric_label", values="r").reindex(index=COHORTS)

    fig, ax = plt.subplots(figsize=(15.5, 3.4))
    im = ax.imshow(heat.values, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title(title)

    p_lookup = {}
    if "p" in stats_df.columns:
        for _, row in stats_df.iterrows():
            p_lookup[(row["cohort"], row["metric_label"])] = row["p"]

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            cohort = heat.index[i]
            metric_label = heat.columns[j]
            stars = p_to_stars(p_lookup.get((cohort, metric_label), np.nan))
            txt = "" if pd.isna(val) else f"{val:.2f}{stars}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    plt.tight_layout()

    png = OUTDIR / f"{out_prefix}.png"
    pdf = OUTDIR / f"{out_prefix}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap: {png}")
    print(f"Saved heatmap: {pdf}")

    return {"heatmap_png": str(png), "heatmap_pdf": str(pdf)}


def fisher_r_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher-z approximate 95% CI for Pearson r."""
    if pd.isna(r) or n <= 3 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = 1.96
    lo, hi = z - zcrit * se, z + zcrit * se
    return tuple(np.tanh([lo, hi]))


def build_cross_cohort_aggregate(
    pairs_df: pd.DataFrame,
    feature_set: str,
    out_prefix: str,
    title: str,
    use_annualized: bool = False,
) -> dict:
    """
    Figure 4-style pooled cross-cohort summary for longitudinal data.
    Points are colored by cohort; one pooled regression line is fit per panel.
    """
    ycol = "annualized_delta_cBAG" if use_annualized else "delta_cBAG"
    x_prefix = "annualized_delta_" if use_annualized else "delta_"
    y_label = "Annualized ΔcBAG" if use_annualized else "ΔcBAG"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    stats_rows: list[dict] = []
    source_rows: list[dict] = []

    for ax, (metric_col, metric_label) in zip(axes, CROSS_COHORT_METRICS):
        xcol = f"{x_prefix}{metric_col}"

        if xcol not in pairs_df.columns:
            ax.text(
                0.5, 0.5, f"Missing column\n{xcol}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9
            )
            ax.set_title(metric_label)
            stats_rows.append({
                "feature_set": feature_set,
                "metric": metric_col,
                "metric_label": metric_label,
                "cohort": "POOLED",
                "n": 0,
                "r": np.nan,
                "p": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "slope": np.nan,
                "intercept": np.nan,
                "pair_mode": PAIR_MODE,
                "use_annualized": use_annualized,
                "status": "missing_column",
            })
            continue

        sub = pairs_df[["cohort", "participant_id", "delta_visit_order", xcol, ycol]].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[xcol, ycol])

        for _, row in sub.iterrows():
            source_rows.append({
                "feature_set": feature_set,
                "metric": metric_col,
                "metric_label": metric_label,
                "cohort": row["cohort"],
                "participant_id": row["participant_id"],
                "delta_visit_order": row["delta_visit_order"],
                "x_delta_metric": row[xcol],
                "y_delta_cBAG": row[ycol],
                "pair_mode": PAIR_MODE,
                "use_annualized": use_annualized,
            })

        if len(sub) < 3 or sub[xcol].nunique() < 2 or sub[ycol].nunique() < 2:
            ax.text(
                0.5, 0.5,
                f"Insufficient data\nn={len(sub)}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9
            )
            ax.set_title(metric_label)
            ax.set_xlabel(metric_label)
            ax.set_ylabel(y_label)
            stats_rows.append({
                "feature_set": feature_set,
                "metric": metric_col,
                "metric_label": metric_label,
                "cohort": "POOLED",
                "n": len(sub),
                "r": np.nan,
                "p": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "slope": np.nan,
                "intercept": np.nan,
                "pair_mode": PAIR_MODE,
                "use_annualized": use_annualized,
                "status": "insufficient_data",
            })
            continue

        for cohort in COHORTS:
            csub = sub[sub["cohort"] == cohort]
            if csub.empty:
                continue
            ax.scatter(
                csub[xcol],
                csub[ycol],
                s=18,
                alpha=0.72,
                color=COHORT_COLORS.get(cohort, None),
                marker=COHORT_MARKERS.get(cohort, "o"),
                label=cohort,
                edgecolors="white",
                linewidths=0.3,
            )

            cstat = association_stats(csub, xcol=xcol, ycol=ycol)
            stats_rows.append({
                "feature_set": feature_set,
                "metric": metric_col,
                "metric_label": metric_label,
                "cohort": cohort,
                "n": cstat["n"],
                "r": cstat["r"],
                "p": cstat["p"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "slope": cstat["slope"],
                "intercept": cstat["intercept"],
                "pair_mode": PAIR_MODE,
                "use_annualized": use_annualized,
                "status": "cohort_specific",
            })

        pooled = association_stats(sub, xcol=xcol, ycol=ycol)
        if pd.notna(pooled["slope"]):
            xx = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
            yy = pooled["slope"] * xx + pooled["intercept"]
            ax.plot(xx, yy, color="black", linewidth=1.5)

        ci_low, ci_high = fisher_r_ci(pooled["r"], pooled["n"])

        txt = f"pooled r={pooled['r']:.2f}"
        if pd.notna(ci_low):
            txt += f" [{ci_low:.2f}, {ci_high:.2f}]"
        txt += f"\np={pooled['p']:.3g}\nn={pooled['n']}"

        ax.text(
            0.04,
            0.94,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.15),
        )

        ax.set_title(metric_label, fontsize=11)
        ax.set_xlabel(metric_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(alpha=0.20)
        ax.tick_params(axis="both", labelsize=9)

        stats_rows.append({
            "feature_set": feature_set,
            "metric": metric_col,
            "metric_label": metric_label,
            "cohort": "POOLED",
            "n": pooled["n"],
            "r": pooled["r"],
            "p": pooled["p"],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "slope": pooled["slope"],
            "intercept": pooled["intercept"],
            "pair_mode": PAIR_MODE,
            "use_annualized": use_annualized,
            "status": "pooled",
        })

    handles = []
    labels = []
    for cohort in COHORTS:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker=COHORT_MARKERS.get(cohort, "o"),
                color="none",
                markerfacecolor=COHORT_COLORS.get(cohort, "gray"),
                markeredgecolor="white",
                markersize=7,
                linestyle="",
            )
        )
        labels.append(cohort)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(COHORTS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(title, fontsize=16, y=0.985)
    fig.tight_layout(rect=[0.02, 0.06, 1.0, 0.94])

    png = OUTDIR / f"{out_prefix}.png"
    pdf = OUTDIR / f"{out_prefix}.pdf"
    stats_csv = OUTDIR / f"{out_prefix}_stats.csv"
    source_csv = OUTDIR / f"{out_prefix}_source_data.csv"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
    pd.DataFrame(source_rows).to_csv(source_csv, index=False)

    print(f"Saved cross-cohort figure: {png}")
    print(f"Saved cross-cohort figure: {pdf}")
    print(f"Saved cross-cohort stats: {stats_csv}")
    print(f"Saved cross-cohort source data: {source_csv}")

    return {
        "cross_cohort_png": str(png),
        "cross_cohort_pdf": str(pdf),
        "cross_cohort_stats_csv": str(stats_csv),
        "cross_cohort_source_csv": str(source_csv),
    }


def build_feature_set_aggregate_heatmap(
    all_stats_df: pd.DataFrame,
    out_prefix: str,
    title: str,
) -> dict:
    """
    Aggregate Supplementary Figure S5 heatmap.

    Rows are cohort × feature-set combinations.
    Columns are longitudinal neuroimaging/graph metrics.
    Cell values are Pearson r for ΔcBAG vs Δmetric.
    """
    if all_stats_df.empty:
        return {}

    stats = all_stats_df.copy()
    stats = stats[stats["cohort"].isin(COHORTS)].copy()
    stats["feature_set_label"] = stats["feature_set"].map(FEATURE_SET_LABELS).fillna(stats["feature_set"])
    stats["row_label"] = stats["cohort"] + " | " + stats["feature_set_label"]

    metric_order = [label for _, label in METRICS]
    row_order = []
    for cohort in COHORTS:
        for fs in FEATURE_SETS:
            row_order.append(f"{cohort} | {FEATURE_SET_LABELS[fs]}")

    pivot_r = stats.pivot_table(index="row_label", columns="metric_label", values="r", aggfunc="first")
    pivot_p = stats.pivot_table(index="row_label", columns="metric_label", values="p", aggfunc="first")
    pivot_n = stats.pivot_table(index="row_label", columns="metric_label", values="n", aggfunc="first")

    pivot_r = pivot_r.reindex(index=row_order, columns=metric_order)
    pivot_p = pivot_p.reindex(index=row_order, columns=metric_order)
    pivot_n = pivot_n.reindex(index=row_order, columns=metric_order)

    fig_width = max(14.0, 1.6 * len(metric_order))
    fig_height = max(6.0, 0.48 * len(row_order) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(pivot_r.values, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels(metric_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)

    for i in range(pivot_r.shape[0]):
        for j in range(pivot_r.shape[1]):
            r = pivot_r.iloc[i, j]
            p = pivot_p.iloc[i, j]
            n = pivot_n.iloc[i, j]
            if pd.isna(r):
                txt = "NA"
            else:
                txt = f"{r:.2f}{p_to_stars(p)}\nn={int(n) if pd.notna(n) else 0}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    ax.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Pearson r", rotation=90)

    plt.tight_layout()

    png = OUTDIR / f"{out_prefix}.png"
    pdf = OUTDIR / f"{out_prefix}.pdf"
    source_csv = OUTDIR / f"{out_prefix}_source_data.csv"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    stats.to_csv(source_csv, index=False)

    print(f"Saved aggregate S5 heatmap: {png}")
    print(f"Saved aggregate S5 heatmap: {pdf}")
    print(f"Saved aggregate S5 heatmap source data: {source_csv}")

    return {
        "aggregate_heatmap_png": str(png),
        "aggregate_heatmap_pdf": str(pdf),
        "aggregate_heatmap_source_csv": str(source_csv),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("FIGURE 5 LONGITUDINAL cBAG–NEUROIMAGING ASSOCIATIONS")
    print("=" * 100)
    print(f"RESULTS_ROOT: {RESULTS_ROOT}")
    print(f"OUTDIR: {OUTDIR}")
    print(f"PAIR_MODE: {PAIR_MODE}")
    print(f"COHORTS: {COHORTS}")
    print(f"FEATURE_SETS: {FEATURE_SETS}")

    manifest_rows: list[dict] = []
    all_pair_counts: list[dict] = []
    all_pairs_frames: list[pd.DataFrame] = []
    all_stats_frames: list[pd.DataFrame] = []

    for feature_set in FEATURE_SETS:
        pair_frames: list[pd.DataFrame] = []

        for cohort in COHORTS:
            path = enriched_input_path(cohort, feature_set)
            print(f"\nReading: {path}")
            if not path.exists():
                print(f"[MISSING] {path}")
                all_pair_counts.append({
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "n_pairs": 0,
                    "pair_mode": PAIR_MODE,
                    "source_path": str(path),
                    "status": "missing_input",
                })
                continue

            df = pd.read_csv(path, low_memory=False)
            try:
                pairs = build_longitudinal_pairs(df, cohort=cohort, feature_set=feature_set)
            except Exception as exc:
                print(f"[ERROR] {cohort} | {feature_set}: {exc}")
                all_pair_counts.append({
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "n_pairs": 0,
                    "pair_mode": PAIR_MODE,
                    "source_path": str(path),
                    "status": f"error: {exc}",
                })
                continue

            print(f"{cohort} | {feature_set} | rows={len(df)} | longitudinal_pairs={len(pairs)}")

            all_pair_counts.append({
                "cohort": cohort,
                "feature_set": feature_set,
                "n_rows_input": len(df),
                "n_pairs": len(pairs),
                "pair_mode": PAIR_MODE,
                "source_path": str(path),
                "status": "ok",
            })

            if not pairs.empty:
                pair_frames.append(pairs)
                all_pairs_frames.append(pairs)

        if not pair_frames:
            print(f"[SKIP] No longitudinal pairs for feature set: {feature_set}")
            continue

        pairs_df = pd.concat(pair_frames, ignore_index=True)

        if feature_set == "imaging_only":
            fig_prefix = "Figure5_delta_cBAG_neuroimaging_associations_imaging_only"
            fig_title = "Figure 5. Longitudinal ΔcBAG versus Δneuroimaging/graph measures (imaging-only model)"
        else:
            supp_label = SUPP_LABELS[feature_set]
            fig_prefix = f"SupplementaryFigure{supp_label}_delta_cBAG_neuroimaging_associations_{feature_set}"
            fig_title = (
                "Supplementary Figure "
                f"{supp_label}. Longitudinal ΔcBAG versus Δneuroimaging/graph measures "
                f"({FEATURE_SET_LABELS[feature_set]})"
            )

        fig_info = build_scatter_figure(
            pairs_df=pairs_df,
            feature_set=feature_set,
            out_prefix=fig_prefix,
            title=fig_title,
            use_annualized=False,
        )

        stats_df = pd.read_csv(fig_info["stats_csv"])
        all_stats_frames.append(stats_df)

        heatmap_prefix = f"{fig_prefix}_heatmap"
        heatmap_info = build_heatmap(
            stats_df,
            out_prefix=heatmap_prefix,
            title=f"{fig_title} — Pearson r heatmap",
        )

        cross_info = {}
        if feature_set == "imaging_only":
            cross_prefix = "SupplementaryFigureS5F_longitudinal_cross_cohort_summary_imaging_only"
            cross_title = (
                "Supplementary Figure S5F. Cross-cohort longitudinal summary of "
                "ΔcBAG versus Δneuroimaging/graph measures (imaging-only model)"
            )
            cross_info = build_cross_cohort_aggregate(
                pairs_df=pairs_df,
                feature_set=feature_set,
                out_prefix=cross_prefix,
                title=cross_title,
                use_annualized=False,
            )

        manifest_rows.append({
            "feature_set": feature_set,
            "feature_set_label": FEATURE_SET_LABELS[feature_set],
            "pair_mode": PAIR_MODE,
            "use_annualized": False,
            **fig_info,
            **heatmap_info,
            **cross_info,
        })

    # Aggregate Supplementary Figure S5 heatmap across all feature-set models.
    if all_stats_frames:
        all_stats_df = pd.concat(all_stats_frames, ignore_index=True)
        agg_info = build_feature_set_aggregate_heatmap(
            all_stats_df=all_stats_df,
            out_prefix="SupplementaryFigureS5G_longitudinal_feature_set_aggregate_heatmap",
            title=(
                "Supplementary Figure S5G. Longitudinal ΔcBAG–Δneuroimaging "
                "associations across feature-set models"
            ),
        )
        if agg_info:
            manifest_rows.append({
                "feature_set": "all_feature_sets",
                "feature_set_label": "All feature sets",
                "pair_mode": PAIR_MODE,
                "use_annualized": False,
                "figure_prefix": "SupplementaryFigureS5G_longitudinal_feature_set_aggregate_heatmap",
                **agg_info,
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = OUTDIR / "Figure5_longitudinal_manifest.csv"
    manifest_xlsx = OUTDIR / "Figure5_longitudinal_manifest.xlsx"
    manifest_df.to_csv(manifest_csv, index=False)
    manifest_df.to_excel(manifest_xlsx, index=False)

    pair_count_df = pd.DataFrame(all_pair_counts)
    pair_count_csv = OUTDIR / "Figure5_longitudinal_pair_counts.csv"
    pair_count_xlsx = OUTDIR / "Figure5_longitudinal_pair_counts.xlsx"
    pair_count_df.to_csv(pair_count_csv, index=False)
    pair_count_df.to_excel(pair_count_xlsx, index=False)

    if all_pairs_frames:
        all_pairs_df = pd.concat(all_pairs_frames, ignore_index=True)
        all_pairs_csv = OUTDIR / "Figure5_longitudinal_all_pairs_source_table.csv"
        all_pairs_df.to_csv(all_pairs_csv, index=False)
        print(f"Saved all-pairs source table: {all_pairs_csv}")

    print("\nSaved manifest:")
    print(manifest_csv)
    print(manifest_xlsx)
    print("\nSaved pair counts:")
    print(pair_count_csv)
    print(pair_count_xlsx)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        raise
