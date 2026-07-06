#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-cohort transfer validation grid for brain-age GNN models
==============================================================

DEFAULT
-------
Start simple with imaging_only only:

    python transfer_validation_train_test_grid_imaging_first.py

This builds a train-cohort × test-cohort grid:

    train ADNI      -> test ADNI, ADRC, HABS, AD_DECODE
    train ADRC      -> test ADNI, ADRC, HABS, AD_DECODE
    train HABS      -> test ADNI, ADRC, HABS, AD_DECODE
    train AD_DECODE -> test ADNI, ADRC, HABS, AD_DECODE

For diagonal cells, it uses OOF-global predictions from the original training
rather than final-model-on-training-cohort predictions.

For off-diagonal cells, it loads the final train-cohort model and applies it to
the test cohort's all-subject graph set.

Default feature set:
    imaging_only

Later, run more:
    python transfer_validation_train_test_grid_imaging_first.py --feature-sets imaging_only,imaging_demographics
    python transfer_validation_train_test_grid_imaging_first.py --feature-sets ALL

Outputs
-------
Default output directory:

    /mnt/newStor/paros/paros_WORK/ines/results/
    BrainAgeTransferValidation_TrainTestGrid_OOFGlobal/

Per feature set:
    <output-root>/<feature_set>/
        transfer_grid_metrics.csv
        transfer_grid_predictions_all.csv
        transfer_grid_compatibility.csv
        heatmap_MAE.png
        heatmap_RMSE.png
        heatmap_R2.png
        heatmap_r.png
        heatmap_cBAG_age_slope.png
        per_cell/train_<TRAIN>__test_<TEST>/
            predictions.csv
            metrics.csv
            predicted_age_vs_age.png
            cBAG_vs_age.png

Notes
-----
- This does not train new models. It reuses final models from training052026.py.
- It checks node/global/edge dimensions before inference.
- If dimensions do not match, the cell is marked incompatible and skipped.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_mean_pool
except Exception as e:
    raise ImportError(
        "This script requires torch_geometric. Run it in the same environment used for training052026.py."
    ) from e


# =============================================================================
# Defaults matching final training/validation structure
# =============================================================================
DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_RESULTS_ROOT = os.path.join(DEFAULT_WORK, "ines/results")
DEFAULT_OUTPUT_ROOT = os.path.join(
    DEFAULT_RESULTS_ROOT,
    "BrainAgeTransferValidation_TrainTestGrid_OOFGlobal",
)

DEFAULT_COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]
DEFAULT_FEATURE_SETS = ["imaging_only"]

ALL_FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

EXPERIMENT_TAG = "stratified_groupcv_targetnorm_bagbiascorr_oofglobal"

COHORT_CONFIG = {
    "ADNI": {
        "prefix": "adni",
        "file_prefix": "adni",
        "display": "ADNI",
    },
    "ADRC": {
        "prefix": "adrc",
        "file_prefix": "adrc",
        "display": "ADRC",
    },
    "HABS": {
        "prefix": "habs",
        "file_prefix": "habs",
        "display": "HABS",
    },
    "AD_DECODE": {
        "prefix": "addecode",
        "file_prefix": "ad_decode",
        "display": "AD-DECODE",
    },
}

BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build train-cohort x test-cohort transfer validation grid."
    )
    p.add_argument("--work", default=DEFAULT_WORK)
    p.add_argument("--results-root", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--cohorts", default=",".join(DEFAULT_COHORTS))
    p.add_argument(
        "--feature-sets",
        default=",".join(DEFAULT_FEATURE_SETS),
        help="Comma-separated feature sets, or ALL. Default: imaging_only",
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available.",
    )
    p.add_argument(
        "--no-diagonal-oof",
        action="store_true",
        help="Skip diagonal OOF cells instead of filling them from OOF-global predictions.",
    )
    return p.parse_args()


# =============================================================================
# Model architecture copied from training052026.py
# =============================================================================
class GNNBrainAge(nn.Module):
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim=64, dropout=0.2, edge_dim=1):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)
        self.conv3 = GINEConv(nn3, edge_dim=edge_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        fusion_in = hidden_dim + global_feat_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))

        gnn_emb = global_mean_pool(x, batch)

        if hasattr(data, "global_features") and data.global_features is not None:
            gf = data.global_features.float()
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
        else:
            gf = torch.zeros((gnn_emb.shape[0], 0), device=gnn_emb.device)

        fused = torch.cat([gnn_emb, gf], dim=1)
        return self.regressor(fused).squeeze(-1)


# =============================================================================
# Paths
# =============================================================================
def ensure_dir(path: str | Path) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def graph_cohort_dirname(cohort: str) -> str:
    return "AD_DECODE" if cohort == "AD_DECODE" else cohort


def prediction_dir_stem(cohort: str) -> str:
    return {
        "ADNI": "BrainAgePredictionADNI",
        "ADRC": "BrainAgePredictionADRC",
        "HABS": "BrainAgePredictionHABS",
        "AD_DECODE": "BrainAgePredictionADDECODE",
    }[cohort]


def get_ablation_paths(cohort: str, feature_set: str, results_root: str) -> Dict[str, str]:
    c = COHORT_CONFIG[cohort]
    out_dir = os.path.join(results_root, f"{prediction_dir_stem(cohort)}_{EXPERIMENT_TAG}")
    ablation_dir = os.path.join(out_dir, f"ablation_{feature_set}")
    prefix = f"{c['prefix']}_{feature_set}"
    graph_dir = os.path.join(results_root, "harmonized", graph_cohort_dirname(cohort), "graphs", feature_set)

    return {
        "ablation_dir": ablation_dir,
        "prefix": prefix,
        "model_raw": os.path.join(ablation_dir, f"brainage_{prefix}_prediction_model.pt"),
        "model_bc": os.path.join(ablation_dir, f"brainage_{prefix}_prediction_BIAS_CORRECTED_model.pt"),
        "oof_csv": os.path.join(ablation_dir, f"{prefix}_cv_oof_predictions.csv"),
        "oof_xlsx": os.path.join(ablation_dir, f"{prefix}_cv_oof_predictions.xlsx"),
        "cv_summary_csv": os.path.join(ablation_dir, f"{prefix}_cv_summary_metrics.csv"),
        "final_summary_csv": os.path.join(ablation_dir, f"{prefix}_final_model_summary.csv"),
        "graph_path_all": os.path.join(graph_dir, f"graph_data_list_{c['file_prefix']}_all.pt"),
        "metadata_path_all": os.path.join(graph_dir, f"{c['file_prefix']}_metadata_all_aligned.csv"),
        "graph_path_train": os.path.join(graph_dir, f"graph_data_list_{c['file_prefix']}.pt"),
        "metadata_path_train": os.path.join(graph_dir, f"{c['file_prefix']}_metadata_aligned.csv"),
    }


def find_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# =============================================================================
# Tensor/graph helpers
# =============================================================================
def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_graph_identifier(g, idx: int) -> str:
    candidates = [
        "graph_id", "connectome_key", "connectome_id", "subject_id",
        "Subject_ID", "match_id", "ptid", "PTID", "runno", "RID", "ID", "name",
    ]
    for attr in candidates:
        if hasattr(g, attr):
            val = getattr(g, attr)
            if val is not None:
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                return str(val)
    return f"graph_{idx:05d}"


def sanitize_graph(g):
    if hasattr(g, "x") and g.x is not None:
        g.x = g.x.float()
    if hasattr(g, "edge_attr") and g.edge_attr is not None:
        if g.edge_attr.dim() == 1:
            g.edge_attr = g.edge_attr.view(-1, 1)
        g.edge_attr = g.edge_attr.float()
    if hasattr(g, "global_features") and g.global_features is not None:
        gf = g.global_features.float()
        if gf.dim() == 1:
            gf = gf.unsqueeze(0)
        g.global_features = gf
    else:
        g.global_features = torch.zeros((1, 0), dtype=torch.float32)
    if hasattr(g, "y") and g.y is not None:
        g.y = g.y.float().view(-1)
    return g


def load_graphs(path: str) -> List:
    graphs = safe_torch_load(path, map_location="cpu")
    graphs = [sanitize_graph(g) for g in graphs]
    return graphs


def graph_dims(graphs: List) -> Dict[str, int]:
    if not graphs:
        return {"node_feat_dim": np.nan, "global_feat_dim": np.nan, "edge_dim": np.nan}
    g = graphs[0]
    return {
        "node_feat_dim": int(g.x.shape[1]) if hasattr(g, "x") and g.x is not None else 0,
        "global_feat_dim": int(g.global_features.shape[1]) if hasattr(g, "global_features") and g.global_features is not None else 0,
        "edge_dim": int(g.edge_attr.shape[1]) if hasattr(g, "edge_attr") and g.edge_attr is not None else 0,
    }


def inverse_transform_target(y, target_mean, target_std):
    return np.asarray(y, dtype=float) * float(target_std) + float(target_mean)


def apply_bias_correction(age_true, pred_raw, beta, alpha):
    age_true = np.asarray(age_true, dtype=float)
    pred_raw = np.asarray(pred_raw, dtype=float)
    bag = pred_raw - age_true
    expected_bag = float(beta) * age_true + float(alpha)
    cbag = bag - expected_bag
    return age_true + cbag


@torch.no_grad()
def predict_model(model, graphs: List, device, batch_size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        y_true.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
    graph_ids = [get_graph_identifier(g, i) for i, g in enumerate(graphs)]
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), graph_ids


# =============================================================================
# Metrics and plots
# =============================================================================
def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def safe_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.nanstd(x) == 0:
        return np.nan, np.nan
    lr = linregress(x, y)
    return float(lr.slope), float(lr.intercept)


def compute_metrics(y_true, pred, cbag=None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(pred)
    yt, yp = y_true[mask], pred[mask]
    if len(yt) < 3:
        out = {"n": int(len(yt)), "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "r": np.nan}
    else:
        out = {
            "n": int(len(yt)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2": float(r2_score(yt, yp)),
            "r": safe_pearsonr(yt, yp),
        }

    if cbag is None:
        cbag = pred - y_true
    slope, intercept = safe_slope(y_true, cbag)
    out["cBAG_age_r"] = safe_pearsonr(y_true, cbag)
    out["cBAG_age_slope"] = slope
    out["cBAG_age_intercept"] = intercept
    return out


def save_scatter_pred_age(df: pd.DataFrame, out_png: str, title: str):
    if df.empty:
        return
    x = pd.to_numeric(df["age_true"], errors="coerce")
    y = pd.to_numeric(df["pred_bias_corrected_global"], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return

    metrics = compute_metrics(x.values, y.values)

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.scatter(x, y, alpha=0.72, edgecolors="k")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], linestyle=":", label="identity")
    try:
        lr = linregress(x, y)
        xx = np.linspace(lo, hi, 100)
        ax.plot(xx, lr.slope * xx + lr.intercept, linestyle="--", label="fit")
    except Exception:
        pass
    ax.set_xlabel("Chronological age")
    ax.set_ylabel("Predicted age, bias-corrected")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.03, 0.97,
        f"n={metrics['n']}\nMAE={metrics['MAE']:.2f}\nRMSE={metrics['RMSE']:.2f}\nR={metrics['r']:.3f}\nR²={metrics['R2']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_scatter_cbag_age(df: pd.DataFrame, out_png: str, title: str):
    if df.empty:
        return
    x = pd.to_numeric(df["age_true"], errors="coerce")
    y = pd.to_numeric(df["cBAG_global"], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return

    slope, intercept = safe_slope(x.values, y.values)
    r = safe_pearsonr(x.values, y.values)

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.scatter(x, y, alpha=0.72, edgecolors="k")
    ax.axhline(0, linestyle=":", linewidth=1)
    if np.isfinite(slope):
        xx = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xx, slope * xx + intercept, linestyle="--")
    ax.set_xlabel("Chronological age")
    ax.set_ylabel("cBAG")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.03, 0.97,
        f"n={len(x)}\nR(age,cBAG)={r:.3f}\nslope={slope:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_heatmap(metrics_df: pd.DataFrame, metric: str, cohorts: List[str], out_png: str, title: str):
    grid = pd.DataFrame(index=cohorts, columns=cohorts, dtype=float)
    labels = pd.DataFrame(index=cohorts, columns=cohorts, dtype=object)

    for _, row in metrics_df.iterrows():
        tr = row["train_cohort"]
        te = row["test_cohort"]
        val = row.get(metric, np.nan)
        grid.loc[tr, te] = val
        if pd.isna(val):
            labels.loc[tr, te] = "NA"
        else:
            suffix = "OOF" if str(row.get("cell_type", "")) == "diagonal_oof" else "EXT"
            labels.loc[tr, te] = f"{val:.2f}\n{suffix}"

    arr = grid.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(cohorts)))
    ax.set_yticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts, rotation=35, ha="right")
    ax.set_yticklabels(cohorts)
    ax.set_xlabel("Test cohort")
    ax.set_ylabel("Train cohort")
    ax.set_title(title)

    for i in range(len(cohorts)):
        for j in range(len(cohorts)):
            txt = labels.iloc[i, j]
            if txt is not None:
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Loading checkpoints and predictions
# =============================================================================
def load_checkpoint(train_cohort: str, feature_set: str, results_root: str) -> Tuple[Optional[dict], Optional[str]]:
    p = get_ablation_paths(train_cohort, feature_set, results_root)
    ckpt_path = find_existing([p["model_bc"], p["model_raw"]])
    if ckpt_path is None:
        return None, None
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    return ckpt, ckpt_path


def load_model_from_ckpt(ckpt: dict, device) -> GNNBrainAge:
    model = GNNBrainAge(
        node_feat_dim=int(ckpt["node_feat_dim"]),
        global_feat_dim=int(ckpt["global_feat_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 64)),
        dropout=float(ckpt.get("dropout", 0.35)),
        edge_dim=int(ckpt.get("edge_dim", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def get_target_norm(ckpt: dict) -> Tuple[bool, float, float]:
    tn = ckpt.get("target_normalization", {}) or {}
    used = bool(tn.get("used", False))
    mean = float(tn.get("target_mean", 0.0))
    std = float(tn.get("target_std", 1.0))
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return used, mean, std


def get_bias_params(ckpt: dict) -> Tuple[float, float]:
    bc = ckpt.get("bias_correction", {}) or {}
    beta = bc.get("oof_global_bias_beta", None)
    alpha = bc.get("oof_global_bias_alpha", None)

    if beta is None:
        beta = ckpt.get("global_bias_beta", 0.0)
    if alpha is None:
        alpha = ckpt.get("global_bias_alpha", 0.0)

    try:
        beta = float(beta)
    except Exception:
        beta = 0.0
    try:
        alpha = float(alpha)
    except Exception:
        alpha = 0.0
    return beta, alpha


def load_diagonal_oof_predictions(cohort: str, feature_set: str, results_root: str) -> Optional[pd.DataFrame]:
    p = get_ablation_paths(cohort, feature_set, results_root)
    path = find_existing([p["oof_csv"], p["oof_xlsx"]])
    if path is None:
        return None
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    pred_col = None
    for c in ["pred_bias_corrected_oof_global", "Predicted_Age_GlobalCorrected", "pred_bias_corrected_global", "pred_bias_corrected", "pred_raw"]:
        if c in df.columns:
            pred_col = c
            break
    if pred_col is None:
        return None

    age_col = None
    for c in ["age_true", "Real_Age", "Age", "age"]:
        if c in df.columns:
            age_col = c
            break
    if age_col is None:
        return None

    out = pd.DataFrame()
    out["graph_id"] = df["graph_id"].astype(str) if "graph_id" in df.columns else [f"oof_{i:05d}" for i in range(len(df))]
    out["age_true"] = pd.to_numeric(df[age_col], errors="coerce")
    out["pred_raw"] = pd.to_numeric(df["pred_raw"], errors="coerce") if "pred_raw" in df.columns else pd.to_numeric(df[pred_col], errors="coerce")
    out["pred_bias_corrected_global"] = pd.to_numeric(df[pred_col], errors="coerce")

    if "BAG_raw" in df.columns:
        out["BAG_raw"] = pd.to_numeric(df["BAG_raw"], errors="coerce")
    else:
        out["BAG_raw"] = out["pred_raw"] - out["age_true"]

    if "cBAG_oof_global" in df.columns:
        out["cBAG_global"] = pd.to_numeric(df["cBAG_oof_global"], errors="coerce")
    elif "cBAG_global" in df.columns:
        out["cBAG_global"] = pd.to_numeric(df["cBAG_global"], errors="coerce")
    elif "cBAG" in df.columns:
        out["cBAG_global"] = pd.to_numeric(df["cBAG"], errors="coerce")
    else:
        out["cBAG_global"] = out["pred_bias_corrected_global"] - out["age_true"]

    return out


def evaluate_external_cell(
    train_cohort: str,
    test_cohort: str,
    feature_set: str,
    results_root: str,
    outdir: str,
    device,
    batch_size: int,
    dpi: int,
) -> Tuple[Dict, Optional[pd.DataFrame]]:
    cell_dir = ensure_dir(Path(outdir) / "per_cell" / f"train_{train_cohort}__test_{test_cohort}")
    pred_path = os.path.join(cell_dir, "predictions.csv")
    metrics_path = os.path.join(cell_dir, "metrics.csv")

    ckpt, ckpt_path = load_checkpoint(train_cohort, feature_set, results_root)
    if ckpt is None:
        row = {
            "train_cohort": train_cohort,
            "test_cohort": test_cohort,
            "feature_set": feature_set,
            "cell_type": "external",
            "status": "missing_model",
            "reason": "No final model checkpoint found",
        }
        pd.DataFrame([row]).to_csv(metrics_path, index=False)
        return row, None

    test_paths = get_ablation_paths(test_cohort, feature_set, results_root)
    graph_path_all = test_paths["graph_path_all"]
    if not os.path.exists(graph_path_all):
        row = {
            "train_cohort": train_cohort,
            "test_cohort": test_cohort,
            "feature_set": feature_set,
            "cell_type": "external",
            "status": "missing_test_graphs",
            "reason": f"Missing {graph_path_all}",
            "model_path": ckpt_path,
        }
        pd.DataFrame([row]).to_csv(metrics_path, index=False)
        return row, None

    graphs = load_graphs(graph_path_all)
    dims = graph_dims(graphs)

    train_dims = {
        "node_feat_dim": int(ckpt.get("node_feat_dim", -1)),
        "global_feat_dim": int(ckpt.get("global_feat_dim", -1)),
        "edge_dim": int(ckpt.get("edge_dim", -1)),
    }

    compatible = (
        train_dims["node_feat_dim"] == dims["node_feat_dim"]
        and train_dims["global_feat_dim"] == dims["global_feat_dim"]
        and train_dims["edge_dim"] == dims["edge_dim"]
    )

    if not compatible:
        row = {
            "train_cohort": train_cohort,
            "test_cohort": test_cohort,
            "feature_set": feature_set,
            "cell_type": "external",
            "status": "incompatible_dimensions",
            "reason": "Input dimensions do not match",
            "model_path": ckpt_path,
            "test_graph_path": graph_path_all,
            "train_node_dim": train_dims["node_feat_dim"],
            "test_node_dim": dims["node_feat_dim"],
            "train_global_dim": train_dims["global_feat_dim"],
            "test_global_dim": dims["global_feat_dim"],
            "train_edge_dim": train_dims["edge_dim"],
            "test_edge_dim": dims["edge_dim"],
        }
        pd.DataFrame([row]).to_csv(metrics_path, index=False)
        return row, None

    model = load_model_from_ckpt(ckpt, device)
    y_true, pred_model_scale, graph_ids = predict_model(model, graphs, device, batch_size=batch_size)

    used_norm, target_mean, target_std = get_target_norm(ckpt)
    if used_norm:
        pred_raw = inverse_transform_target(pred_model_scale, target_mean, target_std)
    else:
        pred_raw = pred_model_scale

    beta, alpha = get_bias_params(ckpt)
    pred_bc = apply_bias_correction(y_true, pred_raw, beta, alpha)

    pred_df = pd.DataFrame({
        "graph_id": graph_ids,
        "age_true": y_true,
        "pred_raw": pred_raw,
        "pred_bias_corrected_global": pred_bc,
    })
    pred_df["BAG_raw"] = pred_df["pred_raw"] - pred_df["age_true"]
    pred_df["expected_BAG_global_from_train_oof"] = beta * pred_df["age_true"] + alpha
    pred_df["cBAG_global"] = pred_df["pred_bias_corrected_global"] - pred_df["age_true"]
    pred_df["train_cohort"] = train_cohort
    pred_df["test_cohort"] = test_cohort
    pred_df["feature_set"] = feature_set
    pred_df.to_csv(pred_path, index=False)

    metrics = compute_metrics(pred_df["age_true"], pred_df["pred_bias_corrected_global"], pred_df["cBAG_global"])
    row = {
        "train_cohort": train_cohort,
        "test_cohort": test_cohort,
        "feature_set": feature_set,
        "cell_type": "external",
        "status": "ok",
        "reason": "",
        "model_path": ckpt_path,
        "test_graph_path": graph_path_all,
        "bias_beta_from_train_oof": beta,
        "bias_alpha_from_train_oof": alpha,
        "target_normalization_used": used_norm,
        "target_mean": target_mean,
        "target_std": target_std,
        "train_node_dim": train_dims["node_feat_dim"],
        "test_node_dim": dims["node_feat_dim"],
        "train_global_dim": train_dims["global_feat_dim"],
        "test_global_dim": dims["global_feat_dim"],
        "train_edge_dim": train_dims["edge_dim"],
        "test_edge_dim": dims["edge_dim"],
        **metrics,
    }
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    save_scatter_pred_age(
        pred_df,
        os.path.join(cell_dir, "predicted_age_vs_age.png"),
        f"{feature_set}: train {train_cohort} → test {test_cohort}",
    )
    save_scatter_cbag_age(
        pred_df,
        os.path.join(cell_dir, "cBAG_vs_age.png"),
        f"{feature_set}: train {train_cohort} → test {test_cohort}, cBAG vs age",
    )

    return row, pred_df


def evaluate_diagonal_oof_cell(
    cohort: str,
    feature_set: str,
    results_root: str,
    outdir: str,
    dpi: int,
) -> Tuple[Dict, Optional[pd.DataFrame]]:
    cell_dir = ensure_dir(Path(outdir) / "per_cell" / f"train_{cohort}__test_{cohort}")
    pred_path = os.path.join(cell_dir, "predictions.csv")
    metrics_path = os.path.join(cell_dir, "metrics.csv")

    pred_df = load_diagonal_oof_predictions(cohort, feature_set, results_root)
    if pred_df is None or pred_df.empty:
        row = {
            "train_cohort": cohort,
            "test_cohort": cohort,
            "feature_set": feature_set,
            "cell_type": "diagonal_oof",
            "status": "missing_oof",
            "reason": "No usable OOF-global prediction file found",
        }
        pd.DataFrame([row]).to_csv(metrics_path, index=False)
        return row, None

    pred_df["train_cohort"] = cohort
    pred_df["test_cohort"] = cohort
    pred_df["feature_set"] = feature_set
    pred_df.to_csv(pred_path, index=False)

    metrics = compute_metrics(pred_df["age_true"], pred_df["pred_bias_corrected_global"], pred_df["cBAG_global"])
    row = {
        "train_cohort": cohort,
        "test_cohort": cohort,
        "feature_set": feature_set,
        "cell_type": "diagonal_oof",
        "status": "ok",
        "reason": "",
        **metrics,
    }
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    save_scatter_pred_age(
        pred_df,
        os.path.join(cell_dir, "predicted_age_vs_age.png"),
        f"{feature_set}: {cohort} OOF-global",
    )
    save_scatter_cbag_age(
        pred_df,
        os.path.join(cell_dir, "cBAG_vs_age.png"),
        f"{feature_set}: {cohort} OOF-global cBAG vs age",
    )

    return row, pred_df


# =============================================================================
# Main workflow
# =============================================================================
def run_feature_set(
    feature_set: str,
    cohorts: List[str],
    results_root: str,
    output_root: str,
    device,
    batch_size: int,
    dpi: int,
    include_diagonal_oof: bool = True,
) -> List[str]:
    outdir = ensure_dir(Path(output_root) / feature_set)
    rows = []
    all_preds = []

    print("\n" + "=" * 100)
    print(f"TRANSFER VALIDATION GRID: {feature_set}")
    print("=" * 100)
    print(f"Output: {outdir}")

    for train_cohort in cohorts:
        for test_cohort in cohorts:
            print(f"\nCell: train={train_cohort} -> test={test_cohort}")

            if train_cohort == test_cohort and include_diagonal_oof:
                row, pred_df = evaluate_diagonal_oof_cell(
                    cohort=train_cohort,
                    feature_set=feature_set,
                    results_root=results_root,
                    outdir=outdir,
                    dpi=dpi,
                )
            elif train_cohort == test_cohort and not include_diagonal_oof:
                row = {
                    "train_cohort": train_cohort,
                    "test_cohort": test_cohort,
                    "feature_set": feature_set,
                    "cell_type": "diagonal_skipped",
                    "status": "skipped",
                    "reason": "Diagonal OOF disabled",
                }
                pred_df = None
            else:
                row, pred_df = evaluate_external_cell(
                    train_cohort=train_cohort,
                    test_cohort=test_cohort,
                    feature_set=feature_set,
                    results_root=results_root,
                    outdir=outdir,
                    device=device,
                    batch_size=batch_size,
                    dpi=dpi,
                )

            rows.append(row)
            if pred_df is not None and not pred_df.empty:
                all_preds.append(pred_df)

    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(outdir, "transfer_grid_metrics.csv")
    metrics_xlsx = os.path.join(outdir, "transfer_grid_metrics.xlsx")
    metrics_df.to_csv(metrics_csv, index=False)
    try:
        metrics_df.to_excel(metrics_xlsx, index=False)
    except Exception as e:
        print(f"Could not save xlsx: {e}")

    compatibility_cols = [
        "train_cohort", "test_cohort", "feature_set", "cell_type", "status", "reason",
        "model_path", "test_graph_path",
        "train_node_dim", "test_node_dim",
        "train_global_dim", "test_global_dim",
        "train_edge_dim", "test_edge_dim",
    ]
    compatibility_df = metrics_df[[c for c in compatibility_cols if c in metrics_df.columns]].copy()
    compatibility_csv = os.path.join(outdir, "transfer_grid_compatibility.csv")
    compatibility_df.to_csv(compatibility_csv, index=False)

    predictions_csv = os.path.join(outdir, "transfer_grid_predictions_all.csv")
    if all_preds:
        pred_all = pd.concat(all_preds, ignore_index=True, sort=False)
        pred_all.to_csv(predictions_csv, index=False)
    else:
        pd.DataFrame().to_csv(predictions_csv, index=False)

    ok_df = metrics_df[metrics_df["status"] == "ok"].copy()
    for metric in ["MAE", "RMSE", "R2", "r", "cBAG_age_slope"]:
        save_heatmap(
            metrics_df=ok_df,
            metric=metric,
            cohorts=cohorts,
            out_png=os.path.join(outdir, f"heatmap_{metric}.png"),
            title=f"{feature_set}: train cohort × test cohort {metric}",
        )

    manifest = os.path.join(outdir, "transfer_grid_manifest.csv")
    outputs = [
        metrics_csv,
        metrics_xlsx,
        compatibility_csv,
        predictions_csv,
        os.path.join(outdir, "heatmap_MAE.png"),
        os.path.join(outdir, "heatmap_RMSE.png"),
        os.path.join(outdir, "heatmap_R2.png"),
        os.path.join(outdir, "heatmap_r.png"),
        os.path.join(outdir, "heatmap_cBAG_age_slope.png"),
    ]
    pd.DataFrame([{
        "feature_set": feature_set,
        "cohorts": ",".join(cohorts),
        "output_dir": outdir,
        "outputs": ";".join(outputs),
    }]).to_csv(manifest, index=False)
    outputs.append(manifest)

    print("\nSaved feature-set transfer outputs:")
    for p in outputs:
        print(f"  {p}")

    return outputs


def main():
    args = parse_args()

    work = os.path.abspath(args.work)
    results_root = os.path.abspath(args.results_root) if args.results_root else os.path.join(work, "ines/results")
    output_root = os.path.abspath(args.output_root) if args.output_root else os.path.join(
        results_root,
        "BrainAgeTransferValidation_TrainTestGrid_OOFGlobal",
    )

    cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]
    invalid_cohorts = [c for c in cohorts if c not in COHORT_CONFIG]
    if invalid_cohorts:
        raise ValueError(f"Unsupported cohorts: {invalid_cohorts}")

    if args.feature_sets.strip().upper() == "ALL":
        feature_sets = ALL_FEATURE_SETS
    else:
        feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]

    device = torch.device("cpu") if args.use_cpu else DEVICE

    ensure_dir(output_root)
    print("\n" + "=" * 100)
    print("CROSS-COHORT TRANSFER VALIDATION GRID")
    print("=" * 100)
    print(f"WORK:         {work}")
    print(f"RESULTS_ROOT: {results_root}")
    print(f"OUTPUT_ROOT:  {output_root}")
    print(f"Device:       {device}")
    print(f"Cohorts:      {', '.join(cohorts)}")
    print(f"Feature sets: {', '.join(feature_sets)}")
    print("=" * 100)

    all_outputs = []
    for fs in feature_sets:
        all_outputs.extend(
            run_feature_set(
                feature_set=fs,
                cohorts=cohorts,
                results_root=results_root,
                output_root=output_root,
                device=device,
                batch_size=args.batch_size,
                dpi=args.dpi,
                include_diagonal_oof=not args.no_diagonal_oof,
            )
        )

    combined_metrics = []
    combined_compat = []
    for fs in feature_sets:
        m = os.path.join(output_root, fs, "transfer_grid_metrics.csv")
        c = os.path.join(output_root, fs, "transfer_grid_compatibility.csv")
        if os.path.exists(m):
            combined_metrics.append(pd.read_csv(m))
        if os.path.exists(c):
            combined_compat.append(pd.read_csv(c))

    if combined_metrics:
        pd.concat(combined_metrics, ignore_index=True, sort=False).to_csv(
            os.path.join(output_root, "combined_transfer_grid_metrics.csv"),
            index=False,
        )

    if combined_compat:
        pd.concat(combined_compat, ignore_index=True, sort=False).to_csv(
            os.path.join(output_root, "combined_transfer_grid_compatibility.csv"),
            index=False,
        )

    overall_manifest = os.path.join(output_root, "transfer_grid_OVERALL_manifest.csv")
    pd.DataFrame([{
        "results_root": results_root,
        "output_root": output_root,
        "cohorts": ",".join(cohorts),
        "feature_sets": ",".join(feature_sets),
        "outputs": ";".join(all_outputs),
    }]).to_csv(overall_manifest, index=False)

    print("\nFinished transfer validation grid.")
    print(f"Overall manifest: {overall_manifest}")


if __name__ == "__main__":
    main()
