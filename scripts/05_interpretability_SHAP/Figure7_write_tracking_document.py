#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write a tracking document for Figure7_SHAP.

This script records:
  - scripts used
  - run command context
  - output directories
  - subject-level SHAP file counts by cohort/model/kind
  - key manifest files
  - final figure/table locations

Run after Figure7_SHAP pipeline finishes.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_OUT_BASE = (
    Path(DEFAULT_WORK)
    / "ines/results"
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Figure7_SHAP"
)
DEFAULT_CODE_DIR = Path(DEFAULT_WORK) / "ines/code/BAG_Stability052627"


def parse_args():
    p = argparse.ArgumentParser(description="Write Figure7_SHAP tracking document.")
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE))
    p.add_argument("--code-dir", default=str(DEFAULT_CODE_DIR))
    p.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", "NA"))
    p.add_argument("--log-out", default="")
    p.add_argument("--log-err", default="")
    return p.parse_args()


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"FAILED: {e}"


def file_count_by_kind(out_base: Path) -> pd.DataFrame:
    patterns = {
        "global_feature_shap": "global_feature_shap_subject_*.csv",
        "node_feature_shap": "node_feature_shap_subject_*.csv",
        "edge_shap": "edge_shap_subject_*.csv",
    }
    rows = []
    for kind, pat in patterns.items():
        for p in out_base.rglob(pat):
            parts = p.parts
            try:
                i = parts.index("Figure7_SHAP")
            except ValueError:
                continue
            if len(parts) <= i + 3:
                continue
            rows.append({"cohort_slug": parts[i + 1], "model": parts[i + 2], "kind": parts[i + 3], "file": str(p)})
    if not rows:
        return pd.DataFrame(columns=["cohort_slug", "model", "kind", "n_subject_files"])
    df = pd.DataFrame(rows)
    return (
        df.groupby(["cohort_slug", "model", "kind"], as_index=False)
        .agg(n_subject_files=("file", "count"))
        .sort_values(["cohort_slug", "model", "kind"])
    )


def collect_outputs(out_base: Path) -> pd.DataFrame:
    rows = []
    for p in out_base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".png", ".pdf", ".xlsx", ".md", ".txt"}:
            continue
        if p.name.startswith(("global_feature_shap_subject_", "node_feature_shap_subject_", "edge_shap_subject_")):
            continue
        rows.append({
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size_bytes": p.stat().st_size,
            "path": str(p),
        })
    return pd.DataFrame(rows).sort_values(["modified", "path"]) if rows else pd.DataFrame(columns=["modified", "size_bytes", "path"])


def main():
    args = parse_args()
    out_base = Path(args.out_base).expanduser().resolve()
    code_dir = Path(args.code_dir).expanduser().resolve()
    track_dir = out_base / "tracking"
    track_dir.mkdir(parents=True, exist_ok=True)

    counts = file_count_by_kind(out_base)
    counts_csv = track_dir / "Figure7_SHAP_subject_file_counts_by_cohort_model_kind.csv"
    counts.to_csv(counts_csv, index=False)

    outputs = collect_outputs(out_base)
    outputs_csv = track_dir / "Figure7_SHAP_output_inventory.csv"
    outputs.to_csv(outputs_csv, index=False)

    git_status = run(["git", "-C", str(code_dir), "status", "--short"])
    git_head = run(["git", "-C", str(code_dir), "rev-parse", "HEAD"])
    git_branch = run(["git", "-C", str(code_dir), "rev-parse", "--abbrev-ref", "HEAD"])

    doc = track_dir / "Figure7_SHAP_TRACKING.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    key_paths = [
        out_base / "combined_shap_run_summary.csv",
        out_base / "Figure6_SHAP_figure_manifest.csv",
        out_base / "cross_cohort_aggregated" / "Figure6_MASTER_v3_workflow_manifest.csv",
        out_base / "cross_cohort_aggregated" / "final_figures" / "Figure6_final_outputs_manifest.csv",
        out_base / "common4_consensus" / "Figure7_common4_SHAP_consensus_manifest.csv",
        out_base / "common4_consensus" / "Figure7_common4_SHAP_consensus_manifest.csv",
    ]

    with open(doc, "w", encoding="utf-8") as f:
        f.write("# Figure7_SHAP tracking document\n\n")
        f.write(f"Generated: `{now}`\n\n")
        f.write("## Purpose\n\n")
        f.write("Figure7_SHAP contains the SHAP interpretability workflow for OOF-global brain-age GNN models. ")
        f.write("It computes subject-level global, node-feature, and edge SHAP values, then builds cross-cohort aggregate figures and consensus tables/plots for features present across all cohorts.\n\n")

        f.write("## Run context\n\n")
        f.write(f"- Host: `{platform.node()}`\n")
        f.write(f"- Python: `{platform.python_version()}`\n")
        f.write(f"- User: `{os.environ.get('USER', 'NA')}`\n")
        f.write(f"- SLURM job ID: `{args.slurm_job_id}`\n")
        f.write(f"- Code directory: `{code_dir}`\n")
        f.write(f"- Output base: `{out_base}`\n")
        if args.log_out:
            f.write(f"- STDOUT log: `{args.log_out}`\n")
        if args.log_err:
            f.write(f"- STDERR log: `{args.log_err}`\n")
        f.write("\n")

        f.write("## Scripts used\n\n")
        for script in [
            "Figure6_buildSHAP.py",
            "FIgure6_aggregateC.py",
            "Figure7_common4_SHAP_consensus.py",
            "Figure7_SHAP_pipeline.py",
            "Figure7_write_tracking_document.py",
            "run_Figure7_SHAP_full_recompute.slurm",
            "run_Figure7_SHAP_aggregate_only.slurm",
        ]:
            p = code_dir / script
            status = "present" if p.exists() else "missing"
            f.write(f"- `{p}` — {status}\n")
        f.write("\n")

        f.write("## Git state\n\n")
        f.write(f"- Branch: `{git_branch}`\n")
        f.write(f"- Commit: `{git_head}`\n")
        f.write("- Uncommitted changes:\n\n")
        f.write("```text\n")
        f.write(git_status + "\n")
        f.write("```\n\n")

        f.write("## Main command templates\n\n")
        f.write("Full recompute, including edges:\n\n")
        f.write("```bash\n")
        f.write("sbatch run_Figure7_SHAP_full_recompute.slurm\n")
        f.write("```\n\n")
        f.write("Aggregate/replot only from existing SHAP CSVs:\n\n")
        f.write("```bash\n")
        f.write("sbatch run_Figure7_SHAP_aggregate_only.slurm\n")
        f.write("```\n\n")

        f.write("## Output organization\n\n")
        f.write(f"- Subject-level SHAP CSVs: `{out_base}/<cohort>/<model>/<kind>/*_subject_*.csv`\n")
        f.write(f"- Per-cohort summary CSVs: `{out_base}/<cohort>/<model>/<kind>/*_summary_all_subjects.csv`\n")
        f.write(f"- Cross-cohort aggregate figures/tables: `{out_base}/cross_cohort_aggregated/`\n")
        f.write(f"- Common-4-cohort consensus figures/tables: `{out_base}/common4_consensus/`\n")
        f.write(f"- Tracking files: `{track_dir}`\n\n")

        f.write("## Key manifests and summaries\n\n")
        for p in key_paths:
            f.write(f"- `{p}` — {'present' if p.exists() else 'missing'}\n")
        f.write("\n")

        f.write("## Subject-level SHAP file counts\n\n")
        f.write(f"Detailed CSV: `{counts_csv}`\n\n")
        if counts.empty:
            f.write("No subject-level SHAP files found at tracking time.\n\n")
        else:
            f.write(counts.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Output inventory\n\n")
        f.write(f"Detailed CSV: `{outputs_csv}`\n\n")
        f.write("The inventory excludes per-subject SHAP CSVs to keep the table readable.\n\n")

        f.write("## Manuscript interpretation note\n\n")
        f.write("The consensus outputs rank features by mean absolute SHAP magnitude among features observed in all requested cohorts. ")
        f.write("These plots should be interpreted as cross-cohort stability/consensus summaries, complementary to top-N SHAP plots that may be driven by a subset of cohorts.\n")

    print("Tracking document:", doc)
    print("Counts CSV:", counts_csv)
    print("Output inventory:", outputs_csv)


if __name__ == "__main__":
    main()
