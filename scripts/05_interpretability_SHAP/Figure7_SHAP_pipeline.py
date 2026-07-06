#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 SHAP pipeline wrapper
==============================

Runs the Figure7_SHAP workflow from the command line.

This wrapper deliberately uses the existing, validated SHAP computation and
aggregation scripts:
  - Figure6_buildSHAP.py        : computes subject-level SHAP CSVs
  - FIgure6_aggregateC.py       : aggregates cross-cohort SHAP outputs
  - Figure7_common4_SHAP_consensus.py : adds common-cohort consensus plots/tables

The only conceptual change is the output root:
  .../Figure7_SHAP/

Modes
-----
compute-and-aggregate
    Recompute global/node/edge SHAP, then aggregate, then run common-cohort consensus.

aggregate-only
    Do not recompute SHAP. Rebuild summary figures/tables from existing CSVs, then consensus.

aggregate-master-only
    Skip Figure6_buildSHAP.py entirely. Run aggregation and common-cohort consensus only.

consensus-only
    Run only Figure7_common4_SHAP_consensus.py from existing summary CSVs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
DEFAULT_OUT_BASE = (
    Path(DEFAULT_WORK)
    / "ines/results"
    / "BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation"
    / "Figure7_SHAP"
)
DEFAULT_COHORTS = "ADNI,ADRC,HABS,AD_DECODE"
DEFAULT_FEATURE_SETS = "imaging_only,imaging_demographics,imaging_biomarkers,full,full_no_cardiovascular"
DEFAULT_MAIN_MODEL = "imaging_only"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Figure 7 SHAP compute/aggregate/consensus workflow.")
    p.add_argument("--mode", choices=["compute-and-aggregate", "aggregate-only", "aggregate-master-only", "consensus-only"], default="aggregate-only")
    p.add_argument("--build-script", default="Figure6_buildSHAP.py")
    p.add_argument("--aggregate-script", default="FIgure6_aggregateC.py")
    p.add_argument("--consensus-script", default="Figure7_common4_SHAP_consensus.py")
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE))
    p.add_argument("--cohorts", default=DEFAULT_COHORTS)
    p.add_argument("--feature-sets", "--models", dest="feature_sets", default=DEFAULT_FEATURE_SETS)
    p.add_argument("--main-model", default=DEFAULT_MAIN_MODEL)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--max-subjects", default="None")
    p.add_argument("--run-global-shap", type=int, choices=[0, 1], default=1)
    p.add_argument("--run-node-shap", type=int, choices=[0, 1], default=1)
    p.add_argument("--run-edge-shap", type=int, choices=[0, 1], default=1)
    p.add_argument("--formats", default="png,pdf")
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--top-n-consensus", type=int, default=25)
    p.add_argument("--min-cohorts", type=int, default=None)
    p.add_argument("--skip-beeswarm", action="store_true")
    p.add_argument("--skip-final-assembly", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_script(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if p.exists():
        return p.resolve()
    cwd_candidate = Path.cwd() / path_like
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    raise FileNotFoundError(f"Script not found: {path_like}")


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("\n" + "=" * 100)
    print("RUN:", " ".join(shlex.quote(x) for x in cmd))
    print("=" * 100)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def set_thread_caps() -> None:
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ.setdefault(k, "1")


def main() -> None:
    args = parse_args()
    set_thread_caps()

    out_base = str(Path(args.out_base).expanduser())
    build_script = resolve_script(args.build_script) if args.mode in {"compute-and-aggregate", "aggregate-only"} else None
    aggregate_script = resolve_script(args.aggregate_script) if args.mode in {"compute-and-aggregate", "aggregate-only", "aggregate-master-only"} else None
    consensus_script = resolve_script(args.consensus_script)

    print("Figure 7 SHAP pipeline")
    print("mode:", args.mode)
    print("out_base:", out_base)
    print("cohorts:", args.cohorts)
    print("feature_sets:", args.feature_sets)
    print("main_model:", args.main_model)
    print("run_edge_shap:", args.run_edge_shap)

    if args.mode in {"compute-and-aggregate", "aggregate-only"}:
        skip_computation = "0" if args.mode == "compute-and-aggregate" else "1"
        build_cmd = [
            sys.executable,
            str(build_script),
            "--cohorts", args.cohorts,
            "--feature-sets", args.feature_sets,
            "--main-feature-set", args.main_model,
            "--out-base", out_base,
            "--skip-computation", skip_computation,
            "--formats", args.formats,
            "--dpi", str(args.dpi),
            "--n-jobs", str(args.n_jobs),
            "--run-global-shap", str(args.run_global_shap),
            "--run-node-shap", str(args.run_node_shap),
            "--run-edge-shap", str(args.run_edge_shap),
        ]
        if args.mode == "compute-and-aggregate":
            build_cmd.extend(["--max-subjects", str(args.max_subjects)])
        run_cmd(build_cmd, dry_run=args.dry_run)

    if args.mode in {"compute-and-aggregate", "aggregate-only", "aggregate-master-only"}:
        agg_cmd = [
            sys.executable,
            str(aggregate_script),
            "--out-base", out_base,
            "--cohorts", args.cohorts,
            "--models", args.feature_sets,
            "--main-model", args.main_model,
            "--dpi", str(args.dpi),
        ]
        if args.skip_beeswarm:
            agg_cmd.append("--skip-beeswarm")
        if args.skip_final_assembly:
            agg_cmd.append("--skip-final-assembly")
        run_cmd(agg_cmd, dry_run=args.dry_run)

    consensus_cmd = [
        sys.executable,
        str(consensus_script),
        "--out-base", out_base,
        "--cohorts", args.cohorts,
        "--models", args.feature_sets,
        "--top-n", str(args.top_n_consensus),
        "--formats", args.formats,
        "--dpi", str(args.dpi),
    ]
    if args.min_cohorts is not None:
        consensus_cmd.extend(["--min-cohorts", str(args.min_cohorts)])
    run_cmd(consensus_cmd, dry_run=args.dry_run)

    print("\n[DONE] Figure 7 SHAP pipeline finished.")
    print("Check outputs under:")
    print(Path(out_base))
    print(Path(out_base) / "cross_cohort_aggregated")
    print(Path(out_base) / "common4_consensus")


if __name__ == "__main__":
    main()
