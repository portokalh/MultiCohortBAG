# Code Release

Created: 2026-07-06T17:15:12

This directory contains the curated scripts and provenance files for the brain-age / cBAG manuscript.

## Contents

* `scripts/00_graph_builder/`: graph construction and cohort graph-builder scripts.
* `scripts/01_training/`: model training, OOF prediction, and cBAG generation.
* `scripts/02_validation/`: validation and full-cohort cBAG source generation.
* `scripts/03_figures/`: figure-generation scripts.
* `scripts/04_tables/`: table-generation scripts.
* `scripts/05_interpretability_SHAP/`: SHAP and interpretability scripts.
* `scripts/06_transferability/`: recovered transferability and recalibration scripts originally developed under the `Figure10_*` label.
* `manuscript_provenance/`: final cBAG provenance workbook and CSV.
* `CODE_TRACKING_TABLE.csv/xlsx`: mapping from copied release scripts to original analysis paths, expressed relative to `$WORK` where applicable.

## Known unresolved items

* Figure 9 source/provenance.
* Some supplementary tables may need individual mapping if included in the final manuscript.

## Notes

This release directory is a curated publication-facing copy, not a complete working clone of the analysis environment. Large datasets, model checkpoints, and protected/raw cohort data are not copied here.

Paths beginning with `$WORK` refer to the local project work root used during analysis.

## Figure 10 / transferability resolution

The primary cross-cohort transferability runner was recovered from:

`$WORK/ines/code/BAG_Stability061926/Figure10_cross_cohort_transferability_train_test_grid.py`

It generated the all-model train-cohort/test-cohort transferability grid under:

`$WORK/ines/results/BrainAgeValidation_AllCohorts_BAGBiasCorr_OOFGlobal_BiologicalValidation/Figure10_cross_cohort_transferability_ALL_MODELS/`

The run included ADNI, ADRC, HABS, and AD_DECODE, with five feature sets: imaging-only, imaging+demographics, imaging+biomarkers, full, and full-no-cardiovascular.

Recommended manuscript use: retain transferability as supplementary calibration/deployment material. Use imaging-only transferability as the primary supplementary result because it aligns with the main imaging-only cBAG validation and SHAP analyses.

Recommended Supplementary Figure S8 source:

`Figure10_imaging_only_audited_transfer_heatmaps_panel_CLEAN.pdf/png`

Optional Supplementary Figure S9 source:

`Figure10_recalibration_sensitivity_panel_FIXED.pdf/png`

Primary source tables:

* `transfer_grid_metrics_RAW_vs_cBAG_WITH_ROBUST_QC.csv`
* `transfer_grid_metrics_RECALIBRATION_SENSITIVITY_FIXED.csv`
* `Figure10_recalibration_summary_FIXED.csv`

Older non-fixed recalibration panels should be archived but not used for manuscript submission.

## Transferability figure numbering note

The transferability scripts retain their original `Figure10_*` filenames because those names match the recovered analysis scripts and output directories. In the manuscript, this material is recommended for supplementary placement, primarily as Supplementary Figure S8 for clean imaging-only transferability and optionally Supplementary Figure S9 for fixed recalibration sensitivity.

Thus, `Figure10_*` in the code release refers to the original development label, not necessarily the final manuscript figure number.
