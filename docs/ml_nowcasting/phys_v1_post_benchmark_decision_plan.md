# Phys V1 Post-Benchmark Decision Plan

This document defines what happens after the `residual_windsup_sst_prev_phys_v1`
dataset rebuild and benchmark finish.

## Goal

Decide whether the new physical signals should replace or extend the current
champion model.

The current champion reference is:

- RMSE: `1.268019`
- MAE: `0.930465`
- target RMSE: `< 0.9`

## Required Inputs

The post-benchmark decision report waits for:

- base LightGBM audit:
  `/srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_prev_phys_v1_2024_2025_to_2026_v1/tabular_rmse09_audit.json`
- calibrated ExtraTrees result:
  `/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_phys_v1/calibration_results.json`
- calibrated 2026 predictions:
  `/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_phys_v1/calibrated_predictions_2026.parquet`
- physical feature audit:
  `/srv/data/corsewind/ml_dataset/training_tables/phys_v1_required_feature_audit.json`
- physical signal coverage:
  `/srv/data/corsewind/ml_dataset/training_tables/phys_v1_signal_coverage.json`

## Outputs

Expected report outputs:

- `/srv/data/corsewind/ml_dataset/benchmarks/phys_v1_decision_report.json`
- `/srv/data/corsewind/ml_dataset/benchmarks/phys_v1_decision_report.md`

## Decision Rules

- `target_achieved_candidate`: best `phys_v1` RMSE is below `0.9`.
- `promote_candidate`: best `phys_v1` RMSE beats the champion by at least
  `0.005`.
- `small_improvement`: best `phys_v1` beats the champion but by less than
  `0.005`.
- `not_improved`: best `phys_v1` does not beat the champion.
- `incomplete`: required benchmark metrics are missing.

Promotion is not automatic. If `phys_v1` improves global RMSE, it still needs a
spot/horizon check, especially on:

- La Tonnara
- Santa Manza
- Balistra
- `+45` and `+60 min`
- high-wind regimes
- thermal-signal quartiles when available

## Lightweight Watcher

The independent watcher is:

`scripts/ml_dataset/z2_phys_v1_decision_report_watcher.sh`

It does not touch the running rebuild or training jobs. It only waits for final
artifacts, then calls:

`scripts/ml_dataset/summarize_phys_v1_decision_report.py`

This keeps the final interpretation reproducible and prevents us from manually
piecing together RMSE, feature coverage, spot errors and horizon errors.
