# Current Champion Provenance

Generated: 2026-06-30

## Status

The current locked wind-mean champion and operational gust champion are known
and their evaluation artifacts are preserved, but the original run directories
do not all contain full reproducible run manifests with command line, git SHA,
dependency versions, and input checksums.

This document records the verified provenance we currently have and the gaps to
close before treating the champion recipes as fully reproducible.

## Wind-Mean Champion Identity

Run id:

```text
prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
```

Remote artifact root:

```text
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
```

Current live inference defaults reference this run:

```text
scripts/ml_dataset/run_live_wind_mean_inference.py
scripts/ml_dataset/run_live_wind_and_gust_inference.py
```

## Gust Champion Identity

Operational champion:

```text
new_scale070_gust_recipe
```

Base model run:

```text
tabular_lgbm_225k_prev_lowmem_gust_from_wind_champion_recipe_2024_2025_to_2026_v1
```

Second-stage calibrator run:

```text
prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1
```

Remote artifact roots:

```text
/srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_prev_lowmem_gust_from_wind_champion_recipe_2024_2025_to_2026_v1
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1
```

Current combined live inference defaults now reference this gust recipe:

```text
scripts/ml_dataset/run_live_wind_and_gust_inference.py
```

Verified 2026 score:

```text
rows: 31,429
base corrected RMSE: 1.501342
base corrected MAE: 1.096947
calibrated RMSE: 1.484221
calibrated MAE: 1.073906
calibrated bias: +0.056219
```

Decision note:

```text
docs/ml_nowcasting/gust_from_wind_champion_recipe_v1.md
```

## Wind Preserved Artifacts

The run directory contains:

```text
calibrator.joblib
calibrated_predictions_2026.parquet
calibrated_predictions_2025h2_reconstructed.parquet
calibration_results.json
calibration_results.md
gap_audit/rmse09_gap_audit_scale070.json
gap_audit/rmse09_gap_audit_scale070.md
```

Important local documentation:

```text
docs/ml_nowcasting/rmse_0_9_path_to_target.md
docs/ml_nowcasting/scientific_error_diagnostic_report.md
docs/ml_nowcasting/rmse09_gap_audit_current_best_scale070.md
docs/ml_nowcasting/rmse09_gap_audit_current_best_scale070.json
docs/ml_nowcasting/rmse_0_9_iteration_notes.md
```

## Wind Verified Score

Locked 2026 evaluation:

```text
rows: 31,429
evaluation window: 2026-01-01T00:00:00Z -> 2026-07-01T00:00:00Z
lead times: +15, +30, +45, +60 min
base corrected RMSE: 1.276846
base corrected MAE: 0.943833
calibrated RMSE: 1.268019
calibrated MAE: 0.930465
calibrated bias: +0.018767
gap to RMSE 0.9: 0.368019
```

By lead:

| Lead | Rows | RMSE | MAE | Bias |
| ---: | ---: | ---: | ---: | ---: |
| +15 | 7,275 | 1.100990 | 0.813567 | +0.003552 |
| +30 | 7,045 | 1.250536 | 0.915433 | +0.001041 |
| +45 | 7,102 | 1.311420 | 0.971726 | +0.036616 |
| +60 | 10,007 | 1.359074 | 0.996750 | +0.029640 |

## Verified Inputs

`calibration_results.json` records these inputs:

```text
calibration_predictions:
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_calbase_2024_to_2025h2_v1/calibration_predictions_2025h2.parquet

evaluation_predictions:
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1/tabular_holdout_predictions.parquet

calibration window:
  2025-07-01T00:00:00Z -> 2026-01-01T00:00:00Z

evaluation window:
  2026-01-01T00:00:00Z -> 2026-07-01T00:00:00Z
```

The duplicated `lead_minutes` list in the JSON is a known artifact of the
script using `action="append"` with a non-empty default. Operationally the
evaluated horizons are `15, 30, 45, 60`.

## Verified Model

The saved `calibrator.joblib` loads as a scikit-learn `Pipeline`:

```text
ColumnTransformer
  numeric:
    SimpleImputer(strategy="median")
  categorical:
    SimpleImputer(strategy="constant", fill_value="__missing__")
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

ExtraTreesRegressor
  n_estimators=300
  min_samples_leaf=80
  n_jobs=2
  random_state=42
```

Feature counts recorded in `calibration_results.json`:

```text
feature columns: 302
numeric columns: 300
categorical columns: 2
categorical columns: spot_id, station_id
```

The model is a second-stage residual calibrator. It uses the first-stage
corrected wind prediction as the base and predicts a residual correction for
wind mean.

Output columns in `calibrated_predictions_2026.parquet` include:

```text
corrected_wind_mean_ms
predicted_second_stage_residual_ms
calibrated_wind_mean_ms
actual_wind_mean_ms
```

For this run, the older result JSON does not include newer fields such as
`target`, `actual_column`, `base_prediction_column`, `scale_selection`, or
`group_modeling`.

## Reconstructible Script Path

The relevant script is:

```text
scripts/ml_dataset/train_prediction_residual_calibrator.py
```

Current script behavior:

- loads calibration and evaluation prediction Parquet files;
- filters time windows and lead times;
- defines the target as `actual_wind_mean_ms - corrected_wind_mean_ms`;
- keeps allowed base, `features__*`, and `baselines__*` columns;
- removes leaky actual/error columns;
- imputes numeric and categorical columns;
- fits the selected residual model;
- writes calibrated predictions, model joblib, JSON, and Markdown.

## Reproduction Skeleton

This is the known shape of the run. It should be treated as a reconstruction
skeleton, not as the exact original command, because the original command was
not preserved in the champion directory.

```bash
python3 scripts/ml_dataset/train_prediction_residual_calibrator.py \
  --calibration-predictions /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_calbase_2024_to_2025h2_v1/calibration_predictions_2025h2.parquet \
  --evaluation-predictions /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1/tabular_holdout_predictions.parquet \
  --calibration-start-utc 2025-07-01T00:00:00Z \
  --calibration-end-utc 2026-01-01T00:00:00Z \
  --evaluation-start-utc 2026-01-01T00:00:00Z \
  --evaluation-end-utc 2026-07-01T00:00:00Z \
  --lead-minute 15 \
  --lead-minute 30 \
  --lead-minute 45 \
  --lead-minute 60 \
  --model-family extra_trees \
  --max-iter 300 \
  --min-samples-leaf 80 \
  --n-jobs 2 \
  --clip-correction-ms 2.0 \
  --correction-scale 0.70 \
  --output-predictions <output>/calibrated_predictions_2026.parquet \
  --output-model <output>/calibrator.joblib \
  --output-json <output>/calibration_results.json \
  --output-md <output>/calibration_results.md
```

The `scale070` suffix strongly indicates a fixed or selected correction scale
of `0.70`, and later scripts use explicit scale candidates including `0.70`.
However, because this older JSON lacks `scale_selection`, the exact scale
selection method for the original run is not fully recoverable from the run
directory alone.

## Provenance Gaps

Missing from the original champion artifact:

- exact shell command;
- git commit SHA;
- dependency versions;
- input file checksums;
- complete first-stage base training manifest;
- explicit `scale_selection` record;
- exact feature name list in JSON.

These gaps do not invalidate the recorded score, because predictions, metrics,
and the saved model exist. They do mean the champion is not yet documented to
the standard required for reproducible model governance.

## Required Hardening

Before the next champion candidate can replace this model, the training scripts
must write a complete manifest containing:

- command line and normalized args;
- git SHA and dirty-worktree flag;
- Python and package versions;
- host name;
- input paths, row counts, date ranges, and file hashes;
- selected target, base prediction, calibrated prediction columns;
- selected feature column list;
- model hyperparameters;
- scale-selection method and candidates;
- output paths and hashes;
- common-key evaluation contract.

Candidate models should not be promoted unless their manifest is at least this
complete and they beat the current champion on identical keyed rows.
