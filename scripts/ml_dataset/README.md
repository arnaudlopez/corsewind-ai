# ML Dataset Scripts

This directory contains the data-collection and dataset-construction utilities
for the CorseWind.ai ML nowcasting work.

It is intentionally separate from the operational forecast/rendering scripts in
`scripts/`.

Current responsibilities:

- archive normalized model-layer snapshots for ML history;
- sample normalized model layers at ML spot coordinates;
- import Beacon Live spots and observations;
- collect Météo-France observations;
- collect public WindsUp historical spot observations for true windsurf spots;
- inventory Météo-France AROME/AROME-PI WCS variables for feature planning;
- collect extra Météo-France AROME/AROME-PI thermal/context fields at ML spots;
- collect AROME 0.025 isobaric vertical profiles at ML spots;
- build the first canonical 15-minute spot feature store;
- inventory Copernicus Marine datasets useful for thermal-wind nowcasting;
- collect Copernicus Marine SST and sample it at ML spot coordinates;
- inventory EUMETSAT satellite datasets useful for cloud/land-heating features;
- scan the EUMETSAT catalogue for additional thermal/convection candidates;
- collect EUMETSAT MTG Cloud Mask products and sample them at ML spot coordinates;
- collect EUMETSAT Cloud Type, Land Surface Temperature, and Global Instability
  Indices products with the generic spot-product sampler;
- build residual-correction training rows from the spot feature store;
- export SAPHIR-style structured sequence datasets from residual training rows;
- audit and benchmark those SAPHIR-style sequence datasets;
- build sequence benchmark cases for foundation time-series models;
- benchmark Chronos-2, TimesFM, and Moirai on saved sequence cases;
- score the current HGB residual model on the same sequence benchmark cases;
- profile which data is currently available and where gaps remain.

Default outputs are written under:

```text
data/processed/ml_dataset/
```

That output tree is ignored by git and can be regenerated.

For large backfills, do not use the repository disk. Point `ML_DATASET_ROOT` to
an external disk or large volume and run the storage preflight first:

```bash
export ML_DATASET_ROOT=/Volumes/<large-disk>/corsewind/ml_dataset
python3 scripts/ml_dataset/storage_preflight.py \
  --ml-root "$ML_DATASET_ROOT" \
  --min-free-gb 250 \
  --create
```

The forecast engine and standalone ML collectors use `ML_DATASET_ROOT` for
raw caches, normalized samples, inventories, and feature-store outputs.

Optional Copernicus Marine setup:

```bash
pip install -r requirements-ml-dataset.txt
export COPERNICUSMARINE_SERVICE_USERNAME=...
export COPERNICUSMARINE_SERVICE_PASSWORD=...
```

Inventory relevant Copernicus products:

```bash
python3 scripts/ml_dataset/inventory_copernicus_marine_products.py
```

Inventory relevant EUMETSAT products:

```bash
python3 scripts/ml_dataset/inventory_eumetsat_products.py
```

Scan the broader EUMETSAT catalogue by thermal-wind keywords:

```bash
python3 scripts/ml_dataset/inventory_eumetsat_catalog_keywords.py
```

Collect and sample SST:

```bash
python3 scripts/ml_dataset/collect_copernicus_marine_sst.py \
  --start-datetime 2026-06-22T12:00:00 \
  --end-datetime 2026-06-22T15:00:00
```

Collect and sample extra AROME/AROME-PI WCS fields at spots:

```bash
python3 scripts/ml_dataset/collect_meteo_france_nwp_spot_features.py \
  --source arome \
  --input visualizations/wind2d/arome-corsica-latest.json \
  --max-steps 24 \
  --include-context-spots

python3 scripts/ml_dataset/collect_meteo_france_nwp_spot_features.py \
  --source aromepi \
  --input visualizations/wind2d/aromepi-corsica-latest.json \
  --max-steps 24 \
  --include-context-spots
```

Or let the forecast engine collect those fields after each source refresh:

```bash
ML_NWP_EXTRA_FIELDS_ENABLED=true \
python3 scripts/run_forecast_update_engine.py --enable-ml-nwp-extra-fields
```

Collect AROME 0.025 vertical profiles on isobaric levels:

```bash
python3 scripts/ml_dataset/collect_meteo_france_vertical_profiles.py \
  --input visualizations/wind2d/arome-corsica-latest.json \
  --max-steps 5 \
  --pressure-level-hpa 1000 \
  --pressure-level-hpa 925 \
  --pressure-level-hpa 850 \
  --include-context-spots
```

Or let the forecast engine run the conservative profile sampler after AROME:

```bash
ML_NWP_VERTICAL_PROFILES_ENABLED=true \
ML_NWP_VERTICAL_PROFILES_PRESSURE_LEVELS_HPA=1000,925,850 \
python3 scripts/run_forecast_update_engine.py --enable-ml-nwp-vertical-profiles
```

Build the 15-minute feature store used by the first training experiments:

```bash
python3 scripts/ml_dataset/build_spot_feature_store.py
```

Target observations are selected on the 15-minute grid from `use_for_ml=true`
spot observations. When multiple normalized observations can fill the same
`spot_id + target_time_utc`, the builder prefers the source/station declared in
`configs/ml_spots.json`, then the highest-priority source family
(`windsup`, `meteofrance`, `wunderground`, ...), then the finest source
resolution and closest timestamp. The selected source metadata is written in
`targets.observation_*` and carried into residual training labels.

Context-station slots include static geometry and pre-target upwind features
after rebuilding the feature store: station bearing from/to the spot, east/north
offsets, altitude delta, and `upwind_score_from_target_wind`. These are used to
make neighboring stations closer to the SAPHIR-style spatial inputs while
keeping the target time leakage-safe.

Or let the forecast engine rebuild it at the end of each cycle:

```bash
ML_FEATURE_STORE_ENABLED=true \
python3 scripts/run_forecast_update_engine.py --enable-ml-feature-store
```

Outputs:

```text
data/processed/ml_dataset/feature_store/spot_forecast_15min.jsonl
data/processed/ml_dataset/feature_store/spot_forecast_15min_profile.json
data/processed/ml_dataset/feature_store/spot_forecast_15min_feature_columns.csv
docs/ml_nowcasting/feature_store_schema.md
```

Build the first residual-correction training table:

```bash
python3 scripts/ml_dataset/build_residual_training_table.py \
  --feature-store "$ML_DATASET_ROOT/feature_store/pilot_20260622/spot_forecast_15min.jsonl" \
  --output-root "$ML_DATASET_ROOT/training_tables/residual_correction_pilot_20260622" \
  --lead-minutes 60,120,180,360
```

This table keeps issue-time observations, context stations, SST, satellite, and
model context as features, adds the NWP forecast at the target horizon as the
baseline, then labels the row with the observed target wind, gust, residuals
against the baseline, and windsurf threshold exceedance flags.

Evaluate raw NWP against a simple recent-error persistence correction:

```bash
python3 scripts/ml_dataset/evaluate_residual_training_table.py \
  --training-rows "$ML_DATASET_ROOT/training_tables/residual_correction_pilot_20260622/training_rows.jsonl" \
  --output-json "$ML_DATASET_ROOT/training_tables/residual_correction_pilot_20260622/evaluation.json" \
  --output-md "$ML_DATASET_ROOT/training_tables/residual_correction_pilot_20260622/evaluation.md"
```

Build and benchmark a small sequential foundation-model pilot:

```bash
python scripts/ml_dataset/benchmark_chronos2_sequences.py \
  --training-table-root "$ML_DATASET_ROOT/training_tables" \
  --run-id-prefix residual_windsup_sst_prev \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --eval-start 2026-01-01T00:00:00Z \
  --context-length 96 \
  --prediction-length 4 \
  --skip-hgb \
  --output-root "$ML_DATASET_ROOT/benchmarks/chronos2_sequence_pilot_2026_windsurf_1h"

python scripts/ml_dataset/score_hgb_sequence_benchmark.py \
  --benchmark-root "$ML_DATASET_ROOT/benchmarks/chronos2_sequence_pilot_2026_windsurf_1h" \
  --training-table-root "$ML_DATASET_ROOT/training_tables" \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --hgb-model-root "$ML_DATASET_ROOT/models/residual_windsup_sst_prev_2024_01_2026_06_hgb_regression_sample"

python scripts/ml_dataset/benchmark_chronos2_saved_sequences.py \
  --benchmark-root "$ML_DATASET_ROOT/benchmarks/chronos2_sequence_pilot_2026_windsurf_1h" \
  --cross-learning

python scripts/ml_dataset/benchmark_chronos2_residual_sequences.py \
  --benchmark-root "$ML_DATASET_ROOT/benchmarks/chronos2_sequence_pilot_2026_windsurf_1h" \
  --cross-learning

python scripts/ml_dataset/benchmark_timesfm_sequences.py \
  --benchmark-root "$ML_DATASET_ROOT/benchmarks/chronos2_sequence_pilot_2026_windsurf_1h"

python scripts/ml_dataset/benchmark_moirai_sequences.py \
  --benchmark-root "$ML_DATASET_ROOT/benchmarks/chronos2_sequence_pilot_2026_windsurf_1h"
```

Train a temporal residual calibrator from saved sequence benchmarks:

```bash
python scripts/ml_dataset/train_sequence_calibrator.py \
  --benchmark-root "$ML_DATASET_ROOT/benchmarks/sequence_2025_windsurf_1h_rmse09_v1" \
  --benchmark-root "$ML_DATASET_ROOT/benchmarks/sequence_2026_windsurf_1h_rmse09_v1" \
  --predictions-file predictions_with_timesfm.parquet \
  --output-root "$ML_DATASET_ROOT/benchmarks/calibrator_2025_to_2026_hgb_v1" \
  --train-end 2026-01-01T00:00:00Z \
  --eval-start 2026-01-01T00:00:00Z \
  --target-mode residual \
  --residual-baseline raw_wind_mean_ms \
  --model-family hist_gradient_boosting
```

Export, audit, and benchmark a SAPHIR-style sequence dataset:

```bash
python scripts/ml_dataset/export_corsewind_saphir_sequence_dataset.py \
  --training-table-root "$ML_DATASET_ROOT/training_tables" \
  --run-id-prefix residual_windsup_sst_prev_phys_v3_dem_fetch \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --output-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1" \
  --lead-minutes 15,30,45,60 \
  --context-length 32 \
  --max-samples-per-spot 400 \
  --issue-hour-start 8 \
  --issue-hour-end 17 \
  --train-end 2025-12-31T23:59:59Z \
  --eval-start 2026-01-01T00:00:00Z

python scripts/ml_dataset/audit_corsewind_saphir_sequence_dataset.py \
  --dataset-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1"

python scripts/ml_dataset/benchmark_corsewind_saphir_sequence_dataset.py \
  --dataset-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1" \
  --output-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1/benchmark" \
  --model-family hgb \
  --model-family extra_trees \
  --model-family ridge \
  --max-numeric-features 500 \
  --min-feature-non-null 50

python scripts/ml_dataset/benchmark_corsewind_saphir_neural_dataset.py \
  --dataset-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1" \
  --output-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1/saphir_neural_v1" \
  --epochs 96 \
  --patience 14 \
  --batch-size 512 \
  --hidden-dim 64 \
  --max-static-features 192 \
  --save-model
```

Export, audit, and benchmark the SAPHIR-style dictionary V2 dataset with real
neighbor-station histories and same-sample tabular/neural tests:

```bash
python scripts/ml_dataset/export_corsewind_saphir_dictionary_v2.py \
  --training-table-root "$ML_DATASET_ROOT/training_tables" \
  --run-id-prefix residual_windsup_sst_prev_phys_v3_dem_fetch \
  --start-month 2025-01 \
  --end-month 2026-06 \
  --output-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_dictionary_v2_medium_light" \
  --lead-minutes 15,30,45,60 \
  --context-length 24 \
  --max-context-stations 6 \
  --max-samples-per-spot 180 \
  --issue-hour-start 8 \
  --issue-hour-end 17 \
  --train-end 2026-01-01T00:00:00Z \
  --eval-start 2026-01-01T00:00:00Z \
  --require-gust

python scripts/ml_dataset/audit_corsewind_saphir_dictionary_v2.py \
  --dataset-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_dictionary_v2_medium_light"

python scripts/ml_dataset/benchmark_corsewind_saphir_dictionary_v2_tabular.py \
  --dataset-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_dictionary_v2_medium_light" \
  --output-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_dictionary_v2_medium_light/benchmark_v2_tabular" \
  --model-family ridge \
  --model-family hgb \
  --max-numeric-features 900 \
  --max-static-features 256 \
  --history-windows 4,8,16,32

python scripts/ml_dataset/benchmark_corsewind_saphir_dictionary_v2_neural.py \
  --dataset-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_dictionary_v2_medium_light" \
  --output-root "$ML_DATASET_ROOT/benchmarks/corsewind_saphir_dictionary_v2_medium_light/benchmark_v2_neural" \
  --epochs 100 \
  --patience 14 \
  --batch-size 256 \
  --hidden-dim 96
```

The first z2 V2 result is documented in:

```text
docs/ml_nowcasting/saphir_dictionary_v2_results.md
```

This calibrator is designed for the scientific approach used in the nowcasting
work: keep raw NWP as the physical prior, learn the recent/local residual, and
evaluate on a later time period. It adds persistence, short trend, cyclical
time, model disagreement, and probabilistic spread features when those columns
are available. When `--training-table-root` is provided, it also merges
leakage-safe `features__*` columns from `training_rows.parquet` by
`spot_id + issue_time_utc + lead_time_minutes`, while excluding labels.

Launch the full z2 RMSE-0.9 sequence experiment from the local repo:

```bash
python3 scripts/ml_dataset/launch_z2_rmse09_sequence_experiment.py
```

This syncs `scripts/ml_dataset/` to `/srv/data/corsewind/backfill_runner`,
builds the 2025 sequence benchmark if missing, adds Chronos-2 univariate and
TimesFM predictions, then trains the temporal 2025->2026 calibrator sweep.
The default benchmark now samples up to 240 cutoffs per spot, because the
previous smoke-test scale of 12 cutoffs per spot is too small for the RMSE-0.9
objective. Use `--remote-dry-run` or pass a lower `--max-cutoffs-per-spot`
directly to `run_rmse09_sequence_experiment.py` only for fast diagnostics.
The calibrator sweep includes Ridge, HistGradientBoosting, RandomForest, and
ExtraTrees by default; `--include-lightgbm` adds LightGBM when installed.
The sweep also trains lead-stratified variants for Ridge,
HistGradientBoosting, and ExtraTrees by default. These variants fit one
calibrator per forecast horizon when enough training rows exist, then fall back
to the global model for sparse or unseen groups. This makes the benchmark test
the scientific assumption that +15/+30/+45/+60 minute corrections are different
problems, instead of forcing a single global compromise.
The launcher also syncs `configs/`, `docs/ml_nowcasting/`, and
`requirements-ml-dataset.txt`, then runs a blocking environment preflight that
checks required files, Python packages, venv imports, GPU visibility, disk
space, training tables, and the 2026 evaluation benchmark.
By default this sweep includes the training-table context/upwind feature merge.
It also audits the monthly training-table Parquet schemas before calibration;
if the audit reports stale shards, rebuild them with:

```bash
python3 scripts/ml_dataset/launch_z2_rebuild_training_shards.py
```

It also writes `rmse09_audit.json` and `rmse09_audit.md`; only a `pass` verdict
there should be considered evidence that the RMSE-0.9 goal has been met. After
the audit it writes `rmse09_error_analysis.json/md`, which breaks down failures
by spot, lead time, actual wind bin, raw-NWP error bin, issue hour, and worst
issue days. It then writes `rmse09_decision.json/md`, which classifies the run
as `achieved`, `calibration_gap`, `routing_or_feature_gap`,
`input_signal_gap`, `inconclusive`, or `needs_more_evidence`.
The run also persists `rmse09_run_manifest.json` with commands, run options,
dataset roots, expected artifacts, git provenance, and the final assertion
command.
The audit recomputes the best model score from `calibrator_predictions.parquet`
when available, adds a bootstrap confidence interval, and lists worst spots and
per-lead RMSE. When prediction diagnostics are available, the audit verdict uses
the recomputed parquet score as `effective_rmse`; the sweep JSON score is only
metadata. The default pass gate also requires at least 3 spots, 4 lead times,
and 20 issue days in the evaluation predictions. The default confidence
interval uses an `issue_day` block bootstrap, not a row bootstrap, so correlated
lead-time rows from the same day do not make the result look artificially
stable. New calibrator runs also publish train/test split coverage; the audit
requires at least 3 training spots, 4 training lead times, and 60 training issue
days by default.
Use `--include-lightgbm` to add the optional LightGBM calibrator when that
package is installed in the z2 ML environment.
Use `--require-ci-upper-below-threshold` when you want the strictest proof mode:
the 95% bootstrap upper bound must also stay below the RMSE threshold.
Use `--assert-goal` when you want the experiment command itself to exit nonzero
unless the final `assert_rmse09_goal.py` gate proves the objective.
Use `--allow-preflight-fail` only when you want a diagnostic report without
protecting the expensive run.

Before spending GPU time, the local proof chain can be smoke-tested with:

```bash
python3 scripts/ml_dataset/smoke_test_rmse09_pipeline.py
```

This creates synthetic prediction artifacts and verifies the `audit -> error
analysis -> decision` path for `achieved`, `inconclusive`, `calibration_gap`,
and `input_signal_gap`. It also verifies that `assert_rmse09_goal.py` exits
successfully only for the `achieved` case.

The broader local readiness gate is:

```bash
python3 scripts/ml_dataset/check_rmse09_local_readiness.py
```

It runs syntax checks, the smoke test, dry-runs the experiment and z2 launcher,
checks the status command output, and scans the repo for the known pasted
secrets.

When z2 is intermittently unreachable, leave the guarded launcher waiting for
SSH and let it start the strict background experiment as soon as the machine is
available:

```bash
python3 scripts/ml_dataset/wait_for_z2_and_launch_rmse09.py \
  --max-wait-minutes -1
```

The default guarded launch runs the local readiness gate, then starts
`launch_z2_rmse09_sequence_experiment.py --background --include-lightgbm
--require-fresh-training-features --assert-goal`. Use
`--require-ci-upper-below-threshold` for the strictest proof mode, where the
95% block-bootstrap upper bound must also stay below 0.9 m/s. Preview the
command without waiting or launching remote work with:

```bash
python3 scripts/ml_dataset/wait_for_z2_and_launch_rmse09.py \
  --dry-run \
  --launch-dry-run
```

After a real z2 run, use the final assertion gate:

```bash
python3 scripts/ml_dataset/assert_rmse09_goal.py \
  --audit-json /srv/data/corsewind/ml_dataset/benchmarks/calibrator_2025_to_2026_sweep_context_v1/rmse09_audit.json \
  --decision-json /srv/data/corsewind/ml_dataset/benchmarks/calibrator_2025_to_2026_sweep_context_v1/rmse09_decision.json
```

This command exits with code `0` only when the leakage-safe audit passed,
`effective_rmse < 0.9`, prediction diagnostics are present, and the decision is
`achieved`.

Preview without running remote work:

```bash
python3 scripts/ml_dataset/launch_z2_rmse09_sequence_experiment.py \
  --dry-run \
  --remote-dry-run
```

For long z2 runs, prefer the background mode and inspect status separately:

```bash
python3 scripts/ml_dataset/launch_z2_rebuild_training_shards.py --background
python3 scripts/ml_dataset/check_z2_rmse09_status.py

python3 scripts/ml_dataset/launch_z2_rmse09_sequence_experiment.py --background
python3 scripts/ml_dataset/check_z2_rmse09_status.py
```

Background logs, PID files, and exit-status files are written under
`/srv/data/corsewind/ml_dataset/run_logs/`.

The first foundation-model pilot is documented in
`docs/ml_nowcasting/foundation_sequence_benchmark_2026_06_25.md`.

For the larger 2026 windsurf-hour benchmark, use
`docs/ml_nowcasting/foundation_sequence_benchmark_v2_2026_windsurf.md`.

The sequence builder also has an experimental +6h mode:

```bash
python scripts/ml_dataset/benchmark_chronos2_sequences.py \
  --training-table-root "$ML_DATASET_ROOT/training_tables" \
  --run-id-prefix residual_windsup_sst_prev \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --eval-start 2026-01-01T00:00:00Z \
  --context-length 96 \
  --prediction-length 24 \
  --allow-interpolated-future \
  --issue-hour-start 8 \
  --issue-hour-end 17 \
  --max-cutoffs-per-spot 2 \
  --skip-hgb \
  --output-root "$ML_DATASET_ROOT/benchmarks/sequence_2026_windsurf_6h_interpolated_smoke"
```

This mode interpolates sparse future NWP covariates onto a 15-minute grid. It is
useful for pipeline validation, but not yet a decision-grade +6h benchmark.

Run the chunked training backfill pipeline over a longer period:

```bash
python3 scripts/ml_dataset/run_training_backfill_pipeline.py \
  --start-date 2024-06-01 \
  --end-date 2024-06-30 \
  --start-hour-utc 10 \
  --end-hour-utc 18 \
  --chunk-days 31 \
  --run-id residual_backfill_2024_06 \
  --lead-minutes 60,120,180,360
```

Add `--collect-open-meteo` when the Open-Meteo historical forecast files for
the period are not already present. Add `--train-models` after rebuilding the
ML dataset Docker image with `scikit-learn` installed.

For monthly historical shards, use the monthly runner so every month gets the
same run id, chunking, evaluation, and Parquet export:

```bash
python3 scripts/ml_dataset/run_monthly_training_shards.py \
  --ml-root "$ML_DATASET_ROOT" \
  --registry configs/ml_spots.json \
  --context-registry configs/ml_context_stations.json \
  --start-month 2024-07 \
  --end-month 2024-08 \
  --run-id-prefix residual_windsup_spots \
  --train-smoke
```

Train first sklearn residual-correction models from an existing table:

```bash
python3 scripts/ml_dataset/train_residual_correction_model.py \
  --training-rows "$ML_DATASET_ROOT/training_tables/residual_backfill_2024_06/training_rows.jsonl" \
  --output-root "$ML_DATASET_ROOT/models/residual_backfill_2024_06"
```

Train residual-correction models from monthly Parquet shards:

```bash
python3 scripts/ml_dataset/train_residual_correction_parquet.py \
  --training-table-root "$ML_DATASET_ROOT/training_tables" \
  --run-id-prefix residual_windsup_sst_prev \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --output-root "$ML_DATASET_ROOT/models/residual_windsup_sst_prev_2024_01_2026_06_hgb_regression_sample" \
  --run-id residual_windsup_sst_prev_2024_01_2026_06_hgb_regression_sample \
  --model-family hist_gradient_boosting \
  --max-train-rows 300000 \
  --max-test-rows 80000 \
  --max-iter 120 \
  --skip-classification
```

The Parquet trainer discovers monthly shards, builds a temporal split, filters
empty/sparse/constant features, derives issue-time calendar features, and writes
models plus `training_results.json`, `training_results.md`, and
`feature_columns.json`.

Train first threshold-probability models from the same Parquet shards:

```bash
python3 scripts/ml_dataset/train_residual_correction_parquet.py \
  --training-table-root "$ML_DATASET_ROOT/training_tables" \
  --run-id-prefix residual_windsup_sst_prev \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --output-root "$ML_DATASET_ROOT/models/residual_windsup_sst_prev_2024_01_2026_06_hgb_threshold_sample" \
  --run-id residual_windsup_sst_prev_2024_01_2026_06_hgb_threshold_sample \
  --model-family hist_gradient_boosting \
  --max-train-rows 300000 \
  --max-test-rows 80000 \
  --max-iter 80 \
  --only-target labels__target_wind_gt_15kt \
  --only-target labels__target_wind_gt_20kt \
  --only-target labels__target_gust_gt_20kt \
  --only-target labels__target_gust_gt_25kt
```

Export a training table to Parquet for faster downstream filtering/training:

```bash
python3 scripts/ml_dataset/export_training_table_parquet.py \
  --training-rows "$ML_DATASET_ROOT/training_tables/residual_windsup_spots_2024_06/training_rows.jsonl" \
  --output-root "$ML_DATASET_ROOT/training_tables/residual_windsup_spots_2024_06"
```

Collect and sample EUMETSAT MTG Cloud Mask:

```bash
export EUMETSAT_CONSUMER_KEY=...
export EUMETSAT_CONSUMER_SECRET=...

python3 scripts/ml_dataset/collect_eumetsat_cloud_mask.py \
  --start-datetime 2026-06-23T13:00:00Z \
  --end-datetime 2026-06-23T14:50:00Z \
  --include-context-spots
```

Or let the forecast engine run it on an 8-minute cadence:

```bash
ML_EUMETSAT_CLOUD_MASK_ENABLED=true \
EUMETSAT_CONSUMER_KEY=... \
EUMETSAT_CONSUMER_SECRET=... \
python3 scripts/run_forecast_update_engine.py --enable-ml-eumetsat-cloud-mask
```

Collect and sample the next EUMETSAT thermal/convection products directly:

```bash
python3 scripts/ml_dataset/collect_eumetsat_spot_product.py \
  --product cloud_type \
  --start-datetime 2026-06-23T13:00:00Z \
  --end-datetime 2026-06-23T15:00:00Z \
  --include-context-spots

python3 scripts/ml_dataset/collect_eumetsat_spot_product.py \
  --product land_surface_temperature \
  --start-datetime 2026-06-23T13:00:00Z \
  --end-datetime 2026-06-23T15:00:00Z \
  --include-context-spots

python3 scripts/ml_dataset/collect_eumetsat_spot_product.py \
  --product global_instability_indices \
  --start-datetime 2026-06-23T13:00:00Z \
  --end-datetime 2026-06-23T15:00:00Z \
  --include-context-spots
```

Or let the forecast engine run all three thermal/convection products:

```bash
ML_EUMETSAT_THERMAL_PRODUCTS_ENABLED=true \
EUMETSAT_CONSUMER_KEY=... \
EUMETSAT_CONSUMER_SECRET=... \
python3 scripts/run_forecast_update_engine.py --enable-ml-eumetsat-thermal-products
```

Profile current local availability:

```bash
python3 scripts/ml_dataset/profile_data_availability.py
```

Backfill public WindsUp spot history:

```bash
python3 scripts/ml_dataset/collect_windsup_observations.py \
  --start-date 2026-06-22 \
  --end-date 2026-06-24 \
  --spot-id porticcio
```

The public pages expose day-level history with spot/date-dependent coverage.
Run broad backfills, then inspect the generated data availability profile to
identify exact gaps by station. Older downloadable archives may still require a
separate authenticated collector if the public pages do not expose enough depth
for a given spot.
