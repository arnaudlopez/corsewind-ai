# Scientific Error Diagnostic Notes

## 2026-06-28T09:15+02:00 - scope

Goal:
  Build a scientific diagnostic of the remaining wind-mean error, not another
  blind model benchmark.

Champion reference:
  run_id:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
  prediction artifact:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/calibrated_predictions_2026.parquet
  validated score:
    RMSE: 1.268019
    MAE: 0.930465
    bias: 0.018767
    count: 31429

Scientific questions:
  1. Which spots, horizons, hours, months, and wind regimes dominate SSE?
  2. Is the error mostly bias, variance, timing, or sparse-source instability?
  3. Are there deployable route/selector opportunities, or only non-deployable
     oracle gains?
  4. Which missing data families are plausibly blocking progress toward RMSE 0.9?
  5. What is the next experiment with the best expected value?

Planned diagnostics:
  - SSE contribution by spot, lead, spot+lead, hour, month, wind bin.
  - Tail analysis: how much RMSE is carried by the worst 5/10/20% rows.
  - Raw-vs-first-stage-vs-second-stage comparison.
  - Bias and asymmetric failure modes by spot/horizon.
  - Existing-model oracle and routeability analysis.
  - Feature/source coverage around high-error regimes.

## 2026-06-28T09:35+02:00 - diagnostics executed

Remote output directory:
  /srv/data/corsewind/ml_dataset/benchmarks/scientific_error_diagnostic_v1

Local copied artifacts:
  docs/ml_nowcasting/scientific_error_diagnostic_v1/scientific_error_diagnostic.md
  docs/ml_nowcasting/scientific_error_diagnostic_v1/gap_oracles_full_common.md
  docs/ml_nowcasting/scientific_error_diagnostic_v1/feature_family_coverage_hard_regimes.md

Scripts used:
  scripts/ml_dataset/scientific_error_diagnostic.py
  scripts/ml_dataset/analyze_rmse09_gap_oracles.py
  scripts/ml_dataset/audit_feature_family_coverage.py

Main findings:
  - Raw model prior RMSE is 2.187306.
  - First-stage correction RMSE is 1.276846.
  - Final champion RMSE is 1.268019.
  - Raw to first-stage gain is very large, but second-stage calibration only
    adds 0.691% RMSE gain.
  - Reaching RMSE 0.9 requires a 49.623% MSE reduction.
  - The worst 7.452% rows contain enough excess SSE to explain the full
    remaining gap to 0.9.
  - Top 5% rows carry 40.993% of SSE; top 10% carry 56.654%.

Dominant error regimes:
  - La Tonnara and Santa Manza dominate the spot error budget.
  - Lead +45/+60 min dominates the horizon error budget.
  - Actual wind >= 8 m/s carries about 31.99% of SSE on only 16.73% rows.
  - Very light wind 0-2 m/s also carries 19.844% of SSE because the model
    tends to overpredict light wind.
  - Strong wind is underpredicted: actual 10+ m/s has bias -0.777700.
  - Very light wind is overpredicted: actual 0-2 m/s has bias +0.460936.

Routeability/oracle:
  - Existing candidate models contain some complementary information.
  - A target-leaky row-wise oracle across the full common candidate set reaches
    RMSE 1.187225, not 0.9.
  - Therefore a deployable selector/router is useful, but cannot by itself
    close the target gap with the current feature/model set.

Feature/source gaps:
  - Vertical profiles are absent from the champion prediction artifact.
  - Previous-runs forecast features are absent from the champion prediction
    artifact.
  - Land-sea, air-sea, and land-air thermal deltas are absent.
  - Upwind station aggregates are absent.
  - Coastal/inland and coastal/relief pressure and temperature deltas are
    absent as explicit features.
  - EUMETSAT LST and instability have schema/availability flags but low mean
    value coverage in the champion artifact.

Working interpretation:
  We have already corrected broad AROME/NWP bias. The remaining problem is
  physical-regime identification: thermal onset/decay, high-energy wind, and
  local spot effects on La Tonnara/Santa Manza at +45/+60 min. More generic
  model tweaking is unlikely to reach RMSE 0.9 without adding the missing
  physical signals or substantially more reliable labels for those regimes.

## 2026-06-28T10:05+02:00 - selected missing physical signals

Decision document:
  docs/ml_nowcasting/physical_signal_selection.md

Selected P0 families:
  1. Low-level vertical stability and mixing.
  2. Land-sea and air-sea thermal contrast.
  3. Coast-inland-relief pressure and temperature gradients.
  4. Upwind station propagation and mountain/coastal exchange.
  5. Wind-direction conditional spot exposure.
  6. Forecast evolution and run-to-run instability.

Rationale:
  These are the signals most directly tied to the measured failures:
  La Tonnara/Santa Manza, +45/+60 min, high-wind underprediction, light-wind
  overprediction, and thermal onset/decay. They have better expected value than
  adding another generic model on the current table.

## 2026-06-28T10:25+02:00 - physical signal availability audit

Decision document:
  docs/ml_nowcasting/physical_signal_availability_matrix.md

Audited data:
  z2:
    /srv/data/corsewind/ml_dataset/training_tables/residual_windsup_sst_prev_regime_v1_YYYY_MM/training_rows.parquet
  months:
    2024-01 .. 2026-06
  rows:
    1,528,776

Key result:
  - Thermal air-SST and inland/relief temperature deltas are already present
    with about 99%+ coverage.
  - Upwind features are already present; all/coastal coverage is strong, while
    inland/relief upwind coverage is about 45-46%.
  - Forecast horizon ramps and previous Open-Meteo runs are already present
    with about 98-99% coverage.
  - Pressure-gradient features exist in schema but the station-based pressure
    deltas are effectively empty in current regime_v1 tables.
  - EUMETSAT LST values are effectively absent historically: 0.109% coverage.
  - Vertical derived features are absent from the training table.
  - Static spot exposure/fetch/relief-blocking features are absent.

Important source finding:
  Open-Meteo raw historical forecast files already contain pressure-level fields
  for meteofrance_arome_france: 904/904 audited daily files had
  temperature_1000hPa and sampled files also had humidity, wind and geopotential
  levels. The next vertical-profile step is therefore mostly integration and
  feature derivation, not a new provider search.

## 2026-06-28T10:45+02:00 - first physical signal implementation

Implementation status:
  docs/ml_nowcasting/physical_signal_implementation_status.md

Code changes:
  - collect_open_meteo_historical_forecast.py:
      pressure-level variables are now part of the default hourly set;
      existing-file completeness now checks requested variables.
  - generate_open_meteo_offset_registry.py:
      new script generating virtual NWP offset spots.
  - build_spot_feature_store.py:
      consumes n/e/s/w Open-Meteo offset points and derives NWP gradients;
      injects optional spot_static_* features.
  - build_residual_training_table.py:
      keeps nwp_offset_* and spot_static_* features.
  - run_training_backfill_pipeline.py and run_monthly_training_shards.py:
      can generate/collect Open-Meteo offset points in monthly rebuilds.

Smoke test:
  z2 path:
    /srv/data/corsewind/ml_dataset/smoke_vertical_offset_2025_07_12
  scope:
    Balistra offsets on 2025-07-12, leads 15/30/45/60.
  result:
    170 training rows;
    10/10 open_meteo_vertical_* features non-null on all rows;
    nwp_offset_gradient_* features present and non-null on 26 Balistra-related
    rows after collecting the four offset points.

Decision:
  The next logical step is a full 2024-01..2026-06 monthly rebuild under a new
  prefix, then a coverage audit and a model benchmark against the champion.

## 2026-06-28T11:00+02:00 - full physical-signal rebuild launched

Remote run:
  prefix:
    residual_windsup_sst_prev_phys_v1
  range:
    2024-01 .. 2026-06
  log:
    /srv/data/corsewind/ml_dataset/run_logs/rebuild_phys_v1_2024_2026.log
  status:
    /srv/data/corsewind/ml_dataset/run_logs/rebuild_phys_v1_2024_2026.status
  pid:
    /srv/data/corsewind/ml_dataset/run_logs/rebuild_phys_v1_2024_2026.pid

Initial state:
  z2 memory available: about 14 GiB.
  active month: 2024-01.
  active step: collect_open_meteo_historical_forecast.py for 2024-01-01..2024-01-08.

## 2026-06-28T12:57+02:00 - monitoring armed for phys_v1

Current remote state:
  rebuild prefix:
    residual_windsup_sst_prev_phys_v1
  status:
    still running.
  active month:
    2024-01.
  active step:
    collect_open_meteo_historical_forecast.py.
  active chunk observed:
    2024-01-22..2024-01-29, including offset registry collection.
  completed Parquet shards:
    0 so far.

Watchers launched:
  physical signal audit:
    /srv/data/corsewind/ml_dataset/run_logs/phys_v1_signal_audit_watcher.status
    /srv/data/corsewind/ml_dataset/run_logs/phys_v1_signal_audit_watcher.outer.log
  low-memory model benchmark:
    /srv/data/corsewind/ml_dataset/run_logs/phys_v1_post_rebuild_lowmem.status
    /srv/data/corsewind/ml_dataset/run_logs/phys_v1_post_rebuild_lowmem.outer.log

The audit watcher will validate the required physical signals after rebuild:
  - vertical thickness/lapse/shear
  - offset pressure gradients east-west/north-south/aligned with wind
  - air-SST thermal delta
  - upwind context wind aggregate

The benchmark watcher will train the low-memory LightGBM baseline, the
calibration base, and the ExtraTrees residual calibrator under the phys_v1
suffix, then write comparison artifacts for deciding whether the new physical
signals improve the champion.

## 2026-06-28T16:32+02:00 - phys_v1 rebuild checkpoint

Remote state:
  rebuild prefix:
    residual_windsup_sst_prev_phys_v1
  status:
    still running.
  elapsed:
    about 4h10.
  active month:
    2024-05.
  active step:
    collect_open_meteo_historical_forecast.py for offset spots.
  active chunk observed:
    2024-05-15..2024-05-22.

Completed monthly Parquet shards:
  - 2024-01: 42,551 rows, 17.6 MiB
  - 2024-02: 39,667 rows, 17.2 MiB
  - 2024-03: 42,962 rows, 18.1 MiB
  - 2024-04: 40,819 rows, 17.6 MiB

Total completed so far:
  165,999 rows across 4 monthly shards.

Disk state:
  /srv/data has about 218 GiB free.

Watchers:
  physical signal audit watcher still polling every 2 minutes.
  low-memory benchmark watcher still polling every 2 minutes.

No final phys_v1 audit or benchmark artifact yet because the full 2024-01..2026-06
rebuild is still in progress.

## 2026-06-28T22:09+02:00 - phys_v1 ETA checkpoint

Remote state:
  rebuild prefix:
    residual_windsup_sst_prev_phys_v1
  status:
    still running.
  elapsed:
    about 9h47 since launch.
  completed monthly shards:
    12 months, 2024-01..2024-12.
  active month:
    2025-01.
  active step:
    collect_open_meteo_historical_forecast.py for offset spots.
  active chunk observed:
    2025-01-22..2025-01-29.

Timing:
  first completed shard:
    2024-01 at 13:19:41.
  latest completed shard:
    2024-12 at 21:39:29.
  observed completion cadence:
    about 45 minutes per completed month after the first shard.

Estimate:
  remaining range after active 2025-01:
    2025-02..2026-06, 17 months.
  likely remaining wall time:
    about 13 to 15 hours including the rest of 2025-01 and variability.
  expected rebuild finish window:
    Monday 2026-06-29 around 11:00..13:30 Europe/Paris.

Post-rebuild work:
  The physical signal audit and low-memory benchmark watchers will start only
  after the rebuild process exits. Their runtime is extra and should be counted
  separately from the dataset rebuild ETA.

## 2026-06-28T22:26+02:00 - phys_v1 decision report prepared

Added a lightweight post-benchmark decision layer:
  script:
    scripts/ml_dataset/summarize_phys_v1_decision_report.py
  watcher:
    scripts/ml_dataset/z2_phys_v1_decision_report_watcher.sh
  plan:
    docs/ml_nowcasting/phys_v1_post_benchmark_decision_plan.md

Purpose:
  Once the already-running rebuild, feature audit and low-memory benchmark have
  produced their artifacts, generate a single decision report comparing phys_v1
  against the current champion RMSE 1.268019 / MAE 0.930465.

Decision categories:
  - target_achieved_candidate
  - promote_candidate
  - small_improvement
  - not_improved
  - incomplete

The report will include global RMSE/MAE, feature/audit gates, physical signal
coverage, by-lead metrics, worst spots, worst spot-leads, high-wind bins, local
hour bins and thermal-signal bins when the prediction parquet contains the
relevant physical columns.

## 2026-06-28T22:45+02:00 - DEM static features prepared

User asked to skip GPS-only proxy features and go directly to the available
30 m DEM.

Implemented:
  script:
    scripts/ml_dataset/generate_dem_spot_static_features.py
  output:
    configs/ml_spot_static_features.json
  documentation:
    docs/ml_nowcasting/dem_static_spot_features_v1.md
  dependency:
    added rasterio>=1.4 to requirements-ml-dataset.txt

DEM source:
  Copernicus GLO-30 tiles in data/raw/dem/copernicus_glo30.

Generated features:
  25 spots, 193 unique static feature names, roughly 190 features for normal
  coastal/land spots.

Feature families:
  - spot/reference elevation
  - nearest-land elevation/distance for coastal/offshore edge cases
  - radial elevation/relief stats at 1/2/5/10/20 km
  - 8-sector 20 km elevation/relief stats
  - sector barrier max/p90/share
  - nearest sector barrier distance
  - cross-sector relief gradients

Coverage note:
  N43 E009 was downloaded after DNS/network recovery, so Cap Corse now has
  complete radial/sector sampling in the configured 20 km DEM window.
  Ajaccio buoy has no DEM reference because it is offshore.

Operational decision:
  Do not copy configs/ml_spot_static_features.json to z2 under the default name
  while phys_v1 is running. Otherwise late monthly shards would contain
  spot_static_* while earlier phys_v1 shards would not. Keep phys_v1 coherent
  and use DEM static features in a clean phys_v2_dem rebuild.

## 2026-06-28T22:55+02:00 - phys_v2_dem rail prepared

Prepared explicit `--spot-static-features` propagation:
  - run_training_backfill_pipeline.py now passes the selected static feature
    file to build_spot_feature_store.py.
  - run_monthly_training_shards.py now accepts and forwards
    --spot-static-features.

Prepared z2 scripts:
  - scripts/ml_dataset/z2_launch_phys_v2_dem_rebuild.sh
  - scripts/ml_dataset/z2_phys_v2_dem_signal_audit_watcher.sh

Prepared documentation:
  - docs/ml_nowcasting/phys_v2_dem_rebuild_plan.md

Default safety:
  z2_launch_phys_v2_dem_rebuild.sh refuses to start unless the phys_v1 decision
  report watcher is complete, unless REQUIRE_PHYS_V1_DONE=0 is explicitly set.

Important:
  The DEM JSON remains staged as configs/ml_spot_static_features.dem_v1.json on
  z2. It is not activated as configs/ml_spot_static_features.json, so the
  running phys_v1 rebuild remains coherent.
