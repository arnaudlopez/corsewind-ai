# Rebuild Waiting Work Checklist

This tracks the useful work prepared while `phys_v1` rebuilds on z2.

## 1. Final `phys_v1` Evaluation

Status: done.

Implemented by:

- `scripts/ml_dataset/summarize_phys_v1_decision_report.py`
- `scripts/ml_dataset/z2_phys_v1_decision_report_watcher.sh`

Coverage:

- global RMSE/MAE
- comparison against champion `RMSE 1.268019 / MAE 0.930465`
- by lead/horizon
- worst spots
- worst spot/horizon pairs
- actual wind bins, including high wind
- local target-hour approximation
- thermal-signal quartiles when available
- feature/audit gates

## 2. Automatic Post-Benchmark Report

Status: done.

Outputs after artifacts are ready:

- `/srv/data/corsewind/ml_dataset/benchmarks/phys_v1_decision_report.json`
- `/srv/data/corsewind/ml_dataset/benchmarks/phys_v1_decision_report.md`

Decision categories:

- `target_achieved_candidate`
- `promote_candidate`
- `small_improvement`
- `not_improved`
- `incomplete`

## 3. Static Spot Features

Status: DEM v1 done, DEM exposure proxy done, true raster land/sea fetch prepared.

Implemented:

- `scripts/ml_dataset/generate_dem_spot_static_features.py`
- `configs/ml_spot_static_features.json`
- `docs/ml_nowcasting/dem_static_spot_features_v1.md`

Done:

- Copernicus GLO-30 DEM tiles
- radial relief stats
- 8-sector relief/barrier stats
- 8-sector lowland/sea share proxy
- 8-sector open-exposure proxy
- nearest 500 m mountain-barrier distance by sector
- nearest-land fallback
- Cap Corse full coverage after adding `N43 E009`

Prepared but not activated inside `phys_v1`:

- z2 staged file:
  `/srv/data/corsewind/backfill_runner/configs/ml_spot_static_features.dem_v1.json`
- next clean rebuild:
  `residual_windsup_sst_prev_phys_v2_dem`

Remaining static geography work:

- coastline orientation and cross-shore/alongshore angle
- explicit Venturi/channel scores
- distance to sea/mountain from vector geometry

## 4. Chronos / TimesFM / Moirai Benchmarks

Status: prepared, not launched.

Config:

- `configs/ml_sequence_benchmark_phys_v1.json`

Launcher:

- `scripts/ml_dataset/z2_launch_phys_v1_sequence_benchmarks.sh`

Guard:

- refuses to start until `phys_v1` decision report watcher is complete

Default prepared benchmark:

- Chronos-2 covariate
- Chronos-2 univariate cross-learning
- TimesFM
- Moirai disabled by default until Chronos/TimesFM sanity passes

## 5. Pipeline Status Script

Status: done.

Script:

- `scripts/ml_dataset/summarize_ml_pipeline_status.py`

Reports:

- rebuild/process status
- shard count
- row count when PyArrow is available
- disk free/used
- audit artifacts
- decision artifacts
- best calibration result found under benchmarks

Suggested z2 command:

```bash
ssh z2 'cd /srv/data/corsewind/backfill_runner && /home/z2/corsewind-ml-smoke/.venv/bin/python scripts/ml_dataset/summarize_ml_pipeline_status.py --ml-root /srv/data/corsewind/ml_dataset --disk-path /srv/data --output-json /srv/data/corsewind/ml_dataset/run_logs/ml_pipeline_status.json --output-md /srv/data/corsewind/ml_dataset/run_logs/ml_pipeline_status.md'
```

## 6. Land/Sea Fetch Static Features

Status: prepared and generated.

Implemented:

- `scripts/ml_dataset/generate_landsea_fetch_static_features.py`
- `configs/ml_spot_static_features.fetch_v1.json`
- `docs/ml_nowcasting/landsea_fetch_static_features_v1.md`

Source:

- ESA WorldCover 2021 v200, 10 m

Downloaded full Corsica tile set:

- `N39E006`
- `N39E009`
- `N42E006`
- `N42E009`

Prepared rebuild:

- `residual_windsup_sst_prev_phys_v3_dem_fetch`

Prepared z2 scripts:

- `scripts/ml_dataset/z2_launch_phys_v3_dem_fetch_rebuild.sh`
- `scripts/ml_dataset/z2_phys_v3_dem_fetch_signal_audit_watcher.sh`

## 7. Temporal Integrity / Leakage Gate

Status: prepared and watcher launched on z2.

Implemented:

- `scripts/ml_dataset/audit_training_table_temporal_integrity.py`
- `scripts/ml_dataset/z2_phys_v1_post_backfill_quality_watcher.sh`

Checks:

- expected monthly shards exist;
- `issue_time_utc + lead_time_minutes = target_time_utc`;
- `target_time_utc` is never before `issue_time_utc`;
- target observation timestamp is close to target time;
- duplicate `(spot_id, issue_time_utc, target_time_utc, lead_time_minutes)` rows;
- negative feature age values, which would imply future data leakage;
- very stale observation-age features, kept as warnings rather than failures;
- suspicious feature names that look like labels or future truth inside model inputs.

Smoke result on `2026-01`:

- verdict: `warn`;
- failures: none;
- warning source: stale `obs_lag_15m/60m_age_minutes` values when recent station observations are unavailable;
- lead alignment: `0` mismatches;
- target observation max distance: `7.45 min`.

Final outputs after full backfill:

- `/srv/data/corsewind/ml_dataset/training_tables/phys_v1_temporal_integrity_audit.json`
- `/srv/data/corsewind/ml_dataset/training_tables/phys_v1_temporal_integrity_audit.md`
- refreshed `/srv/data/corsewind/ml_dataset/run_logs/ml_pipeline_status.md`
