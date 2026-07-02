# Physical Signal Implementation Status

Date: 2026-06-28

This note records the first implementation pass after selecting the missing
physical signals.

## Implemented

### Open-Meteo Pressure-Level Defaults

`scripts/ml_dataset/collect_open_meteo_historical_forecast.py` now includes the
pressure-level variables by default:

- temperature at `1000/950/925/900/850 hPa`
- relative humidity at `1000/950/925/900/850 hPa`
- geopotential height at `1000/950/925/900/850 hPa`
- wind speed and wind direction at `1000/950/925/900/850 hPa`

The collector now checks requested hourly variables when deciding whether an
existing day is complete. This prevents old 24-row files without pressure-level
fields from being incorrectly skipped.

### Vertical Derived Features

The residual table already had the derivation code. The smoke test confirms it
now works once the feature store is rebuilt from Open-Meteo files containing
pressure-level fields.

Derived features validated:

- `open_meteo_vertical_geopotential_thickness_1000_850_m`
- `open_meteo_vertical_temperature_lapse_rate_1000_850_c_per_km`
- `open_meteo_vertical_temperature_delta_1000_850_c`
- `open_meteo_vertical_temperature_delta_1000_950_c`
- `open_meteo_vertical_temperature_delta_950_850_c`
- `open_meteo_vertical_relative_humidity_mean_1000_850_pct`
- `open_meteo_vertical_relative_humidity_delta_1000_850_pct`
- `open_meteo_vertical_wind_shear_speed_1000_850_ms`
- `open_meteo_vertical_wind_shear_direction_1000_850_deg`
- `open_meteo_vertical_low_level_inversion_strength_c`

### NWP Offset Points

New script:

`scripts/ml_dataset/generate_open_meteo_offset_registry.py`

It creates virtual Open-Meteo spots around each ML spot. Default offsets:

- `n10`: 10 km north
- `e10`: 10 km east
- `s10`: 10 km south
- `w10`: 10 km west

The feature store now consumes these virtual points when present and emits:

- `nwp_offset_<name>_*`
- `nwp_offset_<name>_delta_vs_center_*`
- `nwp_offset_gradient_east_west_*`
- `nwp_offset_gradient_north_south_*`
- `nwp_offset_gradient_pressure_msl_magnitude_hpa_per_km`
- `nwp_offset_gradient_pressure_msl_aligned_with_wind_hpa_per_km`

The residual table now keeps `nwp_offset_*` features.

### Static Spot Feature Injection

The feature store now accepts:

`--spot-static-features configs/ml_spot_static_features.json`

Any scalar feature in that file is emitted as `spot_static_*`, and the residual
table now keeps the `spot_static_` prefix.

This is the hook for the next static geography pass:

- fetch by wind sector
- open-sea exposure
- coastline/cross-shore angle
- relief blocking
- channeling/venturi score

## z2 Smoke Test

Smoke root:

`/srv/data/corsewind/ml_dataset/smoke_vertical_offset_2025_07_12`

Scope:

- date: `2025-07-12`
- spot offsets collected for Balistra only
- offsets: `n10/e10/s10/w10`
- leads: `15/30/45/60`

Result:

- training rows: `170`
- vertical features: `10/10` present, `170/170` non-null
- offset gradient feature keys checked: present
- offset gradients non-null on `26` rows, matching rows where Balistra offset
  context was available in the mini table

Smoke output excerpt:

```text
open_meteo_vertical_geopotential_thickness_1000_850_m 170
open_meteo_vertical_temperature_lapse_rate_1000_850_c_per_km 170
open_meteo_vertical_wind_shear_speed_1000_850_ms 170
nwp_offset_gradient_east_west_pressure_msl_per_km 26
nwp_offset_gradient_north_south_pressure_msl_per_km 26
nwp_offset_gradient_pressure_msl_aligned_with_wind_hpa_per_km 26
```

## Next Full Rebuild Command

Recommended next rebuild prefix:

`residual_windsup_sst_prev_phys_v1`

Command shape:

```bash
ssh z2 'cd /srv/data/corsewind/backfill_runner && \
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
/home/z2/corsewind-ml-smoke/.venv/bin/python \
scripts/ml_dataset/run_monthly_training_shards.py \
  --ml-root /srv/data/corsewind/ml_dataset \
  --registry configs/ml_spots.json \
  --context-registry configs/ml_context_stations.json \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --run-id-prefix residual_windsup_sst_prev_phys_v1 \
  --collect-open-meteo \
  --collect-open-meteo-offsets \
  --cleanup-jsonl-after-parquet \
  --continue-on-error'
```

This will:

1. ensure Open-Meteo files include pressure levels;
2. generate virtual offset spots;
3. collect NWP offset points;
4. rebuild feature stores;
5. rebuild residual training tables;
6. export monthly Parquet shards.

## Full Rebuild Launch

Launched on z2:

- prefix: `residual_windsup_sst_prev_phys_v1`
- range: `2024-01` to `2026-06`
- log: `/srv/data/corsewind/ml_dataset/run_logs/rebuild_phys_v1_2024_2026.log`
- status: `/srv/data/corsewind/ml_dataset/run_logs/rebuild_phys_v1_2024_2026.status`
- pid file: `/srv/data/corsewind/ml_dataset/run_logs/rebuild_phys_v1_2024_2026.pid`

Observed state after launch:

- active month: `2024-01`
- active step: `collect_open_meteo_historical_forecast.py`
- active first chunk: `2024-01-01` to `2024-01-08`
- memory available: about `14 GiB`

## 2026-06-28 - Phys V1 Rebuild Monitoring

Observed state around `12:57 Europe/Paris`:

- rebuild prefix: `residual_windsup_sst_prev_phys_v1`
- rebuild status: still running
- active month: `2024-01`
- active step: Open-Meteo historical forecast collection
- active chunk observed: `2024-01-22` to `2024-01-29`
- exported Parquet shards so far: `0`

Two z2 post-rebuild watchers are now armed:

- physical signal audit watcher:
  `/srv/data/corsewind/ml_dataset/run_logs/phys_v1_signal_audit_watcher.status`
- low-memory benchmark/calibration watcher:
  `/srv/data/corsewind/ml_dataset/run_logs/phys_v1_post_rebuild_lowmem.status`

Expected audit outputs once the rebuild finishes:

- `/srv/data/corsewind/ml_dataset/training_tables/phys_v1_required_feature_audit.json`
- `/srv/data/corsewind/ml_dataset/training_tables/phys_v1_required_feature_audit.md`
- `/srv/data/corsewind/ml_dataset/training_tables/phys_v1_signal_coverage.json`
- `/srv/data/corsewind/ml_dataset/training_tables/phys_v1_signal_coverage.md`

Expected benchmark run IDs:

- `tabular_lgbm_225k_prev_phys_v1_2024_2025_to_2026_v1`
- `tabular_lgbm_calbase_phys_v1_2024_to_2025h2_v1`
- `prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_phys_v1`

## Remaining Work

1. Run the full 2024-01 to 2026-06 rebuild on z2.
2. Audit coverage for:
   - `open_meteo_vertical_*`
   - `nwp_offset_gradient_*`
   - `spot_static_*`
3. Build the static coastline/DEM feature generator for real spot exposure.
4. Train/evaluate against the existing champion, with hard-regime metrics:
   La Tonnara, Santa Manza, `+45/+60`, actual wind `>= 8 m/s`.
