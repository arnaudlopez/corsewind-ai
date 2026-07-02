# RMSE 0.9 Feature Gap Audit

Generated: 2026-06-27

## Why This Audit Exists

The current best model is stuck around RMSE `1.269403`, and both shallow
ensembles and third-stage hard-regime specialists failed to generalize to 2026.
The next question is therefore not "which model next?", but "which physical
signals are actually present in the training/evaluation tables?"

This audit compares:

- current best:
  `prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1`
- newer top700/post-relief chain:
  `prediction_residual_calibrator_top700_2025h2_to_2026_extratrees_autoscale_v1`

Hard rows use the same business-critical mask used in the RMSE gap work:

- spots: `la_tonnara`, `santa_manza`, `balistra`, `cap_corse`, `la_parata`
- OR `lead_time_minutes >= 45`

## Current Best Coverage

Artifact:

`/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1/feature_family_coverage_audit.json`

Rows:

- total rows: `31,429`
- hard rows: `23,748`
- RMSE: `1.269403`
- hard RMSE: `1.335017`

Present:

- SST: present, strong coverage
- EUMETSAT cloud type/mask: present, but many numeric product columns sparse
- EUMETSAT instability indices: present, but very sparse product columns
- radiation / shortwave baseline: present
- pressure and coastal/inland/relief station context: present
- recent observation tendencies: present

Missing in current best:

- `thermal_land_minus_sst_c`
- `thermal_air_minus_sst_c`
- `thermal_land_minus_air_c`
- upwind-weighted station aggregates
- explicit coastal-inland and coastal-relief thermal/pressure deltas
- all vertical AROME profile features

Important coverage notes:

- land-surface-temperature family has `100%` row-level "any" coverage mostly
  because availability flags exist, but mean numeric column coverage is only
  about `5%`
- instability family is similar: row-level availability flags exist, but mean
  numeric coverage is only about `2.8%`
- vertical profile coverage is exactly `0%`

## Top700/Post-Relief Coverage

Artifact:

`/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_top700_2025h2_to_2026_extratrees_autoscale_v1/feature_family_coverage_audit.json`

Rows:

- total rows: `59,641`
- hard rows: `44,908`
- RMSE: `1.316977`
- hard RMSE: `1.375580`

Newly present versus current best:

- `thermal_land_minus_sst_c`
- `thermal_air_minus_sst_c`
- `thermal_land_minus_air_c`
- `thermal_inland_minus_coastal_temperature_c`
- `thermal_relief_minus_coastal_temperature_c`
- `thermal_inland_minus_coastal_pressure_hpa`
- `thermal_relief_minus_coastal_pressure_hpa`
- upwind-weighted station aggregates
- previous-run Open-Meteo features
- more pressure, relief, radiation, and thermal derived fields

Still missing:

- vertical temperature profile
- vertical humidity profile
- vertical motion profile
- geopotential thickness / lower-troposphere thickness

Top700 has the right direction of feature work, but the score worsens. This
means the added features are not yet enough, not stable enough, or the evaluation
row mix/model setup changed enough to offset their signal.

## Vertical Profile Status

The code path exists:

- collector:
  `scripts/ml_dataset/collect_meteo_france_vertical_profiles.py`
- feature-store integration:
  `build_spot_feature_store.py` writes `vertical_arome_*`
- collected variables:
  `temperature_c`, `relative_humidity_pct`,
  `vertical_velocity_pressure_pa_s`, `geopotential_height_m`,
  `pseudo_adiabatic_potential_temperature_c`
- derived variables:
  `geopotential_thickness_1000_850_m`,
  `low_level_inversion_strength_c`,
  `relative_humidity_mean_1000_850_pct`,
  `temperature_lapse_rate_1000_850_c_per_km`,
  `vertical_velocity_pressure_850_pa_s`

But available rows are currently too limited:

- NWP vertical profile rows: `725`
- spots: `25`
- time range: `2026-06-24T09:00:00Z -> 2026-06-26T22:00:00Z`
- pressure levels: `1000`, `950`, `925`, `900`, `850`

This is operationally useful going forward, but it is not enough to train or
validate the 2025 -> 2026 RMSE target.

## Data Source Status

Available useful historical sources:

- Copernicus SST:
  - rows: `204,150`
  - time range: `2024-01-01 -> 2026-06-24`
  - spot count: `25`
- Météo-France observations:
  - rows: `4,291,065`
- Meteonet normalized ground stations:
  - rows: `9,934,994`
- training tables:
  - total rows: `5,834,394`

Recent-only sources:

- EUMETSAT Cloud Type:
  - rows: `3,825`
  - time range starts `2026-06-22`
- EUMETSAT Land Surface Temperature:
  - rows: `3,695`
  - time range starts `2026-06-22`
- EUMETSAT Global Instability Indices:
  - rows: `3,775`
  - time range starts `2026-06-22`
- AROME vertical profiles:
  - rows: `725`
  - time range starts `2026-06-24`

These recent-only sources explain why physically promising features exist in
newer tables but cannot yet drive a robust 2025 -> 2026 model.

## Decision

For RMSE `0.9`, the next data work should be:

1. Backfill Open-Meteo Historical Forecast pressure-level fields for
   `meteofrance_arome_france` from `2024-01-02` to `2026-06-23` for pressure
   levels `1000/950/925/900/850`, then rebuild feature store and training
   tables. A z2 smoke test for Balistra on `2025-06-15` returned `24/24`
   non-null hourly rows for temperature, relative humidity, geopotential height,
   wind speed and wind direction at those levels.

2. If historical EUMETSAT LST/cloud/instability cannot be backfilled, treat them
   as forward-collection features only and do not rely on them for the locked
   2025 -> 2026 score.

3. Prioritize feature families with enough history:
   - Copernicus SST
   - Météo-France observations
   - Meteonet/nearby/upwind stations
   - Open-Meteo previous runs
   - AROME current/forecast scalar fields

4. Use vertical profile data as the P0 missing physical source. Native
   Météo-France WCS profiles remain the preferred forward/recent source, but
   Open-Meteo pressure levels are now the practical historical backfill route.
   Without this family, the remaining hard-regime errors are likely
   underdetermined by the existing table.
