# Residual Correction Training Table

This table is the first concrete training surface for the CorseWind.ai
nowcasting model.

It follows the scientific strategy selected for the project:

```text
NWP forecast at target horizon
+ issue-time observations
+ recent model error
+ nearby/coastal/inland/relief context stations
+ SST, land heating, cloud, instability context
-> local residual correction and windsurf threshold probabilities
```

## Current Pilot

- Remote root: `/srv/data/corsewind/ml_dataset/training_tables/residual_correction_pilot_20260622`
- Rows: `168`
- Spots: `7`
- Source feature-store rows: `63`
- Horizons currently materialized: `+60m`, `+120m`, `+180m`, `+360m`
- Missing baseline wind rows: `0`
- Missing target wind rows: `0`

The pilot source feature store is hourly. Once the feature store is produced at
the true 15-minute cadence, the same builder can emit `+15m`, `+30m`, and
`+45m` rows as well.

## Row Identity

Each row is one supervised example:

```text
spot_id + issue_time_utc + target_time_utc + lead_time_minutes
```

The issue time is the moment when we would make the forecast. The target time is
the future observation we want to predict.

## Feature Groups

### `features`

These are copied only from the issue-time feature-store row:

- `obs_*`: last local observation, lags, deltas, freshness.
- `context_nearest_*`: nearest context stations.
- `context_coastal_*`: closest coastal station context.
- `context_inland_*`: inland thermal station context.
- `context_relief_*`: mountain/relief station context.
- `context_global_*`: wider context stations around the spot.
- `context_agg_*`: aggregated gradients and station summaries.
- `context_*_bearing_*`, `context_*_east_offset_km`,
  `context_*_north_offset_km`, `context_*_altitude_delta_m`,
  `context_*_upwind_score_from_target_wind`: neighboring-station geometry and
  upwind alignment computed from pre-target information only.
- `sst_*`: sampled Copernicus sea-surface temperature.
- `eumetsat_*`: sampled cloud type, land-surface temperature, instability.
- `model_open_meteo_meteofrance_arome_france_*`: NWP state at issue time.
- `model_error_now_wind_mean_ms`: current observed wind minus current NWP wind.
- `model_error_now_gust_ms`: current observed gust minus current NWP gust.
- `lead_time_minutes`, `issue_hour_*`, `issue_dayofyear_*`.

Timestamp fields from satellite/model products are intentionally excluded from
the training feature dict to reduce leakage risk and keep age/freshness fields
as the explicit timing signal.

### `baselines`

These are the NWP forecast values for the target horizon:

- `baseline_wind_mean_ms`
- `baseline_gust_ms`
- `baseline_wind_direction_deg`
- `baseline_temperature_2m_c`
- `baseline_pressure_msl_hpa`
- `baseline_surface_pressure_hpa`
- `baseline_shortwave_radiation`
- `baseline_cloud_cover_pct`
- `baseline_cape`

The model should learn corrections around these values, not replace them.

### `labels`

These are the supervised targets:

- `target_wind_mean_ms`
- `target_gust_ms`
- `target_wind_direction_deg`
- `residual_wind_mean_ms`
- `residual_gust_ms`
- `target_wind_gt_15kt`
- `target_wind_gt_20kt`
- `target_gust_gt_20kt`
- `target_gust_gt_25kt`

The first regression target is the residual:

```text
observed target wind - NWP target wind
```

The threshold labels are for probabilistic windsurf decision metrics.

## Outputs

```text
training_rows.jsonl
training_profile.json
training_columns.csv
```

`training_rows.jsonl` keeps nested groups so it is easy to audit. A downstream
trainer can flatten `features.*`, `baselines.*`, and selected categorical
columns into a matrix.
