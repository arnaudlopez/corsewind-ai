# v_next Feature Signal Smoke Test

Generated: `2026-07-01`

## Purpose

The scientific error-floor audit showed that the current champion evaluation frame
does not contain the physical feature families that should plausibly help the
hardest regimes:

- previous Open-Meteo runs
- explicit thermal deltas
- low-level vertical profile derivatives
- static DEM/exposure proxies

This smoke test checks whether the current pipeline can now generate those
features and whether they show a useful signal on a strict temporal mini split.

## Data Built

New monthly shards on z2:

- `/srv/data/corsewind/ml_dataset/training_tables/residual_windsup_sst_prev_vnext_2025_12`
- `/srv/data/corsewind/ml_dataset/training_tables/residual_windsup_sst_prev_vnext_2026_01`

Rows:

- `2025-12`: `41,183`
- `2026-01`: `38,601`

Feature audit verdict: `pass`.

The v_next shards contain `1,767` columns and include:

- `features__previous_run_open_meteo_best_match_day1_*`
- `features__previous_run_open_meteo_best_match_day2_*`
- `features__thermal_*`
- `features__open_meteo_vertical_*`
- `features__spot_static_dem_sector_*`

The old comparable shard has about `749` selected feature columns in the baseline
training run, while v_next selected `1,474`.

## Strict Common-Key Comparison

Train:

- old prefix: `residual_windsup_sst_prev_2025_12`
- v_next prefix: `residual_windsup_sst_prev_vnext_2025_12`

Eval:

- common rows in `2026-01`
- common keys: `spot_id`, `issue_time_utc`, `target_time_utc`, `lead_time_minutes`
- leads: `15/30/45/60`
- common eval rows: `36,485`

| Model | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| raw NWP | `2.353` | `1.816` | `+0.538` |
| old feature set | `1.737` | `1.267` | `-0.366` |
| v_next feature set | `1.531` | `1.119` | `-0.081` |

Relative to the old feature set, v_next improves common-key RMSE by about
`11.9%`.

## By Lead

| Lead | Old RMSE | v_next RMSE | Gain |
| ---: | ---: | ---: | ---: |
| 15 min | `1.621` | `1.379` | `14.98%` |
| 30 min | `1.708` | `1.495` | `12.45%` |
| 45 min | `1.782` | `1.567` | `12.09%` |
| 60 min | `1.807` | `1.635` | `9.50%` |

## Strong-Wind Regimes

| Actual wind | Old RMSE | v_next RMSE | Notes |
| --- | ---: | ---: | --- |
| `>=12 kt` | `2.526` | `2.026` | v_next reduces underprediction bias |
| `>=15 kt` | `2.660` | `2.096` | meaningful gain |
| `>=20 kt` | `3.080` | `2.299` | large gain |
| `>=25 kt` | `3.561` | `2.557` | largest business-relevant gain |

This is the most important signal: v_next improves exactly the high-wind rows
that dominate the error budget.

## Hard Spots

| Spot | Old RMSE | v_next RMSE |
| --- | ---: | ---: |
| `la_tonnara` | `2.244` | `2.015` |
| `santa_manza` | `1.877` | `1.586` |
| `piantarella` | `1.694` | `1.349` |
| `balistra` | `1.616` | `1.360` |
| `porticcio` | `1.291` | `1.227` |

## Feature Importance Signal

Top v_next features include the expected families:

- `features__nwp_horizon_wind_ramp_ms`
- `baselines__baseline_wind_mean_ms`
- `features__model_error_now_wind_mean_ms`
- `features__open_meteo_vertical_geopotential_thickness_1000_850_m`
- `features__previous_run_open_meteo_best_match_day1_wind_u_10m`
- `features__previous_run_open_meteo_best_match_day1_wind_speed_10m`
- `features__open_meteo_vertical_relative_humidity_delta_1000_850_pct`

This suggests the new columns are not just present; the model is actually using
some of them.

## Decision

This is a positive signal, not a champion promotion.

We should now rebuild v_next over a wider temporal range and benchmark it against
the current champion protocol. The correct next test is:

- train on at least `2025-07` to `2025-12`
- evaluate on `2026-01` to `2026-06`
- compare common rows against the current champion and the old prefix
- include wind mean and gust targets
- report global RMSE, strong-wind RMSE, threshold precision/recall, and hard-spot metrics

If the multi-month result keeps even half of this smoke-test gain, v_next becomes
the most credible path we have found so far toward reducing the current error
floor.
