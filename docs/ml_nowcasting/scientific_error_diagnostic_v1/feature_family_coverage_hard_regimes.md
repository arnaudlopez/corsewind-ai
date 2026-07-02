# Feature Family Coverage Audit

Generated: `2026-06-28T09:31:37.386176Z`
Prediction file: `/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/calibrated_predictions_2026.parquet`
Rows: `31429`
Hard rows: `22411`
Prediction RMSE: `1.268019`
Hard RMSE: `1.362068`

## Required Concepts

| Concept | Present | Columns | Examples |
| --- | --- | ---: | --- |
| `sea_surface_temperature` | `True` | 2 | `features__sst_c`, `features__sst_k` |
| `land_sea_delta` | `False` | 0 |  |
| `air_sea_delta` | `False` | 0 |  |
| `land_air_delta` | `False` | 0 |  |
| `shortwave_ramp` | `True` | 2 | `baselines__baseline_shortwave_radiation`, `features__model_open_meteo_meteofrance_arome_france_shortwave_radiation` |
| `cloud_type` | `True` | 45 | `features__eumetsat_cloud_type_age_minutes`, `features__eumetsat_cloud_type_available`, `features__eumetsat_cloud_type_cloud_phase`, `features__eumetsat_cloud_type_cloud_phase_neighborhood_max` |
| `cloud_mask` | `True` | 1 | `features__eumetsat_cloud_mask_available` |
| `instability_indices` | `True` | 45 | `features__eumetsat_global_instability_indices_age_minutes`, `features__eumetsat_global_instability_indices_available`, `features__eumetsat_global_instability_indices_k_index`, `features__eumetsat_global_instability_indices_k_index_neighborhood_max` |
| `upwind_station_aggregates` | `False` | 0 |  |
| `coastal_inland_temperature_delta` | `False` | 0 |  |
| `coastal_relief_temperature_delta` | `False` | 0 |  |
| `coastal_inland_pressure_delta` | `False` | 0 |  |
| `coastal_relief_pressure_delta` | `False` | 0 |  |
| `recent_temperature_tendency` | `True` | 1 | `features__obs_delta_60m_temperature_c` |
| `recent_pressure_tendency` | `True` | 1 | `features__obs_delta_60m_pressure_hpa` |
| `vertical_temperature_profile` | `False` | 0 |  |
| `vertical_humidity_profile` | `False` | 0 |  |
| `vertical_motion_profile` | `False` | 0 |  |
| `geopotential_thickness` | `False` | 0 |  |

## Family Coverage

| Family | Columns | Any coverage | Mean coverage | >=90% columns | Hard any coverage | Hard mean coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sst` | 5 | 100.0% | 99.956% | 5 | 100.0% | 99.968% |
| `land_surface_temperature` | 22 | 100.0% | 5.031% | 1 | 100.0% | 5.036% |
| `thermal_derived` | 0 | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| `cloud` | 56 | 100.0% | 12.904% | 7 | 100.0% | 12.906% |
| `instability` | 45 | 100.0% | 2.818% | 1 | 100.0% | 2.821% |
| `radiation` | 3 | 100.0% | 73.418% | 2 | 100.0% | 71.437% |
| `surface_pressure` | 57 | 100.0% | 53.998% | 25 | 100.0% | 54.921% |
| `vertical_profile` | 0 | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| `upwind` | 0 | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| `coastal_inland_relief` | 217 | 100.0% | 62.412% | 120 | 100.0% | 61.778% |
| `recent_obs_trends` | 17 | 100.0% | 56.914% | 9 | 100.0% | 58.532% |
| `previous_runs` | 0 | 0.0% | 0.0% | 0 | 0.0% | 0.0% |

## Missing Concepts

- `land_sea_delta`
- `air_sea_delta`
- `land_air_delta`
- `upwind_station_aggregates`
- `coastal_inland_temperature_delta`
- `coastal_relief_temperature_delta`
- `coastal_inland_pressure_delta`
- `coastal_relief_pressure_delta`
- `vertical_temperature_profile`
- `vertical_humidity_profile`
- `vertical_motion_profile`
- `geopotential_thickness`
