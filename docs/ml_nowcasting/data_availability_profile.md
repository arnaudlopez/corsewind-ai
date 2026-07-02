# Data Availability Profile

Generated at: `2026-06-25T07:49:42.181084Z`

## Summary

| Source | Status | Rows / coverage | Notes |
| --- | --- | ---: | --- |
| Spot registry | OK | 25 spots | {'False': 5, 'True': 20} |
| Beacon Live snapshot | missing | 0 obs | live fields:  |
| Model samples | missing | 0 rows | sampled model forecasts at spots |
| NWP extra fields | missing | 0 rows | AROME/AROME-PI thermal/context fields at spots |
| NWP vertical profiles | OK | 100 rows | AROME 0.025 isobaric profiles at spots |
| Feature store 15 min | OK | 4976 rows | canonical training rows by spot/time |
| Residual training tables | OK | 24184 rows | NWP baseline + issue-time features + residual labels |
| Trained residual models | OK | 2 runs | latest: residual_backfill_2024_06 |
| In-situ observations | OK | 3948348 rows | normalized Meteo-France + WindsUp spot obs |
| MeteoNet ground stations | OK | 9934994 rows | 6-minute Corsica station observations for pretraining |
| Copernicus Marine SST | OK | 1260 rows | sampled sea-surface temperature at spots |
| EUMETSAT Cloud Mask | missing | 0 rows | sampled MTG cloud mask at spots |
| EUMETSAT Cloud Type | OK | 1300 rows | sampled variables: cloud_type, quality_overall_processing, cloud_phase, quality_illumination, quality_nwp_parameters, quality_MTG_parameters |
| EUMETSAT Land Surface Temperature | OK | 1220 rows | sampled variables: QFLAGS, LST, LST_uncertainty |
| EUMETSAT Global Instability Indices | OK | 1300 rows | sampled variables: lifted_index, k_index, prec_water_low, prec_water_mid, prec_water_high, prec_water_total |
| Copernicus Marine inventory | missing | 0 candidates | {} |
| EUMETSAT inventory | missing | 0 candidates | {} |
| EUMETSAT catalogue keyword scan | missing | 0 matches | {} |
| Meteo-France WCS inventory | missing | 0 services | model variables available beyond wind |

## In-Situ Observation Fields

### `dpclim_station_6min`

- rows: `1320240`
- time range: `2024-01-02T00:00:00Z` -> `2026-06-23T23:54:00Z`
- sources: `7`, mapped spots: `7`

| Field | Non-null | Coverage |
| --- | ---: | ---: |

| Spot | Rows | Time range | Source ids |
| --- | ---: | --- | --- |
| `cap_corse` | 37200 | `2024-04-16T00:00:00Z` -> `2026-06-23T23:54:00Z` | `20107001` |
| `la_parata` | 216960 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:54:00Z` | `20004003` |
| `lfkf` | 216960 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:54:00Z` | `20114002` |
| `lfkj` | 216960 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:54:00Z` | `20004002` |
| `lfks` | 216960 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:54:00Z` | `20342001` |
| `lfvf` | 216960 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:54:00Z` | `20093002` |
| `lfvh` | 198240 | `2024-01-02T00:00:00Z` -> `2026-04-27T23:54:00Z` | `20041001` |

### `dpclim_station_hourly`

- rows: `678528`
- time range: `2024-01-02T00:00:00Z` -> `2026-06-23T23:00:00Z`
- sources: `33`, mapped spots: `7`

| Field | Non-null | Coverage |
| --- | ---: | ---: |
| `cloud_cover_octa` | 68809 | 10% |
| `dewpoint_c` | 600593 | 89% |
| `direct_radiation_j_cm2` | 21287 | 3% |
| `global_radiation_j_cm2` | 86544 | 13% |
| `gust_instant_ms` | 644627 | 95% |
| `gust_max_ms` | 600175 | 88% |
| `humidity_pct` | 601728 | 89% |
| `low_cloud_cover_octa` | 142813 | 21% |
| `precipitation_1h_mm` | 619959 | 91% |
| `pressure_station_hpa` | 193582 | 29% |
| `sea_level_pressure_hpa` | 191297 | 28% |
| `sunshine_duration_minutes` | 64457 | 9% |
| `temperature_c` | 665963 | 98% |
| `visibility_m` | 142929 | 21% |
| `wind_direction_deg` | 640958 | 94% |
| `wind_mean_ms` | 644456 | 95% |

| Spot | Rows | Time range | Source ids |
| --- | ---: | --- | --- |
| `cap_corse` | 21696 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:00:00Z` | `20107001` |
| `la_parata` | 21696 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:00:00Z` | `20004003` |
| `lfkf` | 21696 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:00:00Z` | `20114002` |
| `lfkj` | 21696 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:00:00Z` | `20004002` |
| `lfks` | 21696 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:00:00Z` | `20342001` |
| `lfvf` | 21696 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:00:00Z` | `20093002` |
| `lfvh` | 21696 | `2024-01-02T00:00:00Z` -> `2026-06-23T23:00:00Z` | `20041001` |

### `windsup_public_spot_history`

- rows: `1949580`
- time range: `2023-12-31T23:08:00Z` -> `2026-06-24T21:59:55Z`
- sources: `7`, mapped spots: `7`

| Field | Non-null | Coverage |
| --- | ---: | ---: |
| `gust_kt_raw` | 1949580 | 100% |
| `gust_ms` | 1949580 | 100% |
| `wind_direction_deg` | 1949580 | 100% |
| `wind_mean_kt_raw` | 1949580 | 100% |
| `wind_mean_ms` | 1949580 | 100% |

| Spot | Rows | Time range | Source ids |
| --- | ---: | --- | --- |
| `balistra` | 487003 | `2024-01-02T22:25:09Z` -> `2026-06-24T21:58:57Z` | `1693` |
| `figari_eole` | 142840 | `2024-05-05T16:58:07Z` -> `2026-06-24T14:25:03Z` | `1661` |
| `la_tonnara` | 255534 | `2024-01-01T00:05:52Z` -> `2026-06-24T21:18:15Z` | `51` |
| `piantarella` | 197060 | `2023-12-31T23:08:00Z` -> `2026-06-24T21:58:31Z` | `1659` |
| `porticcio` | 361320 | `2025-06-23T22:00:50Z` -> `2026-06-24T21:59:55Z` | `1726` |
| `porto_polo` | 313414 | `2024-01-01T00:57:15Z` -> `2026-06-24T21:01:33Z` | `84` |
| `santa_manza` | 192409 | `2024-01-01T00:50:11Z` -> `2026-06-24T21:00:34Z` | `1549` |

## MeteoNet Ground Stations

- rows: `9934994`
- stations: `51`
- time range: `2016-01-01T00:00:00Z` -> `2018-12-31T23:54:00Z`
- raw archives: `3`
- station registry: `/srv/data/corsewind/ml_dataset/research/meteonet/normalized/ground_stations/zone=SE/stations.json`

| Year | Rows | Stations | Wind speed | Wind direction | Temperature | Pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2016` | 3044773 | 39 | 67% | 67% | 89% | 25% |
| `2017` | 3344351 | 42 | 63% | 63% | 90% | 23% |
| `2018` | 3545870 | 50 | 64% | 64% | 93% | 22% |

## Model Samples

| Source | Rows | Spots | Time range | Wind | Gust | Direction |
| --- | ---: | ---: | --- | ---: | ---: | ---: |

## Meteo-France WCS Families


## NWP Extra Fields

| Source | Rows | Spots | Time range | Features |
| --- | ---: | ---: | --- | --- |


## NWP Vertical Profiles

| Source | Rows | Spots | Time range | Pressure levels | Profile features | Derived features |
| --- | ---: | ---: | --- | --- | --- | --- |
| `arome_0025` | 100 | 20 | `2026-06-24T09:00:00Z` -> `2026-06-24T13:00:00Z` | `1000`, `950`, `925`, `900`, `850` | `geopotential_height_m`, `pseudo_adiabatic_potential_temperature_c`, `relative_humidity_pct`, `temperature_c`, `vertical_velocity_pressure_pa_s` | `geopotential_thickness_1000_850_m`, `low_level_inversion_strength_c`, `relative_humidity_mean_1000_850_pct`, `temperature_lapse_rate_1000_850_c_per_km`, `vertical_velocity_pressure_850_pa_s` |


## Feature Store 15 min

- rows: `4976`
- spots: `7`
- time range: `2024-06-01T10:00:00Z` -> `2024-07-01T00:00:00Z`

| Source flag | Rows with source |
| --- | ---: |
| `context_stations` | 4976 |
| `model_open_meteo_meteofrance_arome_france` | 4976 |

| Target | Non-null rows |
| --- | ---: |
| `wind_mean_ms` | 4976 |
| `gust_ms` | 4976 |
| `temperature_c` | 4976 |
| `observation_timestamp_utc` | 4976 |
| `observation_distance_minutes` | 4976 |
| `observation_source_project` | 4976 |
| `observation_source_dataset` | 4976 |
| `wind_direction_deg` | 4955 |
| `pressure_hpa` | 4245 |

## Residual Training Tables

- tables: `3`
- latest: `residual_backfill_2024_06`
- total rows: `24184`

| Table | Rows | Source rows | Leads | Spots | Wind RMSE gain | Gust RMSE gain | Missing baseline wind | Missing target wind |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `residual_backfill_2024_06` | 19732 | 19732 | `+60m:4933`, `+120m:4933`, `+180m:4933`, `+360m:4933` | 7 | -0.104 | 4.118 | `{}` | `{}` |
| `residual_backfill_smoke_2024_06_01_07` | 4284 | 4284 | `+60m:1071`, `+120m:1071`, `+180m:1071`, `+360m:1071` | 7 | -5.02 | -0.026 | `{}` | `{}` |
| `residual_correction_pilot_20260622` | 168 | 63 | `+60m:56`, `+120m:49`, `+180m:42`, `+360m:21` | 7 | -9.594 | 37.815 | `{}` | `{}` |

## Trained Residual Models

- runs: `2`
- latest: `residual_backfill_2024_06`

| Run | Rows | Train | Test | Target | Type | RMSE gain vs raw | Test metric |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| `residual_backfill_2024_06` | 19732 | 15792 | 3940 | `residual_gust_ms` | `regression` | 30.402 | RMSE `1.603525`, MAE `1.148377` |
| `residual_backfill_2024_06` | 19732 | 15792 | 3940 | `residual_wind_mean_ms` | `regression` | 17.766 | RMSE `1.585554`, MAE `1.144753` |
| `residual_backfill_2024_06` | 19732 | 15792 | 3940 | `target_gust_gt_20kt` | `classification` | None | Brier `0.040287`, positives `208` |
| `residual_backfill_2024_06` | 19732 | 15792 | 3940 | `target_gust_gt_25kt` | `classification` | None | Brier `0.01102`, positives `44` |
| `residual_backfill_2024_06` | 19732 | 15792 | 3940 | `target_wind_gt_15kt` | `classification` | None | Brier `0.074906`, positives `448` |
| `residual_backfill_2024_06` | 19732 | 15792 | 3940 | `target_wind_gt_20kt` | `classification` | None | Brier `0.021772`, positives `100` |
| `residual_backfill_smoke_2024_06_01_07` | 4284 | 3416 | 868 | `residual_gust_ms` | `regression` | 39.105 | RMSE `1.218738`, MAE `0.950362` |
| `residual_backfill_smoke_2024_06_01_07` | 4284 | 3416 | 868 | `residual_wind_mean_ms` | `regression` | 18.577 | RMSE `1.312423`, MAE `0.996284` |
| `residual_backfill_smoke_2024_06_01_07` | 4284 | 3416 | 868 | `target_gust_gt_20kt` | `classification` | None | Brier `0.009216`, positives `8` |
| `residual_backfill_smoke_2024_06_01_07` | 4284 | 3416 | 868 | `target_wind_gt_15kt` | `classification` | None | Brier `0.022287`, positives `15` |
| `residual_backfill_smoke_2024_06_01_07` | 4284 | 3416 | 868 | `target_wind_gt_20kt` | `classification` | None | Brier `0.004648`, positives `4` |

## Copernicus Marine SST

- rows: `1260`
- time range: `2026-06-16T10:00:00Z` -> `2026-06-22T18:00:00Z`
- spots: `20`

| Field | Non-null | Coverage |
| --- | ---: | ---: |
| `sst_c` | 1260 | 100% |
| `sst_k` | 1260 | 100% |
| `sst_pixel_latitude` | 1260 | 100% |
| `sst_pixel_longitude` | 1260 | 100% |
| `sst_sample_distance_km` | 1260 | 100% |

## EUMETSAT Cloud Mask

- rows: `0`
- products: `0`
- time range: `None` -> `None`
- spots: `0`
- cloud state counts: `{}`

| Field | Non-null | Coverage |
| --- | ---: | ---: |

## EUMETSAT Cloud Type

- rows: `1300`
- products: `65`
- time range: `2026-06-22T09:50:00Z` -> `2026-06-24T12:00:00Z`
- spots: `20`
- sampled variables: `{'cloud_type': 1300, 'quality_overall_processing': 1300, 'cloud_phase': 1300, 'quality_illumination': 1300, 'quality_nwp_parameters': 1300, 'quality_MTG_parameters': 1300}`

| Field | Non-null | Coverage |
| --- | ---: | ---: |
| `neighborhoods` | 1300 | 100% |
| `product_completeness` | 1300 | 100% |
| `product_quality` | 1300 | 100% |
| `product_timeliness` | 1300 | 100% |
| `sample_distance_km` | 1300 | 100% |
| `sampled_values` | 1300 | 100% |
| `sampled_values_c` | 1300 | 100% |

## EUMETSAT Land Surface Temperature

- rows: `1220`
- products: `61`
- time range: `2026-06-22T10:00:07Z` -> `2026-06-24T11:50:07Z`
- spots: `20`
- sampled variables: `{'QFLAGS': 1220, 'LST': 687, 'LST_uncertainty': 687}`

| Field | Non-null | Coverage |
| --- | ---: | ---: |
| `neighborhoods` | 1220 | 100% |
| `sample_distance_km` | 1220 | 100% |
| `sampled_values` | 1220 | 100% |
| `sampled_values_c` | 1220 | 100% |

## EUMETSAT Global Instability Indices

- rows: `1300`
- products: `65`
- time range: `2026-06-22T09:50:00Z` -> `2026-06-24T12:00:00Z`
- spots: `20`
- sampled variables: `{'lifted_index': 1300, 'k_index': 1300, 'prec_water_low': 1300, 'prec_water_mid': 1300, 'prec_water_high': 1300, 'prec_water_total': 1300, 'percent_cloud_free': 1300, 'number_of_iterations': 1300}`

| Field | Non-null | Coverage |
| --- | ---: | ---: |
| `neighborhoods` | 1300 | 100% |
| `product_completeness` | 1300 | 100% |
| `product_timeliness` | 1300 | 100% |
| `sample_distance_km` | 1300 | 100% |
| `sampled_values` | 1300 | 100% |
| `sampled_values_c` | 1300 | 100% |

## External Access Configuration

| Source | Configured | Required env | Target data |
| --- | --- | --- | --- |
| `copernicus_marine` | `False` | `COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD` | `sst_nearest_c`, `land_minus_sea_temp_c` |
| `eumetsat` | `False` | `EUMETSAT_CONSUMER_KEY`, `EUMETSAT_CONSUMER_SECRET` | `cloud_fraction_satellite`, `cloud_type`, `cloud_top_height` |
| `cds_era5` | `False` | `CDSAPI_URL`, `CDSAPI_KEY` | `historical_reanalysis_context` |
