# Spot Feature Store 15 min

Generated at: `2026-06-23T17:15:19.976890Z`

## Grain

One row per `spot_id + target_time_utc`, rounded to a 15-minute grid from available target observations.

## Outputs

- `data/processed/ml_dataset/feature_store/spot_forecast_15min.jsonl`
- `data/processed/ml_dataset/feature_store/spot_forecast_15min_profile.json`
- `data/processed/ml_dataset/feature_store/spot_forecast_15min_feature_columns.csv`

## Current Coverage

- rows: `8`
- spots: `7`
- time range: `2026-06-22T14:00:00Z` -> `2026-06-23T17:00:00Z`

### Source Flag Counts

| Source | Rows with source |
| --- | ---: |
| `eumetsat_cloud_mask` | 5 |
| `eumetsat_cloud_type` | 5 |
| `eumetsat_global_instability_indices` | 5 |
| `eumetsat_land_surface_temperature` | 5 |
| `model_arome` | 5 |
| `model_aromepi` | 5 |
| `sst` | 8 |

## Targets

| Name |
| --- |
| `gust_ms` |
| `observation_distance_minutes` |
| `observation_source_dataset` |
| `observation_source_project` |
| `observation_timestamp_utc` |
| `pressure_hpa` |
| `temperature_c` |
| `wind_direction_deg` |
| `wind_mean_ms` |

## Features

Context station slots now also include static geometry and dynamic upwind
features when the feature store is rebuilt:

- `context_<slot>_bearing_from_spot_deg`
- `context_<slot>_bearing_to_spot_deg`
- `context_<slot>_east_offset_km`
- `context_<slot>_north_offset_km`
- `context_<slot>_altitude_delta_m`
- `context_<slot>_upwind_score_from_target_wind`
- `context_agg_<group>_bearing_from_spot_deg_*`
- `context_agg_<group>_east_offset_km_*`
- `context_agg_<group>_north_offset_km_*`
- `context_agg_<group>_upwind_score_from_target_wind_*`

These fields are computed from station geometry and the latest available
pre-target wind direction only. They are intended to make neighboring stations
usable in a SAPHIR-like way without leaking the target observation.

| Name |
| --- |
| `eumetsat_cloud_mask_age_minutes` |
| `eumetsat_cloud_mask_available` |
| `eumetsat_cloud_state` |
| `eumetsat_cloud_state_fraction_2` |
| `eumetsat_cloud_state_fraction_3` |
| `eumetsat_cloud_state_mode` |
| `eumetsat_cloud_state_valid_count` |
| `eumetsat_cloud_type_age_minutes` |
| `eumetsat_cloud_type_available` |
| `eumetsat_cloud_type_cloud_phase` |
| `eumetsat_cloud_type_cloud_phase_neighborhood_max` |
| `eumetsat_cloud_type_cloud_phase_neighborhood_mean` |
| `eumetsat_cloud_type_cloud_phase_neighborhood_min` |
| `eumetsat_cloud_type_cloud_phase_neighborhood_valid_count` |
| `eumetsat_cloud_type_cloud_type` |
| `eumetsat_cloud_type_cloud_type_neighborhood_fraction_11` |
| `eumetsat_cloud_type_cloud_type_neighborhood_fraction_12` |
| `eumetsat_cloud_type_cloud_type_neighborhood_fraction_13` |
| `eumetsat_cloud_type_cloud_type_neighborhood_fraction_6` |
| `eumetsat_cloud_type_cloud_type_neighborhood_fraction_7` |
| `eumetsat_cloud_type_cloud_type_neighborhood_fraction_8` |
| `eumetsat_cloud_type_cloud_type_neighborhood_fraction_9` |
| `eumetsat_cloud_type_cloud_type_neighborhood_mode` |
| `eumetsat_cloud_type_cloud_type_neighborhood_valid_count` |
| `eumetsat_cloud_type_product_completeness` |
| `eumetsat_cloud_type_product_quality` |
| `eumetsat_cloud_type_product_timeliness` |
| `eumetsat_cloud_type_quality_MTG_parameters` |
| `eumetsat_cloud_type_quality_MTG_parameters_neighborhood_fraction_1` |
| `eumetsat_cloud_type_quality_MTG_parameters_neighborhood_mode` |
| `eumetsat_cloud_type_quality_MTG_parameters_neighborhood_valid_count` |
| `eumetsat_cloud_type_quality_illumination` |
| `eumetsat_cloud_type_quality_illumination_neighborhood_fraction_3` |
| `eumetsat_cloud_type_quality_illumination_neighborhood_mode` |
| `eumetsat_cloud_type_quality_illumination_neighborhood_valid_count` |
| `eumetsat_cloud_type_quality_nwp_parameters` |
| `eumetsat_cloud_type_quality_nwp_parameters_neighborhood_fraction_2` |
| `eumetsat_cloud_type_quality_nwp_parameters_neighborhood_mode` |
| `eumetsat_cloud_type_quality_nwp_parameters_neighborhood_valid_count` |
| `eumetsat_cloud_type_quality_overall_processing` |
| `eumetsat_cloud_type_quality_overall_processing_neighborhood_fraction_1` |
| `eumetsat_cloud_type_quality_overall_processing_neighborhood_mode` |
| `eumetsat_cloud_type_quality_overall_processing_neighborhood_valid_count` |
| `eumetsat_cloud_type_sample_distance_km` |
| `eumetsat_cloud_type_sensing_start_utc` |
| `eumetsat_global_instability_indices_age_minutes` |
| `eumetsat_global_instability_indices_available` |
| `eumetsat_global_instability_indices_k_index` |
| `eumetsat_global_instability_indices_k_index_neighborhood_max` |
| `eumetsat_global_instability_indices_k_index_neighborhood_mean` |
| `eumetsat_global_instability_indices_k_index_neighborhood_min` |
| `eumetsat_global_instability_indices_k_index_neighborhood_valid_count` |
| `eumetsat_global_instability_indices_lifted_index` |
| `eumetsat_global_instability_indices_lifted_index_neighborhood_max` |
| `eumetsat_global_instability_indices_lifted_index_neighborhood_mean` |
| `eumetsat_global_instability_indices_lifted_index_neighborhood_min` |
| `eumetsat_global_instability_indices_lifted_index_neighborhood_valid_count` |
| `eumetsat_global_instability_indices_number_of_iterations` |
| `eumetsat_global_instability_indices_number_of_iterations_neighborhood_max` |
| `eumetsat_global_instability_indices_number_of_iterations_neighborhood_mean` |
| `eumetsat_global_instability_indices_number_of_iterations_neighborhood_min` |
| `eumetsat_global_instability_indices_number_of_iterations_neighborhood_valid_count` |
| `eumetsat_global_instability_indices_percent_cloud_free` |
| `eumetsat_global_instability_indices_percent_cloud_free_neighborhood_max` |
| `eumetsat_global_instability_indices_percent_cloud_free_neighborhood_mean` |
| `eumetsat_global_instability_indices_percent_cloud_free_neighborhood_min` |
| `eumetsat_global_instability_indices_percent_cloud_free_neighborhood_valid_count` |
| `eumetsat_global_instability_indices_prec_water_high` |
| `eumetsat_global_instability_indices_prec_water_high_neighborhood_max` |
| `eumetsat_global_instability_indices_prec_water_high_neighborhood_mean` |
| `eumetsat_global_instability_indices_prec_water_high_neighborhood_min` |
| `eumetsat_global_instability_indices_prec_water_high_neighborhood_valid_count` |
| `eumetsat_global_instability_indices_prec_water_low` |
| `eumetsat_global_instability_indices_prec_water_low_neighborhood_max` |
| `eumetsat_global_instability_indices_prec_water_low_neighborhood_mean` |
| `eumetsat_global_instability_indices_prec_water_low_neighborhood_min` |
| `eumetsat_global_instability_indices_prec_water_low_neighborhood_valid_count` |
| `eumetsat_global_instability_indices_prec_water_mid` |
| `eumetsat_global_instability_indices_prec_water_mid_neighborhood_max` |
| `eumetsat_global_instability_indices_prec_water_mid_neighborhood_mean` |
| `eumetsat_global_instability_indices_prec_water_mid_neighborhood_min` |
| `eumetsat_global_instability_indices_prec_water_mid_neighborhood_valid_count` |
| `eumetsat_global_instability_indices_prec_water_total` |
| `eumetsat_global_instability_indices_prec_water_total_neighborhood_max` |
| `eumetsat_global_instability_indices_prec_water_total_neighborhood_mean` |
| `eumetsat_global_instability_indices_prec_water_total_neighborhood_min` |
| `eumetsat_global_instability_indices_prec_water_total_neighborhood_valid_count` |
| `eumetsat_global_instability_indices_product_completeness` |
| `eumetsat_global_instability_indices_product_quality` |
| `eumetsat_global_instability_indices_product_timeliness` |
| `eumetsat_global_instability_indices_sample_distance_km` |
| `eumetsat_global_instability_indices_sensing_start_utc` |
| `eumetsat_land_surface_temperature_LST` |
| `eumetsat_land_surface_temperature_LST_c` |
| `eumetsat_land_surface_temperature_LST_neighborhood_max` |
| `eumetsat_land_surface_temperature_LST_neighborhood_mean` |
| `eumetsat_land_surface_temperature_LST_neighborhood_min` |
| `eumetsat_land_surface_temperature_LST_neighborhood_valid_count` |
| `eumetsat_land_surface_temperature_LST_uncertainty` |
| `eumetsat_land_surface_temperature_LST_uncertainty_neighborhood_max` |
| `eumetsat_land_surface_temperature_LST_uncertainty_neighborhood_mean` |
| `eumetsat_land_surface_temperature_LST_uncertainty_neighborhood_min` |
| `eumetsat_land_surface_temperature_LST_uncertainty_neighborhood_valid_count` |
| `eumetsat_land_surface_temperature_QFLAGS` |
| `eumetsat_land_surface_temperature_QFLAGS_neighborhood_fraction_0` |
| `eumetsat_land_surface_temperature_QFLAGS_neighborhood_fraction_10` |
| `eumetsat_land_surface_temperature_QFLAGS_neighborhood_fraction_101` |
| `eumetsat_land_surface_temperature_QFLAGS_neighborhood_fraction_102` |
| `eumetsat_land_surface_temperature_QFLAGS_neighborhood_mode` |
| `eumetsat_land_surface_temperature_QFLAGS_neighborhood_valid_count` |
| `eumetsat_land_surface_temperature_age_minutes` |
| `eumetsat_land_surface_temperature_available` |
| `eumetsat_land_surface_temperature_product_completeness` |
| `eumetsat_land_surface_temperature_product_quality` |
| `eumetsat_land_surface_temperature_product_timeliness` |
| `eumetsat_land_surface_temperature_sample_distance_km` |
| `eumetsat_land_surface_temperature_sensing_start_utc` |
| `eumetsat_sample_distance_km` |
| `model_arome_available` |
| `model_arome_gust_speed_ms` |
| `model_arome_lead_minutes` |
| `model_arome_run_age_minutes` |
| `model_arome_run_time_utc` |
| `model_arome_valid_offset_minutes` |
| `model_arome_valid_time_utc` |
| `model_arome_wind_direction_deg` |
| `model_arome_wind_speed_ms` |
| `model_arome_wind_u_ms` |
| `model_arome_wind_v_ms` |
| `model_aromepi_available` |
| `model_aromepi_gust_speed_ms` |
| `model_aromepi_lead_minutes` |
| `model_aromepi_run_age_minutes` |
| `model_aromepi_run_time_utc` |
| `model_aromepi_valid_offset_minutes` |
| `model_aromepi_valid_time_utc` |
| `model_aromepi_wind_direction_deg` |
| `model_aromepi_wind_speed_ms` |
| `model_aromepi_wind_u_ms` |
| `model_aromepi_wind_v_ms` |
| `model_icon2i_available` |
| `model_moloch_available` |
| `obs_delta_15m_gust_ms` |
| `obs_delta_15m_pressure_hpa` |
| `obs_delta_15m_temperature_c` |
| `obs_delta_15m_wind_mean_ms` |
| `obs_lag_15m_age_minutes` |
| `obs_lag_15m_available` |
| `obs_lag_15m_gust_ms` |
| `obs_lag_15m_pressure_hpa` |
| `obs_lag_15m_temperature_c` |
| `obs_lag_15m_wind_direction_deg` |
| `obs_lag_15m_wind_mean_ms` |
| `obs_lag_60m_age_minutes` |
| `obs_lag_60m_available` |
| `obs_last_age_minutes` |
| `obs_last_available` |
| `obs_last_dewpoint_c` |
| `obs_last_global_radiation_raw` |
| `obs_last_gust_ms` |
| `obs_last_humidity_pct` |
| `obs_last_precipitation_mm` |
| `obs_last_pressure_hpa` |
| `obs_last_sea_level_pressure_hpa` |
| `obs_last_temperature_c` |
| `obs_last_wind_direction_deg` |
| `obs_last_wind_mean_ms` |
| `sst_age_minutes` |
| `sst_available` |
| `sst_c` |
| `sst_k` |
| `sst_sample_distance_km` |
