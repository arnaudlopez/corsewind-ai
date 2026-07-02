# CorseWind Scientific Error Floor Audit

Generated: `2026-07-01T12:06:23.864995Z`

## Executive Verdict

- Current wind champion: 1.268 m/s (2.46 kt), MAE 0.930, bias +0.019 on 31429 rows.
- Current gust champion: 1.484 m/s (2.89 kt), MAE 1.074, bias +0.056 on 31429 rows.
- Wind RMSE 0.9 needs 49.623% MSE reduction; top 5% rows carry 40.993% of SSE.
- Gust RMSE 0.9 needs 63.23% MSE reduction; top 5% rows carry 42.866% of SSE.
- Existing wind row-oracle across raw/base/calibrated reaches 1.039 m/s, so current model variants alone do not prove a path to 0.9.
- Existing gust row-oracle across raw/base/calibrated reaches 1.243 m/s.

## Champion Stage Metrics

| Target | Raw | Base corrected | Calibrated champion | Raw to calibrated gain |
| --- | ---: | ---: | ---: | ---: |
| wind_mean | 2.187 m/s (4.25 kt), MAE 1.667, bias +0.390 | 1.277 m/s (2.48 kt), MAE 0.944, bias +0.035 | 1.268 m/s (2.46 kt), MAE 0.930, bias +0.019 | 42.028% |
| gust | 3.936 m/s (7.65 kt), MAE 3.022, bias +2.582 | 1.501 m/s (2.92 kt), MAE 1.097, bias +0.132 | 1.484 m/s (2.89 kt), MAE 1.074, bias +0.056 | 62.29% |

## Error Concentration

| Target | Rows needed perfect to hit 0.9 | Top 1% SSE | Top 5% SSE | Top 10% SSE | Top 20% SSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| wind_mean | 7.452% | 16.983% | 40.993% | 56.654% | 74.579% |
| gust | 12.126% | 18.315% | 42.866% | 58.446% | 76.119% |

## Diagnostic Bias Oracles

These use observed 2026 labels to remove grouped mean residuals. They are diagnostic upper bounds, not deployable models.

### Wind

| Oracle | RMSE | MAE | Bias | Rows |
| --- | ---: | ---: | ---: | ---: |
| `global_mean_residual_removed` | 1.268 | 0.930 | +0.000 | 31429 |
| `spot_id` | 1.258 | 0.915 | +0.000 | 31429 |
| `lead_time_minutes` | 1.268 | 0.930 | -0.000 | 31429 |
| `spot_id+lead_time_minutes` | 1.257 | 0.915 | +0.000 | 31429 |
| `spot_id+lead_time_minutes+target_hour_local` | 1.234 | 0.902 | -0.000 | 31429 |
| `spot_id+lead_time_minutes+wind_actual_bin_ms` | 1.162 | 0.839 | -0.000 | 31429 |

### Gust

| Oracle | RMSE | MAE | Bias | Rows |
| --- | ---: | ---: | ---: | ---: |
| `global_mean_residual_removed` | 1.483 | 1.072 | +0.000 | 31429 |
| `spot_id` | 1.476 | 1.060 | +0.000 | 31429 |
| `lead_time_minutes` | 1.483 | 1.072 | +0.000 | 31429 |
| `spot_id+lead_time_minutes` | 1.475 | 1.059 | +0.000 | 31429 |
| `spot_id+lead_time_minutes+target_hour_local` | 1.451 | 1.046 | +0.000 | 31429 |
| `spot_id+lead_time_minutes+gust_actual_bin_ms` | 1.376 | 0.978 | -0.000 | 31429 |

## Wind Hard Groups

### By Spot

| Group | Rows | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `spot_id=la_tonnara` | 4287 | 1.520 | 1.139 | +0.009 | 19.61% | 1.137 |
| `spot_id=santa_manza` | 4096 | 1.497 | 1.113 | -0.125 | 18.18% | 1.147 |
| `spot_id=balistra` | 4858 | 1.235 | 0.924 | +0.133 | 14.65% | 1.171 |
| `spot_id=porticcio` | 6366 | 1.010 | 0.701 | -0.191 | 12.86% | 1.184 |
| `spot_id=porto_polo` | 3761 | 1.171 | 0.931 | +0.314 | 10.21% | 1.202 |
| `spot_id=piantarella` | 4002 | 1.129 | 0.844 | +0.046 | 10.09% | 1.202 |
| `spot_id=figari_eole` | 1170 | 1.259 | 0.976 | +0.161 | 3.67% | 1.245 |
| `spot_id=cap_corse` | 407 | 1.825 | 1.369 | -0.059 | 2.68% | 1.251 |

### By Actual Wind Bin

| Group | Rows | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wind_actual_bin_ms=0-2` | 9343 | 1.036 | 0.759 | +0.461 | 19.84% | 1.135 |
| `wind_actual_bin_ms=10+` | 2544 | 1.959 | 1.500 | -0.778 | 19.31% | 1.139 |
| `wind_actual_bin_ms=2-4` | 7265 | 1.094 | 0.799 | +0.083 | 17.20% | 1.154 |
| `wind_actual_bin_ms=4-6` | 5491 | 1.208 | 0.925 | -0.092 | 15.87% | 1.163 |
| `wind_actual_bin_ms=6-8` | 4084 | 1.370 | 1.052 | -0.167 | 15.16% | 1.168 |
| `wind_actual_bin_ms=8-10` | 2702 | 1.535 | 1.168 | -0.428 | 12.61% | 1.185 |

### By Spot + Lead

| Group | Rows | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `spot_id=la_tonnara, lead_time_minutes=60.0` | 1091 | 1.705 | 1.264 | +0.017 | 6.28% | 1.228 |
| `spot_id=la_tonnara, lead_time_minutes=45.0` | 1065 | 1.598 | 1.218 | +0.054 | 5.38% | 1.233 |
| `spot_id=santa_manza, lead_time_minutes=60.0` | 1003 | 1.574 | 1.188 | -0.156 | 4.92% | 1.236 |
| `spot_id=santa_manza, lead_time_minutes=30.0` | 1031 | 1.545 | 1.138 | -0.178 | 4.87% | 1.237 |
| `spot_id=santa_manza, lead_time_minutes=45.0` | 1028 | 1.513 | 1.140 | -0.093 | 4.66% | 1.238 |
| `spot_id=la_tonnara, lead_time_minutes=30.0` | 1076 | 1.462 | 1.112 | -0.017 | 4.55% | 1.239 |
| `spot_id=balistra, lead_time_minutes=45.0` | 1262 | 1.305 | 1.000 | +0.151 | 4.25% | 1.241 |
| `spot_id=balistra, lead_time_minutes=60.0` | 1153 | 1.334 | 1.009 | +0.091 | 4.06% | 1.242 |
| `spot_id=santa_manza, lead_time_minutes=15.0` | 1034 | 1.351 | 0.987 | -0.075 | 3.73% | 1.244 |
| `spot_id=porticcio, lead_time_minutes=60.0` | 1561 | 1.085 | 0.765 | -0.158 | 3.64% | 1.245 |

## Gust Hard Groups

### By Spot

| Group | Rows | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `spot_id=la_tonnara` | 4287 | 1.802 | 1.330 | -0.010 | 20.11% | 1.327 |
| `spot_id=santa_manza` | 4096 | 1.837 | 1.344 | -0.025 | 19.96% | 1.328 |
| `spot_id=balistra` | 4858 | 1.458 | 1.089 | +0.200 | 14.92% | 1.369 |
| `spot_id=piantarella` | 4002 | 1.538 | 1.114 | +0.078 | 13.67% | 1.379 |
| `spot_id=porticcio` | 6366 | 1.154 | 0.805 | -0.140 | 12.24% | 1.390 |
| `spot_id=porto_polo` | 3761 | 1.312 | 1.027 | +0.323 | 9.35% | 1.413 |
| `spot_id=figari_eole` | 1170 | 1.523 | 1.165 | +0.162 | 3.92% | 1.455 |
| `spot_id=cap_corse` | 407 | 1.486 | 1.106 | -0.133 | 1.30% | 1.475 |

### By Actual Gust Bin

| Group | Rows | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gust_actual_bin_ms=10+` | 4930 | 2.101 | 1.566 | -0.664 | 31.42% | 1.229 |
| `gust_actual_bin_ms=0-2` | 7713 | 1.159 | 0.859 | +0.554 | 14.96% | 1.369 |
| `gust_actual_bin_ms=2-4` | 6249 | 1.242 | 0.887 | +0.238 | 13.93% | 1.377 |
| `gust_actual_bin_ms=6-8` | 3998 | 1.550 | 1.168 | -0.049 | 13.88% | 1.377 |
| `gust_actual_bin_ms=4-6` | 5267 | 1.306 | 0.969 | +0.013 | 12.97% | 1.385 |
| `gust_actual_bin_ms=8-10` | 3272 | 1.649 | 1.248 | -0.181 | 12.85% | 1.386 |

## Windsurf Thresholds

| Target | Threshold | Actual event rate | Pred event rate | Precision | Recall | F1 | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wind | `ge_12kt` | 26.119% | 27.497% | 0.836 | 0.880 | 0.858 | 984 | 1417 |
| wind | `ge_15kt` | 19.698% | 17.325% | 0.874 | 0.768 | 0.818 | 1434 | 688 |
| wind | `ge_20kt` | 8.044% | 6.733% | 0.840 | 0.703 | 0.765 | 751 | 339 |
| wind | `ge_25kt` | 2.81% | 2.746% | 0.774 | 0.757 | 0.765 | 215 | 195 |
| gust | `ge_15kt` | 29.046% | 27.456% | 0.899 | 0.849 | 0.873 | 1374 | 874 |
| gust | `ge_20kt` | 15.607% | 14.417% | 0.853 | 0.788 | 0.819 | 1041 | 667 |
| gust | `ge_25kt` | 6.72% | 6.255% | 0.799 | 0.743 | 0.770 | 542 | 396 |
| gust | `ge_30kt` | 3.064% | 2.953% | 0.765 | 0.737 | 0.751 | 253 | 218 |

## Label And Observation Diagnostics

- Training parquet files audited: `30`.
- Label rows audited: `2585922` across `14` spots.
- Target observation distance minutes: `{'count': 2585922, 'mean': 0.522123, 'p50': 0.35, 'p90': 1.083, 'p95': 1.483, 'p99': 4.567, 'max': 7.5}`.
- Duplicate target consistency: `{'target_groups': 475649, 'groups_with_multiple_rows': 461158, 'wind_range': {'count': 461158, 'mean': 0.0, 'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}, 'gust_range': {'count': 461153, 'mean': 0.0, 'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}, 'wind_groups_range_gt_0_05_ms_pct': 0.0, 'gust_groups_range_gt_0_05_ms_pct': 0.0}`.
- Short-term target volatility <=20 min: `{'pair_count': 313603, 'wind_abs_delta_ms': {'count': 313603, 'mean': 0.80437, 'p50': 0.514445, 'p90': 2.057777, 'p95': 2.572222, 'p99': 3.601111, 'max': 23.15}, 'gust_abs_delta_ms': {'count': 313603, 'mean': 0.980025, 'p50': 0.514445, 'p90': 2.057778, 'p95': 3.086666, 'p99': 4.63, 'max': 22.121111}, 'wind_delta_gt_1ms_pct': 40.572, 'wind_delta_gt_2ms_pct': 10.482, 'gust_delta_gt_2ms_pct': 16.088}`.
- Training source coverage: `{'row_count': 2585922, 'columns': {'issue_feature_sources__previous_run_open_meteo_best_match_day1': {'present': True, 'non_null_count': 2585922, 'non_null_pct': 100.0}, 'issue_feature_sources__previous_run_open_meteo_best_match_day2': {'present': True, 'non_null_count': 2585922, 'non_null_pct': 100.0}, 'target_feature_sources__previous_run_open_meteo_best_match_day1': {'present': True, 'non_null_count': 2585922, 'non_null_pct': 100.0}, 'target_feature_sources__previous_run_open_meteo_best_match_day2': {'present': True, 'non_null_count': 2585922, 'non_null_pct': 100.0}, 'issue_feature_sources__vertical_arome': {'present': True, 'non_null_count': 2585922, 'non_null_pct': 100.0}, 'target_feature_sources__vertical_arome': {'present': True, 'non_null_count': 2585922, 'non_null_pct': 100.0}}}`.

## Feature Coverage Snapshot

| Concept | Present | Columns | Any non-null | Top columns |
| --- | ---: | ---: | ---: | --- |
| `sst` | `True` | 5 | 100.0% | `features__sst_age_minutes` 100.0%, `features__sst_available` 100.0%, `features__sst_c` 99.927% |
| `land_surface_temperature` | `True` | 22 | 100.0% | `features__eumetsat_land_surface_temperature_available` 100.0%, `features__eumetsat_land_surface_temperature_age_minutes` 1.091%, `features__eumetsat_land_surface_temperature_LST_neighborhood_max` 0.554% |
| `land_sea_or_thermal_delta` | `False` | 0 | 0.0% |  |
| `cloud_type_or_cover` | `True` | 50 | 100.0% | `baselines__baseline_cloud_cover_pct` 100.0%, `features__eumetsat_cloud_type_available` 100.0%, `features__model_open_meteo_meteofrance_arome_france_cloud_cover` 100.0% |
| `instability_or_cape` | `True` | 45 | 100.0% | `features__eumetsat_global_instability_indices_available` 100.0%, `features__eumetsat_global_instability_indices_age_minutes` 1.104%, `features__eumetsat_global_instability_indices_k_index` 0.598% |
| `radiation` | `True` | 3 | 100.0% | `baselines__baseline_shortwave_radiation` 100.0%, `features__model_open_meteo_meteofrance_arome_france_shortwave_radiation` 100.0%, `features__context_global_coastal_1_global_radiation_raw` 20.255% |
| `pressure` | `True` | 57 | 100.0% | `baselines__baseline_pressure_msl_hpa` 100.0%, `baselines__baseline_surface_pressure_hpa` 100.0%, `features__context_agg_all_delta_vs_target_pressure_hpa_count` 100.0% |
| `recent_obs_and_model_error` | `True` | 17 | 100.0% | `features__model_error_now_gust_ms` 100.0%, `features__model_error_now_wind_mean_ms` 100.0%, `features__obs_lag_15m_available` 100.0% |
| `context_stations` | `True` | 278 | 100.0% | `features__context_agg_all_delta_vs_target_pressure_hpa_count` 100.0%, `features__context_agg_all_delta_vs_target_temperature_c_count` 100.0%, `features__context_agg_all_pressure_hpa_count` 100.0% |
| `previous_runs` | `False` | 0 | 0.0% |  |
| `dem_static` | `True` | 9 | 100.0% | `features__context_agg_inland_altitude_m_count` 100.0%, `features__context_agg_inland_altitude_m_max` 100.0%, `features__context_agg_inland_altitude_m_mean` 100.0% |
| `fetch_static` | `False` | 0 | 0.0% |  |
| `vertical_profile` | `False` | 0 | 0.0% |  |

## Interpretation

- The current champions already remove most of the raw AROME/Open-Meteo error, especially for gusts; the remaining gap is concentrated in a limited set of high-energy rows and south-coast spots.
- Static bias correction is not the missing magic lever: even diagnostic in-sample bias oracles should be treated as upper bounds, not production evidence.
- The most credible next gains are not more blind model families; they are denser fresh observations, better strong-wind/thermal event labeling, candidate models with genuinely different errors, and probabilistic threshold heads.
- If the label audit shows large observation-distance or short-term volatility, the RMSE floor may be partly imposed by target noise and by trying to predict a 6-15 minute turbulent signal with point labels.

## Recommended Next Steps

1. Build an event-weighted evaluation set for thermal start, thermal collapse, and strong wind bins >=12/15/20/25 kt; optimize these explicitly alongside RMSE.
2. Add a label-quality gate: exclude or down-weight targets with stale observations, inconsistent duplicate labels, or extreme 15-minute jumps during baseline training.
3. Train specialist heads for high-wind and threshold probability, but promote only if they improve both RMSE and recall/precision on windsurf thresholds.
4. Keep collecting live data; the current 2026 test window is too short to prove subtle feature gains, especially for rare regimes.
5. Re-run foundation/model-router work only when candidate predictions have dense overlap; sparse oracle wins are not enough for production promotion.
