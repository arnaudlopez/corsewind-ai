# Scientific Error Diagnostic

Generated: `2026-06-28T09:33:57.621258Z`
Prediction file: `/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/calibrated_predictions_2026.parquet`
Rows: `31429`
Champion RMSE: `1.268019`
Champion MAE: `0.930465`
MSE reduction needed for threshold: `49.623%`

## Stage Metrics

| Stage | Count | RMSE | MAE | Bias | P90 abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw_wind_mean_ms` | 31429 | 2.187306 | 1.66659 | 0.39034 | 3.496667 |
| `corrected_wind_mean_ms` | 31429 | 1.276846 | 0.943833 | 0.03505 | 2.013794 |
| `calibrated_wind_mean_ms` | 31429 | 1.268019 | 0.930465 | 0.018767 | 1.98403 |

## Tail

- `row_count`: `31429`
- `current_sse`: `50533.7806`
- `target_sse`: `25457.49`
- `excess_sse`: `25076.2906`
- `mse_reduction_needed_pct`: `49.623`
- `perfect_rows_needed`: `2342`
- `perfect_rows_needed_pct`: `7.452`
- `top_1_pct_sse_share_pct`: `16.983`
- `top_2_pct_sse_share_pct`: `25.314`
- `top_5_pct_sse_share_pct`: `40.993`
- `top_10_pct_sse_share_pct`: `56.654`
- `top_20_pct_sse_share_pct`: `74.579`

## SSE By Spot Lead

| Group | Count | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `{'spot_id': 'la_tonnara', 'lead_time_minutes': 60.0}` | 1091 | 1.705399 | 1.263573 | 0.017305 | 6.279% | 1.227563 |
| `{'spot_id': 'la_tonnara', 'lead_time_minutes': 45.0}` | 1065 | 1.597801 | 1.218209 | 0.05363 | 5.38% | 1.233435 |
| `{'spot_id': 'santa_manza', 'lead_time_minutes': 60.0}` | 1003 | 1.573746 | 1.187939 | -0.155721 | 4.916% | 1.23646 |
| `{'spot_id': 'santa_manza', 'lead_time_minutes': 30.0}` | 1031 | 1.545343 | 1.137971 | -0.178294 | 4.872% | 1.236743 |
| `{'spot_id': 'santa_manza', 'lead_time_minutes': 45.0}` | 1028 | 1.512747 | 1.139558 | -0.092896 | 4.655% | 1.238152 |
| `{'spot_id': 'la_tonnara', 'lead_time_minutes': 30.0}` | 1076 | 1.462087 | 1.112238 | -0.016637 | 4.552% | 1.238824 |
| `{'spot_id': 'balistra', 'lead_time_minutes': 45.0}` | 1262 | 1.30471 | 1.000121 | 0.150837 | 4.251% | 1.240773 |
| `{'spot_id': 'balistra', 'lead_time_minutes': 60.0}` | 1153 | 1.333993 | 1.009135 | 0.090626 | 4.06% | 1.242009 |
| `{'spot_id': 'santa_manza', 'lead_time_minutes': 15.0}` | 1034 | 1.350661 | 0.98725 | -0.074784 | 3.733% | 1.244127 |
| `{'spot_id': 'porticcio', 'lead_time_minutes': 60.0}` | 1561 | 1.085197 | 0.765176 | -0.157581 | 3.638% | 1.244741 |
| `{'spot_id': 'porticcio', 'lead_time_minutes': 45.0}` | 1566 | 1.080815 | 0.733278 | -0.177626 | 3.62% | 1.244856 |
| `{'spot_id': 'balistra', 'lead_time_minutes': 30.0}` | 1181 | 1.21842 | 0.893298 | 0.10765 | 3.469% | 1.245828 |
| `{'spot_id': 'la_tonnara', 'lead_time_minutes': 15.0}` | 1055 | 1.275772 | 0.955807 | -0.017618 | 3.398% | 1.246289 |
| `{'spot_id': 'porticcio', 'lead_time_minutes': 30.0}` | 1572 | 0.978192 | 0.684297 | -0.180998 | 2.977% | 1.249004 |
| `{'spot_id': 'porto_polo', 'lead_time_minutes': 60.0}` | 960 | 1.244904 | 1.005687 | 0.38448 | 2.944% | 1.249213 |
| `{'spot_id': 'piantarella', 'lead_time_minutes': 60.0}` | 1056 | 1.182833 | 0.888102 | 0.067128 | 2.924% | 1.249345 |
| `{'spot_id': 'balistra', 'lead_time_minutes': 15.0}` | 1262 | 1.072341 | 0.800577 | 0.175983 | 2.872% | 1.249679 |
| `{'spot_id': 'piantarella', 'lead_time_minutes': 45.0}` | 1017 | 1.161359 | 0.874316 | 0.026933 | 2.714% | 1.250691 |
| `{'spot_id': 'cap_corse', 'lead_time_minutes': 60.0}` | 407 | 1.825083 | 1.368928 | -0.059446 | 2.683% | 1.250894 |
| `{'spot_id': 'porto_polo', 'lead_time_minutes': 45.0}` | 876 | 1.241592 | 0.974361 | 0.339136 | 2.672% | 1.250961 |
| `{'spot_id': 'porticcio', 'lead_time_minutes': 15.0}` | 1667 | 0.892501 | 0.627663 | -0.245192 | 2.628% | 1.251248 |
| `{'spot_id': 'piantarella', 'lead_time_minutes': 30.0}` | 943 | 1.153707 | 0.841784 | 0.094091 | 2.484% | 1.252172 |
| `{'spot_id': 'porto_polo', 'lead_time_minutes': 30.0}` | 946 | 1.150313 | 0.913158 | 0.242459 | 2.477% | 1.252215 |
| `{'spot_id': 'porto_polo', 'lead_time_minutes': 15.0}` | 979 | 1.043915 | 0.836038 | 0.292133 | 2.111% | 1.254562 |
| `{'spot_id': 'la_parata', 'lead_time_minutes': 60.0}` | 415 | 1.554714 | 1.152496 | -0.15345 | 1.985% | 1.25537 |

## SSE By Spot

| Group | Count | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `{'spot_id': 'la_tonnara'}` | 4287 | 1.520349 | 1.138581 | 0.009216 | 19.609% | 1.136918 |
| `{'spot_id': 'santa_manza'}` | 4096 | 1.497476 | 1.112557 | -0.125203 | 18.176% | 1.147007 |
| `{'spot_id': 'balistra'}` | 4858 | 1.234581 | 0.924454 | 0.13258 | 14.653% | 1.171442 |
| `{'spot_id': 'porticcio'}` | 6366 | 1.010447 | 0.701348 | -0.191237 | 12.862% | 1.183666 |
| `{'spot_id': 'porto_polo'}` | 3761 | 1.170954 | 0.930957 | 0.314158 | 10.205% | 1.201579 |
| `{'spot_id': 'piantarella'}` | 4002 | 1.12882 | 0.844032 | 0.045986 | 10.091% | 1.202338 |
| `{'spot_id': 'figari_eole'}` | 1170 | 1.259202 | 0.975709 | 0.161141 | 3.671% | 1.244526 |
| `{'spot_id': 'cap_corse'}` | 407 | 1.825083 | 1.368928 | -0.059446 | 2.683% | 1.250894 |
| `{'spot_id': 'la_parata'}` | 415 | 1.554714 | 1.152496 | -0.15345 | 1.985% | 1.25537 |
| `{'spot_id': 'lfvh'}` | 443 | 1.386233 | 0.996058 | 0.018025 | 1.685% | 1.257293 |
| `{'spot_id': 'lfkf'}` | 471 | 1.248207 | 0.906333 | 0.129269 | 1.452% | 1.258778 |
| `{'spot_id': 'lfvf'}` | 357 | 1.402601 | 1.019475 | 0.146691 | 1.39% | 1.259176 |
| `{'spot_id': 'lfks'}` | 404 | 1.000393 | 0.696147 | 0.012789 | 0.8% | 1.262936 |
| `{'spot_id': 'lfkj'}` | 392 | 0.975864 | 0.714774 | 0.111859 | 0.739% | 1.263326 |

## SSE By Lead

| Group | Count | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `{'lead_time_minutes': 60.0}` | 10007 | 1.359074 | 0.99675 | 0.02964 | 36.577% | 1.009832 |
| `{'lead_time_minutes': 45.0}` | 7102 | 1.31142 | 0.971726 | 0.036616 | 24.17% | 1.104194 |
| `{'lead_time_minutes': 30.0}` | 7045 | 1.250536 | 0.915433 | 0.001041 | 21.802% | 1.121306 |
| `{'lead_time_minutes': 15.0}` | 7275 | 1.10099 | 0.813567 | 0.003552 | 17.451% | 1.152078 |

## SSE By Local Target Hour

| Group | Count | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `{'target_hour_local': 17}` | 1931 | 1.336378 | 1.0001 | 0.060177 | 6.824% | 1.223987 |
| `{'target_hour_local': 16}` | 1954 | 1.327208 | 0.988122 | 0.01148 | 6.811% | 1.224074 |
| `{'target_hour_local': 12}` | 1845 | 1.328896 | 1.002078 | 0.102259 | 6.448% | 1.226459 |
| `{'target_hour_local': 13}` | 1868 | 1.308609 | 1.008928 | 0.075868 | 6.33% | 1.227229 |
| `{'target_hour_local': 18}` | 1963 | 1.274634 | 0.951539 | 0.097158 | 6.311% | 1.227353 |
| `{'target_hour_local': 15}` | 1895 | 1.285866 | 0.970391 | 0.119789 | 6.2% | 1.228079 |
| `{'target_hour_local': 14}` | 1829 | 1.301944 | 0.975681 | 0.112727 | 6.135% | 1.228506 |
| `{'target_hour_local': 9}` | 1547 | 1.408345 | 1.00971 | -0.133711 | 6.072% | 1.228919 |
| `{'target_hour_local': 11}` | 1652 | 1.332378 | 0.998104 | 0.124701 | 5.803% | 1.230675 |
| `{'target_hour_local': 10}` | 1612 | 1.340212 | 1.008961 | -0.122205 | 5.73% | 1.231156 |
| `{'target_hour_local': 19}` | 1842 | 1.217485 | 0.912511 | 0.081406 | 5.403% | 1.233287 |
| `{'target_hour_local': 8}` | 1520 | 1.2837 | 0.905573 | -0.012294 | 4.957% | 1.236194 |
| `{'target_hour_local': 20}` | 1546 | 1.206005 | 0.900511 | 0.034416 | 4.45% | 1.239486 |
| `{'target_hour_local': 21}` | 1254 | 1.216613 | 0.876536 | -0.055177 | 3.673% | 1.244514 |
| `{'target_hour_local': 7}` | 1216 | 1.192286 | 0.854235 | 0.026909 | 3.421% | 1.246142 |
| `{'target_hour_local': 22}` | 890 | 1.147883 | 0.810264 | -0.052408 | 2.321% | 1.253219 |
| `{'target_hour_local': 6}` | 854 | 1.155981 | 0.816925 | -0.059994 | 2.258% | 1.253619 |
| `{'target_hour_local': 2}` | 605 | 1.281834 | 0.808977 | -0.018242 | 1.967% | 1.255485 |
| `{'target_hour_local': 3}` | 586 | 1.188172 | 0.861916 | -0.006087 | 1.637% | 1.257596 |
| `{'target_hour_local': 23}` | 655 | 1.122652 | 0.794119 | -0.143188 | 1.634% | 1.257619 |
| `{'target_hour_local': 5}` | 668 | 1.069831 | 0.742411 | -0.049598 | 1.513% | 1.25839 |
| `{'target_hour_local': 0}` | 574 | 1.127524 | 0.777519 | -0.07272 | 1.444% | 1.25883 |
| `{'target_hour_local': 4}` | 570 | 1.110852 | 0.770941 | -0.108553 | 1.392% | 1.259163 |
| `{'target_hour_local': 1}` | 553 | 1.075817 | 0.753957 | -0.199334 | 1.267% | 1.259963 |

## SSE By Month

| Group | Count | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `{'issue_month': '2026-01'}` | 5219 | 1.409449 | 1.017604 | -0.013358 | 20.517% | 1.130483 |
| `{'issue_month': '2026-05'}` | 6520 | 1.208664 | 0.91412 | -0.022386 | 18.849% | 1.142283 |
| `{'issue_month': '2026-03'}` | 5537 | 1.277849 | 0.92565 | 0.052381 | 17.892% | 1.148998 |
| `{'issue_month': '2026-02'}` | 3898 | 1.486098 | 1.05435 | 0.013784 | 17.036% | 1.154973 |
| `{'issue_month': '2026-04'}` | 5545 | 1.131992 | 0.852027 | 0.06072 | 14.061% | 1.175497 |
| `{'issue_month': '2026-06'}` | 4710 | 1.117862 | 0.852014 | 0.02655 | 11.647% | 1.19189 |

## SSE By Actual Wind Bin

| Group | Count | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `{'actual_wind_bin_ms': '0-2'}` | 9343 | 1.036014 | 0.759438 | 0.460936 | 19.844% | 1.135253 |
| `{'actual_wind_bin_ms': '10+'}` | 2544 | 1.958638 | 1.500033 | -0.7777 | 19.313% | 1.139012 |
| `{'actual_wind_bin_ms': '2-4'}` | 7265 | 1.093912 | 0.798544 | 0.083126 | 17.204% | 1.153802 |
| `{'actual_wind_bin_ms': '4-6'}` | 5491 | 1.208492 | 0.925246 | -0.09193 | 15.869% | 1.163062 |
| `{'actual_wind_bin_ms': '6-8'}` | 4084 | 1.369858 | 1.051684 | -0.167023 | 15.165% | 1.167917 |
| `{'actual_wind_bin_ms': '8-10'}` | 2702 | 1.535373 | 1.167671 | -0.427542 | 12.605% | 1.185413 |

## SSE By Error Sign

| Group | Count | RMSE | MAE | Bias | SSE share | RMSE if perfect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `{'error_sign_bin': 'over_1_2'}` | 4189 | 1.432843 | 1.406384 | 1.406384 | 17.019% | 1.15509 |
| `{'error_sign_bin': 'under_3plus'}` | 499 | 4.102384 | 3.974764 | -3.974764 | 16.618% | 1.157872 |
| `{'error_sign_bin': 'under_1_2'}` | 3868 | 1.431596 | 1.404349 | -1.404349 | 15.687% | 1.16432 |
| `{'error_sign_bin': 'over_3plus'}` | 444 | 4.12734 | 3.976498 | 3.976498 | 14.967% | 1.169281 |
| `{'error_sign_bin': 'over_2_3'}` | 1068 | 2.409984 | 2.39523 | 2.39523 | 12.275% | 1.187647 |
| `{'error_sign_bin': 'under_2_3'}` | 1065 | 2.412576 | 2.397122 | -2.397122 | 12.267% | 1.187703 |
| `{'error_sign_bin': 'ok_-1_1'}` | 20296 | 0.527289 | 0.446404 | 0.016909 | 11.167% | 1.195125 |

## Feature Availability In High Error Tail

| Family | Columns | High-error coverage | Other coverage | Delta |
| --- | ---: | ---: | ---: | ---: |
| `sst` | 5 | 100.0% | 100.0% | 0.0% |
| `cloud` | 56 | 100.0% | 100.0% | 0.0% |
| `instability` | 47 | 100.0% | 100.0% | 0.0% |
| `land_surface_temperature` | 22 | 100.0% | 100.0% | 0.0% |
| `surface_pressure` | 57 | 100.0% | 100.0% | 0.0% |
| `temperature` | 101 | 100.0% | 100.0% | 0.0% |
| `radiation` | 5 | 100.0% | 100.0% | 0.0% |
| `context_wind` | 283 | 100.0% | 100.0% | 0.0% |
| `recent_obs` | 16 | 100.0% | 100.0% | 0.0% |
| `vertical_profile` | 0 | None% | None% | None% |

## Top Spearman Correlations With Absolute Error

| Feature | Spearman | Coverage |
| --- | ---: | ---: |
| `features__context_inland_1_gust_ms` | 0.296054 | 23.733% |
| `features__context_inland_1_wind_mean_ms` | 0.294084 | 23.73% |
| `features__obs_lag_15m_pressure_hpa` | -0.258938 | 7.827% |
| `features__obs_lag_60m_pressure_hpa` | -0.258938 | 7.827% |
| `features__obs_last_pressure_hpa` | -0.258938 | 7.827% |
| `features__context_agg_inland_gust_ms_min` | 0.216107 | 100.0% |
| `features__context_agg_inland_gust_ms_mean` | 0.208064 | 100.0% |
| `features__context_agg_inland_wind_mean_ms_min` | 0.207438 | 99.997% |
| `features__obs_lag_15m_gust_ms` | 0.206574 | 99.905% |
| `features__obs_lag_15m_wind_mean_ms` | 0.201893 | 99.905% |
| `features__context_agg_inland_wind_mean_ms_mean` | 0.201241 | 99.997% |
| `features__context_global_inland_1_gust_ms` | 0.200598 | 98.571% |
| `features__context_agg_inland_gust_ms_max` | 0.194963 | 100.0% |
| `features__context_global_coastal_1_pressure_hpa` | -0.193144 | 81.683% |
| `features__context_global_nearest_1_pressure_hpa` | -0.193144 | 81.683% |
| `features__context_global_inland_1_wind_mean_ms` | 0.190716 | 98.549% |
| `features__context_agg_inland_wind_mean_ms_max` | 0.186545 | 99.997% |
| `features__obs_last_sea_level_pressure_hpa` | -0.183721 | 7.872% |
| `features__context_global_coastal_1_gust_ms` | 0.178139 | 98.67% |
| `features__context_agg_all_delta_vs_target_pressure_hpa_max` | 0.17734 | 6.574% |
| `features__context_agg_all_delta_vs_target_pressure_hpa_mean` | 0.17734 | 6.574% |
| `features__context_agg_all_delta_vs_target_pressure_hpa_min` | 0.17734 | 6.574% |
| `features__context_agg_coastal_delta_vs_target_pressure_hpa_max` | 0.17734 | 6.574% |
| `features__context_agg_coastal_delta_vs_target_pressure_hpa_mean` | 0.17734 | 6.574% |
| `features__context_agg_coastal_delta_vs_target_pressure_hpa_min` | 0.17734 | 6.574% |
