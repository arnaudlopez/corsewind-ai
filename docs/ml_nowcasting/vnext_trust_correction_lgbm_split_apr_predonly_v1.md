# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:52:28.953482Z`
Train end: `2026-04-01`
Validation end / holdout start: `2026-05-01`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.151712 | 1.208008 | 1.265205 | `do_not_promote` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.343904 | 1.36819 | 1.489806 | `do_not_promote` |

## `wind_mean`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_lightgbm_scale0.5_clip1.5`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.151712`
- holdout RMSE: `1.208008`
- full RMSE: `1.265205`
- deltas vs champion: `{'validation_rmse_delta': -0.019037, 'holdout_rmse_delta': 0.002325, 'full_rmse_delta': -0.033637}`
- official gate: `1.268019`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_lightgbm_scale0.5_clip1.5` | `residual_regressor` | 1.151712 | 0.869154 | 0.096607 |
| `residual_lightgbm_scale0.5_clip2` | `residual_regressor` | 1.151712 | 0.869154 | 0.096607 |
| `residual_lightgbm_scale0.5_clip1` | `residual_regressor` | 1.152045 | 0.869428 | 0.096703 |
| `residual_lightgbm_scale0.4_clip1.5` | `residual_regressor` | 1.153458 | 0.872141 | 0.100081 |
| `residual_lightgbm_scale0.4_clip2` | `residual_regressor` | 1.153458 | 0.872141 | 0.100081 |
| `residual_lightgbm_scale0.4_clip1` | `residual_regressor` | 1.153755 | 0.87236 | 0.100158 |
| `residual_lightgbm_scale0.5_clip0.75` | `residual_regressor` | 1.154928 | 0.872424 | 0.099278 |
| `residual_lightgbm_scale0.3_clip1.5` | `residual_regressor` | 1.156244 | 0.875726 | 0.103555 |
| `residual_lightgbm_scale0.3_clip2` | `residual_regressor` | 1.156244 | 0.875726 | 0.103555 |
| `residual_lightgbm_scale0.4_clip0.75` | `residual_regressor` | 1.156307 | 0.874756 | 0.102217 |

## `gust`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_lightgbm_scale0.5_clip2`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.343904`
- holdout RMSE: `1.36819`
- full RMSE: `1.489806`
- deltas vs champion: `{'validation_rmse_delta': -0.029229, 'holdout_rmse_delta': -0.002873, 'full_rmse_delta': -0.048612}`
- official gate: `1.484221`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_lightgbm_scale0.5_clip2` | `residual_regressor` | 1.343904 | 0.992695 | 0.151873 |
| `residual_lightgbm_scale0.5_clip1.5` | `residual_regressor` | 1.343975 | 0.992761 | 0.151683 |
| `residual_lightgbm_scale0.5_clip1` | `residual_regressor` | 1.34473 | 0.993255 | 0.151554 |
| `residual_lightgbm_scale0.4_clip2` | `residual_regressor` | 1.347486 | 0.998272 | 0.156476 |
| `residual_lightgbm_scale0.4_clip1.5` | `residual_regressor` | 1.347566 | 0.998351 | 0.156324 |
| `residual_lightgbm_scale0.5_clip0.75` | `residual_regressor` | 1.347567 | 0.996841 | 0.154682 |
| `residual_lightgbm_scale0.4_clip1` | `residual_regressor` | 1.348283 | 0.998769 | 0.156221 |
| `residual_lightgbm_scale0.4_clip0.75` | `residual_regressor` | 1.350866 | 1.001656 | 0.158723 |
| `residual_lightgbm_scale0.5_clip0.5` | `residual_regressor` | 1.351781 | 1.002663 | 0.159862 |
| `residual_lightgbm_scale0.3_clip2` | `residual_regressor` | 1.352213 | 1.004595 | 0.161079 |
