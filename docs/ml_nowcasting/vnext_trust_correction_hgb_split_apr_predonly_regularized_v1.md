# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:47:45.628353Z`
Train end: `2026-04-01`
Validation end / holdout start: `2026-05-01`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.153436 | 1.209527 | 1.275085 | `do_not_promote` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.345054 | 1.369427 | 1.499439 | `do_not_promote` |

## `wind_mean`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip1`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.153436`
- holdout RMSE: `1.209527`
- full RMSE: `1.275085`
- deltas vs champion: `{'validation_rmse_delta': -0.017313, 'holdout_rmse_delta': 0.003844, 'full_rmse_delta': -0.023757}`
- official gate: `1.268019`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.153436 | 0.871043 | 0.091491 |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.153436 | 0.871043 | 0.091491 |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.153436 | 0.871043 | 0.091491 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.15504 | 0.8727 | 0.093274 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.155231 | 0.873862 | 0.095988 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.155231 | 0.873862 | 0.095988 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.155231 | 0.873862 | 0.095988 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.156614 | 0.875188 | 0.097415 |
| `residual_hist_gradient_boosting_scale0.5_clip0.5` | `residual_regressor` | 1.157435 | 0.875521 | 0.101253 |
| `residual_hist_gradient_boosting_scale0.3_clip1` | `residual_regressor` | 1.157866 | 0.877222 | 0.100485 |

## `gust`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip1.5`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.345054`
- holdout RMSE: `1.369427`
- full RMSE: `1.499439`
- deltas vs champion: `{'validation_rmse_delta': -0.028079, 'holdout_rmse_delta': -0.001636, 'full_rmse_delta': -0.038979}`
- official gate: `1.484221`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.345054 | 0.99445 | 0.144619 |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.345054 | 0.99445 | 0.144619 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.345362 | 0.994889 | 0.145153 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.348167 | 0.998242 | 0.148682 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.34847 | 0.999785 | 0.150673 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.34847 | 0.999785 | 0.150673 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.348829 | 1.000207 | 0.1511 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.351374 | 1.002946 | 0.153923 |
| `residual_hist_gradient_boosting_scale0.5_clip0.5` | `residual_regressor` | 1.35139 | 1.002785 | 0.157525 |
| `trust_hist_gradient_boosting_p0.3_a0.4_clip2` | `trust_classifier` | 1.352326 | 1.00608 | 0.171688 |
