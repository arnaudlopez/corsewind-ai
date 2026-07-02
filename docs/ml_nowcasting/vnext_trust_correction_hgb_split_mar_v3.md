# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:31:31.340765Z`
Train end: `2026-03-01T00:00:00Z`
Validation end / holdout start: `2026-04-01T00:00:00Z`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.292646 | 1.201984 | 1.236669 | `reliable_candidate` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.538671 | 1.378795 | 1.459323 | `do_not_promote` |

## `wind_mean`

- split rows: `{'train': 7136, 'validation': 4289, 'holdout': 12851}`
- feature counts: `{'numeric': 1413, 'categorical': 28, 'total': 1441}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip1.5`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.292646`
- holdout RMSE: `1.201984`
- full RMSE: `1.236669`
- deltas vs champion: `{'validation_rmse_delta': -0.049813, 'holdout_rmse_delta': 0.007757, 'full_rmse_delta': -0.062173}`
- official gate: `1.268019`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.292646 | 0.920755 | 0.091447 |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.292721 | 0.920926 | 0.091273 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.295858 | 0.924309 | 0.094503 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.299319 | 0.930716 | 0.091802 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.29935 | 0.930818 | 0.091663 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.301499 | 0.93143 | 0.100598 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.302248 | 0.933735 | 0.094247 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.307341 | 0.939806 | 0.099123 |
| `residual_hist_gradient_boosting_scale0.3_clip1.5` | `residual_regressor` | 1.307671 | 0.941719 | 0.092157 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.307673 | 0.94174 | 0.092052 |

## `gust`

- split rows: `{'train': 7136, 'validation': 4289, 'holdout': 12851}`
- feature counts: `{'numeric': 1413, 'categorical': 28, 'total': 1441}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip2`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.538671`
- holdout RMSE: `1.378795`
- full RMSE: `1.459323`
- deltas vs champion: `{'validation_rmse_delta': -0.050572, 'holdout_rmse_delta': 0.007046, 'full_rmse_delta': -0.079095}`
- official gate: `1.484221`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.538671 | 1.075534 | 0.15013 |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.539831 | 1.075964 | 0.14983 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.544359 | 1.080529 | 0.151806 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.544726 | 1.082922 | 0.147615 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.545807 | 1.083274 | 0.147375 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.549697 | 1.086997 | 0.155283 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.550108 | 1.087428 | 0.148955 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.552846 | 1.092245 | 0.1451 |
| `residual_hist_gradient_boosting_scale0.3_clip1.5` | `residual_regressor` | 1.553769 | 1.092592 | 0.14492 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.555111 | 1.093055 | 0.151737 |
