# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:44:02.139334Z`
Train end: `2026-04-01`
Validation end / holdout start: `2026-05-01`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.151897 | 1.208142 | 1.266107 | `do_not_promote` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.343737 | 1.368488 | 1.484729 | `do_not_promote` |

## `wind_mean`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip1.5`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.151897`
- holdout RMSE: `1.208142`
- full RMSE: `1.266107`
- deltas vs champion: `{'validation_rmse_delta': -0.018852, 'holdout_rmse_delta': 0.002459, 'full_rmse_delta': -0.032735}`
- official gate: `1.268019`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.151897 | 0.869644 | 0.092364 |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.151897 | 0.869644 | 0.092364 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.152018 | 0.869725 | 0.092206 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.153782 | 0.87272 | 0.096686 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.153782 | 0.87272 | 0.096686 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.153907 | 0.872793 | 0.09656 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.154465 | 0.872346 | 0.094263 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.156117 | 0.874903 | 0.098205 |
| `residual_hist_gradient_boosting_scale0.3_clip1.5` | `residual_regressor` | 1.156618 | 0.87636 | 0.101009 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.156618 | 0.87636 | 0.101009 |

## `gust`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip2`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.343737`
- holdout RMSE: `1.368488`
- full RMSE: `1.484729`
- deltas vs champion: `{'validation_rmse_delta': -0.029396, 'holdout_rmse_delta': -0.002575, 'full_rmse_delta': -0.053689}`
- official gate: `1.484221`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.343737 | 0.992127 | 0.144547 |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.343799 | 0.992153 | 0.144437 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.345158 | 0.993636 | 0.145702 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.347224 | 0.997762 | 0.150616 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.34732 | 0.997826 | 0.150527 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.348044 | 0.997099 | 0.149501 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.348623 | 0.999088 | 0.15154 |
| `residual_hist_gradient_boosting_scale0.5_clip0.5` | `residual_regressor` | 1.351057 | 1.001847 | 0.157011 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.351281 | 1.001959 | 0.154579 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.351922 | 1.004201 | 0.156684 |
