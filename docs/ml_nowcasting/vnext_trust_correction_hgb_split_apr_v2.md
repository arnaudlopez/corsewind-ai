# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:34:34.872337Z`
Train end: `2026-04-01`
Validation end / holdout start: `2026-05-01`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.155771 | 1.213023 | 1.238199 | `do_not_promote` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.341581 | 1.374747 | 1.441421 | `reliable_candidate` |

## `wind_mean`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 1452, 'categorical': 29, 'total': 1481}`
- selected candidate: `residual_hist_gradient_boosting_scale0.4_clip1.5`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.155771`
- holdout RMSE: `1.213023`
- full RMSE: `1.238199`
- deltas vs champion: `{'validation_rmse_delta': -0.014978, 'holdout_rmse_delta': 0.00734, 'full_rmse_delta': -0.060643}`
- official gate: `1.268019`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.155771 | 0.875252 | 0.113451 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.155839 | 0.875287 | 0.113487 |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.15593 | 0.873595 | 0.11332 |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.156024 | 0.87364 | 0.113365 |
| `residual_hist_gradient_boosting_scale0.3_clip1.5` | `residual_regressor` | 1.157182 | 0.87764 | 0.113582 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.157228 | 0.877667 | 0.113609 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.158718 | 0.87796 | 0.116452 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.159217 | 0.876962 | 0.117071 |
| `residual_hist_gradient_boosting_scale0.3_clip1` | `residual_regressor` | 1.159627 | 0.879672 | 0.115833 |
| `residual_hist_gradient_boosting_scale0.2_clip1.5` | `residual_regressor` | 1.160158 | 0.881026 | 0.113714 |

## `gust`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 1452, 'categorical': 29, 'total': 1481}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip2`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.341581`
- holdout RMSE: `1.374747`
- full RMSE: `1.441421`
- deltas vs champion: `{'validation_rmse_delta': -0.031552, 'holdout_rmse_delta': 0.003684, 'full_rmse_delta': -0.096997}`
- official gate: `1.484221`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.341581 | 0.993685 | 0.144344 |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.341941 | 0.993907 | 0.144661 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.344524 | 0.996974 | 0.149592 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.344762 | 0.998381 | 0.150453 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.345085 | 0.998559 | 0.150707 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.347256 | 1.000744 | 0.157819 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.347581 | 1.00128 | 0.154652 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.349527 | 1.004233 | 0.156562 |
| `residual_hist_gradient_boosting_scale0.3_clip1.5` | `residual_regressor` | 1.349795 | 1.004381 | 0.156752 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.350296 | 1.004506 | 0.161233 |
