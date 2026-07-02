# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:37:14.928092Z`
Train end: `2026-04-01`
Validation end / holdout start: `2026-05-01`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.155771 | 1.213023 | 1.238199 | `do_not_promote` |

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
| `residual_hist_gradient_boosting_scale0.32_clip1.5` | `residual_regressor` | 1.156774 | 0.877084 | 0.113556 |
| `residual_hist_gradient_boosting_scale0.32_clip2` | `residual_regressor` | 1.156824 | 0.877113 | 0.113585 |
| `residual_hist_gradient_boosting_scale0.3_clip1.5` | `residual_regressor` | 1.157182 | 0.87764 | 0.113582 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.157228 | 0.877667 | 0.113609 |
| `residual_hist_gradient_boosting_scale0.28_clip1.5` | `residual_regressor` | 1.157652 | 0.878227 | 0.113609 |
| `residual_hist_gradient_boosting_scale0.28_clip2` | `residual_regressor` | 1.157694 | 0.878252 | 0.113634 |
