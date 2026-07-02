# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:42:29.713192Z`
Train end: `2026-03-01T00:00:00Z`
Validation end / holdout start: `2026-04-01T00:00:00Z`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.297834 | 1.190722 | 1.261666 | `reliable_candidate` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.537002 | 1.363656 | 1.48616 | `do_not_promote` |

## `wind_mean`

- split rows: `{'train': 7136, 'validation': 4289, 'holdout': 12851}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip2`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.297834`
- holdout RMSE: `1.190722`
- full RMSE: `1.261666`
- deltas vs champion: `{'validation_rmse_delta': -0.044625, 'holdout_rmse_delta': -0.003505, 'full_rmse_delta': -0.037176}`
- official gate: `1.268019`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.297834 | 0.929442 | 0.073359 |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.297868 | 0.929506 | 0.07326 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.299326 | 0.931111 | 0.072516 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.302282 | 0.934956 | 0.073943 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.304539 | 0.938404 | 0.077331 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.304583 | 0.938455 | 0.077253 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.305986 | 0.939852 | 0.076657 |
| `residual_hist_gradient_boosting_scale0.5_clip0.5` | `residual_regressor` | 1.308497 | 0.942682 | 0.077615 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.308683 | 0.943148 | 0.077799 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.312375 | 0.94825 | 0.081304 |

## `gust`

- split rows: `{'train': 7136, 'validation': 4289, 'holdout': 12851}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_hist_gradient_boosting_scale0.5_clip2`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.537002`
- holdout RMSE: `1.363656`
- full RMSE: `1.48616`
- deltas vs champion: `{'validation_rmse_delta': -0.052241, 'holdout_rmse_delta': -0.008093, 'full_rmse_delta': -0.052258}`
- official gate: `1.484221`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.537002 | 1.076643 | 0.123779 |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.538916 | 1.077463 | 0.12264 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.542996 | 1.080778 | 0.124168 |
| `residual_hist_gradient_boosting_scale0.4_clip2` | `residual_regressor` | 1.544022 | 1.084396 | 0.126534 |
| `residual_hist_gradient_boosting_scale0.4_clip1.5` | `residual_regressor` | 1.545789 | 1.085265 | 0.125623 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.547854 | 1.084725 | 0.126776 |
| `residual_hist_gradient_boosting_scale0.4_clip1` | `residual_regressor` | 1.549693 | 1.088518 | 0.126845 |
| `residual_hist_gradient_boosting_scale0.3_clip2` | `residual_regressor` | 1.552788 | 1.093761 | 0.129289 |
| `residual_hist_gradient_boosting_scale0.4_clip0.75` | `residual_regressor` | 1.554086 | 1.091968 | 0.128932 |
| `residual_hist_gradient_boosting_scale0.3_clip1.5` | `residual_regressor` | 1.554286 | 1.094474 | 0.128606 |
