# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:45:51.314294Z`
Train end: `2026-05-01`
Validation end / holdout start: `2026-06-01`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.227993 | 1.143857 | 1.292404 | `reliable_candidate` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.388426 | 1.301244 | 1.523721 | `reliable_candidate` |

## `wind_mean`

- split rows: `{'train': 15681, 'validation': 5069, 'holdout': 3526}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `static_vnext_blend_a0.4_clip2`
- selected candidate type: `static_vnext_blend`
- validation RMSE: `1.227993`
- holdout RMSE: `1.143857`
- full RMSE: `1.292404`
- deltas vs champion: `{'validation_rmse_delta': -0.010676, 'holdout_rmse_delta': -0.012759, 'full_rmse_delta': -0.006438}`
- official gate: `1.268019`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `static_vnext_blend_a0.4_clip2` | `static_vnext_blend` | 1.227993 | 0.938267 | -0.014231 |
| `static_vnext_blend_a0.4_clip1` | `static_vnext_blend` | 1.228571 | 0.938625 | -0.014876 |
| `static_vnext_blend_a0.3_clip2` | `static_vnext_blend` | 1.229297 | 0.939418 | -0.014296 |
| `static_vnext_blend_a0.4_clip0.5` | `static_vnext_blend` | 1.229309 | 0.939141 | -0.016116 |
| `static_vnext_blend_a0.3_clip1` | `static_vnext_blend` | 1.229829 | 0.939765 | -0.01478 |
| `trust_hist_gradient_boosting_p0.2_a0.4_clip2` | `trust_classifier` | 1.229981 | 0.939437 | -0.013966 |
| `trust_hist_gradient_boosting_p0.3_a0.4_clip2` | `trust_classifier` | 1.229989 | 0.939695 | -0.014447 |
| `trust_hist_gradient_boosting_p0.2_a0.4_clip1` | `trust_classifier` | 1.230557 | 0.939796 | -0.01461 |
| `trust_hist_gradient_boosting_p0.3_a0.4_clip1` | `trust_classifier` | 1.230595 | 0.940058 | -0.01503 |
| `static_vnext_blend_a0.3_clip0.5` | `static_vnext_blend` | 1.230714 | 0.940364 | -0.01571 |

## `gust`

- split rows: `{'train': 15681, 'validation': 5069, 'holdout': 3526}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `static_vnext_blend_a0.4_clip1`
- selected candidate type: `static_vnext_blend`
- validation RMSE: `1.388426`
- holdout RMSE: `1.301244`
- full RMSE: `1.523721`
- deltas vs champion: `{'validation_rmse_delta': -0.014363, 'holdout_rmse_delta': -0.022878, 'full_rmse_delta': -0.014697}`
- official gate: `1.484221`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `static_vnext_blend_a0.4_clip1` | `static_vnext_blend` | 1.388426 | 1.061179 | 0.034984 |
| `static_vnext_blend_a0.4_clip2` | `static_vnext_blend` | 1.388809 | 1.061283 | 0.035594 |
| `trust_hist_gradient_boosting_p0.3_a0.4_clip1` | `trust_classifier` | 1.389019 | 1.061684 | 0.03291 |
| `residual_hist_gradient_boosting_scale0.5_clip2` | `residual_regressor` | 1.389041 | 1.059549 | 0.008084 |
| `trust_hist_gradient_boosting_p0.2_a0.4_clip1` | `trust_classifier` | 1.389147 | 1.061505 | 0.035114 |
| `residual_hist_gradient_boosting_scale0.5_clip1.5` | `residual_regressor` | 1.389207 | 1.059747 | 0.007791 |
| `trust_hist_gradient_boosting_p0.3_a0.4_clip2` | `trust_classifier` | 1.389237 | 1.061796 | 0.033231 |
| `residual_hist_gradient_boosting_scale0.5_clip0.75` | `residual_regressor` | 1.389374 | 1.060211 | 0.00791 |
| `residual_hist_gradient_boosting_scale0.5_clip1` | `residual_regressor` | 1.389535 | 1.060167 | 0.007133 |
| `trust_hist_gradient_boosting_p0.2_a0.4_clip2` | `trust_classifier` | 1.389539 | 1.06165 | 0.035682 |
