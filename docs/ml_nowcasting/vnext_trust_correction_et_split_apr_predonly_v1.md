# v_next Trust Correction Benchmark

Generated: `2026-07-01T20:49:22.785868Z`
Train end: `2026-04-01`
Validation end / holdout start: `2026-05-01`

| Target | Rows | Champion full RMSE | v_next full RMSE | Oracle RMSE | Selected validation RMSE | Selected holdout RMSE | Selected full RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.153148 | 1.212507 | 1.268607 | `do_not_promote` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.343139 | 1.368102 | 1.49393 | `do_not_promote` |

## `wind_mean`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_extra_trees_scale0.5_clip1.5`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.153148`
- holdout RMSE: `1.212507`
- full RMSE: `1.268607`
- deltas vs champion: `{'validation_rmse_delta': -0.017601, 'holdout_rmse_delta': 0.006824, 'full_rmse_delta': -0.030235}`
- official gate: `1.268019`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_extra_trees_scale0.5_clip1.5` | `residual_regressor` | 1.153148 | 0.869414 | 0.093759 |
| `residual_extra_trees_scale0.5_clip2` | `residual_regressor` | 1.153148 | 0.869414 | 0.093759 |
| `residual_extra_trees_scale0.5_clip1` | `residual_regressor` | 1.154204 | 0.870558 | 0.094919 |
| `residual_extra_trees_scale0.4_clip1.5` | `residual_regressor` | 1.154638 | 0.872321 | 0.097803 |
| `residual_extra_trees_scale0.4_clip2` | `residual_regressor` | 1.154638 | 0.872321 | 0.097803 |
| `residual_extra_trees_scale0.4_clip1` | `residual_regressor` | 1.155568 | 0.873236 | 0.09873 |
| `residual_extra_trees_scale0.5_clip0.75` | `residual_regressor` | 1.156617 | 0.873294 | 0.099027 |
| `residual_extra_trees_scale0.3_clip1.5` | `residual_regressor` | 1.157151 | 0.875967 | 0.101846 |
| `residual_extra_trees_scale0.3_clip2` | `residual_regressor` | 1.157151 | 0.875967 | 0.101846 |
| `residual_extra_trees_scale0.4_clip0.75` | `residual_regressor` | 1.157781 | 0.87548 | 0.102017 |

## `gust`

- split rows: `{'train': 11425, 'validation': 4256, 'holdout': 8595}`
- feature counts: `{'numeric': 13, 'categorical': 2, 'total': 15}`
- selected candidate: `residual_extra_trees_scale0.5_clip1.5`
- selected candidate type: `residual_regressor`
- validation RMSE: `1.343139`
- holdout RMSE: `1.368102`
- full RMSE: `1.49393`
- deltas vs champion: `{'validation_rmse_delta': -0.029994, 'holdout_rmse_delta': -0.002961, 'full_rmse_delta': -0.044488}`
- official gate: `1.484221`

Top validation candidates:

| Candidate | Type | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| `residual_extra_trees_scale0.5_clip1.5` | `residual_regressor` | 1.343139 | 0.991654 | 0.14605 |
| `residual_extra_trees_scale0.5_clip2` | `residual_regressor` | 1.343139 | 0.991654 | 0.14605 |
| `residual_extra_trees_scale0.5_clip1` | `residual_regressor` | 1.344378 | 0.993322 | 0.147919 |
| `residual_extra_trees_scale0.5_clip0.75` | `residual_regressor` | 1.346772 | 0.996661 | 0.151366 |
| `residual_extra_trees_scale0.4_clip1.5` | `residual_regressor` | 1.347183 | 0.997663 | 0.151818 |
| `residual_extra_trees_scale0.4_clip2` | `residual_regressor` | 1.347183 | 0.997663 | 0.151818 |
| `residual_extra_trees_scale0.4_clip1` | `residual_regressor` | 1.348305 | 0.999033 | 0.153313 |
| `residual_extra_trees_scale0.4_clip0.75` | `residual_regressor` | 1.350453 | 1.001705 | 0.15607 |
| `trust_extra_trees_p0.4_a0.4_clip2` | `trust_classifier` | 1.350813 | 1.005731 | 0.171967 |
| `residual_extra_trees_scale0.5_clip0.5` | `residual_regressor` | 1.35085 | 1.001736 | 0.15669 |
