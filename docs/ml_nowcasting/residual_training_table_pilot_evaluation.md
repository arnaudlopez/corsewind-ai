# Residual Training Table Evaluation

- rows: `168`
- wind RMSE gain, error persistence vs raw: `-9.594`%
- gust RMSE gain, error persistence vs raw: `37.815`%

## Overall

| Baseline | Count | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: | ---: |
| `gust_error_persistence` | 168 | 1.15 | 1.569539 | -0.167857 |
| `gust_raw_nwp` | 168 | 2.067262 | 2.523968 | 1.993452 |
| `wind_error_persistence` | 168 | 0.879881 | 1.105351 | 0.059762 |
| `wind_raw_nwp` | 168 | 0.749048 | 1.008583 | -0.120833 |

## By Lead

| Lead | Baseline | Count | MAE | RMSE | Bias |
| --- | --- | ---: | ---: | ---: | ---: |
| `120 min` | `gust_error_persistence` | 49 | 1.138776 | 1.538221 | -0.102041 |
| `120 min` | `gust_raw_nwp` | 49 | 2.087755 | 2.530185 | 2.022449 |
| `120 min` | `wind_error_persistence` | 49 | 0.754082 | 0.944796 | 0.042653 |
| `120 min` | `wind_raw_nwp` | 49 | 0.745102 | 1.014554 | -0.145102 |
| `180 min` | `gust_error_persistence` | 42 | 1.366667 | 1.75784 | -0.280952 |
| `180 min` | `gust_raw_nwp` | 42 | 1.985714 | 2.430755 | 1.909524 |
| `180 min` | `wind_error_persistence` | 42 | 0.995952 | 1.271123 | 0.040714 |
| `180 min` | `wind_raw_nwp` | 42 | 0.786905 | 1.052461 | -0.15881 |
| `360 min` | `gust_error_persistence` | 21 | 1.485714 | 1.894855 | -0.561905 |
| `360 min` | `gust_raw_nwp` | 21 | 1.966667 | 2.516895 | 1.833333 |
| `360 min` | `wind_error_persistence` | 21 | 0.956667 | 1.152907 | 0.081429 |
| `360 min` | `wind_raw_nwp` | 21 | 0.672381 | 0.926864 | -0.010476 |
| `60 min` | `gust_error_persistence` | 56 | 0.871429 | 1.286884 | 0.007143 |
| `60 min` | `gust_raw_nwp` | 56 | 2.148214 | 2.588953 | 2.091071 |
| `60 min` | `wind_error_persistence` | 56 | 0.874107 | 1.08355 | 0.080893 |
| `60 min` | `wind_raw_nwp` | 56 | 0.752857 | 0.999078 | -0.1125 |

## Threshold Positives

| Label | Count | Positive | Rate |
| --- | ---: | ---: | ---: |
| `target_gust_gt_20kt` | 168 | 0 | 0.0 |
| `target_gust_gt_25kt` | 168 | 0 | 0.0 |
| `target_wind_gt_15kt` | 168 | 0 | 0.0 |
| `target_wind_gt_20kt` | 168 | 0 | 0.0 |
