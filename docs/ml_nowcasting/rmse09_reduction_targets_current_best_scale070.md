# RMSE 0.9 Reduction Targets

Generated: `2026-06-27T00:01:52.032272Z`
Audit source: `docs/ml_nowcasting/rmse09_gap_audit_current_best_scale070.json`
Current RMSE: `1.268019`
Target RMSE: `0.9`
MSE reduction needed: `49.623%`

## Composite Masks

| Mask | Rows | SSE share | Current RMSE | Required RMSE | Required reduction | Possible alone |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `actual_8plus` | 5258 | 31.99% | 1.753427 | None | None% | False |
| `actual_8plus_or_lead_45_60` | 19426 | 72.785% | 1.376005 | 0.776227 | 43.588% | True |
| `critical_spots` | 12966 | 52.658% | 1.432584 | 0.343938 | 75.992% | True |
| `critical_spots_or_actual_8plus` | 15402 | 64.909% | 1.459334 | 0.708193 | 51.471% | True |
| `critical_spots_or_lead_45_60` | 23230 | 81.89% | 1.334694 | 0.837812 | 37.228% | True |
| `lead_45_60` | 17109 | 60.747% | 1.339495 | 0.573208 | 57.207% | True |
| `lead_60` | 10007 | 36.577% | 1.359074 | None | None% | False |

## spot_id

| Group | Rows | SSE share | Current RMSE | Required RMSE | Required reduction | Possible alone |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `la_tonnara` | 4287 | 19.609% | 1.520349 | None | None% | False |
| `santa_manza` | 4096 | 18.176% | 1.497476 | None | None% | False |
| `balistra` | 4858 | 14.653% | 1.234581 | None | None% | False |
| `porticcio` | 6366 | 12.862% | 1.010447 | None | None% | False |
| `porto_polo` | 3761 | 10.205% | 1.170954 | None | None% | False |
| `piantarella` | 4002 | 10.091% | 1.12882 | None | None% | False |
| `figari_eole` | 1170 | 3.671% | 1.259202 | None | None% | False |
| `cap_corse` | 407 | 2.683% | 1.825083 | None | None% | False |
| `la_parata` | 415 | 1.985% | 1.554714 | None | None% | False |
| `lfvh` | 443 | 1.685% | 1.386233 | None | None% | False |
| `lfkf` | 471 | 1.452% | 1.248207 | None | None% | False |
| `lfvf` | 357 | 1.39% | 1.402601 | None | None% | False |

## lead_time_minutes

| Group | Rows | SSE share | Current RMSE | Required RMSE | Required reduction | Possible alone |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `60.0` | 10007 | 36.577% | 1.359074 | None | None% | False |
| `45.0` | 7102 | 24.17% | 1.31142 | None | None% | False |
| `30.0` | 7045 | 21.802% | 1.250536 | None | None% | False |
| `15.0` | 7275 | 17.451% | 1.10099 | None | None% | False |

## actual_wind_bin_ms

| Group | Rows | SSE share | Current RMSE | Required RMSE | Required reduction | Possible alone |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `8+` | 5246 | 31.917% | 1.753438 | None | None% | False |
| `0-2` | 9343 | 19.844% | 1.036014 | None | None% | False |
| `2-4` | 7265 | 17.204% | 1.093912 | None | None% | False |
| `4-6` | 5491 | 15.869% | 1.208492 | None | None% | False |
| `6-8` | 4084 | 15.165% | 1.369858 | None | None% | False |

## spot_id+lead_time_minutes

| Group | Rows | SSE share | Current RMSE | Required RMSE | Required reduction | Possible alone |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `{'lead_time_minutes': 60.0, 'spot_id': 'la_tonnara'}` | 1091 | 6.279% | 1.705399 | None | None% | False |
| `{'lead_time_minutes': 45.0, 'spot_id': 'la_tonnara'}` | 1065 | 5.38% | 1.597801 | None | None% | False |
| `{'lead_time_minutes': 60.0, 'spot_id': 'santa_manza'}` | 1003 | 4.916% | 1.573746 | None | None% | False |
| `{'lead_time_minutes': 30.0, 'spot_id': 'santa_manza'}` | 1031 | 4.872% | 1.545343 | None | None% | False |
| `{'lead_time_minutes': 45.0, 'spot_id': 'santa_manza'}` | 1028 | 4.655% | 1.512747 | None | None% | False |
| `{'lead_time_minutes': 30.0, 'spot_id': 'la_tonnara'}` | 1076 | 4.552% | 1.462087 | None | None% | False |
| `{'lead_time_minutes': 45.0, 'spot_id': 'balistra'}` | 1262 | 4.251% | 1.30471 | None | None% | False |
| `{'lead_time_minutes': 60.0, 'spot_id': 'balistra'}` | 1153 | 4.06% | 1.333993 | None | None% | False |
| `{'lead_time_minutes': 15.0, 'spot_id': 'santa_manza'}` | 1034 | 3.733% | 1.350661 | None | None% | False |
| `{'lead_time_minutes': 60.0, 'spot_id': 'porticcio'}` | 1561 | 3.638% | 1.085197 | None | None% | False |
| `{'lead_time_minutes': 45.0, 'spot_id': 'porticcio'}` | 1566 | 3.62% | 1.080815 | None | None% | False |
| `{'lead_time_minutes': 30.0, 'spot_id': 'balistra'}` | 1181 | 3.469% | 1.21842 | None | None% | False |
