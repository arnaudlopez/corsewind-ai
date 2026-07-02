# phys_v3 Underperformance Analysis

Generated: 2026-06-30

## Question

Why does the earlier champion beat `phys_v3_dem_fetch`, even though `phys_v3`
has a better organized dataset and richer physical features?

## Short Answer

`phys_v3` is better organized, but it was not a stronger predictive experiment.
It changed several things at once:

- fewer training rows;
- more than twice as many feature columns;
- a lower-capacity benchmark rail;
- many new static features that are not yet direction-conditioned;
- new features whose measured model importance is low, especially maritime fetch.

The new data structure is a better foundation, but the model did not extract
enough useful signal from it to beat the older champion.

## Compared Runs

Champion:

- base run: `tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1`
- calibrator: `prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1`
- final wind mean RMSE: `1.268019`
- final wind mean MAE: `0.930465`

phys_v3:

- base run: `tabular_lgbm_phys_v3_dem_fetch_150k_bin63_2024_2025_to_2026_v1`
- wind calibrator: `prediction_residual_calibrator_phys_v3_dem_fetch_150k_bin63_wind_mean_2025h2_to_2026_v1`
- final wind mean RMSE: `1.305533`
- final wind mean MAE: `0.987419`
- first serious gust RMSE: `1.527744`

## Protocol Differences

| Item | Champion | phys_v3 |
| --- | ---: | ---: |
| Source rows | `2,585,922` | `1,324,473` |
| Unique issue times | `77,423` | `32,133` |
| Max train rows | `225,000` | `150,000` |
| Actual wind train rows | `224,797` | `149,966` |
| Feature columns | `758` | `1,575` |
| Base short-lead RMSE | `1.276846` | `1.314577` |
| Calibrated short-lead RMSE | `1.268019` | `1.305533` |

This is not an apples-to-apples feature-only comparison. `phys_v3` had richer
schema but less training data and a much larger feature space.

## Common-Row Comparison

On rows shared by both final calibrated prediction files:

| Model | Rows | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: | ---: |
| champion | `16,570` | `1.287046` | `0.968034` | `0.073995` |
| phys_v3 | `16,570` | `1.303469` | `0.984130` | `0.143334` |

So the verdict is not only due to different row coverage. On the common subset,
`phys_v3` is still weaker globally.

## Where phys_v3 Helps

`phys_v3` beats the champion by RMSE on 8 spots/stations, but most gains are
small.

| Spot | Champion RMSE | phys_v3 RMSE | Delta |
| --- | ---: | ---: | ---: |
| `cap_corse` | `1.851566` | `1.771847` | `-0.079719` |
| `lfkf` | `1.188975` | `1.132945` | `-0.056030` |
| `lfkj` | `1.015924` | `1.007029` | `-0.008895` |
| `lfvf` | `1.234596` | `1.225810` | `-0.008786` |
| `lfvh` | `1.323670` | `1.314942` | `-0.008728` |
| `figari_eole` | `1.296528` | `1.288377` | `-0.008151` |
| `balistra` | `1.216428` | `1.210281` | `-0.006146` |
| `la_tonnara` | `1.499933` | `1.497591` | `-0.002342` |

Interpretation: `phys_v3` contains a local signal, especially for `cap_corse`
and `lfkf`, but it is not strong or broad enough for global promotion.

## Feature Importance

Champion LightGBM importance by family:

| Family | Importance share |
| --- | ---: |
| context observations | `48.639%` |
| NWP baselines | `22.194%` |
| other/time/location | `17.006%` |
| recent obs/model error | `10.856%` |
| SST/thermal | `1.306%` |

Top champion signals:

- `baselines__baseline_wind_mean_ms`
- `baselines__baseline_shortwave_radiation`
- `features__lead_time_minutes`
- `features__model_error_now_wind_mean_ms`
- `baselines__baseline_gust_ms`
- `features__model_open_meteo_meteofrance_arome_france_wind_speed_10m`
- `baselines__baseline_wind_direction_deg`
- recent observation lags and deltas

phys_v3 LightGBM importance by family:

| Family | Importance share |
| --- | ---: |
| context observations | `29.852%` |
| other/time/location | `22.019%` |
| NWP baselines | `17.852%` |
| recent obs/model error | `11.722%` |
| NWP offsets | `8.130%` |
| previous runs | `5.759%` |
| DEM static features | `2.574%` |
| SST/thermal | `1.167%` |
| vertical profile | `0.833%` |
| fetch static features | `0.093%` |

The important signal is still recent observations, NWP baseline, lead time,
NWP ramp, and model error now. The new fetch features were almost not used.

## Why Better Organization Did Not Improve RMSE

1. Dataset organization is not equivalent to predictive signal.

The `phys_v3` schema is cleaner and more complete, but RMSE improves only if
the model can connect those fields to target errors.

2. `phys_v3` is underpowered relative to its feature count.

It trained on `150k` sampled rows with `1,575` features. The champion trained on
`225k` sampled rows with `758` features. That is a much easier learning
problem for the champion.

3. Static fetch/DEM are not enough.

The useful physical question is not "what is the fetch to the east?". It is:

```text
given the forecast/recent wind direction now,
what is the upwind fetch, lee blocking, cross-shore angle, and terrain exposure?
```

`phys_v3` gives the model sector features, but not the dynamic directional
interaction. LightGBM can learn some interactions, but with sparse/local cases
and many columns this is inefficient.

4. More features dilute tree capacity.

With many additional columns, splits are spent exploring weak or redundant
features. This is visible in importance: fetch accounts for only `0.093%`.

5. The dominant signal is still observation/NWP correction.

Both models mostly rely on:

- raw NWP wind;
- current model error;
- recent observations;
- lead time;
- context stations;
- radiation/season/hour.

The new physical features did not dominate those.

6. `phys_v3` has a higher positive bias.

On common rows:

- champion bias: `+0.073995`
- phys_v3 bias: `+0.143334`

So `phys_v3` tends to overcorrect upward more often.

## Interpretation

The result does not mean DEM/fetch are useless. It means the current
representation is not the right one.

The evidence says:

- DEM/fetch can help some places (`cap_corse`, `lfkf`);
- static fetch sectors are too weak globally;
- dynamic direction-conditioned geometry is the next correct experiment;
- `phys_v3` should be treated as a candidate specialist input, not a global
replacement.

## Next Experiment

Do not rerun another generic rebuild immediately.

Build `phys_v4_directional` features:

- `fetch_upwind_km`
- `fetch_crosswind_left_km`
- `fetch_crosswind_right_km`
- `upwind_relief_blocking_m`
- `lee_index`
- `cross_shore_angle_deg`
- `alongshore_angle_deg`
- `thermal_breeze_alignment`
- `venturi_channel_score`

These should be computed from the forecast/recent wind direction for each row,
not just stored as static sector columns.

Then test two paths:

1. Global model with directional features and same capacity as champion.
2. Router/specialist using `phys_v3/phys_v4` only where it proves better:
   `cap_corse`, `lfkf`, maybe `balistra` and `la_tonnara`.
