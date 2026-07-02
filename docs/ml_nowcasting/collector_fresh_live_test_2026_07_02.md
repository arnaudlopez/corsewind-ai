# Collector Fresh Live Test - 2026-07-02

## Scope

Tested the data currently present on z2 under:

- `/srv/data/corsewind/ml_dataset`
- existing live inference runs under `/srv/data/corsewind/ml_dataset/live_inference`
- fresh collector data for 2026-07-01 and 2026-07-02

The scoring uses only spots with direct Meteo-France observations available for validation:

`cap_corse`, `la_parata`, `lfkf`, `lfkj`, `lfks`, `lfvf`, `lfvh`.

## Collector Health

The collector is producing usable data.

Latest inspected collector state:

- Meteo-France observations: ok, including 6 min and hourly rows for 2026-07-02.
- AROME-PI surface/extra fields: ok, latest run key `aromepi:2026-07-02T05:00:00Z`.
- AROME vertical profiles: ok, latest run key `arome_profiles:2026-07-02T03:00:00Z`.
- Copernicus SST: ok, latest available SST window ending 2026-06-30.
- EUMETSAT cloud mask/type/LST/GII: ok, sampled on 25 spots/context points.

For a fresh 2026-07-01 hindcast, feature generation produced:

- `840` feature-store rows.
- `820` inference/training rows.
- `20` forecasted spots.
- `1262` columns in the inference parquet.
- `nwp_aromepi`: available on all `840` feature rows.
- EUMETSAT cloud mask/type/LST/GII: available on all `840` feature rows.
- SST: available on all `840` feature rows.
- AROME vertical profile: available on `780/840` feature rows.

## Existing Live Run Score

Existing live run:

`/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260630T17_shadow_v2_offset_fallback/shadow_foundation_fresh/predictions_strong_gated_v2_active/predictions.parquet`

Scored against observations:

- target window: `2026-06-30T18:00:00Z` to `2026-06-30T23:00:00Z`
- scored rows: `97`

| Target | Model | RMSE m/s | MAE m/s | Bias m/s |
| --- | --- | ---: | ---: | ---: |
| wind mean | raw | 1.229 | 1.019 | -0.659 |
| wind mean | champion | 0.957 | 0.704 | -0.018 |
| wind mean | strong gated | 0.963 | 0.710 | -0.035 |
| gust | raw | 1.636 | 1.359 | -0.082 |
| gust | champion | 1.256 | 1.012 | -0.242 |
| gust | strong gated | 1.283 | 1.023 | -0.281 |

This was a light-wind evening. The model clearly improved raw AROME/AROME-PI.

## Fresh Collector Hindcast

Fresh hindcast generated from collector data:

`/srv/data/corsewind/ml_dataset/live_inference/collector_fresh_20260701T0645_champion_v1`

Setup:

- AROME-PI run: `2026-07-01T06:00:00Z`
- issue time: `2026-07-01T06:45:00Z`
- target window: `2026-07-01T07:00:00Z` to `2026-07-01T17:00:00Z`
- scored rows: `202`

| Target | Model | RMSE m/s | MAE m/s | Bias m/s |
| --- | --- | ---: | ---: | ---: |
| wind mean | raw | 3.369 | 2.454 | -1.758 |
| wind mean | champion | 3.216 | 2.314 | -1.608 |
| wind mean | strong gated | 2.894 | 2.082 | -1.314 |
| gust | raw | 3.703 | 2.943 | -0.863 |
| gust | champion | 4.278 | 3.256 | -2.835 |
| gust | strong gated | 3.863 | 2.885 | -2.436 |
| gust | high heuristic | 3.756 | 2.891 | -1.410 |

Interpretation:

- Wind mean: ML still improves raw, especially the strong-gated variant.
- Gusts: champion is worse than raw on this strong-wind day.
- Strong-gated helps gusts versus champion, but raw still wins globally on gust RMSE.
- The main error is not random noise. It is a systematic underprediction of strong episodes.

## Strong Wind Failure Mode

For actual gusts above 25 kt on 2026-07-01:

| Model | RMSE m/s | Bias m/s |
| --- | ---: | ---: |
| raw gust | 5.090 | -4.146 |
| champion gust | 6.870 | -6.353 |
| strong-gated gust | 5.838 | -5.162 |
| high heuristic | 5.416 | -4.588 |

Peak examples:

| Spot | Actual wind peak kt | Strong-gated wind peak error kt | Actual gust peak kt | Strong-gated gust peak error kt |
| --- | ---: | ---: | ---: | ---: |
| cap_corse | 35.6 | -15.0 | 43.5 | -18.9 |
| lfvf | 31.3 | -15.5 | 42.2 | -16.2 |
| lfkf | 20.8 | -3.0 | 33.6 | -9.0 |
| lfvh | 22.4 | -2.5 | 30.5 | -5.9 |

Threshold behavior:

- `gust >= 20 kt`: strong-gated CSI `0.608`, precision `0.983`, recall `0.615`.
- `gust >= 25 kt`: raw CSI `0.542`, strong-gated CSI `0.120`, champion CSI `0.0`.
- `wind >= 15 kt`: strong-gated CSI `0.537`, champion/raw CSI `0.325`.

## Opinion

The collector data is good enough to run real live/hindcast tests now.

The current model is useful on normal/light conditions and improves wind mean versus raw NWP. But on the fresh 2026-07-01 strong-wind case it is too conservative, especially for gusts and peaks. This confirms the previous concern: optimizing global RMSE made the model safer on average, but it learned to compress high-wind extremes.

The next improvement should not be another generic feature dump. The next rational step is a dedicated strong-wind correction layer:

- detect strong-wind regime from raw NWP, pressure/vertical profile, context stations, and recent observed acceleration;
- use a conditional expert for `wind >= 15/20/25 kt` and `gust >= 20/25 kt`;
- allow upward corrections when raw NWP and station context indicate a strong episode;
- evaluate with both RMSE and threshold metrics, especially CSI/recall for 20/25 kt gusts.

Do not promote the new `vnext_trust` artifact to live yet. The live parquets scored here do not include its required `vnext` prediction columns, so this test validates the current live pipeline and the strong-gated variant, not the latest `vnext_trust` challenger.
