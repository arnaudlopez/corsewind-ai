# Training Shards Status

Generated on `2026-06-25`.

## Current Dataset

Remote root:

```text
/srv/data/corsewind/ml_dataset
```

Current enriched run prefix:

```text
residual_windsup_sst_prev
```

Current training table roots:

```text
/srv/data/corsewind/ml_dataset/training_tables/residual_windsup_sst_prev_YYYY_MM
```

We now have a continuous monthly shard set from `2024-01` to `2026-06`.
June 2026 is naturally partial because source observations and forecasts were
available only up to `2026-06-23/24` at generation time.

The enriched shards include:

- target wind observations from Winds-Up public spot history and DPClim station
  observations;
- raw Open-Meteo `meteofrance_arome_france` forecast prior;
- recent observation and model-error features;
- context-station features;
- Copernicus Marine SST spot samples;
- Open-Meteo Previous Runs `best_match` features for `day1` and `day2`.

## Monthly Shards

| Month | Rows | Parquet bytes | Parquet columns |
| --- | ---: | ---: | ---: |
| `2024-01` | `81,313` | `20,651,286` | `872` |
| `2024-02` | `75,091` | `19,202,820` | `872` |
| `2024-03` | `81,911` | `20,880,642` | `872` |
| `2024-04` | `78,920` | `20,468,129` | `872` |
| `2024-05` | `88,808` | `21,968,387` | `872` |
| `2024-06` | `87,752` | `21,958,589` | `872` |
| `2024-07` | `92,810` | `22,234,212` | `872` |
| `2024-08` | `90,541` | `22,249,261` | `872` |
| `2024-09` | `91,201` | `22,143,012` | `872` |
| `2024-10` | `86,195` | `21,253,315` | `872` |
| `2024-11` | `78,534` | `19,893,777` | `872` |
| `2024-12` | `81,267` | `20,445,508` | `873` |
| `2025-01` | `79,415` | `20,121,399` | `873` |
| `2025-02` | `65,620` | `16,785,400` | `873` |
| `2025-03` | `70,340` | `18,578,833` | `872` |
| `2025-04` | `67,948` | `17,780,454` | `872` |
| `2025-05` | `68,867` | `18,192,669` | `872` |
| `2025-06` | `98,412` | `23,822,085` | `874` |
| `2025-07` | `103,881` | `25,557,689` | `874` |
| `2025-08` | `100,360` | `24,557,920` | `873` |
| `2025-09` | `93,189` | `22,970,953` | `874` |
| `2025-10` | `99,328` | `25,064,096` | `874` |
| `2025-11` | `84,058` | `21,619,924` | `874` |
| `2025-12` | `96,237` | `22,954,416` | `874` |
| `2026-01` | `90,435` | `22,960,351` | `875` |
| `2026-02` | `68,564` | `18,479,066` | `875` |
| `2026-03` | `94,267` | `23,126,770` | `874` |
| `2026-04` | `95,685` | `23,088,693` | `874` |
| `2026-05` | `109,960` | `26,535,992` | `874` |
| `2026-06` | `85,013` | `19,796,791` | `983` |

Total shards: `30`.

Total rows: `2,585,922`.

Rows by year:

| Year | Rows |
| --- | ---: |
| `2024` | `1,014,343` |
| `2025` | `1,027,655` |
| `2026` | `543,924` |

Each shard has:

- JSONL audit table: `training_rows.jsonl`;
- compressed Parquet table: `training_rows.parquet`;
- baseline evaluation: `evaluation.json` / `evaluation.md`;
- generation profile: `training_profile.json`;
- Parquet export profile: `parquet_export_profile.json`.

## Source Coverage

SST sample coverage:

| Source | Value |
| --- | ---: |
| Copernicus Marine SST rows | `203,850` |
| Copernicus Marine valid SST rows | `203,000` |
| Copernicus Marine broken days | `0` |
| Copernicus Marine date span | `2024-01-01` to `2026-06-24` |

Feature-store source coverage across the 30 enriched shards:

| Feature source flag | Rows | Coverage |
| --- | ---: | ---: |
| `sst` | `479,007` | `100.00000%` |
| `previous_run_open_meteo_best_match_day1` | `479,007` | `100.00000%` |
| `previous_run_open_meteo_best_match_day2` | `479,007` | `100.00000%` |
| `model_open_meteo_meteofrance_arome_france` | `478,225` | `99.83675%` |
| `context_stations` | `478,378` | `99.86869%` |

Open-Meteo Previous Runs `day3` was collected but is not attached to the
current feature store yet. The current design attaches `day1` and `day2` only.

## Interpretation

The dataset is now long enough for a serious first training pass. It covers:

- one full recent year: `2025`;
- the available `2024` history back to January;
- the current `2026` season up to the latest available source data.

This addresses the previous issue where the enriched training set started only
at `2024-06`. We no longer need to treat the existing dataset as a short smoke
sample.

The target structure still follows the core modeling strategy:

```text
NWP prior
+ recent local observation error
+ context stations
+ thermal/ocean context
-> local residual correction
```

## Existing Smoke ML Correction

The earlier smoke model was run before the full SST + previous-runs enrichment.
It remains useful as a sanity check that residual correction is learnable, but
it is not the final model pass for the enriched shards.

Smoke settings:

- deterministic row sample: `20,000` rows per month;
- temporal split: last `20%` issue times as test;
- model family: sklearn `ExtraTrees`;
- trees per target: `40`;
- classification thresholds skipped.

| Month | Target | Raw NWP RMSE | Corrected RMSE | RMSE gain |
| --- | --- | ---: | ---: | ---: |
| `2024-06` | `wind_mean_ms` | `1.854727` | `1.469240` | `20.784%` |
| `2024-06` | `gust_ms` | `2.895484` | `1.698223` | `41.349%` |
| `2024-07` | `wind_mean_ms` | `1.750177` | `1.359422` | `22.327%` |
| `2024-07` | `gust_ms` | `2.469430` | `1.481194` | `40.019%` |
| `2024-08` | `wind_mean_ms` | `1.625985` | `1.329176` | `18.254%` |
| `2024-08` | `gust_ms` | `3.056055` | `1.639478` | `46.353%` |

## First Parquet Training Pass

The first Parquet trainer is implemented:

```text
scripts/ml_dataset/train_residual_correction_parquet.py
```

It reads monthly `training_rows.parquet` shards directly, builds a temporal
split, filters empty/sparse/constant features, derives issue-time calendar
features, and trains sklearn models from the flat Parquet surface.

Shared run settings:

| Setting | Value |
| --- | ---: |
| Source rows | `2,585,922` |
| Source shards | `30` |
| Temporal split | `2025-12-26T06:30:00Z` |
| Pre-sample train rows | `2,025,857` |
| Pre-sample test rows | `560,065` |
| Sampled train rows | `300,000` |
| Sampled test rows | `80,000` |
| Feature columns kept | `758` |
| Numeric features | `745` |
| Categorical features | `13` |
| Dropped sparse features | `110` |
| Dropped constant features | `80` |

Regression run:

```text
/srv/data/corsewind/ml_dataset/models/residual_windsup_sst_prev_2024_01_2026_06_hgb_regression_sample
```

| Target | Raw RMSE | Corrected RMSE | RMSE gain | Raw MAE | Corrected MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `labels__residual_wind_mean_ms` | `2.183094` | `1.448212` | `33.662%` | `1.649727` | `1.066787` |
| `labels__residual_gust_ms` | `3.817749` | `1.678889` | `56.024%` | `2.914374` | `1.225667` |

Threshold-probability run:

```text
/srv/data/corsewind/ml_dataset/models/residual_windsup_sst_prev_2024_01_2026_06_hgb_threshold_sample
```

| Target | Test positives | Positive rate | Accuracy | Brier |
| --- | ---: | ---: | ---: | ---: |
| `labels__target_wind_gt_15kt` | `15,004` | `0.187550` | `0.929188` | `0.050418` |
| `labels__target_wind_gt_20kt` | `5,329` | `0.066613` | `0.969550` | `0.022097` |
| `labels__target_gust_gt_20kt` | `10,211` | `0.127637` | `0.948025` | `0.037152` |
| `labels__target_gust_gt_25kt` | `5,722` | `0.071525` | `0.967050` | `0.023658` |

These are sampled training runs, not the final production models. They validate
that the full enriched Parquet dataset is trainable and that residual correction
adds a large signal over the raw NWP prior.

## Operational Notes

The monthly runner is now the preferred command for backfill shards:

```bash
python3 scripts/ml_dataset/run_monthly_training_shards.py \
  --ml-root /srv/data/corsewind/ml_dataset \
  --registry /work/configs/ml_spots.json \
  --context-registry /work/configs/ml_context_stations.json \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --run-id-prefix residual_windsup_sst_prev \
  --command-timeout-sec 7200 \
  --parquet-batch-size 25000 \
  --parquet-compression zstd
```

Current limitations:

- the Parquet export is correct but slow because it infers schema from the JSONL
  in a full pass;
- Parquet training currently uses sampled rows, not the full `2.58M` rows;
- threshold probabilities are trained but not calibrated yet;
- evaluation by `spot_id`, `lead_time_minutes`, target source, and thermal-day
  regime still needs to be added to the Parquet training report;
- `day3` previous-run features are collected but not attached yet.

## Next Steps

1. Add per-spot and per-lead evaluation tables to the Parquet trainer outputs.
2. Add probability calibration and reliability curves for threshold models.
3. Run ablations for context stations, SST, previous runs, and thermal features.
4. Decide whether to attach Open-Meteo Previous Runs `day3`.
5. Package the regression + threshold models for inference in the update engine.
