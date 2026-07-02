# Residual Backfill 2024-06 Results

Run id: `residual_windsup_spots_2024_06`

Remote outputs:

```text
/srv/data/corsewind/ml_dataset/feature_store/residual_windsup_spots_2024_06
/srv/data/corsewind/ml_dataset/training_tables/residual_windsup_spots_2024_06
/srv/data/corsewind/ml_dataset/models/residual_windsup_spots_2024_06_smoke
```

## Dataset

- period: `2024-06-01` to `2024-06-30`
- issue hours: `10:00Z` to `18:00Z`
- residual training rows: `87,752`
- feature-store disk size: `455M`
- training table disk size: `2.5G`
- Parquet export size: `21M`
- Parquet export columns: `864`
- missing baseline wind rows: `0`
- missing target wind rows: `0`

Rows by lead:

| Lead | Rows |
| ---: | ---: |
| `+15m` | `10,486` |
| `+30m` | `10,349` |
| `+45m` | `10,260` |
| `+60m` | `15,032` |
| `+120m` | `14,753` |
| `+180m` | `14,130` |
| `+360m` | `12,742` |

Rows by target observation source:

| Source | Rows |
| --- | ---: |
| `windsup_public_spot_history` | `69,700` |
| `dpclim_station_hourly` | `18,052` |

Rows by spot:

| Spot | Rows |
| --- | ---: |
| `balistra` | `17,941` |
| `figari_eole` | `8,777` |
| `la_tonnara` | `11,097` |
| `piantarella` | `9,900` |
| `porto_polo` | `10,882` |
| `santa_manza` | `11,103` |
| `cap_corse` | `2,580` |
| `la_parata` | `2,572` |
| `lfkf` | `2,580` |
| `lfkj` | `2,580` |
| `lfks` | `2,580` |
| `lfvf` | `2,580` |
| `lfvh` | `2,580` |

## Baseline Check

The table compares raw Open-Meteo `meteofrance_arome_france` against a simple
current-error persistence correction.

Overall:

| Baseline | Count | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: | ---: |
| `wind_raw_nwp` | `87,752` | `1.531976` | `1.986233` | `-0.474164` |
| `wind_error_persistence` | `87,752` | `1.444781` | `1.973822` | `0.021864` |
| `gust_raw_nwp` | `87,752` | `2.528194` | `3.192197` | `1.985507` |
| `gust_error_persistence` | `87,752` | `1.788589` | `2.462871` | `0.032024` |

Gain vs raw NWP:

- wind mean RMSE: `+0.625%`
- gust RMSE: `+22.847%`

## Parquet Export

The nested JSONL audit table was exported to a flat Parquet table:

```text
/srv/data/corsewind/ml_dataset/training_tables/residual_windsup_spots_2024_06/training_rows.parquet
```

Export profile:

- rows: `87,752`
- columns: `864`
- numeric columns: `752`
- string columns: `112`
- compression: `zstd`

This is the preferred surface for fast filtering and model experiments. Keep
the JSONL for audit/debug, but avoid loading it for every training iteration.

## Smoke ML Correction

The full ExtraTrees run on all `87,752` rows was too slow for an interactive
feedback loop with the current JSONL trainer, so it was stopped and replaced by
a deterministic smoke run:

- source rows: `87,752`
- sampled rows: `20,000`
- train rows: `15,939`
- test rows: `4,061`
- model family: sklearn `ExtraTrees`
- trees per target: `40`
- classification thresholds skipped for this smoke run

Regression results:

| Target | Raw NWP RMSE | Corrected RMSE | RMSE gain |
| --- | ---: | ---: | ---: |
| `wind_mean_ms` | `1.854727` | `1.469240` | `20.784%` |
| `gust_ms` | `2.895484` | `1.698223` | `41.349%` |

This is not yet the production training result, but it is a strong sanity check:
the WindsUp-aware dataset contains learnable local correction signal.

Threshold positives:

| Label | Positive | Rate |
| --- | ---: | ---: |
| `target_wind_gt_15kt` | `24,410` | `0.278170` |
| `target_wind_gt_20kt` | `5,369` | `0.061184` |
| `target_gust_gt_20kt` | `16,330` | `0.186093` |
| `target_gust_gt_25kt` | `6,865` | `0.078232` |

## Interpretation

This run is the first usable windsurf-oriented historical training shard:

```text
Open-Meteo AROME prior + recent spot observations + context stations
-> residual correction labels at 15/30/45/60/120/180/360 min
```

The important validation is not the simple persistence baseline itself. It is
that the dataset now contains true spot-level WindsUp targets for the coastal
windsurf spots, while keeping official Météo-France stations as additional
target/context anchors.

The next scale-up should be monthly or seasonal shards, not one giant combined
2024-2026 JSONL file. The June shard is already `2.5G` as JSONL but only `21M`
as Parquet. Once enough shards are built, training should read Parquet directly,
filter rare/empty columns, and evaluate by spot, lead time, target source, and
thermal-day regime.
