# Relief Context Window Filter Fix

Date: 2026-06-26

## Problem

The RMSE 0.9 gap audit showed that the largest error contributors are:

- `la_tonnara`, `santa_manza`, and `balistra`;
- high wind rows;
- +45/+60 minute horizons.

The same audit showed that recent observations and SST are well covered, but
relief/mountain context is missing on the critical spots.

Root cause: the context slot selection chose the nearest relief station by
distance, even if that station was inactive during the training window. For the
south Corsica critical spots, `CARBINI-COL DE MELA` (`20061002`) was selected
as `global_relief_1`, but its `station_end` is `2009-06-01`. Therefore the slot
was structurally present but observation-empty for 2024-2026.

## Fix

`scripts/ml_dataset/build_spot_feature_store.py` now filters context stations
against the requested build window:

- exclude a station when `station_end < start_datetime`;
- exclude a station when `station_start > end_datetime`;
- keep the previous behavior when no build window is provided.

This avoids selecting inactive stations for current/historical ML shards.

## Probe

Probe window:

```text
2026-01-01T00:00:00Z -> 2026-01-03T23:59:59Z
```

Remote output:

```text
/srv/data/corsewind/ml_dataset/feature_store/relief_window_filter_probe_2026_01_01_03
```

Critical-spot results:

| Spot | Rows | `global_relief_1` | Available | Wind non-null | Temperature non-null |
| --- | ---: | --- | ---: | ---: | ---: |
| `balistra` | 288 | `20254006` QUENZA | 100% | 100% | 100% |
| `la_tonnara` | 201 | `20160001` MOCA-CROCE | 100% | 100% | 100% |
| `santa_manza` | 203 | `20254006` QUENZA | 100% | 100% | 100% |

The full rebuild of `residual_windsup_sst_prev_regime_v1_2024_01..2026_06` was
then launched on `z2` with the patched feature store.

Log:

```text
/srv/data/corsewind/ml_dataset/run_logs/rebuild_training_shards.log
```

PID:

```text
/srv/data/corsewind/ml_dataset/run_logs/rebuild_training_shards.pid
```

Early rebuild evidence from `2024-01`, last chunk:

```text
feature_source_hits.context_global_relief_1 = 1433
```

## Expected Impact

This does not guarantee RMSE < 0.9 by itself. It fixes a concrete missing signal
identified by the gap audit:

- critical south Corsica spots now receive active mountain/relief observations;
- thermal gradients such as `thermal_relief_minus_coastal_temperature_c` and
  `thermal_coastal_minus_relief_wind_ms` become usable on those spots;
- the next valid test is to retrain the temporal best pipeline after the rebuilt
  shards finish, then rerun the RMSE gap audit.
