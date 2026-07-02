# SAPHIR vs CorseWind Dataset Comparison

Date: 2026-06-30

## Objective

Understand how the SAPHIR/Baggio et al. v2 code organizes data, compare it with
the current CorseWind dataset/pipeline, and derive the next model-ready dataset
shape.

Reference SAPHIR v2 archive:

- `/srv/data/corsewind/reference/saphir_predict/v2/Zenodo2026.zip`
- extracted code: `/srv/data/corsewind/reference/saphir_predict/v2/extracted_code/Zenodo_BM_2026`
- verified MD5: `61add72af3e68f51f3df835cd22164e3`

Current CorseWind data root:

- `/srv/data/corsewind/ml_dataset`

## Executive Conclusion

SAPHIR and CorseWind are solving the same family of problem, but our data is not
organized like theirs yet.

SAPHIR keeps the physical structure of the inputs until the neural model:

- station time sequence;
- target station plus neighboring stations;
- AROME spatial grids;
- ARPEGE vertical/spatial grids;
- constants/context vector;
- train-only normalization;
- probabilistic wind head.

CorseWind currently has more operational richness and better cadence, but most
of it is flattened early into a wide tabular residual-correction table:

- one row per spot, issue time, target time and lead;
- many point-sampled and engineered columns;
- NWP prior plus residual labels;
- context stations as slots and aggregates;
- sequence benchmark derived from the flat table, not from a SAPHIR-like tensor
  sample.

The key deduction is simple: adding more flat features is not the same as giving
the model the structured problem. The next serious experiment should build a
CorseWind "SAPHIR-style" tensor dataset, then compare it fairly against the
current tabular champion.

## Current CorseWind Dataset Snapshot

From z2 on 2026-06-30:

| Family | Shards | Rows | Columns | Spots | Date range | Leads |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `residual_windsup_sst_prev` | 153 | 8,222,690 | 872-2008 | 14 | 2024-01-01 to 2026-06-24 | 15, 30, 45, 60, 120, 180, 360 min |
| `residual_windsup_sst_prev_phys_v1` | 30 | 1,324,473 | 1424-1550 | 14 | 2024-01-01 to 2026-06-24 | 15, 30, 45, 60, 120, 180, 360 min |
| `residual_windsup_sst_prev_phys_v3_dem_fetch` | 30 | 1,324,473 | 1862-1972 | 14 | 2024-01-01 to 2026-06-24 | 15, 30, 45, 60, 120, 180, 360 min |
| `residual_windsup_spots` | 19 | 1,635,949 | 864-866 | 14 | 2024-06-01 to 2025-12-31 | 15, 30, 45, 60, 120, 180, 360 min |

Current best locked wind-mean champion from the scientific diagnostic:

- run: `prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1`
- rows: 31,429
- RMSE: 1.268019 m/s
- MAE: 0.930465 m/s
- raw prior RMSE: 2.187306 m/s

The first full sequence benchmark exists, but its exported sequence is thin
compared with SAPHIR:

- `past_context.parquet`: 96 x 15 min context rows per case, 10 columns
- `future_covariates.parquet`: 4 future rows per case, 8 columns
- no context-station tensor;
- no 2D NWP grid tensor;
- no vertical profile tensor;
- no rich static/geometry tensor.

For `sequence_2026_windsurf_1h_rmse09_v1`, this produced:

- 40,320 past rows;
- 1,680 future rows;
- 1,680 prediction rows;
- 15-minute horizon from +15 to +60 min.

## Structural Comparison

| Dimension | SAPHIR v2 | CorseWind current | Deduction |
| --- | --- | --- | --- |
| Primary grain | `station_yyyymmdd_index` sample | `spot_id + issue_time + target_time + lead` row | Our supervised identity is better for live nowcasting. Keep it. |
| Cadence | hourly samples | 15 min rows, with some 6 min source observations | Our cadence is better for windsurf. Do not downgrade to hourly. |
| Target | observed station wind speed | wind mean, gust, direction, thresholds and residuals | We are richer. Keep wind mean and gust as first-class targets. |
| Target strategy | mostly direct prediction | NWP prior plus residual correction | Our residual strategy is more operationally appropriate. Keep it. |
| Station history | 7 timesteps, target + 10 neighbors, 3 variables | flat lags/deltas plus context station slots/aggregates | We need a true station-history tensor, not only flattened summaries. |
| Neighbor stations | nearest stations selected in builder | role-based context stations, coastal/inland/relief/global, age and geometry | Our selection logic is more domain-aware, but the model receives it too flattened. |
| NWP surface fields | AROME tensor `(6, 11, 11, 5)` | mostly point/offset features and target-horizon baselines | Biggest missing structural difference: no real 2D grid tensor in training. |
| Vertical atmosphere | ARPEGE tensor `(6, 7, 5, 5, 4)` | Open-Meteo pressure-level features and limited native AROME profile samples | We have useful vertical scalar features, but not a vertical/spatial tensor. |
| Static context | time, day, lat, lon, altitude, neighbor geometry | time, spot static DEM/fetch, land/sea, context geometry, provider freshness | Our static context is richer, but direction-conditioned interaction is still weak. |
| Raw storage | MeteoNet -> SAPHIR station/day NetCDF -> per-station HDF5 | provider-specific JSONL/parquet -> feature store -> monthly training parquets | Our storage is operationally flexible but less canonical for model training. |
| Model input | multiple tensors consumed by neural encoders | wide flat table for LightGBM/HGB/ExtraTrees plus thin sequence export | We need a model-ready tensor export layer. |
| Normalization | per-source mean/std on train keys | tree models mostly raw; sequence/foundation mostly unnormalized or model-specific | For neural models, add train-only normalization metadata. |
| Split | fixed train/validation/test key lists from 2016-2018 | chronological train/eval split, usually 2024/2025 -> 2026 | Our split is safer for live forecasting. Do not copy SAPHIR's random/day split blindly. |
| Probabilistic output | dedicated probabilistic head and threshold metrics | some quantiles from foundation models; tabular direct/residual mostly point | Need probabilistic residual/wind outputs as a first-class supervised objective. |

## What SAPHIR Does Better

1. It preserves spatial structure.

The model sees AROME as a small image around the station, not just the nearest
grid point or hand-engineered offset gradients. That lets a CNN/LSTM learn
upstream gradients, coastal discontinuities and local displacement patterns.

2. It preserves station-neighbor time structure.

The target station and ten neighbors are presented as one coherent sequence.
Our current context station features are good, but a tree model sees them as
hundreds of individual columns. It cannot easily learn propagation as a
structured sequence.

3. It gives the model vertical atmosphere as a tensor.

SAPHIR's ARPEGE block keeps pressure levels and spatial grid together. Our
vertical features are currently derived scalars. Useful, but a weaker
representation for learning stability, shear and mixing structure.

4. It has a clean model boundary.

SAPHIR has a clear model-ready dictionary:

- `station`
- `arome`
- `arpege`
- `const`
- `obs`

CorseWind has a strong ingestion system, but the model boundary is currently
spread across feature store, training table, sequence exports, calibrators and
benchmarks.

5. It trains the architecture on the same structure it wants to exploit.

Our foundation-model benchmark was not a fair test of a SAPHIR-like approach:
it was zero-shot/univariate or thin-covariate sequence forecasting, not a
locally supervised multi-source neural residual model.

## What CorseWind Does Better

1. It is operationally aligned.

Our row identity uses issue time, target time and lead time. That matches live
forecasting and avoids ambiguity about what is known at forecast time.

2. It has better cadence for the business problem.

SAPHIR trains hourly. Windsurf nowcasting needs +15, +30, +45, +60 min, and it
can benefit from 6-minute or 15-minute observations.

3. It predicts the thing we actually need.

We have wind mean, gusts and windsurf thresholds. SAPHIR's wind setup focuses
on wind speed.

4. It keeps data freshness and missingness.

For live heterogeneous observations, `age_minutes`, `available`, source
identity and forward-fill flags matter. SAPHIR is cleaner because MeteoNet is a
research dataset; live CorseWind is messier, so freshness must stay explicit.

5. It has domain-specific context.

Coastal/inland/relief station roles, DEM, fetch, SST, land-surface temperature,
cloud type and instability are all physically relevant for Corsican thermal
wind. SAPHIR does not model this business/domain layer explicitly.

6. It uses a leakage-safer validation protocol.

Strict chronological validation is better for production than random splits
over the same years.

## Why More Flat Features Did Not Solve RMSE

The `phys_v3_dem_fetch` result is the warning sign.

It added richer physical/schema features but did not beat the older champion:

- champion final RMSE: 1.268019
- `phys_v3_dem_fetch` final RMSE: 1.305533

The reason is not that DEM/fetch/physical signals are useless. It is that:

- the model had fewer training rows;
- the feature space got much wider;
- many new features were static or weakly direction-conditioned;
- the dominant signal remains recent observations, NWP prior, current model
  error and context stations;
- the physical structure is still mostly flattened.

So the lesson is: do not keep adding columns blindly. Build fewer, better
structured inputs.

## Main Missing Piece

The missing piece is a canonical model-ready dataset with SAPHIR-like structure
but CorseWind semantics.

Proposed object name:

```text
corsewind_saphir_sequence_v1
```

One sample should represent:

```text
spot_id + issue_time_utc + forecast horizon block
```

with all information known at issue time, then targets at future leads.

## Proposed CorseWind-SAPHIR Dataset V1

### `samples.parquet`

One row per sample:

- `sample_id`
- `spot_id`
- `issue_time_utc`
- `split`
- `available_leads`
- target columns by lead:
  - `target_wind_mean_ms_lead_15`
  - `target_gust_ms_lead_15`
  - ...
- baseline columns by lead:
  - `baseline_wind_mean_ms_lead_15`
  - `baseline_gust_ms_lead_15`
  - ...
- residual labels by lead:
  - `residual_wind_mean_ms_lead_15`
  - `residual_gust_ms_lead_15`
  - ...

### `station_history`

Tensor:

```text
[sample, time, station_slot, variable]
```

Initial shape target:

```text
[N, 96, 11, V]
```

where:

- `96` = 24 h at 15 min cadence;
- `11` = target spot/station + 10 context stations;
- `V` should include at least wind U, wind V, wind speed, gust, temperature,
  pressure, humidity, age, available/missing mask.

For true 6-minute sources, either add a second high-frequency tensor for the
last 2-3 hours, or aggregate carefully to 15 min with count/min/max/trend.

### `nwp_future_surface`

Tensor:

```text
[sample, lead, grid_point, variable]
```

V1 can use our available offset points:

- center;
- north/south/east/west offsets;
- optional diagonal offsets later.

This is not as good as SAPHIR's 11x11 AROME image, but it is a structured
bridge from the current flat `nwp_offset_*` columns.

V2 should move toward true local AROME/AROME-PI grids when historical run
coverage is available.

### `nwp_vertical_profile`

Tensor:

```text
[sample, lead, pressure_level, variable]
```

Initial pressure levels:

- 1000 hPa
- 950 hPa
- 925 hPa
- 900 hPa
- 850 hPa

Variables:

- temperature;
- relative humidity;
- geopotential height;
- wind speed or U/V;
- vertical velocity when available.

This uses our Open-Meteo historical pressure-level coverage first. Native
Meteo-France AROME vertical profiles can be used for recent/live enrichment.

### `static_context`

Vector:

- time of day and day of year cyclic features;
- spot lat/lon/altitude;
- DEM exposure;
- land/sea/fetch features;
- direction-conditioned fetch/relief once available;
- station-slot geometry: distance, bearing, altitude delta, upwind alignment.

### `availability_masks`

Separate masks are important:

- station variable missingness;
- station age;
- NWP availability;
- satellite availability;
- vertical profile availability.

This is one place where CorseWind should improve over SAPHIR, because live data
is heterogeneous and irregular.

## What To Reuse From SAPHIR

Reuse conceptually:

- source-separated tensors;
- target station plus neighbors;
- wind U/V representation;
- spatial NWP context;
- vertical profile context;
- train-only normalization;
- probabilistic wind/threshold outputs.

Do not copy blindly:

- hourly-only sampling;
- random/frozen split style;
- direct wind-only target;
- MeteoNet-specific NetCDF assumptions;
- exact architecture before proving our tensor export is correct.

## Practical Next Experiments

1. Build `export_corsewind_saphir_sequence_dataset.py`.

It should read the existing monthly `training_rows.parquet` shards and export a
compact tensor dataset. No new provider backfill is needed for V1.

2. Start with a controlled subset.

Use:

- spots: `la_tonnara`, `santa_manza`, `balistra`, `piantarella`, `porticcio`,
  `figari_eole`, `porto_polo`;
- issue hours: 08-17 UTC;
- leads: 15, 30, 45, 60 min;
- train: 2024-2025;
- test: locked 2026.

3. Train three comparable models.

- Current tabular champion baseline.
- SAPHIR-style deterministic residual neural model.
- SAPHIR-style probabilistic/quantile residual neural model.

4. Keep acceptance gates hard.

Promote the new dataset/model only if it improves at least one of:

- full locked 2026 RMSE below 1.20;
- +45/+60 min RMSE below 1.15;
- critical spots RMSE below 1.10;
- high-wind regime RMSE below 1.35 without worsening light-wind bias.

5. Use foundation models only after the sequence is rich enough.

Chronos/TimesFM/Moirai should be retested with richer sequences, but the next
most scientific test is not another zero-shot run. It is a supervised
CorseWind-SAPHIR model trained on the structured tensors.

## Bottom Line

Our data volume and provider coverage are now strong enough to move beyond
"wide tabular residual correction".

The highest-value deduction from SAPHIR is not a single model choice. It is the
data interface:

```text
structured station history
+ structured NWP future context
+ structured vertical atmosphere
+ static local context
+ residual/probabilistic targets
```

That is the missing bridge between our current operational collector and the
kind of model architecture used in the reference study.
