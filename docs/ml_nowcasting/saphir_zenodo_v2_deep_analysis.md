# SAPHIR Zenodo V2 Deep Analysis

Date: 2026-06-30

## Objective

Understand exactly what the SAPHIR/Baggio reference code and Zenodo V2 data do,
then compare that design with the current CorseWind SAPHIR-style V1 benchmark.

This note is intentionally operational: its goal is to decide the next
experiment most likely to improve CorseWind wind-mean RMSE.

Follow-up strategy note: `docs/ml_nowcasting/saphir_plus_best_practices_strategy.md`.

## Source Inspected

- Paper: `arXiv:2503.18797v2`
- Zenodo concept DOI: `10.5281/zenodo.15222910`
- Inspected record: Zenodo record `20327672`, version `v 2.0`
- z2 archive: `/srv/data/corsewind/reference/saphir_predict/v2/Zenodo2026.zip`
- Extracted tree: `/srv/data/corsewind/reference/saphir_predict/v2/extracted_code/Zenodo_BM_2026`
- Main inspected code:
  - `script_examples/loop_build_dics.py`
  - `script_examples/benchmark_wind_Corsica.py`
  - `src/saphir_predict/dataproc/dic_from_saphir_V2.py`
  - `src/saphir_predict/models/neural_network_utils_pytorch.py`
  - `src/saphir_predict/models/neural_network_utils_pytorch_proba.py`
  - `src/saphir_predict/models/pytorch_layers.py`
  - `src/saphir_predict/models/baseline_models.py`
  - `src/saphir_predict/data_generators/data_generation_pytorch.py`
  - `storage/setup_files/setup_wind.yaml`

## Zenodo Data Inventory

The V2 archive contains one ready-made Corsica/Ajaccio station dictionary:

```text
storage/dics/dic_stations/20004002/data_20004002.h5
storage/dics/dic_stations/20004002/features_20004002.pkl
```

Important caveat: the archive does not include the frozen global partition
files used by the scripts:

```text
id_training_full.pkl.gz
id_validation_full.pkl
id_test_full.pkl
```

So the data and model logic are inspectable, but the exact paper benchmark
cannot be reproduced byte-for-byte from the archive alone unless those split
files are regenerated or recovered.

## HDF5 Structure

The HDF5 is a dictionary-like store keyed by sample ids such as:

```text
20004002_20160101_6
20004002_20160101_7
...
20004002_20181230_16
```

Top-level groups:

```text
arome
arpege
const
keys
obs
shape
station
```

Observed counts and shapes:

| Source | Shape per sample | Interpretation |
| --- | ---: | --- |
| `station` | `(7, 33)` | 7 hourly timesteps, target station + 10 neighbors, 3 vars each |
| `arome` | `(6, 11, 11, 5)` | 6 forecast horizons, 11x11 grid, 5 surface variables |
| `arpege` | `(6, 7, 5, 5, 4)` | 6 horizons, 7 vertical levels, 5x5 grid, 4 variables |
| `const` | `(47,)` | local time/position plus 10 neighbor geometry blocks |
| `obs` | `(6,)` | target wind speed at horizons +1h to +6h |

Ajaccio dictionary:

- keys: `15796`
- date span in keys: `2016-01-01` to `2018-12-30`
- source period in metadata: `2016-01-01` to `2018-12-31`
- learning cadence: hourly
- horizons: `[1, 2, 3, 4, 5, 6]`
- past window: `6`, which yields 7 station rows including current time
- neighbors: `10`

Sample stats over the first 512 keys confirm that source tensors are already
normalized, while `obs` remains in physical wind speed units:

| Source | Mean | Std | P50 | P90 |
| --- | ---: | ---: | ---: | ---: |
| `station` | `-0.198` | `1.134` | `-0.298` | `1.235` |
| `arome` | `-0.017` | `1.276` | `-0.258` | `1.739` |
| `arpege` | `-0.219` | `1.254` | `-0.421` | `1.672` |
| `const` | `0.123` | `0.245` | `0.006` | `0.419` |
| `obs` | `2.632` | `1.761` | `2.300` | `4.900` |

## Dataset Construction Logic

The wind dictionary builder in `loop_build_dics.py` uses:

```text
time_step = 60 minutes
past = 6
horizon = [1, 2, 3, 4, 5, 6]
nNeigh = 10
field_out = ["wind_speed_station"]
predictionScale = 1
```

Input variables before U/V conversion:

| Source | Variables |
| --- | --- |
| `station` | wind speed, wind direction, temperature |
| `arome` | relative humidity, wind speed, wind direction, temperature 2m, sea-level pressure |
| `arpege` | wind speed, wind direction, temperature, pressure |

After `add_UV_to_dic`, wind speed/direction are converted into eastward and
northward components:

| Source | Variables after conversion |
| --- | --- |
| `station` | `wind_eastward_`, `wind_northward_`, `temperature_` |
| `arome` | `relative_humidity_`, `wind_eastward_U`, `wind_northward_V`, `temperature_2m_`, `sealevel_pressure_` |
| `arpege` | `wind_eastward_U`, `wind_northward_V`, `temperature_`, `pressure_` |

### Key Point: Neighbor History Is In The Station Tensor

For each sample, SAPHIR slices the target station history:

```text
sample_index - past ... sample_index
```

Then it concatenates the same time window from each neighbor station along the
feature axis. With 10 neighbors and 3 station variables:

```text
7 timesteps x ((1 target + 10 neighbors) * 3 vars) = (7, 33)
```

This is not a simple current-time neighbor snapshot. It lets the model learn
propagation, delays, and recent upwind/downwind evolution.

### NWP Is Preserved As Horizon-Structured Tensors

For each sample and horizon, the builder selects a model field valid near the
target timestamp:

- AROME tolerance: 90 minutes
- ARPEGE tolerance: 180 minutes

If any requested source cannot serve a sample, the whole sample is skipped.
This keeps `obs`, `station`, `arome`, `arpege`, and `const` perfectly aligned.

The builder also skips horizons crossing the reference day, because the SAPHIR
daily files contain one forecast issue per day.

### Static Context Is Compact

`const` has only 47 values:

- 7 local values:
  - cosine/sine hour
  - cosine/sine day of year
  - latitude / 100
  - longitude / 100
  - altitude / 1000
- 10 neighbor blocks x 4 values:
  - delta latitude / 100
  - delta longitude / 100
  - altitude delta / 10000
  - distance / 100

This is much smaller and cleaner than our current `static_context` table.

## Normalization Logic

SAPHIR normalizes each source with train-only statistics, but keeps the full
tensor position shape:

```text
station mean/std: (7, 33)
arome mean/std:   (6, 11, 11, 5)
arpege mean/std:  (6, 7, 5, 5, 4)
```

This means each horizon, grid point, pressure level, station slot, and variable
position gets its own normalization. The constants are not normalized in the
same function.

This matters for us because our current neural V1 normalizes more by feature
families or channel axes after reshaping. That is convenient, but it is not the
same inductive bias.

## Model Architecture

The deterministic model `NeuralNetwork_PyTorch` and probabilistic model
`NeuralNetwork_PyTorch_Proba` both process sources according to tensor shape.

### Station Branch

`station` shape `(T, C)` goes into `LSTMWithContext`:

- local context encoder: first 8 values from `const`
- neighbor context encoder: remaining neighbor geometry values from `const`
- input at every timestep is station tensor + repeated local context + repeated
  neighbor context
- output is an LSTM embedding

### AROME Branch

`arome` shape `(T, H, W, C)` goes into `LSTM2DWithContext`:

- 2D convolution on each horizon/grid slice
- average pooling
- flatten
- repeated local context
- stacked LSTMs over forecast horizons

### ARPEGE Branch

`arpege` shape `(T, D, H, W, C)` goes into `LSTM3DWithContext`:

- 3D convolution over vertical/spatial cube
- flatten
- repeated local context
- LSTM aggregation over horizons

### Aggregation

Source embeddings are concatenated and passed through a dense block. For the
deterministic model, local context is appended again before the dense block.
For the probabilistic wind model, source embeddings go into a probabilistic
head.

### Probabilistic Wind Head

For wind mode, the probabilistic model outputs three vectors per horizon:

- `nu`: positive location via `Softplus`
- `sigma`: scale through exponential inside the loss
- `eps`: bounded heterogeneity parameter via scaled sigmoid, max `0.25`

The loss is a Rice-like likelihood with Gauss-Hermite quadrature. The model
learns a distribution of positive wind speed, not only a point residual.

## Baselines

SAPHIR includes two important sanity baselines:

- raw AROME/ARPEGE: center grid point, U/V converted back to wind speed when
  needed;
- persistence: last station observation repeated across all horizons.

This is important because the neural model is not evaluated in isolation. It is
compared against physically meaningful simple forecasts.

## Current CorseWind V1 Comparison

Our current medium SAPHIR-style V1 export:

```text
/srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1
```

Rows:

| Table | Rows |
| --- | ---: |
| `samples.parquet` | `5600` |
| `future_targets.parquet` | `22400` |
| `station_history.parquet` | `179200` |
| `context_station_snapshot.parquet` | `42400` |
| `nwp_surface_offsets.parquet` | `22400` |
| `nwp_vertical_profile.parquet` | `27900` |
| `static_context.parquet` | `5600` |

Best medium results:

| Target | Best model | RMSE |
| --- | --- | ---: |
| wind mean | HGB/Ridge flat residual | about `1.328` |
| gust | Ridge flat residual | about `1.522` |
| wind mean neural V1 | structured residual net | about `1.366` to `1.401` |
| gust neural V1 | structured residual net | about `1.631` to `1.667` |

So the flat models beat our neural V1 on this medium dataset.

## Why This Is Not As Weird As It Looks

The current neural benchmark is not yet a faithful SAPHIR replica.

Major differences:

| Topic | SAPHIR V2 | CorseWind current V1 |
| --- | --- | --- |
| Sample unit | one issue sample predicts all horizons | one row per lead in neural target table |
| Target | direct wind speed vector `(6,)` | residual wind/gust per row |
| Station context | target + 10 neighbor histories inside `(7, 33)` | target history only; neighbors mostly current snapshot |
| NWP surface | true 11x11 AROME grid per horizon | offset table / flattened selected offsets |
| Vertical NWP | ARPEGE 7-level 5x5 cube per horizon | point vertical profile, currently not equivalent |
| Static context | compact 47 values | large selected static table, often hundreds of features |
| Normalization | train-only per tensor position | train-only, but less position-specific after reshaping |
| Architecture | LSTM + Conv2D/Conv3D + context repeated into branches | GRU + masked set MLPs + flat vertical/static MLPs |
| Output | multi-horizon trajectory | independent lead rows with lead as feature |
| Loss | deterministic MSE or probabilistic Rice likelihood | residual SmoothL1/Huber |

The flat Ridge/HGB models can perform better because they are well matched to
our current representation: wide, sparse-ish, partially flattened engineered
features. The neural model is being asked to infer structure after we already
removed or diluted some of the structure SAPHIR relies on.

## Main Scientific Interpretation

The most likely missing signal is not "one more physical scalar". It is the
dynamic propagation signal from neighboring stations.

SAPHIR's station tensor is the clearest design choice:

```text
target station recent history
+ neighbor station recent histories
+ neighbor geometry
```

Our current context stations are useful, but mostly as present-time descriptors.
That can help a tabular model, but it gives a sequence model much less temporal
information about how wind changes travel through the local network.

For CorseWind short horizons, this is probably even more important than in
SAPHIR because our horizons are +15/+30/+45/+60 minutes. Propagation,
acceleration, and delayed thermal onset are exactly the signals we need.

## Next Best Experiment

Build `corsewind_saphir_dictionary_v2`, closer to SAPHIR's actual data model.

### Required Dataset Changes

One sample should be:

```text
spot_id + issue_time
```

Not:

```text
spot_id + issue_time + lead
```

Targets should be multi-horizon vectors:

```text
wind_mean_target: [lead_15, lead_30, lead_45, lead_60]
gust_target:      [lead_15, lead_30, lead_45, lead_60]
```

Core tensors:

| Tensor | Proposed shape | Notes |
| --- | ---: | --- |
| `station` | `(T, (1 + K) * F)` | target + context station histories |
| `nwp_surface` | `(H, O, V)` or `(H, Y, X, V)` | preserve horizon dimension; offsets are acceptable first |
| `vertical` | `(H, L, V)` | pressure profile per lead when available |
| `const` | compact vector | local time/position + neighbor geometry + optional static terrain/fetch |
| `obs` | `(H, targets)` | multi-horizon target vector |

Start with:

- `T`: 16 to 32 steps at 15-minute cadence, or 10 to 16 steps at mixed/native
  cadence if 6-minute data are reliable;
- `K`: 8 to 12 context stations;
- `H`: `[15, 30, 45, 60]`;
- station variables: wind U/V, wind speed, gust if available, temperature,
  pressure, freshness/missingness;
- context station selection: nearest + upwind + coastal + mountain/relief role,
  but represented as histories, not only snapshots.

### Required Model Changes

Train a true SAPHIR-like neural model:

1. Station branch: LSTM/GRU over the concatenated target+neighbor history tensor.
2. Surface NWP branch: horizon-aware encoder over offsets/grid.
3. Vertical branch: horizon-aware profile encoder.
4. Compact static/context branch.
5. Multi-horizon output head.
6. Compare three targets:
   - direct wind speed/gust;
   - residual over NWP;
   - probabilistic quantiles or Gaussian/Rice-like distribution.

The first acceptance gate should be simple:

```text
Does the same model beat the flat Ridge/HGB V1 on the exact same sample set?
```

Then only if that passes, scale up to the full dataset and more expensive
probabilistic training.

## Concrete Next Steps

1. Implement a dictionary/tensor exporter inspired by SAPHIR:
   `export_corsewind_saphir_dictionary_dataset.py`.
2. Include context station histories, not only context snapshots.
3. Store arrays as `.npz` or HDF5 with a manifest:
   - `sample_ids`
   - `station`
   - `nwp_surface`
   - `vertical`
   - `const`
   - `obs_wind`
   - `obs_gust`
   - train/eval split masks
   - train-only normalization stats
4. Add an audit:
   - no future observation in station history;
   - all target timestamps are after issue time;
   - all horizons complete;
   - no eval data in normalization.
5. Train a small SAPHIR-like model on the medium sample set.
6. Benchmark against:
   - raw NWP;
   - persistence;
   - current flat Ridge/HGB on the same samples.
7. Only then run the heavier full dataset.

## Decision

Do not spend the next iteration on more flat static/physical columns. We have
already seen that richer flat physical features do not automatically improve
RMSE.

The best next move is to repair the structural mismatch with SAPHIR:

```text
neighbor histories + multi-horizon output + position-specific normalization
```

This is the most plausible explanation for why our neural benchmark did not
beat the smaller flat models, and it is the cleanest experiment to run next.
