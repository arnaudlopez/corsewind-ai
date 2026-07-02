# CorseWind SAPHIR-Style Sequence V1 Benchmark

Generated on 2026-06-30.

## Why This Exists

The SAPHIR/Baggio reference code keeps the learning problem structured:

- station history as a time tensor;
- neighboring station context as spatial/context inputs;
- NWP fields as structured forecast tensors;
- train-only normalization;
- residual/local correction rather than replacing the NWP model.

Our previous CorseWind training rows were rich, but flattened very early. This
iteration creates an intermediate SAPHIR-style dataset and a first benchmark on
top of it. The benchmark is intentionally conservative: before building a
heavier neural model, the same data must beat raw NWP and persistence with a
simple, leakage-safe supervised residual correction.

## Scripts Added

- `scripts/ml_dataset/export_corsewind_saphir_sequence_dataset.py`
- `scripts/ml_dataset/audit_corsewind_saphir_sequence_dataset.py`
- `scripts/ml_dataset/benchmark_corsewind_saphir_sequence_dataset.py`

## Dataset Shape

Medium export on z2:

```text
/srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1
```

Command:

```bash
cd /srv/data/corsewind/backfill_runner
/home/z2/corsewind-ml-smoke/.venv/bin/python \
  scripts/ml_dataset/export_corsewind_saphir_sequence_dataset.py \
  --training-table-root /srv/data/corsewind/ml_dataset/training_tables \
  --run-id-prefix residual_windsup_sst_prev_phys_v3_dem_fetch \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --output-root /srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1 \
  --lead-minutes 15,30,45,60 \
  --context-length 32 \
  --max-samples-per-spot 400 \
  --issue-hour-start 8 \
  --issue-hour-end 17 \
  --train-end 2025-12-31T23:59:59Z \
  --eval-start 2026-01-01T00:00:00Z
```

Exported tables:

| Table | Rows | Columns |
| --- | ---: | ---: |
| `samples.parquet` | 5600 | 67 |
| `future_targets.parquet` | 22400 | 26 |
| `station_history.parquet` | 179200 | 27 |
| `context_station_snapshot.parquet` | 42400 | 38 |
| `nwp_surface_offsets.parquet` | 22400 | 39 |
| `nwp_vertical_profile.parquet` | 27900 | 12 |
| `static_context.parquet` | 5600 | 1206 |

Audit:

```bash
/home/z2/corsewind-ml-smoke/.venv/bin/python \
  scripts/ml_dataset/audit_corsewind_saphir_sequence_dataset.py \
  --dataset-root /srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1
```

Verdict: `pass`.

The audit checked that target times are after issue times, lead minutes match
the issue-to-target delta, and station history never uses timestamps after the
issue time.

## Medium Benchmark

Main benchmark output:

```text
/srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1/benchmark
```

Command:

```bash
/home/z2/corsewind-ml-smoke/.venv/bin/python \
  scripts/ml_dataset/benchmark_corsewind_saphir_sequence_dataset.py \
  --dataset-root /srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1 \
  --output-root /srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1/benchmark \
  --model-family hgb \
  --model-family extra_trees \
  --model-family ridge \
  --max-numeric-features 500 \
  --min-feature-non-null 50 \
  --hgb-max-iter 180 \
  --extra-trees-estimators 180 \
  --extra-trees-min-samples-leaf 6 \
  --n-jobs 4 \
  --write-flat-table
```

Eval split:

- train rows: 11200;
- eval rows: 11200;
- spots: 7;
- horizons: +15, +30, +45, +60 minutes.

### Overall Eval Metrics

| Target | Model | RMSE | MAE | Bias | Count |
| --- | --- | ---: | ---: | ---: | ---: |
| wind mean | raw NWP | 2.125021 | 1.640945 | 0.302860 | 11200 |
| wind mean | persistence | 1.467946 | 1.019719 | 0.157018 | 10884 |
| wind mean | spot/lead bias correction | 2.035246 | 1.555415 | 0.243352 | 11200 |
| wind mean | HGB residual | 1.339277 | 1.013703 | 0.090914 | 11200 |
| wind mean | ExtraTrees residual | 1.451827 | 1.109854 | 0.096121 | 11200 |
| wind mean | Ridge residual | 1.329020 | 0.988037 | 0.028197 | 11200 |
| gust | raw NWP | 4.041999 | 3.195093 | 2.875589 | 11200 |
| gust | persistence | 1.729764 | 1.211478 | 0.164344 | 10884 |
| gust | spot/lead bias correction | 2.775964 | 2.048670 | 0.469287 | 11200 |
| gust | HGB residual | 1.612760 | 1.207884 | 0.183476 | 11200 |
| gust | ExtraTrees residual | 1.797402 | 1.349027 | 0.164278 | 11200 |
| gust | Ridge residual | 1.548223 | 1.158397 | 0.079162 | 11200 |

### By-Horizon RMSE

Wind mean:

| Lead | Raw NWP | Persistence | HGB | Ridge |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 2.133726 | 1.186912 | 1.183695 | 1.150897 |
| 30 | 2.131398 | 1.358553 | 1.292833 | 1.270405 |
| 45 | 2.131055 | 1.562997 | 1.398267 | 1.394840 |
| 60 | 2.103762 | 1.709409 | 1.465246 | 1.476857 |

Gust:

| Lead | Raw NWP | Persistence | HGB | Ridge |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 4.005645 | 1.439511 | 1.426703 | 1.370721 |
| 30 | 4.031578 | 1.609747 | 1.538365 | 1.459734 |
| 45 | 4.037094 | 1.840960 | 1.696820 | 1.628444 |
| 60 | 4.093176 | 1.978817 | 1.767126 | 1.710687 |

### Feature-Count Ablation

| Run | Best wind mean RMSE | Best wind model | Best gust RMSE | Best gust model |
| --- | ---: | --- | ---: | --- |
| `500` features | 1.329020 | Ridge | 1.548223 | Ridge |
| `250` features | 1.337535 | Ridge | 1.521571 | Ridge |
| `800` features | 1.330837 | HGB | 1.582691 | Ridge |
| `1100` features | 1.328369 | HGB | 1.610616 | HGB |

Best medium result:

- wind mean: `1.328369 m/s` with HGB and 1100 numeric features;
- gust: `1.521571 m/s` with Ridge and 250 numeric features.

## Interpretation

This is a successful dataset/benchmark milestone, not yet a new production
champion.

What it proves:

- the SAPHIR-style export is technically valid on real multi-year shards;
- the temporal audit passes;
- the structured dataset strongly improves over raw NWP;
- recent observations remain a very strong short-horizon signal;
- residual correction is the right framing;
- evaluating by horizon exposes the expected degradation from +15 to +60 min.

What it does not prove yet:

- it does not beat the best historical CorseWind champion on wind mean;
- the benchmark still flattens the SAPHIR-style tables before training;
- the medium run is capped at 400 samples per spot and only 7 spots survive the
  selected coverage constraints;
- Ridge winning often means the signal is real but the feature representation is
  noisy or too wide for the available supervised rows.

## Difference From The SAPHIR Study

SAPHIR trains neural models directly on structured tensors:

- target + neighbor station history;
- gridded AROME/ARPEGE windows;
- train-only normalization;
- deterministic and probabilistic losses.

CorseWind V1 now has analogous pieces, but the benchmark still converts them
into flat features for HGB/Ridge. The next scientific step is therefore not just
"more features"; it is a model that preserves the tensor structure:

- target station time tensor;
- context-station tensor with distance/upwind/role metadata;
- NWP horizon tensor;
- vertical profile tensor;
- static geography tensor;
- residual probabilistic output for wind mean and gust.

## Recommended Next Step

Run a full or larger capped export, then train two model families on the same
cases:

1. Flat supervised residual benchmark: HGB/Ridge/LightGBM, to keep a strong
   operational baseline.
2. SAPHIR-style neural residual model preserving the tensor inputs, with
   probabilistic outputs P10/P50/P90.

The target to beat remains the current CorseWind champion, not raw NWP. Raw NWP
is already beaten decisively in this benchmark.

## Neural Retest

Added script:

```text
scripts/ml_dataset/benchmark_corsewind_saphir_neural_dataset.py
```

Main run:

```bash
/home/z2/corsewind-ml-smoke/.venv/bin/python \
  scripts/ml_dataset/benchmark_corsewind_saphir_neural_dataset.py \
  --dataset-root /srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1 \
  --output-root /srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_sequence_medium_phys_v3_v1/saphir_neural_v1 \
  --epochs 96 \
  --patience 14 \
  --batch-size 512 \
  --hidden-dim 64 \
  --dropout 0.15 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --max-static-features 192 \
  --save-model
```

Architecture:

- GRU over target-station history;
- masked MLP encoder over context stations;
- masked MLP encoder over NWP surface offsets;
- MLP over vertical pressure-level profile;
- MLP over selected static features;
- MLP over lead/baseline/future covariates;
- spot embedding;
- two residual outputs: wind mean and gust.

### Neural Results

| Run | Wind mean RMSE | Gust RMSE | Notes |
| --- | ---: | ---: | --- |
| `saphir_neural_v1` | 1.375008 | 1.667227 | main structured NN |
| `saphir_neural_small_static` | 1.366307 | 1.655393 | 64 static features, smaller hidden |
| `saphir_neural_no_static` | 1.401135 | 1.666314 | confirms static features help slightly |
| `saphir_neural_wide_reg` | 1.393458 | 1.631634 | wider, more regularized |

Best neural result:

- wind mean: `1.366307 m/s`;
- gust: `1.631634 m/s`.

Compared with the flat benchmark:

- best flat wind mean: `1.328369 m/s`;
- best flat gust: `1.521571 m/s`.

Conclusion: the structured neural model already learns useful residuals and
beats raw NWP by a large margin, but this first architecture does not beat the
flat Ridge/HGB baseline. A simple stacking test using flat predictions plus
neural predictions also did not improve the 2026 eval score.

This suggests the next improvement should not be "make the neural model larger"
yet. The likely missing pieces are:

- more training cases per spot before training a neural model;
- explicit residual-error history features at every step;
- probabilistic/quantile loss instead of only deterministic residual MSE/Huber;
- better station-context ordering by upwind and coastal/relief role;
- cleaner removal of empty tensor channels before normalization;
- possibly target-specific heads or horizon-specific heads.
