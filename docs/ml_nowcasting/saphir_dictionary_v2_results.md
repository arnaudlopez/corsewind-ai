# CorseWind SAPHIR Dictionary V2 Results

Generated: 2026-06-30

## Goal

Build a SAPHIR-like dictionary dataset for CorseWind, closer to the reference
study structure:

- one sample per `spot_id + issue_time`;
- real target-station history;
- real neighboring/context-station history;
- multi-horizon targets at `+15/+30/+45/+60 min`;
- lead-aware NWP surface and vertical profile context;
- static spot/context geometry;
- same-sample tabular and neural benchmarks.

This is not yet a production replacement for the current champion. It is a
structural benchmark to test whether the SAPHIR-style organization gives us a
better learning surface.

## Implemented Scripts

New scripts:

- `scripts/ml_dataset/export_corsewind_saphir_dictionary_v2.py`
- `scripts/ml_dataset/audit_corsewind_saphir_dictionary_v2.py`
- `scripts/ml_dataset/benchmark_corsewind_saphir_dictionary_v2_tabular.py`
- `scripts/ml_dataset/benchmark_corsewind_saphir_dictionary_v2_neural.py`

Remote execution root on z2:

```text
/srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_dictionary_v2_medium_light
```

## Dataset Built

The completed evaluation run is `medium_light`:

- source run prefix: `residual_windsup_sst_prev_phys_v3_dem_fetch`;
- months: `2025-01` to `2026-06`;
- split: train before `2026-01-01`, eval from `2026-01-01`;
- spots: `balistra`, `figari_eole`, `la_tonnara`, `piantarella`, `porticcio`,
  `porto_polo`, `santa_manza`;
- samples: `2520`;
- future target rows: `10080`;
- station sequence rows: `397440`;
- context stations: target + `6` neighbors;
- context length: `24` steps, on a `15 min` grid;
- output size: `37M`.

Main tensor shapes:

| Tensor | Shape |
| --- | ---: |
| `station_tensor` | `2520 x 24 x 7 x 18` |
| `station_mask` | `2520 x 24 x 7` |
| `baseline_tensor` | `2520 x 4 x 9` |
| `y_actual` | `2520 x 4 x 2` |
| `y_residual` | `2520 x 4 x 2` |
| `static_tensor` | `2520 x 256` |

Audit verdict: `pass`.

Important checks passed:

- future targets are after issue time;
- lead minutes match target deltas;
- every sample has all four leads;
- station sequence contains no future observations;
- station sequence has no duplicate `sample + station slot + time` rows;
- tensors match the sample axis;
- target tensors contain no NaN;
- mean station mask coverage: `0.9387755`.

Disk status after the run:

- `/srv/data`: `552G` free;
- no remaining `corsewind_saphir_dictionary_v2` process was running.

## Baselines on Eval

Raw NWP on the V2 eval split:

| Target | RMSE | MAE | Bias | Rows |
| --- | ---: | ---: | ---: | ---: |
| wind mean | 2.123768 | 1.636067 | 0.289876 | 5040 |
| gust | 4.071346 | 3.191652 | 2.865934 | 5040 |

Persistence:

| Target | RMSE | MAE | Bias | Rows |
| --- | ---: | ---: | ---: | ---: |
| wind mean | 1.426155 | 0.991306 | 0.169601 | 4832 |
| gust | 1.699481 | 1.181455 | 0.220704 | 4832 |

## Tabular Benchmark

Best standard tabular run:

```text
/srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_dictionary_v2_medium_light/benchmark_v2_tabular
```

Global eval:

| Target | Best model | RMSE | MAE | Bias | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| wind mean | HGB | 1.376859 | 1.056345 | 0.162613 | 5040 |
| gust | persistence | 1.699481 | 1.181455 | 0.220704 | 4832 |

Wind mean by horizon:

| Horizon | Raw NWP RMSE | Persistence RMSE | HGB RMSE |
| ---: | ---: | ---: | ---: |
| +15 min | 2.121472 | 1.158588 | 1.208524 |
| +30 min | 2.099448 | 1.351157 | 1.333113 |
| +45 min | 2.143593 | 1.523546 | 1.458328 |
| +60 min | 2.130317 | 1.626816 | 1.489469 |

Gust by horizon:

| Horizon | Raw NWP RMSE | Persistence RMSE | HGB RMSE | Ridge RMSE |
| ---: | ---: | ---: | ---: | ---: |
| +15 min | 4.077221 | 1.418936 | 1.616479 | 1.619763 |
| +30 min | 4.084281 | 1.628162 | 1.759766 | 1.715449 |
| +45 min | 4.091067 | 1.793301 | 1.851563 | 1.777177 |
| +60 min | 4.032556 | 1.916434 | 1.868346 | 1.839712 |

The wider tabular run added ExtraTrees and more features:

```text
/srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_dictionary_v2_medium_light/benchmark_v2_tabular_wide
```

It did not improve:

- wind mean HGB RMSE: `1.379380`;
- gust best remained persistence: `1.699481`;
- ExtraTrees was worse on both targets.

## Neural Benchmark

Run:

```text
/srv/data/corsewind/ml_dataset/benchmarks/corsewind_saphir_dictionary_v2_medium_light/benchmark_v2_neural
```

Architecture used for this first pass:

- station-history encoder over `time x station slots`;
- static feature branch;
- baseline feature branch;
- residual prediction for all horizons and both targets;
- Huber loss, early stopping;
- device: CUDA on z2.

Global eval:

| Target | Raw NWP RMSE | Persistence RMSE | V2 NN RMSE | V2 NN MAE |
| --- | ---: | ---: | ---: | ---: |
| wind mean | 2.123768 | 1.426155 | 1.490269 | 1.136360 |
| gust | 4.071346 | 1.699481 | 1.795497 | 1.360256 |

Wind mean by horizon:

| Horizon | Persistence RMSE | V2 NN RMSE |
| ---: | ---: | ---: |
| +15 min | 1.158588 | 1.314122 |
| +30 min | 1.351157 | 1.467076 |
| +45 min | 1.523546 | 1.565343 |
| +60 min | 1.626816 | 1.598150 |

The neural model improves raw NWP, but does not beat HGB on wind mean and does
not beat persistence on gust. This first neural architecture should not be
scaled blindly.

## Comparison With Existing CorseWind Results

Known locked wind-mean champion:

- RMSE: `1.268019`;
- MAE: `0.930465`.

Previous SAPHIR sequence V1 best medium result:

- wind mean HGB RMSE: `1.328369`;
- gust HGB RMSE: `1.610616`.

Current V2 medium-light result:

- wind mean HGB RMSE: `1.376859`;
- gust best RMSE: `1.699481` with persistence.

Conclusion: V2 is structurally healthier and closer to SAPHIR, but this first
medium-light benchmark does not beat the current production champion, and it
does not beat the previous V1 sequence benchmark.

This should not be interpreted as "neighbor histories are useless". The sample
set, horizon range, max samples per spot, feature selection, and model families
are not identical to the historical champion/V1 benchmark. It means this exact
V2 implementation and training recipe is not yet the winner.

## Scientific Interpretation

What worked:

- the dictionary organization is feasible;
- the leakage audit passes;
- real neighboring stations can be represented as historical sequences;
- the model learns meaningful residual corrections versus raw NWP;
- HGB improves wind mean over persistence globally and especially from `+30` to
  `+60 min`.

What did not work yet:

- V2 HGB is worse than the locked champion;
- the simple neural model is underpowered for this dataset size/shape;
- gust prediction remains better with persistence at short horizons;
- adding more flat features and ExtraTrees did not help.

The most important modeling signal is the horizon split:

- `+15 min`: persistence remains extremely strong;
- `+30/+45/+60 min`: HGB starts to add value for wind mean;
- gusts are still mostly a nowcasting/persistence problem.

## Decision

Do not replace the current champion with V2.

Keep the V2 exporter and audit as the new structured research format, because
it gives us the right SAPHIR-like object for more serious experiments.

Operationally, the best immediate deployable idea from this benchmark is a
router/blend:

- wind mean `+15 min`: keep persistence/current champion;
- wind mean `+30/+45/+60 min`: evaluate HGB V2 as an additional candidate in a
  same-split ensemble;
- gust `+15/+30/+45 min`: keep persistence/current gust route;
- gust `+60 min`: test Ridge/HGB as weak candidates, but do not switch yet.

## Next Experiments

Priority 1: same-sample comparison

Run the current champion, V1 sequence HGB, and V2 HGB on exactly the same
`medium_light` samples. Without this, comparisons remain directionally useful
but not decisive.

Priority 2: per-horizon models

Train separate models per target and per lead instead of one global model with
`lead_time_minutes` as a feature. The horizon behavior is different enough that
one global model is likely averaging incompatible regimes.

Priority 3: context ablation

Run V2 with controlled feature groups:

- target station only;
- target + coastal neighbor;
- target + mountain/inland neighbor;
- all context stations;
- all context stations without NWP error features.

This will tell us whether the SAPHIR-style neighbor histories are actually
helping or whether the model is mostly using target persistence/error features.

Priority 4: better neural architecture

The next neural version should not just scale the current GRU/MLP. It should use
one of:

- station attention over context slots;
- temporal convolution over recent station history;
- separate lead heads;
- residual + direct multitask outputs;
- explicit masking and station metadata embeddings.

Priority 5: optimize full V2 export

The larger full export attempt was killed after running too long. It was a time
and memory-efficiency issue, not a disk issue. Before exporting full
2024-2026/400 samples per spot, optimize:

- context-source series construction;
- station-sequence materialization;
- optional streaming writes instead of keeping all frames until the end.
