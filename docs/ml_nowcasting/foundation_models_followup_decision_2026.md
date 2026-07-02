# Foundation Models Follow-up Decision

Generated: 2026-06-30

## Why Investigate Further

The previous complementarity audit found a striking result on a small 2026
foundation benchmark:

- rows: `336`;
- base HGB sequence RMSE: `1.276752`;
- row-wise oracle across foundation/HGB/raw models: `0.628239`.

That result is not production proof because the sample is small. But it is too
large to ignore: the models are making different errors.

## Larger Existing Benchmark

We found a larger saved sequence benchmark:

```text
/srv/data/corsewind/ml_dataset/benchmarks/sequence_2026_windsurf_1h_rmse09_v1/predictions_with_timesfm.parquet
```

Rows: `1680`.

Available wind-mean columns:

- `raw_wind_mean_ms`;
- `chronos2_univar_wind_mean_ms_mean`;
- `timesfm_wind_mean_ms_mean`.

`hgb_wind_mean_ms` exists in the file but is empty for this benchmark, so it was
not used as a valid base.

## Larger Foundation Oracle

Audit:

```text
/srv/data/corsewind/ml_dataset/benchmarks/prediction_complementarity_foundation_2026_rmse09_v1_chronos_base
```

| Model | RMSE | MAE | Bias | Rows |
| --- | ---: | ---: | ---: | ---: |
| Chronos2 univariate | 1.680607 | 1.116305 | 0.069110 | 1680 |
| TimesFM | 1.691706 | 1.132719 | 0.021046 | 1680 |
| Raw NWP | 2.126312 | 1.621158 | 0.170418 | 1652 |
| Row-wise oracle | 1.039008 | - | - | 1680 |

Oracle selection:

- TimesFM: `35.6%`;
- Chronos2: `32.7%`;
- raw NWP: `31.7%`.

By horizon, oracle RMSE:

- `+15`: `0.860460`;
- `+30`: `1.014970`;
- `+45`: `1.116680`;
- `+60`: `1.140447`.

This confirms the small-sample signal: foundation models and raw NWP have
strong complementary error patterns. But the individual foundation models are
not competitive yet.

## Direct Champion Intersection

Audit:

```text
/srv/data/corsewind/ml_dataset/benchmarks/prediction_complementarity_champion_foundation_2026_rmse09_v1
```

Intersection with the true champion is only `197` rows, so this is a smoke
signal only.

| Model | RMSE | Rows |
| --- | ---: | ---: |
| TimesFM | 1.412857 | 197 |
| champion scale070 | 1.422246 | 197 |
| Chronos2 univariate | 1.440426 | 197 |
| raw NWP | 2.300353 | 197 |
| row-wise oracle | 0.870311 | 197 |

This is important but not production-grade evidence. It says: on the rows where
they overlap, the foundation models sometimes beat the champion, but we do not
yet have enough overlap to validate a deployable system.

## Can We Learn The Oracle?

We built a 2024+2025 calibration file:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_router_calibration_2024_2025_rmse09_v1.parquet
```

Rows: `3120`.

Then we evaluated routers on 2026.

### HGB Routers

| Router | Base RMSE | Alt RMSE | Oracle RMSE | Router RMSE | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chronos2 -> TimesFM | 1.680607 | 1.691706 | 1.546498 | 1.684015 | -0.20% |
| Chronos2 -> raw NWP | 1.690946 | 2.126312 | 1.108815 | 1.690019 | 0.055% |
| TimesFM -> raw NWP | 1.701807 | 2.126312 | 1.147270 | 1.701807 | 0.00% |

The routers do not learn the oracle. They either avoid switching almost
entirely or switch in ways that do not improve RMSE.

### Weighted Ensembles

| Ensemble | Base RMSE | Alt RMSE | Ensemble RMSE | Gain |
| --- | ---: | ---: | ---: | ---: |
| Chronos2 + TimesFM | 1.680607 | 1.691706 | 1.659747 | 1.241% |
| Chronos2 + raw NWP | 1.680607 | 2.126312 | 1.606214 | 4.427% |
| TimesFM + raw NWP | 1.691706 | 2.126312 | 1.629702 | 3.665% |

Weighted blending helps more than routing, but these scores are still far from
the champion.

## Decision

Yes, foundation models are worth investigating further.

But the right next step is not to simply run bigger Chronos/TimesFM/Moirai
benchmarks. The current evidence says:

1. Foundation models contain different signal.
2. Their individual predictions are not good enough.
3. Their row-wise oracle is strong.
4. Current simple routers cannot predict when to trust them.
5. Weighted blending helps a little, but not enough.

So the next useful experiment is:

- build a larger same-sample benchmark where champion, raw NWP, Chronos2,
  TimesFM, Moirai, and possibly Chronos residual all overlap;
- train a meta-model on 2024+2025 to predict either:
  - the best model family for each row;
  - or the future champion error;
  - or calibrated weights across model families;
- evaluate strictly on 2026.

## Proposed Next Foundation Experiment

Create:

```text
foundation_superbench_2024_2026_v1
```

Requirements:

- at least `5k-10k` eval rows if possible;
- exact same `spot_id + issue_time_utc + lead_time_minutes` keys for all models;
- include true champion predictions, not only sequence HGB;
- include Chronos2, TimesFM, Moirai, raw NWP, and persistence;
- include issue-time router features:
  - model disagreement;
  - spread between foundation predictions;
  - raw NWP vs persistence disagreement;
  - recent target observation trend;
  - recent raw-model error trend;
  - lead, spot, hour, month;
  - thermal/cloud/static regime features when available.

Success criteria:

- individual foundation model does not need to beat champion;
- meta-blend/router must beat champion on same-sample 2026 by at least `1%`
  before we scale;
- if oracle remains strong but router remains weak, focus shifts to failure
  prediction and missing issue-time regime features.

## Champion-Aligned Superbench Result

Update: 2026-06-30.

We built the first champion-aligned superbench on z2:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_sequence_champion_aligned_2026_windsurf_v1
/srv/data/corsewind/ml_dataset/benchmarks/foundation_superbench_champion_aligned_2026_windsurf_v1
```

Models included:

- raw NWP baseline;
- Chronos covariate sequence benchmark;
- Chronos2 univariate sequence benchmark;
- TimesFM;
- Moirai;
- current wind-mean champion;
- best separate gust champion.

Important caveat: the foundation benchmark has `1120` rows, but the champion
does not cover all these rows after key alignment:

- wind champion coverage: `324 / 1120`;
- gust champion coverage: `417 / 1120`.

Therefore, global foundation scores and champion scores are not directly
comparable unless we use the overlap subsets.

### Global Foundation Rows

| Target | Model | RMSE | MAE | Rows |
| --- | --- | ---: | ---: | ---: |
| wind mean | raw NWP | 2.055389 | 1.591356 | 1120 |
| wind mean | Chronos2 univariate | 1.529181 | 1.075778 | 1120 |
| wind mean | TimesFM p50 | 1.536660 | 1.077207 | 1120 |
| wind mean | Moirai p50 | 1.723305 | 1.206202 | 1120 |
| wind mean | row-wise oracle | 0.794014 | 0.485310 | 1120 |
| gust | raw NWP | 3.928518 | 3.153143 | 1120 |
| gust | Chronos2 univariate | 1.836629 | 1.268885 | 1120 |
| gust | TimesFM p50 | 1.838873 | 1.283454 | 1120 |
| gust | Moirai p50 | 2.013893 | 1.429235 | 1120 |
| gust | row-wise oracle | 1.062598 | 0.636683 | 1120 |

### Fair Wind Champion Overlap

On the `324` rows where the wind champion exists:

| Model | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| wind champion | 1.216885 | 0.912317 | 0.107417 |
| Chronos2 univariate | 1.556209 | 1.103138 | 0.100633 |
| TimesFM mean | 1.563320 | 1.104234 | 0.063279 |
| TimesFM p50 | 1.564752 | 1.105430 | 0.063078 |
| raw NWP | 2.023291 | 1.532973 | 0.331505 |
| row-wise oracle | 0.758569 | 0.443742 | 0.102329 |

Interpretation: none of the foundation models beats the champion alone on the
fair overlap. But the oracle is much lower than the champion, which proves that
there are rows where the foundation predictions contain useful complementary
signal.

### Fair Gust Champion Overlap

On the `417` rows where the gust champion exists:

| Model | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| gust champion | 1.444354 | 1.093665 | 0.151142 |
| Chronos2 univariate | 1.855357 | 1.293658 | 0.149993 |
| TimesFM p50 | 1.843858 | 1.315385 | 0.086478 |
| raw NWP | 3.845477 | 3.140485 | 2.845739 |
| row-wise oracle | 0.998607 | 0.591952 | 0.191517 |

Same conclusion for gusts: the separate gust champion remains better as a
standalone model, but the oracle shows a strong exploitable complementarity.

## Current Decision

The champion has not been replaced by Chronos2, TimesFM, or Moirai.

The useful result is different: foundation models are not the final forecaster,
but they are promising extra experts for a stacking/router model. The next
scientifically valid step is to train a deployable meta-model on past years that
learns when the champion is likely wrong and how much weight to give each
expert.

Next experiment:

```text
foundation_meta_stack_champion_aligned_v1
```

Required changes:

- build the same aligned superbench for 2024 and 2025, not only 2026;
- keep the 2026 split fully held out;
- train a meta-regressor or calibrated weighted stack using only issue-time
  features and model predictions;
- evaluate against the champion on strict same-key overlap;
- report both wind mean and gusts.

Success criterion:

- beat the current wind champion on the same-key 2026 overlap;
- first target: any stable improvement above `1%`;
- stretch target: approach the oracle direction without overfitting.

## 2025h2 Mirror Bench And Router Smoke Test

Update: 2026-06-30, second pass.

We built a mirrored 2025h2 foundation benchmark to act as calibration data:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_sequence_champion_aligned_2025h2_windsurf_v1
/srv/data/corsewind/ml_dataset/benchmarks/foundation_superbench_champion_aligned_2025h2_windsurf_v1
```

Global 2025h2 foundation rows: `1120`.

| Target | Model | RMSE | MAE | Rows |
| --- | --- | ---: | ---: | ---: |
| wind mean | raw NWP | 1.979750 | 1.587186 | 1120 |
| wind mean | Chronos2 univariate | 1.497450 | 1.085338 | 1120 |
| wind mean | TimesFM p50 | 1.513067 | 1.094093 | 1120 |
| wind mean | Moirai p50 | 1.682832 | 1.212142 | 1120 |
| wind mean | row-wise oracle | 0.818453 | 0.508676 | 1120 |
| gust | raw NWP | 3.328529 | 2.660471 | 1120 |
| gust | Chronos2 univariate | 1.791183 | 1.289399 | 1120 |
| gust | TimesFM p50 | 1.821766 | 1.320137 | 1120 |
| gust | Moirai p50 | 2.004415 | 1.471178 | 1120 |
| gust | row-wise oracle | 1.093903 | 0.676737 | 1120 |

Champion overlap remains the blocking point:

- 2025h2 wind champion coverage on the sequence keys: `224 / 1120`;
- 2026 wind champion coverage on the sequence keys: `324 / 1120`;
- the 2025h2 reconstructed champion has many short-lead rows overall, but almost
  never the full four horizons for the same issue time. Only `2` issue times
  have all `15/30/45/60` leads on the 7 windsurf spots.

This means the aligned superbench is currently good enough for diagnosis, but
too sparse for a robust deployable router.

### Pair Router Smoke Test

Calibration: 2025h2 overlap.
Evaluation: 2026 overlap.
Base model: wind champion.

| Alternative | Eval rows | Champion RMSE | Alt RMSE | Router RMSE | Pair oracle RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chronos2 univariate | 324 | 1.216885 | 1.556209 | 1.218275 | 1.032973 | not improved |
| TimesFM p50 | 324 | 1.216885 | 1.564752 | 1.224504 | 1.025223 | not improved |

Interpretation:

- the router result is not surprising with only `224` calibration rows;
- one-foundation pair oracles are much weaker than the all-model oracle;
- the all-model oracle still shows complementarity, but the current overlap is
  too small and sparse to learn it safely.

## Updated Next Step

Do not spend more time tuning routers on the current sparse overlap.

The next required engineering step is to produce dense champion-compatible
predictions on the same keys as the sequence benchmark. Two viable options:

1. Re-run/reconstruct the champion residual model on all foundation sequence
   keys for 2025h2 and 2026.
2. Or build the foundation sequence benchmark from the exact champion prediction
   keys, accepting sparse/irregular horizons but changing the sequence builder
   to support row-level evaluation rather than requiring complete four-step
   issues.

Preferred option: `1`, because it gives a clean same-key table for stacking:

```text
spot_id + issue_time_utc + lead_time_minutes
actual
raw_nwp
champion_wind
champion_gust
chronos2
timesfm
moirai
issue-time features
```

Once dense overlap exists, rerun:

- all-model meta-stack;
- pair routers;
- error predictor for champion residual;
- evaluation by spot, lead, thermal regime, and strong-wind regime.

## Dense Champion Overlap And First Positive Blends

Update: 2026-06-30, dense champion pass.

We fixed the key limitation above: the champion has now been rescored densely on
the exact foundation sequence keys.

Dense wind champion artifacts:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_dense_champion_wind_2025h2_windsurf_200cut_scale070_source_v1
/srv/data/corsewind/ml_dataset/benchmarks/foundation_dense_champion_wind_2026_windsurf_scale070_source_v1
```

Dense gust champion artifacts:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_dense_champion_gust_2025h2_windsurf_200cut_old_signal_v1
/srv/data/corsewind/ml_dataset/benchmarks/foundation_dense_champion_gust_2026_windsurf_old_signal_v1
```

The dense wind champion reproduces the official champion exactly on the previous
overlap:

- 2025h2 official overlap rows: `224`, max prediction delta: `0.0`;
- 2026 official overlap rows: `324`, max prediction delta: `0.0`.

The dense gust champion also reproduces the official gust champion exactly on
the 2026 overlap:

- 2026 official overlap rows: `417`, max prediction delta: `0.0`.

### Dense Same-Key Benchmarks

Training/calibration superbench:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_superbench_dense_champion_wind_gust_2025h2_windsurf_200cut_v1
```

Rows: `5600`.

Evaluation superbench:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_superbench_dense_champion_wind_gust_2026_windsurf_v1
```

Rows: `1120`.

On 2026:

| Target | Model | RMSE | MAE/Bias note |
| --- | --- | ---: | --- |
| wind mean | dense champion scale070 | 1.250266 | MAE 0.946621 |
| wind mean | TimesFM p50 | 1.536660 | standalone worse |
| wind mean | Chronos2 univariate | 1.529181 | standalone worse |
| wind mean | oracle | 0.747830 | diagnostic only |
| gust | dense champion old_signal | 1.513977 | MAE 1.140131 |
| gust | TimesFM p50 | 1.838873 | standalone worse |
| gust | Chronos2 univariate | 1.836629 | standalone worse |
| gust | oracle | 0.949492 | diagnostic only |

### First Positive Production-Shaped Results

Flexible meta-stacks still overfit easily for wind mean. The first robust
improvement is a conservative static blend:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_static_blend_wind_2025h2_200cut_to_2026_v1
```

Weights learned on 2025h2:

- `92.5%` dense champion;
- `7.5%` TimesFM p50.

2026 result:

| Wind model | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| dense champion | 1.250266 | 0.946621 | 0.079530 |
| static blend | 1.240036 | 0.938861 | 0.073098 |
| oracle | 0.747830 | 0.440693 | 0.049895 |

Gain vs champion: `0.818%`.

For gusts, both a conservative static blend and a direct ExtraTrees meta-stack
improve the dense champion:

| Gust model | RMSE | Gain vs champion |
| --- | ---: | ---: |
| dense champion | 1.513977 | baseline |
| static champion + TimesFM blend | 1.502680 | 0.746% |
| direct ExtraTrees meta-stack | 1.492289 | 1.433% |
| oracle | 0.949492 | diagnostic only |

Best gust meta-stack artifact:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_meta_stack_gust_2025h2_200cut_to_2026_direct_extra_trees_v1
```

## Updated Decision

We have the first same-key improvements over the champions:

- wind mean: small but clean improvement with a static TimesFM blend;
- gusts: clearer improvement with ExtraTrees meta-stack.

This does not solve the `RMSE < 0.9` target, but it changes the conclusion:
foundation models are no longer just diagnostic. Used conservatively as extra
experts, they can improve the current champions on held-out 2026 data.

Next experiments should focus on stability:

- expand 2026 evaluation beyond `1120` rows;
- build 2025h2 plus earlier 2025/2024 calibration superbenches if compute allows;
- evaluate by spot, lead, thermal regime, and wind-strength regime;
- promote only the conservative static wind blend unless the meta-stack proves
  stable on a larger split;
- keep the gust ExtraTrees stack as a candidate, but require a larger eval
  before considering it champion.
