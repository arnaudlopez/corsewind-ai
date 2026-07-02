# Foundation Sequence Benchmark v2 - 2026 Windsurf Hours

This is the second foundation-model benchmark pass. It expands the first pilot
from 20 sequences to 84 sequences, keeps the same +15..+60 minute horizons, and
adds a smoke test for a +6 hour 15-minute grid.

## 1h Benchmark

Remote root:

```text
/srv/data/corsewind/ml_dataset/benchmarks/sequence_2026_windsurf_1h_v2
```

Scope:

- evaluation period: issue times from 2026-01-01 onward;
- hours: 08..17 UTC;
- context: 96 x 15-minute observations = 24 hours;
- horizon: +15, +30, +45, +60 minutes;
- cases: 84;
- prediction rows: 336;
- rows common to all models with raw NWP/HGB available: 308;
- spots: `balistra`, `figari_eole`, `la_tonnara`, `piantarella`,
  `porticcio`, `porto_polo`, `santa_manza`.

Final common-row file:

```text
benchmark_results_final_common_rows.json
benchmark_results_final_common_rows.md
predictions_final_all_models.parquet
```

## 1h Results

Fair comparison on 308 common rows:

| Target | Model | RMSE | MAE | Bias | Count |
| --- | --- | ---: | ---: | ---: | ---: |
| wind mean | Chronos-2 univariate cross | 1.269650 | 0.943105 | 0.184721 | 308 |
| wind mean | HGB residual | 1.276752 | 1.015342 | 0.154688 | 308 |
| wind mean | TimesFM | 1.310452 | 0.968488 | 0.154784 | 308 |
| wind mean | Chronos-2 covariate/multivariate | 1.444525 | 0.981802 | 0.093371 | 308 |
| wind mean | Moirai | 1.456665 | 1.030630 | 0.171570 | 308 |
| wind mean | Chronos-2 residual correction | 1.495813 | 1.120848 | -0.162007 | 308 |
| wind mean | raw NWP | 1.900426 | 1.453470 | 0.274120 | 308 |
| gust | Chronos-2 univariate cross | 1.513633 | 1.063257 | 0.272513 | 308 |
| gust | TimesFM | 1.545868 | 1.118788 | 0.238694 | 308 |
| gust | HGB residual | 1.589091 | 1.235237 | 0.339539 | 308 |
| gust | Moirai | 1.692379 | 1.213607 | 0.277161 | 308 |
| gust | Chronos-2 covariate/multivariate | 1.771654 | 1.170547 | 0.239680 | 308 |
| gust | Chronos-2 residual correction | 2.331977 | 1.574069 | -0.053334 | 308 |
| gust | raw NWP | 3.799771 | 2.926955 | 2.608074 | 308 |

## Interpretation

Chronos-2 univariate + cross-learning is the best 1h candidate in this v2
benchmark. It is slightly ahead of the current HGB sample model on wind mean and
clearly ahead on gust. The margin is not large enough to decide production by
itself, but it is large enough to make Chronos-2 a primary benchmark candidate.

TimesFM is close and operationally simple. It should remain in the comparison
set, but Chronos-2 is currently the stronger foundation-model candidate.

The naive residual-sequence approach did not work. Forecasting
`observed - NWP` with Chronos and adding it back to future NWP degraded both wind
mean and gust. This does not invalidate residual correction as a concept; it
only invalidates this simple zero-shot residual formulation.

The Chronos-2 multivariate/covariate mode remains suspicious. It improved over
raw NWP but underperformed the simpler univariate Chronos mode. Do not use this
mode for ranking until its input contract and covariate semantics have been
audited.

## 6h Smoke Test

Remote root:

```text
/srv/data/corsewind/ml_dataset/benchmarks/sequence_2026_windsurf_6h_interpolated_smoke
```

Scope:

- same 2026 windsurf-hour filters;
- context: 96 x 15-minute observations;
- horizon: 24 x 15-minute steps = +6h;
- cases: 14;
- rows: 336;
- future NWP covariates are interpolated from sparse available lead times.

Results:

| Target | Model | RMSE | MAE | Bias | Count |
| --- | --- | ---: | ---: | ---: | ---: |
| wind mean | raw NWP | 1.535728 | 1.219818 | 0.065509 | 336 |
| wind mean | Chronos-2 univariate cross | 2.087458 | 1.456361 | -0.479621 | 336 |
| wind mean | Chronos-2 covariate/multivariate | 2.428687 | 1.784059 | 0.647404 | 336 |
| gust | Chronos-2 univariate cross | 2.554903 | 1.840717 | -0.529911 | 336 |
| gust | raw NWP | 2.555636 | 2.085079 | 1.701710 | 336 |
| gust | Chronos-2 covariate/multivariate | 3.203466 | 2.243666 | 0.751285 | 336 |

This is only a smoke test, not a final +6h benchmark. Still, it matches the
expected structure: observations dominate the very short range, while raw NWP
becomes much more competitive at longer horizons.

## Current Decision

Use this ranking for the next engineering pass:

1. Chronos-2 univariate + cross-learning as the leading foundation-model
   benchmark.
2. HGB residual as the current supervised local baseline and production-style
   reference.
3. TimesFM as the second foundation-model baseline.
4. Moirai as optional, lower priority.
5. Chronos-2 covariate mode and naive Chronos residual mode as ablations to
   debug, not candidates to promote.

## Next Work

- Increase the 1h benchmark from 84 cases to a larger sample once the sequence
  builder is optimized.
- Add a proper ensemble/calibration layer:
  `features + HGB + Chronos p10/p50/p90 + raw NWP -> calibrated p10/p50/p90`.
- Replace the naive residual-sequence test with a supervised residual model or
  a Chronos residual formulation that includes horizon, recent NWP error, and
  station/spot identity more explicitly.
- Build a non-interpolated 15-minute future NWP feature grid for +6h before
  treating long-horizon results as decision-grade.
