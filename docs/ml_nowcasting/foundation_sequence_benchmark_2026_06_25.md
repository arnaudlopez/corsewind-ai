# Foundation Sequence Benchmark - 2026-06-25

This note records the first sequential dataset and foundation-model benchmark
for CorseWind.ai short-range wind nowcasting.

## Goal

Compare the current supervised residual-correction baseline against zero-shot
time-series foundation models on the same spot/horizon cases:

- raw NWP baseline from the residual training table;
- current HGB residual model trained on the 2024-01..2026-06 sample;
- Chronos-2, with two tested modes:
  - direct multivariate/covariate mode from the first dataset builder;
  - corrected univariate + cross-learning mode on saved sequences;
- TimesFM 2.5 200M PyTorch;
- Moirai 1.1 small.

The benchmark is intentionally small. Its purpose is to validate the sequence
dataset format, GPU environments, and scoring pipeline before scaling to all
spots and more cutoffs.

## Sequential Dataset

Remote root:

```text
/srv/data/corsewind/ml_dataset/benchmarks/chronos2_sequence_pilot_2026_windsurf_1h
```

Files:

- `past_context.parquet`: 96 x 15-minute observed context points per case.
- `future_covariates.parquet`: NWP future covariates for the same horizons.
- `predictions.parquet`: actuals, raw NWP, Chronos forecasts.
- `predictions_with_hgb.parquet`: adds current HGB residual predictions.
- `predictions_with_timesfm.parquet`: adds TimesFM forecasts.
- `predictions_with_moirai.parquet`: adds Moirai forecasts.
- `predictions_with_chronos2_univariate.parquet`: adds the corrected
  Chronos-2 univariate + cross-learning forecasts.
- `benchmark_results_common_rows.json`: fair comparison on rows where every
  model and the raw baseline are available.
- `benchmark_results_common_rows_corrected_chronos.json`: fair comparison with
  the corrected Chronos-2 univariate mode.

Pilot scope:

- spots: `piantarella`, `la_tonnara`, `figari_eole`, `porticcio`, `balistra`;
- cutoffs: 4 per spot;
- context: 96 steps = 24 hours;
- horizon: 4 steps = +15, +30, +45, +60 minutes;
- total sequence cases: 20;
- total forecast rows: 80;
- common rows with HGB/raw NWP available: 60.

The first 6-hour Chronos run was not valid with exact future covariates because
the current residual table has sparse lead times: `15,30,45,60,120,180,360`.
A 6-hour 15-minute sequential benchmark needs either interpolated/rebuilt
future covariates at every 15-minute step or a model mode that does not require
future covariates.

## Environments On z2

- Chronos: `/home/z2/corsewind-ml-smoke/.venv`
  - `chronos-forecasting==2.3.0`
  - `torch==2.6.0+cu124`
- TimesFM: `/home/z2/corsewind-ml-smoke/.venv-timesfm`
  - `timesfm==2.0.1`
  - `torch==2.6.0+cu124`
  - the default PyPI torch install targeted CUDA 13 and did not work with the
    z2 driver, so it was replaced with the CUDA 12.4 build.
- Moirai: `/home/z2/corsewind-ml-smoke/.venv-moirai`
  - `uni2ts==1.1.1`
  - `torch==2.6.0+cu124`
  - available model path is Moirai 1.1; Moirai-2/MoE imports were not available
    in this resolved environment.
- HGB scoring: `corsewind-ml-dataset-runner:latest`
  - used because the Chronos environment cannot load the existing HGB joblib
    artifacts due a numpy/joblib bitgenerator compatibility mismatch.

## Scripts Added

- `scripts/ml_dataset/benchmark_chronos2_sequences.py`
- `scripts/ml_dataset/benchmark_chronos2_saved_sequences.py`
- `scripts/ml_dataset/score_hgb_sequence_benchmark.py`
- `scripts/ml_dataset/benchmark_timesfm_sequences.py`
- `scripts/ml_dataset/benchmark_moirai_sequences.py`

## Main Results

Fair comparison on the 60 common rows, after correcting Chronos-2 to run as a
univariate sequence model with `cross_learning=True`:

| Target | Model | RMSE | MAE | Bias | Count |
| --- | --- | ---: | ---: | ---: | ---: |
| wind mean | HGB residual | 0.841094 | 0.705134 | 0.177848 | 60 |
| wind mean | Chronos-2 univariate cross | 1.151478 | 0.793380 | 0.560453 | 60 |
| wind mean | TimesFM p50 | 1.284156 | 0.828741 | 0.652606 | 60 |
| wind mean | Moirai p50 | 1.495024 | 0.927688 | 0.763278 | 60 |
| wind mean | Chronos-2 covariate/multivariate p50 | 1.941574 | 1.100922 | 0.803230 | 60 |
| wind mean | raw NWP | 1.988817 | 1.480852 | 0.600889 | 60 |
| gust | HGB residual | 1.145624 | 1.001891 | 0.273452 | 60 |
| gust | Chronos-2 univariate cross | 1.210740 | 0.874177 | 0.557375 | 60 |
| gust | TimesFM p50 | 1.445307 | 0.976202 | 0.719797 | 60 |
| gust | Moirai p50 | 1.873760 | 1.295607 | 0.867528 | 60 |
| gust | Chronos-2 covariate/multivariate p50 | 2.578802 | 1.464733 | 1.075475 | 60 |
| gust | raw NWP | 4.228222 | 3.259481 | 2.956444 | 60 |

On all 80 foundation-model rows, Chronos-2 univariate + cross-learning leads
the zero-shot foundation models:

| Target | Model | RMSE | MAE | Bias | Count |
| --- | --- | ---: | ---: | ---: | ---: |
| wind mean | Chronos-2 univariate cross | 1.059828 | 0.718382 | 0.432009 | 80 |
| wind mean | TimesFM p50 | 1.228558 | 0.797285 | 0.594424 | 80 |
| wind mean | Moirai p50 | 1.407585 | 0.895646 | 0.659580 | 80 |
| wind mean | Chronos-2 covariate/multivariate p50 | 1.743514 | 0.972891 | 0.625628 | 80 |
| gust | Chronos-2 univariate cross | 1.143878 | 0.819816 | 0.454881 | 80 |
| gust | TimesFM p50 | 1.397967 | 0.936003 | 0.660222 | 80 |
| gust | Moirai p50 | 1.698196 | 1.140947 | 0.681353 | 80 |
| gust | Chronos-2 covariate/multivariate p50 | 2.318978 | 1.303127 | 0.863568 | 80 |

## Interpretation

The current HGB residual model is still the best model in this first pilot. That
is expected: it is supervised on CorseWind-specific spot, NWP, context station,
and derived features. The foundation models were used zero-shot and mostly as
univariate recent-history forecasters.

Chronos-2 is the strongest zero-shot candidate so far when used in the corrected
univariate + cross-learning configuration. It is close to HGB on gust RMSE in
this small pilot and beats TimesFM/Moirai on both wind mean and gust.

The first Chronos-2 multivariate/covariate configuration was misleading. It
worked technically, but underperformed badly. That points to a configuration or
data-format issue around the way future NWP covariates and multiple targets were
fed to Chronos-2, not to a weak Chronos model. Treat those covariate-mode numbers
as an ablation failure, not the primary Chronos result.

TimesFM remains a strong and simple zero-shot baseline. It is easier to operate
than Moirai and still beats the raw NWP baseline by a wide margin.

Moirai 1.1 is viable but not leading in this pilot. Its environment is heavier
and more fragile than TimesFM and Chronos.

## Next Steps

1. Scale the benchmark from 20 sequence cases to a statistically meaningful
   evaluation set: all key spots, windsurf hours, and many cutoffs across 2025
   and 2026.
2. Produce a 6-hour sequential dataset on a true 15-minute grid. This requires
   rebuilding/interpolating future NWP covariates for every 15-minute step.
3. Add residual-error sequence targets:
   `observed_now - nwp_forecast_for_now` as context, then forecast future NWP
   residuals instead of direct wind/gust.
4. Re-test Chronos-2 covariate mode only after validating the exact input
   contract and the 15-minute future covariate grid. Do not use the first
   covariate-mode result as a model ranking signal.
5. Test TimesFM with covariates (`timesfm[xreg]`) once the 15-minute future
   covariate grid exists.
6. Train a full HGB/ExtraTrees benchmark on all available data, not only the
   300k/80k sample, then compare again.
7. Keep HGB as the current production candidate and treat foundation models as
   candidates for hybrid residual ensembles until they beat the supervised local
   baseline on a larger benchmark.

## Reference APIs

- TimesFM API examples: https://github.com/google-research/timesfm
- Uni2TS/Moirai examples: https://github.com/SalesforceAIResearch/uni2ts
- Chronos repository/model family: https://github.com/amazon-science/chronos-forecasting
