# Gust Quantile Anti-Smoothing P0 Results

Date: 2026-07-02

This run responds to the SAPHIR-vs-CorseWind review: the gust pipeline was
optimizing point-estimate RMSE with L2 losses and a `scale070` shrinkage, which
systematically reduced variance and under-called strong gusts.

## Change Tested

Same training/evaluation scope as the current gust champion:

- calibration: `2025-07-01` to `2026-01-01`;
- evaluation: `2026-01-01` to `2026-07-01`;
- leads: `15,30,45,60` minutes;
- rows: `31,429`;
- base first-stage run:
  `tabular_lgbm_225k_prev_lowmem_gust_from_wind_champion_recipe_2024_2025_to_2026_v1`.

New second-stage heads:

- LightGBM quantile objective / pinball loss;
- quantiles: `q33,q40,q50,q60,q67,q70,q75,q90`;
- no `scale070`;
- wider correction clip: `5.0 m/s`;
- output columns are suffixed, for example `calibrated_gust_ms_q60`, so the
  current champion column is not overwritten.

Artifacts:

```text
/srv/data/corsewind/ml_dataset/benchmarks/gust_quantile_p0_lgbm_pinball_2025h2_to_2026_v1
```

Key files:

```text
predictions_quantile_merged.parquet
predictions_quantile_merged_with_q33_q40.parquet
anti_smoothing_audit.json
anti_smoothing_audit_with_q33_q40.json
anti_smoothing_audit.md
```

## Overall Result

| Rail | RMSE m/s | MAE m/s | Bias m/s | Variance ratio |
| --- | ---: | ---: | ---: | ---: |
| `corrected_gust_ms` | 1.501342 | 1.096947 | 0.132077 | 0.883168 |
| `champion_scale070_gust_ms` | 1.484221 | 1.073906 | 0.056219 | 0.920166 |
| `calibrated_gust_ms_q33` | 1.572365 | 1.124519 | -0.547220 | 0.891346 |
| `calibrated_gust_ms_q40` | 1.505651 | 1.072243 | -0.326542 | 0.905983 |
| `calibrated_gust_ms_q50` | 1.474077 | 1.055054 | -0.019976 | 0.935749 |
| `calibrated_gust_ms_q60` | 1.522944 | 1.102714 | 0.326135 | 0.969673 |
| `calibrated_gust_ms_q75` | 1.770661 | 1.326825 | 0.888022 | 1.025283 |
| `calibrated_gust_ms_q90` | 2.355572 | 1.858755 | 1.673428 | 1.105025 |

Interpretation:

- `q50` is the best central-value candidate: it beats the current gust champion
  on RMSE and MAE while raising variance ratio from `0.920` to `0.936`.
- `q60` is not a central-value candidate, but it is the best threshold/event
  rail. It restores variance close to observed variance without behaving like
  raw AROME.
- `q75/q90` are too aggressive for deterministic display, but remain useful as
  upper-risk bands.

## Threshold Skill

| Rail | CSI >=12 kt | CSI >=15 kt | CSI >=20 kt | CSI >=25 kt |
| --- | ---: | ---: | ---: | ---: |
| `champion_scale070_gust_ms` | 0.808078 | 0.775267 | 0.693467 | 0.610515 |
| `calibrated_gust_ms_q33` | 0.780099 | 0.736507 | 0.636346 | 0.554326 |
| `calibrated_gust_ms_q40` | 0.796935 | 0.758358 | 0.661743 | 0.579721 |
| `calibrated_gust_ms_q50` | 0.808864 | 0.776348 | 0.688993 | 0.615442 |
| `calibrated_gust_ms_q60` | 0.816909 | 0.785109 | 0.705526 | 0.625568 |
| `calibrated_gust_ms_q67` | 0.813355 | 0.782378 | 0.702068 | 0.625000 |
| `calibrated_gust_ms_q75` | 0.802241 | 0.772897 | 0.683667 | 0.612152 |
| `calibrated_gust_ms_q90` | 0.754596 | 0.722059 | 0.623150 | 0.541737 |

`q60` wins all four tested threshold CSI scores. Follow-up q33/q40 tests were
added to check the SAPHIR quantile-decision intuition; on this split they
under-trigger relative to q50/q60 and do not improve CSI. The current best match
to the SAPHIR lesson is therefore: keep a central q50 rail, but make threshold
decisions from q60 / calibrated exceedance probabilities rather than from a
shrunk L2 mean.

## Strong-Gust Regime

For actual gusts `>=25 kt`:

| Rail | RMSE m/s | Bias m/s | Variance ratio |
| --- | ---: | ---: | ---: |
| `champion_scale070_gust_ms` | 2.417757 | -0.966166 | 1.365001 |
| `calibrated_gust_ms_q50` | 2.452074 | -0.982059 | 1.429085 |
| `calibrated_gust_ms_q60` | 2.303932 | -0.432662 | 1.440166 |
| `calibrated_gust_ms_q75` | 2.302547 | 0.389893 | 1.476672 |

`q60` cuts the strong-gust under-bias by more than half and improves strong
gust RMSE versus the current champion. It does this at the cost of calm-regime
overprediction, so it should be routed/gated rather than used as the single
display value.

## Decision

Do not promote a single quantile as the full gust product yet.

Recommended next product split:

- displayed/central gust: test-promote `calibrated_gust_ms_q50` against the
  current gust champion;
- threshold/session decisions: use `calibrated_gust_ms_q60` or a calibrated
  probability derived from the quantile family;
- upper-risk band: expose `q75/q90` internally for uncertainty, not as the
  deterministic gust forecast.

Promotion gates must now include:

- RMSE/MAE non-regression for the central gust rail;
- CSI at `12,15,20,25 kt`;
- variance ratio, with a target floor near `0.9` overall and no collapse in
  strong-gust regimes;
- calm-regime overprediction guard, because q60/q75/q90 are intentionally more
  sensitive to events.

## Code Changes

- `scripts/ml_dataset/train_prediction_residual_calibrator.py`
  - added `--objective quantile`;
  - added `--quantile-alpha`;
  - added suffixed output columns via `--prediction-suffix`;
  - added pinball loss support for scale-selection diagnostics;
  - added variance-ratio metrics.
- `scripts/ml_dataset/audit_gust_quantile_anti_smoothing.py`
  - new audit for RMSE/MAE/bias, variance ratio, regime metrics and threshold
    CSI for quantile gust rails.

## Follow-Up Implementation

Additional work completed after the first benchmark:

- `scripts/ml_dataset/run_live_wind_and_gust_inference.py`
  - loads saved gust quantile calibrators from
    `/srv/data/corsewind/ml_dataset/benchmarks/gust_quantile_p0_lgbm_pinball_2025h2_to_2026_v1`;
  - emits `calibrated_gust_ms_q50/q60/q75/q90` and knot equivalents;
  - keeps the existing deterministic gust rail intact.
- `scripts/ml_dataset/score_live_hindcast_predictions.py`
  - scores gust quantile rails when observed gusts are available;
  - adds `variance_ratio` to overall metrics;
  - adds threshold CSI for quantile gust rails.
  - adds empirical quantile coverage overall, by spot, by lead bucket, by
    spot/lead bucket, and by actual gust regime;
  - emits conformal additive offsets from `actual - quantile` residuals;
  - reports the best quantile rail per gust threshold overall and by issue day.
- `scripts/ml_dataset/summarize_shadow_suites.py`
  - rolls up quantile metrics across shadow suites;
  - carries variance ratio into JSON and Markdown summaries.
  - carries false alarm ratio/rate through threshold rollups.
- `scripts/ml_dataset/assert_shadow_promotion_gate.py`
  - now supports `--min-variance-ratio`;
  - optionally supports a regime-specific variance floor with
    `--require-variance-regime --variance-regime group/regime`.
  - now checks maximum false-alarm-ratio regression versus baselines.
- `scripts/ml_dataset/review_shadow_promotion_candidates.py`
  - propagates variance-ratio promotion checks across candidate review;
  - defaults gust candidate review to `--gust-min-variance-ratio 0.85`.
- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`
  - applies the gust anti-smoothing floor in daily shadow postprocess via
    `GUST_MIN_VARIANCE_RATIO`, default `0.85`.
  - applies a windy-regime variance floor on `actual_gust/>=25kt` via
    `GUST_MIN_WINDY_VARIANCE_RATIO`, default `0.80`;
  - applies a false-alarm-ratio regression guard, default `0.03`.
- `scripts/ml_dataset/run_live_wind_and_gust_inference.py`
  - sorts quantile rails monotonically per row before serving, so q90 cannot be
    lower than q50 even if independently trained heads cross.
- `scripts/ml_dataset/extract_wind2d_patch_features.py`
  - new extractor for local NWP patch statistics around each spot.

## Live Smoke

Smoke input:

```text
/srv/data/corsewind/ml_dataset/live_inference/hindcast_auto_20260630T0645_foundation_v1/training_rows/training_rows.parquet
```

Smoke output:

```text
/srv/data/corsewind/ml_dataset/benchmarks/gust_quantile_live_smoke_monotone_v1/predictions.parquet
```

Result:

- rows: `820`;
- spots: `20`;
- produced q50/q60/q75/q90 gust columns successfully;
- monotonic crossings after write: `0`;
- hindcast scoring also produced `gust_quantile_q50_kt`,
  `gust_quantile_q60_kt`, `gust_quantile_q75_kt`, `gust_quantile_q90_kt`.

Fresh-day coverage smoke on `2026-06-30`, official scored spots:

| Rail | Expected coverage | Empirical coverage | Conformal offset kt |
| --- | ---: | ---: | ---: |
| `q50` | 0.50 | 0.440678 | 0.800857 |
| `q60` | 0.60 | 0.480226 | 1.145135 |
| `q75` | 0.75 | 0.598870 | 1.804478 |
| `q90` | 0.90 | 0.711864 | 3.875760 |

Interpretation: the rails are promising on the large locked split, but this
fresh-day smoke is under-covered. Promotion now requires multi-day coverage
stability and conformal calibration by spot/lead bucket.

This proves the live/shadow plumbing. It is not a promotion proof because this
smoke is a small fresh-day slice.

## Wind2D Patch Smoke

Smoke input:

```text
/srv/data/corsewind/ml_dataset/model_runs/arome/run_20260702T150000Z/arome_20260702T150000Z.json.gz
```

Smoke output:

```text
/srv/data/corsewind/ml_dataset/benchmarks/wind2d_patch_smoke_v1
```

Result:

- rows: `1,225`;
- shape: `25 spots * 49 steps`;
- patch radii: `1` and `2` grid cells;
- features include center, min, mean, max, std, range and max-minus-center for
  wind speed / u / v / gust fields when available.

This gives the next likely physical lever for gusts: local NWP gradients and
nearby stronger cells, instead of only the value sampled exactly at the spot.

## Gust-Factor Smoke

Tested the proposed `log(actual_gust / wind_reference)` path on the fresh
hindcast smoke table.

Artifact:

```text
/srv/data/corsewind/ml_dataset/benchmarks/gust_factor_smoke_v1/results.json
```

Result on `177` rows:

| Rail | RMSE m/s | MAE m/s | Bias m/s | Variance ratio |
| --- | ---: | ---: | ---: | ---: |
| `calibrated_gust_factor_ms_q50` | 1.503218 | 0.911302 | -0.095102 | 0.422360 |

Verdict: do not promote this path yet. The framing is physically attractive,
but this small smoke collapses variance badly. It should only be revisited on a
larger backtest and likely with explicit patch features / high-wind weighting.
