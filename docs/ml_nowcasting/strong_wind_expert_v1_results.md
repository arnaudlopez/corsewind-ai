# Strong Wind Expert v1

Generated: `2026-06-30`

Objective: test whether explicit strong-wind weighting improves CorseWind
predictions on the useful/high-wind regimes without blindly replacing the
global champion.

## Code Changes

Implemented in:

```text
scripts/ml_dataset/train_residual_correction_parquet.py
scripts/ml_dataset/compare_strong_wind_regime_runs.py
```

The residual trainer now supports progressive sample-weight rules:

```text
--target-high-wind-weight-rule-kt 12:...
--target-high-wind-weight-rule-kt 15:...
--target-high-wind-weight-rule-kt 20:...
--target-high-wind-weight-rule-kt 25:...
```

It also reports metrics by observed regime:

```text
<12kt
12-15kt
15-20kt
20-25kt
>=25kt
```

and threshold-detection metrics for:

```text
>=12kt
>=15kt
>=20kt
>=25kt
```

## Runs

Baseline, same data and model settings, no weighting:

```text
/srv/data/corsewind/ml_dataset/benchmarks/strong_wind_expert_lgbm_unweighted_v1
```

Aggressive weighted expert:

```text
/srv/data/corsewind/ml_dataset/benchmarks/strong_wind_expert_lgbm_weighted_12_15_20_25_v1
```

Weights:

```text
12kt -> 2x
15kt -> 4x
20kt -> 8x
25kt -> 12x
```

Soft weighted expert:

```text
/srv/data/corsewind/ml_dataset/benchmarks/strong_wind_expert_lgbm_weighted_soft_12_15_20_25_v1
```

Weights:

```text
12kt -> 1.25x
15kt -> 1.75x
20kt -> 2.5x
25kt -> 4x
```

All three runs use:

```text
training rows: 225000
test rows: 80000
split: 2026-01-01T00:00:00Z
model: LightGBM
targets: wind mean residual, gust residual
```

## Global Result

The unweighted baseline remains the best global median model in this benchmark.

| Target | Unweighted RMSE | Soft weighted RMSE | Aggressive weighted RMSE |
| --- | ---: | ---: | ---: |
| wind mean | 1.444 m/s | 1.469 m/s | 1.560 m/s |
| gust | 1.655 m/s | 1.712 m/s | 1.832 m/s |

Conclusion: do not promote either weighted model as the main median champion.

## Strong-Wind Regime Result

The weighting does improve the high-wind regimes.

Soft weighted vs unweighted:

| Target | Regime | Unweighted RMSE | Soft weighted RMSE | Gain |
| --- | --- | ---: | ---: | ---: |
| wind mean | 15-20kt | 1.545 | 1.511 | +2.2% |
| wind mean | 20-25kt | 1.868 | 1.712 | +8.4% |
| wind mean | >=25kt | 2.476 | 2.298 | +7.2% |
| gust | 20-25kt | 1.861 | 1.791 | +3.7% |
| gust | >=25kt | 2.485 | 2.314 | +6.9% |

Aggressive weighted vs unweighted:

| Target | Regime | Unweighted RMSE | Aggressive weighted RMSE | Gain |
| --- | --- | ---: | ---: | ---: |
| wind mean | 20-25kt | 1.868 | 1.576 | +15.6% |
| wind mean | >=25kt | 2.476 | 2.240 | +9.6% |
| gust | 20-25kt | 1.861 | 1.687 | +9.3% |
| gust | >=25kt | 2.485 | 2.297 | +7.6% |

But aggressive weighting strongly degrades low wind:

| Target | Regime | Unweighted RMSE | Aggressive weighted RMSE | Change |
| --- | --- | ---: | ---: | ---: |
| wind mean | <12kt | 1.311 | 1.542 | -17.6% |
| gust | <12kt | 1.467 | 1.829 | -24.7% |

## Detection Result

Soft weighted vs unweighted:

| Target | Threshold | Unweighted CSI | Soft weighted CSI | Unweighted recall | Soft weighted recall |
| --- | --- | ---: | ---: | ---: | ---: |
| wind mean | >=20kt | 0.602 | 0.619 | 0.674 | 0.726 |
| wind mean | >=25kt | 0.603 | 0.613 | 0.692 | 0.739 |
| gust | >=20kt | 0.681 | 0.691 | 0.779 | 0.831 |
| gust | >=25kt | 0.618 | 0.622 | 0.745 | 0.795 |

Aggressive weighted improves recall more, but with more false positives and
larger degradation on low wind.

## Decision

Do not replace the median champion.

Keep:

```text
unweighted model / current champion -> median p50
soft weighted model -> candidate strong-wind expert
aggressive weighted model -> diagnostic upper/risk expert only
```

The next useful step is a gated blend:

```text
if strong-wind probability/risk is low:
    use champion median
elif risk >=20kt:
    blend champion with soft weighted expert
elif risk >=25kt:
    optionally blend more strongly or expose aggressive expert as p80/p90
```

This matches what the numbers say: weighting helps the high-wind tail, but
should be activated only when the probability heads indicate a strong-wind
regime.

## Live Hindcast Gated Blend

Implemented in:

```text
scripts/ml_dataset/run_live_wind_and_gust_inference.py
scripts/ml_dataset/score_live_hindcast_predictions.py
```

The live inference now emits extra columns:

```text
strong_soft_wind_mean_kt
strong_soft_gust_kt
strong_aggressive_wind_mean_kt
strong_aggressive_gust_kt
strong_wind_total_weight
strong_gated_wind_mean_kt
strong_gated_gust_kt
```

Two gates were tested on the same 6 pseudo-live hindcasts:

```text
/srv/data/corsewind/ml_dataset/live_inference/hindcast_batch_strong_gated_v1
/srv/data/corsewind/ml_dataset/live_inference/hindcast_batch_strong_gated_v2_active
```

### Gate v1 Conservative

Parameters:

```text
p20_start=0.30
p20_full=0.65
p25_start=0.18
p25_full=0.35
soft_max_weight=0.35
aggressive_max_weight=0.20
cap_delta_ms=1.5
```

This gate was too cautious. It changed 661 / 3560 prediction rows, but the
mean total weight was only `0.0062`.

| Target | Raw RMSE | Champion RMSE | Strong gated RMSE |
| --- | ---: | ---: | ---: |
| wind mean | 4.167 kt | 3.615 kt | 3.613 kt |
| gust | 6.755 kt | 5.065 kt | 5.062 kt |

Conclusion: v1 is safe but almost a no-op.

### Gate v2 Active

Parameters now used as live defaults:

```text
p20_start=0.25
p20_full=0.35
p25_start=0.15
p25_full=0.20
soft_max_weight=0.65
aggressive_max_weight=0.35
cap_delta_ms=2.5
```

This gate changed 1603 / 3560 prediction rows. Mean total weight was `0.1529`,
with p95 at `0.7375`.

| Target | Raw RMSE | Champion RMSE | Strong gated RMSE | Change vs champion |
| --- | ---: | ---: | ---: | ---: |
| wind mean | 4.167 kt | 3.615 kt | 3.541 kt | +2.0% |
| gust | 6.755 kt | 5.065 kt | 4.970 kt | +1.9% |

By actual wind regime:

| Regime | Champion RMSE | Strong gated RMSE | Change |
| --- | ---: | ---: | ---: |
| <12 kt | 2.669 kt | 2.760 kt | -3.4% |
| 12-15 kt | 4.569 kt | 4.341 kt | +5.0% |
| 15-20 kt | 7.010 kt | 6.511 kt | +7.1% |
| 20-25 kt | 9.333 kt | 8.925 kt | +4.4% |

By actual gust regime:

| Regime | Champion RMSE | Strong gated RMSE | Change |
| --- | ---: | ---: | ---: |
| <15 kt | 3.244 kt | 3.314 kt | -2.2% |
| 15-20 kt | 5.577 kt | 5.505 kt | +1.3% |
| 20-25 kt | 8.299 kt | 7.872 kt | +5.1% |
| >=25 kt | 14.345 kt | 13.878 kt | +3.3% |

Threshold detection remains weak:

| Threshold | Champion CSI | Strong gated CSI | Raw CSI |
| --- | ---: | ---: | ---: |
| wind >=12 kt | 0.081 | 0.125 | 0.147 |
| wind >=15 kt | 0.059 | 0.086 | 0.071 |
| gust >=15 kt | 0.253 | 0.285 | 0.391 |
| gust >=20 kt | 0.032 | 0.039 | 0.200 |
| gust >=25 kt | 0.000 | 0.000 | 0.000 |

## Updated Decision

Use `strong_gated_*` as the candidate strong-wind forecast in shadow/live
evaluation. It improves global RMSE and the strong regimes, but it should not
be considered a solved gust-alert model yet.

Current interpretation:

```text
champion_* -> safest median baseline
strong_gated_* -> better candidate when we care about windsurf/strong-wind days
gust_high_* -> useful diagnostic/risk upper variant, not calibrated enough
raw AROME/AROME-PI -> still detects some gust exceedances better, but with many false positives
```

Next scientific step: train a direct regime/router model that chooses between
champion, raw, gust_high, and strong_gated per row. The current probability
gate helps, but it still misses too many >=20 kt gust events.

Live artifact generated after this decision:

```text
/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260630T17_shadow_v2_offset_fallback/shadow_foundation_fresh/predictions_strong_gated_v2_active/predictions.parquet
```

On this evening live run, the gate mostly affected Cap Corse and corrected
gusts downward by up to about `0.64 kt`; this was a low-to-moderate risk run,
not a true strong-wind validation case.
