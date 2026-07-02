# Hindcast Evaluation Review - 2026-06-30

## Objective

Validate the live nowcasting rail as a pseudo-live hindcast:

- build a prediction as if it was issued earlier in the day;
- compare it later against real observations;
- compare `AROME/AROME-PI raw` vs `ML champion` vs, where available, the guarded
  foundation shadow.

## Implemented Tooling

New scripts:

```text
scripts/ml_dataset/score_live_hindcast_predictions.py
scripts/ml_dataset/run_aromepi_hindcast_evaluation.py
```

`score_live_hindcast_predictions.py` joins predictions to observations by spot
and nearest timestamp, then reports:

- RMSE, MAE, bias and p90 absolute error;
- global metrics;
- metrics by spot;
- metrics by lead bucket: `0-1h`, `1-3h`, `3-6h`, `6h+`;
- metrics by actual and raw gust regimes;
- deterministic threshold scores for wind `>=15 kt`, gust `>=20 kt`,
  gust `>=25 kt`;
- peak wind/gust errors by spot.

`run_aromepi_hindcast_evaluation.py` orchestrates:

1. AROME-PI pseudo-live grid metadata;
2. feature-store build;
3. residual input rows;
4. champion inference, optionally with Foundation shadow;
5. hindcast scoring.

## Runs

Batch root:

```text
/srv/data/corsewind/ml_dataset/live_inference/hindcast_batch_v1
```

Foundation shadow run:

```text
/srv/data/corsewind/ml_dataset/live_inference/hindcast_auto_20260630T0645_foundation_v1
```

Scored observation spots:

```text
cap_corse, la_parata, lfkf, lfkj, lfks, lfvf, lfvh
```

These are the spots with same-day Meteo-France observations available for this
review.

## Main Batch Result

Pooled over 6 pseudo-live runs, 783 scored rows:

| Target | ML RMSE | Raw AROME RMSE | ML MAE | Raw AROME MAE | ML Bias | Raw Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wind mean | 3.62 kt | 4.17 kt | 2.83 kt | 3.23 kt | -0.83 kt | -1.50 kt |
| gust | 5.06 kt | 6.76 kt | 3.86 kt | 4.83 kt | -2.11 kt | +1.69 kt |

Interpretation:

- the ML champion improves global wind mean error;
- the ML champion improves global gust RMSE/MAE;
- the ML champion flips AROME's positive gust bias into a negative bias;
- this is good for average error, but dangerous for strong-gust detection.

## By Day

| Day | Wind ML RMSE | Wind Raw RMSE | Gust ML RMSE | Gust Raw RMSE | Gust ML Bias | Gust Raw Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-29 | 4.15 kt | 4.84 kt | 5.99 kt | 4.92 kt | -4.30 kt | -1.11 kt |
| 2026-06-30 | 3.26 kt | 3.71 kt | 4.43 kt | 7.64 kt | -0.81 kt | +3.35 kt |

Interpretation:

- on `2026-06-30`, the ML correction is clearly useful, especially for rafales;
- on `2026-06-29`, the ML is too conservative for rafales and worse than raw
  AROME;
- the current champion is not uniformly reliable across regimes.

## By Lead Bucket

| Lead bucket | Wind ML RMSE | Wind Raw RMSE | Gust ML RMSE | Gust Raw RMSE |
| --- | ---: | ---: | ---: | ---: |
| 0-1h | 3.22 kt | 3.80 kt | 4.75 kt | 6.34 kt |
| 1-3h | 3.94 kt | 4.46 kt | 5.09 kt | 6.93 kt |
| 3-6h | 3.58 kt | 4.21 kt | 5.09 kt | 7.18 kt |
| 6h+ | 3.48 kt | 3.96 kt | 5.15 kt | 6.19 kt |

Interpretation:

- the ML champion improves global RMSE across all lead buckets;
- the improvement does not solve the peak problem.

## Strong Gust Regimes

Pooled rows where actual gust exceeds threshold:

| Actual gust regime | N | ML RMSE | Raw RMSE | ML Bias | Raw Bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| `>=15 kt` | 254 | 7.56 kt | 6.03 kt | -6.58 kt | -2.77 kt |
| `>=20 kt` | 110 | 9.55 kt | 7.12 kt | -8.93 kt | -4.99 kt |
| `>=25 kt` | 18 | 14.34 kt | 11.76 kt | -14.12 kt | -11.36 kt |

Interpretation:

- the champion is better on average but worse on strong gusts;
- it suppresses peak amplitude too aggressively;
- raw AROME is also low on the largest observed peaks, but less low than ML.

## Threshold Detection

For the `2026-06-30 06:45 UTC` pseudo-live run:

| Threshold | ML Recall | Raw Recall | ML CSI | Raw CSI |
| --- | ---: | ---: | ---: | ---: |
| wind `>=15 kt` | 0.67 | 0.67 | 0.40 | 0.22 |
| gust `>=20 kt` | 0.17 | 0.83 | 0.11 | 0.32 |
| gust `>=25 kt` | 0.00 | 0.00 | 0.00 | 0.00 |

Interpretation:

- for navigable wind mean threshold, ML is cleaner;
- for gust warning threshold, raw AROME currently catches more events;
- neither system catches `>=25 kt` gusts well in this sample.

## Foundation Shadow

Foundation shadow run:

```text
/srv/data/corsewind/ml_dataset/live_inference/hindcast_auto_20260630T0645_foundation_v1
```

For `2026-06-30 06:45 UTC`:

| Target | Champion RMSE | Shadow RMSE |
| --- | ---: | ---: |
| wind mean | 3.288 kt | 3.276 kt |
| gust | 4.621 kt | 4.626 kt |

Interpretation:

- Foundation shadow is neutral on this run;
- it slightly improves wind mean;
- it slightly degrades gust;
- this is not enough to promote it.

## Spot-Level Gust Findings

| Spot | ML Gust RMSE | Raw Gust RMSE | ML Bias | Raw Bias | Actual Max | ML Max | Raw Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cap_corse | 7.69 | 6.72 | -4.64 | -3.55 | 34.8 | 15.0 | 18.3 |
| la_parata | 5.29 | 4.59 | -3.99 | -1.61 | 20.0 | 15.4 | 21.4 |
| lfkf | 4.21 | 4.44 | -2.05 | +2.55 | 24.9 | 15.8 | 21.0 |
| lfkj | 4.47 | 9.51 | -0.55 | +4.91 | 19.8 | 21.5 | 35.0 |
| lfks | 3.44 | 8.76 | -0.16 | +5.73 | 23.1 | 20.5 | 34.4 |
| lfvf | 4.40 | 3.86 | -2.34 | +0.16 | 23.5 | 15.1 | 18.5 |
| lfvh | 4.84 | 4.22 | -2.44 | +1.28 | 30.1 | 19.3 | 23.3 |

Interpretation:

- ML strongly fixes AROME over-forecasting at `lfkj` and `lfks`;
- ML under-forecasts peak-prone sites like `cap_corse`, `lfvh`, `lfvf`;
- a single gust model is probably not enough.

## Scientific Conclusion

The current champion is doing the right thing for the average objective:

```text
reduce systematic AROME/AROME-PI gust over-forecasting
```

But for the windsurf product, average error is not enough. The important
failure mode is now:

```text
over-smoothing high gust regimes and missing peak amplitude
```

So the next model iteration should not replace the champion. It should add a
second decision layer for strong gust risk.

## Recommended Next Steps

1. Keep the current champion for median wind/gust correction.

2. Add a `gust_peak_guard` output:

   - use raw AROME gust as an alarm feature;
   - use actual ML correction only when peak-risk features are weak;
   - limit downward correction when raw AROME gust is high and the context is
     physically supportive.

3. Train/evaluate a separate high-quantile gust model:

   - target: observed gust P90/P95 or daily/rolling peak;
   - metrics: recall and CSI for `gust >=20 kt` and `gust >=25 kt`;
   - not optimized only for global RMSE.

4. Add product outputs:

   ```text
   gust_median_kt
   gust_high_kt
   P(gust >= 20kt)
   P(gust >= 25kt)
   peak_risk_level
   ```

5. Continue daily hindcast automation:

   - run at `06:45`, `08:45`, `10:45`, `13:45 UTC`;
   - score when observations are available;
   - keep a rolling leaderboard by day, spot, lead, threshold and regime.

6. Fix Beacon/WindsUp live freshness.

   The current reliable same-day scoring is mostly Meteo-France station based.
   We still need fresh Beacon/WindsUp observations for windsurf-specific spots
   to validate Balistra, Tonnara, Piantarella, Santa Manza, Porto Polo, etc.

## Implemented Follow-Up - Gust Peak Guard v1

Implemented in:

```text
scripts/ml_dataset/run_live_wind_and_gust_inference.py
scripts/ml_dataset/score_live_hindcast_predictions.py
```

The live inference now keeps:

```text
champion_gust_kt
```

as the median/main gust forecast, and adds:

```text
gust_high_kt
prob_gust_ge_20kt
prob_gust_ge_25kt
peak_risk_level
```

as a peak-risk rail.

Default recipe:

```text
if raw_arome_gust_kt >= 12 kt and raw_arome_gust_kt > champion_gust_kt:
    gust_high_kt = champion_gust_kt + min(0.80 * (raw_arome_gust_kt - champion_gust_kt), 5 kt)
else:
    gust_high_kt = champion_gust_kt
```

This is intentionally not the production median forecast. It is an upper/risk
output designed to recover part of AROME's peak signal when the champion model
is likely over-smoothing.

Validation batch:

```text
/srv/data/corsewind/ml_dataset/live_inference/hindcast_batch_gust_peak_guard_v1
```

Pooled over the same 6 hindcasts and 783 rows:

| Gust output | RMSE | MAE | Bias | Recall `>=20 kt` | CSI `>=20 kt` |
| --- | ---: | ---: | ---: | ---: | ---: |
| champion median | 5.06 kt | 3.86 kt | -2.11 kt | 0.04 | 0.03 |
| gust_high v1 | 5.27 kt | 4.13 kt | +0.21 kt | 0.12 | 0.09 |
| raw AROME | 6.76 kt | 4.83 kt | +1.69 kt | 0.35 | 0.20 |

On strong observed gusts:

| Actual gust regime | Champion RMSE | Gust high RMSE | Raw AROME RMSE |
| --- | ---: | ---: | ---: |
| `>=15 kt` | 7.56 kt | 6.00 kt | 6.03 kt |
| `>=20 kt` | 9.55 kt | 7.42 kt | 7.12 kt |
| `>=25 kt` | 14.34 kt | 12.26 kt | 11.76 kt |

Conclusion:

- `gust_high` should not replace the champion gust median;
- it materially improves the high-gust failure mode versus champion;
- it remains less aggressive than raw AROME;
- raw AROME is still the strongest recall signal for `>=20 kt`, but with many
  false alarms and worse global RMSE.

Live regenerated output:

```text
/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260630T17_shadow_v2_offset_fallback/shadow_foundation_fresh/predictions_peak_guard_v1
```

For the `2026-06-30T17:45Z` issue, the peak guard activated on 25 rows, with
no capped deltas. Risk remains low on the inspected spots, because raw AROME is
itself weak for the evening window.

## Implemented Follow-Up - Gust Probability Heads v1

Implemented in:

```text
scripts/ml_dataset/train_gust_threshold_probability.py
scripts/ml_dataset/run_live_wind_and_gust_inference.py
scripts/ml_dataset/score_live_hindcast_predictions.py
```

Trained binary ExtraTrees probability heads for:

```text
labels__target_gust_gt_20kt
labels__target_gust_gt_25kt
```

Training run:

```text
/srv/data/corsewind/ml_dataset/benchmarks/gust_threshold_probability_extratrees_2024_2025_to_2026_v1
```

Training split: 2024-2025 train, 2026 test sample, capped at 220k train rows
and 100k test rows.

| Target | Brier | AUC | AP | Positive rate | Best CSI | Best threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gust `>=20 kt` | 0.067 | 0.966 | 0.855 | 15.7% | 0.620 | 0.65 |
| gust `>=25 kt` | 0.050 | 0.974 | 0.806 | 8.7% | 0.567 | 0.70 |

Important: this clean offline score does not transfer directly to the small
pseudo-live hindcast window. On the 6-run hindcast batch the probability heads
rank risk better, but are not calibrated enough to use a generic `0.50`
threshold.

Validation batch:

```text
/srv/data/corsewind/ml_dataset/live_inference/hindcast_batch_gust_probability_v1
```

Pooled over the same 6 hindcasts and 783 rows:

| Probability output | Brier | AUC | AP | Mean probability | Positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| model `P(gust >=20 kt)` | 0.123 | 0.779 | 0.297 | 0.253 | 14.0% |
| heuristic `P(gust >=20 kt)` | 0.123 | 0.756 | 0.281 | 0.181 | 14.0% |
| model `P(gust >=25 kt)` | 0.031 | 0.858 | 0.226 | 0.119 | 2.3% |
| heuristic `P(gust >=25 kt)` | 0.034 | 0.654 | 0.036 | 0.072 | 2.3% |

Best operational CSI thresholds on this hindcast:

| Target | Output | Threshold | TP | FP | FN | Precision | Recall | CSI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `>=20 kt` | model | 0.30 | 66 | 132 | 44 | 0.33 | 0.60 | 0.273 |
| `>=20 kt` | heuristic | 0.28 | 61 | 132 | 49 | 0.32 | 0.55 | 0.252 |
| `>=25 kt` | model | 0.18 | 8 | 22 | 10 | 0.27 | 0.44 | 0.200 |
| `>=25 kt` | heuristic | 0.03 | 18 | 467 | 0 | 0.04 | 1.00 | 0.037 |

Conclusion:

- keep the champion median for RMSE/MAE;
- keep `gust_high_kt` as an upper/risk rail;
- use probability heads as ranking/alert features, not as raw calibrated
  probabilities yet;
- provisional alert thresholds from hindcast are `P20_model >= 0.30` and
  `P25_model >= 0.18`;
- the `>=25 kt` model is the strongest new signal from this iteration because
  it separates rare strong-gust risk far better than the heuristic.

Live regenerated output with probability heads:

```text
/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260630T17_shadow_v2_offset_fallback/shadow_foundation_fresh/predictions_probability_heads_v1
```

For the evening window `2026-06-30T18:00Z` to `2026-06-30T23:00Z`, no spot
crosses the provisional `P25_model >= 0.18` alert threshold. `cap_corse` is the
only spot crossing the provisional `P20_model >= 0.30` threshold, while median
and raw gust forecasts remain weak.

## Implemented Follow-Up - Operational Gust Alert Flags v1

Implemented in:

```text
scripts/ml_dataset/calibrate_gust_probability_alerts.py
scripts/ml_dataset/run_live_wind_and_gust_inference.py
scripts/ml_dataset/score_live_hindcast_predictions.py
```

The probability thresholds are now stored as a deployable artifact:

```text
/srv/data/corsewind/ml_dataset/benchmarks/gust_probability_alert_thresholds_hindcast_v1/gust_probability_alert_thresholds.json
```

The live inference consumes this file and emits:

```text
gust_alert_ge_20kt
gust_alert_ge_20kt_probability
gust_alert_ge_20kt_threshold
gust_alert_ge_25kt
gust_alert_ge_25kt_probability
gust_alert_ge_25kt_threshold
gust_operational_risk_level
```

Pooled verification over the same 6 hindcasts and 783 rows:

| Alert flag | Threshold source | TP | FP | FN | Precision | Recall | CSI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gust_alert_ge_20kt` | `prob_gust_ge_20kt_model >= 0.30` | 66 | 132 | 44 | 0.33 | 0.60 | 0.273 |
| `gust_alert_ge_25kt` | `prob_gust_ge_25kt_model >= 0.18` | 8 | 22 | 10 | 0.27 | 0.44 | 0.200 |

Live regenerated output with operational alerts:

```text
/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260630T17_shadow_v2_offset_fallback/shadow_foundation_fresh/predictions_probability_alerts_v1
```

For `2026-06-30T18:00Z` to `2026-06-30T23:00Z`, the live output has 399 rows
with risk `none` and 21 rows with risk `moderate`. Only `cap_corse` triggers
`gust_alert_ge_20kt`; no spot triggers `gust_alert_ge_25kt`.
