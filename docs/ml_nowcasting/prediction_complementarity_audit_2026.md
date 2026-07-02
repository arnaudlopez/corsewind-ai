# Prediction Complementarity Audit 2026

Generated: 2026-06-30

## Question

We have repeatedly added new data/model families without beating the locked
wind-mean champion. The goal of this audit was to answer a more basic
scientific question:

Do the other models contain complementary signal, or are all models making the
same errors?

If a row-wise oracle is much better than the champion, then an improvement may
exist but we need a better router/blender. If the oracle is barely better, then
the bottleneck is probably target noise, missing input signal, or a genuine
short-term predictability ceiling.

## Important Correction

The first local audit accidentally used `corrected_wind_mean_ms` from the
champion file. The locked champion is `calibrated_wind_mean_ms`.

All conclusions below use the true champion column:

```text
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/calibrated_predictions_2026.parquet
column: calibrated_wind_mean_ms
```

## Script Added

New audit script:

```text
scripts/ml_dataset/audit_prediction_complementarity.py
```

It computes:

- native model metrics;
- strict same-sample intersections;
- row-wise oracle metrics;
- oracle model selection shares;
- pairwise oracle vs base;
- error correlations and same-side error share;
- metrics by lead/spot/bin.

New deployable router probe:

```text
scripts/ml_dataset/train_prediction_pair_router.py
```

It trains on calibration predictions, selects a threshold on an internal
temporal validation split, then evaluates on 2026.

## Pairwise Oracle Results

All rows below use strict intersections with the true champion.

| Candidate | Rows | Champion RMSE on intersection | Candidate/Alt RMSE | Oracle RMSE | Oracle gain vs champion |
| --- | ---: | ---: | ---: | ---: | ---: |
| existing router | 31429 | 1.268019 | 1.276534 | 1.226632 | 3.264% |
| LGBM top700 | 18162 | 1.306431 | 1.309026 | 1.182041 | 9.521% |
| weighted oldbest/top700 | 18162 | 1.306431 | 1.291417 | 1.234013 | 5.543% |
| SAPHIR V1 HGB | 1237 | 1.243410 | 1.263867 | 1.091136 | 12.246% |
| SAPHIR V2 HGB | 554 | 1.279930 | 1.396033 | 1.148891 | 10.238% |

Interpretation:

- The existing router has almost the same errors as the champion. Error
  correlation is extremely high, so it is not a useful new direction.
- LGBM top700 is almost tied with the champion on its intersection, but the
  oracle is much better. This means it is sometimes better on different rows.
- SAPHIR V1/V2 also show complementary behavior, but their intersections are
  small, especially V2, so they are smoke signals rather than production
  evidence.

## Foundation Model Audit

Small 2026 sequence benchmark:

```text
/srv/data/corsewind/ml_dataset/benchmarks/sequence_2026_windsurf_1h_v2/predictions_final_all_models.parquet
```

Rows: `336`.

| Model | RMSE |
| --- | ---: |
| Chronos2 univariate | 1.235208 |
| TimesFM | 1.271544 |
| HGB sequence | 1.276752 |
| Chronos2 residual | 1.495813 |
| Moirai | 1.683513 |
| Raw NWP | 1.900426 |
| Row-wise oracle | 0.628239 |

This is too small to be production evidence, but it strongly suggests that the
foundation models are not simply useless. They are often wrong in different
ways. The problem is routing/calibration, not only raw model quality.

## Deployable Router Probe

The most promising large intersection was champion vs LGBM top700:

- champion RMSE: `1.306431`;
- LGBM top700 RMSE: `1.309026`;
- row-wise oracle RMSE: `1.182041`.

We reconstructed champion calibration predictions for 2025h2 with:

```text
corrected_wind_mean_ms + 0.70 * predicted_second_stage_residual_wind_mean_ms
```

Then we tested two leakage-safe deployable selectors on 2025h2 -> 2026.

Weighted ensemble:

| Strategy selected | Base RMSE | Alt RMSE | Ensemble RMSE | Gain |
| --- | ---: | ---: | ---: | ---: |
| spot_lead | 1.306431 | 1.309026 | 1.304860 | 0.120% |

HGB pair router:

| Model | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| base champion | 1.306431 | 0.983239 | 0.071672 |
| alt LGBM | 1.309026 | 0.984899 | 0.102249 |
| HGB router | 1.306223 | 0.982894 | 0.071628 |
| oracle | 1.182041 | 0.850980 | 0.071873 |

ExtraTrees pair router:

| Model | RMSE |
| --- | ---: |
| base champion | 1.306431 |
| alt LGBM | 1.309026 |
| ExtraTrees router | 1.306431 |
| oracle | 1.182041 |

The HGB router selected the alternative only `0.0826%` of the time. ExtraTrees
selected it `0%` of the time.

## Main Conclusion

There is complementary signal in the predictions, but our current issue-time
features do not reliably identify when to switch models.

This explains the repeated failures:

1. New models often contain useful information.
2. They do not dominate the champion globally.
3. The row-wise oracle can improve a lot.
4. But a leakage-safe router trained on current features cannot capture that
   oracle yet.

So the bottleneck is not simply "try another ML model". The bottleneck is
decision information: we need features that predict model trustworthiness
before the target observation is known.

## What This Means For The RMSE Goal

The target RMSE `0.9` is not proven impossible, but current evidence says:

- naive ensembling will not get us there;
- adding model families without a trust/router signal will not get us there;
- SAPHIR-style data organization alone will not get us there;
- foundation models may help only if routed/calibrated correctly;
- the remaining gain is concentrated in knowing when the champion will fail.

## Next Best Work

The next research task should be a "forecast failure predictor", not another
plain residual regressor.

Candidate target:

```text
will_abs_error_champion_exceed_1_5ms
or
will_alt_model_beat_champion_by_0_25ms
```

Candidate features must be available at issue time:

- disagreement between AROME / AROME-PI / Open-Meteo previous runs;
- recent observed acceleration/deceleration;
- recent model error trend;
- direction shift rate;
- gust factor instability;
- thermal regime flags;
- land-sea temperature gradient and its trend;
- cloud/solar regime;
- station freshness and station disagreement;
- spot/lead-specific historical reliability.

If we cannot predict champion failure better than chance, then the practical
ceiling is probably close to the current score with the current observation
quality.
