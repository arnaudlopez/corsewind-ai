# Scientific Error Diagnostic Report

Generated: 2026-06-28

Champion run:
`prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1`

Prediction artifact:
`/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/calibrated_predictions_2026.parquet`

Local diagnostic artifacts:

- `docs/ml_nowcasting/scientific_error_diagnostic_v1/scientific_error_diagnostic.md`
- `docs/ml_nowcasting/scientific_error_diagnostic_v1/gap_oracles_full_common.md`
- `docs/ml_nowcasting/scientific_error_diagnostic_v1/feature_family_coverage_hard_regimes.md`

## Executive Summary

The current best validated wind-mean model is strong compared with the raw
forecast prior, but it is not close to RMSE 0.9 yet.

Current champion:

- RMSE: 1.268019 m/s
- MAE: 0.930465 m/s
- Bias: +0.018767 m/s
- Rows: 31,429

The raw AROME-like prior is already reduced from RMSE 2.187306 to 1.276846 by
the first correction stage. The final residual calibration only improves that
to 1.268019. This means the easy global bias correction has mostly been done.
The remaining error is not a simple scale problem.

To reach RMSE 0.9, we need a 49.623% reduction in MSE. The key point is that
the gap is tail-dominated:

- Worst 1% rows carry 16.983% of SSE.
- Worst 5% rows carry 40.993% of SSE.
- Worst 10% rows carry 56.654% of SSE.
- Only 7.452% of rows, if perfectly corrected, would mathematically close the
  gap to RMSE 0.9.

The most likely path is therefore not "one more generic model". It is a
scientific/regime strategy: identify and correct the physical situations where
the current system collapses.

## Main Diagnosis

The remaining error is dominated by four interacting axes:

1. Hard spots: especially La Tonnara and Santa Manza.
2. Longer short-term horizons: +45 and +60 minutes.
3. High-wind regimes: actual wind >= 8 m/s.
4. Compression bias: the model overpredicts very light wind and underpredicts
   strong wind.

Worst spots by SSE:

| Spot | Rows | RMSE | Bias | SSE share |
| --- | ---: | ---: | ---: | ---: |
| La Tonnara | 4,287 | 1.520349 | +0.009216 | 19.609% |
| Santa Manza | 4,096 | 1.497476 | -0.125203 | 18.176% |
| Balistra | 4,858 | 1.234581 | +0.132580 | 14.653% |
| Porticcio | 6,366 | 1.010447 | -0.191237 | 12.862% |
| Porto Polo | 3,761 | 1.170954 | +0.314158 | 10.205% |

Worst horizons:

| Lead | Rows | RMSE | SSE share | Global RMSE if perfect |
| --- | ---: | ---: | ---: | ---: |
| +60 min | 10,007 | 1.359074 | 36.577% | 1.009832 |
| +45 min | 7,102 | 1.311420 | 24.170% | 1.104194 |
| +30 min | 7,045 | 1.250536 | 21.802% | 1.121306 |
| +15 min | 7,275 | 1.100990 | 17.451% | 1.152078 |

Worst spot+horizon pairs:

| Spot + lead | Rows | RMSE | Bias | SSE share |
| --- | ---: | ---: | ---: | ---: |
| La Tonnara +60 | 1,091 | 1.705399 | +0.017305 | 6.279% |
| La Tonnara +45 | 1,065 | 1.597801 | +0.053630 | 5.380% |
| Santa Manza +60 | 1,003 | 1.573746 | -0.155721 | 4.916% |
| Santa Manza +30 | 1,031 | 1.545343 | -0.178294 | 4.872% |
| Santa Manza +45 | 1,028 | 1.512747 | -0.092896 | 4.655% |

## Bias Structure

The model is not simply biased globally. Global bias is almost zero. The issue
is conditional bias:

| Actual wind bin | Rows | RMSE | Bias | SSE share |
| --- | ---: | ---: | ---: | ---: |
| 0-2 m/s | 9,343 | 1.036014 | +0.460936 | 19.844% |
| 8-10 m/s | 2,702 | 1.535373 | -0.427542 | 12.605% |
| 10+ m/s | 2,544 | 1.958638 | -0.777700 | 19.313% |

Interpretation:

- In very light wind, the model tends to invent wind.
- In strong wind, the model tends to damp the event.
- The mean bias hides this because the two errors compensate.

This is classic regression-to-the-mean behavior. A naive global calibration can
look clean on bias while still failing the regimes that matter for windsurf.

## Counterfactuals

The counterfactuals show where effort matters.

| Corrected subset, if perfect | Rows | Row share | SSE share | Global RMSE |
| --- | ---: | ---: | ---: | ---: |
| Actual wind >= 8 m/s | 5,258 | 16.730% | 31.990% | 1.045713 |
| Lead +60 only | 10,007 | 31.840% | 36.577% | 1.009832 |
| Lead +45/+60 | 17,109 | 54.437% | 60.747% | 0.794439 |
| Critical spots | 9,205 | 29.288% | 42.453% | 0.961916 |
| Critical spots or actual >= 8 m/s | 11,769 | 37.446% | 55.926% | 0.841819 |

This tells us two things:

1. Fixing hard spots alone nearly reaches the target, but not quite.
2. Fixing hard spots plus high-wind regimes would be enough in theory.

So the target is scientifically plausible, but only if we attack the correct
regimes. A generic average-score improvement will not be enough.

## Existing Model Oracle

I tested whether the models we already trained contain enough complementary
information for a router/selector to solve the problem.

Result: a target-leaky row-wise oracle over the full common candidate set gets:

- Oracle RMSE: 1.187225
- Champion RMSE: 1.268019

This is useful but not enough. Even with impossible knowledge of the target,
choosing the best existing model per row does not approach RMSE 0.9.

Conclusion:

- A deployable router is still worth building.
- But router work alone cannot close the gap with the current candidate models.
- We need better signals, not only better selection.

## Missing Signals

The current champion artifact is missing several physical features that are
central to the thermal/local-wind hypothesis.

Missing explicit concepts:

- Land-sea temperature delta.
- Air-sea temperature delta.
- Land-air temperature delta.
- Upwind station aggregates.
- Coastal-inland temperature delta.
- Coastal-relief temperature delta.
- Coastal-inland pressure delta.
- Coastal-relief pressure delta.
- Vertical temperature profile.
- Vertical humidity profile.
- Vertical motion profile.
- Geopotential thickness.
- Previous forecast runs.

Important nuance: some EUMETSAT and satellite columns exist, but actual value
coverage is low in this champion artifact. For example, land surface
temperature and instability have availability/schema presence, but weak mean
payload coverage. We should treat them as not yet operationally reliable until
the filled values are verified in the training matrix.

## Scientific Interpretation

The current system has learned the broad NWP correction. What it still lacks is
regime awareness.

Likely failure mechanisms:

1. Thermal onset and decay timing:
   The model sees recent observations and NWP, but does not yet see enough
   explicit land/sea heating contrast, vertical mixing, or coastal-relief
   pressure/temperature gradients.

2. High-wind amplitude compression:
   Strong events are underpredicted, especially above 8-10 m/s. This suggests
   the loss/objective and training distribution pull predictions toward the
   center.

3. Local spot physics:
   La Tonnara and Santa Manza dominate the SSE budget. Their errors are not
   just random noise; they look like spot-specific physical effects.

4. Horizon degradation:
   +45/+60 minutes are where the model must extrapolate regime evolution. That
   is exactly where missing physical drivers hurt most.

## Decision

We should stop treating RMSE 0.9 as a model-selection problem. The next phase
should be a data and regime-identification phase.

The target remains possible, but the current data/model stack does not contain
enough signal. The proof is the existing-model oracle: even an impossible
selector only reaches RMSE 1.187225.

## Recommended Next Experiments

1. Build a hard-regime dataset slice:
   Focus on La Tonnara, Santa Manza, lead +45/+60, actual/predicted high-wind,
   and thermal hours. Every new experiment should report both global score and
   score on this slice.

2. Add missing physical deltas:
   Create explicit features for land-sea, air-sea, land-air, coastal-inland,
   and coastal-relief pressure/temperature deltas. This is the most direct
   response to the thermal hypothesis.

3. Add vertical profile features:
   Add pressure-level temperature, humidity, wind shear, vertical velocity, and
   geopotential thickness at the 5 selected levels. These features should be
   summarized as gradients and stability indicators, not only raw levels.

4. Make upwind station features explicit:
   Build station context by wind direction and distance: nearest, coastal,
   inland, relief, and upwind aggregates. Include observation age and source
   reliability.

5. Train asymmetric/regime-specific calibrators:
   Do not repeat naive high-wind weighting globally. Instead, train targeted
   calibrators for high-wind underprediction and light-wind overprediction.

6. Build a deployable router after the new signals are present:
   Existing router/oracle gains show routeability, but insufficient signal.
   Routing should come after the regime features are actually populated.

7. Keep probabilistic targets:
   For windsurf decisions, continue producing P10/P50/P90 and threshold
   probabilities. RMSE alone over-penalizes rare extremes, while the business
   need is also "will a navigable window happen and when?".

## Bottom Line

The best current RMSE is 1.268019. To reach 0.9, we need to cut almost half of
the remaining MSE. That will not come from another generic benchmark pass.

The most rational path is:

AROME correction stays as the prior, then we add missing physical/regime
signals, then we specialize the correction for hard spots, high wind, and
+45/+60 minute evolution.
