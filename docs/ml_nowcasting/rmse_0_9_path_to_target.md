# RMSE 0.9 Path To Target

Generated: 2026-06-27

## Current Status

The current best validated score is:

- run: `prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1`
- period: locked 2026 evaluation
- rows: `31,429`
- wind mean RMSE: `1.268019`
- wind mean MAE: `0.930465`
- gap to RMSE 0.9: `0.368019`

The recent leakage-safe ensemble between the current best and the top700
post-relief model improves only on their common subset:

- rows: `18,162`
- old-best overlap RMSE: `1.306305`
- top700 overlap RMSE: `1.307041`
- ensemble overlap RMSE: `1.291417`
- gain vs overlap base: `1.14%`

This confirms complementarity, but it does not change the order of magnitude.

## What A RMSE 0.9 Score Requires

For the current best model, reaching RMSE `0.9` still requires removing about
half of the total squared error:

- MSE reduction needed: `49.623%`
- rows needing perfect correction: `2,342` (`7.452%`)
- top 1% worst rows carry `16.983%` of SSE
- top 5% worst rows carry `40.993%` of SSE
- top 10% worst rows carry `56.654%` of SSE

So the target is not a small calibration problem. It requires a large reduction
on the tail cases.

## Where The Error Lives

The current best error is concentrated in:

- `lead_time_minutes` 45 and 60: `60.747%` of SSE
- high observed wind `actual >= 8 m/s`: `31.917%` of SSE
- critical exposed spots: `52.658%` of SSE
- worst spot-leads: `la_tonnara +60`, `la_tonnara +45`,
  `santa_manza +60`, `santa_manza +30/+45`, then `balistra` and exposed
  airport/coastal spots

Counterfactuals show that perfect correction of the hard zones would cross the
threshold, but realistic required subgroup RMSEs are extremely low:

- correcting only `lead 45/60` would require that subgroup to fall from about
  `1.339495` RMSE to `0.573208` RMSE
- correcting only critical spots would require them to fall from about
  `1.432584` RMSE to `0.343938` RMSE
- correcting only high wind `actual >= 8 m/s` cannot reach global `0.9` alone;
  the rest of the dataset is already too high
- correcting `critical spots OR lead 45/60` would require that large subset to
  fall from about `1.334694` RMSE to `0.837812` RMSE

## What Current Models Prove

The diagnostic row-wise oracle chooses the best prediction per row using the
observed target, so it is not deployable. It is useful as an upper-bound probe.

On the current best audit with existing raw/calibrated variants:

- best deployable model: `1.268019`
- row-wise existing-model oracle among raw NWP, scale070, and autoscale:
  `1.051416`

On the recent old-best/top700/ensemble overlap:

- best deployable overlap ensemble: `1.291417`
- row-wise oracle among old-best/top700/ensemble: `1.178522`

Even cheating by choosing the best existing raw/calibrated prediction per row
does not reach `0.9`. That strongly suggests the current family of tabular
residual models is missing signal, not just needing another shallow blend.

## Decision

Do not spend the next phase on more shallow global blends or more granular
spot/lead weights. We already see:

- global blending helps only slightly
- spot/lead weighting overfits validation
- per-spot/per-lead grouped models degrade
- high-wind weighting degrades
- top-N feature pruning loses signal

The route to RMSE `0.9` is now:

1. Improve target and context data depth for the hard zones.
   We need longer, denser, cleaner history on the exact coastal spots,
   especially `la_tonnara`, `santa_manza`, `balistra`, `porto_polo`,
   `cap_corse`, `la_parata`, plus nearby/upwind mountain and coastal stations.

2. Build a hard-regime specialist, not a shallow ensemble.
   The specialist should target `critical spots OR lead 45/60` and thermal
   onset/high-wind regimes, while being selected by a conservative validation
   gate. It must use only forecast-time features.

   First third-stage tests using only the current calibrated-prediction features
   and Q4 2025 as honest calibration did not generalize:

  - HGB broad hard mask: `1.269403 -> 1.270628`
  - LightGBM broad hard mask: `1.269403 -> 1.274176`
  - ExtraTrees broad hard mask: `1.269403 -> 1.276871`
  - HGB lead>=45: `1.269403 -> 1.270308`
   - HGB critical spots AND lead>=45: validation selected scale `0.0`

   This means the specialist direction remains conceptually right, but the
   current feature/history stack is not enough. The specialist needs longer
   hard-regime history and new physical/regime inputs, not just another model on
   the same Q4 feature table.

3. Add regime features that are not just reweighted copies of the current
   features:
   - thermal onset indicators
   - land-sea temperature gradient history
   - pressure-gradient vertical-column summaries
   - station freshness and station disagreement
   - upwind/coastal/mountain propagation deltas
   - model-error recent trend features

4. Validate with a locked protocol:
   - choose all routing/weights/specialist thresholds on 2025 only
   - evaluate once on 2026
   - report full 2026 RMSE and hard-zone RMSE separately
   - reject any model whose gain is only visible on a narrowed overlap

## Feature Gap Update

A dedicated feature coverage audit is available here:

`docs/ml_nowcasting/rmse_0_9_feature_gap_audit.md`

Key result:

- current best is missing the later thermal deltas, upwind aggregates, and all
  vertical profile features
- top700/post-relief contains the newer thermal/upwind features, but still lacks
  all vertical profile features and does not beat the locked best
- AROME vertical profile collection exists and is wired into the feature store,
  but only `725` rows are currently available, from `2026-06-24` to
  `2026-06-26`
- EUMETSAT LST/cloud/instability samples are also recent-only at the moment,
  starting around `2026-06-22`

So the P0 data question is now very concrete: can we backfill historical AROME
isobaric vertical profiles, or do we need another historical vertical-atmosphere
source? Without that, the current 2025 -> 2026 RMSE `0.9` target is not credible
from the available feature stack alone.

## Practical Acceptance Gate

The next model family is only worth keeping if it demonstrates at least one of:

- full locked 2026 RMSE below `1.20`
- hard subset `critical spots OR lead 45/60` RMSE below `1.10`
- `lead 45/60` RMSE below `1.15`
- high-wind `actual >= 8 m/s` diagnostic RMSE below `1.35` without worsening
  low-wind bias materially

Without one of those intermediate breakthroughs, RMSE `0.9` is not credible
with the current data/model stack.
