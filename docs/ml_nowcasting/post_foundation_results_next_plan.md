# Post Foundation Results Next Plan

Date: 2026-06-30

## Champion Update

Update: 2026-06-30.

The operational gust champion is now:

```text
new_scale070_gust_recipe
```

Artifacts:

```text
/srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_prev_lowmem_gust_from_wind_champion_recipe_2024_2025_to_2026_v1
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1
```

Verified 2026 score from the aligned-scope decision note:

| Target | Rows | Base RMSE | Calibrated RMSE | MAE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| gust | 31,429 | 1.501342 | 1.484221 | 1.073906 | +0.056219 |

Follow-up completed: the foundation/blend gust experiments were rerun against
`new_scale070_gust_recipe`. The selected guarded blend is now:

```text
gust_guarded = new_scale070_gust_recipe + clip(0.10 * (TimesFM_mean - new_scale070_gust_recipe), -0.25, +0.25)
```

On the 2026 foundation same-key split:

| Target | Champion RMSE | Guarded RMSE | MAE | Bias | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| gust | 1.542160 | 1.525986 | 1.158341 | +0.170094 | 1.049% |

## Executive Position

The latest foundation-model pass changes the project state, but not in the
simple way we initially hoped.

Chronos2, TimesFM, and Moirai do not beat the local champions as standalone
forecasters. That confirms the earlier best-practice warning: generic
foundation time-series models do not know Corsican topography, local station
roles, AROME error structure, thermal timing, or spot exposure.

However, after building dense same-key champion predictions and validating on
the larger 2026 same-key split, we now have clean but modest improvements over
the champions:

- wind mean: guarded blend `champion + clipped 10% Chronos2 delta`, RMSE
  `1.271776` versus dense champion `1.285917`;
- gust: guarded blend on top of `new_scale070_gust_recipe`, RMSE `1.525986`
  versus dense new champion `1.542160`.

The gain is small, but important: foundation models have moved from
diagnostic-only to usable weak experts.

## How This Fits Earlier Conclusions

### The SAPHIR framing remains right

The best framing is still:

```text
NWP prior + recent observations + neighbor/context dynamics -> local correction
```

The champion remains the production floor. The new result does not say
"replace AROME correction with a foundation model". It says:

```text
champion correction + small foundation-model expert weight
```

This is aligned with SAPHIR, ORCA, and our own diagnosis.

### Dense same-sample comparison was the missing methodological step

The earlier router failures were partly methodological. Champion/foundation
overlap was sparse, so we were trying to learn routing decisions from too few
rows.

After rescoring the champion densely on the exact foundation keys, the
comparison became valid. That is why the result is now actionable.

Rule going forward:

```text
no model comparison without exact same spot_id + issue_time_utc + lead_time_minutes keys
```

### The scientific error diagnosis is still valid

The old diagnostic said the RMSE 0.9 target cannot be reached by generic model
selection alone:

- errors are tail-dominated;
- hard spots and +45/+60 min dominate SSE;
- high wind is compressed;
- missing physical/regime signals matter.

The new foundation oracle is much better than the champion on the small
same-key set, but the deployable learned gains are still below 2%. That
confirms the earlier conclusion: routing/blending helps, but it is not enough
to close the RMSE 0.9 gap by itself.

### Flexible routers remain dangerous

Wind mean flexible meta-stacks overfit or degrade. The stable wind improvement
comes from a very small static Chronos2 weight plus a delta cap.

For gusts, the rerun on `new_scale070_gust_recipe` keeps the same lesson:
foundation is useful only as a small capped correction. The new selected
formula uses `TimesFM mean`, alpha `0.10`, and a tighter `0.25 m/s` cap.

Current operational posture:

- wind: guarded conservative blend is the only promotion candidate;
- gust: `new_scale070_gust_recipe` is the champion; guarded TimesFM blend is
  a promotion candidate on top of it;
- row-wise oracle remains diagnostic only.

## What This Invalidates

1. Foundation models are not useful as standalone replacements.
2. Bigger flexible meta-stacks are not automatically better.
3. More feature/model complexity is not justified until it beats a conservative
   blend on same-key eval.
4. Small overlap router experiments should no longer drive decisions.
5. Global RMSE alone is not enough to promote anything.

## What This Proves

1. The champion can be rescored densely on arbitrary benchmark keys.
2. Foundation experts contain complementary signal.
3. The signal is weak but real on held-out 2026.
4. Conservative blending is currently more reliable than row-level routing for
   wind mean.
5. Gusts may be more routeable than wind mean, but need a larger validation.

## Next Workstream

### Phase 1: Validate Stability Before Promotion

Status: completed below on 2026-06-30.

Goal: decide whether the new blends are real or split noise.

Do next:

1. Build a larger 2026 foundation evaluation set.
   - Current eval: `1120` rows.
   - Target: at least `5600` rows, same as the enlarged 2025h2 calibration set.
   - Keep exact same-key comparison.

2. Recompute dense champions on the larger 2026 set.
   - wind champion scale070;
   - gust champion `new_scale070_gust_recipe`.

3. Rerun only conservative candidates first:
   - wind static champion + TimesFM blend;
   - wind static champion + Chronos2 blend;
   - gust static blend;
   - gust ExtraTrees stack.

4. Report by:
   - global RMSE/MAE/bias;
   - spot;
   - lead;
   - hard spots: La Tonnara, Santa Manza;
   - high wind bins;
   - thermal hours/day regimes.

Promotion gate:

```text
candidate must improve global RMSE and not degrade hard-regime RMSE materially
```

For wind, require at least `0.5%` stable gain before considering deployment.
For gust, require at least `1%` stable gain on larger eval before considering
the ExtraTrees stack.

### Phase 2: Turn Blends Into A Production-Safe Layer

Status: first implementation completed below on 2026-06-30.

If Phase 1 holds:

1. Implement a small final expert-blend layer:

```text
wind_mean_p50 = champion + clip(0.10 * (Chronos2_univar - champion), -0.5, +0.5)
gust_p50 = champion + clip(0.055725 * (TimesFM_p50 - champion), -0.5, +0.5)
```

2. Keep the champion prediction in outputs for audit:

```text
champion_wind_mean_ms
foundation_chronos2_wind_mean_ms
blended_wind_mean_ms
blend_delta_ms
```

3. Add guardrails:

- max absolute blend delta;
- missing foundation prediction -> fallback to champion;
- per-lead/per-spot metrics in monitoring.

4. For gusts, keep the guarded TimesFM blend behind an experiment flag until it
passes another time-forward split or a live shadow period.

### Phase 3: Attack The RMSE 0.9 Gap With Regime Signals

The foundation blend is not enough for RMSE 0.9. The old scientific diagnostic
still points to the main gap.

Priority features/regimes:

1. High-wind amplitude correction.
   - Target actual wind >= 8 m/s and >= 10 m/s.
   - Evaluate compression bias directly.

2. Hard spot specialists.
   - La Tonnara;
   - Santa Manza;
   - Balistra/Porticcio if they dominate current SSE on the new split.

3. Lead-specific correction.
   - +45/+60 min deserve separate specialists.
   - One global model is likely averaging incompatible horizon behavior.

4. Thermal regime indicators.
   - land-sea delta;
   - land-air delta;
   - air-sea delta;
   - coastal-inland/relief pressure and temperature gradients;
   - cloud/LST/SST availability checks.

5. Context station histories.
   - target station history;
   - coastal station;
   - mountain/relief station;
   - upwind station selected by wind direction;
   - freshness and missingness.

### Phase 4: Return To SAPHIR V2, But With A Narrower Question

The SAPHIR dictionary V2 proved the structure is feasible but did not beat the
champion. The next SAPHIR-like experiment should not try to win globally in one
shot.

Better question:

```text
Can structured neighbor histories improve the hard regimes where the champion
fails?
```

Use V2 for targeted experiments:

- train per-lead or per-regime models;
- evaluate only same-key against dense champion;
- test station-history encoders as extra experts, then blend with champion;
- do not promote a neural model unless it beats the conservative blend on the
  same split.

## Immediate Command-Level Plan

1. Build larger 2026 foundation sequence benchmark:

```text
foundation_sequence_champion_aligned_2026_windsurf_200cut_v1
```

2. Run:

```text
Chronos covariate
TimesFM
Chronos2 univariate
Moirai
```

3. Export matching training rows and dense champions:

```text
foundation_dense_champion_wind_2026_windsurf_200cut_scale070_source_v1
foundation_dense_champion_gust_2026_windsurf_200cut_new_scale070_gust_recipe_v1
```

4. Build:

```text
foundation_superbench_dense_champion_wind_gust_2026_windsurf_200cut_v1
```

5. Re-evaluate:

```text
static wind blend
static gust blend
gust ExtraTrees stack
regime/spot/lead breakdown
```

6. Decide:

- if wind blend holds: prepare production-safe blend layer;
- if gust stack holds: keep as candidate behind flag;
- if either fails: treat latest gain as split noise and continue with regime
  feature work.

## Current Decision

Do not chase larger generic meta-stacks right now.

The next best move is a stability audit on a larger same-key 2026 split. If the
small conservative gains survive, we promote a guarded blend layer. In parallel,
the path to RMSE 0.9 remains the regime/physics path: hard spots, high wind,
thermal timing, context station histories, and +45/+60 minute specialists.

## Stability Audit Result

Update: 2026-06-30, larger 2026 same-key split.

We built the larger 2026 same-key foundation evaluation:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_sequence_champion_aligned_2026_windsurf_200cut_v1
/srv/data/corsewind/ml_dataset/benchmarks/foundation_superbench_dense_champion_wind_gust_2026_windsurf_200cut_v1
/srv/data/corsewind/ml_dataset/benchmarks/foundation_stability_audit_2025h2_200cut_to_2026_200cut_v1
```

Rows: `3580`.

Note: the original stability audit used `old_signal_gust`; the gust rerun with
`new_scale070_gust_recipe` is recorded below and supersedes the old gust
decision.

Standalone foundation models remain worse than the champions:

| Target | Model | RMSE |
| --- | --- | ---: |
| wind mean | dense champion | 1.285917 |
| wind mean | Chronos2 univariate | 1.570379 |
| wind mean | TimesFM p50 | 1.600344 |
| gust | dense old_signal champion | 1.537145 |
| gust | Chronos2 univariate | 1.863339 |
| gust | TimesFM p50 | 1.899027 |

Oracles remain low:

| Target | Oracle RMSE |
| --- | ---: |
| wind mean | 0.730291 |
| gust | 0.919222 |

### Wind Stability

Best static wind blend learned from 2025h2 enlarged calibration:

```text
90% dense champion + 10% Chronos2 univariate
```

2026 enlarged eval:

| Model | RMSE | MAE | Bias | Gain |
| --- | ---: | ---: | ---: | ---: |
| dense champion | 1.285917 | 0.983218 | 0.127772 | baseline |
| static blend | 1.273055 | 0.969880 | 0.126103 | 1.000% |

By lead, the blend improves all horizons:

| Lead | Champion RMSE | Blend RMSE | Delta |
| ---: | ---: | ---: | ---: |
| +15 | 1.131468 | 1.109744 | -0.021724 |
| +30 | 1.219884 | 1.206499 | -0.013385 |
| +45 | 1.348769 | 1.334622 | -0.014147 |
| +60 | 1.423662 | 1.419258 | -0.004404 |

By spot, it improves most important spots, including La Tonnara and Santa
Manza, but not all spots:

| Spot | Delta RMSE |
| --- | ---: |
| La Tonnara | -0.025100 |
| Santa Manza | -0.004973 |
| Porto Polo | -0.051207 |
| Porticcio | -0.011597 |
| Figari Eole | -0.016592 |
| Balistra | +0.001990 |
| Piantarella | +0.026007 |

By wind bin, it improves most bins, but slightly worsens `10+ m/s`:

| Wind bin | Delta RMSE |
| --- | ---: |
| 0-2 | -0.044128 |
| 2-4 | -0.003798 |
| 4-6 | -0.015484 |
| 6-8 | -0.010081 |
| 8-10 | -0.010369 |
| 10+ | +0.009439 |

Interpretation: the wind blend is now a real promotion candidate, but it needs
a guardrail for high-wind regimes and possibly Piantarella.

### Gust Stability

Historical old-reference result. This subsection used the previous
`old_signal_gust` base.

Best static gust blend on `old_signal_gust`:

```text
gust_champion + 0.055725 * (TimesFM p50 - gust_champion)
```

2026 enlarged eval:

| Model | RMSE | MAE | Bias | Gain |
| --- | ---: | ---: | ---: | ---: |
| dense champion | 1.537145 | 1.167838 | 0.176615 | baseline |
| static blend | 1.526393 | 1.157125 | 0.168572 | 0.699% |
| residual ExtraTrees stack | 1.531066 | 1.162146 | 0.208857 | 0.395% |

The static gust blend is more stable than the ExtraTrees stack on the larger
split. It improves all leads:

| Lead | Champion RMSE | Static blend RMSE | Delta |
| ---: | ---: | ---: | ---: |
| +15 | 1.382434 | 1.369273 | -0.013161 |
| +30 | 1.490324 | 1.478545 | -0.011779 |
| +45 | 1.579440 | 1.567313 | -0.012127 |
| +60 | 1.680608 | 1.673927 | -0.006681 |

By spot, it improves almost all spots, but slightly worsens Piantarella:

| Spot | Delta RMSE |
| --- | ---: |
| La Tonnara | -0.007650 |
| Santa Manza | -0.004113 |
| Figari Eole | -0.013309 |
| Balistra | -0.005595 |
| Porto Polo | -0.038952 |
| Porticcio | -0.011903 |
| Piantarella | +0.005547 |

By gust bin, it worsens only the rare `16+ m/s` bin:

| Gust bin | Delta RMSE |
| --- | ---: |
| 0-4 | -0.023500 |
| 4-8 | -0.007940 |
| 8-12 | -0.004974 |
| 12-16 | -0.016340 |
| 16+ | +0.027315 |

Interpretation at the time: the gust static blend was a safer candidate than
the gust ExtraTrees stack. Current interpretation: superseded by the new
champion rerun below.

## Stability Decision

The larger split validates the conservative blend idea.

Promotion candidates:

- wind mean: static blend `90% champion + 10% Chronos2`;
- gust: `new_scale070_gust_recipe` plus guarded TimesFM mean blend from the
  rerun below.

Do not promote:

- flexible wind meta-stacks;
- gust ExtraTrees stack as default.

Required guardrails before production:

- fallback to champion when foundation prediction is missing;
- cap blend delta;
- do not add spot/threshold/lead routing unless it survives a future split;
- monitor by spot, especially Piantarella;
- monitor high gust `16+ m/s`;
- keep champion and blend outputs side-by-side for audit.

## Guardrail Audit Result

Update: 2026-06-30, production-safe blend layer.

We tested deployable guardrails on the same 2025h2 calibration -> 2026 eval
setup. The result is important: rules that look smarter on calibration do not
generalize cleanly enough.

Train-selected rules:

| Target | Train-selected rule | Eval RMSE | Outcome |
| --- | --- | ---: | --- |
| wind mean | cap `0.75 m/s`, only blend leads `<=45 min` | 1.273580 | worse than raw static blend |
| gust | predicted gust `>=16 m/s` fallback, disagreement `>=5 m/s` fallback, only blend leads `<=45 min` | 1.529493 | worse than raw static blend |

Decision: no hard spot, threshold, or lead routing for now. Keep the layer
boring and auditable:

```text
wind_guarded = champion + clip(0.10 * (Chronos2_univar - champion), -0.5, +0.5)
gust_guarded_old_signal_only = champion + clip(0.055725 * (TimesFM_p50 - champion), -0.5, +0.5)
missing foundation -> champion
```

2026 enlarged eval:

| Target | Champion RMSE | Raw static blend RMSE | Guarded RMSE | Guarded MAE | Bias | Gain vs champion | Capped deltas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wind mean | 1.285917 | 1.273055 | 1.271776 | 0.969362 | 0.127490 | 1.100% | 14 / 3580 |
| gust, old_signal base | 1.537145 | 1.526393 | 1.525892 | 1.156782 | 0.169009 | 0.732% | 8 / 3580 |

This is now implemented as:

```text
scripts/ml_dataset/apply_foundation_blend_guardrails.py
```

Validated output on z2:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2026_windsurf_200cut_v1/predictions.parquet
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2026_windsurf_200cut_v1/summary.json
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2026_windsurf_200cut_v1/summary.md
```

Operational interpretation:

- wind mean passes the previous `0.5%` promotion gate on this split;
- gust old-signal blend improved, but is superseded by the
  `new_scale070_gust_recipe` champion decision;
- the layer is not a path to RMSE `0.9` by itself;
- next serious RMSE work remains high-wind/thermal/context-station
  specialization.

## Gust Rerun With New Champion

Update: 2026-06-30.

We reran the gust foundation blend using `new_scale070_gust_recipe` as the
dense same-key champion.

Artifacts:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_dense_champion_gust_2025h2_windsurf_200cut_new_scale070_gust_recipe_v1/predictions.parquet
/srv/data/corsewind/ml_dataset/benchmarks/foundation_dense_champion_gust_2026_windsurf_200cut_new_scale070_gust_recipe_v1/predictions.parquet
/srv/data/corsewind/ml_dataset/benchmarks/foundation_superbench_dense_champion_wind_gust_2025h2_windsurf_200cut_new_gust_v1/foundation_superbench.parquet
/srv/data/corsewind/ml_dataset/benchmarks/foundation_superbench_dense_champion_wind_gust_2026_windsurf_200cut_new_gust_v1/foundation_superbench.parquet
/srv/data/corsewind/ml_dataset/benchmarks/foundation_stability_audit_2025h2_200cut_to_2026_200cut_new_gust_v1/stability_summary.md
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2026_windsurf_200cut_new_gust_v1/summary.md
```

Selected on 2025h2:

```text
gust_guarded = new_scale070_gust_recipe + clip(0.10 * (TimesFM_mean - new_scale070_gust_recipe), -0.25, +0.25)
```

2026 same-key result:

| Variant | RMSE | MAE | Bias | Gain vs champion |
| --- | ---: | ---: | ---: | ---: |
| `new_scale070_gust_recipe` | 1.542160 | 1.173474 | +0.180008 | baseline |
| guarded TimesFM mean blend | 1.525986 | 1.158341 | +0.170094 | 1.049% |
| old TimesFM p50 alpha `0.055725` | 1.530884 | 1.163242 | +0.171777 | 0.731% |

By lead, the guarded blend improves every horizon:

| Lead | Champion RMSE | Blend RMSE | Delta |
| ---: | ---: | ---: | ---: |
| +15 | 1.367945 | 1.345066 | -0.022879 |
| +30 | 1.491236 | 1.472932 | -0.018304 |
| +45 | 1.597832 | 1.580441 | -0.017391 |
| +60 | 1.692601 | 1.684639 | -0.007962 |

By spot:

| Spot | Champion RMSE | Blend RMSE | Delta |
| --- | ---: | ---: | ---: |
| Porto Polo | 1.395928 | 1.335394 | -0.060534 |
| Porticcio | 1.261779 | 1.240779 | -0.021000 |
| Figari Eole | 1.597219 | 1.580348 | -0.016871 |
| La Tonnara | 1.724115 | 1.709246 | -0.014869 |
| Balistra | 1.563211 | 1.556264 | -0.006947 |
| Santa Manza | 1.614298 | 1.614820 | +0.000522 |
| Piantarella | 1.582574 | 1.584119 | +0.001545 |

By actual gust bin, the blend improves all bins except rare `16+ m/s`:

| Gust bin | Champion RMSE | Blend RMSE | Delta |
| --- | ---: | ---: | ---: |
| 0-4 | 1.422485 | 1.388712 | -0.033773 |
| 4-8 | 1.433111 | 1.421417 | -0.011695 |
| 8-12 | 1.582696 | 1.576566 | -0.006131 |
| 12-16 | 1.914655 | 1.894988 | -0.019667 |
| 16+ | 2.276502 | 2.288498 | +0.011996 |

Decision: promote the guarded TimesFM mean blend as a gust promotion candidate
on top of `new_scale070_gust_recipe`, with high-gust monitoring. Do not add
spot-specific routing for Santa Manza/Piantarella yet; the degradation is tiny
and likely too small to justify another rule.

## Remaining Error Map After Guarded Blend

Diagnostic artifacts:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2026_windsurf_200cut_new_gust_v1/error_diagnostic.json
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2026_windsurf_200cut_new_gust_v1/error_diagnostic.md
```

The largest remaining wind-mean SSE cells are:

| Cell | Count | Guarded RMSE | Bias | SSE share |
| --- | ---: | ---: | ---: | ---: |
| La Tonnara +60 | 130 | 1.686118 | 0.012073 | 6.38% |
| Santa Manza +45 | 133 | 1.504348 | -0.082660 | 5.20% |
| Figari Eole +60 | 132 | 1.470447 | 0.148236 | 4.93% |
| Porto Polo +45 | 136 | 1.436643 | 0.310457 | 4.85% |
| La Tonnara +45 | 130 | 1.455914 | 0.048754 | 4.76% |

The largest remaining gust SSE cells are:

| Cell | Count | Guarded RMSE | Bias | SSE share |
| --- | ---: | ---: | ---: | ---: |
| La Tonnara +60 | 130 | 1.983245 | -0.181270 | 6.13% |
| Santa Manza +45 | 133 | 1.863265 | 0.055430 | 5.54% |
| Figari Eole +60 | 132 | 1.742038 | 0.256139 | 4.81% |
| Piantarella +60 | 120 | 1.820382 | 0.171920 | 4.77% |
| Figari Eole +30 | 132 | 1.681043 | 0.395249 | 4.47% |

Next attack should therefore be narrow:

- per-lead specialists for `+45/+60`;
- hard-spot specialists for La Tonnara, Santa Manza, Figari Eole;
- amplitude correction by actual-regime proxy, using deployable signals only
  such as champion/raw level, recent observation level, recent model error,
  direction, and context station gradients;
- do not spend more time on generic foundation routing until these cells move.

## Hard-Regime Specialist V1 Result

Update: 2026-06-30.

We tested the obvious next idea: train a small residual specialist on the
remaining hard wind cells, calibrated on 2025h2 and evaluated on 2026.

Inputs tested:

- superbench-only features: champion/foundation/raw predictions, spot, lead,
  time;
- full feature merge: guarded predictions plus the full
  `training_rows_for_sequence_keys.parquet` feature table.

Merged artifacts:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2025h2_windsurf_200cut_v1/predictions_with_full_features.parquet
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2026_windsurf_200cut_v1/predictions_with_full_features.parquet
```

Leaderboard:

| Run | Features | Hard rule | Selected scale | Eval RMSE | Gain |
| --- | ---: | --- | ---: | ---: | ---: |
| HGB full features all spots +45/+60 | 1259 | lead >= 45 | 0.25 | 1.271477 | 0.024% |
| LGBM full features hard spots +45/+60 | 1074 | La Tonnara/Santa Manza/Figari and lead >= 45 | 0.05 | 1.271615 | 0.013% |
| HGB full features hard spots +45/+60 | 1074 | La Tonnara/Santa Manza/Figari and lead >= 45 | 0.00 | 1.271776 | 0.000% |
| superbench-only specialists | 8 | La Tonnara/Santa Manza/Figari and lead >= 45 | 0.00 | 1.271776 | 0.000% |

Decision: do not promote hard-regime specialist v1. The effect is too small
and the validation often selects `scale=0`, which means the residual correction
is not stable enough.

Implication: the next meaningful RMSE step probably requires different target
structure or better data, not another small residual calibrator on the same
features. Candidate directions:

- model direct wind/gust quantiles instead of only residual mean;
- train separate lead models from the base training table, not a second-stage
  correction on already corrected predictions;
- improve recent observation histories and context-station/upwind histories;
- add explicit thermal-timing labels/features and evaluate only thermal
  startup/drop cells.
