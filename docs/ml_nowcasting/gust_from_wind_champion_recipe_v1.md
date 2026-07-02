# Gust Candidate From Wind-Champion Recipe v1

Generated: 2026-06-30

## Purpose

Apply the current wind-mean champion recipe to gust prediction:

- same two-stage structure;
- LightGBM residual base model;
- ExtraTrees second-stage residual calibrator;
- fixed `scale070` second-stage correction;
- same 2025h2 calibration window;
- same locked 2026 evaluation window;
- same short leads `+15/+30/+45/+60`.

This run intentionally tests the wind champion recipe for gusts rather than
reusing older gust experiments.

## Run IDs

Base 2026 model:

```text
tabular_lgbm_225k_prev_lowmem_gust_from_wind_champion_recipe_2024_2025_to_2026_v1
```

Calibration base:

```text
tabular_lgbm_calbase_gust_from_wind_champion_recipe_2024_to_2025h2_v1
```

Second-stage gust calibrator:

```text
prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1
```

Remote artifact root:

```text
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1
```

## Recipe

First-stage base model:

```text
target: labels__residual_gust_ms
model family: LightGBM
train split: 2024-01 through 2025-12 -> 2026 evaluation
max train rows: 225000
max test rows: 60000
max_iter: 180
learning_rate: 0.04
max_leaf_nodes: 31
min_samples_leaf: 20
lightgbm_max_bin: 127
feature_fraction: 0.85
bagging_fraction: 0.85
bagging_freq: 1
```

Second-stage calibrator:

```text
target: gust
calibration window: 2025-07-01T00:00:00Z -> 2026-01-01T00:00:00Z
evaluation window: 2026-01-01T00:00:00Z -> 2026-07-01T00:00:00Z
leads: 15, 30, 45, 60 minutes
model family: ExtraTrees
n_estimators: 300
min_samples_leaf: 80
n_jobs: 2
clip_correction_ms: 2.0
correction_scale: 0.70
feature columns: 302
```

## Results

Locked 2026 evaluation:

```text
rows: 31,429
base gust RMSE: 1.501342
base gust MAE: 1.096947
base gust bias: +0.132077

calibrated gust RMSE: 1.484221
calibrated gust MAE: 1.073906
calibrated gust bias: +0.056219

RMSE gain vs base: 1.14%
```

By lead:

| Lead | Rows | Base RMSE | Calibrated RMSE | Calibrated MAE | Bias |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +15 | 7,275 | 1.342993 | 1.321811 | 0.964246 | +0.045261 |
| +30 | 7,045 | 1.509094 | 1.490154 | 1.082372 | +0.041879 |
| +45 | 7,102 | 1.594504 | 1.582131 | 1.149908 | +0.085887 |
| +60 | 10,007 | 1.536340 | 1.519450 | 1.093729 | +0.053227 |

Worst spots:

| Spot | Rows | Calibrated RMSE | MAE | Bias |
| --- | ---: | ---: | ---: | ---: |
| `santa_manza` | 4,096 | 1.836823 | 1.343765 | -0.024957 |
| `la_tonnara` | 4,287 | 1.802347 | 1.329784 | -0.009521 |
| `piantarella` | 4,002 | 1.538033 | 1.114299 | +0.077901 |
| `figari_eole` | 1,170 | 1.522576 | 1.164687 | +0.162427 |
| `cap_corse` | 407 | 1.486215 | 1.105751 | -0.132792 |

## Interpretation

The wind champion recipe transfers cleanly to gusts and improves the new gust
base model:

```text
1.501342 -> 1.484221 RMSE
```

The gain is real but modest. The second stage mostly reduces positive bias and
slightly improves every evaluated lead.

This is a valid gust candidate from the wind-champion recipe. After the
aligned-scope follow-up below, it should be treated as the current operational
gust champion candidate by RMSE, with one caveat: old historical prediction
files are not fully comparable because they emit a different key set.

## Strict Common-Key Comparison

Comparison artifact:

```text
/srv/data/corsewind/ml_dataset/benchmarks/gust_from_wind_champion_recipe_comparison_v1/comparison.md
```

Common-key definition:

```text
spot_id + issue_time_utc + lead_time_minutes
```

All-way overlap across the new recipe candidate and four prior gust candidates:

```text
common rows: 16,570
```

| Candidate | RMSE | MAE | Bias | P90 abs error |
| --- | ---: | ---: | ---: | ---: |
| `old_signal_gust` | 1.512454 | 1.122432 | +0.140692 | 2.368504 |
| `v4_directional_pruned_gust` | 1.516386 | 1.129759 | +0.144778 | 2.370433 |
| `new_scale070_gust_recipe` | 1.521890 | 1.119652 | +0.115436 | 2.380732 |
| `v3_pruned_gust` | 1.523131 | 1.133691 | +0.149919 | 2.385190 |
| `v3_dem_fetch_gust` | 1.524443 | 1.133979 | +0.155309 | 2.385234 |

Pairwise common-row verdict against the new candidate:

| Prior candidate | Common rows | New RMSE | Prior RMSE | Verdict |
| --- | ---: | ---: | ---: | --- |
| `old_signal_gust` | 16,570 | 1.521890 | 1.512454 | prior wins RMSE |
| `v4_directional_pruned_gust` | 16,570 | 1.521890 | 1.516386 | prior wins RMSE |
| `v3_pruned_gust` | 16,570 | 1.521890 | 1.523131 | new wins narrowly |
| `v3_dem_fetch_gust` | 16,570 | 1.521890 | 1.524443 | new wins narrowly |

The new recipe candidate has the best global RMSE among the checked
`calibration_results.json` files. This strict historical overlap shows a
caveat: on the 16,570 rows emitted by all compared historical prediction files,
it does not beat `old_signal_gust` or `v4_directional_pruned_gust` on RMSE.

The useful signal is different: `new_scale070_gust_recipe` has lower MAE and
lower positive bias than the RMSE-leading candidates. It should be kept as a
candidate for blending, routing, or bias-sensitive product metrics.

## Aligned-Scope Follow-Up

Because the historical candidates do not emit the same rows, a second report
was generated on the exact key scope of the new recipe candidate.

Artifact:

```text
/srv/data/corsewind/ml_dataset/benchmarks/gust_from_wind_champion_recipe_aligned_scope_v1/comparison.md
```

Scope:

```text
new_scale070_gust_recipe keys: 31,429 rows
```

Historical global comparison:

| Model | Rows | Base RMSE | Calibrated RMSE | MAE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| `new_scale070_gust_recipe` | 31,429 | 1.501342 | 1.484221 | 1.073906 | +0.056219 |
| `old_signal_gust` | 33,214 | 1.524039 | 1.516817 | 1.124876 | +0.137016 |
| `v4_directional_pruned_gust` | 33,214 | 1.531345 | 1.521491 | 1.131693 | +0.140771 |
| `v3_pruned_gust` | 33,214 | 1.535513 | 1.526835 | 1.134749 | +0.145641 |
| `v3_dem_fetch_gust` | 33,214 | 1.534463 | 1.527744 | 1.135021 | +0.151301 |

Historical predictions on the new-scope overlap:

| Prior model | Covered/new rows | Coverage | New RMSE | Prior RMSE | New MAE | Prior MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `old_signal_gust` | 16,570 / 31,429 | 52.722% | 1.521890 | 1.512454 | 1.119652 | 1.122432 |
| `v4_directional_pruned_gust` | 16,570 / 31,429 | 52.722% | 1.521890 | 1.516386 | 1.119652 | 1.129759 |
| `v3_pruned_gust` | 16,570 / 31,429 | 52.722% | 1.521890 | 1.523131 | 1.119652 | 1.133691 |
| `v3_dem_fetch_gust` | 16,570 / 31,429 | 52.722% | 1.521890 | 1.524443 | 1.119652 | 1.133979 |

Saved calibrators applied on the exact new scope:

| Calibrator | Rows | Missing expected features | Scale | Base RMSE | Applied RMSE | MAE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `new_scale070_gust_recipe` | 31,429 | 0 / 302 | 0.7 | 1.501342 | 1.484221 | 1.073906 | +0.056219 |
| `v3_pruned_gust` | 31,429 | 227 / 526 | 0.5 | 1.501342 | 1.493361 | 1.087938 | +0.100699 |
| `v3_dem_fetch_gust` | 31,429 | 227 / 526 | 0.5 | 1.501342 | 1.493533 | 1.087532 | +0.099898 |
| `v4_directional_pruned_gust` | 31,429 | 229 / 528 | 0.5 | 1.501342 | 1.493622 | 1.088160 | +0.101558 |
| `old_signal_gust` | 31,429 | 227 / 526 | 0.5 | 1.501342 | 1.494995 | 1.089362 | +0.110274 |

This compatibility test is not a full replay of the old first-stage recipes:
it applies their saved second-stage calibrators to the new first-stage base
predictions. It is still the fairest available same-scope test without
rerunning the old first-stage pipelines to emit the exact new key set.

Decision: promote `new_scale070_gust_recipe` as the current operational gust
champion candidate by RMSE, while preserving the historical-overlap caveat.

## Artifacts

```text
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1/calibrated_predictions_2026.parquet
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1/calibrator.joblib
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1/calibration_results.json
/srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1/calibration_results.md
```
