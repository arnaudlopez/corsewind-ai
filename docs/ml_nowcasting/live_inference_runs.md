# Live Inference Runs

## 2026-06-29 - AROME-PI 12Z Bridge

- Run id: `live_aromepi_20260629T12_v1`
- Dataset root: `/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260629T12_v1`
- Source NWP: Meteo-France AROME-PI run `2026-06-29T12:00:00Z`
- Issue time used: `2026-06-29T13:30:00Z`
- Forecast targets: `2026-06-29T13:45:00Z` to `2026-06-29T18:00:00Z`, every 15 minutes
- Spots: 20 ML spots
- Rows predicted: 360
- Output parquet: `/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260629T12_v1/predictions/predictions.parquet`
- Output JSON: `/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260629T12_v1/predictions/predictions_by_spot.json`

What was implemented:

- Added inference-grid target generation to `build_spot_feature_store.py`.
- Added AROME-PI native wind fields to `collect_meteo_france_nwp_spot_features.py`.
- Added AROME-PI fallback mapping into residual baselines in `build_residual_training_table.py`.
- Added `create_meteo_france_forecast_grid_layer.py`.
- Added `run_live_wind_mean_inference.py`.

Result:

- AROME-PI wind WCS fields were available and sampled successfully:
  - `wind_speed_10m_ms`
  - `wind_u_10m_ms`
  - `wind_v_10m_ms`
- Feature store built successfully: 380 rows.
- Model rows built successfully: 360 rows, with no missing wind baseline.
- Inference completed successfully with base tabular model plus second-stage calibrator.
- Calibration scale used: `0.70`.

Important caveats:

- This is a bridge inference path. The champion model was trained mostly with Open-Meteo AROME-style baselines, while this live run uses native Meteo-France AROME-PI fields mapped into the same baseline schema.
- Several target spots had no fresh direct spot observation at issue time inside the 180-minute window, so this run is not yet a full live observation-corrected nowcast for every spot.
- Leads beyond `+60 min` are useful operationally, but the strongest validation of the current champion remains on `+15/+30/+45/+60 min`.

Next validation:

- Compare the `13:45-18:00 UTC` predictions against observations as they arrive.
- Track raw AROME-PI vs corrected vs calibrated error by spot and lead.
- Decide whether to keep the AROME-PI bridge as-is, train a native AROME-PI calibrator, or blend AROME-PI raw with the Open-Meteo-trained correction more conservatively.

## 2026-06-30 - Champion + Foundation Shadow Rail

The live wind/gust inference script now emits both the production champion
prediction and a guarded foundation-shadow prediction.

Code:

```text
scripts/ml_dataset/build_live_foundation_sequence_inputs.py
scripts/ml_dataset/run_live_foundation_shadow_pipeline.py
scripts/ml_dataset/run_live_wind_and_gust_inference.py
scripts/ml_dataset/evaluate_shadow_foundation_blends.py
```

Production champion defaults:

- wind mean: `prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1`;
- gust: `new_scale070_gust_recipe`, implemented through
  `prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_gust_from_wind_champion_recipe_v1`.

Shadow foundation defaults:

- wind mean: `champion_wind_mean_ms + clip(0.10 * (chronos2_univar_wind_mean_ms_mean - champion_wind_mean_ms), -0.50, +0.50)`;
- gust: `champion_gust_ms + clip(0.10 * (timesfm_gust_ms_mean - champion_gust_ms), -0.25, +0.25)`.

Output columns added:

```text
champion_wind_mean_ms
guarded_foundation_wind_mean_ms
guarded_foundation_wind_mean_delta_ms
guarded_foundation_wind_mean_used_foundation
guarded_foundation_wind_mean_delta_was_capped
champion_gust_ms
guarded_foundation_gust_ms
guarded_foundation_gust_delta_ms
guarded_foundation_gust_used_foundation
guarded_foundation_gust_delta_was_capped
```

Historical shadow evaluation on the 2026 common-key foundation benchmark:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_guarded_blend_2026_windsurf_200cut_new_gust_v1/shadow_evaluation.md
```

Result:

| Target | Champion RMSE | Guarded RMSE | Gain |
| --- | ---: | ---: | ---: |
| wind mean | 1.285917 | 1.271776 | 1.099682% |
| gust | 1.542160 | 1.525986 | 1.048789% |

The guarded blend improves every `+15/+30/+45/+60 min` lead on this benchmark.
It slightly degrades high gust events `>=16 m/s`, with gust RMSE moving from
`2.276502` to `2.288498`; therefore it remains a shadow/monitoring candidate,
not the operational champion.

Smoke run:

```text
/srv/data/corsewind/ml_dataset/benchmarks/live_shadow_smoke_2026_windsurf_200cut_v1/predictions.parquet
/srv/data/corsewind/ml_dataset/benchmarks/live_shadow_smoke_2026_windsurf_200cut_v1/predictions_by_spot.json
```

The smoke input did not contain the foundation expert columns. The live script
correctly fell back to `guarded_foundation_* = champion_*` and recorded:

```text
fallback_reason: missing_foundation_expert_column
```

Next operational step: wire generation of the live foundation expert columns
before inference if we want the shadow deltas to be active on real daily runs.

Implementation update:

- `build_live_foundation_sequence_inputs.py` exports a live saved-sequence root
  with `predictions.parquet`, `past_context.parquet`, and coverage diagnostics.
- `run_live_foundation_shadow_pipeline.py` orchestrates sequence export,
  optional Chronos-2, optional TimesFM, merge of foundation expert columns, and
  final champion/shadow inference.

Operational guardrail:

- do not run Chronos/TimesFM when `past_context_coverage.items_with_observed_context`
  is `0`; this means the available history source is stale or does not cover
  the live issue time;
- for the benchmark score `wind mean champion RMSE = 1.285917`, the input rows
  are the `scale070_source` rows:

```text
/srv/data/corsewind/ml_dataset/benchmarks/foundation_sequence_champion_aligned_2026_windsurf_200cut_v1/training_rows_for_sequence_keys_scale070_source.parquet
```

Using the non-`scale070_source` rows changes the wind champion input baseline and
does not reproduce the reference score, even though the shadow mechanics still
work.

## 2026-06-30 - AROME-PI 17Z Fresh Live Run

- Run id: `live_aromepi_20260630T17_shadow_v2_offset_fallback`
- Dataset root: `/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260630T17_shadow_v2_offset_fallback`
- Source NWP: Meteo-France AROME-PI run `2026-06-30T17:00:00Z`
- Issue time used: `2026-06-30T17:45:00Z`
- Forecast targets: `2026-06-30T18:00:00Z` to `2026-06-30T23:00:00Z`, every 15 minutes
- Spots: 20 ML spots
- Rows predicted: 420
- Output parquet: `/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260630T17_shadow_v2_offset_fallback/shadow_foundation_fresh/predictions/predictions.parquet`
- Output JSON: `/srv/data/corsewind/ml_dataset/live_inference/live_aromepi_20260630T17_shadow_v2_offset_fallback/shadow_foundation_fresh/predictions/predictions_by_spot.json`

What changed:

- `build_live_foundation_sequence_inputs.py` can now ingest fresh observation
  JSONL files in addition to historical parquet rows.
- `run_live_foundation_shadow_pipeline.py` passes those observation JSONL files
  through and stops before Chronos/TimesFM if no observed live context exists.
- `build_residual_training_table.py` falls back to the mean of available
  `nwp_offset_{e10,n10,s10,w10}` wind/gust fields when the central NWP wind
  baseline is missing.

Result:

- Final prediction table has 420 rows, 20 spots, and no missing raw/champion/
  guarded wind or gust values.
- Foundation expert columns are present on 80 rows: 20 spots x the first 4
  leads (`+15/+30/+45/+60 min`).
- Fresh observed context came from Meteo-France observation JSONL for
  `2026-06-30`; coverage was 7 items/spots with observed wind and gust context.
- Beacon Live/WindsUp observations in the current dataset were only fresh up to
  `2026-06-29T12:05:07Z`, so they did not feed this live issue time.

Important caveats:

- This is an inference run, not a scored benchmark, because future observations
  for `18:00-23:00 UTC` were not yet available at issue time.
- The spatial-offset fallback is operationally useful for missing central NWP
  points, but should be evaluated against observations before promoting it as a
  permanent training-time rule.
