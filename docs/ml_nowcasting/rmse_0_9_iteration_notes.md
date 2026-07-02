# RMSE 0.9 Iteration Notes

Objective: drive wind mean RMSE below `0.9` without data leakage.

## Working Log Contract

This file is the persistent work log for the RMSE09 effort. Every meaningful
experiment must leave a trace here before the thread is considered in a clean
state.

For each iteration, record:

- date/time and machine/path;
- dataset scope and split;
- model or feature change tested;
- command or watcher path when relevant;
- measured result, not only the intuition;
- decision: keep, retry, discard, or wait for more data;
- next action.

Rule: a failed test is still useful evidence. Do not delete it from the story;
summarize why it failed and what it ruled out.

Current live state to re-check after any interruption:

- z2 rebuild `regime_v1` is the current completed dense data-generation path;
- latest confirmed completed shard is `2026-06`;
- latest confirmed full audit is `2024-01..2026-06`, verdict `pass`;
- full rebuilt dataset has `1,528,776` rows across 30 Parquet shards;
- global post-rebuild LightGBM 150k was stopped by the memory guard before z2
  OOM;
- manual global LightGBM 100k completed and was audited:
  `corrected short RMSE = 1.340857`, verdict `not_achieved`;
- best historical short-horizon LightGBM RMSE remains `1.278997`, so the global
  100k rebuild is worse than the previous best;
- grouped-by-lead LightGBM 100k is the current live probe, using
  `lead_time_minutes` groups;
- grouped-by-spot+lead and selector/assertion watchers are still waiting for
  grouped audits;
- the goal is not achieved until `assert_tabular_rmse09_goal.py` passes.

Historical reference benchmark, before the RMSE09 dense rebuild:

```text
/srv/data/corsewind/ml_dataset/benchmarks/sequence_2026_windsurf_1h_v2
```

Scope:

- 2026 issue times;
- windsurf hours 08..17 UTC;
- 7 spots;
- horizons +15, +30, +45, +60 minutes;
- 308 common rows with raw NWP, HGB, Chronos, TimesFM, and Moirai.

The active RMSE09 experiment now builds dedicated dense train/eval roots:

```text
/srv/data/corsewind/ml_dataset/benchmarks/sequence_2025_windsurf_1h_rmse09_v1
/srv/data/corsewind/ml_dataset/benchmarks/sequence_2026_windsurf_1h_rmse09_v1
```

Both years are sampled with the same `--max-cutoffs-per-spot` setting, 240 by
default, so the holdout is no longer silently inherited from the short v2 smoke
benchmark.

## Current Best

Best validated model on the common rows:

| Model | Wind RMSE | Wind MAE |
| --- | ---: | ---: |
| Chronos-2 univariate cross-learning | 1.269650 | 0.943105 |
| HGB residual global sample | 1.276752 | 1.015342 |
| TimesFM | 1.310452 | 0.968488 |
| Raw NWP | 1.900426 | 1.453470 |

Simple convex averaging of HGB + Chronos + TimesFM improves only to about
`1.1787` RMSE. This is useful but not near `0.9`.

## What Did Not Work

- In-sample calibration can go below `0.9`, but temporal and leave-one-spot
  checks degrade above `1.2`. This is overfit and should not be counted.
- Naive Chronos residual forecasting, where Chronos forecasts
  `observation - NWP` and the result is added to future NWP, degraded the
  benchmark.
- Spot-specific HGB residual models trained on 2024-2025 did not improve in the
  first pass: RMSE was about `1.31`.
- Larger sklearn HGB training is memory-heavy on z2. A 1M-row run was killed
  with code `137`; a 500k-row run reached roughly 15 GB RSS / 95% memory and was
  stopped before completion.

## Error Structure

The problem is concentrated by spot:

| Spot | Chronos RMSE | HGB RMSE |
| --- | ---: | ---: |
| porticcio | 0.924607 | 0.964534 |
| figari_eole | 1.012742 | 1.302135 |
| porto_polo | 1.063968 | 1.458802 |
| balistra | 1.226294 | 1.076632 |
| piantarella | 1.339856 | 1.365746 |
| la_tonnara | 1.485164 | 1.441239 |
| santa_manza | 1.663366 | 1.247046 |

Even an in-sample oracle choosing the best existing model by spot+lead only gets
to about `1.07` RMSE. That means the existing model outputs do not contain
enough independent signal to reach `0.9` by routing alone.

## Scientific Alignment

The path remains consistent with SAPHIR/Baggio-style hybrid nowcasting:

- keep NWP as prior;
- use recent observations and neighboring/upwind stations;
- fine-tune/calibrate locally per spot/station;
- evaluate by horizon because +15/+30/+45/+60 behave differently.

The first naive residual Chronos attempt failed, but that does not invalidate
residual correction. It suggests that residual correction needs supervised local
features, not zero-shot residual extrapolation alone.

## Paper Alignment Checklist

Reference: <https://arxiv.org/html/2503.18797v2>

The SAPHIR-style model uses richer inputs than the current sequence benchmark:

- station observations: target station plus 10 neighboring stations;
- recent station history: current value plus 6 past values;
- observed variables: wind vector components and 2 m temperature;
- AROME fields: 10 m wind components, 2 m temperature, relative humidity, and
  mean-sea-level pressure on a local spatial subgrid;
- ARPEGE fields: wind, temperature, and pressure information over vertical
  levels on a broader grid;
- temporal/static features: hour/day cyclical encodings, station location, and
  relative neighbor positions;
- leakage control: data split at day level so adjacent samples do not cross
  train/validation/test boundaries;
- local adaptation: global model followed by station-specific fine-tuning.

Current CorseWind benchmark already covers the NWP-prior and sequence-model
side, but it is still thinner than the paper on three points:

1. no true neighboring/upwind station tensor yet;
2. no spatial AROME/ARPEGE subgrid tensor yet, only sampled point/context
   features;
3. no station-specific fine-tuning layer validated on a larger temporal
   holdout.

If the 2025->2026 calibrator sweep remains above `0.9`, these are the next
scientifically justified additions before trying larger foundation models.

## Next Valid Paths

1. Build a larger calibration dataset of Chronos/TimesFM predictions over 2025,
   then train the ensemble/calibrator on 2025 and test on 2026. The current
   benchmark has too few rows to train a reliable meta-model.
2. Replace sklearn HGB scaling experiments with a memory-efficient learner:
   LightGBM, XGBoost histogram, or CatBoost, ideally with native categorical
   handling and early stopping.
3. Add source-quality and anomaly features:
   observation age, source resolution, sudden drop/ramp flags, target station
   continuity, and station-specific reliability.
4. Improve local context for the hard spots, especially Santa Manza and
   La Tonnara:
   nearest coastal station, mountain/relief station, upwind station, and
   pressure/temperature gradients.
5. Rebuild a true sequential training table from 6-minute observations where
   available. The current 15-minute grid may be too coarse to catch the sharp
   changes that dominate RMSE.

## Current Conclusion

`RMSE < 0.9` is not reachable honestly with the current v2 benchmark using only
simple ensembling of existing outputs. The most credible route is:

```text
more historical sequence predictions
+ memory-efficient supervised calibrator
+ better spot/upwind/context quality features
+ 6-minute recent observations
```

Until then, the honest target for the current artifacts is closer to `1.15-1.20`
with simple ensembling, and `1.0-1.1` if a well-validated local calibrator can be
trained on a much larger historical prediction set.

## Next Executable Experiment

The next experiment should be temporal, not random:

```text
train/calibrate on 2025 sequence benchmark
evaluate on untouched 2026 sequence benchmark
```

This is stricter than in-sample ensembling and prevents the score from being
inflated by learning the 2026 target values.

The local one-command launcher is:

```bash
python3 scripts/ml_dataset/launch_z2_rmse09_sequence_experiment.py
```

It synchronizes the ML scripts to z2 and runs the detailed sequence below from
`/srv/data/corsewind/backfill_runner`. Use `--remote-dry-run` to preview the
remote commands after sync.

If `training_table_feature_audit.json` reports stale shards, rebuild the monthly
training tables with the current feature-store code first:

```bash
python3 scripts/ml_dataset/launch_z2_rebuild_training_shards.py --background
python3 scripts/ml_dataset/check_z2_rmse09_status.py
```

Then rerun the RMSE experiment. This matters because the new upwind/context
geometry features only appear in `training_rows.parquet` after the shards have
been regenerated.

For the final experiment, also prefer background mode:

```bash
python3 scripts/ml_dataset/launch_z2_rmse09_sequence_experiment.py --background
python3 scripts/ml_dataset/check_z2_rmse09_status.py
```

### 1. Build 2025 Sequence Cases

```bash
cd /srv/data/corsewind/backfill_runner

/home/z2/corsewind-ml-smoke/.venv/bin/python scripts/ml_dataset/benchmark_chronos2_sequences.py \
  --training-table-root /srv/data/corsewind/ml_dataset/training_tables \
  --run-id-prefix residual_windsup_sst_prev \
  --start-month 2024-01 \
  --end-month 2026-06 \
  --eval-start 2025-01-01T00:00:00Z \
  --eval-end 2025-12-31T23:59:59Z \
  --context-length 96 \
  --prediction-length 4 \
  --issue-hour-start 8 \
  --issue-hour-end 17 \
  --max-cutoffs-per-spot 240 \
  --skip-hgb \
  --output-root /srv/data/corsewind/ml_dataset/benchmarks/sequence_2025_windsurf_1h_rmse09_v1 \
  --run-id sequence_2025_windsurf_1h_rmse09_v1_chronos_covariate \
  --batch-size 128
```

### 2. Add Zero-Shot Foundation Model Outputs

```bash
/home/z2/corsewind-ml-smoke/.venv/bin/python scripts/ml_dataset/benchmark_chronos2_saved_sequences.py \
  --benchmark-root /srv/data/corsewind/ml_dataset/benchmarks/sequence_2025_windsurf_1h_rmse09_v1 \
  --predictions-file predictions.parquet \
  --run-id sequence_2025_windsurf_1h_rmse09_v1_chronos2_univar_cross \
  --context-length 96 \
  --prediction-length 4 \
  --cross-learning

/home/z2/corsewind-ml-smoke/.venv-timesfm/bin/python scripts/ml_dataset/benchmark_timesfm_sequences.py \
  --benchmark-root /srv/data/corsewind/ml_dataset/benchmarks/sequence_2025_windsurf_1h_rmse09_v1 \
  --predictions-file predictions_with_chronos2_univariate.parquet \
  --run-id sequence_2025_windsurf_1h_rmse09_v1_timesfm \
  --context-length 96 \
  --prediction-length 4
```

Moirai can be added afterward, but it is not required for the first calibration
test because it was weaker than Chronos-2 and TimesFM on the 2026 benchmark.

### 3. Train A Leakage-Safe Residual Calibrator

```bash
/home/z2/corsewind-ml-smoke/.venv/bin/python scripts/ml_dataset/train_sequence_calibrator.py \
  --benchmark-root /srv/data/corsewind/ml_dataset/benchmarks/sequence_2025_windsurf_1h_rmse09_v1 \
  --benchmark-root /srv/data/corsewind/ml_dataset/benchmarks/sequence_2026_windsurf_1h_rmse09_v1 \
  --predictions-file predictions_with_timesfm.parquet \
  --output-root /srv/data/corsewind/ml_dataset/benchmarks/calibrator_2025_to_2026_ridge_v1 \
  --train-end 2026-01-01T00:00:00Z \
  --eval-start 2026-01-01T00:00:00Z \
  --target-mode residual \
  --residual-baseline raw_wind_mean_ms \
  --model-family ridge

/home/z2/corsewind-ml-smoke/.venv/bin/python scripts/ml_dataset/train_sequence_calibrator.py \
  --benchmark-root /srv/data/corsewind/ml_dataset/benchmarks/sequence_2025_windsurf_1h_rmse09_v1 \
  --benchmark-root /srv/data/corsewind/ml_dataset/benchmarks/sequence_2026_windsurf_1h_rmse09_v1 \
  --predictions-file predictions_with_timesfm.parquet \
  --output-root /srv/data/corsewind/ml_dataset/benchmarks/calibrator_2025_to_2026_hgb_v1 \
  --train-end 2026-01-01T00:00:00Z \
  --eval-start 2026-01-01T00:00:00Z \
  --target-mode residual \
  --residual-baseline raw_wind_mean_ms \
  --model-family hist_gradient_boosting \
  --max-iter 160 \
  --learning-rate 0.04 \
  --max-leaf-nodes 15 \
  --l2-regularization 0.2
```

The calibrator now adds:

- recent persistence and recent trend from `past_context.parquet`;
- cyclical hour/month features;
- model disagreement and quantile-spread features;
- optional training-table context features merged by
  `spot_id + issue_time_utc + lead_time_minutes`;
- neighboring-station geometry/upwind fields from `features__context_*` when
  the feature store/training tables have been rebuilt;
- residual learning against raw NWP by default;
- per-spot and per-lead diagnostic metrics.

The current one-command z2 launcher enables the training-table context merge by
default and writes:

```text
/srv/data/corsewind/ml_dataset/benchmarks/calibrator_2025_to_2026_sweep_context_v1/
```

This makes the experiment closer to SAPHIR: foundation-model predictions and
raw NWP remain the prior, while context-station features provide local spatial
correction signal.

### Success Criterion

Count only the 2026 temporal holdout score:

```text
rmse09_audit.json -> verdict == "pass"
```

The audit recomputes the best model RMSE from
`<sweep_root>/<model>/calibrator_predictions.parquet` when available and writes
a bootstrap confidence interval plus by-spot/by-lead diagnostics. The point RMSE
used for acceptance is `effective_rmse`, which comes from the prediction parquet
when available. The default pass gate also requires coverage across at least 3
spots, 4 lead times, and 20 issue days. The CI uses an `issue_day` block
bootstrap by default, because lead rows from the same day are correlated. The CI
and worst-spot table are the guardrail for deciding whether the result is robust
enough to trust operationally.
For the strictest validation run, launch the experiment with
`--require-ci-upper-below-threshold`; in that mode the audit only passes if the
95% bootstrap upper bound is also below `0.9`.
Add `--assert-goal` when the command should fail unless the final
machine-checkable RMSE09 gate passes.
When we want to avoid selecting the best model family directly on the final
holdout, first run a validation sweep, select a run with
`select_sequence_calibrator_run.py`, then pass that run to the final audit via
`--selected-run-name`. In that mode `audit_rmse09_results.py` evaluates the
preselected run exactly instead of picking the best final-holdout RMSE from the
sweep.

Every final run also writes:

```text
rmse09_error_analysis.json
rmse09_error_analysis.md
rmse09_decision.json
rmse09_decision.md
rmse09_run_manifest.json
```

Those files are the first place to look if the audit fails: they compare the
calibrated model to raw NWP and foundation-model outputs, then rank errors by
spot, lead time, wind regime, raw-NWP error regime, hour, and issue day.
They also include oracle bounds over the available prediction columns. If the
oracle by `spot_id + lead_time_minutes` is still above `0.9`, the current model
outputs probably do not contain enough independent signal and the next move is
new inputs/features. If the row oracle is below `0.9` but the trainable
calibrator is not, the signal exists but the calibration architecture or amount
of training data is still insufficient.
The decision file turns this evidence into the next action category:
`achieved`, `calibration_gap`, `routing_or_feature_gap`, `input_signal_gap`,
`inconclusive`, or `needs_more_evidence`.
The final machine-checkable gate is `assert_rmse09_goal.py`; it should be the
last command before claiming the objective is achieved. The manifest preserves
the exact run options, commands, dataset roots, expected artifacts, and git
provenance for the run being asserted.

### 2026-06-26 Routing Probes

After the first stale-context run failed with `routing_or_feature_gap`, we added
an `error_selector_extra_trees` calibrator family. It trains expected squared
error models for each available predictor and chooses the predictor with the
lowest predicted error at inference time. This is a non-oracle approximation of
the row oracle diagnostic.

Quick probes on the old `/srv/data/corsewind/ml_dataset` sequence benchmark
showed that routing alone is not enough:

```text
extra_trees global                  RMSE 1.409718
error_selector_extra_trees global   RMSE 1.420998
extra_trees by lead                 RMSE 1.429522
error_selector by lead              RMSE 1.452949
error_selector by spot              RMSE 1.488522
extra_trees by spot                 RMSE 1.510284
error_selector by spot+lead         RMSE 1.538137
extra_trees by spot+lead            RMSE 1.572240
```

The selector remains useful as a benchmark candidate, but these probes suggest
the route to `<0.9` is unlikely to be routing architecture alone. The fresh
feature rebuild must provide stronger explanatory inputs: recent model-error
state, upwind/context stations, thermal proxies, and observation freshness.

### 2026-06-26 Previous-Run Feature Fix

The Open-Meteo previous-run features were present in the spot feature store but
were filtered out when building residual training rows. The training table kept
the current AROME/Open-Meteo prefix, observations, context stations, SST, and
EUMETSAT fields, but not `previous_run_open_meteo_*`.

This meant earlier fresh rebuild attempts did not actually test the
"learn the NWP error by horizon / previous forecast run" hypothesis. We now keep
the `previous_run_open_meteo_` prefix in `build_residual_training_table.py` and
the feature audit requires both:

```text
features__previous_run_open_meteo_best_match_day1_wind_speed_10m
features__previous_run_open_meteo_best_match_day2_wind_speed_10m
```

A z2 smoke test on an existing feature-store chunk produced 44 previous-run
feature columns in the residual table and preserved the day-1/day-2 wind-speed
columns in Parquet. The fresh rebuild must be rerun end-to-end after this fix;
mixed old/new shards are not acceptable for the RMSE09 assertion.

The calibrator feature merge must also include these columns. The default
training-feature prefixes now include `features__previous_run_open_meteo_`, the
selection is prefix-ordered instead of raw-schema ordered, and the default
feature cap is `1400` so the corrected table is not silently truncated before
the previous-run columns.

The monthly shard builder also needed a per-day issue-hour guard. Previously,
`--start-hour-utc 8 --end-hour-utc 17` was applied as one continuous datetime
range per chunk, so multi-day chunks kept overnight issue rows between the
first and last day. `build_residual_training_table.py` now accepts
`--issue-start-hour-utc/--issue-end-hour-utc`, and
`run_training_backfill_pipeline.py` forwards the configured hours. A z2 smoke
test on a multi-day chunk and the rebuilt `2024-01` shard confirmed that only
hours `08..17` remain while the previous-run and SST audit still passes.

After a z2 out-of-memory reboot during the fresh rebuild/benchmark cycle, the
sklearn calibrator sweep was made less aggressive. RandomForest, ExtraTrees,
the error selector, and LightGBM now use a configurable `--n-jobs` setting
instead of `n_jobs=-1`; the default is `2`, and the RMSE09 runner forwards it
through `--calibrator-n-jobs`. The z2 watcher was also lowered to
`--batch-size 4` for the final sequence benchmark. After the reboot, the
watcher was restarted with `--max-train-rows 250000` so the 2026 holdout remains
complete while the heaviest supervised fits are bounded in memory. LightGBM
`4.6.0` was installed in the z2 smoke-test virtualenv and the watcher now passes
`--include-lightgbm`. The RMSE09 launcher now supports repeated
`--calibrator-model-family` flags; the z2 watcher is restricted to `ridge`,
`hist_gradient_boosting`, and `lightgbm` for the first post-reboot assertion run.
This keeps the experiment slower but makes it much less likely that the final
sweep masks a modeling result behind another machine-level OOM.

During the fresh rebuild, the `2025-08` Parquet export crashed in native
PyArrow code (`returncode -11`) with `--parquet-batch-size 5000`, leaving a file
without valid Parquet footer bytes. The JSONL source was complete
(`58467` rows), so the corrupted Parquet was removed and regenerated with
`--batch-size 1000`; the regenerated shard passed the freshness, feature, lead,
and `08..17 UTC` issue-hour audit. The remaining rebuild was restarted from
`2025-09` with `--parquet-batch-size 1000` to avoid repeating the native export
crash on later months.

If the score remains above `0.9`, the next move is not another in-sample
ensemble. It is to increase the 2025 training cases, add true neighbor/upwind
station sequences, and move the supervised learner to LightGBM/XGBoost for a
larger residual-correction table.

### 2026-06-26 z2 OOM Recovery

After a second z2 memory failure, the remote RMSE watcher was stopped before it
could launch the `250000`-row LightGBM sweep. The monthly rebuild itself was
left running because memory usage was low and the already-finished shards were
valid. The `2025-09` shard was audited after recovery:

```text
rows: 54920
fresh: true
missing critical features: none
outside 08..17 UTC: none
leads: 15, 30, 45, 60, 120, 180, 360
```

The next benchmark launch is intentionally conservative:

```text
batch_size: 2
max_train_rows: 120000
max_training_features: 900
calibrator_n_jobs: 1
model families: ridge, hist_gradient_boosting
LightGBM: disabled for the first post-OOM pass
```

The standard z2 launchers now forward `--max-train-rows`,
`--max-training-features`, `--calibrator-n-jobs`, and repeated
`--calibrator-model-family` flags so the intended memory limits cannot be lost
between the local watcher and the remote RMSE09 runner. Once this stable pass
finishes, LightGBM should be reintroduced as a second, smaller isolated run
rather than bundled into the first post-reboot benchmark.

The first low-memory setting exposed another modeling hazard: with the previous
prefix order, `--max-training-features 900` selected only observations and the
large context aggregates. It silently excluded the very features the fresh
rebuild was meant to test:

```text
features__model_error_now_*
features__previous_run_open_meteo_*
features__sst_*
features__eumetsat_*
```

The calibrator feature priority is now:

```text
obs -> model_error_now -> issue time -> previous runs -> SST -> EUMETSAT
-> nearest/coastal/inland/relief/global context -> context aggregates
```

On the fresh `2025-10` shard, the corrected 900-feature selection keeps all
critical families:

```text
obs: 34
model_error_now: 2
issue cyclic/table features: 4
previous_run_open_meteo: 44
sst: 5
eumetsat: 8
context_nearest/coastal/inland/relief/global: 523
context_agg: 280
```

`run_rmse09_sequence_experiment.py` now requires the selected training features
to include recent model error, previous-run day-1/day-2 wind, SST, and EUMETSAT
patterns. If a future memory cap would drop them again, the sweep fails loudly
instead of producing a misleading RMSE.

The existing sequence benchmark files under
`ml_dataset_z2_rebuild/benchmarks/sequence_{2025,2026}_windsurf_1h_rmse09_v1`
were older than the fresh rebuild cutoff. The low-memory watcher now launches
`run_rmse09_sequence_experiment.py` with `--force`, so Chronos/TimesFM sequence
predictions are regenerated from the same fresh monthly shards that feed the
calibrator. This avoids mixing fresh training-table features with stale
sequence predictions.

To make this guard repeatable, `audit_calibrator_feature_selection.py` now
applies the exact calibrator feature-selection logic to the monthly Parquet
schemas and fails if required patterns are not selected under the configured
feature cap. The low-memory watcher runs this audit between the fresh-shard
audit and the RMSE09 benchmark. A check on fresh `2025-09..2025-11` passed with
`--max-training-features 900`.

During the 2026 holdout rebuild, `2026-04` exposed a second Parquet safety
issue. The JSONL and evaluation contained `54143` rows, but the Parquet file
contained only `9000` rows because `export_training_table_parquet.py` failed
mid-write after PyArrow saw a mixed-type feature-source column. The monthly
runner was in `--continue-on-error`, so it advanced to the next month and left
the partial Parquet file in place.

The exporter now writes to a process-scoped temporary Parquet path and replaces
`training_rows.parquet` only after the writer closes successfully. On failure,
the temporary file is removed, so partial Parquet output cannot masquerade as a
valid shard. `2026-04` was re-exported from its complete JSONL after this fix:

```text
input_row_count: 54143
parquet_row_count: 54143
columns: 1165
string columns: 119
```

Manual audits now compare JSONL and Parquet row counts for newly finished 2026
holdout shards. `2026-01`, `2026-02`, `2026-03`, `2026-04`, `2026-05`, and
`2026-06` were checked or scheduled for this stricter validation before the
fresh RMSE09 benchmark.

## 2026-06-26 Iteration Log After z2 Reboot

z2 rebooted after an out-of-memory event. The RMSE09 run was relaunched in
low-memory mode and completed without a new machine crash. The final assertion
failed because the score remained above target, not because of infrastructure:

```text
run: calibrator_2025_to_2026_sweep_context_fresh_lowmem_v1
train/eval: 2025 -> 2026
models: ridge, hist_gradient_boosting
max_cutoffs_per_spot: 60
max_train_rows: 120000
max_training_features: 900
best model: hist_gradient_boosting
effective RMSE: 1.521771
raw NWP RMSE on same holdout: 2.126312
Chronos-2 univariate RMSE: 1.690946
TimesFM RMSE: 1.703295
decision: routing_or_feature_gap
row oracle RMSE: 0.574946
```

Interpretation: the available predictors contain row-level signal, but the
calibrator does not yet learn when to trust which source. Static spot/lead
routing is not enough; the oracle by spot+lead stayed around `1.61` RMSE.

An isolated `error_selector_extra_trees` run was tested because it directly
tries to learn the best predictor per row. The global selector was worse than
the HGB calibrator:

```text
run: calibrator_2025_to_2026_sweep_selector_lowmem_v1/error_selector_extra_trees
RMSE: 1.728719
MAE: 1.138816
```

The grouped selector was stopped after the global result because it was slow
and no longer looked like the main path to `0.9`.

To address the small sequential training set, a fresh 2024 sequence benchmark
was generated from the same rebuilt shards:

```text
run: sequence_2024_windsurf_1h_rmse09_v1
rows: 1440
spots: 6
Chronos-2 covariate wind RMSE: 1.581359
Chronos-2 univariate wind RMSE: 1.558440
TimesFM wind RMSE: 1.588919
raw NWP wind RMSE: 2.072568
```

Then a two-year sequence calibrator was trained on `2024+2025` and evaluated on
untouched `2026`:

```text
run: calibrator_2024_2025_to_2026_sweep_context_two_years_lowmem_v1
train rows: 3104
test rows: 1652
best model: extra_trees global
effective RMSE: 1.413998
MAE: 1.049443
raw RMSE delta: -0.712314
CI by issue-day: 1.280721 .. 1.550762
```

This is a real improvement over the one-year sequential calibrator:

```text
2025 -> 2026 best:      1.521771
2024+2025 -> 2026 best: 1.413998
gain:                  -0.107773 RMSE
```

The improvement confirms that more historical sequence predictions help. It
also confirms that the current sequence benchmark is still too small to support
spot-specific fine-tuning: spot/horizon variants did not beat the global model.

Worst 2026 holdout spots for the two-year best model:

```text
la_tonnara:   1.783319
balistra:     1.549483
piantarella:  1.388865
santa_manza:  1.388640
porto_polo:   1.284590
figari_eole:  1.274264
porticcio:    1.132880
```

Next hypothesis under test: the sequential benchmark is too sparse because it
samples only a limited number of cutoffs per spot/year. A tabular residual
model can train directly on the much larger monthly training shards.

One full tabular HGB attempt with `250000` train rows and `120000` test rows
was stopped before another OOM:

```text
run: tabular_hgb_2024_2025_to_2026_v1
split: train < 2026-01-01, test >= 2026-01-01
target: labels__residual_wind_mean_ms
peak observed RSS: about 15.4 GB
action: killed intentionally before z2 crashed
```

The tabular trainer was patched to support:

```text
--split-time-utc
--n-jobs
```

so future tabular runs can use the same strict temporal split and avoid
unbounded ExtraTrees CPU parallelism.

A smaller tabular HGB was then completed successfully:

```text
run: tabular_hgb_100k_2024_2025_to_2026_v1
split: train < 2026-01-01, test >= 2026-01-01
max_train_rows: 100000
max_test_rows: 60000
target: labels__residual_wind_mean_ms
feature columns: 952
train rows: 100000 sampled / 99855 usable target rows
test rows: 60000 sampled / 59695 usable target rows
raw NWP RMSE: 2.168151
corrected NWP RMSE: 1.472366
MAE: 1.107115
RMSE gain vs raw: 32.091%
```

By horizon:

```text
+15 min: 1.216212
+30 min: 1.331891
+45 min: 1.397157
+60 min: 1.450307
+120 min: 1.542998
+180 min: 1.615809
+360 min: 1.629277
```

Interpretation: dense tabular residual learning is useful and stable at 100k
rows, but this HGB setup still does not beat the two-year sequence calibrator
best score (`1.413998`). The very short horizon is promising (`+15 min` at
`1.216212`), but the model degrades with horizon and is not yet enough for the
`0.9` target.

LightGBM was then added to the same strict temporal tabular trainer because HGB
was memory-heavy and LightGBM should scale better on the dense shard table:

```text
run: tabular_lgbm_100k_2024_2025_to_2026_v1
split: train < 2026-01-01, test >= 2026-01-01
max_train_rows: 100000
max_test_rows: 60000
target: labels__residual_wind_mean_ms
feature columns: 952
train rows: 100000 sampled / 99855 usable target rows
test rows: 60000 sampled / 59695 usable target rows
raw NWP RMSE: 2.168151
corrected NWP RMSE: 1.458552
MAE: 1.091891
RMSE gain vs raw: 32.728%
```

By horizon:

```text
+15 min: 1.189317
+30 min: 1.315006
+45 min: 1.379619
+60 min: 1.432046
+120 min: 1.540433
+180 min: 1.610865
+360 min: 1.610795
```

Important correction of interpretation: the tabular global RMSE includes
`+120`, `+180`, and `+360` minute horizons, while the RMSE09 nowcasting goal is
primarily judged on the short `+15/+30/+45/+60` window. On that comparable
short-horizon subset:

```text
HGB 100k short horizons:      RMSE 1.359840 / MAE 1.031198
LightGBM 100k short horizons: RMSE 1.340337 / MAE 1.011708
raw NWP short horizons:       RMSE 2.169640 / MAE 1.657979
LightGBM gain vs raw:         38.223%
```

Interpretation: LightGBM is now the best dense tabular path tested so far for
short nowcasting horizons. It still does not reach `0.9`, but it is materially
better than the sequence calibrator when measured on the same `+15` to `+60`
minute business window. Next comparison should therefore report both:

```text
all available horizons RMSE
short nowcasting horizons RMSE (+15/+30/+45/+60)
```

The next memory-safe iteration should either increase the LightGBM training
sample to `150000` rows, or add an official `--eval-lead-minute` option to the
tabular trainer so the short-horizon score is written into each
`training_results.json` instead of being computed manually after the run.

Both were done next:

```text
code change: train_residual_correction_parquet.py now supports
  --eval-lead-minute 15 --eval-lead-minute 30 ...

run: tabular_lgbm_150k_2024_2025_to_2026_v1
split: train < 2026-01-01, test >= 2026-01-01
max_train_rows: 150000
max_test_rows: 60000
target: labels__residual_wind_mean_ms
feature columns: 757
train rows: 150000 sampled
test rows: 60000 sampled / 59660 usable target rows
raw NWP all-horizon RMSE: 2.193686
corrected all-horizon RMSE: 1.424298
MAE all horizons: 1.044858
RMSE gain vs raw all horizons: 35.073%

raw NWP short-horizon RMSE: 2.187306
corrected short-horizon RMSE: 1.281894
MAE short horizons: 0.948549
RMSE gain vs raw short horizons: 41.394%
```

Short-horizon detail:

```text
+15 min: 1.110513
+30 min: 1.263007
+45 min: 1.327085
+60 min: 1.375315
```

Interpretation: increasing the dense LightGBM sample from `100000` to `150000`
rows improved the short-horizon RMSE from `1.340337` to `1.281894`. This is the
first clearly scalable path after the z2 reboot. Memory stayed acceptable:
observed RSS peaked around `9.4 GB`, with no swap increase.

A `200000` row LightGBM run was then tested under close memory monitoring:

```text
run: tabular_lgbm_200k_2024_2025_to_2026_v1
split: train < 2026-01-01, test >= 2026-01-01
max_train_rows: 200000
max_test_rows: 60000
target: labels__residual_wind_mean_ms
feature columns: 757
train rows: 200000 sampled
test rows: 60000 sampled / 59660 usable target rows
raw NWP all-horizon RMSE: 2.193686
corrected all-horizon RMSE: 1.418875
MAE all horizons: 1.040925
RMSE gain vs raw all horizons: 35.320%

raw NWP short-horizon RMSE: 2.187306
corrected short-horizon RMSE: 1.278997
MAE short horizons: 0.946159
RMSE gain vs raw short horizons: 41.526%
```

Short-horizon detail:

```text
+15 min: 1.111091
+30 min: 1.261230
+45 min: 1.321803
+60 min: 1.371257
```

Interpretation: `200000` rows improved only marginally over `150000`
(`1.281894` -> `1.278997`, about `-0.0029` RMSE), while observed RSS briefly
rose to about `13.9 GB` with only about `1.4 GB` memory available. This is too
close to the z2 OOM boundary to justify a blind `250000` row attempt. The next
meaningful lever is no longer "more rows of the same shape"; it should be
better row-level regime features and/or target-specific filtering.

Current decision state:

```text
RMSE < 0.9: not achieved
best validated fresh sequence score: 1.413998
best validated dense tabular short-horizon score: 1.278997
most useful proven levers so far: more historical training data + dense
tabular residual learning
most suspicious blocker: missing row-level regime features for thermal/upwind
routing, especially on hard coastal spots
next action: stop increasing sample size blindly; add explicit thermal/upwind
regime features for La Tonnara, Balistra, Santa Manza, and Piantarella, then
retest the 150k/200k LightGBM setup
```

## Next Feature Iteration: Thermal And Upwind Regimes

After the `200000` row LightGBM run, the scaling curve is nearly flat:

```text
100k LightGBM short RMSE: 1.340337
150k LightGBM short RMSE: 1.281894
200k LightGBM short RMSE: 1.278997
```

The next implementation therefore adds regime features rather than simply
adding more rows:

```text
thermal_land_minus_sst_c
thermal_air_minus_sst_c
thermal_land_minus_air_c
thermal_clear_sky_fraction
thermal_low_cloud_fraction
thermal_insolation_proxy
thermal_land_sea_insolation_index
thermal_air_sea_insolation_index
thermal_recent_heating_rate_c_per_h
thermal_recent_pressure_tendency_hpa_per_h
thermal_cape_x_land_sea
thermal_low_cloud_suppression_index
thermal_inland_minus_coastal_temperature_c
thermal_relief_minus_coastal_temperature_c
thermal_inland_minus_coastal_pressure_hpa
thermal_relief_minus_coastal_pressure_hpa
thermal_coastal_minus_inland_wind_ms
thermal_coastal_minus_relief_wind_ms

nwp_horizon_wind_ramp_ms
nwp_horizon_gust_ramp_ms
nwp_horizon_temperature_ramp_c
nwp_horizon_pressure_msl_ramp_hpa
nwp_horizon_surface_pressure_ramp_hpa
nwp_horizon_shortwave_ramp
nwp_horizon_cloud_cover_ramp_pct
nwp_horizon_cape_ramp
nwp_horizon_wind_direction_delta_deg
nwp_error_persistence_plus_wind_ramp_ms
nwp_error_persistence_plus_gust_ramp_ms

context_agg_<group>_upwind_weighted_<field>_mean
context_agg_<group>_upwind_weight_sum
```

Leakage audit for these features:

- thermal features use only issue-time satellite/SST/NWP/context values;
- NWP horizon ramps use target-horizon NWP forecasts, which are known at issue
  time and are already part of the baseline, not target observations;
- upwind weighted context features use stations strictly before target time in
  the feature store builder;
- no future observation label is used.

Expected validation path:

```text
1. rebuild a small shard and confirm new columns are non-null;
2. rebuild 2024-2026 monthly shards if coverage is acceptable;
3. rerun LightGBM 150k or 200k with the same split and eval leads;
4. compare against the current short-horizon benchmark RMSE 1.278997.
```

Validation started on z2:

```text
smoke run: residual_windsup_sst_prev_regime_smoke_2026_01
rows: 52355
columns: 1357
parquet size: 18 MB
jsonl intermediate size: 2.4 GB
status: ok
```

Smoke coverage:

```text
features__nwp_horizon_*: non-null on 52355/52355 rows
features__nwp_error_persistence_*: non-null on 52355/52355 rows
features__thermal_*: 11 useful non-null columns
features__context_agg_all_upwind_*: 37 useful non-null columns
features__context_agg_coastal_upwind_*: 37 useful non-null columns
features__context_agg_inland_upwind_*: 35 useful non-null columns
features__context_agg_relief_upwind_*: 35 useful non-null columns
```

Full rebuild launched:

```text
run prefix: residual_windsup_sst_prev_regime_v1
range: 2024-01 .. 2026-06
status at first audit: 2024-01 completed, 2024-02 running
first shard rows: 48711
first shard columns: 1357
```

First historical shard coverage (`2024-01`):

```text
features__thermal_air_minus_sst_c: 47084 non-null
features__thermal_inland_minus_coastal_temperature_c: 47650 non-null
features__nwp_horizon_wind_ramp_ms: 47505 non-null
features__nwp_error_persistence_plus_wind_ramp_ms: 47505 non-null
features__context_agg_all_upwind_weighted_wind_mean_ms_mean: 43204 non-null
features__context_agg_inland_upwind_weighted_temperature_c_mean: 27060 non-null
features__thermal_land_minus_sst_c: present but 0 non-null
```

Interpretation: the feature iteration is technically valid on old shards, but
true EUMETSAT land-surface temperature is not present historically in the
current 2024-01 data. The useful historical thermal path is therefore mostly
air-SST, NWP radiation/cloud proxies, context station gradients, and upwind
station weighting.

Operational follow-up:

```text
full rebuild process:
  /srv/data/corsewind/ml_dataset/run_logs/rebuild_regime_v1_2024_2026.sh

post-rebuild watcher:
  /srv/data/corsewind/ml_dataset/run_logs/regime_v1_after_rebuild_audit_train.sh

watcher behavior:
  1. wait for rebuild_regime_v1_2024_2026.status to finish with status 0;
  2. require at least 30 monthly Parquet shards;
  3. write /srv/data/corsewind/ml_dataset/training_tables/regime_v1_feature_audit.json;
  4. run LightGBM 150k with the same temporal split and eval leads;
  5. stop safely if memory drops below 1.4 GB available or training RSS exceeds
     about 14.2 GB.

benchmark output:
  /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150k_regime_v1_2024_2025_to_2026_v1
```

Progress snapshot after watcher setup:

```text
2024-01: complete, 48711 rows
2024-02: complete, 45481 rows
2024-03: complete, 49248 rows
2024-04: exporting
disk: about 425 GB free
memory: stable
```

The first rebuild pass kept very large JSONL intermediates:

```text
combined monthly training_rows.jsonl: about 2.1-2.4 GB per month
chunk training_rows.jsonl: about another 2 GB+ per month
monthly Parquet: tens of MB
```

To keep the full 30-month rebuild practical, the pipeline was patched with:

```text
run_training_backfill_pipeline.py --cleanup-jsonl-after-parquet
run_monthly_training_shards.py --cleanup-jsonl-after-parquet
```

The running rebuild was restarted after `2024-04` with skip-existing Parquet and
cleanup enabled. Completed JSONL intermediates for `2024-01` to `2024-04` were
removed after verifying their Parquet files existed:

```text
deleted_files: 24
deleted_gb: 16.846
```

Cleanup was then verified on the next month:

```text
2024-05 rows: 52661
2024-05 Parquet: present
2024-05 combined training_rows.jsonl: removed
2024-05 chunk training_rows.jsonl files: removed
2024-05 deleted_bytes: 4894552552
```

This keeps the authoritative training artifact as Parquet and avoids letting
temporary JSONL files dominate storage during the remaining rebuild.

The post-rebuild watcher was then strengthened before the final training run:

```text
primary attempt: LightGBM 150000 rows
fallback: LightGBM 100000 rows if memory guard kills the primary attempt
memory guard:
  - kill if MemAvailable < 1.4 GB
  - kill if training RSS > about 14.2 GB
```

Progress after the fallback watcher was installed:

```text
completed Parquet shards: 7/30
latest completed: 2024-07
current month in progress: 2024-08
cleanup verified through 2024-07
```

Progress after the z2 reboot/status check:

```text
rebuild status: restarted_cleaning 2026-06-26T14:40:15+02:00
watcher status: started 2026-06-26T15:00:55+02:00
completed Parquet shards: 8/30
latest completed: 2024-08
current month in progress: 2024-09
cleanup verified through 2024-08
```

Latest verified cleanup:

```text
2024-08 rows: 54698
2024-08 deleted_jsonl_bytes: 5078081452
2024-08 Parquet: present
```

Progress after installing the tabular audit watcher:

```text
completed Parquet shards: 9/30
latest completed: 2024-09
current month in progress: 2024-10
2024-09 rows: 55040
JSONL leftovers: 0 combined files, 3 chunk files for the active month
disk: about 436 GB free on /srv/data
memory: about 13 GB available
```

An explicit tabular RMSE09 audit helper was added:

```text
script: scripts/ml_dataset/audit_tabular_rmse09_result.py
purpose: parse training_results.json, enforce the temporal/data-size gates,
         compare corrected RMSE against 0.9 and against the previous best
         short-horizon LightGBM score of 1.278997.
```

The helper is deployed on z2 and a separate audit watcher is running:

```text
status: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_rmse09_audit.status
log: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_rmse09_audit.log
pid: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_rmse09_audit.pid
```

It waits for either:

```text
/srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_2024_2025_to_2026_v1/training_results.json
/srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_2024_2025_to_2026_v1/training_results.json
```

and then writes:

```text
tabular_regime_v1_rmse09_audit.json
tabular_regime_v1_rmse09_audit.md
```

`check_z2_rmse09_status.py` now reports this whole path as well:

```text
active regime/RMSE09 processes
regime_v1 Parquet count
regime_v1 JSONL leftovers
LightGBM 150000/100000 training_results.json
tabular_regime_v1_rmse09_audit.json
```

Next automatic iteration prepared:

```text
watcher: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_lead_after_global.status
log: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_lead_after_global.log
pid: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_lead_after_global.pid
```

Behavior:

```text
1. wait for the global tabular regime_v1 audit;
2. if global LightGBM reaches RMSE < 0.9, skip;
3. otherwise run LightGBM with --fit-group-column lead_time_minutes;
4. audit the grouped result with audit_tabular_rmse09_result.py;
5. fallback from 150000 to 100000 sampled train rows if the memory guard fires.
```

Rationale: this follows the paper-aligned observation that short horizons do not
share the same error dynamics. The z2 synthetic smoke test confirmed that the
new grouped trainer path works and can beat the global model when the residual
law differs by lead:

```text
synthetic global eval RMSE: 0.370703
synthetic grouped-by-lead eval RMSE: 0.10586
```

Latest live rebuild snapshot:

```text
completed Parquet shards: 10/30
latest completed: 2024-10
current month in progress: 2024-11
all three downstream watchers are alive:
  - global post-rebuild audit/train
  - tabular RMSE09 audit
  - grouped-by-lead fallback if global misses 0.9
disk: about 435 GB free on /srv/data
memory: about 13 GB available
secret scan: clean
```

The tabular trainer now emits local error diagnostics for every regression run:

```text
corrected_nwp_by_spot
raw_nwp_by_spot
corrected_nwp_by_spot_lead
raw_nwp_by_spot_lead
```

The tabular RMSE09 audit now summarizes the worst spot groups and worst
spot+horizon groups in both JSON and Markdown. This is needed for the next
decision point: if the global/grouped models miss `0.9`, the remaining error
must be attributed to concrete spots and horizons rather than only a global
average.

Second automatic fallback prepared:

```text
watcher: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_spot_lead_after_lead.status
log: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_spot_lead_after_lead.log
pid: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_spot_lead_after_lead.pid
```

Behavior:

```text
1. wait for global and grouped-by-lead audits;
2. skip if either already reaches RMSE < 0.9;
3. otherwise run LightGBM with --fit-group-column spot_id and
   --fit-group-column lead_time_minutes;
4. audit the result with the same RMSE09 gate;
5. fallback from 150000 to 100000 sampled rows if memory guard fires.
```

Latest snapshot after deploying spot diagnostics:

```text
completed Parquet shards: 11/30
latest completed: 2024-11
current month in progress: 2024-12
watchers alive:
  - global post-rebuild audit/train
  - tabular RMSE09 audit
  - grouped-by-lead fallback
  - grouped-by-spot-lead fallback
disk: about 435 GB free on /srv/data
memory: about 13 GB available
secret scan: clean
```

Final tabular selector prepared:

```text
script: scripts/ml_dataset/select_tabular_rmse09_result.py
watcher status: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_selection.status
watcher log: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_selection.log
watcher pid: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_selection.pid
output json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_regime_v1_selection/tabular_regime_v1_selection.json
output md: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_regime_v1_selection/tabular_regime_v1_selection.md
```

Behavior:

```text
1. watch for any tabular_regime_v1_rmse09_audit.json;
2. if any run achieves RMSE < 0.9, write the selection and stop;
3. otherwise wait until the final spot+lead fallback is done;
4. select the best valid RMSE and include by-lead, worst-spot, and
   worst-spot-lead diagnostics.
```

Monitoring improvement:

```text
python3 scripts/ml_dataset/check_z2_rmse09_status.py --compact --tail-lines 5
```

The compact mode reports only the active RMSE09/regime path:

```text
resources
watcher statuses
active processes
regime_v1 Parquet count
JSONL leftovers
tabular audits
tabular selection
short watcher tails
```

Latest compact snapshot:

```text
completed Parquet shards: 12/30
latest completed: 2024-12
current month in progress: 2025-01
tabular audits: none yet
tabular selection: missing until first audit is produced
JSONL leftovers: 0 combined files, 2 chunk files for active month
disk: about 435 GB free on /srv/data
memory: about 14 GB available
```

Final assertion gate prepared:

```text
script: scripts/ml_dataset/assert_tabular_rmse09_goal.py
watcher status: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_assertion.status
watcher log: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_assertion.log
watcher pid: /srv/data/corsewind/ml_dataset/run_logs/tabular_regime_v1_assertion.pid
output json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_regime_v1_selection/tabular_regime_v1_assertion.json
```

This is the hard proof gate for the tabular path. It only passes if:

```text
selection decision == achieved
best corrected RMSE < 0.9
best audit verdict == achieved
selection RMSE and audit RMSE match
source Parquet count >= 30
train rows >= 100000
test rows >= 10000
metric rows >= 10000
temporal split == 2026-01-01T00:00:00Z
best audit has no reasons
```

The active goal should not be marked complete from tabular results unless this
assertion passes and the resulting JSON is inspected.

Partial 2024 regime_v1 audit:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_row_summary.json
verdict: pass
shards: 12/12
stale shards: 0
columns per shard: 1357
total rows 2024: 611879
min monthly rows: 45481
max monthly rows: 56286
```

Required feature patterns checked:

```text
features__thermal_air_minus_sst_c
features__thermal_inland_minus_coastal_temperature_c
features__nwp_horizon_wind_ramp_ms
features__nwp_error_persistence_plus_wind_ramp_ms
features__context_agg_all_upwind_weighted_wind_mean_ms_mean
```

Latest rebuild snapshot:

```text
completed Parquet shards: 13/30
latest completed: 2025-01
current month in progress: 2025-02
2025-01 rows: 48133
2025-01 deleted_jsonl_bytes: 4511475074
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-01` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_01_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_01_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_01_row_summary.json
verdict: pass
shards: 13/13
stale shards: 0
missing shards: 0
total rows: 660012
min monthly rows: 45481
max monthly rows: 56286
```

Extended partial audit after `2025-02` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_02_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_02_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_02_row_summary.json
verdict: pass
shards: 14/14
stale shards: 0
missing shards: 0
total rows: 698063
min monthly rows: 38051
max monthly rows: 56286
2025-02 rows: 38051
2025-02 deleted_jsonl_bytes: 3706532818
```

Latest rebuild snapshot:

```text
completed Parquet shards: 14/30
latest completed: 2025-02
current month in progress: 2025-03
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Journal maintenance update:

```text
date: 2026-06-26 15:36 CEST
change: added an explicit Working Log Contract at the top of this file.
reason: keep decisions, failed tests, benchmark evidence, and z2 paths
        recoverable after interruptions/reboots/context compaction.
decision: every meaningful experiment must now record scope, command/path,
          measured result, interpretation, and next action.
```

Extended partial audit after `2025-03` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_03_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_03_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_03_row_summary.json
verdict: pass
shards: 15/15
stale shards: 0
missing shards: 0
total rows: 738880
min monthly rows: 38051
max monthly rows: 56286
2025-03 rows: 40817
2025-03 deleted_jsonl_bytes: 3944545696
```

Latest rebuild snapshot:

```text
completed Parquet shards: 15/30
latest completed: 2025-03
current month in progress: 2025-04
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-04` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_04_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_04_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_04_row_summary.json
verdict: pass
shards: 16/16
stale shards: 0
missing shards: 0
total rows: 778115
min monthly rows: 38051
max monthly rows: 56286
2025-04 rows: 39235
2025-04 deleted_jsonl_bytes: 3783508640
```

Latest rebuild snapshot:

```text
completed Parquet shards: 16/30
latest completed: 2025-04
current month in progress: 2025-05
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-05` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_05_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_05_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_05_row_summary.json
verdict: pass
shards: 17/17
stale shards: 0
missing shards: 0
total rows: 817909
min monthly rows: 38051
max monthly rows: 56286
2025-05 rows: 39794
2025-05 deleted_jsonl_bytes: 3791407712
```

Latest rebuild snapshot:

```text
completed Parquet shards: 17/30
latest completed: 2025-05
current month in progress: 2025-06
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-06` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_06_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_06_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_06_row_summary.json
verdict: pass
shards: 18/18
stale shards: 0
missing shards: 0
total rows: 874620
min monthly rows: 38051
max monthly rows: 56711
2025-06 rows: 56711
2025-06 deleted_jsonl_bytes: 5312568538
```

Latest rebuild snapshot:

```text
completed Parquet shards: 18/30
latest completed: 2025-06
current month in progress: 2025-07
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-07` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_07_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_07_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_07_row_summary.json
verdict: pass
shards: 19/19
stale shards: 0
missing shards: 0
total rows: 935771
min monthly rows: 38051
max monthly rows: 61151
2025-07 rows: 61151
2025-07 deleted_jsonl_bytes: 5688405250
```

Latest rebuild snapshot:

```text
completed Parquet shards: 19/30
latest completed: 2025-07
current month in progress: 2025-08
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-08` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_08_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_08_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_08_row_summary.json
verdict: pass
shards: 20/20
stale shards: 0
missing shards: 0
total rows: 995229
min monthly rows: 38051
max monthly rows: 61151
2025-08 rows: 59458
2025-08 deleted_jsonl_bytes: 5565313712
```

Latest rebuild snapshot:

```text
completed Parquet shards: 20/30
latest completed: 2025-08
current month in progress: 2025-09
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-09` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_09_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_09_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_09_row_summary.json
verdict: pass
shards: 21/21
stale shards: 0
missing shards: 0
total rows: 1051082
min monthly rows: 38051
max monthly rows: 61151
2025-09 rows: 55853
2025-09 deleted_jsonl_bytes: 5277245024
```

Latest rebuild snapshot:

```text
completed Parquet shards: 21/30
latest completed: 2025-09
current month in progress: 2025-10
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-10` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_10_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_10_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_10_row_summary.json
verdict: pass
shards: 22/22
stale shards: 0
missing shards: 0
total rows: 1109732
min monthly rows: 38051
max monthly rows: 61151
2025-10 rows: 58650
2025-10 deleted_jsonl_bytes: 5585197476
```

Latest rebuild snapshot:

```text
completed Parquet shards: 22/30
latest completed: 2025-10
current month in progress: 2025-11
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-11` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_11_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_11_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_11_row_summary.json
verdict: pass
shards: 23/23
stale shards: 0
missing shards: 0
total rows: 1159076
min monthly rows: 38051
max monthly rows: 61151
2025-11 rows: 49344
2025-11 deleted_jsonl_bytes: 4714713030
```

Latest rebuild snapshot:

```text
completed Parquet shards: 23/30
latest completed: 2025-11
current month in progress: 2025-12
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2025-12` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_12_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_12_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2025_12_row_summary.json
verdict: pass
shards: 24/24
stale shards: 0
missing shards: 0
total rows: 1214769
min monthly rows: 38051
max monthly rows: 61151
2025-12 rows: 55693
2025-12 deleted_jsonl_bytes: 5434681652
```

Latest rebuild snapshot:

```text
completed Parquet shards: 24/30
latest completed: 2025-12
current month in progress: 2026-01
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2026-01` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_01_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_01_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_01_row_summary.json
verdict: pass
shards: 25/25
stale shards: 0
missing shards: 0
total rows: 1267124
min monthly rows: 38051
max monthly rows: 61151
2026-01 rows: 52355
2026-01 columns: 1357
2026-01 deleted_jsonl_bytes: 5086652052
```

Latest rebuild snapshot:

```text
completed Parquet shards: 25/30
latest completed: 2026-01
current month in progress: 2026-02
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2026-02` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_02_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_02_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_02_row_summary.json
verdict: pass
shards: 26/26
stale shards: 0
missing shards: 0
total rows: 1307942
min monthly rows: 38051
max monthly rows: 61151
2026-02 rows: 40818
2026-02 columns: 1357
2026-02 deleted_jsonl_bytes: 3940527612
```

Latest rebuild snapshot:

```text
completed Parquet shards: 26/30
latest completed: 2026-02
current month in progress: 2026-03
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2026-03` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_03_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_03_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_03_row_summary.json
verdict: pass
shards: 27/27
stale shards: 0
missing shards: 0
total rows: 1363274
min monthly rows: 38051
max monthly rows: 61151
2026-03 rows: 55332
2026-03 columns: 1357
2026-03 deleted_jsonl_bytes: 5405143104
```

Latest rebuild snapshot:

```text
completed Parquet shards: 27/30
latest completed: 2026-03
current month in progress: 2026-04
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2026-04` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_04_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_04_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_04_row_summary.json
verdict: pass
shards: 28/28
stale shards: 0
missing shards: 0
total rows: 1418382
min monthly rows: 38051
max monthly rows: 61151
2026-04 rows: 55108
2026-04 columns: 1357
2026-04 deleted_jsonl_bytes: 5273032382
```

Latest rebuild snapshot:

```text
completed Parquet shards: 28/30
latest completed: 2026-04
current month in progress: 2026-05
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Extended partial audit after `2026-05` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_05_feature_audit.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_05_feature_audit.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_partial_2024_to_2026_05_row_summary.json
verdict: pass
shards: 29/29
stale shards: 0
missing shards: 0
total rows: 1481605
min monthly rows: 38051
max monthly rows: 63223
2026-05 rows: 63223
2026-05 columns: 1357
2026-05 deleted_jsonl_bytes: 6024258828
```

Latest rebuild snapshot:

```text
completed Parquet shards: 29/30
latest completed: 2026-05
current month in progress: 2026-06
tabular audits: none yet
tabular selection/assertion: waiting for first audit
```

Full regime_v1 audit after `2026-06` completed:

```text
audit json: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_feature_audit_manual.json
audit md: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_feature_audit_manual.md
row summary: /srv/data/corsewind/ml_dataset/training_tables/regime_v1_full_2024_to_2026_06_row_summary.json
verdict: pass
shards: 30/30
stale shards: 0
missing shards: 0
total rows: 1528776
min monthly rows: 38051
max monthly rows: 63223
2026-06 rows: 47171
2026-06 columns: 1467
2026-06 deleted_jsonl_bytes: 4425102512
```

Latest rebuild snapshot:

```text
completed Parquet shards: 30/30
latest completed: 2026-06
rebuild status: finished 2026-06-26T16:58:34+02:00 status 0
post-rebuild watcher: launched LightGBM 150000 rows
training run: tabular_lgbm_150000_regime_v1_2024_2025_to_2026_v1
training split: 2026-01-01T00:00:00Z
target: labels__residual_wind_mean_ms
eval leads: 15, 30, 45, 60
tabular audits: waiting for training_results.json
tabular selection/assertion: waiting for first audit
note: 2026-06 has 1467 columns while earlier regime_v1 shards have 1357; the
      required-feature audit passes, but downstream trainer results should be
      checked for schema-union behavior.
```

Post-rebuild watcher evidence:

```text
internal full audit rows: 1528776
internal full audit path_count: 30
features__thermal_air_minus_sst_c non_null: 1518902
features__thermal_inland_minus_coastal_temperature_c non_null: 1525894
features__nwp_horizon_wind_ramp_ms non_null: 1525343
features__nwp_error_persistence_plus_wind_ramp_ms non_null: 1525343
features__context_agg_all_upwind_weighted_wind_mean_ms_mean non_null: 1365846
LightGBM 150000 training started: 2026-06-26T17:00:59+02:00
first memory sample: rss_kb=11944, mem_available_kb=15175300
second memory sample: rss_kb=2912852, mem_available_kb=12008608
memory guard sample before kill: rss_kb=15417212, mem_available_kb=106484
memory guard killed 150000-row run before z2 OOM.
watcher bug: it logged first_training_attempt_status=0 after the memory guard
             kill, so it did not launch its intended 100000-row fallback.
manual fallback launched:
  status: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_lgbm_100k_manual_guard.status
  log: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_lgbm_100k_manual_guard.log
  pid file: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_lgbm_100k_manual_guard.pid
  benchmark: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_2024_2025_to_2026_v1
```

Manual fallback result:

```text
run: tabular_lgbm_100000_regime_v1_2024_2025_to_2026_v1
training_results: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_2024_2025_to_2026_v1/training_results.json
audit json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_2024_2025_to_2026_v1/tabular_regime_v1_rmse09_audit.json
audit md: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_2024_2025_to_2026_v1/tabular_regime_v1_rmse09_audit.md
verdict: not_achieved
train rows: 100000
test rows: 60000
source parquet count: 30
feature columns: 1124
temporal split: 2026-01-01T00:00:00Z
metric source: eval_leads
eval leads: 15, 30, 45, 60
raw short RMSE: 2.167359
corrected short RMSE: 1.340857
corrected short MAE: 1.013049
gain vs raw: 38.134%
previous best short RMSE: 1.278997
gain vs previous best: -4.837%
gap to 0.9: 0.440857
by lead:
  +15 min: 1.190478
  +30 min: 1.313680
  +45 min: 1.390199
  +60 min: 1.426402
worst spots:
  cap_corse: 2.067265
  porto_polo: 1.658465
  la_tonnara: 1.648611
  lfvf: 1.639672
  santa_manza: 1.621757
```

Interpretation: the full `regime_v1` rebuild is valid, but the global
100000-row LightGBM is worse than the earlier dense 200000-row short-horizon
baseline. More complete 2026 holdout coverage and the new schema did not
automatically improve the global model. The next automatic probes, already
watching, are grouped-by-lead and grouped-by-spot+lead LightGBM; these test
whether the new thermal/upwind features help only after separating regimes.

Watcher repair / next probe:

```text
issue: grouped-by-lead watcher had a fragile verdict read and was killed before
       it could consume the global 100000 audit.
manual grouped-by-lead run launched:
  status: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_lead_100k_manual_guard.status
  log: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_lead_100k_manual_guard.log
  pid file: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_lead_100k_manual_guard.pid
  benchmark: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_grouped_by_lead_2024_2025_to_2026_v1
  grouped by: lead_time_minutes
  train rows: 100000
  eval leads: 15, 30, 45, 60
```

Grouped-by-lead first run failed before producing metrics:

```text
date: 2026-06-26 17:18 CEST
failure: pandas.errors.IndexingError, unalignable boolean Series in grouped
         evaluation path
root cause: prediction_series is indexed on the test subset, while
            test.loc[predicted_mask] expects a mask aligned to the full test
            dataframe
local fix: convert predicted_mask to predicted_indices before selecting
           eval_test
verification: python3 -m py_compile scripts/ml_dataset/train_residual_correction_parquet.py
decision: keep grouped-by-lead hypothesis, relaunch after syncing patched
          trainer to z2
```

Relaunch after grouped eval patch:

```text
date: 2026-06-26 17:19:50 CEST
remote script: /tmp/run_regime_v1_grouped_by_lead_100k_guarded.sh
remote pid: 67150
status: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_lead_100k_manual_guard.status
log: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_lead_100k_manual_guard.log
next action: wait for training_results.json and audit; if it still fails, inspect
             grouped residual evaluation before launching spot+lead.
```

Grouped-by-lead result:

```text
date: 2026-06-26 17:25:30 CEST
run: tabular_lgbm_100000_regime_v1_grouped_by_lead_2024_2025_to_2026_v1
audit json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_grouped_by_lead_2024_2025_to_2026_v1/tabular_regime_v1_rmse09_audit.json
verdict: not_achieved
train rows: 100000
test rows: 60000
feature columns: 1124
fit groups: 7 lead_time_minutes groups
skipped groups: 0
metric rows on +15/+30/+45/+60: 32415
raw short RMSE: 2.167359
corrected short RMSE: 1.326416
corrected short MAE: 0.992984
gain vs raw: 38.8%
previous best short RMSE: 1.278997
gain vs previous best: -3.708%
gap to 0.9: 0.426416
by lead:
  +15 min: 1.126259
  +30 min: 1.300475
  +45 min: 1.374834
  +60 min: 1.441701
worst spots:
  cap_corse: 2.045962
  lfvf: 1.687915
  porto_polo: 1.670715
  la_tonnara: 1.659509
  lfvh: 1.622210
  santa_manza: 1.610270
decision: keep the evidence, but do not keep this as best; it beats the global
          100k rebuild slightly, yet remains worse than the previous 1.278997
          baseline and far from 0.9.
```

Next probe triggered automatically:

```text
date: 2026-06-26 17:27:44 CEST
watcher: /tmp/regime_v1_grouped_by_spot_lead_after_lead.sh
run: tabular_lgbm_150000_regime_v1_grouped_by_spot_lead_2024_2025_to_2026_v1
status: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_spot_lead_after_lead.status
log: /srv/data/corsewind/ml_dataset/run_logs/regime_v1_grouped_by_spot_lead_after_lead.log
grouping: spot_id + lead_time_minutes
train rows requested: 150000
eval leads: 15, 30, 45, 60
reason: this is closer to the SAPHIR-style local adaptation/fine-tuning idea
        than a single global model or lead-only grouping.
next action: monitor memory guard; if 150k is killed, watcher should fallback
             to 100k; then audit the resulting training_results.json.
```

Audit hardening while spot+lead is running:

```text
date: 2026-06-26 17:29 CEST
file: scripts/ml_dataset/audit_tabular_rmse09_result.py
change: report fit_group_count, skipped_fit_group_count, model_test_row_count,
        and invalidate any grouped result that skipped groups overlapping
        requested eval leads +15/+30/+45/+60
reason: a future RMSE below 0.9 must not be accepted if it was obtained by
        silently dropping difficult spot/lead groups.
verification: python3 -m py_compile scripts/ml_dataset/audit_tabular_rmse09_result.py
remote sync: copied to z2 before spot+lead 150k finished, so its audit should
             use the stricter gate.
```

Grouped-by-spot+lead result:

```text
date: 2026-06-26 17:37:44 CEST
run: tabular_lgbm_150000_regime_v1_grouped_by_spot_lead_2024_2025_to_2026_v1
audit json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_grouped_by_spot_lead_2024_2025_to_2026_v1/tabular_regime_v1_rmse09_audit.json
verdict: invalid
train rows: 150000
test rows: 60000
model test rows covered: 50976
metric rows on +15/+30/+45/+60: 27429
feature columns: 1124
fit groups: 70
skipped groups: 7
skipped eval-lead groups: 4
skipped eval groups:
  porticcio +15 train=576 test=1232
  porticcio +30 train=522 test=1217
  porticcio +45 train=554 test=1231
  porticcio +60 train=533 test=1306
raw short RMSE on covered rows: 2.137728
corrected short RMSE on covered rows: 1.456295
corrected short MAE on covered rows: 1.087992
gain vs raw on covered rows: 31.877%
previous best short RMSE: 1.278997
gain vs previous best: -13.862%
gap to 0.9: 0.556295
by lead:
  +15 min: 1.266633
  +30 min: 1.435874
  +45 min: 1.524176
  +60 min: 1.544969
worst spots:
  cap_corse: 2.119764
  porto_polo: 2.070308
  lfvf: 1.718890
  santa_manza: 1.695386
  la_tonnara: 1.675695
decision: discard as a primary path. Full spot+lead models fragment the data too
          much, skip valid eval groups, and degrade RMSE. The better direction
          is a global/lead model with lightweight local calibration or fallback,
          not a full tree model per micro-group.
```

Global low-memory LightGBM probe:

```text
date: 2026-06-26 17:42:43 CEST
script patch: train_residual_correction_parquet.py now exposes LightGBM
              max_bin, feature_fraction, bagging_fraction, bagging_freq, and
              force_col_wise
verification: python3 -m py_compile scripts/ml_dataset/train_residual_correction_parquet.py
remote script: /tmp/run_regime_v1_lgbm_200k_lowmem_guarded.sh
remote pid: 68372
run: tabular_lgbm_200000_regime_v1_lowmem_2024_2025_to_2026_v1
train rows requested: 200000
settings: max_bin=63, feature_fraction=0.75, bagging_fraction=0.8,
          bagging_freq=1, force_col_wise=true, min_samples_leaf=30,
          l2=0.3, max_iter=700
reason: the best validated tabular direction remains global/lead residual
        correction; the earlier 150k global was stopped by memory, so this
        tests whether a memory-bounded global model can use more rows without
        fragmenting by spot.
next action: monitor memory guard and audit result.
```

Global low-memory LightGBM 150k result:

```text
date: 2026-06-26 17:54:44 CEST
run: tabular_lgbm_150000_regime_v1_lowmem_2024_2025_to_2026_v1
audit json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_lowmem_2024_2025_to_2026_v1/tabular_regime_v1_rmse09_audit.json
verdict: not_achieved
train rows: 150000
test rows: 60000
metric rows on +15/+30/+45/+60: 32415
raw short RMSE: 2.167359
corrected short RMSE: 1.326988
corrected short MAE: 1.004131
gain vs raw: 38.774%
previous best short RMSE: 1.278997
gain vs previous best: -3.752%
gap to 0.9: 0.426988
by lead:
  +15 min: 1.184457
  +30 min: 1.302225
  +45 min: 1.372264
  +60 min: 1.408852
decision: valid but not best. Memory-bounded 150k trains safely, but lowering
          LightGBM memory knobs costs enough accuracy that it remains worse than
          the previous dense 150k/200k-era best.
```

Short-horizon-only global LightGBM probe:

```text
date: 2026-06-26 17:5x CEST
script patch: train_residual_correction_parquet.py now supports
              --include-lead-minute before split sampling
verification: python3 -m py_compile scripts/ml_dataset/train_residual_correction_parquet.py
remote script: /tmp/run_regime_v1_lgbm_150k_shortleads_guarded.sh
remote pid: 68885
run: tabular_lgbm_150000_regime_v1_shortleads_lowmem_2024_2025_to_2026_v1
train rows requested: 150000
included leads: 15, 30, 45, 60
settings: same low-memory LightGBM settings as 150k lowmem
reason: previous global models trained on 15/30/45/60/120/180/360 while the
        RMSE09 objective evaluates only +15..+60; this tests whether dedicating
        all training rows/model capacity to the short horizons improves the
        true objective.
next action: monitor memory guard and audit.
```

Top-feature short-horizon 175k result:

```text
date: 2026-06-26 18:18:17 CEST
run: tabular_lgbm_175000_regime_v1_shortleads_top300_2024_2025_to_2026_v1
audit json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_175000_regime_v1_shortleads_top300_2024_2025_to_2026_v1/tabular_regime_v1_rmse09_audit.json
verdict: not_achieved
train rows: 175000
test rows: 60000
feature columns: 300
metric rows on +15/+30/+45/+60: 59641
raw short RMSE: 2.174881
corrected short RMSE: 1.306696
corrected short MAE: 0.982257
gain vs raw: 39.919%
previous best short RMSE: 1.278997
gain vs previous best: -2.166%
gap to 0.9: 0.406696
by lead:
  +15 min: 1.133619
  +30 min: 1.280520
  +45 min: 1.355753
  +60 min: 1.407727
decision: valid but slightly worse than 150k top-300. More rows alone are not
          improving this feature/model family; keep 150k top-300 as the best
          new short-only/top-feature probe.
```

Top-feature short-horizon 200k result:

```text
date: 2026-06-26 18:13:39 CEST
run: tabular_lgbm_200000_regime_v1_shortleads_top300_2024_2025_to_2026_v1
status: memory_guard
peak observed before kill: rss_kb=15411208, mem_available_kb=80484
training_results: none
decision: 200k top-300 still exceeds z2 memory. The viable ceiling is between
          150k and 200k, so test 175k with the same top-300 configuration.
```

Top-feature short-horizon 175k launch:

```text
date: 2026-06-26 18:14 CEST
remote script: /tmp/run_regime_v1_lgbm_175k_short_top300_guarded.sh
remote pid: 69756
run: tabular_lgbm_175000_regime_v1_shortleads_top300_2024_2025_to_2026_v1
train rows requested: 175000
included leads: 15, 30, 45, 60
feature count requested: top 300 numeric features
settings: same as 150k/200k top-300
next action: monitor memory guard and audit.
```

Top-feature short-horizon 150k result:

```text
date: 2026-06-26 18:09:23 CEST
run: tabular_lgbm_150000_regime_v1_shortleads_top300_2024_2025_to_2026_v1
audit json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_shortleads_top300_2024_2025_to_2026_v1/tabular_regime_v1_rmse09_audit.json
verdict: not_achieved
train rows: 150000
test rows: 60000
feature columns: 300
metric rows on +15/+30/+45/+60: 59641
raw short RMSE: 2.174881
corrected short RMSE: 1.305132
corrected short MAE: 0.981467
gain vs raw: 39.991%
previous best short RMSE: 1.278997
gain vs previous best: -2.043%
gap to 0.9: 0.405132
by lead:
  +15 min: 1.133024
  +30 min: 1.279322
  +45 min: 1.352174
  +60 min: 1.406756
decision: keep and scale. Top-300 feature restriction improves memory and RMSE
          versus 100k short-only, so the next logical test is 200k top-300.
```

Top-feature short-horizon 200k launch:

```text
date: 2026-06-26 18:10 CEST
remote script: /tmp/run_regime_v1_lgbm_200k_short_top300_guarded.sh
remote pid: 69595
run: tabular_lgbm_200000_regime_v1_shortleads_top300_2024_2025_to_2026_v1
train rows requested: 200000
included leads: 15, 30, 45, 60
feature count requested: top 300 numeric features
settings: same as 150k top-300
next action: monitor memory guard and audit.
```

Short-horizon-only global LightGBM 100k result:

```text
date: 2026-06-26 18:03:47 CEST
run: tabular_lgbm_100000_regime_v1_shortleads_lowmem_2024_2025_to_2026_v1
audit json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_shortleads_lowmem_2024_2025_to_2026_v1/tabular_regime_v1_rmse09_audit.json
verdict: not_achieved
train rows: 100000
test rows: 60000
metric rows on +15/+30/+45/+60: 59641
raw short RMSE: 2.174881
corrected short RMSE: 1.311605
corrected short MAE: 0.986970
gain vs raw: 39.693%
previous best short RMSE: 1.278997
gain vs previous best: -2.549%
gap to 0.9: 0.411605
by lead:
  +15 min: 1.145450
  +30 min: 1.285408
  +45 min: 1.356650
  +60 min: 1.411344
decision: keep as useful evidence. Short-only training improves over global
          lowmem and lead-only variants, but still misses both 0.9 and the
          previous 1.278997 best.
```

Top-feature short-horizon probe:

```text
date: 2026-06-26 18:06:03 CEST
allowlist: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/feature_allowlists/shortleads_lgbm100k_top300.json
allowlist source: top 300 nonzero LightGBM importances from the 100k
                  shortleads lowmem run
script patch: train_residual_correction_parquet.py now supports
              --feature-allowlist-json
verification: python3 -m py_compile scripts/ml_dataset/train_residual_correction_parquet.py
remote script: /tmp/run_regime_v1_lgbm_150k_short_top300_guarded.sh
remote pid: 69385
run: tabular_lgbm_150000_regime_v1_shortleads_top300_2024_2025_to_2026_v1
train rows requested: 150000
included leads: 15, 30, 45, 60
feature count requested: top 300 numeric features
settings: max_bin=127, feature_fraction=0.9, bagging_fraction=0.85,
          learning_rate=0.025, max_iter=900
reason: reduce memory/noise enough to train 150k short-horizon rows while
        preserving the strongest recent-error, NWP-ramp, previous-run, thermal,
        and context features.
next action: monitor memory guard and audit.
```

Short-horizon-only 150k result:

```text
date: 2026-06-26 17:59:23 CEST
run: tabular_lgbm_150000_regime_v1_shortleads_lowmem_2024_2025_to_2026_v1
status: memory_guard
peak observed before kill: rss_kb=15406620, mem_available_kb=80068
training_results: none
decision: short-lead filtering is still worth testing, but 150k short-only rows
          exceed z2 memory with the current 1124-feature design. Retry at 100k.
```

Short-horizon-only global LightGBM 100k launch:

```text
date: 2026-06-26 18:00 CEST
remote script: /tmp/run_regime_v1_lgbm_100k_shortleads_guarded.sh
remote pid: 69046
run: tabular_lgbm_100000_regime_v1_shortleads_lowmem_2024_2025_to_2026_v1
train rows requested: 100000
included leads: 15, 30, 45, 60
settings: same short-lead low-memory LightGBM settings
next action: monitor memory guard and audit.
```

Global low-memory LightGBM 200k result:

```text
date: 2026-06-26 17:45:24 CEST
run: tabular_lgbm_200000_regime_v1_lowmem_2024_2025_to_2026_v1
status: memory_guard
peak observed before kill: rss_kb=15369508, mem_available_kb=118188
training_results: none
decision: 200k is still too large for the current z2 memory envelope even with
          max_bin=63/feature_fraction=0.75/bagging. Retry the same low-memory
          configuration at 150k before trying more aggressive feature/bin
          reduction.
```

Global low-memory LightGBM 150k launch:

```text
date: 2026-06-26 17:46 CEST
remote script: /tmp/run_regime_v1_lgbm_150k_lowmem_guarded.sh
remote pid: 68592
run: tabular_lgbm_150000_regime_v1_lowmem_2024_2025_to_2026_v1
train rows requested: 150000
settings: same low-memory LightGBM settings as the 200k run
next action: monitor memory guard and audit result.
```

Latest explicit tabular selection:

```text
date: 2026-06-26 18:19:42 CEST
output json: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_regime_v1_selection/tabular_regime_v1_selection_latest.json
output md: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_regime_v1_selection/tabular_regime_v1_selection_latest.md
decision: not_achieved
audit_count: 7
valid_audit_count: 6
best run: tabular_lgbm_150000_regime_v1_shortleads_top300_2024_2025_to_2026_v1
best RMSE: 1.305132
best MAE: 0.981467
gap to 0.9: 0.405132
process status: no active z2 training process after selection
decision: stop pure row-count brute force for now. The feature/input gap still
          dominates. Next iteration should diagnose hard spots/hours and add
          local calibration or stronger local observations rather than only
          increasing train rows.
```

Tabular holdout error diagnosis tooling:

```text
date: 2026-06-26 18:26-18:35 CEST
script added: scripts/ml_dataset/analyze_tabular_rmse09_errors.py
purpose: rebuild the exact tabular holdout, score a saved LightGBM model, write
         row-level predictions, and diagnose errors by spot, lead, issue hour,
         month, wind bins, raw-error bins, and key feature quantiles.
verification:
  python3 -m py_compile scripts/ml_dataset/analyze_tabular_rmse09_errors.py
remote output, top300 regime run:
  /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_shortleads_top300_2024_2025_to_2026_v1/tabular_holdout_predictions.parquet
  /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_shortleads_top300_2024_2025_to_2026_v1/tabular_error_diagnosis.json
  /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_shortleads_top300_2024_2025_to_2026_v1/tabular_error_diagnosis.md
result reproduced audit exactly:
  corrected RMSE: 1.305132
  raw RMSE: 2.174881
main findings:
  worst spots/leads: la_tonnara +60, cap_corse +60, la_tonnara +45,
                     santa_manza +45/+60, porto_polo +60
  error grows with lead: +15 1.133024, +60 1.406756
  high actual winds are underpredicted: actual 8+ m/s RMSE 1.688396,
                                      bias -0.627926
  very low winds are overpredicted: actual 0-2 m/s RMSE 1.293863,
                                  bias +0.937225
  January/February are much worse than April/May/June.
decision: the ceiling is not caused by a generic audit bug. The model still
          regresses toward the mean and misses local/high-wind regimes.
```

Top300 plus local categorical identity probe:

```text
date: 2026-06-26 18:28-18:31 CEST
allowlist: /srv/data/corsewind/ml_dataset_z2_rebuild/benchmarks/feature_allowlists/shortleads_lgbm100k_top300_plus_local_cats.json
added categoricals: spot_id, station_id, spot_kind, spot_source_type
run: tabular_lgbm_150000_regime_v1_shortleads_top300_localcats_2024_2025_to_2026_v1
status: success
corrected RMSE: 1.307472
MAE: 0.983447
raw RMSE: 2.174881
metric rows: 59641
feature columns: 303
decision: reject as new best. Local categorical identity alone does not unlock
          the error ceiling and is slightly worse than top300 numeric-only
          (1.305132).
```

Historical best diagnosis and selection hardening:

```text
date: 2026-06-26 18:33-18:35 CEST
run diagnosed: tabular_lgbm_200k_2024_2025_to_2026_v1
audit output:
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_200k_2024_2025_to_2026_v1/tabular_rmse09_audit.json
diagnosis output:
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_200k_2024_2025_to_2026_v1/tabular_error_diagnosis.json
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_200k_2024_2025_to_2026_v1/tabular_error_diagnosis.md
corrected RMSE: 1.278997
raw RMSE: 2.187306
metric rows: 31429
main findings:
  same hard spots: cap_corse, la_parata, la_tonnara, santa_manza
  same lead degradation: +15 1.111091, +60 1.371257
  actual 8+ m/s bin RMSE 1.799837, bias -0.714778
  January/February are again worst.
script patch:
  analyze_tabular_rmse09_errors.py now separates reconstruction lead filters
  from metric lead filters, so old all-lead runs can be diagnosed correctly.
  select_tabular_rmse09_result.py now discovers both tabular_regime_v1_rmse09_audit.json
  and tabular_rmse09_audit.json.
decision: the true best branch before the next run is the older
          residual_windsup_sst_prev family, not regime_v1.
```

Previous-feature-family row-count probes:

```text
date: 2026-06-26 18:36-18:47 CEST
base feature family: residual_windsup_sst_prev
reason: historical best uses this 757-feature family and beats all regime_v1
        runs, so test whether the best signal benefits from more rows under
        memory-safe LightGBM settings.

run: tabular_lgbm_250k_prev_lowmem_2024_2025_to_2026_v1
status: memory_guard
finished: 2026-06-26 18:40:24 CEST
peak before kill: rss_kb=15249916, mem_available_kb=108908
training_results: none
decision: 250k is too high for z2 with this feature family.

run: tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1
status: success
finished: 2026-06-26 18:47:58 CEST
audit:
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1/tabular_rmse09_audit.json
corrected RMSE: 1.276846
corrected MAE: 0.943833
raw RMSE: 2.187306
metric rows: 31429
train rows: 225000
feature columns: 758
gain vs previous best 1.278997: +0.168%
gap to 0.9: 0.376846
by lead:
  +15 min: 1.111586
  +30 min: 1.260469
  +45 min: 1.318601
  +60 min: 1.367343
decision: new valid best, but the gain is tiny. More rows help slightly, but
          cannot plausibly close the remaining 0.376846 RMSE gap alone.
```

Latest global tabular selection:

```text
date: 2026-06-26 18:48:39 CEST
selection output:
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_rmse09_selection/tabular_selection_latest.json
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_rmse09_selection/tabular_selection_latest.md
audit_count: 10
valid_audit_count: 9
decision: not_achieved
best run: tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1
best RMSE: 1.276846
best MAE: 0.943833
gap to 0.9: 0.376846
next: row count is nearly exhausted on z2. The next serious path is not larger
      LightGBM, but reducing hard-regime errors: high-wind tail, January/February
      synoptic regimes, and local spots cap_corse/la_tonnara/santa_manza/porto_polo
      via stronger observations, regime-specific models, or valid out-of-fold
      calibration/stacking.
```

Temporal second-stage calibration probe:

```text
date: 2026-06-26 18:50-19:09 CEST
script added/updated:
  scripts/ml_dataset/train_prediction_residual_calibrator.py
purpose:
  Train a second-stage correction from prediction parquet files, with explicit
  anti-leakage exclusions:
    - excludes actual_wind_mean_ms from features
    - excludes raw_error_ms/corrected_error_ms/abs errors
    - excludes actual/error bins
    - uses only calibration-period labels, then evaluates on 2026 holdout

base evaluation predictions:
  /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1/tabular_holdout_predictions.parquet
base RMSE on eval leads: 1.276846

calibration base model:
  run: tabular_lgbm_calbase_2024_to_2025h2_v1
  train window: before 2025-07-01
  calibration prediction window: 2025-07-01 to 2026-01-01
  calibration predictions:
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_calbase_2024_to_2025h2_v1/calibration_predictions_2025h2.parquet
  calibration base RMSE on 15-60 min leads: 1.312450

second-stage calibrators, trained on 2025-H2 and evaluated on 2026:
  HGB:
    output: /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_hgb_v1/calibration_results.json
    RMSE: 1.272756
    gain vs base: +0.320%
  LightGBM:
    output: /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_lgbm_v1/calibration_results.json
    RMSE: 1.275868
    gain vs base: +0.077%
  ExtraTrees fixed scale=1.0:
    output: /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_v1/calibration_results.json
    RMSE: 1.269943
    gain vs base: +0.541%

diagnostic only, not official:
  A post-hoc sweep of ExtraTrees correction scale on the 2026 holdout showed
  an apparent optimum near scale=0.70, RMSE about 1.2680. This is not counted
  as a valid score because the scale was selected on the evaluation holdout.

official autoscale result:
  output: /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1/calibration_results.json
  model family: ExtraTrees
  scale selection:
    fit rows: 16315
    validation rows: 15417
    validation window: 2025-10-01 to 2026-01-01
    selected scale: 0.95
  evaluation rows: 31429
  calibrated RMSE: 1.269403
  calibrated MAE: 0.930905
  bias: 0.012952
  gain vs base 1.276846: +0.583%
  gap to 0.9: 0.369403
  by lead:
    +15 min: 1.102812
    +30 min: 1.251636
    +45 min: 1.313328
    +60 min: 1.360039
decision:
  New valid best for the whole pipeline, but still far from 0.9. Temporal
  stacking helps consistently and reduces bias, yet only closes about 0.0074
  RMSE. The remaining gap is not a simple calibration issue; it likely requires
  new signal/coverage for hard local/high-wind regimes.
```

High-wind weighted LightGBM probe:

```text
date: 2026-06-26 19:10-19:25 CEST
script patch:
  scripts/ml_dataset/train_residual_correction_parquet.py now supports
  target-dependent sample weights for regression training:
    --target-high-wind-weight-threshold-ms
    --target-high-wind-weight
  The weight uses labels only in the training loss, not as prediction features.

reason:
  Error diagnosis showed persistent underprediction for actual wind >= 8 m/s.
  Test whether emphasizing high-wind training examples improves global RMSE.

run 1:
  run: tabular_lgbm_225k_prev_weight8x15_2024_2025_to_2026_v1
  threshold: target wind >= 8 m/s
  high wind weight: 1.5
  status: memory_guard
  finished: 2026-06-26 19:16:50 CEST
  peak before kill: rss_kb=15280324, mem_available_kb=115340
  decision: 225k weighted is too high for z2 memory.

run 2:
  run: tabular_lgbm_200k_prev_weight8x15_2024_2025_to_2026_v1
  status: success
  audit:
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_200k_prev_weight8x15_2024_2025_to_2026_v1/tabular_rmse09_audit.json
  diagnosis:
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_200k_prev_weight8x15_2024_2025_to_2026_v1/tabular_error_diagnosis.json
  sample weighting:
    training rows: 199812
    high-wind rows: 32511
    mean weight: 1.081354
  corrected RMSE: 1.283157
  corrected MAE: 0.950688
  raw RMSE: 2.187306
  comparison:
    base 200k RMSE: 1.278997
    base 225k RMSE: 1.276846
    best calibrated pipeline RMSE: 1.269403
  by lead:
    +15 min: 1.115551
    +30 min: 1.263975
    +45 min: 1.329712
    +60 min: 1.373621
  actual wind bins:
    weighted actual 8+ RMSE: 1.736154, bias -0.608595
    base 200k actual 8+ RMSE: 1.799837, bias -0.714778
    base 225k actual 8+ RMSE: 1.787061, bias -0.719912
decision:
  Reject as main model: global RMSE worsens. Keep as evidence that high-wind
  specialization can improve the high-wind tail, but it must be routed or
  ensembled using a non-leaky gate selected on a calibration period.
```

Non-leaky high-wind routing probe:

```text
date: 2026-06-26 19:27-19:36 CEST
script added:
  scripts/ml_dataset/route_prediction_models.py
purpose:
  Route between a base model and the high-wind weighted model using only
  non-leaky prediction-time columns. Gate candidates are selected on 2025-H2
  calibration data, then applied once to the 2026 holdout.

calibration weighted model:
  run: tabular_lgbm_calbase_weight8x15_2024_to_2025h2_v1
  train window: before 2025-07-01
  calibration prediction window: 2025-H2
  prediction parquet:
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_calbase_weight8x15_2024_to_2025h2_v1/calibration_predictions_2025h2.parquet
  calibration weighted RMSE on 15-60 min leads: 1.312138

route selection:
  output:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_router_base225_vs_weighted200_cal2025h2_to_2026_v1/routing_results.json
  calibration rows: 31749
  evaluation rows: 31636
  selected gate: alt_predicted_residual_wind_mean_ms >= 1.879612
  selected use-alt rate on calibration: 10.0003%

route applied to 2026 base225 vs weighted200:
  base RMSE: 1.276846
  weighted alt RMSE: 1.283157
  routed RMSE: 1.276534
  gain vs base: +0.024%
  decision: tiny valid gain, not competitive with the calibrated pipeline.

same gate applied to calibrated base pipeline vs weighted200:
  calibrated base RMSE: 1.269403
  weighted alt RMSE: 1.283157
  routed RMSE: 1.270708
  use-alt rate on 2026: 7.6776%
  decision: reject. The gate that slightly improves the uncalibrated base
            harms the best calibrated pipeline and does not improve the
            high-wind bin.

decision:
  High-wind weighting/routing does not currently improve the best pipeline.
  Keep the evidence: high-wind weighting can reduce high-wind underprediction
  in isolation, but the available prediction-time gates are too weak to route
  it safely. Further progress likely needs stronger high-wind/thermal/local
  signals rather than loss weighting alone.
```

RMSE gap/oracle audit:

```text
date: 2026-06-26 19:42-19:48 CEST
script added:
  scripts/ml_dataset/analyze_rmse09_gap_oracles.py
purpose:
  Quantify what it would take to reach RMSE < 0.9 from the current best valid
  model, and separate deployable evidence from diagnostic/oracle evidence.

primary model:
  /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1/calibrated_predictions_2026.parquet
  prediction column: calibrated_wind_mean_ms

local report copies:
  docs/ml_nowcasting/rmse09_gap_audit_current_best_v1.md
  docs/ml_nowcasting/rmse09_gap_audit_current_best_v1.json
  docs/ml_nowcasting/rmse09_feature_coverage_error_modes_current_best_v1.json

remote report:
  /srv/data/corsewind/ml_dataset/benchmarks/rmse09_gap_audit_current_best_v1/gap_audit.json
  /srv/data/corsewind/ml_dataset/benchmarks/rmse09_gap_audit_current_best_v1/gap_audit.md
  /srv/data/corsewind/ml_dataset/benchmarks/rmse09_gap_audit_current_best_v1/feature_coverage_error_modes.json

overall:
  rows: 31429
  RMSE: 1.269403
  MAE: 0.930905
  bias: 0.012952
  gap to 0.9: 0.369403
  current SSE: 50644.207955
  target SSE for 0.9: 25457.49
  MSE reduction needed: 49.733%

error tail:
  top 1% rows contain 17.022% of SSE
  top 5% rows contain 41.033% of SSE
  top 10% rows contain 56.666% of SSE
  perfect correction of the worst 2350 rows (7.477%) would be required to
  reach 0.9, if every other row stayed unchanged.

diagnostic existing-model oracle:
  compared: current calibrated, base225, weighted200, HGB calibrator,
            LightGBM calibrator, fixed-scale ExtraTrees, weighted router
  row-wise best-existing-model oracle RMSE: 1.134896
  note: this oracle uses observed 2026 targets to choose the best model per row,
        so it is not deployable.
  decision implication:
    Even perfect routing among the existing model outputs would not reach 0.9.
    The current family of predictions lacks enough independent signal.

largest error contributors:
  actual wind >= 8 m/s:
    rows: 5258
    SSE share: 31.671%
    global RMSE if perfect: 1.049306
  lead +60 min:
    rows: 10007
    SSE share: 36.549%
    global RMSE if perfect: 1.011157
  lead +45/+60 min:
    rows: 17109
    SSE share: 60.737%
    global RMSE if perfect: 0.795411
  critical spots la_tonnara/santa_manza/balistra:
    rows: 13241
    SSE share: 52.365%
    global RMSE if perfect: 0.876118
  critical spots OR actual wind >= 8 m/s:
    rows: 15137
    SSE share: 64.337%
    global RMSE if perfect: 0.758069

feature coverage on hard/error regimes:
  obs lag 15 min:
    available globally: 99.9%
    available top 5% errors: 99.75%
    median age: about 1.1-1.4 min
  SST:
    available globally: 100%
    available top 5% errors: 100%
  EUMETSAT land surface temperature:
    available globally: 0.55%
    available top 5% errors: 0.32%
  global coastal context station:
    available globally: 98.68%
  global inland context station:
    available globally: 98.86%
  global relief context station:
    available globally: 40.0%
    available on critical spots la_tonnara/santa_manza/balistra: 0.0%

decision:
  Do not spend the next iteration on another shallow ensemble/routing of the
  same predictions. The measured gap is too large and the best possible
  row-wise oracle across existing outputs is still 1.134896.

next logical work:
  1. Improve signals for the critical spots first: la_tonnara, santa_manza,
     balistra.
  2. Fill missing relief/mountain context for those spots; current relief
     context is completely absent for the critical set.
  3. Make EUMETSAT land surface temperature coverage usable, or replace it with
     a reliable land-heating proxy. Current coverage is too sparse to help the
     thermal regime.
  4. For 45-60 min leads, add stronger future-context predictors: pressure
     gradients, vertical profile/instability, land-sea thermal contrast, and
     upwind station trends.
  5. Only after those data additions, rerun the best temporal calibrator and
     this gap audit.
```

Relief context fix and rebuild:

```text
date: 2026-06-26 19:50-19:55 CEST
root cause:
  The context registry contains active mountain/relief stations near the hard
  south Corsica spots, but the slot selector chose the nearest relief station
  regardless of station activity window.

  For la_tonnara/santa_manza/balistra, the nearest relief candidate was:
    station_id: 20061002
    name: CARBINI-COL DE MELA
    altitude: 1105 m
    station_end: 2009-06-01

  Because it ended in 2009, it produced no useful 2024-2026 observations and
  occupied the `context_global_relief_1` slot ahead of active stations.

code change:
  scripts/ml_dataset/build_spot_feature_store.py

  Added window-aware context station filtering:
    - exclude if station_end < build start_datetime
    - exclude if station_start > build end_datetime
    - preserve previous behavior when no build window is supplied

probe:
  remote output:
    /srv/data/corsewind/ml_dataset/feature_store/relief_window_filter_probe_2026_01_01_03
  local doc:
    docs/ml_nowcasting/relief_context_window_filter_fix.md

probe result on critical spots:
  balistra:
    rows: 288
    global_relief_1: 20254006 QUENZA
    available: 100%
    wind/temp non-null: 100%
  la_tonnara:
    rows: 201
    global_relief_1: 20160001 MOCA-CROCE
    available: 100%
    wind/temp non-null: 100%
  santa_manza:
    rows: 203
    global_relief_1: 20254006 QUENZA
    available: 100%
    wind/temp non-null: 100%

rebuild launched:
  host: z2
  pid: /srv/data/corsewind/ml_dataset/run_logs/rebuild_training_shards.pid
       currently 75033 wrapper / 75039 runner at launch
  log:
    /srv/data/corsewind/ml_dataset/run_logs/rebuild_training_shards.log
  command scope:
    run_id_prefix: residual_windsup_sst_prev_regime_v1
    start_month: 2024-01
    end_month: 2026-06
    leads: 15,30,45,60,120,180,360

early rebuild evidence:
  2024-01 last chunk feature_source_hits.context_global_relief_1 = 1433
  This confirms the rebuilt shards are no longer empty for global relief
  context.

next action after rebuild completes:
  1. Audit rebuilt training table coverage for critical spots.
  2. Retrain the best 225k LightGBM base and 2025-H2 ExtraTrees temporal
     calibrator.
  3. Rerun RMSE gap audit and compare critical-spot/high-wind/+45/+60 metrics.
```


Post-rebuild low-memory watcher:

```text
date: 2026-06-26 20:10-20:12 CEST
reason:
  z2 was rebooted after an OOM. The current rebuild is still running on
  /srv/data/corsewind/ml_dataset with prefix residual_windsup_sst_prev_regime_v1.
  Do not launch heavy training in parallel with the monthly shard rebuild.

state checked before launch:
  rebuild status file: missing/running
  monthly rebuild runner pid: 75039
  current month at check: 2024-04
  confirmed rebuilt shards from log: 2024-01, 2024-02, 2024-03

script added:
  scripts/ml_dataset/z2_regime_v1_post_rebuild_lowmem_watcher.sh

remote deployed path:
  /srv/data/corsewind/backfill_runner/scripts/ml_dataset/z2_regime_v1_post_rebuild_lowmem_watcher.sh

remote watcher:
  process: bash scripts/ml_dataset/z2_regime_v1_post_rebuild_lowmem_watcher.sh
  observed pid: 76049
  status:
    /srv/data/corsewind/ml_dataset/run_logs/regime_v1_post_rebuild_lowmem.status
  log:
    /srv/data/corsewind/ml_dataset/run_logs/regime_v1_post_rebuild_lowmem.log
  pid file:
    /srv/data/corsewind/ml_dataset/run_logs/regime_v1_post_rebuild_lowmem.pid

watcher behavior:
  1. Wait until monthly rebuild processes stop.
  2. Audit all 2024-01..2026-06 shards for required feature families:
     previous Open-Meteo runs, SST, global relief station, thermal deltas,
     EUMETSAT LST availability.
  3. Audit full relief coverage for critical spots:
     la_tonnara, santa_manza, balistra.
  4. Train the comparable post-relief LightGBM base:
     tabular_lgbm_225k_prev_relief_active_v1_2024_2025_to_2026_v1
     split: 2026-01-01T00:00:00Z
     max_train_rows: 225000
     max_test_rows: 60000
     target: labels__residual_wind_mean_ms only
  5. Generate holdout predictions and tabular RMSE09 audit.
  6. Train the 2025-H2 calibration base:
     tabular_lgbm_calbase_relief_active_v1_2024_to_2025h2_v1
     split: 2025-07-01T00:00:00Z
  7. Train ExtraTrees temporal second-stage calibrator with scale selected on
     2025-Q4, then evaluate on 2026.
  8. Rerun tabular selection and post-relief RMSE gap audit.

memory safety:
  BLAS/OpenMP threads forced to 1.
  LightGBM n_jobs=1.
  training commands run sequentially.
  watcher kills the training process if MemAvailable < 2.2 GB or RSS > 13 GB.

decision rule:
  The post-relief base must be compared primarily against the pre-fix best
  tabular_lgbm_225k_prev_lowmem RMSE 1.276846 on 2026 short leads.
  The post-relief calibrated model must be compared against the current best
  temporal calibrated RMSE 1.269403.
  Any result >= these baselines means relief-slot availability alone did not
  close the gap and the next iteration must target missing LST/land-heating or
  stronger spot-specific/high-wind features.
```

Post-rebuild watcher preflight on completed months:

```text
date: 2026-06-26 20:13 CEST
scope:
  completed rebuilt shards checked before full rebuild completion:
    2024-01, 2024-02, 2024-03, 2024-04

feature-family audit:
  command:
    audit_training_table_features.py on 2024-01..2024-04 with required
    previous-run, SST, global relief, thermal, and EUMETSAT LST columns.
  verdict: pass
  stale_shard_count: 0
  column_count per checked shard: 1343

critical relief coverage audit:
  command:
    audit_relief_context_coverage.py on 2024-01..2024-04
  rows checked on leads 15/30/45/60: 56,763

  balistra:
    rows: 18,792
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

  la_tonnara:
    rows: 19,120
    relief station: 20160001
    global_relief_1_available: 99.163%
    relief wind/temp/deltas: 99.163%

  santa_manza:
    rows: 18,851
    relief station: 20254006
    global_relief_1_available: 99.151%
    relief wind/temp/deltas: 99.151%

decision:
  The watcher audit commands are compatible with the rebuilt shards. The relief
  fix is materially present in the early rebuilt months, so the current wait is
  worthwhile. Do not restart the rebuild unless a later month fails schema or
  coverage checks.
```

Watcher robustness patch:

```text
date: 2026-06-26 20:16 CEST
issue:
  The first watcher instance detected the outer rebuild launcher wrapper pid
  75033 because that wrapper command line also contained
  run_monthly_training_shards.py. That was probably harmless while the rebuild
  was running, but it could become a stale wait condition if the wrapper stayed
  alive after the real child processes finished.

change:
  scripts/ml_dataset/z2_regime_v1_post_rebuild_lowmem_watcher.sh now ignores:
    - its own watcher command line;
    - the outer `setsid bash -lc ... rebuild_training_shards.log` launcher.

remote action:
  deployed patched watcher to z2 and restarted only the watcher.
  rebuild process was not touched.

remote state after restart:
  watcher pid: 76430
  watcher log:
    /srv/data/corsewind/ml_dataset/run_logs/regime_v1_post_rebuild_lowmem.log
  watcher now reports rebuild still running pid=75036, i.e. the real rebuild
  shell/process chain, not the outer wrapper pid=75033.
```

Rebuild checkpoint:

```text
date: 2026-06-26 20:17-20:18 CEST
host: z2
memory:
  total: 15 GiB
  available: about 12 GiB
  swap used: about 252 MiB

state:
  rebuild status: running
  completed fresh shard exports:
    2024-01, 2024-02, 2024-03, 2024-04
  active month:
    2024-05
  active step:
    export_training_table_parquet.py for
    residual_windsup_sst_prev_regime_v1_2024_05

watcher:
  pid: 76430
  status: started
  last log line still reports rebuild running pid=75036.

decision:
  No corrective action needed. Wait for the remaining monthly rebuilds; the
  post-rebuild watcher should then run audits and train the post-relief models.
```

Post-relief summary automation and May shard check:

```text
date: 2026-06-26 20:20-20:21 CEST

script added:
  scripts/ml_dataset/summarize_post_relief_rmse09_iteration.py

purpose:
  Once the post-rebuild watcher finishes, produce a compact JSON/Markdown
  summary comparing:
    - post-relief base RMSE vs current base best 1.276846
    - post-relief calibrated RMSE vs current calibrated best 1.269403
    - best post-relief RMSE vs target 0.9
    - relief coverage by critical spot
    - gap-audit oracle headline metrics

watcher update:
  scripts/ml_dataset/z2_regime_v1_post_rebuild_lowmem_watcher.sh now writes:
    /srv/data/corsewind/ml_dataset/benchmarks/post_relief_iteration_summary_relief_active_v1.json
    /srv/data/corsewind/ml_dataset/benchmarks/post_relief_iteration_summary_relief_active_v1.md

remote deployment:
  deployed both scripts to:
    /srv/data/corsewind/backfill_runner/scripts/ml_dataset/
  validated with:
    py_compile summarize_post_relief_rmse09_iteration.py
    bash -n z2_regime_v1_post_rebuild_lowmem_watcher.sh

watcher restart:
  watcher restarted only; rebuild was not touched.
  new watcher pid: 76697
  watcher now waits on rebuild pid=75036.

rebuild progress:
  fresh export completed:
    2024-05 at 20:19 CEST
  active month after that:
    2024-06

2024-05 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  column_count: 1343

2024-05 critical relief coverage:
  balistra:
    rows: 4,900
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    rows: 4,874
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    rows: 4,900
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  The relief fix remains valid beyond the first four rebuilt months. Continue
  waiting for the full rebuild; post-relief summary will make the next RMSE
  decision immediate once watcher training finishes.
```

June shard checkpoint:

```text
date: 2026-06-26 20:24-20:25 CEST
progress:
  fresh export completed:
    2024-06 at 20:24 CEST
  active month after that:
    2024-07

2024-06 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  column_count: 1343

2024-06 critical relief coverage:
  balistra:
    rows: 4,740
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    rows: 4,737
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    rows: 4,579
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  June confirms the same pattern as May: the active relief station selection is
  working in full monthly shards. Continue rebuild without intervention.
```

## Working rule - experiment journal discipline

```text
date: 2026-06-26
rule:
  Every meaningful experiment, model run, data-source test, z2 restart, feature
  change, score, failure, and decision must be recorded in this journal or in a
  linked markdown/json artifact under docs/ml_nowcasting/.

minimum note content:
  - hypothesis or reason for the action
  - exact dataset/run/artifact path when applicable
  - train/validation/test split when applicable
  - RMSE/MAE/bias or relevant audit metric when applicable
  - decision taken afterward
  - whether the result is deployable, diagnostic only, or blocked

why:
  The RMSE < 0.9 target requires many iterations. Without durable notes, we risk
  retesting the same idea, losing track of leakage risks, or forgetting which
  data fix actually moved the score.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, July 2024

```text
date: 2026-06-26 20:28-20:31 CEST
context:
  z2 had been rebooted after an OOM. The full post-relief rebuild is running
  again under:
    /srv/data/corsewind/ml_dataset/run_logs/rebuild_training_shards.log
  watcher:
    /srv/data/corsewind/ml_dataset/run_logs/regime_v1_post_rebuild_lowmem.log

machine state:
  memory at 20:28:
    total: 15 GiB
    available: 14 GiB
    swap free: 15 GiB
  watcher status:
    started 2026-06-26T20:20:30+02:00
    waiting on rebuild pid=75036

rebuild progress:
  completed fresh parquet profiles:
    2024-01 at 19:57
    2024-02 at 20:02
    2024-03 at 20:07
    2024-04 at 20:13
    2024-05 at 20:19
    2024-06 at 20:24
    2024-07 at 20:30
  active next month after July:
    2024-08

2024-07 table:
  rows: 56,286
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,668
    windsup_public_spot_history: 47,618
  rows_by_critical_spot:
    balistra: 8,575 total rows; 4,900 rows in relief audit target subset
    la_tonnara: 8,037 total rows; 4,860 rows in relief audit target subset
    santa_manza: 8,280 total rows; 4,900 rows in relief audit target subset

2024-07 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2024-07 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  July is validated. The active-station relief fix holds on a dense summer month,
  so continue the rebuild and let the low-memory watcher run the post-relief
  training/calibration once all months through 2026-06 are rebuilt.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, August 2024

```text
date: 2026-06-26 20:35-20:37 CEST
context:
  After the validated July shard, the rebuild advanced to August and completed
  the fresh parquet export at 20:37 CEST. The rebuild then advanced to 2024-09.

machine state:
  memory at 20:35:
    total: 15 GiB
    available: 13 GiB
    swap free: 15 GiB
  no post-relief training artifacts yet because the full rebuild is still
  running.

2024-08 table:
  rows: 54,698
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,673
    windsup_public_spot_history: 46,025
  rows_by_critical_spot:
    balistra: 8,575 total rows; 4,900 rows in relief audit target subset
    la_tonnara: 8,099 total rows; 4,900 rows in relief audit target subset
    santa_manza: 8,280 total rows; 4,900 rows in relief audit target subset

2024-08 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2024-08 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  August is validated. The relief fix is now confirmed on consecutive dense
  summer months. Continue without launching additional heavy jobs; the watcher
  remains the correct mechanism for post-rebuild training to avoid another OOM.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, September 2024

```text
date: 2026-06-26 20:38-20:45 CEST
context:
  September took longer than the previous dense summer months but completed a
  fresh parquet export at 20:43 CEST. The rebuild then advanced to 2024-10.

machine state:
  memory while September was exporting at 20:42:
    total: 15 GiB
    available: 9.6 GiB
    swap free: 15 GiB
  memory after September completed at 20:44:
    total: 15 GiB
    available: 13 GiB
    swap free: 15 GiB

2024-09 table:
  rows: 55,040
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,364
    windsup_public_spot_history: 46,676
  rows_by_lead:
    +15: 7,079
    +30: 7,044
    +45: 7,001
    +60: 9,051
    +120: 8,906
    +180: 8,686
    +360: 7,273
  rows_by_critical_spot:
    balistra: 8,288 total rows; 4,740 rows in relief audit target subset
    la_tonnara: 7,800 total rows; 4,734 rows in relief audit target subset
    santa_manza: 8,009 total rows; 4,740 rows in relief audit target subset

2024-09 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2024-09 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  September is validated. January through September 2024 have now been rebuilt
  after the active-station relief fix, and the critical south Corsica relief
  context is consistently available. Continue monitoring October and keep the
  low-memory watcher as the only post-rebuild trainer.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, October 2024

```text
date: 2026-06-26 20:49-20:52 CEST
context:
  October completed a fresh parquet export at 20:49 CEST. The rebuild advanced
  to 2024-11 and was already exporting November at 20:52 CEST.

machine state:
  memory at 20:52:
    total: 15 GiB
    available: 14 GiB
    swap free: 15 GiB
  no post-relief score yet:
    watcher is alive and still waiting for the full rebuild to finish.

2024-10 table:
  rows: 52,914
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,654
    windsup_public_spot_history: 44,260
  rows_by_lead:
    +15: 6,793
    +30: 6,743
    +45: 6,693
    +60: 8,807
    +120: 8,621
    +180: 8,356
    +360: 6,901
  rows_by_critical_spot:
    balistra: 8,402 total rows; 4,830 rows in relief audit target subset
    la_tonnara: 7,926 total rows; 4,830 rows in relief audit target subset
    santa_manza: 8,118 total rows; 4,815 rows in relief audit target subset

2024-10 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2024-10 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature: 100.0%
    relief wind: 99.669%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature: 100.0%
    relief wind: 99.668%

decision:
  October is validated. Availability is no longer the blocker: the small
  0.33% wind-null gap on the 20254006 relief station should be handled by the
  model's missingness flags/imputation and is not comparable to the previous
  complete relief-context hole on south Corsica. Continue rebuild; next score
  remains pending until the watcher runs post-rebuild training/calibration.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, November 2024

```text
date: 2026-06-26 20:54-20:56 CEST
context:
  November completed a fresh parquet export at 20:54 CEST. The rebuild advanced
  to 2024-12.

machine state:
  memory after November completed:
    total: 15 GiB
    available: 13 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2024-11 table:
  rows: 47,590
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,384
    windsup_public_spot_history: 39,206
  rows_by_lead:
    +15: 5,851
    +30: 5,849
    +45: 5,847
    +60: 7,942
    +120: 7,863
    +180: 7,697
    +360: 6,541
  rows_by_critical_spot:
    balistra: 8,289 total rows; 4,740 rows in relief audit target subset
    la_tonnara: 7,797 total rows; 4,719 rows in relief audit target subset
    santa_manza: 8,017 total rows; 4,740 rows in relief audit target subset

2024-11 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2024-11 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief temperature: 100.0%
    relief wind: 99.661%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  November is validated. January through November 2024 are now rebuilt and
  audited after the active-station relief fix. Continue to December, then keep
  the rebuild moving toward 2026-06 before judging model score.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, December 2024

```text
date: 2026-06-26 20:59-21:03 CEST
context:
  December initially had stale parquet/profile artifacts from 15:23 CEST while
  the fresh export was still running. Validation was deliberately delayed until
  the profile mtime refreshed to 21:00 CEST. The rebuild then advanced to
  2025-01.

stale-artifact guard:
  old December profile:
    mtime: 2026-06-26 15:23 CEST
    column_count observed: 1,357
  fresh December profile:
    mtime: 2026-06-26 21:00 CEST
    column_count observed: 1,343
  decision:
    only the fresh profile was audited/accepted.

machine state:
  memory after December export:
    total: 15 GiB
    available: 14 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2024-12 table:
  rows: 49,007
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,473
    windsup_public_spot_history: 40,534
  rows_by_lead:
    +15: 6,042
    +30: 6,041
    +45: 6,040
    +60: 8,157
    +120: 8,078
    +180: 7,911
    +360: 6,738
  rows_by_critical_spot:
    balistra: 8,575 total rows; 4,900 rows in relief audit target subset
    la_tonnara: 8,040 total rows; 4,866 rows in relief audit target subset
    santa_manza: 8,280 total rows; 4,900 rows in relief audit target subset

2024-12 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2024-12 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 99.671%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

milestone:
  All 2024 monthly shards, 2024-01 through 2024-12, are now rebuilt and audited
  after the active-station relief fix.

decision:
  2024 is accepted as the first fully rebuilt/audited year. Continue 2025
  rebuild. Do not evaluate the post-relief model yet; the target comparison
  requires the complete 2024-01 through 2026-06 rebuilt dataset.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, January 2025

```text
date: 2026-06-26 21:04-21:08 CEST
context:
  January 2025 initially had stale parquet/profile artifacts from 15:28 CEST
  while the fresh export was still running. Validation was delayed until the
  profile mtime refreshed to 21:05 CEST. The rebuild then advanced to 2025-02.

stale-artifact guard:
  old January profile:
    mtime: 2026-06-26 15:28 CEST
  fresh January profile:
    mtime: 2026-06-26 21:05 CEST
    column_count observed: 1,343
  decision:
    only the fresh profile was audited/accepted.

machine state:
  memory after January export:
    total: 15 GiB
    available: 14 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-01 table:
  rows: 48,133
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 7,818
    windsup_public_spot_history: 40,315
  rows_by_lead:
    +15: 6,042
    +30: 6,036
    +45: 6,030
    +60: 7,978
    +120: 7,879
    +180: 7,692
    +360: 6,476
  rows_by_critical_spot:
    balistra: 8,503 total rows; 4,878 rows in relief audit target subset
    la_tonnara: 7,988 total rows; 4,858 rows in relief audit target subset
    santa_manza: 8,216 total rows; 4,878 rows in relief audit target subset

2025-01 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-01 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  January 2025 is validated. Continue 2025 rebuild and keep using fresh mtime
  checks before accepting monthly shards.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, February 2025

```text
date: 2026-06-26 21:08-21:11 CEST
context:
  February 2025 completed a fresh parquet export at 21:09 CEST. The rebuild
  advanced to 2025-03.

machine state:
  memory after February export:
    total: 15 GiB
    available: 13 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-02 table:
  rows: 38,051
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 7,840
    windsup_public_spot_history: 30,211
  rows_by_lead:
    +15: 4,522
    +30: 4,521
    +45: 4,520
    +60: 6,479
    +120: 6,400
    +180: 6,237
    +360: 5,372
  rows_by_critical_spot:
    balistra: 7,756 total rows; 4,432 rows in relief audit target subset
    la_tonnara: 7,291 total rows; 4,424 rows in relief audit target subset
    santa_manza: 770 total rows; 468 rows in relief audit target subset

2025-02 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-02 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

coverage note:
  February passes feature/relief audits, but Santa Manza target coverage is much
  lower than surrounding months:
    2025-01 santa_manza total rows: 8,216
    2025-02 santa_manza total rows: 770
  This is not a relief-context failure. It is likely target-source coverage
  variation and should be checked later if February errors or spot calibration
  behave oddly.

decision:
  February 2025 is accepted with a coverage caveat for Santa Manza. Continue the
  rebuild, and include spot/month coverage in later score diagnostics.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, March 2025

```text
date: 2026-06-26 21:12-21:16 CEST
context:
  March 2025 initially had a stale profile from 15:36 CEST. Validation waited
  until the fresh profile appeared at 21:14 CEST. The rebuild then advanced to
  2025-04.

machine state:
  memory after March export:
    total: 15 GiB
    available: 13 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-03 table:
  rows: 40,817
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,676
    windsup_public_spot_history: 32,141
  rows_by_lead:
    +15: 4,811
    +30: 4,809
    +45: 4,807
    +60: 6,974
    +120: 6,889
    +180: 6,714
    +360: 5,813
  rows_by_critical_spot:
    balistra: 8,575 total rows; 4,900 rows in relief audit target subset
    la_tonnara: 8,077 total rows; 4,900 rows in relief audit target subset
    santa_manza: 0 total rows in profile; no relief audit row

2025-03 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-03 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    no target rows in the March training table, so no relief coverage can be
    measured for this spot/month.

coverage note:
  This is a stronger version of the February Santa Manza coverage caveat:
    2025-01 santa_manza total rows: 8,216
    2025-02 santa_manza total rows: 770
    2025-03 santa_manza total rows: 0
  The monthly shard is valid globally, but Santa Manza local calibration will
  have a target coverage hole around late winter/early spring 2025 unless this
  source gap is backfilled later.

decision:
  March 2025 is accepted for global training, with a significant Santa Manza
  target-coverage caveat. Later score diagnostics must include spot/month
  support counts before interpreting Santa Manza RMSE or calibration quality.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, April 2025

```text
date: 2026-06-26 21:16-21:21 CEST
context:
  April 2025 initially had a stale profile from 15:40 CEST. Validation waited
  until the fresh profile appeared at 21:18 CEST. The rebuild then advanced to
  2025-05.

machine state:
  memory after April export:
    total: 15 GiB
    available: 14 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-04 table:
  rows: 39,235
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,392
    windsup_public_spot_history: 30,843
  rows_by_lead:
    +15: 4,678
    +30: 4,651
    +45: 4,625
    +60: 6,697
    +120: 6,604
    +180: 6,436
    +360: 5,544
  rows_by_critical_spot:
    balistra: 8,281 total rows; 4,732 rows in relief audit target subset
    la_tonnara: 7,753 total rows; 4,709 rows in relief audit target subset
    santa_manza: 0 total rows in profile; no relief audit row

2025-04 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-04 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature: 100.0%
    relief wind: 99.662%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    no target rows in the April training table, so no relief coverage can be
    measured for this spot/month.

coverage note:
  Santa Manza now shows a multi-month target support collapse:
    2025-01 santa_manza total rows: 8,216
    2025-02 santa_manza total rows: 770
    2025-03 santa_manza total rows: 0
    2025-04 santa_manza total rows: 0
  This is a target-source/backfill coverage issue, not a relief-context issue.
  It should be investigated before trusting spot-specific Santa Manza metrics
  for the 2025-Q1/Q2 boundary.

decision:
  April 2025 is accepted for global training, with the same significant Santa
  Manza target-coverage caveat as March. Continue rebuild and keep support
  counts in the post-relief score audit.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, May 2025

```text
date: 2026-06-26 21:21-21:25 CEST
context:
  May 2025 initially had a stale profile from 15:45 CEST. Validation waited
  until the fresh profile appeared at 21:23 CEST. The rebuild then advanced to
  2025-06.

machine state:
  memory after May export:
    total: 15 GiB
    available: 14 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-05 table:
  rows: 39,794
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,680
    windsup_public_spot_history: 31,114
  rows_by_lead:
    +15: 4,771
    +30: 4,736
    +45: 4,700
    +60: 6,834
    +120: 6,708
    +180: 6,510
    +360: 5,535
  rows_by_critical_spot:
    balistra: 8,172 total rows; 4,718 rows in relief audit target subset
    la_tonnara: 7,978 total rows; 4,874 rows in relief audit target subset
    santa_manza: 0 total rows in profile; no relief audit row

2025-05 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-05 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    no target rows in the May training table, so no relief coverage can be
    measured for this spot/month.

coverage note:
  Santa Manza target support collapse persists:
    2025-01 santa_manza total rows: 8,216
    2025-02 santa_manza total rows: 770
    2025-03 santa_manza total rows: 0
    2025-04 santa_manza total rows: 0
    2025-05 santa_manza total rows: 0
  This should become a dedicated backfill/source-quality task if Santa Manza
  remains a priority spot for local calibration.

decision:
  May 2025 is accepted for global training, with the persistent Santa Manza
  target-coverage caveat. Continue rebuild and plan a later targeted source
  coverage investigation for Santa Manza.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, June 2025

```text
date: 2026-06-26 21:26-21:30 CEST
context:
  June 2025 initially had a stale profile from 15:50 CEST. Validation waited
  until the fresh profile appeared at 21:29 CEST. The rebuild then advanced to
  2025-07.

machine state:
  memory after June export:
    total: 15 GiB
    available: 13 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-06 table:
  rows: 56,711
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,400
    windsup_public_spot_history: 48,311
  rows_by_lead:
    +15: 7,315
    +30: 7,267
    +45: 7,219
    +60: 9,278
    +120: 9,123
    +180: 8,898
    +360: 7,611
  rows_by_critical_spot:
    balistra: 8,222 total rows; 4,699 rows in relief audit target subset
    la_tonnara: 7,727 total rows; 4,706 rows in relief audit target subset
    santa_manza: 5,758 total rows; 3,410 rows in relief audit target subset

2025-06 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-06 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature: 100.0%
    relief wind: 96.595%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

coverage note:
  Santa Manza target support returns in June:
    2025-02 santa_manza total rows: 770
    2025-03 santa_manza total rows: 0
    2025-04 santa_manza total rows: 0
    2025-05 santa_manza total rows: 0
    2025-06 santa_manza total rows: 5,758
  This supports the hypothesis of a source-coverage gap during 2025-02/05
  rather than a permanent spot/pipeline mapping failure.

decision:
  June 2025 is accepted. Continue rebuild. Later diagnostics should inspect
  whether Santa Manza's 2025-02/05 gap materially affects spot calibration or
  aggregate RMSE; June coverage itself is usable.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, July 2025

```text
date: 2026-06-26 21:31-21:38 CEST
context:
  July 2025 initially had a stale profile from 15:56 CEST. Validation waited
  until the fresh profile appeared at 21:35 CEST. The rebuild then advanced to
  2025-08.

machine state:
  memory after July export:
    total: 15 GiB
    available: 14 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-07 table:
  rows: 61,151
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,673
    windsup_public_spot_history: 52,478
  rows_by_lead:
    +15: 7,962
    +30: 7,915
    +45: 7,870
    +60: 9,994
    +120: 9,814
    +180: 9,550
    +360: 8,046
  rows_by_critical_spot:
    balistra: 8,575 total rows; 4,900 rows in relief audit target subset
    la_tonnara: 8,055 total rows; 4,900 rows in relief audit target subset
    santa_manza: 8,279 total rows; 4,900 rows in relief audit target subset

2025-07 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-07 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

coverage note:
  Santa Manza has fully recovered in July after the February-May target support
  gap:
    2025-05 santa_manza total rows: 0
    2025-06 santa_manza total rows: 5,758
    2025-07 santa_manza total rows: 8,279

decision:
  July 2025 is accepted as a healthy dense summer month. Continue rebuild and
  later isolate the Santa Manza gap as a period-specific source/backfill issue.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, August 2025

```text
date: 2026-06-26 21:39-21:45 CEST
context:
  August 2025 initially had a stale profile from 16:03 CEST. Validation waited
  until the fresh profile appeared at 21:42 CEST. The rebuild then advanced to
  2025-09.

machine state:
  memory after August export:
    total: 15 GiB
    available: 14 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-08 table:
  rows: 59,458
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,680
    windsup_public_spot_history: 50,778
  rows_by_lead:
    +15: 7,775
    +30: 7,708
    +45: 7,647
    +60: 9,761
    +120: 9,550
    +180: 9,255
    +360: 7,762
  rows_by_critical_spot:
    balistra: 8,391 total rows; 4,827 rows in relief audit target subset
    la_tonnara: 7,878 total rows; 4,804 rows in relief audit target subset
    santa_manza: 8,073 total rows; 4,804 rows in relief audit target subset

2025-08 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-08 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  August 2025 is accepted as another healthy dense summer month. Continue
  rebuild toward 2026-06; no model score comparison until the full rebuilt range
  is available.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, September 2025

```text
date: 2026-06-26 21:46-21:50 CEST
context:
  September 2025 initially had a stale profile from 16:08 CEST. Validation
  waited until the fresh profile appeared at 21:48 CEST. The rebuild then
  advanced to 2025-10.

machine state:
  memory after September export:
    total: 15 GiB
    available: 13 GiB
    swap free: 15 GiB
  no post-relief score yet:
    full rebuild still running; watcher has not started training.

2025-09 table:
  rows: 55,853
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,356
    windsup_public_spot_history: 47,497
  rows_by_lead:
    +15: 7,224
    +30: 7,171
    +45: 7,131
    +60: 9,181
    +120: 9,004
    +180: 8,763
    +360: 7,379
  rows_by_critical_spot:
    balistra: 8,244 total rows; 4,710 rows in relief audit target subset
    la_tonnara: 7,338 total rows; 4,447 rows in relief audit target subset
    santa_manza: 7,961 total rows; 4,710 rows in relief audit target subset

2025-09 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-09 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  September 2025 is accepted. Continue rebuild toward 2026-06 and wait for the
  watcher to train/evaluate only after the complete rebuilt range is available.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, October 2025 in progress

```text
date: 2026-06-26 21:52-21:54 CEST
context:
  The rebuild has advanced from 2025-09 to 2025-10. The October JSONL table is
  freshly rebuilt, but the parquet export/profile are not fresh yet, so October
  must not be counted as audited or accepted at this checkpoint.

machine state:
  memory:
    total: 15 GiB
    available: 11-14 GiB during export
    swap free: about 15 GiB
  active process:
    export_training_table_parquet.py for residual_windsup_sst_prev_regime_v1_2025_10
    RSS observed around 5.0-5.7 GiB
    CPU observed at 100%

2025-10 partial table state:
  fresh JSONL:
    training_rows.jsonl mtime: 2026-06-26 21:51 CEST
    size: about 2.9 GiB
    row_count from pipeline/evaluation: 58,650
  stale parquet/profile still present:
    training_rows.parquet mtime: 2026-06-26 16:14 CEST
    parquet_export_profile.json mtime: 2026-06-26 16:14 CEST

decision:
  Do not audit 2025-10 yet. Wait for parquet_export_profile.json to be freshly
  rewritten, then run the feature-family audit and critical relief coverage
  audit before accepting the month.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, October 2025 accepted

```text
date: 2026-06-26 21:54-21:56 CEST
context:
  The October 2025 parquet export completed after the in-progress checkpoint.
  The fresh parquet_export_profile.json mtime is 2026-06-26 21:54 CEST, so the
  stale 16:14 CEST artifact has been replaced.

2025-10 table:
  rows: 58,650
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,675
    windsup_public_spot_history: 49,975
  rows_by_lead:
    +15: 7,594
    +30: 7,548
    +45: 7,502
    +60: 9,625
    +120: 9,425
    +180: 9,180
    +360: 7,776
  rows_by_critical_spot:
    balistra: 8,505 total rows; 4,866 rows in relief audit target subset
    la_tonnara: 7,081 total rows; 4,327 rows in relief audit target subset
    santa_manza: 8,266 total rows; 4,892 rows in relief audit target subset

2025-10 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-10 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  October 2025 is accepted. The rebuild has advanced to 2025-11. Still no
  post-relief model score; wait for the complete rebuilt range and watcher
  benchmark before comparing RMSE.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, November 2025 accepted

```text
date: 2026-06-26 22:00-22:02 CEST
context:
  November 2025 completed after the 2025-10 checkpoint. The fresh
  parquet_export_profile.json mtime is 2026-06-26 22:00 CEST. The rebuild then
  advanced to 2025-12.

2025-11 table:
  rows: 49,344
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,372
    windsup_public_spot_history: 40,972
  rows_by_lead:
    +15: 6,199
    +30: 6,170
    +45: 6,149
    +60: 8,224
    +120: 8,084
    +180: 7,864
    +360: 6,654
  rows_by_critical_spot:
    balistra: 8,295 total rows; 4,740 rows in relief audit target subset
    la_tonnara: 7,814 total rows; 4,740 rows in relief audit target subset
    santa_manza: 8,009 total rows; 4,740 rows in relief audit target subset

2025-11 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-11 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  November 2025 is accepted. The rebuild has advanced to 2025-12. Still no
  post-relief model score; continue waiting for the complete rebuilt range and
  watcher benchmark before comparing RMSE.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, December 2025 accepted

```text
date: 2026-06-26 22:06-22:09 CEST
context:
  December 2025 completed after the 2025-11 checkpoint. The fresh
  parquet_export_profile.json mtime is 2026-06-26 22:06 CEST. The rebuild then
  advanced to 2026-01.

2025-12 table:
  rows: 55,693
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,668
    windsup_public_spot_history: 47,025
  rows_by_lead:
    +15: 7,000
    +30: 6,994
    +45: 6,987
    +60: 9,147
    +120: 9,039
    +180: 8,840
    +360: 7,686
  rows_by_critical_spot:
    balistra: 8,575 total rows; 4,900 rows in relief audit target subset
    la_tonnara: 7,962 total rows; 4,840 rows in relief audit target subset
    santa_manza: 7,735 total rows; 4,591 rows in relief audit target subset

2025-12 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2025-12 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  December 2025 is accepted. The rebuild has advanced to 2026-01. Still no
  post-relief model score; continue waiting for the complete rebuilt range and
  watcher benchmark before comparing RMSE.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, January 2026 accepted

```text
date: 2026-06-26 22:12-22:13 CEST
context:
  January 2026 completed after the 2025-12 checkpoint. The fresh
  parquet_export_profile.json mtime is 2026-06-26 22:12 CEST. The rebuild then
  advanced to 2026-02.

2026-01 table:
  rows: 52,355
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,517
    windsup_public_spot_history: 43,838
  rows_by_lead:
    +15: 6,528
    +30: 6,526
    +45: 6,521
    +60: 8,650
    +120: 8,552
    +180: 8,361
    +360: 7,217
  rows_by_critical_spot:
    balistra: 6,071 total rows; 3,484 rows in relief audit target subset
    la_tonnara: 7,906 total rows; 4,805 rows in relief audit target subset
    santa_manza: 7,712 total rows; 4,561 rows in relief audit target subset

2026-01 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2026-01 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature/delta temperature: 100.0%
    relief wind/delta wind: 99.082%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature/delta temperature: 100.0%
    relief wind/delta wind: 99.211%

decision:
  January 2026 is accepted. The small relief wind gaps on Balistra and Santa
  Manza are noted but not blocking: global relief station assignment is correct,
  temperature is complete, and wind coverage remains above 99%. The rebuild has
  advanced to 2026-02. Still no post-relief model score; continue waiting for
  the complete rebuilt range and watcher benchmark before comparing RMSE.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, February 2026 accepted

```text
date: 2026-06-26 22:17-22:18 CEST
context:
  February 2026 completed after the 2026-01 checkpoint. The fresh
  parquet_export_profile.json mtime is 2026-06-26 22:17 CEST. The rebuild then
  advanced to 2026-03.

2026-02 table:
  rows: 40,818
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 7,701
    windsup_public_spot_history: 33,117
  rows_by_lead:
    +15: 4,979
    +30: 4,971
    +45: 4,963
    +60: 6,882
    +120: 6,790
    +180: 6,614
    +360: 5,619
  rows_by_critical_spot:
    balistra: 0/null total rows in parquet profile
    la_tonnara: 6,862 total rows; 4,185 rows in relief audit target subset
    santa_manza: 7,299 total rows; 4,336 rows in relief audit target subset

2026-02 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2026-02 critical relief coverage:
  balistra:
    no target rows in this month; not audited for relief coverage
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    direct global relief wind/temp/deltas: 98.375%
    aggregate relief wind/temp counts: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  February 2026 is accepted with a source-coverage caveat: Balistra has no
  target rows in the rebuilt training profile for this month, so it contributes
  nothing to February validation/training. La Tonnara has a small direct relief
  gap but aggregate relief context remains complete. The rebuild has advanced
  to 2026-03. Still no post-relief model score; continue waiting for the
  complete rebuilt range and watcher benchmark before comparing RMSE.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, March 2026 accepted

```text
date: 2026-06-26 22:23-22:24 CEST
context:
  March 2026 completed after the 2026-02 checkpoint. The fresh
  parquet_export_profile.json mtime is 2026-06-26 22:23 CEST. The rebuild then
  advanced to 2026-04.

2026-03 table:
  rows: 55,332
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,675
    windsup_public_spot_history: 46,657
  rows_by_lead:
    +15: 6,933
    +30: 6,930
    +45: 6,928
    +60: 9,094
    +120: 9,006
    +180: 8,832
    +360: 7,609
  rows_by_critical_spot:
    balistra: 6,243 total rows; 3,572 rows in relief audit target subset
    la_tonnara: 7,996 total rows; 4,856 rows in relief audit target subset
    santa_manza: 8,208 total rows; 4,868 rows in relief audit target subset

2026-03 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2026-03 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature/delta temperature: 100.0%
    relief wind/delta wind: 99.552%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature/delta temperature: 100.0%
    relief wind/delta wind: 99.671%

decision:
  March 2026 is accepted. The small direct relief wind gaps on Balistra and
  Santa Manza are noted but not blocking: global relief station assignment is
  correct, temperature is complete, and wind coverage remains above 99.5%. The
  rebuild has advanced to 2026-04. Still no post-relief model score; continue
  waiting for the complete rebuilt range and watcher benchmark before comparing
  RMSE.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, April 2026 accepted

```text
date: 2026-06-26 22:29-22:30 CEST
context:
  April 2026 completed after the 2026-03 checkpoint. The fresh
  parquet_export_profile.json mtime is 2026-06-26 22:29 CEST. The rebuild then
  advanced to 2026-05.

2026-04 table:
  rows: 55,108
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,400
    windsup_public_spot_history: 46,708
  rows_by_lead:
    +15: 7,007
    +30: 6,976
    +45: 6,947
    +60: 9,019
    +120: 8,907
    +180: 8,720
    +360: 7,532
  rows_by_critical_spot:
    balistra: 8,295 total rows; 4,740 rows in relief audit target subset
    la_tonnara: 7,801 total rows; 4,732 rows in relief audit target subset
    santa_manza: 8,010 total rows; 4,740 rows in relief audit target subset

2026-04 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2026-04 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature/delta temperature: 100.0%
    relief wind/delta wind: 99.662%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief temperature/delta temperature: 100.0%
    relief wind/delta wind: 99.662%

decision:
  April 2026 is accepted. The small direct relief wind gaps on Balistra and
  Santa Manza are noted but not blocking: global relief station assignment is
  correct, temperature is complete, and wind coverage remains above 99.6%. The
  rebuild has advanced to 2026-05. Still no post-relief model score; continue
  waiting for the complete rebuilt range and watcher benchmark before comparing
  RMSE.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, May 2026 accepted

```text
date: 2026-06-26 22:36-22:38 CEST
context:
  May 2026 completed after the 2026-04 checkpoint. The fresh
  parquet_export_profile.json mtime is 2026-06-26 22:36 CEST. The rebuild then
  advanced to 2026-06, the last requested monthly shard in this rebuild.

2026-05 table:
  rows: 63,223
  columns: 1,343
  rows_by_source_dataset:
    dpclim_station_hourly: 8,680
    windsup_public_spot_history: 54,543
  rows_by_lead:
    +15: 8,251
    +30: 8,194
    +45: 8,149
    +60: 10,274
    +120: 10,093
    +180: 9,829
    +360: 8,433
  rows_by_critical_spot:
    balistra: 8,575 total rows; 4,900 rows in relief audit target subset
    la_tonnara: 8,078 total rows; 4,900 rows in relief audit target subset
    santa_manza: 8,280 total rows; 4,900 rows in relief audit target subset

2026-05 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability

2026-05 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 100.0%
    relief wind/temp/deltas: 100.0%

decision:
  May 2026 is accepted. It is a dense high-quality shard with complete relief
  coverage on the critical spots. The rebuild has advanced to 2026-06, the last
  month before the watcher can start post-relief training and RMSE comparison.
```

## 2026-06-26 - z2 post-reboot rebuild checkpoint, June 2026 accepted

```text
date: 2026-06-26 22:41-22:44 CEST
context:
  June 2026 completed after the 2026-05 checkpoint. The fresh
  parquet_export_profile.json mtime is 2026-06-26 22:41 CEST. This completed
  the requested 2024-01..2026-06 rebuild with the active-window relief station
  fix. The post-rebuild watcher then started the comparable LightGBM benchmark.

2026-06 table:
  rows: 47,171
  columns: 1,453
  rows_by_source_dataset:
    dpclim_station_hourly: 6,360
    windsup_public_spot_history: 40,811
  rows_by_lead:
    +15: 6,067
    +30: 6,046
    +45: 6,027
    +60: 7,594
    +120: 7,524
    +180: 7,408
    +360: 6,505
  rows_by_critical_spot:
    balistra: 6,621 total rows; 3,790 rows in relief audit target subset
    la_tonnara: 6,182 total rows; 3,740 rows in relief audit target subset
    santa_manza: 6,316 total rows; 3,739 rows in relief audit target subset

2026-06 feature-family audit:
  verdict: pass
  stale_shard_count: 0
  missing_shard_count: 0
  required families present:
    previous_run_open_meteo best_match day1/day2 wind
    SST
    global relief availability, wind, temperature
    thermal land-minus-SST
    thermal inland-minus-coastal temperature
    EUMETSAT land surface temperature availability
  note:
    column_count is 1,453, higher than prior months' 1,343. This is accepted
    for now because the required feature families are present and the watcher
    full-range audit passed. It should be reviewed later if the benchmark
    regresses or feature selection becomes unstable.

2026-06 critical relief coverage:
  balistra:
    relief station: 20254006
    global_relief_1_available: 95.778%
    relief wind/temp/deltas: 95.778%
    aggregate relief wind/temp counts: 100.0%
  la_tonnara:
    relief station: 20160001
    global_relief_1_available: 95.722%
    relief wind/temp/deltas: 95.722%
    aggregate relief wind/temp counts: 100.0%
  santa_manza:
    relief station: 20254006
    global_relief_1_available: 95.721%
    relief wind/temp/deltas: 95.721%
    aggregate relief wind/temp counts: 100.0%

full rebuilt range relief audit from watcher:
  range: 2024-01..2026-06
  month_count: 30
  critical rows audited: 398,479
  balistra:
    rows: 135,230
    station: 20254006
    global_relief_1_available: 99.882%
    direct relief wind: 99.692%
  la_tonnara:
    rows: 141,387
    station: 20160001
    global_relief_1_available: 99.774%
    direct relief wind: 99.703%
  santa_manza:
    rows: 121,862
    station: 20254006
    global_relief_1_available: 99.737%
    direct relief wind: 99.668%

post-rebuild watcher:
  rebuild status: 0
  started benchmark:
    tabular_lgbm_225k_prev_relief_active_v1_2024_2025_to_2026_v1
  command:
    train_residual_correction_parquet.py
    split_time_utc: 2026-01-01T00:00:00Z
    max_train_rows: 225000
    max_test_rows: 60000
    model_family: lightgbm
    n_jobs: 1
    eval leads: 15, 30, 45, 60

decision:
  June 2026 is accepted with a local relief-coverage caveat: direct global
  relief availability is around 95.7% on the three critical spots for this
  month, but aggregate relief context is complete and full rebuilt-range
  coverage remains around 99.7-99.9%. The dataset rebuild is now complete and
  post-relief LightGBM benchmarking is running.
```

## 2026-06-26 - post-relief benchmark memory incident and safe runs

```text
date: 2026-06-26 22:42-23:04 CEST
context:
  The post-rebuild watcher started the comparable 225k LightGBM benchmark after
  the full rebuild completed. This was intended to compare against the previous
  225k base run, but the rebuilt feature set and full range exceeded z2 memory.

watcher 225k run:
  run_id: tabular_lgbm_225k_prev_relief_active_v1_2024_2025_to_2026_v1
  command:
    max_train_rows: 225,000
    max_test_rows: 60,000
    lightgbm_max_bin: 127
    n_jobs: 1
  outcome:
    killed by watcher memory guard
    status/code: 92
    peak observed RSS: about 15.47 GiB
    MemAvailable at kill: about 69 MiB
  produced:
    feature_columns.json only
  decision:
    not a valid score; do not compare to previous RMSE.

safe 100k run:
  run_id: tabular_lgbm_100k_prev_relief_active_v1_2024_2025_to_2026_v1
  command changes:
    max_train_rows: 100,000
    max_test_rows: 40,000
    lightgbm_max_bin: 127
    max_iter: 160
  result:
    train rows: 100,000
    test rows: 40,000
    feature columns: 1,132
    raw NWP eval-leads RMSE: 2.163233
    corrected eval-leads RMSE: 1.349176
    corrected eval-leads MAE: 1.021482
    corrected eval-leads bias: 0.114945
    corrected full-test RMSE: 1.468633
    RMSE gain vs raw: 32.146%
  decision:
    valid safe score but worse than current confirmed best RMSE 1.269403.
    It is not directly comparable to the previous 225k/60k run because the row
    caps differ.

safe 150k/bin63 run:
  run_id: tabular_lgbm_150k_bin63_prev_relief_active_v1_2024_2025_to_2026_v1
  command changes:
    max_train_rows: 150,000
    max_test_rows: 40,000
    lightgbm_max_bin: 63
    max_iter: 160
  result:
    train rows: 150,000
    test rows: 40,000
    feature columns: 1,132
    raw NWP eval-leads RMSE: 2.163233
    corrected eval-leads RMSE: 1.348311
    corrected eval-leads MAE: 1.021276
    corrected eval-leads bias: 0.113911
    corrected full-test RMSE: 1.464124
    RMSE gain vs raw: 32.354%
  memory:
    peak observed RSS during polling: about 7.95 GiB
    completed successfully
  decision:
    valid safe score, marginally better than 100k but still worse than current
    confirmed best RMSE 1.269403. The poor result is therefore not explained
    only by using 100k train rows.

manual 225k/bin63 probe:
  run_id: tabular_lgbm_225k_bin63_prev_relief_active_v1_2024_2025_to_2026_v1
  command changes:
    max_train_rows: 225,000
    max_test_rows: 40,000
    lightgbm_max_bin: 63
    max_iter: 160
  outcome:
    manually killed before machine instability
    status: 143
    peak observed RSS: about 15.46 GiB
    MemAvailable before kill: about 54 MiB
  produced:
    feature_columns.json only
  decision:
    invalid/no score. 225k rows are not safe on z2 with the current full
    rebuilt feature matrix, even with max_bin=63 and max_test_rows=40k.

interpretation:
  The active-window relief station fix successfully restores relief coverage,
  but the current post-relief tabular LightGBM score is worse on safe row caps.
  The next iteration should not assume the relief fix improves RMSE by itself.
  Before attempting heavier training again, reduce feature width / memory load
  and run comparable controlled experiments:
    - compare same row caps against pre-relief or previous-prefix tables if
      still available;
    - investigate the 2026 holdout composition and the 2026-06 extra 110 columns;
    - produce holdout predictions/error diagnosis for the 150k/bin63 run;
    - try feature pruning around duplicated/low-coverage context features before
      any bigger 225k run.
```

2026-06-26 23:08 CEST - short-leads run status

Question from Arnaud:
  Have we improved the score?

Current answer:
  Not proven yet. The best confirmed score is still the calibrated run at
  RMSE 1.269403 on wind_mean. The completed post-relief safe runs did not
  improve it:
    - 100k train / 40k test: RMSE 1.349176
    - 150k train / 40k test / max_bin 63: RMSE 1.348311

Run still in progress:
  run_id: tabular_lgbm_150k_shortleads_bin63_relief_active_v1_2024_2025_to_2026_v1
  intent:
    Train and evaluate only short horizons 15/30/45/60 min, because these are
    the production-critical nowcasting horizons and long-lead rows may inject
    noise into the residual model.
  status at 23:07 CEST:
    still running on z2
    process: 84271
    elapsed: 01:46
    RSS: about 8.87 GiB
    MemAvailable: about 5.9 GiB
    no training_results.json yet

Decision:
  Do not claim improvement until this run finishes. If it beats 1.269403, it is
  the first real post-relief improvement. If it stays around 1.34, the next
  direction should be grouped/per-lead modeling or feature pruning, not simply
  adding more rows to the same model shape.

2026-06-26 23:12 CEST - short-leads run completed

Run:
  run_id: tabular_lgbm_150k_shortleads_bin63_relief_active_v1_2024_2025_to_2026_v1
  train rows: 149,780
  test rows: 59,641
  included/evaluated leads: 15, 30, 45, 60 min
  model: LightGBM, max_bin 63, 180 iterations, one global model

Result:
  raw NWP RMSE: 2.174881
  corrected RMSE: 1.316545
  corrected MAE: 0.990837
  corrected bias: 0.097863
  RMSE gain vs raw: 39.466%

By lead:
  +15 min: RMSE 1.146654, MAE 0.864847
  +30 min: RMSE 1.289613, MAE 0.974123
  +45 min: RMSE 1.361360, MAE 1.035984
  +60 min: RMSE 1.419300, MAE 1.065340

Worst corrected spots:
  cap_corse: RMSE 1.682962, count 560
  lfvh: RMSE 1.544904, count 616
  la_tonnara: RMSE 1.534696, count 9,603
  santa_manza: RMSE 1.466636, count 9,408
  la_parata: RMSE 1.412103, count 618
  porto_polo: RMSE 1.328793, count 8,009

Verdict:
  This is a real improvement over the post-relief safe completed runs
  (1.348311 -> 1.316545), but still worse than the best confirmed calibrated
  run at RMSE 1.269403. It does not achieve the 0.9 target.

Next hypothesis:
  Since +15 min is much easier than +45/+60 min, train grouped models by
  lead_time_minutes on the same short-lead row set. This tests whether the
  single global residual function is underfitting horizon-specific error
  dynamics.

2026-06-26 23:12 CEST - launched grouped-by-lead experiment

Run:
  run_id: tabular_lgbm_150k_shortleads_by_lead_bin63_relief_active_v1_2024_2025_to_2026_v1
  hypothesis:
    The residual correction error dynamics differ enough between +15, +30, +45
    and +60 min that independent per-lead models can reduce RMSE versus the
    single global short-leads model.
  command shape:
    same short-lead row filters and LightGBM memory limits as previous run
    plus --fit-group-column lead_time_minutes
    min group train rows: 5,000
    min group test rows: 500
  remote status at launch:
    z2 process: 84503
    log: /srv/data/corsewind/ml_dataset/run_logs/tabular_lgbm_150k_shortleads_by_lead_bin63_relief_active_v1.log
    output: /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_150k_shortleads_by_lead_bin63_relief_active_v1_2024_2025_to_2026_v1

2026-06-26 23:16 CEST - grouped-by-lead result

Result:
  raw NWP RMSE: 2.174881
  corrected RMSE: 1.316851
  corrected MAE: 0.986199
  corrected bias: 0.091906

By lead:
  +15 min: RMSE 1.129144, MAE 0.829047
  +30 min: RMSE 1.290631, MAE 0.972180
  +45 min: RMSE 1.370750, MAE 1.047349
  +60 min: RMSE 1.423394, MAE 1.070296

Verdict:
  Per-lead models improve +15 min versus the single short-leads model
  (1.146654 -> 1.129144), but degrade +45/+60 and slightly degrade the overall
  short-horizon RMSE (1.316545 -> 1.316851). The horizon mixture is not the main
  current bottleneck.

Next hypothesis:
  The RMSE09 gap audit showed that a small share of high-error/high-wind rows
  dominates the remaining MSE. Test sample weighting on observed wind >= 8 m/s
  with the single short-leads model, because this may reduce the worst-case
  thermal/strong-wind misses without changing temporal validation.

2026-06-26 23:16 CEST - launched high-wind weighted short-leads experiment

Run:
  run_id: tabular_lgbm_150k_shortleads_highwind2_bin63_relief_active_v1_2024_2025_to_2026_v1
  hypothesis:
    Weighting rows with observed target wind >= 8 m/s should reduce the
    high-wind thermal misses that dominate the RMSE09 gap, while preserving the
    same temporal split and short-horizon evaluation.
  command shape:
    same as the single short-leads LightGBM run
    plus --target-high-wind-weight-threshold-ms 8
    plus --target-high-wind-weight 2.0
  remote status at launch:
    z2 process: 84731
    log: /srv/data/corsewind/ml_dataset/run_logs/tabular_lgbm_150k_shortleads_highwind2_bin63_relief_active_v1.log
    output: /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_150k_shortleads_highwind2_bin63_relief_active_v1_2024_2025_to_2026_v1

2026-06-26 23:19 CEST - high-wind weighted 150k stopped

Outcome:
  invalid/no score
  status: 143
  produced: feature_columns.json only
  reason:
    During training, MemAvailable dropped to about 44-49 MiB on z2. The run was
    terminated to avoid another OOM crash. Memory returned to about 14 GiB
    available afterward.

  Decision:
  Do not retry high-wind weighting at 150k/60k with the current feature matrix.
  Run a smaller 100k/40k high-wind weighted test first. If it improves the
  comparable 100k baseline, revisit memory reduction before scaling it.

2026-06-26 23:23 CEST - high-wind weighted 100k result

Run:
  run_id: tabular_lgbm_100k_shortleads_highwind2_bin63_relief_active_v1_2024_2025_to_2026_v1
  train rows: 99,857
  test rows: 39,773
  high-wind weighting:
    observed target wind >= 8 m/s
    weight: 2.0
    weighted high-wind train rows: 20,526
    mean train weight: 1.205554

Result:
  raw NWP RMSE: 2.168088
  corrected RMSE: 1.331300
  corrected MAE: 1.004058
  corrected bias: 0.203853

By lead:
  +15 min: RMSE 1.168299, MAE 0.882503
  +30 min: RMSE 1.297362, MAE 0.986372
  +45 min: RMSE 1.378189, MAE 1.052061
  +60 min: RMSE 1.431990, MAE 1.072642

Verdict:
  Safe run, but not an improvement versus the 150k short-leads model
  (1.316545) and still worse than the best confirmed calibrated run
  (1.269403). It also introduces a stronger positive bias, so simple
  high-wind weighting is not enough by itself.

Next check:
  Run the exact 100k/40k short-leads configuration without high-wind weighting
  to isolate whether the weighting helped or hurt relative to a fair 100k
  baseline.

2026-06-26 23:28 CEST - fair 100k short-leads baseline result

Run:
  run_id: tabular_lgbm_100k_shortleads_bin63_relief_active_v1_2024_2025_to_2026_v1
  train rows: 99,857
  test rows: 39,773
  sample weighting: disabled

Result:
  raw NWP RMSE: 2.168088
  corrected RMSE: 1.311982
  corrected MAE: 0.988769
  corrected bias: 0.105563

By lead:
  +15 min: RMSE 1.148032, MAE 0.866329
  +30 min: RMSE 1.280070, MAE 0.972814
  +45 min: RMSE 1.355843, MAE 1.034509
  +60 min: RMSE 1.413906, MAE 1.058431

Comparison:
  100k unweighted short-leads beats 100k high-wind weighted:
    1.311982 vs 1.331300
  Therefore simple high-wind weighting hurts at equal row cap. It increases
  positive bias and does not solve the high-wind misses cleanly.

Diagnostics:
  Saved holdout predictions and error diagnosis:
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_100k_shortleads_bin63_relief_active_v1_2024_2025_to_2026_v1/tabular_holdout_predictions.parquet
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_100k_shortleads_bin63_relief_active_v1_2024_2025_to_2026_v1/tabular_error_diagnosis.json
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_100k_shortleads_bin63_relief_active_v1_2024_2025_to_2026_v1/tabular_error_diagnosis.md

Worst spot/lead groups:
  la_tonnara +60: RMSE 1.730048, count 1,666
  cap_corse +60: RMSE 1.715704, count 392
  la_tonnara +45: RMSE 1.582554, count 1,571
  santa_manza +60: RMSE 1.524740, count 1,571
  santa_manza +45: RMSE 1.507545, count 1,605

Error modes:
  actual wind >= 8 m/s:
    RMSE 1.682640
    bias -0.600834
    interpretation: model still underpredicts strong-wind cases.
  actual wind 0-2 m/s:
    RMSE 1.315350
    bias +0.977094
    interpretation: model overpredicts calm cases.
  raw wind forecast top quartile:
    RMSE 1.636287
    interpretation: highest NWP wind regimes remain the hardest.
  shortwave radiation quartiles:
    highest radiation quartile RMSE 1.194613
    lowest radiation quartile RMSE 1.403816
    interpretation: in this holdout, radiation alone is not the primary
    aggravating factor; wind regime and exposed spots dominate.

Current conclusion:
  We have not improved the historical best confirmed RMSE 1.269403.
  The best post-relief short-lead score observed here is 1.311982 on a 40k test
  sample, but it is not directly comparable to the 1.269403 calibrated run
  because the evaluation sample differs. The next high-value direction is not
  high-wind weighting; it is regime/spot calibration or a two-stage model that
  explicitly avoids compressing extremes.

2026-06-26 23:52 CEST - regime calibration and feature-pruning iteration

Regime calibration test:
  Added script:
    scripts/ml_dataset/calibrate_predictions_by_regime.py
  Purpose:
    Learn a post-calibration correction on a validation period and apply it to
    2026 without using 2026 truth for selection.

Validation setup:
  First-stage validation calibrator:
    run_id: prediction_residual_calibrator_2025q3_to_2025q4_extratrees_scale095_v1
    fit window: 2025-07-01 to 2025-10-01
    validation/eval window: 2025-10-01 to 2026-01-01
    model: extra_trees, correction scale 0.95
  Result on 2025Q4:
    base RMSE: 1.378996
    calibrated RMSE: 1.365992

Regime selection:
  run_id/output: regime_calibration_2025q4_to_2026_v1
  validation-selected best:
    correction: affine
    group: predicted wind bin
    validation RMSE: 1.357757
  2026 evaluation:
    base calibrated RMSE: 1.269403
    regime calibrated RMSE: 1.274574
    gain vs base: -0.407%
  Forced conservative variants:
    global bias: 1.272987
    global affine: 1.272200
    lead bias: 1.273100
    lead affine: 1.272462
    predicted-bin bias: 1.273149
    predicted-bin affine: 1.273178
  Decision:
    Reject this post-hoc regime calibration family for now. It improved 2025Q4
    but degraded 2026 because 2025Q4 had a negative residual bias while 2026
    was already close to zero-bias after the best calibrator. This is not
    temporally stable enough.

Trainer memory fix:
  Patched scripts/ml_dataset/train_residual_correction_parquet.py so
  --feature-allowlist-json is applied before required_columns are read from
  Parquet. Previously the trainer still read the full feature matrix before
  filtering, so allowlists did not prevent OOM.
  Verification:
    python3 -m py_compile on z2 succeeded.

Feature pruning / 225k runs after the fix:
  allowlist top350:
    path: /srv/data/corsewind/ml_dataset/allowlists/top350_shortleads_importance_v1.json
    columns: 348 numeric + 2 categorical
    run_id: tabular_lgbm_225k_shortleads_top350_preread_bin63_relief_active_v1_2024_2025_to_2026_v1
    train rows: 224,688
    test rows: 59,641
    corrected RMSE: 1.321707
    by lead:
      +15: 1.152708
      +30: 1.292850
      +45: 1.368095
      +60: 1.424159

  allowlist top700:
    path: /srv/data/corsewind/ml_dataset/allowlists/top700_shortleads_importance_v1.json
    columns: 697 numeric + 3 categorical
    run_id: tabular_lgbm_225k_shortleads_top700_preread_bin63_relief_active_v1_2024_2025_to_2026_v1
    train rows: 224,688
    test rows: 59,641
    corrected RMSE: 1.319159
    by lead:
      +15: 1.150134
      +30: 1.291183
      +45: 1.365503
      +60: 1.421027

  failed/unusable before trainer patch:
    top350 and top150 225k runs still OOM-stopped because the allowlist was
    applied after reading all columns.

Interpretation:
  The pre-read allowlist fix is important infrastructure progress: 225k
  training can now complete safely with reduced feature sets. But importance
  pruning from the 100k model did not improve RMSE; it likely removes weaker
  contextual features that still help the full model. More rows alone are not
  solving the gap.

Current best after this iteration:
  Still the existing calibrated run:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1
    RMSE 1.269403

Next highest-value directions:
  1. Generate comparable post-relief calibration/evaluation predictions for the
     full-ish feature set enabled by the pre-read allowlist fix, then compare on
     identical overlapping rows with the old best.
  2. Investigate whether the new relief-active rebuild changes target row mix
     and makes the 59k short-lead evaluation materially harder than the old 31k
     calibrated benchmark.
  3. If feature pruning continues to hurt, move away from scalar residual-only
     models toward quantile/distributional correction or a mixture-of-experts
     trained with a validation gate, because the remaining error is dominated
     by tail behavior rather than mean bias.

2026-06-27 00:15 CEST - comparable top700 post-relief calibration chain

Purpose:
  Test whether the post-relief/top700 base can improve once passed through the
  same second-stage calibrator pattern as the current best historical run.

Inputs:
  2026 evaluation base:
    run_id: tabular_lgbm_225k_shortleads_top700_preread_bin63_relief_active_v1_2024_2025_to_2026_v1
    holdout predictions:
      /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_shortleads_top700_preread_bin63_relief_active_v1_2024_2025_to_2026_v1/tabular_holdout_predictions.parquet
    base RMSE: 1.319159

  2025H2 calibration base:
    run_id: tabular_lgbm_calbase_225k_shortleads_top700_preread_2024_to_2025h2_v1
    split: 2025-07-01T00:00:00Z
    train rows: 224,618
    holdout/calibration rows: 59,957
    holdout predictions:
      /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_calbase_225k_shortleads_top700_preread_2024_to_2025h2_v1/calibration_predictions_2025h2.parquet
    base RMSE on 2025H2: 1.326099

Second-stage calibrator:
  run_id: prediction_residual_calibrator_top700_2025h2_to_2026_extratrees_autoscale_v1
  model: extra_trees
  fit calibration window: 2025-07-01 to 2026-01-01
  scale validation window: 2025-10-01 to 2026-01-01
  selected scale: 0.1
  output:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_top700_2025h2_to_2026_extratrees_autoscale_v1/calibrated_predictions_2026.parquet

Result on 2026:
  base RMSE: 1.319159
  calibrated RMSE: 1.316977
  gain vs base: 0.165%
  gap to 0.9: 0.416977

Verdict:
  Not an improvement over the current best RMSE 1.269403. The top700
  post-relief chain calibrates only slightly and remains materially worse on its
  full 2026 evaluation sample.

Overlap comparison with current best:
  current best:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1
  overlap rows by issue_time_utc + spot_id + lead_time_minutes:
    old rows: 31,429
    top700 rows: 59,641
    intersection: 18,162
  on intersection:
    old calibrated RMSE: 1.306305
    top700 calibrated RMSE: 1.307041
    old uncalibrated corrected RMSE: 1.317238
    top700 uncalibrated corrected RMSE: 1.309026
  by lead on intersection:
    +15 old/top700 calibrated: 1.142967 / 1.146031
    +30 old/top700 calibrated: 1.274289 / 1.269908
    +45 old/top700 calibrated: 1.373961 / 1.367340
    +60 old/top700 calibrated: 1.389305 / 1.397525
  spot-level changes on intersection:
    improved by top700: balistra, la_tonnara, santa_manza, figari_eole,
      porticcio, piantarella
    degraded notably by top700: porto_polo (+0.0627 RMSE), lfvh (+0.0408 RMSE)

Simple overlap ensemble probe:
  weighted average = (1-w) * old_calibrated + w * top700_calibrated
  best tested weight: w=0.5
  overlap RMSE:
    old alone: 1.306305
    top700 alone: 1.307041
    50/50 ensemble: 1.290386
  Interpretation:
    The old and top700 models have complementary errors on common rows. This is
    not yet a locked production score because the overlap is only 18,162 rows,
    but it strongly supports a proper non-leaky ensemble/stacking experiment as
    the next modeling direction.

Next step:
  Build a leakage-safe ensemble evaluation where both base models produce
  predictions on the same locked 2026 rows. Select ensemble weights or a routing
  gate only on a 2025 validation window, then evaluate once on 2026. This is more
  promising than further univariate regime calibration because the complementarity
  is visible without using 2026 labels for training.

## 2026-06-27 - point d'etape score apres reboot z2

Machine:
  z2 rebooted cleanly.
  RAM available at check time: about 14 GiB.
  No training/calibration process running.

Score status:
  Current locked best remains:
    run_id: prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1
    RMSE: 1.269403
    MAE: 0.930905
    bias: 0.012952
    rows: 31,429
    gap to RMSE 0.9: 0.369403

  Best by lead:
    +15 min: RMSE 1.102812
    +30 min: RMSE 1.251636
    +45 min: RMSE 1.313328
    +60 min: RMSE 1.360039

Recent experiments:
  The active-relief rebuild and top700 pre-read feature selection improved data
  coverage and solved the previous memory path, but did not improve the locked
  score. The comparable top700 calibrated chain reached RMSE 1.316977, worse
  than 1.269403.

Conclusion:
  We have not improved the best validated RMSE yet. The latest useful signal is
  complementarity between the old best and the top700 model: a diagnostic 50/50
  blend on overlapping 2026 rows reached RMSE 1.290386 versus 1.306305 for the
  old model on that same overlap. This is not a production score because the
  weight was probed on the evaluation labels, but it points to the next rigorous
  experiment: select ensemble weights on 2025 only, then evaluate once on 2026.

## 2026-06-27 - leakage-safe old-best/top700 weighted ensemble

Implemented:
  script:
    scripts/ml_dataset/weighted_prediction_ensemble.py
  purpose:
    Merge two prediction files on issue_time_utc + spot_id + lead_time_minutes,
    choose blend weights only on a calibration period, then evaluate the selected
    strategy once on a separate evaluation period.

Validation predictions generated for top700:
  run_id: prediction_residual_calibrator_top700_2025q3_to_2025q4_extratrees_scale010_v1
  calibration input:
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_calbase_225k_shortleads_top700_preread_2024_to_2025h2_v1/calibration_predictions_2025h2.parquet
  calibration fit window: 2025-07-01 to 2025-10-01
  validation/eval window: 2025-10-01 to 2026-01-01
  model: ExtraTrees, fixed correction scale 0.1
  validation rows: 28,534
  base RMSE: 1.394615
  calibrated RMSE: 1.394426

Weighted ensemble:
  run_id: weighted_ensemble_oldbest_top700_cal2025q4_to_2026_v1
  output:
    /srv/data/corsewind/ml_dataset/benchmarks/weighted_ensemble_oldbest_top700_cal2025q4_to_2026_v1/ensemble_results.json
    /srv/data/corsewind/ml_dataset/benchmarks/weighted_ensemble_oldbest_top700_cal2025q4_to_2026_v1/ensemble_predictions_2026.parquet
  calibration base:
    prediction_residual_calibrator_2025q3_to_2025q4_extratrees_scale095_v1
  calibration alt:
    prediction_residual_calibrator_top700_2025q3_to_2025q4_extratrees_scale010_v1
  evaluation base:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1
  evaluation alt:
    prediction_residual_calibrator_top700_2025h2_to_2026_extratrees_autoscale_v1
  common calibration rows: 8,974
  common evaluation rows: 18,162
  weight grid: 0.00 to 1.00 by 0.05

Leakage-safe selected result:
  selected strategy by 2025 calibration: spot_lead
  2026 overlap base RMSE: 1.306305
  2026 overlap alt RMSE: 1.307041
  2026 overlap ensemble RMSE: 1.291417
  gain vs base on overlap: 1.14%
  gap to RMSE 0.9: 0.391417

Strategy sensitivity:
  global:
    validation RMSE: 1.389949
    2026 RMSE: 1.290587
  lead:
    validation RMSE: 1.389277
    2026 RMSE: 1.290923
  spot_lead:
    validation RMSE: 1.384919
    2026 RMSE: 1.291417

Interpretation:
  The ensemble gain is real under a no-2026-label selection protocol, but it is
  small and only on the model-overlap subset. The fact that spot_lead wins on
  validation while global is slightly better on 2026 suggests that granular
  weights overfit the 2025Q4 validation sample. This does not beat the current
  locked best full score of 1.269403, so the RMSE < 0.9 goal remains unachieved.

Decision:
  Keep weighted ensembling as a secondary stabilizer, not the main path to 0.9.
  The next material improvement likely needs either stronger target data / longer
  full-history coverage, or a model that directly learns high-wind and thermal
  onset regimes rather than a shallow blend of two similar tabular predictors.

## 2026-06-27 - RMSE 0.9 reduction target decomposition

Implemented:
  script:
    scripts/ml_dataset/compute_rmse09_reduction_targets.py
  purpose:
    Given an RMSE gap audit, compute how low a subgroup's RMSE would need to be
    if that subgroup alone were improved enough to bring global RMSE to 0.9.

Current-best audit regenerated on z2:
  run:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1
  audit:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1/rmse09_gap_audit_current_best_v2.json
  reduction targets:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1/rmse09_reduction_targets_v2.json
  current RMSE: 1.269403
  MSE reduction needed for RMSE 0.9: 49.733%
  row-wise oracle among current available variants: 1.157878

Current-best subgroup targets:
  lead_45_60:
    current subgroup RMSE: 1.340847
    required subgroup RMSE if only this subgroup improves: 0.570735
    required subgroup RMSE reduction: 57.435%
  critical_spots:
    current subgroup RMSE: 1.43294
    required subgroup RMSE if only this subgroup improves: 0.512178
    required subgroup RMSE reduction: 64.257%
  actual_8plus:
    current subgroup RMSE: 1.746568
    cannot reach global 0.9 by correcting only actual>=8 m/s; non-group RMSE
    remains 1.049306.
  critical_spots_or_lead_45_60:
    current subgroup RMSE: 1.335018
    required subgroup RMSE: 0.849523
    required subgroup RMSE reduction: 36.366%

Ensemble overlap audit:
  run:
    weighted_ensemble_oldbest_top700_cal2025q4_to_2026_v1
  audit:
    /srv/data/corsewind/ml_dataset/benchmarks/weighted_ensemble_oldbest_top700_cal2025q4_to_2026_v1/ensemble_rmse09_gap_audit.json
  reduction targets:
    /srv/data/corsewind/ml_dataset/benchmarks/weighted_ensemble_oldbest_top700_cal2025q4_to_2026_v1/ensemble_rmse09_reduction_targets.json
  current overlap RMSE: 1.291417
  MSE reduction needed for RMSE 0.9: 51.432%
  row-wise oracle among old_best/top700/ensemble: 1.178522

Interpretation:
  The required improvements are too large for another shallow global calibration
  pass. Even oracle selection among existing model variants does not approach
  0.9. A credible path now requires a new hard-regime specialist and/or more
  informative input data for critical spots and lead 45/60 thermal/high-wind
  cases.

Decision artifact:
  docs/ml_nowcasting/rmse_0_9_path_to_target.md

## 2026-06-27 - hard-regime third-stage specialist tests

Implemented:
  script:
    scripts/ml_dataset/train_hard_regime_specialist.py
  purpose:
    Train a third-stage residual correction specialist on hard regimes only,
    using forecast-time hard masks and selecting correction scale on a 2025
    validation window before evaluating 2026.

Protocol:
  calibration predictions:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025q3_to_2025q4_extratrees_scale095_v1/calibrated_predictions_2025q4.parquet
  evaluation predictions:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1/calibrated_predictions_2026.parquet
  base prediction column:
    calibrated_wind_mean_ms
  fit window:
    2025-10-01 to 2025-12-01
  scale validation window:
    2025-12-01 to 2026-01-01
  locked evaluation:
    2026 predictions from current best run

Hard mask variants:
  broad:
    critical spots OR lead>=45
    critical spots: la_tonnara, santa_manza, balistra, cap_corse, la_parata
  lead-only:
    lead>=45
  narrow:
    critical spots AND lead>=45

Results:
  hgb broad:
    run_id: hard_regime_specialist_oldbest_q4_to_2026_hgb_v1
    fit rows: 8,078
    validation hard rows: 4,108
    eval hard rows: 23,748
    validation base RMSE: 1.194354
    validation selected RMSE: 1.192261
    selected scale: 0.3
    2026 base RMSE: 1.269403
    2026 specialist RMSE: 1.270628
    2026 hard base/specialist RMSE: 1.335017 / 1.336558
    decision: reject, validation gain did not generalize.

  lightgbm broad:
    run_id: hard_regime_specialist_oldbest_q4_to_2026_lgbm_v1
    validation base RMSE: 1.194354
    validation selected RMSE: 1.189799
    selected scale: 0.45
    2026 specialist RMSE: 1.274176
    2026 hard base/specialist RMSE: 1.335017 / 1.34102
    decision: reject, stronger validation gain but worse 2026 degradation.

  extra_trees broad:
    run_id: hard_regime_specialist_oldbest_q4_to_2026_et_v1
    validation base RMSE: 1.194354
    validation selected RMSE: 1.188941
    selected scale: 0.75
    2026 specialist RMSE: 1.276871
    2026 hard base/specialist RMSE: 1.335017 / 1.344409
    decision: reject, clear overfit.

  hgb lead>=45:
    run_id: hard_regime_specialist_oldbest_q4_to_2026_hgb_lead45_60_v1
    fit rows: 5,659
    validation hard rows: 2,839
    validation base RMSE: 1.191443
    validation selected RMSE: 1.190703
    selected scale: 0.25
    2026 specialist RMSE: 1.270308
    2026 hard base/specialist RMSE: 1.340847 / 1.34242
    decision: reject.

  hgb critical AND lead>=45:
    run_id: hard_regime_specialist_oldbest_q4_to_2026_hgb_critical_and_45_v1
    fit rows: 2,774
    validation hard rows: 1,351
    selected scale: 0.0
    2026 specialist RMSE: 1.269403
    decision: validation gate correctly rejects applying the specialist.

Interpretation:
  With current features and only Q4 2025 as honest third-stage calibration data,
  hard-regime specialists do not generalize to 2026. The broad models show small
  validation improvements, but all degrade locked 2026. This strengthens the
  conclusion that the path to RMSE 0.9 needs more informative/longer hard-regime
  training data and new regime features, not another residual correction layer
  on the same feature family.

## 2026-06-27 - feature family coverage gap audit

Implemented:
  script:
    scripts/ml_dataset/audit_feature_family_coverage.py
  purpose:
    Audit whether prediction artifacts contain the physical feature families
    needed for hard-regime thermal/wind correction, and measure row coverage on
    both all rows and the hard subset.

Current-best audit:
  run:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1
  output:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1/feature_family_coverage_audit.json
  rows: 31,429
  hard rows: 23,748
  RMSE: 1.269403
  hard RMSE: 1.335017
  missing required concepts:
    - land/sea thermal delta
    - air/sea thermal delta
    - land/air thermal delta
    - upwind station aggregates
    - coastal/inland and coastal/relief explicit thermal-pressure deltas
    - vertical temperature/humidity/motion/geopotential profile features

Top700/post-relief audit:
  run:
    prediction_residual_calibrator_top700_2025h2_to_2026_extratrees_autoscale_v1
  output:
    /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_top700_2025h2_to_2026_extratrees_autoscale_v1/feature_family_coverage_audit.json
  rows: 59,641
  hard rows: 44,908
  RMSE: 1.316977
  hard RMSE: 1.37558
  newly present vs current-best:
    - thermal_land_minus_sst_c
    - thermal_air_minus_sst_c
    - thermal_land_minus_air_c
    - upwind-weighted station aggregates
    - coastal/inland and coastal/relief thermal-pressure deltas
    - previous-run features
  still missing:
    - vertical temperature profile
    - vertical humidity profile
    - vertical motion profile
    - geopotential thickness/profile features

Availability profile with ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset:
  output:
    /srv/data/corsewind/ml_dataset/source_inventories/data_availability_profile_latest.json
  Copernicus SST:
    rows: 204,150
    range: 2024-01-01 to 2026-06-24
  EUMETSAT Cloud Type/LST/Global Instability:
    rows: 3,825 / 3,695 / 3,775
    range starts around 2026-06-22
  AROME vertical profiles:
    rows: 725
    spots: 25
    range: 2026-06-24T09:00:00Z to 2026-06-26T22:00:00Z
    levels: 1000, 950, 925, 900, 850 hPa
    derived features available in collector:
      geopotential_thickness_1000_850_m,
      low_level_inversion_strength_c,
      relative_humidity_mean_1000_850_pct,
      temperature_lapse_rate_1000_850_c_per_km,
      vertical_velocity_pressure_850_pa_s

Interpretation:
  The newer feature pipeline is moving in the correct physical direction, but
  the current validated 2025->2026 training/evaluation stack still lacks the
  vertical atmospheric column signal. The code can collect and integrate it, but
  the available history is currently only a few days. Therefore the next data
  gate is historical vertical profile availability/backfill, not another model
  sweep.

Decision artifact:
  docs/ml_nowcasting/rmse_0_9_feature_gap_audit.md

## 2026-06-27 - historical vertical profile backfill feasibility

Question:
  Can we backfill the vertical atmospheric column for 2024-2026, or are native
  Meteo-France AROME WCS profiles only usable from now onward?

Finding:
  Native Meteo-France AROME WCS vertical profiles are implemented and useful,
  but the current z2 inventory only proves recent local coverage:
  2026-06-24 to 2026-06-26, 725 rows. That is not enough to train/evaluate
  the locked 2025 -> 2026 RMSE target.

Open-Meteo pressure-level smoke test:
  run:
    source_inventories/open_meteo_pressure_level_smoke
  model:
    meteofrance_arome_france
  spot/date:
    balistra, 2025-06-15
  result:
    24 hourly rows
    all requested pressure-level fields returned 24/24 non-null values
  tested fields:
    temperature, relative_humidity, geopotential_height, wind_speed,
    wind_direction at 1000/950/925/900/850 hPa.

Implemented:
  - `scripts/ml_dataset/run_training_backfill_pipeline.py` now accepts
    `--open-meteo-hourly`, so the normal backfill runner can request pressure
    levels.
  - `scripts/ml_dataset/build_residual_training_table.py` now derives
    Open-Meteo vertical features when pressure-level variables are present:
      geopotential thickness 1000-850,
      temperature lapse rate 1000-850,
      low-level inversion strength,
      mean humidity 1000-850,
      humidity delta 1000-850,
      wind-speed shear 1000-850,
      wind-direction shear 1000-850.
  - `scripts/ml_dataset/audit_feature_family_coverage.py` now recognizes
    `open_meteo_vertical_*` as vertical profile coverage.
  - `configs/ml_backfill_sources.json` records the pressure-level extension and
    marks native WCS historical depth as unproven.

Decision:
  Use Open-Meteo Historical Forecast pressure-level fields as the practical
  historical vertical-profile backfill for 2024-01-02 -> 2026-06-23. Keep native
  Meteo-France WCS vertical profiles for forward/recent collection and native
  validation, but do not block RMSE09 progress on native multi-year WCS history.

Decision artifact:
  docs/ml_nowcasting/vertical_profile_backfill_feasibility.md

Backfill launched on z2:
  command family:
    collect_open_meteo_historical_forecast.py
  date range:
    2024-01-02 -> 2026-06-23
  model:
    meteofrance_arome_france
  mode:
    include context spots, max 7 days/request, no skip existing complete rows
  fields:
    existing surface/default fields plus pressure-level temperature, humidity,
    geopotential height, wind speed and wind direction at 1000/950/925/900/850
    hPa
  pid file:
    /srv/data/corsewind/ml_dataset/backfill_logs/open_meteo_pressure_levels_20240102_20260623.pid
  log:
    /srv/data/corsewind/ml_dataset/backfill_logs/open_meteo_pressure_levels_20240102_20260623.log
  immediate status:
    process started as PID 90279 and was still running after launch check with
    low memory usage.

Acceleration update:
  The first single-process job was progressing correctly but too slowly because
  it used 7-day chunks over all spots sequentially. It was stopped cleanly and
  replaced with five non-overlapping date partitions, so the processes do not
  write the same `date=YYYY-MM-DD/forecast.jsonl` files:
    - 2024h1: 2024-01-02 -> 2024-06-30, PID 90502
    - 2024h2: 2024-07-01 -> 2024-12-31, PID 90504
    - 2025h1: 2025-01-01 -> 2025-06-30, PID 90506
    - 2025h2: 2025-07-01 -> 2025-12-31, PID 90508
    - 2026h1: 2026-01-01 -> 2026-06-23, PID 90510
  Each partition uses 31-day chunks, includes context spots, and forces
  `--no-skip-existing-complete` so existing surface-only Open-Meteo rows are
  enriched with pressure-level fields.
  Progress audit after relaunch:
    required hPa-complete rows: 5,992 / 542,400 (1.105%)
    active spots: currently filling alistro across all date partitions
    all five processes still running with low memory usage

Rebuild watcher:
  script:
    scripts/ml_dataset/z2_watch_open_meteo_pressure_then_rebuild.sh
  z2 status:
    /srv/data/corsewind/ml_dataset/backfill_logs/open_meteo_pressure_rebuild_watcher.status
  z2 log:
    /srv/data/corsewind/ml_dataset/backfill_logs/open_meteo_pressure_rebuild_watcher.log
  actual watcher PID:
    90900
  behavior:
    waits for the five date-partitioned pressure-level backfills, runs the
    Open-Meteo hPa coverage audit, and launches
    `residual_backfill_2024_2026_short_hpa_v1` if observed-row hPa coverage is
    at least 99.5%. The updated watcher then launches the first strict
    2024-2025 -> 2026 LightGBM benchmark on short horizons:
      tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1
    Benchmark settings:
      split: 2026-01-01T00:00:00Z
      leads: 15/30/45/60
      model: LightGBM
      max_train_rows: 300,000
      max_test_rows: 120,000
      target: residual_wind_mean_ms only
    It also writes:
      tabular_rmse09_audit.json/md
      tabular_error_analysis.json/md
      tabular_holdout_predictions.parquet
  threshold note:
    A stricter 99.9% observed-row gate was relaxed to 99.5% after alistro showed
    54 isolated missing hPa rows across four dates, while otherwise returning
    complete pressure-level coverage. Those sparse holes are acceptable because
    the tabular pipeline imputes missing numeric fields, and blocking the rebuild
    on a few Open-Meteo null hours would not improve the RMSE experiment.

Concurrency adjustment:
  The monthly 30-worker launch was too aggressive for Open-Meteo. It improved
  throughput but produced HTTP 429 `Too many concurrent requests`, timeouts, and
  connection resets in multiple month logs. That run was stopped.

  Stable active launch:
    segments: 8 non-overlapping date ranges
    PIDs:
      seg01 93297 2024-01-02 -> 2024-04-30
      seg02 93298 2024-05-01 -> 2024-08-31
      seg03 93299 2024-09-01 -> 2024-12-31
      seg04 93300 2025-01-01 -> 2025-04-30
      seg05 93301 2025-05-01 -> 2025-08-31
      seg06 93302 2025-09-01 -> 2025-12-31
      seg07 93303 2026-01-01 -> 2026-03-31
      seg08 93304 2026-04-01 -> 2026-06-23
    watcher PID: 93305
    request_sleep_sec: 0.25
    timeout_sec: 150
  Stability check:
    workers_running: 8
    seg_error_logs: 0
    hPa-complete observed rows: 180,996
    observed-row hPa coverage: 33.377%

Watcher correction:
  The watcher now monitors only `open_meteo_pressure_levels_seg*.pid`, not all
  historical pressure backfill pid files. This prevents stale pid files from
  old 5-worker, 10-worker, or failed monthly runs from affecting the rebuild
  gate.

Latest stable progress:
  workers_running: 8
  seg_error_logs: 0
  hPa-complete observed rows: 210,865
  observed-row hPa coverage: 38.885%
  top completed/near-completed context:
    alistro: 21,642 rows complete, 54 missing
    bonifacio: 21,642 rows complete, 54 missing
    revellata: 21,642 rows complete, 54 missing
  rebuild/benchmark status:
    not started yet; watcher is correctly waiting for segment completion.

Score/status checkpoint:
  timestamp_utc: 2026-06-26T23:40:08Z
  conclusion:
    No new RMSE improvement has been measured yet after the vertical-profile
    work, because the Open-Meteo pressure-level backfill is still running and
    the downstream rebuild/benchmark has not started.
  current best locked validation:
    benchmark:
      prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_v1
    calibrated RMSE:
      1.269403
    calibrated MAE:
      0.930905
    gap to RMSE 0.9:
      0.369403
    verdict:
      not_achieved
  latest hPa backfill progress:
    workers_running: 8
    seg_error_logs: 0
    observed_rows: 542,280 / 542,400
    required hPa-complete rows: 223,328
    hPa-complete observed-row coverage: 41.183%
    training table:
      residual_backfill_2024_2026_short_hpa_v1 not created yet
    hPa benchmark:
      tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1 not created yet
  next criterion:
    Once the watcher sees >=99.5% observed-row hPa coverage, it will rebuild the
    short-horizon dataset and run the first LightGBM benchmark using the new
    vertical-profile features. That will be the first valid answer to whether
    these features improve the score.

Extra hPa benchmark watcher:
  timestamp_utc: 2026-06-26T23:42:50Z
  script:
    scripts/ml_dataset/z2_watch_hpa_then_extra_benchmarks.sh
  z2 status:
    /srv/data/corsewind/ml_dataset/backfill_logs/hpa_extra_benchmarks_watcher.status
  z2 log:
    /srv/data/corsewind/ml_dataset/backfill_logs/hpa_extra_benchmarks_watcher.log
  actual watcher PID:
    94509
  behavior:
    waits for the primary hPa LightGBM benchmark:
      tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1
    then launches two additional leakage-safe 2024-2025 -> 2026 short-horizon
    benchmarks:
      tabular_extratrees_260k_short_hpa_v1_2024_2025_to_2026_v1
      tabular_lgbm_bylead_260k_short_hpa_v1_2024_2025_to_2026_v1
  rationale:
    The current locked best was produced by an ExtraTrees calibration chain, so
    testing only LightGBM on the new hPa features would leave a plausible model
    family untested. The by-lead LightGBM variant tests the repeated observation
    that +15/+30/+45/+60 min horizons have different useful signals.
  safety:
    The watcher is passive until the primary benchmark writes
    `training_results.json`; it does not interfere with the active Open-Meteo
    pressure-level collectors or the primary rebuild watcher.

Selection gate update:
  timestamp_utc: 2026-06-26T23:46:06Z
  latest hPa backfill progress:
    workers_running: 8
    seg_error_logs: 0
    observed_rows: 542,280 / 542,400
    required hPa-complete rows: 262,324
    hPa-complete observed-row coverage: 48.374%
    recent_files_last_5_minutes: 781
  hPa benchmark status:
    primary benchmark not created yet
    training table not created yet
  extra watcher update:
    actual watcher PID: 94748
    now runs final selection after the three hPa benchmarks:
      hpa_tabular_rmse09_selection_v1/hpa_tabular_rmse09_selection.json
      hpa_tabular_rmse09_selection_v1/hpa_tabular_rmse09_selection.md
      hpa_tabular_rmse09_selection_v1/hpa_tabular_rmse09_assertion.json
  interpretation:
    The hPa backfill is still progressing and should not be interrupted. No new
    RMSE comparison is valid until the rebuild watcher writes the primary
    benchmark results. The final selection/assertion gate will prevent us from
    mistaking a partial or invalid hPa run for a real RMSE < 0.9 achievement.

Global leaderboard correction:
  timestamp_utc: 2026-06-26T23:50:26Z
  artifact added:
    scripts/ml_dataset/select_wind_mean_rmse_leaderboard.py
  z2 current leaderboard:
    /srv/data/corsewind/ml_dataset/benchmarks/wind_mean_rmse_leaderboard_current.json
    /srv/data/corsewind/ml_dataset/benchmarks/wind_mean_rmse_leaderboard_current.md
  corrected best known wind-mean RMSE:
    run:
      prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
    evidence:
      /srv/data/corsewind/ml_dataset/benchmarks/prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1/calibration_results.json
    RMSE:
      1.268019
    MAE:
      0.930465
    evaluation rows:
      31,429
    evaluation window:
      2026-01-01T00:00:00Z -> 2026-07-01T00:00:00Z
    gap to RMSE 0.9:
      0.368019
  previous operational reference:
    1.269403 was slightly stale; it remains a valid historical run but is no
    longer the best known score.
  watcher correction:
    primary hPa watcher PID 95125 and extra hPa watcher PID 95137 now compare
    new tabular hPa audits against 1.268019, not 1.269403.
  latest hPa coverage remains:
    required hPa-complete rows: 262,324 / 542,280 observed
    observed-row hPa coverage: 48.374%
    segment error logs: 0

Open-Meteo 429 handling:
  timestamp_utc: 2026-06-26T23:52:51Z
  issue:
    The active 8-segment hPa backfill hit Open-Meteo's hourly request limit.
    Segment 08 stopped with HTTP 429 on several 2026-04-01 -> 2026-06-23
    spot chunks. Continuing the other workers would likely create more holes.
  action taken:
    Stopped all active `open_meteo_pressure_levels_seg*` collectors while
    preserving already written daily JSONL files.
    Stopped the primary rebuild watcher so it cannot audit/rebuild from an
    intentionally incomplete hPa backfill.
  repair script:
    scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh
  z2 repair PID:
    95408
  z2 repair log:
    /srv/data/corsewind/ml_dataset/backfill_logs/open_meteo_pressure_repair_after_429.log
  behavior:
    waits 3,900 seconds for quota reset, audits missing hPa rows, generates
    slow per-spot repair tasks, repairs sequentially with 14-day chunks and
    4-second request sleeps, reruns the hPa coverage audit, and relaunches the
    primary rebuild watcher only if coverage is >= 99.5%.
  active state:
    repair watcher sleeping for quota reset
    extra hPa benchmark watcher remains waiting for the primary benchmark
    no pressure-level segment collectors are currently running

Repair hardening:
  timestamp_utc: 2026-06-26T23:55:33Z
  updated repair PID:
    95589
  changes:
    - repair requests now use 7-day chunks instead of 14-day chunks
    - repair sleep is now 8 seconds between requests instead of 4 seconds
    - `audit_open_meteo_coverage.py` now emits the full
      `required_feature_partial_days` list, not only a 20-row sample
    - repair planning uses the full partial-day list when available
  rationale:
    The current audit would generate roughly 176 high-level spot/date repair
    tasks from the 48.374% hPa-complete state. Slower sequential repair is
    preferable after an hourly Open-Meteo quota limit, even if it takes longer,
    because another 429 would delay the downstream RMSE benchmark more than a
    conservative repair cadence.

Repair retry hardening:
  timestamp_utc: 2026-06-26T23:57:54Z
  updated repair PID:
    95739
  latest local repair estimate:
    high-level repair tasks: 176
    missing full days: 11,653
    partial days: 37
    estimated API calls with 7-day chunks: 1,781
    minimum runtime at 8 seconds between requests: 237.5 minutes
  retry behavior:
    Each high-level repair task now writes a per-task attempt log and retries up
    to 3 times. If a task sees `HTTP 429` or `Hourly API request limit exceeded`,
    it waits the full quota reset interval before retrying the same task instead
    of skipping ahead and leaving a silent hole.

Current-best RMSE gap audit:
  timestamp_utc: 2026-06-27T00:00:00Z
  artifacts:
    docs/ml_nowcasting/rmse09_gap_audit_current_best_scale070.json
    docs/ml_nowcasting/rmse09_gap_audit_current_best_scale070.md
  benchmark:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
  RMSE:
    1.268019
  MAE:
    0.930465
  rows:
    31,429
  gap to 0.9:
    0.368019
  MSE reduction needed:
    49.623%
  error concentration:
    top 1% rows SSE share: 16.983%
    top 5% rows SSE share: 40.993%
    top 10% rows SSE share: 56.654%
    rows needing perfect correction to hit 0.9: 2,342 (7.452%)
  diagnostic oracle:
    Choosing row-wise between raw NWP, scale070 calibrated, and autoscale
    calibrated would only reach RMSE 1.051416. This is not deployable because it
    uses the observed target to choose the best row, but it proves that simple
    routing among the already available variants cannot reach 0.9.
  dominant error groups:
    by spot:
      la_tonnara RMSE 1.520349, SSE share 19.609%
      santa_manza RMSE 1.497476, SSE share 18.176%
      balistra RMSE 1.234581, SSE share 14.653%
      porticcio RMSE 1.010447, SSE share 12.862%
      porto_polo RMSE 1.170954, SSE share 10.205%
    by lead:
      +60 min RMSE 1.359074, SSE share 36.577%
      +45 min RMSE 1.311420, SSE share 24.170%
      +30 min RMSE 1.250536, SSE share 21.802%
      +15 min RMSE 1.100990, SSE share 17.451%
    by actual wind:
      actual >=8 m/s RMSE 1.753438, SSE share 31.917%
  implication:
    The path to RMSE 0.9 cannot be only scale tuning or routing between current
    raw/calibrated predictions. We need new signal for high-wind and +45/+60 min
    regimes, especially on La Tonnara and Santa Manza. The pending vertical hPa
    features are aligned with this because they target stability, shear, and
    thermal regime structure, but they still need leakage-safe validation.

Current-best reduction targets:
  timestamp_utc: 2026-06-27T00:05:00Z
  artifacts:
    docs/ml_nowcasting/rmse09_reduction_targets_current_best_scale070.json
    docs/ml_nowcasting/rmse09_reduction_targets_current_best_scale070.md
  key targets:
    lead_45_60:
      current RMSE: 1.339495
      required RMSE if this subgroup alone solves the gap: 0.573208
      required RMSE reduction: 57.207%
    critical_spots_or_lead_45_60:
      current RMSE: 1.334694
      required RMSE if this subgroup alone solves the gap: 0.837812
      required RMSE reduction: 37.228%
    actual_8plus:
      current RMSE: 1.753427
      cannot solve global RMSE 0.9 alone because non-group RMSE is already
      1.045712.
  path document:
    docs/ml_nowcasting/rmse_0_9_path_to_target.md now uses 1.268019 as the
    current best reference.

## 2026-06-27 - pending hPa residual calibrator watcher

Implemented:
  script:
    scripts/ml_dataset/z2_watch_hpa_calibrator_then_leaderboard.sh
  z2 deployed path:
    /srv/data/corsewind/backfill_runner/scripts/ml_dataset/z2_watch_hpa_calibrator_then_leaderboard.sh
  z2 PID:
    96259
  log:
    /srv/data/corsewind/ml_dataset/backfill_logs/hpa_calibrator_watcher.log

Current status:
  timestamp_utc: 2026-06-27T00:08:10Z
  state:
    watcher running, waiting for primary hPa benchmark
  dependency:
    primary hPa benchmark:
      tabular_lgbm_300k_short_hpa_v1_2024_2025_to_2026_v1
    extra hPa watcher status:
      required complete before calibrator training starts by default

Protocol:
  Once hPa feature-store/training-table rebuild and primary benchmark are ready,
  the watcher trains a leakage-safe 2025H2 calibration base on the same hPa
  training table:
    run_id:
      tabular_lgbm_calbase_180k_short_hpa_v1_2024_to_2025h2_v1
    split:
      2025-07-01T00:00:00Z
    train window:
      2024 to 2025H1
    calibration/eval window:
      2025H2
  It then applies the established second-stage temporal residual calibrator to
  the hPa 2026 primary benchmark predictions:
    run_id:
      prediction_residual_calibrator_hpa_2025h2_to_2026_extratrees_scalegrid_v1
    model:
      ExtraTrees residual calibrator
    scale validation:
      2025Q4 only
    locked evaluation:
      2026 only
    scale candidates:
      0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00

Rationale:
  The best current score, 1.268019, came from the same high-level recipe before
  vertical hPa features: base tabular residual model plus temporal residual
  calibration. The hPa features target the largest remaining error families
  (lead 45/60, thermal/high-wind regimes, La Tonnara/Santa Manza), so the first
  fair test after hPa backfill should reuse this proven recipe on the enriched
  dataset before adding more complex specialists.

## 2026-06-27 - hPa repair pre-wake audit

Checked:
  timestamp_utc: 2026-06-27T00:09:44Z
  z2 processes:
    repair watcher:
      scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh
    extra hPa benchmark watcher:
      scripts/ml_dataset/z2_watch_hpa_then_extra_benchmarks.sh
    hPa calibrator watcher:
      scripts/ml_dataset/z2_watch_hpa_calibrator_then_leaderboard.sh
  statuses:
    hpa_calibrator_watcher.status: running
    hpa_extra_benchmarks_watcher.status: running
    open_meteo_pressure_rebuild_watcher.status: stopped_due_to_rate_limit
    open_meteo_pressure_repair_after_429.status: running

Coverage still unchanged before quota wake:
  expected rows:
    542400
  observed rows:
    542280
  missing rows:
    120
  required hPa feature complete rows:
    262324
  required hPa feature missing rows:
    280076
  observed hPa complete coverage:
    48.3743%

Code audit:
  repair script:
    scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh
  collector:
    scripts/ml_dataset/collect_open_meteo_historical_forecast.py
  result:
    The repair script is consistent with the current audit schema. It reads
    `required_feature_complete_rows`, builds tasks from
    `required_feature_missing_day_ranges` and the full
    `required_feature_partial_days` list, and retries the same task after a full
    quota wait if the task log contains `Open-Meteo HTTP 429` or
    `Hourly API request limit exceeded`.
  dedupe behavior:
    The collector writes by day and deduplicates by
    `(model, spot_id, valid_time_utc)`, so repairing one spot/day replaces that
    spot's rows without dropping other already-good rows in the daily file.

RMSE status:
  no new benchmark finished
  current best remains:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
    RMSE 1.268019

## 2026-06-27 - hPa calibrator watcher diagnostics hardening

Implemented:
  script updated:
    scripts/ml_dataset/z2_watch_hpa_calibrator_then_leaderboard.sh
  z2 deployed path:
    /srv/data/corsewind/backfill_runner/scripts/ml_dataset/z2_watch_hpa_calibrator_then_leaderboard.sh
  old watcher PID:
    96259
  new watcher PID:
    96500

Change:
  After the hPa residual calibrator finishes and the leaderboard is refreshed,
  the watcher now writes RMSE-0.9 diagnostic artifacts for the hPa calibrated
  predictions:
    rmse09_gap_audit_hpa_calibrator_v1.json/md
    rmse09_reduction_targets_hpa_calibrator_v1.json/md

Diagnostic comparisons:
  The gap audit treats the hPa calibrated predictions as the primary candidate
  and compares them against:
    - the hPa primary tabular benchmark predictions
    - the previous best scale070 calibrated predictions, when available

Rationale:
  If hPa improves the score but still misses 0.9, we need the next bottleneck
  immediately: dominant spot/lead/wind-bin SSE groups, tail concentration, and
  subgroup reduction targets. This keeps the iteration from stopping at a score
  without explaining what signal or model change is still missing.

## 2026-06-27 - hPa watcher memory hardening after z2 OOM

Implemented:
  scripts updated:
    scripts/ml_dataset/z2_watch_open_meteo_pressure_then_rebuild.sh
    scripts/ml_dataset/z2_watch_hpa_then_extra_benchmarks.sh

Why:
  z2 previously rebooted after an out-of-memory event. The primary hPa benchmark
  and extra hPa benchmark watcher were still launching heavy LightGBM/ExtraTrees
  jobs without the same memory guard used by the calibrator watcher.

Changes:
  - exported single-thread BLAS/OpenMP defaults in both watchers
  - added `MEMORY_MIN_AVAILABLE_KB`, default `2200000`
  - added `MEMORY_MAX_RSS_KB`, default `12000000`
  - wrapped the hPa feature-store/training-table rebuild in `run_guarded`
  - wrapped primary LightGBM hPa training in `run_guarded`
  - wrapped extra hPa ExtraTrees and by-lead LightGBM training in `run_guarded`
  - watchers now write `failed:*` status if a guarded training job fails

Remote state:
  timestamp_utc: 2026-06-27T00:15:31Z
  primary hPa watcher:
    script synced on z2; currently not running, expected to be relaunched by
    the repair watcher after hPa coverage passes.
    remote md5 after rebuild guard:
      710c021dcc44b945961f154fbfa22bef
  extra hPa watcher:
    stale `running` status without a process was detected after the script
    update. It was relaunched explicitly.
    new PID: 96635
  calibrator watcher:
    still running and waiting for the primary hPa benchmark.

RMSE status:
  no new benchmark finished
  current best remains RMSE 1.268019

## 2026-06-27 - hPa repair acceleration and recoverability check

Observed:
  The hPa repair was progressing, but several 7-day Open-Meteo calls suffered
  transient connection resets/timeouts. No 429 or permanent failure occurred,
  but the single-worker speed with `REQUEST_SLEEP_SEC=8` was unnecessarily slow.

Recoverability probe:
  Checked rows that existed with all required hPa features as `null`, then
  refetched them to `/tmp/corsewind_hpa_probe`.

  Probe cases:
    - santa_manza, 2024-01-02
    - santa_manza, 2025-07-15

  Result:
    Both probes returned 24/24 rows with non-null:
      - temperature_1000hPa
      - relative_humidity_850hPa
      - geopotential_height_850hPa
      - wind_speed_850hPa
      - wind_direction_850hPa

  Conclusion:
    The missing hPa rows are recoverable by targeted refetch. The repair is not
    chasing structurally unavailable data.

Tuning:
  Relaunched repair with:
    REQUEST_SLEEP_SEC=3
    TIMEOUT_SEC=240
    MAX_DAYS_PER_REQUEST=7
    REPAIR_TASK_DAYS_PER_RANGE=7
    NON_429_RETRY_WAIT_SECONDS=120

Remote state:
  timestamp_utc: 2026-06-27T01:38:23Z
  repair PID: 100775
  pre-repair hPa coverage after completed refetches: 0.48846352
  complete hPa rows: 264884
  missing hPa rows: 277516
  remaining task count: 1764
  current progress: 4 / 1764
  retries in current run: 0
  permanent failures: 0
  429 count: 0

RMSE status:
  no new benchmark finished
  current best remains RMSE 1.268019

## 2026-06-27 - hPa repair task granularity tuning

Observed:
  After retry hardening, a long repair task for `ajaccio_buoy` from
  2024-09-01 to 2024-10-01 wrote 24 daily files successfully but timed out on
  the chunk 2024-09-08 to 2024-09-14. Retrying the whole month would repeatedly
  refetch already written chunks.

Iteration:
  - first attempted daily task expansion
  - that produced 11,666 tasks, which is too slow for Open-Meteo quotas
  - replaced daily expansion with range chunks controlled by
    `REPAIR_TASK_DAYS_PER_RANGE`, defaulting to `MAX_DAYS_PER_REQUEST`

Implemented:
  script updated:
    scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh
  status script updated:
    scripts/ml_dataset/summarize_hpa_rmse09_status.py

Changes:
  - missing feature ranges now become 7-day task chunks
  - partial feature days remain single-day tasks
  - progress summary now uses the current task index for restarted runs
  - repair pidfile corrected to the actual repair child process on z2

Remote state:
  timestamp_utc: 2026-06-27T01:28:29Z
  repair PID: 99916
  task count after chunking: 1777
  progress: 18 / 1777
  retries: 3
  permanent failures: 0
  429 count: 0

RMSE status:
  no new benchmark finished
  current best remains RMSE 1.268019

## 2026-06-27 - hPa repair status dashboard hardening

Implemented:
  script updated:
    scripts/ml_dataset/summarize_hpa_rmse09_status.py

Why:
  The z2 repair/rebuild chain is currently waiting on an Open-Meteo quota reset.
  The previous compact status output exposed `watcher_statuses`, but not a
  short `statuses` map, and it did not parse repair task progress from the
  repair log. That made quick checks look like `None` even though the watcher
  state was available.

Changes:
  - added a short `statuses` map to the JSON summary
  - added repair task progress parsing from
    `open_meteo_pressure_repair_after_429.log`
  - added repair task failure/429 warning hooks
  - refreshed the remote z2 status script without restarting active jobs

Remote state:
  timestamp_utc: 2026-06-27T00:39:37Z
  next_action: wait_for_repair_wake
  statuses:
    repair: running
    primary: stopped_due_to_rate_limit
    extra: running
    calibrator: running
  expected repair wake: 2026-06-27T01:04:28Z
  repair_remaining_seconds: 1491
  hPa coverage observed ratio: 0.48374272
  hPa complete rows: 262324
  hPa missing rows: 280076
  repair tasks started: 0
  warnings: none

RMSE status:
  no new benchmark finished
  current best remains RMSE 1.268019
  gap to target remains 0.368019

## 2026-06-27 - hPa repair retry hardening

Observed:
  z2 repair woke at 2026-06-27T01:04:28Z and generated 176 repair tasks.
  The first 7 tasks wrote successfully, then task 8 failed on:
    spot: bonifacio
    date: 2024-06-13
    error: Open-Meteo read timeout after 180s

Problem:
  The repair script was intended to retry non-429 transient failures, but with
  `set -euo pipefail` and the global `ERR` trap, the failed collector pipeline
  triggered the trap before the script could inspect `PIPESTATUS` and retry.

Implemented:
  script updated:
    scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh
  status script updated:
    scripts/ml_dataset/summarize_hpa_rmse09_status.py

Changes:
  - disable the `ERR` trap only around the collector pipeline
  - capture `PIPESTATUS[0]`, then restore `set -e` and the trap
  - retry non-429 transient failures after configurable
    `NON_429_RETRY_WAIT_SECONDS` (default 120s)
  - keep 429 failures on the longer quota wait path
  - status parser now scopes repair progress to the latest repair run segment
  - status parser separates transient retry count from permanent failures
  - latest task log is selected by modification time instead of filename

Remote state:
  timestamp_utc: 2026-06-27T01:14:12Z
  repair relaunched with:
    INITIAL_WAIT_SECONDS=0
    WAIT_SECONDS=3900
    NON_429_RETRY_WAIT_SECONDS=120
  progress:
    started tasks: 10 / 176
    retry count: 1
    permanent failures: 0
    429 count: 0

Validation:
  local shell syntax passed for repair script
  local Python compilation passed for status script
  secret scan returned no matches

RMSE status:
  no new benchmark finished
  current best remains RMSE 1.268019

## 2026-06-27 - consolidated hPa/RMSE09 status report

Implemented:
  script:
    scripts/ml_dataset/summarize_hpa_rmse09_status.py
  z2 deployed path:
    /srv/data/corsewind/backfill_runner/scripts/ml_dataset/summarize_hpa_rmse09_status.py
  z2 outputs:
    /srv/data/corsewind/ml_dataset/benchmarks/hpa_rmse09_status_current.json
    /srv/data/corsewind/ml_dataset/benchmarks/hpa_rmse09_status_current.md

Purpose:
  Provide a single machine-readable and human-readable status for:
    - repair/primary/extra/calibrator watcher statuses
    - pidfile validity using `/proc/<pid>/cmdline`
    - hPa required-feature coverage
    - current best wind-mean RMSE leaderboard
    - latest repair wait line and expected wake time
    - next recommended action

First z2 summary:
  generated_at_utc:
    2026-06-27T00:24:30Z
  next_action:
    wait_for_repair_wake
  repair expected wake:
    2026-06-27T01:04:28Z
  repair remaining seconds at summary time:
    2398
  hPa coverage observed ratio:
    0.48374272
  current best RMSE:
    1.268019
  current best run:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1

Rationale:
  The pipeline now has enough moving pieces that status should not require
  manually reading four logs and three JSON files. This status report reduces
  operational ambiguity while waiting for the hPa repair and subsequent
  benchmarks.

Status report enrichment:
  timestamp_utc: 2026-06-27T00:26:23Z
  script updated:
    scripts/ml_dataset/summarize_hpa_rmse09_status.py
  additions:
    - repair task TSV line count
    - repair task log count
    - latest repair task log tail
    - latest task `429` detection
    - expected hPa benchmark/calibrator artifact existence and timestamps
  refreshed z2 summary:
    next_action: wait_for_repair_wake
    repair task count: none yet
    repair task logs: 0
    latest task hit 429: false
    primary hPa training_results exists: false
    primary hPa predictions exists: false
    hPa extra selection exists: false
    hPa calibrator results exists: false
    hPa calibrator predictions exists: false
    hPa calibrator gap audit exists: false
  interpretation:
    The repair has not yet moved past the initial quota wait. No hPa benchmark
    artifact has been produced yet, so the score remains unchanged.

## 2026-06-27 - repair failure status trap

Implemented:
  script updated:
    scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh

Why:
  If the repair script failed after waking, for example at the final coverage
  gate or during an audit command, `set -e` would terminate the process but could
  leave `open_meteo_pressure_repair_after_429.status` as `running`. Downstream
  watchers would then appear alive while the repair had actually stopped.

Change:
  Added an `ERR` trap that writes `failed:<exit_code>` to the repair status file
  and logs the failing line before exiting.

Remote state:
  timestamp_utc:
    2026-06-27T00:28:04Z
  old repair PID:
    96899
  new repair PID:
    97386
  initial wait after restart:
    2184s
  expected wake preserved:
    2026-06-27T01:04:28Z

Refreshed consolidated status:
  next_action:
    wait_for_repair_wake
  repair running:
    true
  best RMSE:
    1.268019
  hPa coverage:
    0.48374272

## 2026-06-27 - hPa status consistency warnings

Implemented:
  script updated:
    scripts/ml_dataset/summarize_hpa_rmse09_status.py

Additions:
  - `warnings` list in the JSON summary
  - warning when a watcher status is `running` but the pidfile does not point
    to a matching cmdline
  - warning when a watcher status is `failed:*` but the process is still alive
  - warning when repair task logs exist but the task TSV is missing
  - warning when the latest repair task log shows Open-Meteo 429
  - warning when primary hPa training artifacts exist before the hPa coverage
    gate is satisfied
  - markdown now displays warning count and warning details

Refreshed z2 status:
  timestamp_utc:
    2026-06-27T00:30:03Z
  next_action:
    wait_for_repair_wake
  warnings:
    []
  repair remaining seconds:
    2065
  repair task file exists:
    false
  repair task count:
    0
  repair task logs:
    0
  best RMSE:
    1.268019
  hPa coverage:
    0.48374272

## 2026-06-27 - hPa post-run decision summary

Implemented:
  script:
    scripts/ml_dataset/summarize_hpa_rmse09_iteration.py
  z2 deployed path:
    /srv/data/corsewind/backfill_runner/scripts/ml_dataset/summarize_hpa_rmse09_iteration.py
  z2 outputs:
    /srv/data/corsewind/ml_dataset/benchmarks/hpa_rmse09_iteration_summary.json
    /srv/data/corsewind/ml_dataset/benchmarks/hpa_rmse09_iteration_summary.md

Purpose:
  Once hPa artifacts exist, summarize whether the hPa iteration:
    - achieves RMSE < 0.9
    - improves the previous best RMSE 1.268019 but not enough
    - fails to improve
    - needs additional input signal rather than more calibration

Inputs:
  - consolidated hPa/RMSE status JSON
  - global wind-mean RMSE leaderboard
  - hPa tabular selection, when available
  - hPa residual calibrator results, when available
  - hPa gap audit and reduction targets, when available

Initial z2 result:
  decision:
    waiting_for_hpa_artifacts
  best_hpa_rmse:
    null
  next_action:
    Wait for repair -> primary rebuild -> hPa benchmark chain.

Rationale:
  This prepares the decision layer before the long hPa repair and benchmark
  chain finishes. When the artifacts appear, the same command will immediately
  produce the next action instead of requiring manual interpretation.

Watcher integration:
  timestamp_utc:
    2026-06-27T00:34:21Z
  script updated:
    scripts/ml_dataset/z2_watch_hpa_calibrator_then_leaderboard.sh
  change:
    The hPa calibrator watcher now calls both:
      - scripts/ml_dataset/summarize_hpa_rmse09_status.py
      - scripts/ml_dataset/summarize_hpa_rmse09_iteration.py
    after leaderboard and gap-audit generation.
  remote state:
    old calibrator watcher PID: 97022
    new calibrator watcher PID: 97714
  effect:
    Once the hPa calibrator finishes, the final status and iteration decision
    summary are generated automatically without manual follow-up.

## 2026-06-27 - post-primary concurrency guard

Implemented:
  scripts updated:
    scripts/ml_dataset/z2_watch_hpa_then_extra_benchmarks.sh
    scripts/ml_dataset/z2_watch_hpa_calibrator_then_leaderboard.sh

Why:
  The extra hPa benchmark watcher previously waited only for the primary
  `training_results.json`. That file appears before the primary watcher has
  necessarily finished audits and prediction export. Starting ExtraTrees or
  by-lead LightGBM at that moment could overlap with primary diagnostics and
  increase memory pressure on z2.

Changes:
  - extra hPa watcher now waits for both:
      - primary `training_results.json`
      - `open_meteo_pressure_rebuild_watcher.status == complete`
  - extra hPa watcher fails explicitly if primary status is `failed:*`
  - hPa calibrator watcher uses the same primary completion gate
  - hPa calibrator watcher fails explicitly if extra watcher status is `failed:*`

Remote state:
  timestamp_utc: 2026-06-27T00:21:44Z
  extra watcher:
    old PID: 96635
    new PID: 97021
  calibrator watcher:
    old PID: 96500
    new PID: 97022
  current logged wait state:
    primary_status=stopped_due_to_rate_limit

RMSE status:
  no new benchmark finished
  current best remains RMSE 1.268019

## 2026-06-27 - stale PID guard for hPa repair/rebuild

Implemented:
  scripts updated:
    scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh
    scripts/ml_dataset/z2_watch_open_meteo_pressure_then_rebuild.sh

Why:
  After OOM/reboot/kill events, old pidfiles can remain. A plain `ps -p <pid>`
  can be fooled if Linux later reuses the PID for an unrelated process. That
  could either prevent the repair script from relaunching the primary watcher,
  or make the primary watcher wait forever on stale segment pidfiles.

Changes:
  - added `pid_matches(pid, needle)` helper using `/proc/<pid>/cmdline`
  - repair now treats the primary rebuild watcher as alive only if cmdline
    contains `z2_watch_open_meteo_pressure_then_rebuild.sh`
  - primary rebuild watcher counts segment pidfiles only when cmdline contains
    `collect_open_meteo_historical_forecast.py`
  - added `INITIAL_WAIT_SECONDS`, separate from `WAIT_SECONDS`, so a repair
    watcher can be restarted without shortening the full wait used for future
    429 retries

Remote state:
  timestamp_utc: 2026-06-27T00:19:28Z
  repair watcher:
    old PID: 96859
    new PID: 96899
    initial wait: 2700s
    retry wait after future 429: 3900s
  expected repair wake:
    approximately 2026-06-27T01:04:28Z

RMSE status:
  no new benchmark finished
  current best remains RMSE 1.268019

## 2026-06-27T01:52Z - live hPa repair and partial signal checkpoint

Machine/path:
  z2:/srv/data/corsewind/backfill_runner

Objective:
  Continue toward `RMSE < 0.9` on wind mean without leakage by testing whether
  pressure-level features add usable signal before the full hPa repair finishes.

Current best validated score:
  benchmark:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
  wind_mean_rmse:
    1.268019
  gap_to_0_9:
    0.368019
  decision:
    not achieved; keep as current baseline.

Live data repair:
  script:
    scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh
  status:
    running
  hPa coverage observed ratio:
    0.48846352
  complete required-feature rows:
    264884
  missing required-feature rows:
    277516
  repair task progress:
    39 / 1764
  retries:
    6 transient retries
  permanent failures:
    0
  Open-Meteo 429:
    0
  current task:
    balistra 2026-05-30 -> 2026-06-01

Live early-signal benchmark:
  script:
    scripts/ml_dataset/z2_run_partial_hpa_early_signal.sh
  run_id:
    residual_backfill_2025h2_2026_short_hpa_partial_signal_v1
  benchmark_id:
    tabular_lgbm_150k_short_hpa_partial_signal_2025h2_to_2026_v1
  split:
    build features on 2025-07-01..2026-06-23, train/evaluate with the
    script's temporal 2025H2 -> 2026 setup
  pre-run hPa coverage:
    94848 / 214728 = 0.44171231
  progress:
    feature/training table rebuild reached chunk 5 / 12
  result files:
    training_results.json missing
    tabular_rmse09_audit.json missing
    tabular_error_analysis.json missing

Decision:
  Do not launch another heavy model while the partial rebuild is active. The
  useful next evidence is the partial hPa RMSE. If it improves the 1.268019
  baseline, keep the pressure-level path and wait for full repair; if it does
  not improve, inspect per-spot/per-lead errors before adding more models.

Next action:
  Monitor the partial hPa benchmark until `training_results.json` appears, then
  compare it against the current best and run the RMSE09 audit.

## 2026-06-27T01:55Z - hPa repair process-timeout hardening

Issue observed:
  The repair task `balistra 2026-05-30 -> 2026-06-01` produced an empty task log
  for several minutes, then returned an Open-Meteo read timeout. The old repair
  watcher would retry, but it had no global process timeout around the collector
  command, only the collector's HTTP timeout.

Change:
  script:
    scripts/ml_dataset/z2_repair_open_meteo_pressure_after_rate_limit.sh
  added:
    COLLECT_PROCESS_TIMEOUT_SEC default 600
  behavior:
    each collector invocation now runs under:
      timeout --kill-after=30s "${COLLECT_PROCESS_TIMEOUT_SEC}s"

Validation:
  local:
    bash -n passed
    secret scan passed with no matches
  z2:
    remote bash -n passed
    grep confirmed `COLLECT_PROCESS_TIMEOUT_SEC` and `timeout --kill-after`

Operational action:
  old repair watcher:
    PID 100775 killed during retry wait
  new repair watcher:
    PID 102258
    INITIAL_WAIT_SECONDS=0
    REQUEST_SLEEP_SEC=3
    TIMEOUT_SEC=240
    COLLECT_PROCESS_TIMEOUT_SEC=600
    MAX_DAYS_PER_REQUEST=7
    REPAIR_TASK_DAYS_PER_RANGE=7

Post-restart evidence:
  new pre-repair audit completed
  new task_count:
    1743
  previous task_count:
    1764
  current task example:
    alistro 2024-05-16 -> 2024-05-16
  process tree:
    timeout wrapper is active around collect_open_meteo_historical_forecast.py

RMSE status:
  no new score yet
  current best remains RMSE 1.268019

Decision:
  Keep the repair running under the new guard. This protects the long backfill
  from silent collector hangs without changing the leakage-safe evaluation
  setup.

## 2026-06-27T02:13Z - partial hPa benchmark rebuild hit memory guard

Machine/path:
  z2:/srv/data/corsewind/backfill_runner

Experiment:
  run_id:
    residual_backfill_2025h2_2026_short_hpa_partial_signal_v1
  benchmark_id:
    tabular_lgbm_150k_short_hpa_partial_signal_2025h2_to_2026_v1

Result:
  status:
    failed before training
  status file:
    /srv/data/corsewind/ml_dataset/backfill_logs/partial_hpa_early_signal.status
  status value:
    failed:19
  failure reason:
    memory guard killed the rebuild wrapper during combined-table processing
  guard detail:
    MemAvailable dropped to 681672 kB

Artifacts produced before failure:
  combined JSONL:
    /srv/data/corsewind/ml_dataset/training_tables/residual_backfill_2025h2_2026_short_hpa_partial_signal_v1/training_rows.jsonl
  combined JSONL size:
    16G
  training_profile:
    training_row_count = 316726
    source_chunk_count = 12
    first_issue_time_utc = 2025-07-01T10:00:00Z
    last_issue_time_utc = 2026-06-23T18:00:00Z

Important interpretation:
  This is not evidence that hPa features failed. No model score was produced.
  The failure is infrastructure/memory during table materialization.

Follow-up implementation:
  added:
    scripts/ml_dataset/sample_residual_training_jsonl.py
    scripts/ml_dataset/z2_resume_partial_hpa_sampled_signal.sh
  purpose:
    stream the large combined JSONL, preserve the strict temporal split, select
    deterministic train/test samples, and write a smaller JSONL before Parquet
    export/training.
  resume script:
    stops any leftover full-Parquet export, samples the 16G JSONL, exports the
    sampled table to Parquet with batch_size=1000, runs LightGBM, then writes the
    RMSE09 audit and error analysis.
  validation:
    python3 -m py_compile passed
    bash -n passed for the resume script
    secret scan passed with no matches

Next action when z2 is reachable:
  1. stop any leftover manual full-Parquet export if it is still running;
  2. sync `sample_residual_training_jsonl.py` to z2;
  3. sample the 16G JSONL to a smaller split-safe JSONL;
  4. export only that sampled JSONL to Parquet with small batches;
  5. run the LightGBM benchmark and audit against RMSE 1.268019.

RMSE status:
  no new score
  current best remains RMSE 1.268019

## 2026-06-28T06:58Z - sampled hPa signal runs after z2 reboot

Context:
  z2 SSH is back after reboot. The partial hPa sampled pipeline completed on:
    /srv/data/corsewind/ml_dataset/training_tables/residual_backfill_2025h2_2026_short_hpa_sampled_signal_v1/training_rows.parquet
  sampled rows:
    train=60000
    test=40000
  split:
    2026-01-01T00:00:00Z
  target:
    labels__residual_wind_mean_ms

Important comparability warning:
  These sampled hPa runs are early-signal experiments, not official replacements
  for the current best. The sampled holdout has a much worse raw NWP baseline
  around RMSE 2.18, and audits are invalid when they use only one parquet shard
  and 60k train rows. Use them to choose directions, not to update the champion.

Runs:
  global LightGBM targetfix:
    run_id: tabular_lgbm_100k_short_hpa_sampled_signal_targetfix_v1
    RMSE: 1.425800
    MAE: 1.058471
    raw RMSE: 2.185678
    gain vs raw: 34.766%
    verdict: not_achieved
  spot-grouped LightGBM:
    run_id: tabular_lgbm_100k_short_hpa_sampled_signal_spotgroup_v1
    RMSE: 1.549502
    MAE: 1.142721
    raw RMSE: 2.179876
    gain vs raw: 28.918%
    verdict: not_achieved
  lead-grouped LightGBM:
    run_id: tabular_lgbm_100k_short_hpa_sampled_signal_leadgroup_v1
    RMSE: 1.411654
    MAE: 1.046431
    raw RMSE: 2.185678
    gain vs raw: 35.413%
    verdict: invalid because sampled/one-shard audit
    by lead RMSE:
      +15m: 1.220658
      +30m: 1.371362
      +45m: 1.472300
      +60m: 1.530660
  global high-wind weighted LightGBM:
    run_id: tabular_lgbm_100k_short_hpa_sampled_signal_highwind2_v1
    RMSE: 1.438012
    MAE: 1.066979
    gain vs raw: 34.208%
    verdict: invalid because sampled/one-shard audit
  lead-grouped high-wind weighted LightGBM:
    run_id: tabular_lgbm_100k_short_hpa_sampled_signal_leadgroup_highwind2_v1
    RMSE: 1.435769
    MAE: 1.061400
    gain vs raw: 34.310%
    verdict: invalid because sampled/one-shard audit

Interpretation:
  - The best sampled hPa variant is lead-grouped LightGBM at RMSE 1.411654.
  - Spot-grouping is too fragile on this sample; it skips/weakens groups and
    Porto Pollo dominates the error tail.
  - High-wind weighting does not help global RMSE and also hurts the lead-grouped
    model.
  - Lead-specific behavior has real signal: +15m reaches RMSE 1.220658 on this
    harder sampled holdout.
  - Many pressure/context feature columns are present in the schema but empty for
    groups, so the next useful data work is coverage/feature-family cleanup, not
    simply adding more sparse columns.

Decision:
  Current champion remains:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
    RMSE: 1.268019
    MAE: 0.930465
  Do not promote any sampled hPa run.
  Next logical iteration:
    run a comparable full-data lead-aware second-stage calibrator, or add
    lead_time_minutes/regime interactions to the current champion calibrator,
    then evaluate on the same 2026 holdout with prediction diagnostics.

## 2026-06-28T07:08Z - comparable lead-aware second-stage calibrator

Code change:
  script:
    scripts/ml_dataset/train_prediction_residual_calibrator.py
  added:
    --fit-group-column
    --min-group-train-rows
    --min-group-eval-rows
    --scale-by-fit-group
    --min-scale-group-rows
  purpose:
    Let the second-stage residual calibrator train separate models per horizon
    and optionally select a correction scale per fitted group.

Validation:
  local:
    python3 -m py_compile scripts/ml_dataset/train_prediction_residual_calibrator.py
  z2:
    /home/z2/corsewind-ml-smoke/.venv/bin/python -m py_compile \
      scripts/ml_dataset/train_prediction_residual_calibrator.py
  result:
    passed

Comparable benchmark setup:
  calibration predictions:
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_calbase_2024_to_2025h2_v1/calibration_predictions_2025h2.parquet
  evaluation predictions:
    /srv/data/corsewind/ml_dataset/benchmarks/tabular_lgbm_225k_prev_lowmem_2024_2025_to_2026_v1/tabular_holdout_predictions.parquet
  calibration window:
    2025-07-01T00:00:00Z -> 2026-01-01T00:00:00Z
  evaluation window:
    2026-01-01T00:00:00Z -> 2026-07-01T00:00:00Z
  rows:
    calibration=31732
    evaluation=31429
  base RMSE:
    1.276846
  model:
    ExtraTrees, 240 trees, min_samples_leaf=50, clip=2.0

Runs:
  fixed scale 0.70, grouped by lead_time_minutes:
    run_id: prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_by_lead_v1
    RMSE: 1.268578
    MAE: 0.929617
    verdict: not_achieved
    note: MAE is slightly better than champion, but RMSE is worse.
  autoscale Q4, grouped by lead_time_minutes:
    run_id: prediction_residual_calibrator_2025h2_to_2026_extratrees_by_lead_autoscale_q4_v1
    selected global scale: 0.65
    RMSE: 1.268393
    MAE: 0.929819
    verdict: not_achieved
  autoscale Q4, grouped model and grouped scale by lead_time_minutes:
    run_id: prediction_residual_calibrator_2025h2_to_2026_extratrees_by_lead_group_scale_q4_v1
    selected group scales:
      +15m: 1.40
      +30m: 0.45
      +45m: 0.50
      +60m: 0.55
    RMSE: 1.269697
    MAE: 0.931628
    verdict: not_achieved

Diagnostic scale oracle on 2026 holdout:
  lead grouped model:
    best oracle scale: 0.60
    oracle RMSE: 1.268327
    oracle MAE: 0.930170
  current champion:
    best oracle scale: 0.675
    oracle RMSE: 1.268012
    oracle MAE: 0.930587

Interpretation:
  Lead-specific second-stage modeling has a small signal but does not beat the
  current champion. Even an invalid holdout oracle scale cannot make it beat
  the champion. The remaining gap is not a scale-selection problem.

Decision:
  Keep champion:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
    RMSE: 1.268019
    MAE: 0.930465
  Do not promote lead-grouped calibrators.
  Next high-leverage work:
    improve data signal/coverage on the worst regimes and worst spots
    (Cap Corse, La Tonnara, Santa Manza, 45-60m horizons), or build a selector
    that routes only regimes where an alternate model is proven better on a
    calibration window.

## 2026-06-27T02:17Z - sampled resume validation while z2 is unreachable

Local implementation update:
  script:
    scripts/ml_dataset/z2_resume_partial_hpa_sampled_signal.sh
  hardening:
    - detects and stops leftover full Parquet exports by command line, not only
      by pidfile
    - stops any previous failed partial rebuild process for the source run
    - optionally stops hPa repair while the sampled benchmark runs, reducing
      memory/network contention on z2
    - can restart hPa repair after completion if `RESTART_REPAIR_AFTER_SAMPLE=1`
    - defaults to a safer sampled benchmark size:
      `SAMPLE_MAX_TRAIN_ROWS=60000`, `SAMPLE_MAX_TEST_ROWS=40000`
      instead of keeping 250k rows from the 316k-row source table

Local validation:
  commands:
    bash -n scripts/ml_dataset/z2_resume_partial_hpa_sampled_signal.sh
    python3 -m py_compile scripts/ml_dataset/sample_residual_training_jsonl.py
  result:
    passed
  sampler smoke test:
    synthetic JSONL with train/test issue times
    max_train_rows=5
    max_test_rows=6
  smoke result:
    sample test ok 11
  secret scan:
    passed with no matches

z2 status:
  SSH test:
    ssh -o ConnectTimeout=8 z2 'echo ok; date -u; uptime; free -h'
  result:
    Connection timed out during banner exchange
  ping:
    2/2 replies from 192.168.1.99, so host is alive but SSH is not responsive

Decision:
  Do not retry heavy remote work until z2 accepts SSH again. The next safe
  remote command is to sync the two sampled-resume scripts and launch
  `z2_resume_partial_hpa_sampled_signal.sh`.

## 2026-06-27T02:21Z - local launcher for sampled hPa resume

Added:
  script:
    scripts/ml_dataset/wait_for_z2_and_launch_hpa_sampled_resume.py

Purpose:
  Wait for z2 SSH to become usable, sync only the scripts needed for the sampled
  hPa resume, then launch:
    scripts/ml_dataset/z2_resume_partial_hpa_sampled_signal.sh
  in a detached `setsid` process.

Why:
  z2 currently answers ping but SSH times out during banner exchange. The local
  launcher makes the next retry deterministic once SSH recovers, without relying
  on manual `scp`/`ssh` sequencing or unavailable local `rsync`.

Synced files:
  - sample_residual_training_jsonl.py
  - z2_resume_partial_hpa_sampled_signal.sh
  - export_training_table_parquet.py
  - train_residual_correction_parquet.py
  - audit_tabular_rmse09_result.py
  - analyze_tabular_rmse09_errors.py
  - z2_repair_open_meteo_pressure_after_rate_limit.sh

Validation:
  commands:
    python3 -m py_compile scripts/ml_dataset/wait_for_z2_and_launch_hpa_sampled_resume.py scripts/ml_dataset/sample_residual_training_jsonl.py
    bash -n scripts/ml_dataset/z2_resume_partial_hpa_sampled_signal.sh
    python3 scripts/ml_dataset/wait_for_z2_and_launch_hpa_sampled_resume.py --dry-run --launch-dry-run --max-wait-minutes 0
  result:
    passed
  secret scan:
    passed with no matches

Ready command once SSH is back:
  python3 scripts/ml_dataset/wait_for_z2_and_launch_hpa_sampled_resume.py \
    --max-wait-minutes 0

Long wait option:
  python3 scripts/ml_dataset/wait_for_z2_and_launch_hpa_sampled_resume.py \
    --max-wait-minutes -1 \
    --poll-seconds 60

RMSE status:
  no new score
  current best remains RMSE 1.268019

## 2026-06-29T15:20+02:00 - strategic RMSE review after phys_v1 failure and phys_v3 launch

Current live remote state:
  machine:
    z2 with 32 GB RAM and sudo NOPASSWD available
  disk:
    /srv/data has roughly 610 GB free after deleting obsolete JSONL
    intermediates
  live rebuild:
    residual_windsup_sst_prev_phys_v3_dem_fetch
    status file:
      /srv/data/corsewind/ml_dataset/run_logs/rebuild_phys_v3_dem_fetch_2024_2026.status
    launched with:
      DEM/static spot features
      true fetch/coastline features
      pressure-level and offset Open-Meteo collection

Best validated score remains:
  run:
    prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
  RMSE:
    1.268019
  MAE:
    0.930465
  Bias:
    +0.018767

Evidence reviewed:
  - scientific_error_diagnostic_report.md
  - rmse09_gap_audit_current_best_scale070.md
  - foundation_sequence_benchmark_v2_2026_windsurf.md
  - phys_v1 decision report on z2

Main conclusion:
  The easy NWP bias correction is already consumed. Raw NWP around 2.19 RMSE
  is brought down to roughly 1.27; the last residual calibration only improves
  marginally. The remaining gap is dominated by tail/regime failures, not by a
  missing global scale factor.

Hard error modes:
  - La Tonnara and Santa Manza dominate spot-level SSE.
  - +45 and +60 minute horizons dominate horizon-level SSE.
  - actual wind >= 8 m/s is heavily underpredicted.
  - 0-2 m/s wind is overpredicted.
  - global bias near zero hides conditional compression of extremes.

Important failed/limited paths:
  - More generic global calibration did not beat the champion.
  - Lead-specific calibration had a tiny signal but did not beat champion.
  - Existing-model oracle/routing alone cannot reach 0.9, so current candidate
    outputs do not contain enough independent signal.
  - phys_v1 physical features had high vertical coverage but worsened the
    stable benchmark versus champion.
  - 225k LightGBM path segfaulted even after RAM upgrade; 150k/bin63 remains the
    safer benchmark rail.
  - Naive Chronos residual sequence degraded.

Most plausible next leverage:
  1. Finish and benchmark phys_v3_dem_fetch with the same stable rail. This is a
     direct test of the missing local exposure/fetch hypothesis.
  2. If phys_v3 is neutral or worse, stop generic feature accumulation and move
     to dynamic geometry features selected by forecast/observed wind direction:
     upwind fetch, crosswind fetch, upwind relief, lee/blocking index,
     cross-shore/alongshore angle, and thermal-breeze alignment.
  3. Build a deployable hard-regime specialist, but gate it using only features
     known at issue time: spot, lead, raw predicted wind, recent observed ramp,
     model error now, thermal/pressure gradients, and directional geometry.
  4. Attack high-wind compression with quantile/asymmetric objectives and
     distributional outputs, not only point RMSE.
  5. Scale Chronos-2 as a feature source, especially p10/p50/p90 and gust
     uncertainty, then feed it into a supervised calibrator rather than using it
     as a zero-shot replacement.
  6. Keep improving labels/history on hard spots; without more clean target
     observations on the high-error regimes, RMSE 0.9 may be mathematically
     plausible but operationally unreachable.

Decision:
  Let phys_v3 finish, then judge it on:
    global RMSE vs 1.268019
    hard spots RMSE/SSE
    +45/+60 RMSE
    high-wind bias and RMSE
    windsurf threshold metrics
  If phys_v3 does not help these slices, do not run phys_v2 as a consolation
  path; move directly to dynamic direction-conditioned geometry and hard-regime
  specialist work.

## 2026-06-29T15:45+02:00 - gust prediction made mandatory

User correction:
  The production system must predict both wind mean and gusts. Optimizing only
  wind mean is not acceptable for the windsurf product or for safety/session
  decisions.

Pipeline audit:
  Existing data/training tables already contain gust fields:
    labels__target_gust_ms
    labels__residual_gust_ms
    baselines__baseline_gust_ms
    features__model_error_now_gust_ms
    features__nwp_horizon_gust_ramp_ms
    features__nwp_error_persistence_plus_gust_ramp_ms
  However, every serious z2 tabular benchmark inspected so far had zero trained
  labels__residual_gust_ms models. The reason is operational, not data-schema:
  benchmark scripts usually forced:
    --only-target labels__residual_wind_mean_ms

Implemented locally and synced to z2:
  - train_prediction_residual_calibrator.py now supports:
      --target wind_mean
      --target gust
    with calibrated_gust_ms / actual_gust_ms paths.
  - analyze_tabular_rmse09_errors.py now diagnoses either:
      labels__residual_wind_mean_ms
      labels__residual_gust_ms
    and writes target-specific prediction parquet columns.
  - Added run_live_wind_and_gust_inference.py to produce wind mean and gust
    predictions in one live artifact.
  - Added z2_run_wind_gust_150k_bin63_benchmark.sh, a z2 benchmark rail that
    trains both residual targets, analyzes both, then trains separate wind mean
    and gust second-stage calibrators.

Validation:
  Local syntax:
    python3 -m py_compile passed for modified Python scripts
    bash -n passed for new z2 benchmark script
  z2 syntax:
    py_compile and bash -n passed inside /home/z2/corsewind-ml-smoke/.venv
  z2 synthetic smoke:
    train labels__residual_gust_ms -> diagnosis -> gust calibrator passed
    smoke train_gust_rmse: 0.005704
    smoke diag_gust_rmse: 0.005704
    smoke calibrated_gust_rmse: 0.004804

Decision:
  From now on, a model/run cannot be considered production-grade unless it
  reports wind mean and gust metrics. The phys_v3 post-rebuild benchmark should
  use z2_run_wind_gust_150k_bin63_benchmark.sh rather than a wind-only rail.

## 2026-06-30T10:55+02:00 - phys_v3 ablation and phys_v4 directional campaign launched

User question:
  Why did the first/champion run beat the better-organized latest dataset?
  Execute the proposed plan rather than only speculate.

Implemented:
  - Added feature include/exclude filters to train_residual_correction_parquet.py
    so we can run controlled ablations without rebuilding shards.
  - Added augment_directional_static_features.py. It reuses phys_v3 shards and
    appends 36 direction-conditioned static features:
      upwind/downwind/crosswind fetch
      upwind water/land share
      upwind DEM barrier/open exposure
      blocking and marine exposure indices
      sea-breeze alignment from max-fetch sector
  - Made z2_run_wind_gust_150k_bin63_benchmark.sh parameterizable for max rows
    and feature filters.
  - Added z2_run_phys_v3_ablation_and_v4_directional.sh as the campaign runner.

z2 validation:
  - python py_compile passed remotely.
  - bash -n passed remotely.
  - One-month smoke augmentation on 2026_06 passed:
      source columns: 1972
      added directional columns: 36
      output columns: 2008

Campaign launched on z2:
  /srv/data/corsewind/backfill_runner/scripts/ml_dataset/z2_run_phys_v3_ablation_and_v4_directional.sh

Runs:
  1. phys_v3_old_signal_225k_bin63
     Purpose: test whether the champion was mainly better because it had more
     rows / less feature dilution.
     Excludes static DEM/fetch, nwp_offset, previous_run, vertical, EUMETSAT.

  2. phys_v3_pruned_200k_bin63
     Purpose: test whether the new feature families help after removing the
     raw sector grids and EUMETSAT noise.

  3. phys_v4_directional_pruned_200k_bin63
     Purpose: test whether dynamic direction-conditioned geometry beats static
     sector columns.

Current state:
  phys_v4_directional shards generated for all 30 months.
  First benchmark running:
    phys_v3_old_signal_225k_bin63

Decision rule:
  Compare wind mean calibrated RMSE/MAE to champion:
    RMSE 1.268019
    MAE 0.930465
  Also report gust RMSE/MAE for production readiness.

## 2026-06-30T11:28+02:00 - phys_v3_old_signal result

Run:
  phys_v3_old_signal_225k_bin63

Result:
  - wind mean base short-lead RMSE: 1.312222
  - wind mean calibrated RMSE: 1.301797
  - wind mean calibrated MAE: 0.982492
  - gust base short-lead RMSE: 1.524039
  - gust calibrated RMSE: 1.516817
  - gust calibrated MAE: 1.124876

Comparison:
  - Better than phys_v3_dem_fetch wind RMSE 1.305533, but only by 0.003736.
  - Still worse than champion wind RMSE 1.268019 by 0.033778.

Interpretation:
  The volume/capacity hypothesis is partially true but not sufficient. More rows
  and a simpler old-signal feature set recover a little RMSE, but they do not
  explain the full gap to the champion. This keeps the pruned-noise and
  direction-conditioned geometry hypotheses alive.

Next running:
  phys_v3_pruned_200k_bin63

## 2026-06-30T11:45+02:00 - phys_v3_pruned 200k partial and recovery decision

Run:
  phys_v3_pruned_200k_bin63

Base result:
  - feature count: 1372
  - train rows: 200000
  - test rows: 60000
  - wind mean base short-lead RMSE: 1.311003
  - wind mean base short-lead MAE: 0.993864
  - gust base short-lead RMSE: 1.534341
  - gust base short-lead MAE: 1.144583

Failure:
  The 2025-H2 calibration-base LightGBM process segfaulted at the same 200k
  row setting:
    exit code 139
    benchmark status code 30
    campaign status code 40
  z2 did not OOM or swap; this looks like a LightGBM/native memory stability
  limit for this feature shape, not a disk issue.

Interpretation:
  Pruning raw static sector grids and EUMETSAT does not create a large base-model
  gain. It is very close to old_signal and still materially worse than the
  champion. The calibrated result is missing because of the segfault, so do not
  use this run as a final decision artifact.

Recovery:
  Relaunch pruned and v4_directional on the safer 150k rail, which is known to
  work from phys_v3_dem_fetch_150k_bin63. This trades some capacity for a stable
  apples-to-apples comparison against the previous phys_v3 benchmark.

Recovery launched:
  /srv/data/corsewind/backfill_runner/scripts/ml_dataset/z2_resume_pruned_v4_directional_150k.sh

Recovery runs:
  - phys_v3_pruned_150k_bin63
  - phys_v4_directional_pruned_150k_bin63

## 2026-06-30T12:21+02:00 - phys_v3_pruned 150k result

Run:
  phys_v3_pruned_150k_bin63

Result:
  - feature count: 1372
  - train rows: 150000
  - wind mean base short-lead RMSE: 1.309661
  - wind mean calibrated RMSE: 1.299284
  - wind mean calibrated MAE: 0.980664
  - gust base short-lead RMSE: 1.535513
  - gust calibrated RMSE: 1.526835
  - gust calibrated MAE: 1.134749

Comparison:
  - Better than phys_v3_dem_fetch 150k wind RMSE 1.305533 by 0.006249.
  - Better than old_signal 225k wind RMSE 1.301797 by 0.002513.
  - Still worse than champion wind RMSE 1.268019 by 0.031265.

Interpretation:
  Pruning raw static sector grids and EUMETSAT helps a little. The gain is real
  but small, so generic feature pruning is not enough. The important remaining
  test is phys_v4_directional: if direction-conditioned geometry is the missing
  physical representation, it should beat pruned_v3 on the same 150k rail.

Next running:
  phys_v4_directional_pruned_150k_bin63

## 2026-06-30T12:50+02:00 - pruned/v4 directional campaign final result

Remote summary:
  /srv/data/corsewind/ml_dataset/benchmarks/pruned_v4_directional_150k_resume_summary.json

Decision:
  keep_champion

Results:
  - champion:
      wind mean RMSE 1.268019
      wind mean MAE 0.930465
  - phys_v3_old_signal_225k_bin63:
      wind mean RMSE 1.301797
      wind mean MAE 0.982492
      gust RMSE 1.516817
      gust MAE 1.124876
  - phys_v3_pruned_150k_bin63:
      wind mean RMSE 1.299284
      wind mean MAE 0.980664
      gust RMSE 1.526835
      gust MAE 1.134749
  - phys_v4_directional_pruned_150k_bin63:
      wind mean RMSE 1.312789
      wind mean MAE 0.992090
      gust RMSE 1.521491
      gust MAE 1.131693
  - phys_v3_pruned_200k_bin63 partial base-only:
      wind mean base RMSE 1.311003
      wind mean base MAE 0.993864
      gust base RMSE 1.534341
      gust base MAE 1.144583

Interpretation:
  - The best new wind-mean run is phys_v3_pruned_150k_bin63 at 1.299284.
  - It improves phys_v3_dem_fetch_150k_bin63 but remains 0.031265 above the
    champion.
  - The v4 directional features as currently built are harmful for global wind
    mean RMSE. They may contain a weak gust signal, but not enough to justify
    promotion.
  - More rows and generic pruning helped by only a few thousandths, so the gap
    to champion is not explained solely by dataset organization.

Next scientific conclusion:
  Stop adding broad static/directional feature families to the global model.
  The next logical path is a gated specialist or router:
    use phys_v3/pruned/v4 only on spots, leads, or gust regimes where they beat
    the champion/common baseline, otherwise keep champion.
