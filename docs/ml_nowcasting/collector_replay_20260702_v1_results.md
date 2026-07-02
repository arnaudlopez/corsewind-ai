# Collector Replay 2026-07-02 v1

## Objective

Move from a two-case smoke test to a wider pseudo-live replay using fresh
collector data from z2.

Goal:

- validate wind mean corrections without calm-regime regression;
- validate strong gust behavior on `>=20kt` and `>=25kt`;
- decide whether `strong_gated`, deterministic gust rules, router, or stacker
  deserve shadow/live promotion.

## Inputs

Cases file:

`configs/ml_collector_hindcast_cases_20260702_replay_v1.json`

z2 output root:

`/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_replay_20260702_v1`

The suite ran 14 requested cases:

- 13 completed and scored;
- 1 failed: `collector_20260702T0245_replay_v1`, likely because the day was
  still too fresh/incomplete for the full pipeline at run time.

Scored rows:

- `1405` rows after joining predictions with observations;
- 7 scored spots: `cap_corse`, `la_parata`, `lfkf`, `lfkj`, `lfks`, `lfvf`,
  `lfvh`;
- validation for router/stacker: leave-one-hindcast-out.

## Suite-Level Result

The wide replay confirms the smoke-test direction for wind mean:

- `strong_gated` beats raw in 12/13 scored cases on wind mean RMSE;
- `strong_gated` is especially useful on the 2026-07-01 strong-wind windows.

But gust behavior is more nuanced:

- `gust_high` helps in some moderate/strong cases;
- raw remains important for `gust >=25kt`;
- deterministic `rule_raw25_else_high` is safer than the champion for strong
  detection, but not the best global RMSE candidate on the full replay.

## Router/Stacker Results

Output:

`/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_replay_20260702_v1/router_v1_rules_all_cases`

Final shadow artifact output:

`/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_replay_20260702_v1/router_v1_shadow_artifact_all_cases`

Created files:

- `router_v1_final_models.joblib` (`11M`)
- `router_v1_final_models_metadata.json`
- `router_v1_results.json`
- `router_v1_results.md`
- OOF prediction parquets for wind and gust

Artifact load check passed on z2:

- format: `corsewind.hindcast_router_v1_final_models`
- targets: `wind`, `gust`
- training rows per target: `1405`
- features per target: `441`
- model types: sklearn `Pipeline` classifier + sklearn `Pipeline` regressor

Metric cells are knots.

### Wind Mean

| Rail | RMSE | MAE | Bias | Notes |
| --- | ---: | ---: | ---: | --- |
| champion | 4.215 | 2.878 | -1.163 | current baseline |
| raw | 4.494 | 3.189 | -1.836 | worse than champion |
| strong_gated | 3.933 | 2.745 | -0.865 | stable practical rail |
| router_classifier | 3.949 | 2.760 | -1.043 | improves champion, not better than `strong_gated` |
| stacker_regressor | 3.374 | 2.535 | +0.011 | best OOF candidate |
| oracle | 3.745 | 2.396 | -1.183 | best candidate selector only |

Important: `stacker_regressor` beats the candidate-selection oracle because it
is not limited to choosing one existing rail. It can interpolate/extrapolate.

Wind threshold CSI:

| Rail | `>=12kt` | `>=15kt` | `>=20kt` | `>=25kt` |
| --- | ---: | ---: | ---: | ---: |
| champion | 0.396 | 0.250 | 0.000 | 0.000 |
| raw | 0.398 | 0.288 | 0.000 | 0.000 |
| strong_gated | 0.440 | 0.415 | 0.061 | 0.000 |
| router_classifier | 0.448 | 0.410 | 0.061 | 0.000 |
| stacker_regressor | 0.516 | 0.545 | 0.620 | 0.260 |

Conclusion wind:

- `strong_gated` is now a defensible shadow rail;
- `stacker_regressor` is the first candidate that materially attacks the
  strong-wind underprediction problem;
- do not promote directly until tested on a later collector day not included in
  this replay.

### Gusts

| Rail | RMSE | MAE | Bias | Notes |
| --- | ---: | ---: | ---: | --- |
| champion | 5.854 | 4.017 | -2.697 | current baseline |
| raw | 5.902 | 4.388 | -0.147 | good for high-gust recall, noisy in calm |
| gust_high | 5.473 | 3.946 | -1.012 | best simple static rail |
| strong_gated | 5.540 | 3.813 | -2.394 | good MAE, too conservative at high gusts |
| rule_raw25_else_high | 5.696 | 4.016 | -0.846 | preserves `>=25kt` CSI but not best RMSE |
| router_classifier | 5.115 | 3.624 | -1.491 | best selector |
| stacker_regressor | 4.879 | 3.543 | +0.054 | best OOF candidate |
| oracle | 4.550 | 2.862 | -1.505 | remaining routeability exists |

Gust threshold CSI:

| Rail | `>=15kt` | `>=20kt` | `>=25kt` |
| --- | ---: | ---: | ---: |
| champion | 0.416 | 0.379 | 0.000 |
| raw | 0.497 | 0.412 | 0.392 |
| gust_high | 0.503 | 0.464 | 0.276 |
| strong_gated | 0.454 | 0.463 | 0.047 |
| rule_raw25_else_high | 0.503 | 0.457 | 0.392 |
| router_classifier | 0.524 | 0.465 | 0.341 |
| stacker_regressor | 0.557 | 0.559 | 0.503 |
| oracle | 0.625 | 0.552 | 0.473 |

Conclusion gust:

- `rule_raw25_else_high` is useful as a conservative fallback because it keeps
  raw's `>=25kt` CSI while reducing bias versus raw.
- The stronger candidate is now `stacker_regressor`: it improves RMSE, MAE,
  bias, and all threshold CSI columns, including `gust >=25kt`.
- This is the first replay where a learned blend looks clearly useful on fresh
  collector data.

## Physical Signal Warning

The router/stacker run still reports fully missing columns:

- `features__thermal_coastal_minus_inland_*`
- `features__thermal_relief_minus_coastal_*`
- `features__thermal_recent_*`
- `features__nwp_offset_*_boundary_layer_height`

Interpretation:

- the replay is already useful, but it is not yet the complete physical design;
- the next improvement path is not just model tuning;
- we should fill these feature gaps, especially thermal station deltas and
  boundary-layer offsets, then rerun the same replay.

## Decision

Do not promote a new production champion yet.

Promote to shadow candidates:

- wind mean: `strong_gated` as stable conservative rail;
- wind mean: `stacker_regressor` as experimental strong-wind rail;
- gusts: `stacker_regressor` as experimental main candidate;
- gusts: `rule_raw25_else_high` as deterministic safety fallback for high-gust
  detection.

Promotion gate for production:

1. Replay on at least one newly collected day not included in this suite.
2. Keep wind calm regime close to champion:
   - `<12kt` RMSE must not degrade materially.
3. Beat champion on wind mean global RMSE and improve `wind >=15kt` CSI.
4. Beat champion on gust global RMSE and improve `gust >=20kt` / `>=25kt` CSI.
5. Confirm no obvious leakage in stacker features and training split.
6. Fill or explicitly remove all always-missing physical columns before
   freezing the production pipeline.

## Next Work

Immediate next step:

- run `router_v1_shadow_artifact_all_cases` on the next fresh collector day as a
  true forward shadow;
- compare after observations arrive.

Second step:

- fix feature availability for thermal deltas and boundary-layer offsets;
- rerun this exact replay suite to see whether physical v2 improves beyond the
  current stacker.

## 2026-07-02 Shadow Plumbing Update

Implemented reusable shadow application/scoring:

- `scripts/ml_dataset/apply_hindcast_router_v1_artifact.py`
  - reads a prediction parquet;
  - loads `router_v1_final_models.joblib`;
  - adds `shadow_router_v1_*` and `shadow_stacker_v1_*` columns;
  - preserves all champion/raw/current columns.
- `scripts/ml_dataset/score_live_hindcast_predictions.py`
  - now scores shadow router/stacker columns in overall, grouped, threshold,
    and peak summaries.
- `scripts/ml_dataset/run_collector_hindcast_suite.py`
  - now accepts `--shadow-artifact`;
  - after each hindcast, it can apply and score the shadow model automatically;
  - suite Markdown now includes shadow RMSE and threshold CSI tables.

Smoke test on z2:

`/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_suite_smoke_20260702`

Smoke case:

`collector_20260701T0645_replay_v1`

Important caveat:

- this smoke reuses a case included in the final artifact training set;
- it proves the end-to-end shadow pipeline, not generalization;
- the stacker numbers are therefore in-sample and must not be used for
  promotion.

Smoke output in m/s:

| Rail | Wind RMSE | Gust RMSE |
| --- | ---: | ---: |
| raw | 3.369 | 3.703 |
| champion | 3.216 | 4.278 |
| strong_gated | 2.892 | 3.863 |
| shadow_router_v1 | 2.888 | 3.332 |
| shadow_stacker_v1 | 0.471 | 0.599 |

Threshold CSI smoke:

| Rail | Wind `>=15kt` | Gust `>=20kt` | Gust `>=25kt` |
| --- | ---: | ---: | ---: |
| raw | 0.325 | 0.516 | 0.542 |
| champion | 0.325 | 0.573 | 0.000 |
| strong_gated | 0.537 | 0.608 | 0.120 |
| shadow_router_v1 | 0.543 | 0.649 | 0.481 |
| shadow_stacker_v1 | 0.857 | 0.929 | 0.885 |

Operational consequence:

- `shadow_router_v1` is now deployable as a measured shadow rail.
- `shadow_stacker_v1` remains the most promising candidate but needs true
  unseen-day shadow validation before any champion decision.

## 2026-07-02 First Unseen Shadow Validation

After the plumbing smoke, a first short unseen validation was run on z2 using
data not included in `router_v1_shadow_artifact_all_cases`.

Output:

`/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1`

Preparation:

- collected Open-Meteo `meteofrance_arome_france` for `2026-07-02`;
- used available Meteo-France observations through about `06:36Z`;
- scored only the already observable morning window, not the windsurf day.

Cases:

| Case | Window UTC | Rows |
| --- | --- | ---: |
| `collector_20260702T0245_unseen_v1` | `03:00` to `06:30` | 83 |
| `collector_20260702T0345_unseen_v1` | `04:00` to `06:30` | 61 |

Overall RMSE in m/s:

| Case | Wind raw | Wind champion | Wind strong | Wind router | Wind stacker | Gust raw | Gust champion | Gust high | Gust router | Gust stacker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0245` | 1.533 | 1.401 | 1.351 | 1.359 | 1.450 | 1.808 | 2.005 | 1.879 | 1.876 | 2.234 |
| `0345` | 1.592 | 1.354 | 1.259 | 1.181 | 1.381 | 1.757 | 2.055 | 1.826 | 1.850 | 1.651 |

Threshold CSI:

| Case | Wind `>=15kt` raw | Wind `>=15kt` strong | Wind `>=15kt` router | Gust `>=25kt` raw | Gust `>=25kt` champion | Gust `>=25kt` router | Gust `>=25kt` stacker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0245` | 0.600 | 0.143 | 0.143 | 1.000 | 0.000 | 1.000 | 0.000 |
| `0345` | 0.750 | 0.500 | 0.750 | 1.000 | 0.000 | 1.000 | 0.000 |

Interpretation:

- This is the first real unseen signal for the shadow artifact.
- `shadow_stacker_v1` is not reliable enough for promotion:
  - good on the `0345` gust RMSE;
  - bad on the `0245` gust RMSE;
  - misses all `gust >=25kt` events in both unseen morning cases.
- `shadow_router_v1` is safer than stacker for strong gust detection:
  - it keeps `gust >=25kt` CSI at `1.0` in both short cases;
  - it improves wind RMSE on `0345`;
  - it does not beat `strong_gated` on wind `0245`.
- The champion still under-detects strong gusts:
  - `gust >=25kt` CSI is `0.0` in both unseen cases.

Decision update:

- no production promotion;
- keep `shadow_router_v1` as the safer measured shadow rail;
- downgrade `shadow_stacker_v1` from "promising candidate" to "needs
  guardrails or regime gating before further consideration";
- rerun on the full `2026-07-02` windsurf window when observations through
  `17:00Z` are available.

Prepared full-day validation:

- config: `configs/ml_collector_shadow_cases_20260702_full_day_v1.json`
- z2 planned output:
  `/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_full_day_v1`
- cases: `06:45Z`, `08:45Z`, `10:45Z`, `12:45Z` issues, all scored through
  `17:00Z`
- dry-run passed on z2.

Command to run after observations through `2026-07-02T17:00:00Z` are present:

```bash
cd /srv/data/corsewind/backfill_runner
/srv/data/corsewind/pyenv/bin/python scripts/ml_dataset/run_collector_hindcast_suite.py \
  --output-root /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_full_day_v1 \
  --cases-json configs/ml_collector_shadow_cases_20260702_full_day_v1.json \
  --source aromepi \
  --shadow-artifact /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_replay_20260702_v1/router_v1_shadow_artifact_all_cases/router_v1_final_models.joblib \
  --shadow-allow-missing-features \
  --continue-on-error \
  --python /srv/data/corsewind/pyenv/bin/python
```

Watcher status:

- script: `scripts/ml_dataset/z2_watch_20260702_full_day_shadow.sh`
- z2 PID file:
  `/srv/data/corsewind/ml_dataset/live_inference/watch_logs/20260702_full_day_shadow.pid`
- z2 log:
  `/srv/data/corsewind/ml_dataset/live_inference/watch_logs/20260702_full_day_shadow.log`
- started at: `2026-07-02T06:57:12Z`
- active PID at start: `85382`
- cadence: refresh Meteo-France observations every `900s`
- launch condition: every scored spot has an observation timestamp at or after
  `2026-07-02T17:00:00Z`

Latest coverage before watcher launch:

| Spot | Latest observation UTC |
| --- | --- |
| `cap_corse` | `2026-07-02T06:48:00Z` |
| `la_parata` | `2026-07-02T06:00:00Z` |
| `lfkf` | `2026-07-02T06:48:00Z` |
| `lfkj` | `2026-07-02T06:48:00Z` |
| `lfks` | `2026-07-02T06:48:00Z` |
| `lfvf` | `2026-07-02T06:48:00Z` |
| `lfvh` | `2026-07-02T06:00:00Z` |

Current state:

- waiting for observations;
- no full-day score yet;
- goal remains active.

## 2026-07-02 Guarded Stacker Retest

The first unseen validation showed a useful but unsafe pattern:

- `shadow_stacker_v1` can reduce gust RMSE on some cases;
- but it missed all `gust >=25kt` events in both short unseen cases;
- `shadow_router_v1` was safer for strong gust detection.

Implemented guarded stacker rule:

- wind: use stacker normally, but when router predicts `>=15kt`, keep at least
  the router value;
- gust: use stacker normally, but when router predicts `>=20kt`, keep at least
  the router value.

Updated scripts:

- `scripts/ml_dataset/apply_hindcast_router_v1_artifact.py`
- `scripts/ml_dataset/score_live_hindcast_predictions.py`
- `scripts/ml_dataset/run_collector_hindcast_suite.py`

Validation:

- local `py_compile`: passed;
- z2 sync: done;
- z2 `py_compile`: passed;
- z2 noise check: pandas `PerformanceWarning` removed from shadow application;
- rescored the two short unseen cases with:
  `hindcast_score_with_guarded_stacker_v1.json`.

Overall RMSE in m/s:

| Case | Wind router | Wind stacker | Wind guarded | Gust raw | Gust router | Gust stacker | Gust guarded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0245` | 1.359 | 1.450 | 1.450 | 1.808 | 1.876 | 2.234 | 1.884 |
| `0345` | 1.181 | 1.381 | 1.381 | 1.757 | 1.850 | 1.651 | 1.596 |

Strong-threshold CSI:

| Case | Wind `>=15kt` router | Wind `>=15kt` guarded | Gust `>=20kt` stacker | Gust `>=20kt` guarded | Gust `>=25kt` stacker | Gust `>=25kt` guarded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0245` | 0.143 | 0.143 | 0.000 | 0.300 | 0.000 | 0.500 |
| `0345` | 0.750 | 0.750 | 0.750 | 0.500 | 0.000 | 1.000 |

Guard activation share:

| Case | Wind guarded by router | Gust guarded by router |
| --- | ---: | ---: |
| `0245` | 1.0% | 8.0% |
| `0345` | 0.0% | 9.5% |

Interpretation:

- Guarding fixes the dangerous failure mode on `gust >=25kt`.
- On `0345`, guarded stacker becomes the best gust RMSE among tested ML
  variants and beats raw gust RMSE.
- On `0245`, guarded stacker is still slightly worse than raw/router gust RMSE,
  but much safer than raw stacker for thresholds.
- For wind mean, router remains better than guarded stacker in both short
  unseen cases.

Decision update:

- no production promotion yet;
- keep `shadow_router_v1` as the safer wind-mean shadow candidate;
- keep `shadow_guarded_stacker_v1` as the next gust candidate to test on the
  full-day unseen validation;
- do not consider raw `shadow_stacker_v1` for promotion without guardrails.

Operational status:

- the z2 watcher is still active;
- the full-day output directory only contains a preparatory summary so far, not
  scored cases;
- the next automatic full-day suite will use the synced guarded-stack scripts.
- latest checked watcher cycle: `2026-07-02T07:12:34Z`;
- latest observations at that cycle:
  - `cap_corse`, `lfkf`, `lfkj`, `lfks`, `lfvf`: `2026-07-02T07:06:00Z`;
  - `la_parata`, `lfvh`: `2026-07-02T07:00:00Z`;
- still waiting for all score spots to reach `2026-07-02T17:00:00Z`.

## Repeatable Daily Shadow Rail

To move from isolated experiments to a reliable champion decision, the daily
shadow suite is now reusable.

Added:

- `scripts/ml_dataset/generate_collector_shadow_cases.py`
- `scripts/ml_dataset/z2_watch_shadow_suite.sh`

Validated:

- local Python compile: passed;
- local Bash syntax check: passed;
- z2 Python compile: passed;
- z2 Bash syntax check: passed;
- generated test cases for `2026-07-02` match the manually written full-day
  cases;
- generated and synced the next-day config:
  `configs/ml_collector_shadow_cases_20260703_full_day_v1.json`.

Generate a new daily validation config:

```bash
python3 scripts/ml_dataset/generate_collector_shadow_cases.py \
  --date 2026-07-03 \
  --output configs/ml_collector_shadow_cases_20260703_full_day_v1.json \
  --overwrite
```

Launch the generic watcher on z2 for a prepared date:

```bash
cd /srv/data/corsewind/backfill_runner
setsid env TARGET_DATE=2026-07-03 \
  scripts/ml_dataset/z2_watch_shadow_suite.sh \
  >/srv/data/corsewind/ml_dataset/live_inference/watch_logs/20260703_full_day_v1_shadow.launch.log 2>&1 &
```

Decision implication:

- do not promote from `2026-07-02` alone;
- collect at least one more fresh full-day shadow validation with the same
  suite shape;
- compare candidates on identical cases before changing any champion.

## Multi-Suite Aggregator

Added:

- `scripts/ml_dataset/summarize_shadow_suites.py`

Purpose:

- aggregate multiple `suite_summary.json` files;
- weight RMSE/MAE/bias by scored row count;
- aggregate threshold `tp/fp/fn/tn` before recomputing CSI;
- automatically read guarded re-score files when present in case directories.

Validated on z2 against the two short unseen cases:

```bash
cd /srv/data/corsewind/backfill_runner
/srv/data/corsewind/pyenv/bin/python scripts/ml_dataset/summarize_shadow_suites.py \
  --suite-summary /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/suite_summary.json \
  --output-json /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/shadow_aggregate_with_guarded_v1.json \
  --output-markdown /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/shadow_aggregate_with_guarded_v1.md
```

Short unseen aggregate, 144 rows:

| Rail | Wind RMSE m/s | Gust RMSE m/s |
| --- | ---: | ---: |
| raw | 1.558 | 1.786 |
| champion | 1.381 | 2.026 |
| strong/high | 1.313 | 1.857 |
| router | 1.286 | 1.865 |
| stacker | 1.421 | 2.008 |
| guarded stacker | 1.421 | 1.768 |

Threshold aggregate:

| Rail | Wind `>=15kt` CSI | Gust `>=25kt` CSI |
| --- | ---: | ---: |
| raw | 0.667 | 1.000 |
| router | 0.364 | 1.000 |
| stacker | 0.364 | 0.000 |
| guarded stacker | 0.364 | 0.667 |

Interpretation:

- short unseen aggregate confirms `wind_router` is the best wind-mean shadow
  candidate so far;
- guarded stacker is the best gust RMSE candidate on this short aggregate;
- raw/router/high still dominate `gust >=25kt` CSI on this tiny sample;
- this is still too short and too morning-biased for promotion.

## Shadow Promotion Gate

Added:

- `scripts/ml_dataset/assert_shadow_promotion_gate.py`

Purpose:

- prevent promotion from a single attractive metric;
- require multi-day/multi-case evidence;
- require global RMSE gain versus baselines;
- require no material CSI regression on windsurf thresholds;
- require no material RMSE regression on calm regimes.

Default strict requirements:

- at least `2` issue days;
- at least `6` scored cases;
- at least `6` shadow cases;
- at least `500` joined rows;
- candidate RMSE must beat each baseline by at least `0.02 m/s`;
- threshold CSI may regress by at most `0.02`;
- calm-regime RMSE may regress by at most `0.02 m/s`.

Validated on z2:

```bash
cd /srv/data/corsewind/backfill_runner
/srv/data/corsewind/pyenv/bin/python scripts/ml_dataset/assert_shadow_promotion_gate.py \
  --aggregate-json /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/shadow_aggregate_with_guarded_v1.json \
  --target wind \
  --candidate router \
  --output-json /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/wind_router_promotion_gate_v1.json \
  --output-markdown /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/wind_router_promotion_gate_v1.md

/srv/data/corsewind/pyenv/bin/python scripts/ml_dataset/assert_shadow_promotion_gate.py \
  --aggregate-json /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/shadow_aggregate_with_guarded_v1.json \
  --target gust \
  --candidate guarded_stacker \
  --output-json /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/gust_guarded_stacker_promotion_gate_v1.json \
  --output-markdown /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/gust_guarded_stacker_promotion_gate_v1.md
```

Current gate decisions:

| Candidate | Decision | Main blockers |
| --- | --- | --- |
| `wind_router` | `do_not_promote` | only `1` day, `2` cases, `144` rows; `wind >=15kt` CSI regresses versus raw |
| `gust_guarded_stacker` | `do_not_promote` | only `1` day, `2` cases, `144` rows; RMSE gain versus raw is below strict `0.02 m/s`; `gust >=25kt` CSI regresses versus raw |

Decision:

- the gate is working as intended;
- current short unseen results are useful research evidence, not promotion
  evidence;
- rerun the gate after the full-day `2026-07-02` suite and at least one more
  fresh day are aggregated.

## Automated Shadow Postprocess

Added:

- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`

The generic watcher now runs postprocessing automatically after a completed
suite:

- `scripts/ml_dataset/z2_watch_shadow_suite.sh`

Postprocessing does three things:

1. Build `shadow_aggregate_v1.json` and `shadow_aggregate_v1.md`.
2. Run the wind gate for `router`.
3. Run the gust gate for `guarded_stacker`.

The promotion gates are run with `--no-fail-on-reject`, because `do_not_promote`
is a valid expected state while evidence is insufficient.

Validated on z2 against the short unseen output:

```bash
cd /srv/data/corsewind/backfill_runner
OUTPUT_ROOT=/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1 \
  scripts/ml_dataset/run_shadow_suite_postprocess.sh
```

Validation result:

- aggregate format: `corsewind.shadow_suite_aggregate.v1`;
- wind gate: `do_not_promote`;
- gust gate: `do_not_promote`;
- script exits successfully despite rejection decisions.

Important operational note:

- the already running `2026-07-02` watcher loaded its Bash functions before this
  postprocess hook was added;
- if it finishes without postprocess artifacts, run the postprocess command
  manually with:

```bash
cd /srv/data/corsewind/backfill_runner
OUTPUT_ROOT=/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_full_day_v1 \
  scripts/ml_dataset/run_shadow_suite_postprocess.sh
```

## Autonomous Full-Day Postprocess Watcher

Added:

- `scripts/ml_dataset/z2_watch_shadow_postprocess.sh`

Purpose:

- monitor an `OUTPUT_ROOT`;
- wait until `suite_summary.json` contains scored cases and shadow scores for
  every case;
- run `run_shadow_suite_postprocess.sh` automatically;
- exit immediately if postprocess artifacts already exist.

Validated on z2:

- complete short unseen suite: detected `2/2` scored and `2/2` shadow cases,
  ran postprocess successfully;
- incomplete full-day suite: detected `0/4` scored and `0/4` shadow cases,
  waited/retried as expected.

Launched for the `2026-07-02` full-day suite:

- main full-day watcher PID: `85382`;
- postprocess watcher PID: `86704`;
- postprocess log:
  `/srv/data/corsewind/ml_dataset/live_inference/watch_logs/20260702_full_day_postprocess.log`;
- target output:
  `/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_full_day_v1`.

Current postprocess status:

- waiting for full-day suite completion;
- last status: `4` planned cases, `0` scored cases, `0` shadow cases;
- no promotion decision can be made from the full-day run yet.

## Shadow Validation Status Command

Added:

- `scripts/ml_dataset/shadow_validation_status.py`

Purpose:

- summarize both watchers;
- report latest observation coverage;
- report full-day suite completion;
- report postprocess artifacts and gate decisions;
- report free disk space.

Validated on z2:

```bash
cd /srv/data/corsewind/backfill_runner
/srv/data/corsewind/pyenv/bin/python scripts/ml_dataset/shadow_validation_status.py \
  --target-date 2026-07-02 \
  --output-json /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_full_day_v1/shadow_validation_status_v1.json \
  --output-markdown /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_full_day_v1/shadow_validation_status_v1.md
```

Current status snapshot:

- main watcher: running, PID `85382`;
- postprocess watcher: running, PID `86704`;
- health: `waiting_for_observations`, `ok=True`;
- latest coverage age: `3.872` minutes at last check;
- suite complete: `False`;
- cases scored: `0/4`;
- shadow cases: `0/4`;
- latest observation coverage: `2026-07-02T07:00Z` to
  `2026-07-02T07:18Z` depending on spot;
- postprocess artifacts: not present yet.

Health states:

- `waiting_for_observations`: watchers alive, suite incomplete, coverage fresh;
- `awaiting_postprocess`: suite complete, postprocess not finished yet;
- `postprocessed`: aggregate and gates exist;
- `attention`: watcher missing, stale coverage, or inconsistent state;
- `running_or_partial_suite`: some cases scored but suite not complete.

## Daily Shadow Validation Launcher

Added:

- `scripts/ml_dataset/z2_launch_shadow_validation_day.sh`

Purpose:

- generate the daily full-day shadow case config;
- launch the main shadow watcher;
- launch the autonomous postprocess watcher;
- use stable PID/log paths derived from `TARGET_DATE`.

Validated:

- local `PREPARE_ONLY=1` for `2026-07-03`: generated 4 cases;
- z2 `PREPARE_ONLY=1` for `2026-07-03`: generated/synced
  `configs/ml_collector_shadow_cases_20260703_full_day_v1.json`;
- no watchers were launched during this prepare-only validation.

Prepare-only command:

```bash
cd /srv/data/corsewind/backfill_runner
PREPARE_ONLY=1 TARGET_DATE=2026-07-03 FORCE_CASES=1 \
  scripts/ml_dataset/z2_launch_shadow_validation_day.sh
```

Launch command for a real daily validation:

```bash
cd /srv/data/corsewind/backfill_runner
TARGET_DATE=2026-07-03 \
  scripts/ml_dataset/z2_launch_shadow_validation_day.sh
```

Operational note:

- do not launch the `2026-07-03` watcher too early with default `MAX_LOOPS`,
  because it needs to survive until observations reach `2026-07-03T17:00:00Z`;
- launch tomorrow morning, or increase `MAX_LOOPS`/`POST_MAX_LOOPS` if launching
  much earlier.

## Multi-Day Shadow Rollup

Added:

- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh`

Purpose:

- discover complete shadow suites under
  `/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen*/suite_summary.json`;
- skip incomplete placeholders;
- aggregate all complete suites;
- run the same wind/gust promotion gates on the aggregate.

Validated on z2:

```bash
cd /srv/data/corsewind/backfill_runner
ROLLUP_ID=shadow_rollup_latest MIN_COMPLETE_SUITES=1 \
  scripts/ml_dataset/run_shadow_multi_day_rollup.sh
```

Current rollup output:

- `/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest`

Current discovered suites:

- `/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/suite_summary.json`

Current rollup status:

- complete suite count: `1`;
- aggregate cases: `2`;
- aggregate rows: `144`;
- wind router gate: `do_not_promote`;
- gust guarded stacker gate: `do_not_promote`.

Interpretation:

- the rollup works and correctly ignores the incomplete full-day placeholder;
- it is not promotion evidence yet;
- rerun after the `2026-07-02` full-day suite completes, then again after
  `2026-07-03` is collected.

## Automatic Rollup Hook

Updated:

- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`

Behavior:

- daily postprocess still writes daily aggregate and daily gates;
- after that, it now refreshes `shadow_rollup_latest` automatically;
- can be disabled with `RUN_ROLLUP=0`.

Validation:

- tested on the short unseen suite with `ROLLUP_ID=shadow_rollup_hook_test`;
- discovered one complete suite;
- aggregate rows: `144`;
- wind gate: `do_not_promote`;
- gust gate: `do_not_promote`.

Implementation note:

- first test revealed an environment leak: daily `OUTPUT_ROOT` was inherited by
  the rollup and caused rollup files to be written into the daily directory;
- fixed by unsetting `OUTPUT_ROOT` in a subshell before calling
  `run_shadow_multi_day_rollup.sh`;
- stale test artifacts written in the wrong daily directory were removed on z2.

Consequence:

- when the full-day `2026-07-02` postprocess eventually runs, the multi-day
  rollup should refresh automatically too.

## Status Dashboard With Rollup

Updated:

- `scripts/ml_dataset/shadow_validation_status.py`

Purpose:

- keep one command for the live validation state;
- include watcher health, observation freshness, suite completion, daily
  artifacts, disk usage, and now the latest multi-day rollup/gate decisions.

Validated on z2:

```bash
cd /srv/data/corsewind/backfill_runner
/srv/data/corsewind/pyenv/bin/python scripts/ml_dataset/shadow_validation_status.py \
  --target-date 2026-07-02 \
  --output-json /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_full_day_v1/shadow_validation_status_v1.json \
  --output-markdown /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_full_day_v1/shadow_validation_status_v1.md
```

Current status at validation time:

- health: `waiting_for_observations`, ok `True`;
- coverage age: about `9.5` minutes;
- full-day suite: `0/4` scored, `0/4` shadow;
- rollup complete suites: `1`;
- rollup rows: `144`;
- wind router gate: `do_not_promote`;
- gust guarded stacker gate: `do_not_promote`.

Interpretation:

- the full-day validation is still waiting for the day to complete;
- the rollup is wired but still only contains the short unseen morning suite;
- no model promotion is allowed until the full-day and multi-day evidence pass
  the gates.

## Multi-Day Shadow Campaign

Added:

- `scripts/ml_dataset/z2_run_shadow_validation_campaign.sh`

Purpose:

- launch daily shadow validations automatically at the right UTC time;
- avoid starting watchers too early and consuming their wait window;
- accumulate several fresh unseen days without manual relaunch.

Validated:

- local prepare-only smoke test;
- z2 prepare-only smoke test;
- z2 background campaign launched.

Current z2 campaign:

- campaign id: `shadow_campaign_20260703_3d_v1`;
- PID: `87605`;
- date range: `2026-07-03` to `2026-07-05`;
- launch time: `05:30 UTC`;
- current state: sleeping until the `2026-07-03` launch.

This gives us the first real multi-day validation path: the 2026-07-02 full-day
suite, then 2026-07-03/04/05 daily suites, all feeding the same
`shadow_rollup_latest` promotion gates.

## Business Threshold Gates

Updated:

- `scripts/ml_dataset/score_live_hindcast_predictions.py`
- `scripts/ml_dataset/summarize_shadow_suites.py`
- `scripts/ml_dataset/assert_shadow_promotion_gate.py`

Change:

- scoring now emits threshold metrics at `12/15/20/25 kt` for wind and gust;
- aggregate rollups preserve those thresholds for raw/champion/current/shadow
  rails;
- promotion gates now require the candidate not to break those threshold CSI
  checks by default.

Validated on z2:

- Python compilation passed;
- scripts synced to `/srv/data/corsewind/backfill_runner`;
- `shadow_rollup_latest` regenerated.

Current rollup after stricter gates:

- rows: `144`;
- threshold keys available from old scores: `20`;
- wind router gate: `do_not_promote`;
- gust guarded stacker gate: `do_not_promote`.

Important interpretation:

- old morning scores were generated before all `12/15/20/25 kt` thresholds
  existed, so some new threshold checks are still missing on that old evidence;
- the running full-day suite and future campaign days will use the updated
  score script and should produce the full threshold grid;
- this is the right behavior: a model cannot be promoted from incomplete
  threshold evidence.

## Campaign-Aware Status Dashboard

Updated:

- `scripts/ml_dataset/shadow_validation_status.py`

Change:

- optional `--campaign-id` support;
- reports campaign PID, log path, and latest campaign log line in the same
  status JSON/Markdown as watcher, postprocess, rollup, and disk.

Validated on z2:

```bash
/srv/data/corsewind/pyenv/bin/python scripts/ml_dataset/shadow_validation_status.py \
  --target-date 2026-07-02 \
  --campaign-id shadow_campaign_20260703_3d_v1
```

Current campaign evidence:

- campaign id: `shadow_campaign_20260703_3d_v1`;
- running: `True`;
- PID: `87605`;
- latest line: waiting before launching target date `2026-07-03`.

## Re-Score Existing Shadow Suites

Added:

- `scripts/ml_dataset/rescore_shadow_suite_metrics.py`

Purpose:

- recompute score JSON files for an existing suite when metrics evolve;
- do not rerun AROME or model inference;
- rescore available prediction files:
  - `predictions.parquet`;
  - `predictions_with_shadow_router_v1.parquet`;
  - `predictions_with_guarded_stacker_v1.parquet`.

Validated on z2:

- dry-run on `/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_shadow_unseen_20260702_v1/suite_summary.json`;
- real re-score with `--overwrite`;
- regenerated `shadow_rollup_latest`.

Result after re-score:

- rollup rows: `144`;
- threshold keys: `52`, matching the available rails over `12/15/20/25 kt`;
- wind gate: `do_not_promote`;
- gust gate: `do_not_promote`.

Gate refinement:

- `assert_shadow_promotion_gate.py` now treats a threshold with no events as
  non-informative instead of a hard failure;
- after this fix, `wind >=25 kt` no longer creates an artificial rejection when
  the sample contains no event at that threshold.

Current real rejection reasons:

- evidence still too small: `1` day, `2` cases, `144` rows;
- wind router improves global RMSE but regresses `wind >=15 kt` CSI vs raw;
- gust guarded stacker does not beat raw RMSE by the required margin and
  regresses several gust threshold CSI checks.

## Multi-Candidate Promotion Review

Added:

- `scripts/ml_dataset/review_shadow_promotion_candidates.py`

Purpose:

- evaluate every available candidate rail against the same promotion gates;
- avoid hard-coding the decision around only `wind_router` and
  `gust_guarded_stacker`;
- make RMSE-vs-threshold tradeoffs explicit before any promotion.

Default candidates:

- wind: `strong_gated`, `router`, `stacker`, `guarded_stacker`;
- gust: `high`, `strong_gated`, `router`, `stacker`, `guarded_stacker`.

Pipeline integration:

- `scripts/ml_dataset/run_shadow_suite_postprocess.sh` now writes daily
  `promotion_candidate_review_v1.json/md`;
- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh` now writes rollup
  `promotion_candidate_review.json/md`;
- `scripts/ml_dataset/shadow_validation_status.py` now reports the best
  candidate per target from the rollup review.

Validated on z2:

- regenerated `shadow_rollup_latest`;
- produced
  `/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/promotion_candidate_review.json`;
- dashboard confirms the review exists.

Current candidate review on the small unseen rollup:

- best wind candidate by gate sort: `router`;
- wind decision: `do_not_promote`;
- wind failed checks: `5`;
- wind RMSE: `1.286 m/s`;
- best gust candidate by gate sort: `high`;
- gust decision: `do_not_promote`;
- gust failed checks: `8`;
- gust RMSE: `1.857 m/s`.

Interpretation:

- `router` is currently the most promising wind rail, but it still fails the
  evidence-size checks and regresses `wind >=15 kt` CSI vs raw;
- `guarded_stacker` has a lower gust RMSE than `high`, but `high` currently
  fails fewer gate checks, which is why the review ranks it first;
- this distinction is useful: future promotion should consider both continuous
  error and threshold behavior, not just RMSE.

## Candidate Review v2: Evidence vs Performance

Updated:

- `scripts/ml_dataset/review_shadow_promotion_candidates.py`
- `scripts/ml_dataset/shadow_validation_status.py`

Change:

- candidate review now separates:
  - global evidence failures: not enough days, cases, shadow cases, rows;
  - performance failures: RMSE, threshold CSI, calm-regime regressions;
- review also exposes `best_by_rmse`, because the lowest-RMSE rail can differ
  from the best gate-compatible rail;
- status dashboard now reports:
  - evidence readiness;
  - evidence progress;
  - best candidate by gate sort;
  - best candidate by RMSE.

Current z2 review:

- evidence ready: `False`;
- days: `1/2`;
- cases: `2/6`;
- rows: `144/500`;
- best wind by gate: `router`, `4` global failures, `1` performance failure;
- best wind by RMSE: `router`, RMSE `1.286 m/s`;
- best gust by gate: `high`, `4` global failures, `4` performance failures;
- best gust by RMSE: `guarded_stacker`, RMSE `1.768 m/s`, but `6`
  performance failures.

Interpretation:

- wind: router is the coherent candidate to keep watching, but the
  `wind >=15 kt` CSI regression remains a real blocker;
- gust: no candidate is clean yet; the lower-RMSE guarded stacker is not the
  most gate-compatible because it damages several threshold checks;
- promotion remains blocked by both lack of evidence and unresolved
  performance failures, so `do_not_promote` is the correct decision.

## Candidate Review v3: Failure Margins

Updated:

- `scripts/ml_dataset/review_shadow_promotion_candidates.py`
- `scripts/ml_dataset/shadow_validation_status.py`

Change:

- performance failures now include explicit margins:
  - RMSE max miss in `m/s`;
  - CSI min miss;
  - calm-regime RMSE miss in `m/s`;
- candidate review exposes a `performance_gap_summary`;
- dashboard surfaces this gap summary for the best gate-sorted candidates.

Current quantified gaps:

- best wind candidate: `router`;
- wind performance failures: `1`;
- wind CSI miss total: `0.283`;
- wind RMSE miss total: `0.000 m/s`;
- wind calm RMSE miss total: `0.000 m/s`;
- best gust candidate by gate: `high`;
- gust performance failures: `4`;
- gust CSI miss total: `0.272`;
- gust RMSE miss total: `0.090 m/s`;
- gust calm RMSE miss total: `0.155 m/s`.

Most actionable interpretation:

- wind `router` is not failing because of overall RMSE or calm regime; it is
  failing because it misses the `wind >=15 kt` threshold behavior versus raw;
- gust `high` is still too weak globally and in calm regime, and also loses
  meaningful CSI versus champion at `gust >=20 kt`;
- gust `guarded_stacker` remains interesting for RMSE, but its CSI miss total is
  much larger than `high`, so it should not be promoted without a dedicated
  threshold guard or calibration.

## Threshold Guard v1 Experiment

Added:

- `scripts/ml_dataset/apply_threshold_guard_v1.py`

Integrated:

- `scripts/ml_dataset/run_collector_hindcast_suite.py`
- `scripts/ml_dataset/score_live_hindcast_predictions.py`
- `scripts/ml_dataset/summarize_shadow_suites.py`
- `scripts/ml_dataset/rescore_shadow_suite_metrics.py`
- `scripts/ml_dataset/review_shadow_promotion_candidates.py`

Purpose:

- create a non-prod shadow rail focused on the current real blockers:
  - wind `>=15 kt` threshold behavior;
  - gust `>=20/25 kt` false positives/misses;
- keep it fully side-by-side with raw/champion/router/stacker/guarded rails.

Rules:

- wind starts from `shadow_router_v1_wind_mean_kt`;
- if raw wind is `>=15 kt` and router falls below `15 kt`, use raw wind for that
  row;
- gust starts from `shadow_guarded_stacker_v1_gust_kt`;
- for gust `>=20/25 kt`, raw/champion are used to reduce obvious guarded false
  positives and repair obvious misses.

Validation on existing small unseen suite:

- applied to the two existing `2026-07-02` morning cases;
- rescored the suite;
- regenerated `shadow_rollup_latest`;
- dashboard now sees `threshold_guard` as a candidate.

Current result on the small unseen rollup:

- wind threshold guard RMSE: `1.179 m/s`;
- previous wind router RMSE: `1.286 m/s`;
- wind raw RMSE: `1.558 m/s`;
- wind `>=15 kt` CSI:
  - raw: `0.667`;
  - router: `0.364`;
  - threshold guard: `0.636`;
- wind CSI miss total versus gates: `0.010`, down from router's `0.283`;
- gust threshold guard RMSE: `1.721 m/s`, best current gust RMSE;
- gust `>=25 kt` CSI improves to `1.000`, matching raw;
- gust `>=20 kt` CSI remains weak at `0.375`.

Interpretation:

- this is the first candidate that improves wind RMSE and nearly repairs the
  `wind >=15 kt` blocker at the same time;
- it is still not promotable because evidence is tiny: `1` day, `2` cases,
  `144` rows;
- the remaining wind threshold gap is now small enough to study on the full-day
  and multi-day campaign;
- for gust, threshold guard improves RMSE and `>=25 kt`, but does not solve
  `>=20 kt`; gust needs a separate calibration strategy.

## Dynamic Promotion Decision

Added:

- `scripts/ml_dataset/decide_shadow_promotion.py`

Integrated:

- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`
- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh`
- `scripts/ml_dataset/shadow_validation_status.py`

Purpose:

- create one final promotion decision artifact from the multi-candidate review;
- avoid treating fixed legacy gates like `wind_router` or
  `gust_guarded_stacker` as the only promotion path;
- promote only if at least one target candidate passes the full gate suite;
- otherwise classify the blocker:
  - `evidence_only`;
  - `performance`;
  - `evidence_and_performance`;
  - `no_candidate`.

Validated on z2:

- regenerated `shadow_rollup_latest`;
- produced
  `/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/promotion_decision.json`;
- status dashboard now reports the final decision.

Current decision:

- global decision: `do_not_promote`;
- evidence ready: `False`;
- wind:
  - decision: `do_not_promote`;
  - blocker: `evidence_and_performance`;
  - best gate candidate: `threshold_guard`;
  - best RMSE candidate: `threshold_guard`;
- gust:
  - decision: `do_not_promote`;
  - blocker: `evidence_and_performance`;
  - best gate candidate: `high`;
  - best RMSE candidate: `threshold_guard`.

Interpretation:

- the decision path is now aligned with the actual best candidates, not the old
  fixed rails;
- `threshold_guard` can become promotable for wind if it confirms on enough
  days and clears the small remaining threshold gap;
- gust still has a genuine performance problem even though `threshold_guard`
  is best by RMSE.

## Threshold Guard Impact Audit

Added:

- `scripts/ml_dataset/audit_threshold_guard_impact.py`

Integrated:

- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh`
- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`
- `scripts/ml_dataset/shadow_validation_status.py`

Purpose:

- audit `threshold_guard_v1` beyond global RMSE;
- compare against raw/champion/router for wind;
- compare against raw/champion/high/guarded stacker for gust;
- report by spot, lead bucket, actual wind/gust regime, and target hour.

Validated on z2:

- rollup now writes:
  `/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/threshold_guard_impact_audit.json`;
- dashboard exposes audit row count and RMSE gains.

Current audit, still tiny sample:

- rows: `144`;
- wind threshold guard RMSE gain:
  - vs raw: `+0.738 kt`;
  - vs champion: `+0.393 kt`;
  - vs router: `+0.210 kt`;
- gust threshold guard RMSE gain:
  - vs raw: `+0.127 kt`;
  - vs champion: `+0.593 kt`;
  - vs high: `+0.264 kt`;
  - vs guarded stacker: `+0.092 kt`.

Local risk notes:

- wind improves versus raw on all listed spots, though `la_parata` gain is tiny
  on only `7` rows;
- gust improves globally but is not uniformly safer:
  - worse than `high` on `cap_corse`, `lfks`, and `lfvf` in this tiny sample;
  - better than `high` on `la_parata`, `lfkf`, `lfkj`, and `lfvh`;
- this supports the current decision:
  - wind `threshold_guard` is a serious candidate to validate;
  - gust `threshold_guard` should remain shadow-only until we understand local
    regressions and `gust >=20 kt` CSI.

## Local Risk Gate

Updated:

- `scripts/ml_dataset/audit_threshold_guard_impact.py`
- `scripts/ml_dataset/decide_shadow_promotion.py`
- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh`
- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`
- `scripts/ml_dataset/shadow_validation_status.py`

Change:

- `threshold_guard` audit now emits local risk flags when, for a group with at
  least `20` rows, candidate RMSE is worse than the best baseline by more than
  `0.10 kt`;
- groups checked:
  - spot;
  - lead bucket;
  - actual wind regime;
  - actual gust regime;
  - target hour UTC;
- promotion decision now receives the threshold guard audit and can reject a
  promotable `threshold_guard` if local risk flags exist.

Current local risk flags:

- total flags: `7`;
- wind flags: `0`;
- gust flags: `7`.

Current flagged gust groups:

- `spot_id=cap_corse`, regression `+2.432 kt` vs champion;
- `spot_id=lfks`, regression `+0.356 kt` vs high;
- `spot_id=lfvf`, regression `+1.141 kt` vs raw;
- `lead_bucket=0-1h`, regression `+0.133 kt` vs champion;
- `actual_gust_regime_kt=15-20kt`, regression `+1.138 kt` vs high;
- `target_hour_utc=3`, regression `+0.911 kt` vs champion;
- `target_hour_utc=6`, regression `+0.604 kt` vs raw.

Interpretation:

- wind `threshold_guard` has no local RMSE risk flag on the current sample;
- gust `threshold_guard` has real local risk despite good global RMSE;
- final decision remains `do_not_promote`;
- future promotion of `threshold_guard` is now protected against hidden
  local regressions, not just global metric regressions.

## Next Specialist Planner

Added:

- `scripts/ml_dataset/plan_next_nowcasting_specialists.py`

Hooked into:

- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`
- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh`

Purpose:

- convert promotion evidence, performance misses, and local risk flags into the
  next concrete modelling work order;
- avoid repeating broad model experiments when the evidence points to a narrow
  failure cell;
- produce both JSON and Markdown artifacts for every daily postprocess and
  multi-day rollup.

Current z2 artifact:

`/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/next_nowcasting_specialist_plan.md`

Current recommendation:

`wait_for_fresh_shadow_evidence_and_keep_preparing_specialists`

Current actions:

- wind:
  - collect more fresh shadow evidence;
  - prepare a threshold probability/event head for the small `wind >=15kt` CSI
    miss.
- gust:
  - collect more fresh shadow evidence;
  - build a local-risk fallback gate before any promotion;
  - prepare a threshold probability/event head because `gust >=20kt` CSI is the
    largest performance miss.

Validation:

- local `py_compile`: passed;
- local Bash syntax check for postprocess/rollup: passed;
- z2 `py_compile`: passed;
- z2 Bash syntax check: passed;
- first z2 plan generation: passed.

## Gust Local Fallback Guard v1

Added:

- `scripts/ml_dataset/apply_local_fallback_guard_v1.py`

Integrated into:

- `scripts/ml_dataset/run_collector_hindcast_suite.py`
- `scripts/ml_dataset/score_live_hindcast_predictions.py`
- `scripts/ml_dataset/summarize_shadow_suites.py`
- `scripts/ml_dataset/review_shadow_promotion_candidates.py`

Purpose:

- start from `threshold_guard_v1_gust`;
- read the latest local-risk audit;
- use only inference-safe risky groups:
  - `spot_id`;
  - `target_hour_utc`;
  - `lead_bucket`;
- fallback to the locally safer baseline from the audit:
  - raw;
  - champion;
  - high;
  - guarded stacker;
- preserve `>=25kt` alerts so a local fallback cannot erase a high-gust event
  that `threshold_guard_v1` had already detected.

Current tiny-sample result on the two short unseen morning cases, `144` rows:

| Rail | Gust RMSE m/s | MAE m/s | Bias m/s | `>=20kt` CSI | `>=25kt` CSI |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 1.786 | 1.476 | -0.099 | 0.375 | 1.000 |
| champion | 2.026 | 1.514 | -1.088 | 0.667 | 0.000 |
| high | 1.857 | 1.506 | -0.455 | 0.375 | 1.000 |
| guarded stacker | 1.768 | 1.331 | -0.115 |  |  |
| threshold guard | 1.721 | 1.308 | -0.139 | 0.375 | 1.000 |
| local fallback guard | 1.437 | 1.155 | -0.335 | 0.667 | 1.000 |

Interpretation:

- this is the strongest gust shadow result so far on the short unseen rollup;
- it fixes the `gust >=20kt` false-positive issue versus threshold guard;
- the first version hurt `gust >=25kt` recall, so `preserve_threshold_kt=25`
  was added and retested;
- after preserve-25, `gust >=25kt` CSI remains `1.0`;
- still no promotion: evidence is only `1` day, `2` cases, `144` rows.

Next validation:

- let the `2026-07-02` full-day suite and the `2026-07-03..05` campaign run
  with this new shadow rail;
- promote only if the same pattern holds under the existing multi-day evidence,
  threshold, calm-regime, and local-risk gates.

## Gust Threshold Event Head Audit

Added:

- `scripts/ml_dataset/audit_gust_threshold_event_heads.py`

Hooked into:

- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`
- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh`
- `scripts/ml_dataset/plan_next_nowcasting_specialists.py`

Purpose:

- evaluate threshold events separately from continuous RMSE;
- compare deterministic rails such as raw/champion/high/threshold guard/local
  fallback guard;
- compare existing probability heads:
  - `prob_gust_ge_20kt`;
  - `prob_gust_ge_20kt_heuristic`;
  - `prob_gust_ge_20kt_model`;
  - `prob_gust_ge_25kt`;
  - `prob_gust_ge_25kt_heuristic`;
  - `prob_gust_ge_25kt_model`;
- report CSI, precision, recall, false positives, false negatives, Brier score,
  and best probability cutoff.

Current z2 artifact:

`/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/gust_threshold_event_head_audit.md`

Current tiny-sample result:

| Threshold | Best deterministic | CSI | Best probability | CSI | Cutoff |
| --- | --- | ---: | --- | ---: | ---: |
| `gust >=20kt` | champion | 0.667 | `prob_gust_ge_20kt_heuristic` | 1.000 | 0.800 |
| `gust >=25kt` | raw | 1.000 | `prob_gust_ge_25kt_heuristic` | 1.000 | 0.400 |

Interpretation:

- this is not a deployable calibration yet because the sample is only `144`
  rows;
- it confirms that threshold-event calibration is worth validating separately
  from RMSE;
- future full-day/multi-day rollups will now show whether probability heads are
  stable enough to become a dedicated alert layer.

## Wind Threshold Event Head Audit

Added:

- `scripts/ml_dataset/audit_wind_threshold_event_heads.py`

Hooked into:

- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`
- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh`
- `scripts/ml_dataset/plan_next_nowcasting_specialists.py`

Purpose:

- evaluate windsurf wind-mean thresholds separately from global RMSE;
- compare deterministic rails:
  - raw;
  - champion;
  - strong gated;
  - router;
  - stacker;
  - guarded stacker;
  - threshold guard.

Current z2 artifact:

`/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/wind_threshold_event_head_audit.md`

Current tiny-sample result:

| Threshold | Best deterministic | CSI | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: |
| `wind >=12kt` | strong gated | 0.433 | 13 | 3 | 14 |
| `wind >=15kt` | raw | 0.667 | 6 | 0 | 3 |
| `wind >=20kt` | raw | 0.000 | 0 | 0 | 4 |
| `wind >=25kt` | no events / no usable best |  |  |  |  |

Interpretation:

- on the short unseen rollup, the wind problem is not only calibration;
- `wind >=20kt` has `4` actual events and every current deterministic rail
  misses them;
- this points to a future high-wind recall head or event-specific fallback,
  but not to immediate promotion;
- the full-day and multi-day shadow campaigns are required before training or
  selecting a new wind event policy.

## Wind High Event Guard v1

Added to:

- `scripts/ml_dataset/apply_local_fallback_guard_v1.py`
- `scripts/ml_dataset/score_live_hindcast_predictions.py`
- `scripts/ml_dataset/summarize_shadow_suites.py`
- `scripts/ml_dataset/review_shadow_promotion_candidates.py`
- `scripts/ml_dataset/audit_wind_threshold_event_heads.py`

Rule:

```text
start from threshold_guard_v1_wind
if 17kt <= wind < 20kt
and local_fallback_guard_v1_gust >= 28kt
then wind = 20kt
```

Rationale:

- the missed `wind >=20kt` events were all near-threshold wind predictions with
  very strong confirmed gust signal;
- this is inference-safe because it uses only predicted wind and predicted gust
  rails, not observed target labels;
- it is deliberately narrow and should remain shadow-only until validated on
  fresh days.

Current tiny-sample result on `144` rows:

| Rail | Wind RMSE m/s | MAE m/s | Bias m/s | `>=15kt` CSI | `>=20kt` CSI |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 1.558 | 1.273 | -1.136 | 0.667 | 0.000 |
| champion | 1.381 | 1.000 | -0.712 |  |  |
| strong gated | 1.313 | 0.935 | -0.580 |  |  |
| threshold guard | 1.179 | 0.870 | -0.503 | 0.636 | 0.000 |
| high event guard | 1.154 | 0.849 | -0.482 | 0.636 | 1.000 |

Interpretation:

- `wind_high_event_guard` becomes the best wind RMSE candidate on the short
  unseen rollup;
- it repairs all four `wind >=20kt` events in this sample without false
  positives;
- the sample is still too small and too morning-biased for promotion;
- this rail is now ready to be tested by the full-day and multi-day shadow
  campaign.

## Shadow Candidate Impact Audit v1

Added:

- `scripts/ml_dataset/audit_shadow_candidate_impact.py`

Hooked into:

- `scripts/ml_dataset/run_shadow_suite_postprocess.sh`
- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh`
- `scripts/ml_dataset/decide_shadow_promotion.py`
- `scripts/ml_dataset/plan_next_nowcasting_specialists.py`

Purpose:

- audit local regressions for each actual promotion candidate, not only for
  `threshold_guard_v1`;
- make the final promotion decision candidate-specific;
- make the next-specialist plan point to the real weak cells of the current
  best candidate.

Current z2 artifact:

`/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/shadow_candidate_impact_audit.md`

Current tiny-sample result on `144` joined rows:

| Target | Candidate | Local Risk Flags |
| --- | --- | ---: |
| wind | `high_event_guard` | 0 |
| wind | `threshold_guard` | 0 |
| gust | `local_fallback_guard` | 2 |
| gust | `threshold_guard` | 7 |

Current best candidates:

| Target | Best Candidate | RMSE m/s | MAE m/s | Gain vs Raw m/s | Gain vs Champion m/s |
| --- | --- | ---: | ---: | ---: | ---: |
| wind | `high_event_guard` | 1.154 | 0.849 | 0.404 | 0.227 |
| gust | `local_fallback_guard` | 1.437 | 1.155 | 0.349 | 0.588 |

Current decision:

```text
do_not_promote
```

Why:

- evidence gate is not ready: `1/2` days, `2/6` cases, `2/6` shadow cases,
  `144/500` joined rows;
- gust `local_fallback_guard` still has local risk on:
  - `spot_id=lfvf`, worse than raw by `0.842 kt`;
  - `spot_id=lfkj`, worse than guarded stacker by `0.833 kt`;
- wind `high_event_guard` has no local risk flag on the current sample, but it
  still slightly misses the `wind >=15kt` CSI non-regression gate versus raw.

Interpretation:

- the wind path is now clearer: keep collecting evidence, then focus the next
  specialist on threshold/event calibration rather than local fallback;
- the gust path is promising globally, but not promotion-safe until the
  `lfvf/lfkj` risk persists or disappears across more fresh days;
- do not convert these two spot risks into prod rules yet, because the current
  sample is too small and could overfit.
