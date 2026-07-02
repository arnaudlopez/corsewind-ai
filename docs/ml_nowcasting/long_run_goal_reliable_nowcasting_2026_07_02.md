# Long-Run Goal: Reliable CorseWind Nowcasting

Date: 2026-07-02

## Mission

Build a genuinely reliable CorseWind nowcasting engine for windsurf decisions:

```text
AROME/AROME-PI prior
+ recent observations
+ local/context station dynamics
+ physical regime signals
+ conservative ML correction
-> calibrated wind mean, gust, and threshold probabilities
```

The goal is not to replace weather models. The goal is to correct them locally,
quickly, and safely on our observed spots.

## Current Truth

The current champions are already strong:

- wind mean champion: `prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1`
  - verified 2026 RMSE: `1.268019 m/s`
  - MAE: `0.930465 m/s`
- gust champion: `new_scale070_gust_recipe`
  - verified 2026 RMSE: `1.484221 m/s`
  - MAE: `1.073906 m/s`

The RMSE `0.9 m/s` target is still useful as a north star, but the scientific
audit says it requires a very large MSE reduction:

- wind mean: about `49.6%` MSE reduction versus current champion;
- gust: about `63.2%` MSE reduction versus current champion.

So "really good" should be defined as a production-grade improvement, not only a
single global RMSE number.

## What We Have Learned

1. The SAPHIR/ORCA framing is right.

   The best architecture is a correction model:

   ```text
   raw NWP + recent obs + recent NWP error + neighbor/context signals
   ```

   Pure ML or standalone foundation models are not the right primary forecast.

2. Generic model churn does not work.

   Chronos, TimesFM, Moirai, flexible routers, and large tabular variants did not
   become strong standalone replacements. Their useful role is as weak experts or
   diagnostic signals.

3. Conservative residual correction works best so far.

   The most reliable improvements have come from constrained residual models and
   capped blends around the champion, not from unrestricted meta-models.

4. Same-key evaluation is mandatory.

   Every comparison must use the exact same:

   ```text
   spot_id + issue_time_utc + target_time_utc + lead_time_minutes
   ```

   Small-overlap results are not promotion evidence.

5. The remaining error is concentrated.

   Hard rows dominate the score:

   - strong wind and strong gust bins;
   - La Tonnara and Santa Manza;
   - +45/+60 minute horizons;
   - thermal start/collapse;
   - high-volatility 6-15 minute labels.

6. Threshold behavior matters as much as RMSE.

   A model that improves RMSE but misses `>=20kt` or `>=25kt` events is not good
   enough for windsurf.

## Definition Of Success

A candidate can be called "really good" only if it passes all of these gates.

### Evidence Gate

- at least `5` fresh issue days;
- at least `20` scored shadow cases;
- at least `3000` joined prediction/observation rows;
- at least one clear thermal day and one strong-wind day;
- same-key comparison versus raw and current champions.

### Global Skill Gate

Minimum promotion target:

- wind mean global RMSE improves by at least `1%` versus champion;
- gust global RMSE improves by at least `1%` versus champion;
- MAE does not degrade;
- absolute bias stays below `0.10 m/s`.

Stretch target:

- wind mean RMSE below `1.20 m/s` on 2026-style evaluation;
- gust RMSE below `1.40 m/s`;
- then re-open the RMSE `0.9` campaign with a realistic error-floor audit.

### Windsurf Gate

For wind mean thresholds `12/15/20/25 kt` and gust thresholds `15/20/25/30 kt`:

- CSI must improve or stay within `0.02` of champion/raw safety baselines;
- false negatives at `>=20kt` and `>=25kt` must not increase materially;
- event timing for thermal start and collapse must be measured, not guessed.

### Regime Gate

The candidate must not win globally by damaging important regimes:

- calm wind `<12kt`;
- wind `>=20kt`;
- gust `>=25kt`;
- La Tonnara;
- Santa Manza;
- +45/+60 minute horizons;
- thermal hours, roughly `11:00-17:00` local.

### Operational Gate

- missing expert prediction falls back to champion;
- max correction delta is capped;
- model output includes champion/raw/candidate side by side;
- every live prediction can be audited later against observations;
- no production promotion from one attractive metric.

## Long-Run Workstreams

### Workstream 1: Fresh Collector Evidence

Use `/srv/data/corsewind/ml_dataset` as the live truth source.

Immediate target:

- finish the current `2026-07-02` full-day shadow suite;
- finish the `2026-07-03` to `2026-07-05` campaign;
- aggregate all fresh shadow days into one same-key benchmark;
- decide only after enough rows and regimes are present.

Output:

```text
fresh_collector_shadow_rollup_v1
fresh_collector_promotion_decision_v1
```

### Workstream 2: Data Quality And Label Physics

Before trying more models, quantify whether the target itself is learnable:

- observation age and source freshness;
- 6-minute versus 15-minute target volatility;
- duplicate label consistency;
- station/spot representativeness;
- strong gust spikes versus sustained wind;
- missing context-station histories.

If the label is too volatile for point RMSE, add probabilistic heads instead of
overfitting the mean.

### Workstream 3: Hard-Regime Specialists

Train specialists only where the error audit says they can matter:

- strong wind bins: `12/15/20/25+ kt`;
- gust bins: `15/20/25/30+ kt`;
- thermal onset;
- thermal collapse;
- La Tonnara and Santa Manza;
- +45/+60 minute leads.

The specialist output should not replace the champion directly. It should be a
guarded residual layer:

```text
candidate = champion + scale * clip(specialist_residual, min_delta, max_delta)
```

### Workstream 4: Physical Signals That Are Still Worth Chasing

Priority signals:

- recent observed model error:
  `observation_now - NWP_prediction_for_now`;
- coastal/inland thermal delta;
- relief/coast pressure and temperature gradients;
- land-sea temperature contrast:
  Copernicus SST versus land surface temperature;
- true fetch and coastline exposure;
- DEM slope/aspect/upwind relief;
- boundary-layer height and low-level vertical wind profile;
- cloud type and instability for thermal inhibition/convection.

Rule:

```text
add a signal only if it improves a hard regime or explains a failure mode
```

### Workstream 5: Foundation Models As Weak Experts

Foundation models stay in the project, but with a narrow role:

- never standalone production forecast;
- always same-key against champion;
- capped blend only;
- promote only if stable across fresh days and hard regimes.

Current lesson:

```text
small capped correction useful
flexible router dangerous until proven on larger fresh overlap
```

### Workstream 6: Probabilistic Windsurf Output

The final product should not expose only one deterministic number.

Needed outputs:

- wind mean P10/P50/P90;
- gust P10/P50/P90;
- `P(wind >= 12/15/20/25 kt)`;
- `P(gust >= 15/20/25/30 kt)`;
- confidence/session score;
- timing confidence for start/drop of wind window.

This aligns better with the decision problem than chasing RMSE alone.

## Immediate Execution Plan

1. Let the current fresh shadow watchers finish.

   Running on z2:

   - `2026-07-02` full-day watcher;
   - postprocess watcher;
   - `2026-07-03` to `2026-07-05` campaign watcher.

2. When the rollup is complete, produce one decision report:

   - global RMSE/MAE/bias;
   - by spot;
   - by lead;
   - by threshold;
   - by thermal hours;
   - by hard wind/gust bins;
   - local risk flags.

3. If no candidate passes:

   - do not declare failure;
   - identify the exact error cells that remain;
   - train only specialists aimed at those cells.

4. Build `fresh_collector_training_pack_v1`:

   - same-key rows from the collector;
   - raw NWP;
   - champion outputs;
   - recent observations;
   - model error now;
   - station-context summaries;
   - regime labels;
   - quality flags.

5. Train the next serious candidate:

   - constrained residual correction around the champion;
   - event-weighted objective;
   - separate wind and gust heads;
   - threshold probability heads;
   - hard-regime validation before global promotion.

## Automation Added

Added:

```text
scripts/ml_dataset/plan_next_nowcasting_specialists.py
```

Purpose:

- read the multi-candidate promotion review;
- read the final promotion decision;
- read the threshold-guard local-risk audit when available;
- turn the evidence into a concrete next work order.

The script is now called automatically by:

```text
scripts/ml_dataset/run_shadow_suite_postprocess.sh
scripts/ml_dataset/run_shadow_multi_day_rollup.sh
```

Generated artifacts:

```text
next_nowcasting_specialist_plan.json
next_nowcasting_specialist_plan.md
```

Current z2 smoke result on the short unseen rollup:

```text
recommendation = wait_for_fresh_shadow_evidence_and_keep_preparing_specialists
```

Current target actions:

- wind:
  - collect more fresh shadow evidence;
  - train/prepare a threshold probability or event head, because the best
    candidate still has a small `wind >=15kt` CSI miss versus raw.
- gust:
  - collect more fresh shadow evidence;
  - build a local fallback gate before promotion;
  - train/prepare a threshold probability or event head, because `gust >=20kt`
    CSI is the dominant performance miss.

Current gust local-risk cells from the tiny sample:

- `spot_id=cap_corse`, threshold guard worse than champion by `2.432 kt`;
- `actual_gust_regime_kt=15-20kt`, worse than high by `1.138 kt`;
- `spot_id=lfvf`, worse than raw by `1.141 kt`;
- `target_hour_utc=6`, worse than raw by `0.604 kt`;
- `target_hour_utc=3`, worse than champion by `0.911 kt`;
- `spot_id=lfks`, worse than high by `0.356 kt`;
- `lead_bucket=0-1h`, worse than champion by `0.133 kt`.

Interpretation:

- wind `threshold_guard` remains a plausible candidate to validate;
- gust `threshold_guard` is not promotion-safe yet despite good global RMSE;
- the next gust model should be a guarded/local fallback or probability-head
  experiment, not a blind global replacement.

Update:

- implemented `local_fallback_guard_v1` for gusts;
- it falls back only on inference-safe risky groups from the audit;
- it preserves `>=25kt` events from `threshold_guard_v1`;
- on the current tiny unseen rollup, gust RMSE improves from `1.721` to
  `1.437 m/s`, `gust >=20kt` CSI improves from `0.375` to `0.667`, and
  `gust >=25kt` CSI stays at `1.000`;
- this is promising but explicitly not promotion evidence because the sample is
  still only `144` rows.

Latest update:

- added `shadow_candidate_impact_audit_v1` so local-risk checks are now tied to
  the actual best candidate:
  - wind `high_event_guard`;
  - gust `local_fallback_guard`;
- re-ran the latest z2 rollup successfully;
- current evidence remains insufficient: `1/2` days, `2/6` cases, `2/6`
  shadow cases, `144/500` joined rows;
- current best wind candidate is `high_event_guard`:
  - RMSE `1.154 m/s`;
  - MAE `0.849 m/s`;
  - gain vs raw `0.404 m/s`;
  - gain vs champion `0.227 m/s`;
  - local risk flags `0`;
- current best gust candidate is `local_fallback_guard`:
  - RMSE `1.437 m/s`;
  - MAE `1.155 m/s`;
  - gain vs raw `0.349 m/s`;
  - gain vs champion `0.588 m/s`;
  - local risk flags `2`, concentrated on `spot_id=lfvf` and `spot_id=lfkj`.

Decision:

- still `do_not_promote`;
- continue full-day/multi-day shadow evidence collection;
- avoid turning the `lfvf/lfkj` gust risks into production rules until they
  persist across more days.

New candidate:

- added `gust_recall_floor_guard_v1`;
- it starts from the current guarded gust candidate and restores low-threshold
  recall when raw NWP is above `12/15 kt` and the corrected prediction is only
  slightly below the threshold;
- current short-rollup result:
  - RMSE improves from `1.437 m/s` to `1.381 m/s`;
  - MAE improves from `1.155 m/s` to `1.100 m/s`;
  - `gust >=15kt` CSI improves from `0.405` to `0.511`, matching raw;
  - `gust >=20kt` CSI remains `0.667`;
  - `gust >=25kt` CSI remains `1.000`;
  - performance failures drop from `2` to `1`;
  - local risk flags remain `2`, on `lfkj` and `lfvf`.

Interpretation:

- this is the strongest recent gust improvement because it improves both RMSE
  and useful threshold behavior on the same rows;
- it must remain shadow-only until fresh multi-day evidence confirms the gain;
- next promotion decision should compare `gust_recall_floor_guard_v1` against
  `local_fallback_guard_v1`, raw, and champion over at least the configured
  evidence gate.

Wind candidate update:

- added `wind_gust_floor_guard_v1`;
- it starts from `wind_high_event_guard_v1`;
- it uses the improved gust rail as a conservative evidence signal:

```text
if 8 kt <= wind < 12 kt
and gust_recall_floor_guard_v1 >= 14 kt
then wind = 12 kt
```

Current short-rollup result on the same `144` rows:

- wind RMSE improves from `1.154 m/s` to `1.079 m/s`;
- wind MAE improves from `0.849 m/s` to `0.786 m/s`;
- bias improves from `-0.482 m/s` to `-0.381 m/s`;
- `wind >=12kt` CSI improves from `0.433` to `0.639`;
- `wind >=15kt` CSI stays `0.636`, still slightly below the raw non-regression
  gate;
- `wind >=20kt` CSI stays `1.000`;
- local risk flags are `0`.

Interpretation:

- this is currently the strongest wind candidate on the tiny unseen rollup;
- it is inference-safe and deliberately limited to the first useful windsurf
  threshold;
- it should remain shadow-only until full-day and multi-day evidence proves the
  gain is stable.

Status tooling update:

- updated `scripts/ml_dataset/shadow_validation_status.py`;
- it now reports the multi-candidate promotion decision directly instead of
  highlighting only the older router/guarded-stacker gates;
- it includes candidate-impact local risk flags by target and candidate;
- it now computes observation coverage wait time from the oldest tracked spot
  to `target_end_utc`;
- it now emits a `recommended_next_action` field so the watcher state is
  immediately actionable;
- added `scripts/ml_dataset/package_shadow_promotion.py`;
- `scripts/ml_dataset/run_shadow_multi_day_rollup.sh` now writes
  `promotion_package.json` and `promotion_package.md` after every rollup;
- the package summarizes evidence readiness, current decision, best wind/gust
  candidates, local risks, and the next operational action;
- the package now adds a per-target readiness state, so we can distinguish:
  - `waiting_for_evidence_only`;
  - `waiting_for_evidence_with_local_risk`;
  - `needs_performance_specialist`;
  - `ready_to_promote`;
- generated z2 status artifact:
  `/srv/data/corsewind/ml_dataset/live_inference/shadow_status_latest.md`.
- the status now reports per-spot observation lag versus the freshest spot, with
  relative states:
  - `fresh`;
  - `slow_or_hourly`;
  - `stale_candidate`;
  - `missing`.

Latest status:

- health: `waiting_for_observations`, ok `True`;
- recommended next action: `wait_for_observations`;
- wait reason: oldest tracked spot is still `480.0` minutes before
  `target_end_utc`;
- `2026-07-02` full-day watcher running;
- postprocess watcher running;
- `shadow_campaign_20260703_3d_v1` running and waiting for `2026-07-03`;
- current slower spots: `la_parata` and `lfvh`, both classified
  `slow_or_hourly`, not `stale_candidate`;
- promotion package:
  `/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/promotion_package.md`;
- current wind readiness: `needs_threshold_margin_review`, because
  `gust_floor_guard` still misses one windsurf threshold gate but the current
  misses are close to the threshold;
- current gust readiness: `waiting_for_evidence_with_local_risk`, because
  `recall_floor_guard` has clean performance checks but still shows local risk
  on `lfkj` and `lfvf`;
- latest rollup remains `do_not_promote`.

Threshold-margin update:

- enriched `scripts/ml_dataset/audit_wind_threshold_event_heads.py`;
- the wind threshold audit now reports false-positive and false-negative
  margins around each business threshold;
- added tolerance-aware CSI at `0.5/1/2 kt` around the observed threshold;
- added the same tolerance-aware structure to
  `scripts/ml_dataset/audit_gust_threshold_event_heads.py`;
- added `scripts/ml_dataset/review_tolerant_threshold_gates.py`;
- the tolerant gate review is generated by the multi-day rollup but does not
  replace the strict promotion gate;
- on the current `144` row rollup, the `wind >=15kt` blocker is not a missed
  raw event:
  - raw: `TP=6`, `FP=0`, `FN=3`, CSI `0.667`;
  - `gust_floor_guard`: `TP=7`, `FP=2`, `FN=2`, CSI `0.636`;
  - all `gust_floor_guard` errors at this threshold are within `2 kt` of the
    threshold;
  - with `2 kt` tolerance, `gust_floor_guard` reaches CSI `1.000` at
    `wind >=15kt`;
- package readiness for wind is now
  `strict_threshold_failures_resolved_by_tolerance`, not a blind call for a
  heavier specialist.

Interpretation:

- the wind candidate may be better for windsurf recall than the raw CSI gate
  suggests;
- before training a new model only to fix this tiny CSI miss, evaluate whether
  near-threshold uncertainty/tolerance should be part of the gate;
- a likely next gate improvement is to keep strict CSI for strong events, but add
  tolerant CSI for near-threshold windsurf decisions;
- do not lower the production gate yet: keep the strict gate as the promotion
  authority until multi-day evidence confirms that tolerance resolves the same
  kind of near-threshold miss repeatedly;
- keep this as a hypothesis until the full-day and multi-day rollups confirm it.

Coverage-cadence update:

- added `coverage_cadence` to `scripts/ml_dataset/shadow_validation_status.py`;
- deployed the status script to z2 and regenerated
  `/srv/data/corsewind/ml_dataset/live_inference/shadow_status_latest.md`;
- the diagnostic uses the last `24` watcher coverage entries, not only the last
  payload;
- current inferred cadence:
  - `cap_corse`, `lfkf`, `lfkj`, `lfks`, `lfvf`, `lfvh`: `sub_hourly_like`;
  - `la_parata`: `hourly_like`, median observed update cadence `60` minutes;
- interpretation: `la_parata` is delayed relative to the freshest spots but
  currently behaves like an hourly source, not a broken/stale source;
- next promotion evidence should keep per-spot cadence in the status report so
  we do not confuse expected hourly lag with a model/data failure.

## Things We Should Stop Doing

- Do not chase RMSE `0.9` with blind model families.
- Do not promote from small samples like `144` rows.
- Do not compare models on non-identical keys.
- Do not trust flexible routers without fresh forward validation.
- Do not let global RMSE hide strong-wind or thermal failures.
- Do not add heavy features without an audit proving they are populated and
  relevant.

## Working Objective

The active long-run objective is:

```text
Build a robust local correction engine that improves the current champions on
fresh same-key shadow validation, especially for windsurf thresholds, thermal
days, strong wind, and gusts, while preserving calm-regime reliability.
```

This is the rail. The next decision should be data-driven:

- if the current shadow candidate passes the gates, package it for promotion;
- if it fails, use the failure cells to define the next specialist;
- if the data is insufficient, keep collecting and do not overfit.
