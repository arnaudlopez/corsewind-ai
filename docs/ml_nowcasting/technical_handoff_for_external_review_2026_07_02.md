# CorseWind ML Nowcasting - Technical Handoff For External Review

Date: 2026-07-02

This document summarizes the current CorseWind ML/nowcasting work so an external
technical reviewer can understand what has been done, where the data lives, what
has worked, what has failed, and where a review could help.

## 1. Mission

CorseWind aims to improve wind forecasts on specific Corsican spots, especially
for windsurf decisions.

The chosen approach is not to replace numerical weather prediction models. The
goal is to locally correct them:

```text
AROME / AROME-PI raw forecast
+ recent observations
+ recent model error
+ neighboring/context station signals
+ physical regime signals
+ conservative ML correction
-> calibrated wind mean, gusts, threshold probabilities, and session confidence
```

The core product is a short-term nowcasting engine for the next minutes/hours:

- wind mean;
- gusts;
- probabilities of windsurf thresholds;
- timing of thermal onset/collapse;
- strong wind and gust risk.

The system should update quickly when fresh observations arrive.

## 2. Scientific Reference And Design Philosophy

The main scientific reference is SAPHIR / Baggio et al. 2025
(`arXiv:2503.18797v2`) plus related work around local NWP correction and
neighboring station signals.

The main lessons we kept:

- do not build a pure ML forecaster from scratch;
- keep AROME/AROME-PI as the physical prior;
- learn local residual corrections;
- use recent observations and recent NWP error explicitly;
- use neighboring/upwind/context stations;
- validate per horizon, not only global RMSE;
- output probabilistic/threshold-oriented information, not only one point value;
- fine-tune/calibrate per local station/spot because Corsican coast/relief
  effects are very local.

The current architecture follows this framing:

```text
raw NWP + obs history + model error history + context stations + physical/static features
-> residual correction / guarded blend
-> corrected wind and gust forecasts
```

## 3. Data Sources Investigated Or Integrated

### Beacon Live / Internal Project

Beacon Live contains the spot list and GPS coordinates. It also has live
observation monitoring logic that has been useful to understand sources such as
Winds-Up.

The key idea is: train and validate on points where observations exist, because
those are the only points where local correction can be supervised reliably.

### Meteo-France

Explored via provided swagger files and API keys:

- public observations;
- observation packages;
- climatology;
- radar packages;
- AROME;
- AROME-PI.

Useful parts:

- real-time observations where available;
- hourly climatology / DPClim-like historical station data;
- AROME/AROME-PI forecasts;
- AROME extra variables beyond wind;
- vertical profile / isobaric-level extraction around spots.

Less useful or limited:

- radar mainly gives precipitation/cloud/rain-related information, useful for
  disturbed-day flags but not a direct thermal wind driver;
- current access did not clearly confirm 6-minute wind history from
  Meteo-France for all needed stations;
- historical AROME runs from Meteo-France are not straightforwardly available
  over multiple years through the current access.

### Open-Meteo

Investigated as a potential historical forecast backfill source:

- Historical Forecast API can provide recent multi-year model history depending
  on model/date;
- Previous Runs API is useful to learn error by forecast horizon.

This is useful for backfill, but it is not the same as having archived AROME-PI
exactly as used operationally.

### MeteoNet

MeteoNet was accepted as a useful research/pretraining source, especially for
2016-2018 style public research data. It is useful for learning general weather
station behavior, but it is not enough alone for production local correction on
our exact Corsican spots.

### Copernicus

Copernicus was integrated/investigated mainly for sea-related physics:

- sea surface temperature (SST);
- historical/reanalysis style products;
- useful for land-sea thermal contrast.

Important clarification: Copernicus SST is not a live buoy network replacement;
it is mostly gridded satellite/reanalysis/model product data. It remains useful
because thermal wind is strongly linked to the land-sea temperature difference.

### EUMETSAT

EUMETSAT access was configured and product discovery was implemented.

Products identified as useful:

- Cloud Type `EO:EUM:DAT:0680`
  - clear sky vs low marine cloud vs high cloud vs convection;
- Land Surface Temperature `EO:EUM:DAT:1088`
  - important for real land heating and land-sea delta with Copernicus SST;
- Global Instability Indices `EO:EUM:DAT:0683`
  - useful to identify convection/instability and disturbed regimes.

These features are conceptually useful for thermal days, but must still be
validated by coverage/population audits before being trusted in models.

### Winds-Up

Winds-Up historical spot observations were explored through Beacon Live parsing
logic. The available history varies by spot and can include 2024 data and other
periods depending on the spot.

This is valuable because it can give spot-level observations closer to the
actual windsurf locations than official airport stations.

### DEM / Static Geography

Static features have been prepared or planned:

- DEM / relief context;
- slope/aspect;
- upwind relief;
- distance to coast;
- maritime fetch;
- coastline exposure;
- Venturi/corridor indicators.

The most recent addition planned/started was true maritime fetch using
coastline/land-sea polygons.

## 4. Data Warehouse And Machine Layout

The heavy data work runs on `z2`.

Main dataset root:

```text
/srv/data/corsewind/ml_dataset
```

Current z2 state at last check:

- `/srv/data` has about 515 GB free;
- shadow watchers are running;
- fresh observations are still being collected for the current full-day shadow
  validation.

The local repo is:

```text
/Users/arnaud/Documents/CorseWind.ai
```

Main docs are under:

```text
docs/ml_nowcasting/
```

Important status artifacts on z2:

```text
/srv/data/corsewind/ml_dataset/live_inference/shadow_status_latest.md
/srv/data/corsewind/ml_dataset/live_inference/shadow_status_latest.json
/srv/data/corsewind/ml_dataset/live_inference/shadow_rollups/shadow_rollup_latest/
```

## 5. Dataset Structure We Are Targeting

The key unit is a same-key forecast/observation sample:

```text
spot_id
issue_time_utc
target_time_utc
lead_time_minutes
```

All model comparisons must use identical keys. This became one of the strongest
lessons of the project: small-overlap or non-identical-key comparisons can make
weak models look artificially good.

Main targets:

- `wind_mean_ms`;
- `gust_ms`;
- residual wind error against raw NWP;
- residual gust error against raw NWP;
- threshold events:
  - wind `>=12/15/20/25 kt`;
  - gust `>=15/20/25/30 kt`.

Main feature families:

- raw AROME/AROME-PI wind and gust;
- lead time;
- recent observation value;
- recent observation deltas/trends;
- recent NWP error:

```text
observation_now - NWP_prediction_for_now
```

- neighboring/context station values;
- observation source, freshness, resolution, forward-fill flags;
- static geography;
- physical regime signals.

The useful context stations are not only nearest stations. We especially care
about:

- nearest coastal station;
- mountain/interior station;
- upwind station depending on direction;
- neighboring windsurf/spot observation stations.

## 6. Current Champions

Current verified champions are already strong.

Wind mean champion:

```text
prediction_residual_calibrator_2025h2_to_2026_extratrees_scale070_v1
```

Verified 2026 metrics:

- RMSE: `1.268019 m/s`;
- MAE: `0.930465 m/s`.

Gust champion:

```text
new_scale070_gust_recipe
```

Verified 2026 metrics:

- RMSE: `1.484221 m/s`;
- MAE: `1.073906 m/s`.

The old goal of RMSE `<0.9 m/s` is still a north star, but the current error
audit says it would require a very large MSE reduction:

- wind mean: about 49.6% MSE reduction vs current champion;
- gust: about 63.2% MSE reduction vs current champion.

So the practical production goal is now:

- improve reliably;
- do not degrade calm regimes;
- improve strong wind/threshold behavior;
- prove the improvement over multiple fresh days.

## 7. Model Families Tested

### Tabular Residual Models

This has been the strongest family so far:

- ExtraTrees;
- HistGradientBoosting;
- LightGBM-style experiments;
- residual correction around AROME/AROME-PI.

Conservative scaling/capping has worked better than unconstrained models.

### SAPHIR-Style Dictionary / Sequence Experiments

We downloaded/analyzed SAPHIR v2 material from Zenodo and compared their data
organization to ours.

Main lesson:

- our approach should mimic the dictionary/same-key/horizon structure;
- but we should adapt it to our spot-level live collection and windsurf metrics;
- output should be multi-horizon and evaluate by regime.

### Foundation Models

Chronos, TimesFM, Moirai and related sequence/foundation approaches were
investigated.

Result so far:

- promising on tiny oracle/small samples;
- not yet a reliable production replacement;
- useful as weak experts or diagnostic signals;
- should be benchmarked only on same-key samples against current champions.

### Physical Feature Variants

Several `phys_v*` variants were tested or planned:

- `phys_v1` did not improve the champion;
- DEM/fetch/static features are still considered important, but only if they
  show value in hard regimes;
- the team decided not to assume `phys_v3_dem_fetch` will help automatically;
  it must prove its gain.

### Strong Wind / Gust Specialists

We explicitly started targeting thresholds:

- wind 12/15/20/25+ kt;
- gust 15/20/25/30+ kt.

The current direction is not to replace the champion with a specialist, but to
apply guarded residual corrections only when evidence supports them.

## 8. Current Shadow Validation Work

A fresh collector shadow validation campaign is running on z2.

Current active objective:

```text
Build a robust local correction engine that improves the current champions on
fresh same-key shadow validation, especially for windsurf thresholds, thermal
days, strong wind, and gusts, while preserving calm-regime reliability.
```

Current gate before promotion:

- at least 5 fresh issue days;
- at least 20 scored shadow cases;
- at least 3000 joined prediction/observation rows;
- same-key comparison versus raw and champions;
- must include useful regimes, including thermal/strong wind.

Current rollup evidence is still too small:

- only 1 day;
- 2 cases;
- 144 rows;
- latest decision: `do_not_promote`.

## 9. Latest Shadow Candidates

### Wind Candidate

Candidate:

```text
wind_gust_floor_guard_v1
```

It starts from a wind high-event guard and uses the improved gust rail as a
conservative evidence signal.

Current short-rollup result on 144 rows:

- wind RMSE improves from `1.154` to `1.079 m/s`;
- wind MAE improves from `0.849` to `0.786 m/s`;
- bias improves from `-0.482` to `-0.381 m/s`;
- `wind >=12kt` CSI improves from `0.433` to `0.639`;
- `wind >=15kt` CSI is `0.636`, slightly below raw non-regression gate;
- `wind >=20kt` CSI stays `1.000`;
- local risk flags: `0`.

Interpretation:

- promising;
- not enough evidence;
- should remain shadow-only.

### Gust Candidate

Candidate:

```text
gust_recall_floor_guard_v1
```

Current short-rollup result:

- gust RMSE: `1.381 m/s`;
- gust MAE: `1.100 m/s`;
- bias: `-0.250 m/s`;
- clean global checks;
- local risks still visible on `lfkj` and `lfvf`.

Interpretation:

- promising but not promotable without more fresh evidence;
- local risk must be watched.

## 10. Threshold Review

A tolerant threshold review was added because some failures are very close to
the threshold and may be decision-uncertainty rather than meaningful misses.

Example from current 144-row rollup:

- strict `wind >=15kt`:
  - raw CSI: `0.667`;
  - candidate CSI: `0.636`;
- with 2 kt tolerance:
  - candidate CSI: `1.000`.

Important: tolerant CSI does not replace strict promotion gates yet. It is used
to understand whether a failure is a real dangerous miss or just near-threshold
noise.

## 11. Current Observation Coverage Diagnostics

The status script now reports:

- latest coverage;
- wait time until target end;
- per-spot lag versus freshest spot;
- inferred observation cadence from the last 24 coverage entries.

Recent inferred cadence:

- `cap_corse`, `lfkf`, `lfkj`, `lfks`, `lfvf`, `lfvh`: `sub_hourly_like`;
- `la_parata`: `hourly_like`, median cadence about 60 minutes.

This matters because a spot can appear delayed while actually behaving normally
for an hourly source. We do not want to blame the model for a data freshness
artifact.

## 12. Main Scripts Added Or Modified Recently

Important scripts:

```text
scripts/ml_dataset/shadow_validation_status.py
scripts/ml_dataset/package_shadow_promotion.py
scripts/ml_dataset/review_tolerant_threshold_gates.py
scripts/ml_dataset/apply_probability_event_guard_v1.py
scripts/ml_dataset/apply_gust_recall_floor_guard_v1.py
scripts/ml_dataset/apply_wind_gust_floor_guard_v1.py
scripts/ml_dataset/audit_wind_threshold_event_heads.py
scripts/ml_dataset/audit_gust_threshold_event_heads.py
scripts/ml_dataset/audit_shadow_candidate_impact.py
scripts/ml_dataset/review_shadow_promotion_candidates.py
scripts/ml_dataset/run_shadow_multi_day_rollup.sh
scripts/ml_dataset/run_shadow_suite_postprocess.sh
scripts/ml_dataset/score_live_hindcast_predictions.py
scripts/ml_dataset/summarize_shadow_suites.py
```

The current toolchain produces:

- status report;
- promotion package;
- threshold gate review;
- candidate impact audit;
- multi-day rollup;
- per-target readiness.

## 13. What Has Not Worked Well

The following paths did not reliably beat the champion:

- blind model-family churn;
- unrestricted routers;
- foundation models as direct replacements;
- physical features added without coverage/regime audit;
- comparing on small or non-identical sample keys;
- optimizing only global RMSE;
- promoting from attractive small samples.

The main current belief:

```text
The champion is hard to beat because the remaining error is concentrated in
specific hard regimes, not because we have not tried enough generic models.
```

Hard regimes:

- strong wind;
- strong gusts;
- La Tonnara / Santa Manza style local effects;
- +45/+60 minute horizons;
- thermal start/collapse;
- volatile 6-15 minute labels.

## 14. Where External Review Would Help Most

Useful review questions:

1. Is the dataset schema correct?

   Especially:

   ```text
   spot_id + issue_time_utc + target_time_utc + lead_time_minutes
   ```

   Are there leakage risks, alignment errors, or missing keys?

2. Are we mixing target resolutions incorrectly?

   We have 6-minute, 15-minute, and hourly observations. We need to decide
   whether to train separate heads or normalize labels differently.

3. Are context stations represented correctly?

   Should context be:

   - fixed nearest stations;
   - direction-dependent upwind stations;
   - graph/attention style station network;
   - explicit coastal/mountain pairs?

4. Should the model predict residuals, quantiles, or thresholds?

   Current residual correction is strongest, but windsurf decisions may need:

   - P10/P50/P90;
   - probability of `>=12/15/20/25kt`;
   - calibrated gust exceedance probabilities.

5. Are physical signals being added at the right layer?

   Maybe SST/LST/fetch/DEM should be used primarily for regime detection and
   gating, not directly as generic tabular columns.

6. Is the validation gate strict enough?

   The current gate requires multiple days, same-key comparison, thresholds,
   local risk, and regime checks. Review whether this is too strict, too weak,
   or missing business metrics.

7. Is RMSE the right north star?

   Current global RMSE is useful but may hide:

   - missed thermal onset;
   - missed strong wind;
   - wrong session window;
   - underpredicted gusts.

## 15. Current Recommended Next Steps

Do not train blindly while the shadow campaign is still collecting.

Next high-value steps:

1. Finish the current full-day and multi-day shadow validation.
2. Aggregate fresh same-key evidence.
3. Decide whether `wind_gust_floor_guard_v1` and `gust_recall_floor_guard_v1`
   survive beyond the tiny 144-row rollup.
4. If they fail, inspect failure cells by:
   - spot;
   - horizon;
   - wind bin;
   - gust bin;
   - thermal hours;
   - observation cadence/source.
5. Train only specialists targeted at the failing cells.
6. Keep the current champions as fallback until fresh evidence proves otherwise.

## 16. One-Sentence Summary

CorseWind has moved from "try more models" to a stricter production-scientific
loop: same-key local residual correction, fresh shadow validation, threshold and
regime gates, explicit data freshness diagnostics, and no champion promotion
until the gain is stable across enough real days.
