# Response To External ML Nowcasting Review

Date: 2026-07-02

This document turns the external review into an execution plan.

## Executive Decision

The review is accepted as directionally correct.

The strongest sentence is:

```text
We industrialized evaluation before fully industrializing the data.
```

That is the right diagnosis. The next phase must prioritize data correctness,
label semantics, and product-representative scoring before more model-family
experiments.

## P0. Operational AROME / AROME-PI Archive

### Review Claim

Training currently relies heavily on Open-Meteo-served AROME-like fields while
production corrects Meteo-France operational AROME/AROME-PI.

This creates train/serve skew.

The operational engine also refreshes latest model layers. If these are not
archived and sampled per run, the correct production baseline is lost every day.

### Our Verification

Accepted.

Current code already contains the right lightweight components:

- `scripts/ml_dataset/archive_model_layer_snapshot.py`;
- `scripts/ml_dataset/sample_model_layers_at_spots.py`;
- `scripts/run_forecast_update_engine.py --enable-ml-dataset-archive`.

But the default was disabled:

```text
ML_DATASET_ARCHIVE_ENABLED=false
```

z2 currently has only sparse operational model archives/samples around
2026-06-25 to 2026-06-28, not a continuous operational history.

### Action Taken

- changed `scripts/run_forecast_update_engine.py` so
  `ML_DATASET_ARCHIVE_ENABLED` defaults to `true`;
- changed `.env.example` to `ML_DATASET_ARCHIVE_ENABLED=true`;
- changed `docker-compose.forecast-engine.yml` so the container default is also
  `ML_DATASET_ARCHIVE_ENABLED=true`;
- documented the operational archive as mandatory-lightweight, distinct from
  heavier extra-field collection;
- added `scripts/ml_dataset/audit_operational_nwp_archive.py`.

### z2 Audit Result

Audit generated:

```text
/srv/data/corsewind/ml_dataset/source_inventories/operational_nwp_archive_audit_latest.md
```

Current result:

- problem count: `2`;
- `arome`: latest archived run is about `177.9h` old;
- `aromepi`: latest archived run is about `171.9h` old;
- both exceed the `36h` freshness gate.

Interpretation:

```text
z2 is not the active forecast-engine runtime. This audit is still useful for the
dataset mirror, but it is not the operational source of truth.
```

### home101 Runtime Result

The active forecast-engine runtime is on `home101`, not z2.

Runtime status:

- container `corsewind-forecast-engine`: healthy;
- container environment had `ML_DATASET_ARCHIVE_ENABLED=false`;
- mounted ML dataset root:
  `/data/compose/17/data/processed/ml_dataset`;
- current layers existed in the container:
  - `visualizations/wind2d/arome-corsica-latest.json`;
  - `visualizations/wind2d/aromepi-corsica-latest.json`.

Immediate backstop action:

- archived current AROME run `2026-07-02T12:00:00Z`;
- archived current AROME-PI runs from the active Wind2D runtime;
- sampled both at ML spots into the mounted ML dataset;
- fixed `sample_model_layers_at_spots.py` so same-date samples preserve existing
  rows and dedupe by `(source, run_time_utc, valid_time_utc, spot_id)`.

Corrected architecture:

- the canonical runtime for continuous dataset collection is
  `corsewind-ml-data-collector` on `home101`;
- added source `remote_wind2d_model_layers` to
  `configs/ml_data_collector_sources.json`;
- implemented the corresponding collector branch in
  `scripts/ml_dataset/run_data_collector.py`;
- deployed the collector source into the active
  `corsewind-ml-data-collector` container;
- launched a 5-minute bridge loop inside the collector, writing to z2 through
  the existing SSHFS mount:
  `/srv/data/corsewind/ml_dataset/model_runs` and
  `/srv/data/corsewind/ml_dataset/model_samples`;
- stopped the temporary host watcher once the collector bridge was confirmed;
- committed the patched active container filesystem back into local image
  `corsewind-ml-dataset-runner:latest`;
- prepared `docker-compose.ml-data-collector-home101.yml` so a future Portainer
  redeploy starts both the official collector loop and the Wind2D bridge loop.

home101 audit result after the backstop:

```text
problem_count = 0
arome latest run = 2026-07-02T12:00:00Z
aromepi latest run = 2026-07-02T15:00:00Z
```

### Next Required Action

Update/redeploy the Portainer stack on `home101` from the patched
`docker-compose.ml-data-collector-home101.yml`, or mirror the same command/env
in Portainer. Until that redeploy, the Wind2D bridge loop is already running via
`docker exec -d`, but a container recreate would need the patched stack command.

The forecast-engine archive fallback remains useful, but it is not the canonical
collector path. The z2 forecast-engine compose file was also patched in place,
but z2 is not the active runtime:

```text
/srv/data/corsewind/forecast_engine/docker-compose.forecast-engine.yml
```

At the time of the patch, the forecast-engine stack was not running on z2, so no
container restart was performed.

The audit should become a health gate:

```text
no recent AROME/AROME-PI model_samples => data collection incident
```

## P0. Double Scoring Track: Official Stations And Winds-Up Product Spots

### Review Claim

The current shadow gate scores mostly official Meteo-France stations:

```text
cap_corse, la_parata, lfkf, lfkj, lfks, lfvf, lfvh
```

But the product promise is windsurf spots such as La Tonnara, Santa Manza,
Piantarella, Balistra, Porto Polo. Those must be in the promotion gate.

### Our Verification

Accepted.

The default shadow suite currently uses official/station-like spots. Winds-Up
history exists and has been used in training experiments, but product-spot
shadow scoring is not yet a first-class promotion gate.

### Required Design

Create two scoring tracks:

```text
official_track:
  purpose: clean meteorological supervision
  sources: Meteo-France / official stations

product_track:
  purpose: product truth at windsurf spots
  sources: Winds-Up / spot observations
  metrics: robust to sensor/source quirks
```

Promotion must require:

- improvement or non-regression on official track;
- no material degradation on product track;
- separate reporting by source and spot.

### Next Required Action

### Action Taken

- added `score_track` to `configs/ml_spots.json`;
- `official` now resolves to:
  `cap_corse,la_parata,lfkf,lfkj,lfks,lfvf,lfvh`;
- `product` now resolves to:
  `balistra,figari_eole,la_tonnara,piantarella,porticcio,porto_polo,santa_manza`;
- `context` contains the other live/context sources;
- updated `scripts/ml_dataset/score_live_hindcast_predictions.py` with
  `--score-track official|product|context|all|legacy_default`;
- updated `run_aromepi_hindcast_evaluation.py` and
  `run_collector_hindcast_suite.py` to propagate the scoring track;
- `--spots` remains available and overrides `--score-track` for manual tests.

The suite can now run:

```text
--score-track official
--score-track product
--score-track all
```

## P0. Canonical Label Semantics And Noise Floor

### Review Claim

`wind_mean_ms` and `gust_ms` currently mix sources with different temporal
semantics:

- 6-minute station data;
- hourly station data;
- Winds-Up quasi-instant/irregular observations;
- gust instant versus gust max windows.

This creates label noise and may impose an irreducible RMSE floor.

### Our Verification

Accepted.

The project already suspected volatile 6-15 minute labels, but it has not yet
quantified the label noise floor by source/window.

### Required Design

Define canonical labels:

```text
canonical_wind_mean_10m_centered_ms
canonical_gust_max_window_ms
canonical_direction_vector_u/v
source_native_wind_value
source_native_gust_value
label_window_minutes
label_source
label_semantics
label_quality_flags
```

Then estimate noise floor:

- intra-window variance;
- source-to-source disagreement at nearby/coincident stations;
- nearest-observation timing error;
- 6-minute versus 15-minute aggregation error;
- gust definition disagreement.

### Next Required Action

### Action Taken

Added:

```text
scripts/ml_dataset/audit_label_semantics_v1.py
```

Generated on z2:

```text
/srv/data/corsewind/ml_dataset/source_inventories/label_semantics_audit_v1_latest.md
```

Recent-window result from `2026-06-25` to `2026-07-02`:

| Track | Rows | Spots | Median cadence | Median offset to 15-min grid |
| --- | ---: | ---: | ---: | ---: |
| `official` | `2864` | `7` | `12.0 min` | `3.0 min` |
| `product` | `809` | `7` | `1.98 min` | `3.93 min` |
| `context` | `1352` | `8` | `0.8 min` | `3.88 min` |

Interpretation:

- product/Winds-Up observations are much more frequent than the official track;
- both product and context observations are often several minutes away from the
  15-minute forecast grid;
- this supports the review claim that nearest-observation labels can inject
  non-trivial alignment noise into RMSE.

### Next Required Action

Build canonical centered-window labels and estimate the true label noise floor
before chasing RMSE `<0.9 m/s` again.

## P1. Quantile / Exceedance Heads

### Review Claim

Manual guards are an ad hoc replacement for probabilistic heads.

### Decision

Accepted.

Current guards are useful as fast production-safe probes, but the target
architecture should expose:

- P10/P50/P90 wind;
- P10/P50/P90 gust;
- calibrated `P(wind >= 12/15/20/25 kt)`;
- calibrated `P(gust >= 15/20/25/30 kt)`.

Manual guards can then become policy decisions on probabilities instead of
hard-coded threshold hacks.

## P1. Direction / Vector Wind

### Review Claim

Speed alone is insufficient for windsurf. Direction is core product semantics.

### Decision

Accepted.

We should predict residuals on vector components:

```text
delta_u_ms
delta_v_ms
```

Then derive:

- corrected speed;
- corrected direction;
- onshore/offshore/side-shore score;
- upwind station logic aligned with output.

## P1. Guard Tuning Separation

### Review Claim

Guard thresholds may be tuned and evaluated on the same tiny rollup.

### Decision

Accepted.

All guard parameters must be frozen on a tuning period and scored only on
disjoint fresh days.

Current `144` row results remain hypothesis-generating only.

## P1. Run Provenance

### Review Claim

Champion runs need manifest provenance: command, git SHA, versions, checksums.

### Decision

Accepted.

This is a reproducibility requirement and should be added to every training and
promotion run.

## P2. Thermal Onset / Collapse Metrics

### Review Claim

Thermal timing is promised but not yet implemented as a label or metric.

### Decision

Accepted.

Before a model head, we need an evaluator:

```text
first sustained crossing of useful wind threshold
last sustained crossing / collapse
timing error in minutes
```

This should be computed by spot and direction regime.

## P2. Physical Feature Hygiene

### Review Claim

The physical feature failure does not refute physics because the feature store
likely includes too many sparse QC flags and low-coverage satellite columns.

### Decision

Accepted.

Physical data should first be reduced to a small regime layer:

- land-sea thermal delta;
- cloud regime;
- instability index;
- fetch/exposure;
- relief/upwind indicators;
- pressure/temperature gradients.

QC flags should control availability/trust, not flood the model as sparse
features.

## Revised Execution Order

1. Ensure operational AROME/AROME-PI archive is always on.
2. Add archive audit as a daily health signal.
3. Add official/product scoring tracks.
4. Build canonical label semantics and label noise-floor audit.
5. Freeze guard tuning/scoring split.
6. Add probabilistic quantile/exceedance heads.
7. Add vector wind residuals.
8. Add thermal onset/collapse evaluator.
9. Rebuild physical regime layer with curated features only.

## Impact On Current Shadow Campaign

The current shadow campaign remains useful, but it is not sufficient for
promotion until:

- product-track scoring exists;
- operational NWP archive continuity is confirmed;
- guard parameters are treated as frozen and evaluated on disjoint days.

Current candidates should remain shadow-only.

## Bottom Line

The next win is probably not a bigger model.

The next win is a better data contract:

```text
same production NWP baseline
+ product-representative spots
+ canonical labels
+ quantified label noise
+ probabilistic/vector outputs
```

Only after that should we expect the champion to become meaningfully beatable.
