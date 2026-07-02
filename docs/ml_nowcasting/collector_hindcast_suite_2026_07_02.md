# Collector Hindcast Suite - 2026-07-02

## What Changed

Added a reusable collector-backed hindcast suite:

- `scripts/ml_dataset/run_collector_hindcast_suite.py`
- extended `scripts/ml_dataset/score_live_hindcast_predictions.py` with wind-mean regimes:
  - `<12kt`
  - `12-15kt`
  - `15-20kt`
  - `20-25kt`
  - `>=25kt`

The suite runs pseudo-live AROME-PI hindcasts from `/srv/data/corsewind/ml_dataset`, scores them against later observations, and writes:

- `suite_summary.json`
- `suite_summary.md`
- one scored parquet per case

z2 smoke run:

`/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_suite_20260702_v1`

## Smoke Cases

| Case | Window | Character |
| --- | --- | --- |
| `collector_20260630T1745_v1` | 2026-06-30 18:00-23:00 UTC | light evening |
| `collector_20260701T0645_v1` | 2026-07-01 07:00-17:00 UTC | strong wind day |

Both cases completed without failures.

## Overall RMSE

RMSE in m/s.

| Case | Rows | Wind raw | Wind champion | Wind strong | Gust raw | Gust champion | Gust high | Gust strong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `collector_20260630T1745_v1` | 97 | 1.229 | 0.950 | 0.956 | 1.636 | 1.256 | 1.238 | 1.278 |
| `collector_20260701T0645_v1` | 202 | 3.369 | 3.216 | 2.892 | 3.703 | 4.278 | 3.756 | 3.863 |

## Gains Versus Raw

Positive means the candidate beats raw NWP on RMSE.

| Case | Wind champion | Wind strong | Gust champion | Gust high | Gust strong |
| --- | ---: | ---: | ---: | ---: | ---: |
| `collector_20260630T1745_v1` | +0.278 | +0.272 | +0.380 | +0.399 | +0.359 |
| `collector_20260701T0645_v1` | +0.153 | +0.477 | -0.575 | -0.053 | -0.160 |

## Threshold CSI

| Case | Wind >=15 raw | Wind >=15 strong | Gust >=20 raw | Gust >=20 strong | Gust >=25 raw | Gust >=25 high | Gust >=25 strong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `collector_20260630T1745_v1` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `collector_20260701T0645_v1` | 0.325 | 0.537 | 0.516 | 0.608 | 0.542 | 0.423 | 0.120 |

## Decision

The suite confirms two different behaviors:

- Wind mean: `strong_gated` is promising. It improves the 2026-07-01 strong-wind day by `+0.477 m/s` RMSE versus raw and improves `wind >=15kt` CSI from `0.325` to `0.537`.
- Gusts: current ML/gated recipes are still too conservative on strong-wind days. On 2026-07-01, raw remains best for gust RMSE and `gust >=25kt` CSI.

So the next high-leverage step is not another global model. It is a gust/strong-wind router:

1. Detect regimes where raw gust should be trusted or only lightly corrected.
2. Keep the champion/gated correction for calm and moderate gust regimes.
3. Add an upward-capable expert for strong events, validated specifically on `gust >=20kt` and `gust >=25kt`.
4. Promote only if the collector suite shows no calm-regime regression and improves strong thresholds.

## Router Probe 2

After the first suite, I added deterministic no-training router baselines in:

- `scripts/ml_dataset/train_hindcast_router_v1.py`

The most useful new gust rule is:

```text
rule_raw25_else_high:
  if raw_gust_kt >= 25: use raw_gust_kt
  else: use gust_high_kt
```

z2 output:

`/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_suite_20260702_v1/router_v1_rules_probe2`

Validation: leave-one-hindcast-out over the two collector suite cases.

Metrics below are in knots.

| Target | Champion RMSE | Best Practical RMSE | Router Classifier RMSE | Oracle RMSE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| wind mean | 5.245 | 4.740 `strong_gated` | 4.812 | 4.656 | keep `strong_gated` as shadow wind rail |
| gust | 6.975 | 6.055 `rule_raw25_else_high` | 6.441 | 5.369 | test `rule_raw25_else_high` as shadow gust rail |

Gust threshold CSI:

| Rail | `>=15kt` | `>=20kt` | `>=25kt` |
| --- | ---: | ---: | ---: |
| champion | 0.552 | 0.573 | 0.000 |
| raw | 0.603 | 0.516 | 0.542 |
| gust_high | 0.648 | 0.594 | 0.423 |
| strong_gated | 0.569 | 0.608 | 0.120 |
| `rule_raw25_else_high` | 0.648 | 0.594 | 0.542 |
| oracle | 0.683 | 0.656 | 0.608 |

Interpretation:

- The flexible classifier still under-trusts raw gusts in the strongest regime.
- The deterministic rule is better aligned with the physics and the observed errors: use the corrected/high rail in normal conditions, but stop damping AROME-PI when it already predicts very strong gusts.
- `rule_raw25_else_high` beats champion, raw, `gust_high`, `strong_gated`, and the classifier on this two-case suite.
- It also keeps the best available `gust >=25kt` CSI among deployable rails.

Promotion status:

- Do not promote yet. Two hindcasts are not enough.
- Promote only after a wider event replay set confirms:
  - no calm-regime regression;
  - stable wind mean gain from `strong_gated`;
  - stable gust RMSE gain from `rule_raw25_else_high`;
  - `gust >=20kt` and `gust >=25kt` CSI not worse than raw.

Important data warning:

The router run emitted missing-feature warnings for several physical features:

- `features__thermal_coastal_minus_inland_*`
- `features__thermal_relief_minus_coastal_*`
- `features__thermal_recent_*`
- `features__nwp_offset_*_boundary_layer_height`

So these two collector hindcasts are not yet using the full physical signal set we designed. The next long-run goal must therefore combine model improvement with feature availability checks, not just train more estimators.

## Re-run Command

```bash
cd /srv/data/corsewind/backfill_runner
/srv/data/corsewind/pyenv/bin/python scripts/ml_dataset/run_collector_hindcast_suite.py \
  --output-root /srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_suite_20260702_v1 \
  --case 'collector_20260630T1745_v1|2026-06-30T17:00:00Z|2026-06-30T17:45:00Z|2026-06-30T23:00:00Z' \
  --case 'collector_20260701T0645_v1|2026-07-01T06:00:00Z|2026-07-01T06:45:00Z|2026-07-01T17:00:00Z' \
  --registry configs/ml_spots.json \
  --context-registry configs/ml_context_stations.json \
  --spot-static-features configs/ml_spot_static_features.json \
  --python /srv/data/corsewind/pyenv/bin/python \
  --continue-on-error
```
