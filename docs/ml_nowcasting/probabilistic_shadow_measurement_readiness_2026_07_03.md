# Probabilistic Shadow Measurement Readiness - 2026-07-03

## Scope

No model training was changed during the 2026-07-03 shadow campaign window. This pass only hardens measurement for the 2026-07-05 decision:

- quantile coverage audit and monotone rail application;
- threshold-specific gust decision rail policy frozen from 2025h2 calibration;
- observation QC and source/sensor bias reporting in live hindcast scoring;
- label cadence/noise proxy audit;
- operational NWP archive and disk checks.

## Quantile Calibration

Saved gust quantile calibrators were applied to the 2025h2 calibration prediction set without retraining.

- rows: 60,000 input, 59,960 usable for coverage;
- loaded rails: q33, q40, q50, q60, q75, q90;
- q67/q70 model files were not present, so they were not included;
- pre-sort quantile crossing rows: 6,146;
- inference already sorts q50/q60/q75/q90 monotonically per row, which removes the serving bug class.

Overall coverage on 2025h2 calibration:

| Rail | Expected | Empirical | Error | Conformal Offset kt |
| --- | ---: | ---: | ---: | ---: |
| q33 | 0.33 | 0.3439 | +0.0139 | -0.049 |
| q40 | 0.40 | 0.4079 | +0.0079 | -0.024 |
| q50 | 0.50 | 0.5005 | +0.0005 | -0.001 |
| q60 | 0.60 | 0.5864 | -0.0136 | +0.044 |
| q75 | 0.75 | 0.7201 | -0.0299 | +0.137 |
| q90 | 0.90 | 0.8644 | -0.0356 | +0.338 |

## Frozen Threshold Policy

Policy file: `configs/gust_quantile_threshold_policy_v1.json`.

The rail choice is selected on 2025h2 calibration only, with FAR cap 0.65. It is frozen before scoring the 2026-07-03 to 2026-07-05 shadow campaign.

| Threshold | Rail | CSI | FAR | Precision | Recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12 kt | q60 | 0.750 | 0.171 | 0.829 | 0.887 |
| 15 kt | q75 | 0.733 | 0.185 | 0.815 | 0.880 |
| 20 kt | q60 | 0.699 | 0.154 | 0.846 | 0.800 |
| 25 kt | q60 | 0.600 | 0.248 | 0.752 | 0.749 |

This is not a promotion. It is a pre-registered rule to evaluate on the fresh campaign.

## Product Track Scoring Smoke

The scorer now includes `observation_qc` and `by_observation_source_dataset`.

Smoke run:

- predictions: `/srv/data/corsewind/ml_dataset/live_inference/collector_hindcast_replay_20260702_v1/collector_20260629T0645_replay_v1/predictions/predictions.parquet`;
- observations: Beacon Live + Winds-Up for 2026-06-29;
- output: `/srv/data/corsewind/ml_dataset/live_inference/product_track_score_20260629T0645_qc_v1.json`;
- joined rows: 48;
- spots: balistra, figari_eole, la_tonnara, piantarella, porticcio, porto_polo, santa_manza;
- target window actually scored: 2026-06-29T08:15Z to 2026-06-29T10:00Z;
- median obs distance: 0.81 min;
- p90 obs distance: 7.30 min;
- QC anomalies: 0 gust-below-wind, 0 wind above 80 kt, 0 gust above 110 kt.

Product smoke metrics:

| Target | Model | MAE kt | RMSE kt | Bias kt | Variance Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| wind mean | ML | 2.441 | 3.067 | -1.129 | 0.279 |
| wind mean | raw | 3.114 | 3.838 | -1.583 | 0.686 |
| gust | ML | 2.962 | 3.589 | -1.145 | 0.482 |
| gust | raw | 3.417 | 4.164 | +1.175 | 0.863 |

This smoke verifies the double-track scoring machinery. It does not validate the campaign because Beacon Live product observations are not currently fresh.

## Label Semantics Audit

Audit output:

- `/srv/data/corsewind/ml_dataset/audits/label_semantics_live_window_20260624_20260703.json`;
- `/srv/data/corsewind/ml_dataset/audits/label_semantics_live_window_20260624_20260703.md`.

Window: 2026-06-24T00:00:29Z to 2026-07-03T06:42:00Z.

| Track | Rows | Spots | Last Obs UTC | Median Cadence min | Gust p90 Step ms |
| --- | ---: | ---: | --- | ---: | ---: |
| product | 4,545 | 7 | 2026-06-29T10:04:50Z | 1.93 | 1.03 |
| official | 3,255 | 7 | 2026-07-03T06:42:00Z | 12.00 | 2.10 |
| context | 1,352 | 8 | 2026-06-29T12:05:07Z | 0.80 | 1.96 |

Interpretation:

- official Météo-France labels are fresh for the active campaign;
- product/context Beacon/Winds-Up labels are high cadence but stale after 2026-06-29;
- short-step gust volatility is already around 1-2 m/s p90 depending on source, so a sub-0.9 RMSE target must be interpreted with source cadence and label definition in mind.

## Operational Archive

Audit output:

- `/srv/data/corsewind/ml_dataset/audits/operational_nwp_archive_20260703.json`;
- `/srv/data/corsewind/ml_dataset/audits/operational_nwp_archive_20260703.md`.

Result: 0 archive problems.

| Source | Runs | Last Run UTC | Latest Age h | Sample Rows |
| --- | ---: | --- | ---: | ---: |
| AROME | 7 | 2026-07-03T03:00:00Z | 3.863 | 8,575 |
| AROME-PI | 15 | 2026-07-03T06:00:00Z | 0.863 | 6,625 |

Disk on z2:

- `/srv/data`: 916 GB total, 510 GB free, 42 percent used;
- model runs: 123 MB;
- model samples: 19 MB;
- observations: 5.8 GB;
- benchmarks: 4.2 GB.

## Operational Issue

home101 `corsewind-ml-data-collector` is running but unhealthy. The failing source is `beacon_live_observations`.

Concrete cause observed on 2026-07-03:

- `/beacon-live/weather-state.json` inside the collector is 0 bytes;
- `beacon-live-weather-api` restarts after an upstream Wunderground 503;
- the collector still writes other sources to z2, including Météo-France, AROME/AROME-PI, EUMETSAT cloud type, land surface temperature, and global instability indices.

Decision implication:

- official-track shadow scoring remains viable;
- product-track scoring for current fresh days is blocked until Beacon Live writes a valid weather-state again;
- do not hide this by making Beacon Live optional in the official collector health. The stale product truth should stay visible.

## 2026-07-05 Decision Checklist

1. Roll up the full 2026-07-03 to 2026-07-05 shadow campaign.
2. Score official track first.
3. Score product track only if Beacon Live product observations are fresh again.
4. Report q50/q60/q75/q90 coverage by spot, lead, day, and gust regime.
5. Apply the frozen threshold rail policy and compare against champion/raw.
6. Check q60 stability by day, not only aggregate.
7. Check FAR at 12 kt in calm regimes.
8. Promote only if gates pass; otherwise report failing cells by spot x horizon x bin x source.
