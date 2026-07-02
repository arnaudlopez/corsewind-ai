# Router OOF Base LGBM Results - 2026-07-01

Objective: start using the full historical 2024/2025 data safely for router
training, without training on rows that are later used as router labels.

## What Was Built

Built an expanding-window quarterly OOF base LGBM rail on z2.

Output:

`/srv/data/corsewind/ml_dataset/benchmarks/router_oof_24_25_v0/base_lgbm_oof_2024q3_2025q4.parquet`

Summary:

`/srv/data/corsewind/ml_dataset/benchmarks/router_oof_24_25_v0/base_lgbm_oof_summary.md`

Scope:

- rows: `480000`
- issue period: `2024-07-01` to `2025-12-31`
- spots: `14`
- leads: `15`, `30`, `45`, `60` minutes
- folds: `2024q3`, `2024q4`, `2025q1`, `2025q2`, `2025q3`, `2025q4`
- train protocol: expanding window, max `150000` sampled train rows per fold,
  max `80000` holdout rows per fold

Important limitation: this is the OOF base LGBM correction rail. It is not yet
the nested OOF version of the production `scale070` wind champion or the
`new_scale070_gust_recipe` gust champion.

## Overall Metrics

| Target | Raw RMSE | OOF base RMSE | Raw MAE | OOF base MAE | Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| wind mean | `2.012504` | `1.277770` | `1.535308` | `0.939159` | `479882` |
| gust | `3.262192` | `1.493430` | `2.525885` | `1.085308` | `479880` |

## Fold Metrics

| Fold | Wind raw RMSE | Wind OOF RMSE | Gust raw RMSE | Gust OOF RMSE | Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2024q3` | `1.859614` | `1.247171` | `2.981630` | `1.477948` | `80000` |
| `2024q4` | `2.033878` | `1.309745` | `3.399819` | `1.559356` | `80000` |
| `2025q1` | `2.302622` | `1.326424` | `3.657606` | `1.543742` | `80000` |
| `2025q2` | `1.800647` | `1.184895` | `2.873629` | `1.334621` | `80000` |
| `2025q3` | `1.858028` | `1.235943` | `3.100033` | `1.439626` | `80000` |
| `2025q4` | `2.169961` | `1.354344` | `3.486421` | `1.590285` | `80000` |

## Threshold Metrics

| Target | Threshold | CSI | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| wind | `>=12kt` | `0.730887` | `0.850359` | `0.838767` |
| wind | `>=15kt` | `0.673011` | `0.744389` | `0.875293` |
| wind | `>=20kt` | `0.563251` | `0.624539` | `0.851624` |
| wind | `>=25kt` | `0.534757` | `0.623117` | `0.790405` |
| gust | `>=15kt` | `0.739157` | `0.806722` | `0.898223` |
| gust | `>=20kt` | `0.671653` | `0.750646` | `0.864545` |
| gust | `>=25kt` | `0.576110` | `0.689558` | `0.777863` |

## Worst Spot/Lead Groups

Worst wind mean groups:

- `cap_corse`, lead `60`: RMSE `1.846401`
- `la_tonnara`, lead `60`: RMSE `1.582459`
- `balistra`, lead `60`: RMSE `1.491927`
- `lfvf`, lead `60`: RMSE `1.490741`
- `la_tonnara`, lead `45`: RMSE `1.484257`

Worst gust groups:

- `la_tonnara`, lead `60`: RMSE `1.860353`
- `santa_manza`, lead `60`: RMSE `1.831563`
- `balistra`, lead `60`: RMSE `1.768354`
- `la_tonnara`, lead `45`: RMSE `1.748358`
- `santa_manza`, lead `45`: RMSE `1.730500`

## Interpretation

This confirms that a large historical OOF rail is feasible and useful. The base
LGBM correction strongly improves raw NWP on every quarter tested.

However, this does not yet prove a router can beat the current production
champions, because the table does not yet contain leakage-safe historical
versions of:

- wind `scale070`;
- gust `new_scale070_gust_recipe`;
- `strong_gated`;
- foundation model candidates on the same dense keys.

The next rigorous step is nested OOF champion generation:

1. For each evaluation quarter, train the base model only on earlier rows.
2. Produce base predictions for an earlier calibration window and the evaluation
   window.
3. Train the second-stage calibrator only on the calibration window.
4. Apply it to the evaluation window.
5. Assemble `raw`, `base_lgbm_oof`, `scale070_oof`, `gust_recipe_oof`, and
   optional foundation rails into one router table.

Until that exists, router conclusions against the production champion remain
incomplete.

## Nested Scale070 Probe

I also tested the first nested second-stage calibration step, using `2024q3`
OOF predictions as calibration and `2024q4` OOF predictions as evaluation.

Configuration:

- target folds: `2024q4`
- calibration fold: `2024q3`
- model family: `extra_trees`
- correction scale: fixed `0.7`
- clip: `2.0 m/s`
- calibration rows: `80000`

Result:

| Target | Base OOF RMSE | Nested calibrated RMSE | Gain vs base |
| --- | ---: | ---: | ---: |
| wind mean | `1.309745` | `1.313362` | `-0.276%` |
| gust | `1.559356` | `1.571196` | `-0.759%` |

Decision: stopped the full nested ExtraTrees campaign after this probe. The
first fold degraded both wind mean and gust, so blindly applying the current
second-stage recipe quarter-to-quarter is not a promising use of compute.

This does not invalidate the production `scale070` champion, because production
uses a specific 2025H2 calibration -> 2026 evaluation setup. It does show that
for historical router training we should not assume `scale070` transfers
naively across every adjacent quarter.

Better next options:

1. Generate nested candidates with a validation-selected scale per fold instead
   of fixed `0.7`.
2. Try a conservative shrinkage-only second stage, constrained to improve a
   validation sub-window before it can modify evaluation rows.
3. Use the OOF base LGBM rail as the dense historical backbone, and add
   production champion/foundation candidates only where they are leakage-safe.

## OOF Base + Foundation Router Diagnostic

I built an overlap table between the dense OOF base rail and the sparse
foundation historical predictions.

Artifacts:

- train:
  `/srv/data/corsewind/ml_dataset/benchmarks/router_oof_foundation_overlap_v0/train_oof_foundation_overlap_2024q3_2025q4.parquet`
- eval:
  `/srv/data/corsewind/ml_dataset/benchmarks/router_oof_foundation_overlap_v0/eval_foundation_base_champion_2026.parquet`
- router output:
  `/srv/data/corsewind/ml_dataset/benchmarks/router_oof_foundation_overlap_v0/router_common_oof_base_to_2026`

Coverage:

- train rows: `1406`
- eval rows: `1048`
- train spots: `7`
- eval spots: `7`

Candidates used on both train and eval:

- `oof_base`
- `raw`
- `chronos`
- `chronos2`
- `timesfm`

Result:

| Target | Base OOF RMSE kt | Router RMSE kt | Stacker RMSE kt | Oracle RMSE kt | Current champion RMSE kt |
| --- | ---: | ---: | ---: | ---: | ---: |
| wind mean | `2.507000` | `2.812525` | `2.802789` | `1.501344` | `2.481946` |
| gust | `2.914791` | `5.473144` | `3.240542` | `1.758841` | `2.883685` |

Decision: do not promote this router.

The oracle is strong, so there is complementary signal in the candidate set.
But with only `1406` training rows, the classifier over-switches and degrades
both wind and gust. This matches the earlier lesson: flexible routers are
dangerous unless the candidate overlap is dense and the trust policy is
well-constrained.

Next useful move: do not tune this sparse router. Either generate dense
foundation/foundation-like candidates on the OOF keys, or train a conservative
residual blend that is constrained to stay close to the base model.

## Conservative Blend v0

After the sparse router failure, I tested a conservative residual blend:

- train signal: dense OOF 2024/2025 base LGBM residuals;
- model: compact HGB;
- historical scale validation: `2025-07-01` to `2026-01-01`;
- scale candidates: `0` to `0.5`;
- correction clip: `0.75 m/s`;
- evaluation: 2026.

Artifacts:

`/srv/data/corsewind/ml_dataset/benchmarks/router_oof_24_25_v0/conservative_blend_v0`

Result:

| Run | Target | Base RMSE m/s | Blend RMSE m/s | Base RMSE kt | Blend RMSE kt | Gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base 2026 | wind mean | `1.276846` | `1.276471` | `2.481990` | `2.481261` | `+0.029%` |
| champion 2026 | wind mean | `1.268019` | `1.272412` | `2.464832` | `2.473371` | `-0.346%` |
| base 2026 | gust | `1.501342` | `1.493848` | `2.918375` | `2.903808` | `+0.499%` |
| champion 2026 | gust | `1.484221` | `1.484008` | `2.885095` | `2.884681` | `+0.014%` |

Decision: do not promote.

The conservative blend behaves better than the flexible router because it does
not blow up, but it does not produce a meaningful improvement over the current
champions. Gust improves by only `0.014%` around the champion, which is
noise-level. Wind degrades around the champion.

Interpretation: post-hoc residual blending is not currently the high-leverage
path unless we add new dense signals or new candidates. The current champions
should remain:

- wind mean: `scale070`;
- gust: `new_scale070_gust_recipe`.
