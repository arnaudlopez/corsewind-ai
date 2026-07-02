# v_next As Champion Expert - Decision Report

Generated: 2026-07-01

## Objective

Test whether `v_next` should be used as a cautious secondary expert around the
current champions:

```text
blend = champion + alpha * clip(v_next - champion)
```

The tested alpha range was deliberately small: `0.05`, `0.10`, `0.15`, `0.20`.

## Protocol

- Evaluation keys: strict same-key intersection between champion predictions
  and v_next predictions.
- Key: `spot_id + issue_time_utc + target_time_utc + lead_time_minutes`.
- Targets: wind mean and gust.
- Rows per target: `24276`.
- Blend selection window: before `2026-04-01T00:00:00Z`.
- Holdout window: from `2026-04-01T00:00:00Z`.
- Promotion gates remain:
  - wind champion official RMSE: `1.268019`
  - gust champion official RMSE: `1.484221`

Artifacts:

- Remote benchmark: `/srv/data/corsewind/ml_dataset/benchmarks/vnext_champion_blends_2026h1_v1`
- Local JSON: `docs/ml_nowcasting/vnext_champion_blend_results_2026h1_v1.json`
- Local auto report: `docs/ml_nowcasting/vnext_champion_blend_results_2026h1_v1.md`

## Wind Mean

Same-key full 2026H1:

| Prediction | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| Raw AROME/Open-Meteo | `2.161885` | `1.651248` | `0.328109` |
| Champion `scale070` | `1.298842` | `0.967489` | `0.044222` |
| v_next | `1.344099` | `0.999217` | `0.143419` |
| Pair oracle, not deployable | `1.174058` | `0.838035` | `0.068534` |

Selected cautious blend:

```text
blend_all_a0.2_clip1
```

| Window | Champion RMSE | Blend RMSE | Gain |
| --- | ---: | ---: | ---: |
| Calibration | `1.407252` | `1.400116` | small positive |
| Holdout | `1.194227` | `1.188602` | `+0.471016%` |
| Full same-key | `1.298842` | `1.292466` | `+0.4909%` |

Threshold CSI changes are tiny but mostly neutral-positive:

| Threshold | Champion CSI | Blend CSI |
| --- | ---: | ---: |
| `>=12 kt` | `0.748944` | `0.750059` |
| `>=15 kt` | `0.689814` | `0.692346` |
| `>=20 kt` | `0.616742` | `0.620435` |
| `>=25 kt` | `0.614764` | `0.613975` |

Decision: do not promote as wind champion. The blend is slightly better on the
same-key test, but full RMSE `1.292466` is still worse than the official
champion gate `1.268019`.

## Gust

Same-key full 2026H1:

| Prediction | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| Raw AROME/Open-Meteo | `3.918143` | `3.010380` | `2.577239` |
| Champion `new_scale070_gust_recipe` | `1.538418` | `1.124678` | `0.081652` |
| v_next | `1.592368` | `1.164728` | `0.161290` |
| Pair oracle, not deployable | `1.374317` | `0.963252` | `0.087785` |

Selected cautious blend:

```text
blend_all_a0.2_clip2
```

| Window | Champion RMSE | Blend RMSE | Gain |
| --- | ---: | ---: | ---: |
| Calibration | `1.706548` | `1.695511` | small positive |
| Holdout | `1.371749` | `1.361503` | `+0.746930%` |
| Full same-key | `1.538418` | `1.527820` | `+0.6889%` |

Threshold CSI changes are small but mostly positive:

| Threshold | Champion CSI | Blend CSI |
| --- | ---: | ---: |
| `>=12 kt` | `0.809206` | `0.811589` |
| `>=15 kt` | `0.777778` | `0.779864` |
| `>=20 kt` | `0.693555` | `0.694546` |
| `>=25 kt` | `0.615136` | `0.618652` |

Decision: do not promote as gust champion. The blend is slightly better on the
same-key test, but full RMSE `1.527820` is still worse than the official
champion gate `1.484221`.

## Interpretation

This is a constructive result, not a dead end.

v_next is weaker than the champions as a standalone model, but it contains
complementary signal. The pair oracle confirms this clearly:

- wind oracle RMSE: `1.174058`, versus champion `1.298842`;
- gust oracle RMSE: `1.374317`, versus champion `1.538418`.

The deployable cautious blend captures only a small part of that oracle gap:

- wind full same-key gain: about `0.49%`;
- gust full same-key gain: about `0.69%`.

So the signal exists, but the current trust rule is too blunt. The best tested
rule is simply a global `20%` move toward v_next, not a narrow regime gate. That
means the model is not yet telling us confidently when v_next should override
the champion; it only says v_next is useful as a small shrinkage expert.

## Decision

Do not replace the production champions.

Keep current champions:

- wind mean: `scale070`;
- gust: `new_scale070_gust_recipe`.

Keep the v_next blend as a research candidate:

```text
wind_candidate = champion + 0.20 * clip(v_next - champion, -1.0, +1.0)
gust_candidate = champion + 0.20 * clip(v_next - champion, -2.0, +2.0)
```

Do not turn this on in production yet. The gains are positive but small, and
they do not beat the official champion gates.

## Next Step

The next useful step is not another unconstrained router. It is a trust-signal
diagnostic:

1. Identify rows where v_next beats champion by more than `0.5 m/s`.
2. Compare their issue-time features: lead, spot, raw/champion gap, thermal
   deltas, vertical features, previous-run disagreement, and strong wind regime.
3. Train a very small binary trust classifier only if the separability is real.
4. Keep the final output constrained to a capped delta around the champion.

Promotion remains blocked until a candidate beats the official gates on the
agreed same-key protocol.
