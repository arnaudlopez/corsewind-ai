# Router v1 Strategy Results

Generated: `2026-07-01`

Objective: test the next strategy after `strong_gated_v2_active`: route between
available forecast rails instead of forcing one global model.

Rails tested:

```text
wind: champion, raw, strong_gated
gust: champion, raw, gust_high, strong_gated
```

Source batch:

```text
/srv/data/corsewind/ml_dataset/live_inference/hindcast_batch_strong_gated_v2_active
```

Rows: `783` scored rows over `6` pseudo-live hindcasts.

## Methods

Two validation modes were tested.

### Leave-One-Hindcast-Out

Output:

```text
/srv/data/corsewind/ml_dataset/benchmarks/router_v1_hindcast_strong_gated_v2_active/router_v1_results.json
```

This is useful as a smoke test, but it can be optimistic because adjacent
hindcasts can cover overlapping target times.

### Target-Day Blocked

Output:

```text
/srv/data/corsewind/ml_dataset/benchmarks/router_v1_target_day_blocked_strong_gated_v2_active/router_v1_results.json
```

This is the more honest scientific check for this tiny sample because it avoids
training on another hindcast that may already include the same target period.

## Leave-One-Hindcast-Out Result

| Target | Champion RMSE | Raw RMSE | Strong/Gust High RMSE | Router RMSE | Stacker RMSE | Oracle RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wind mean | 3.615 | 4.167 | 3.541 strong | 3.563 | 2.783 | 3.284 |
| gust | 5.065 | 6.755 | 5.267 gust_high / 4.970 strong | 4.692 | 3.394 | 3.814 |

Interpretation:

- The router improves gust RMSE by `+7.37%` vs champion.
- The router improves wind mean RMSE by `+1.44%` vs champion, but does not beat
  `strong_gated` on wind mean.
- The stacker looks very strong, but this validation is probably too optimistic.

Gust threshold CSI in this optimistic validation:

| Threshold | Champion | Router | Stacker | Raw | Gust high | Strong gated | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| >=15 kt | 0.253 | 0.443 | 0.611 | 0.391 | 0.369 | 0.285 | 0.591 |
| >=20 kt | 0.032 | 0.045 | 0.424 | 0.200 | 0.094 | 0.039 | 0.286 |
| >=25 kt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Target-Day Blocked Result

| Target | Champion RMSE | Raw RMSE | Strong/Gust High RMSE | Router RMSE | Stacker RMSE | Oracle RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wind mean | 3.615 | 4.167 | 3.541 strong | 3.885 | 4.554 | 3.284 |
| gust | 5.065 | 6.755 | 5.267 gust_high / 4.970 strong | 5.361 | 6.577 | 3.814 |

Interpretation:

- The router does not generalize when target day is held out.
- The stacker collapses in target-day blocked validation, which confirms that
  the optimistic stacker result is not production evidence.
- The oracle remains much better than champion, especially for gusts, so there
  is real complementary signal in the rails.

Gust threshold CSI in target-day blocked validation:

| Threshold | Champion | Router | Stacker | Raw | Gust high | Strong gated | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| >=15 kt | 0.253 | 0.258 | 0.224 | 0.391 | 0.369 | 0.285 | 0.591 |
| >=20 kt | 0.032 | 0.038 | 0.052 | 0.200 | 0.094 | 0.039 | 0.286 |
| >=25 kt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Wind threshold CSI in target-day blocked validation:

| Threshold | Champion | Router | Stacker | Raw | Strong gated | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| >=12 kt | 0.081 | 0.138 | 0.142 | 0.147 | 0.125 | 0.197 |
| >=15 kt | 0.059 | 0.061 | 0.016 | 0.071 | 0.086 | 0.088 |
| >=20 kt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Decision

Do not promote `router_v1` or the stacker yet.

What we learned:

1. A routeable signal exists: the row-wise oracle is materially better than the
   champion.
2. The current 6-hindcast sample is too small and too correlated to train a
   robust router.
3. The route/stacks can look excellent under weak validation and fail under
   target-day blocking.
4. Raw AROME remains the strongest simple detector for `gust >=20 kt`, but it
   has too many false positives to become the median forecast.

## Next Step

The strategy is still right, but the blocker is sample diversity.

Build an event replay set before training the production router:

```text
minimum: 30-50 event days
include: weak, thermal, 15+ kt, 20+ kt, gust 25+ kt
validation: blocked by target_date, then by weather episode
metrics: RMSE/MAE plus CSI/recall/false positives for windsurf thresholds
```

Until then:

```text
champion_* remains median p50
strong_gated_* remains a strong-wind RMSE candidate/shadow rail
raw/gust_high/probability heads remain risk/alert rails
router_v1 remains research-only
```
