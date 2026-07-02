# phys_v4 Directional Campaign

## Goal

Explain and test why `phys_v3_dem_fetch` did not beat the current champion,
despite better-organized input data.

Champion reference:

- wind mean RMSE: `1.268019`
- wind mean MAE: `0.930465`

`phys_v3` reference:

- wind mean calibrated RMSE: `1.305533`
- gust calibrated RMSE: `1.527744`

## Hypotheses

1. Volume/capacity hypothesis:
   The champion may have won because it trained on more rows and fewer features.
   `phys_v3` had cleaner data but fewer sampled rows and a much wider feature
   space.

2. Noise/dilution hypothesis:
   Raw static sector grids, satellite fields, previous-run fields, or vertical
   profiles may dilute the strongest observation/context signals.

3. Directional-geometry hypothesis:
   Static fetch/DEM sectors are physically incomplete unless selected relative
   to the actual wind direction. The model should see upwind fetch, downwind
   relief, blocking, and sea-breeze alignment directly.

## Implemented Tests

### `phys_v3_old_signal_225k_bin63`

Purpose: test volume/capacity against a simpler feature set.

It excludes:

- `features__spot_static_`
- `features__nwp_offset_`
- `features__previous_run_`
- `features__open_meteo_vertical_`
- `features__eumetsat_`

Max train rows: `225000`.

### `phys_v3_pruned_200k_bin63`

Purpose: keep most new physical sources, but remove broad noisy grids.

It excludes:

- raw static fetch sector columns
- raw static DEM sector columns
- EUMETSAT columns

Max train rows: `200000`.

### `phys_v4_directional_pruned_200k_bin63`

Purpose: replace raw static sectors with dynamic direction-conditioned
features.

Generated from the existing `phys_v3_dem_fetch` shards, without a heavy
backfill. Each shard gets 36 new columns, including:

- wind-from sector id/center/delta
- upwind/downwind/crosswind fetch
- upwind water/land share
- upwind first-land and longest-water distances
- upwind DEM barrier, relief, and open exposure
- crosswind and downwind barrier proxies
- blocking index
- marine exposure index
- max-fetch sea-breeze alignment

Raw fetch and DEM sector grids are excluded during the pruned benchmark, while
the new `features__directional_*` columns are kept.

## Decision Rule

Promote only if wind mean calibrated RMSE improves against `1.268019`.

Also report gust RMSE/MAE because production now requires both wind mean and
gust forecasts.

If none of these runs improves, the next logical path is not more generic
feature accumulation. It is either:

- a routed specialist that uses physical features only on spots/regimes where
  they win, or
- improving target/history quality for the hard regimes where the residual
  error remains irreducible with current labels.

## Results

Remote summary:

`/srv/data/corsewind/ml_dataset/benchmarks/pruned_v4_directional_150k_resume_summary.json`

Decision: `keep_champion`

| Run | Status | Wind RMSE | Wind MAE | Gust RMSE | Gust MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| `champion` | current best | `1.268019` | `0.930465` | n/a | n/a |
| `phys_v3_old_signal_225k_bin63` | complete | `1.301797` | `0.982492` | `1.516817` | `1.124876` |
| `phys_v3_pruned_150k_bin63` | complete | `1.299284` | `0.980664` | `1.526835` | `1.134749` |
| `phys_v4_directional_pruned_150k_bin63` | complete | `1.312789` | `0.992090` | `1.521491` | `1.131693` |
| `phys_v3_pruned_200k_bin63` | partial base-only | `1.311003` | `0.993864` | `1.534341` | `1.144583` |

## Conclusions

`phys_v3_pruned_150k_bin63` is the best new wind-mean candidate, but it is still
`0.031265 m/s` worse than the champion RMSE.

The 200k pruned run produced a base model but the calibration-base LightGBM
process segfaulted. The machine did not OOM, so this is treated as a native
LightGBM stability limit for that row/feature shape rather than a disk or RAM
capacity problem.

`phys_v4_directional_pruned_150k_bin63` did not validate the current
direction-conditioned feature design for global wind mean prediction. It is
worse than both `phys_v3_pruned_150k_bin63` and `phys_v3_old_signal_225k_bin63`
on wind mean. It is slightly better than `phys_v3_pruned_150k_bin63` on gust
RMSE, but not better than `phys_v3_old_signal_225k_bin63`.

The result argues against more broad feature accumulation in the global model.
The next useful path is a gated specialist/router: use these richer physical
feature sets only where they prove better by spot, lead time, wind regime, or
gust objective, while keeping the champion as the default.
