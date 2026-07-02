# Phys V2 DEM Rebuild Plan

`phys_v2_dem` is the clean dataset rebuild that adds DEM static geography to the
`phys_v1` physical signals.

## Prefix

`residual_windsup_sst_prev_phys_v2_dem`

## Why A New Prefix

The current `phys_v1` rebuild started before DEM static features were ready.
Injecting `spot_static_*` in the middle of that rebuild would create mixed
monthly shards:

- early months without DEM features;
- later months with DEM features.

So `phys_v1` remains coherent, and DEM features start with a clean rebuild under
`phys_v2_dem`.

## Static Feature File

Staged on z2:

`/srv/data/corsewind/backfill_runner/configs/ml_spot_static_features.dem_v1.json`

It is intentionally not named:

`/srv/data/corsewind/backfill_runner/configs/ml_spot_static_features.json`

so the running `phys_v1` rebuild cannot accidentally pick it up.

## Launch Script

Prepared script:

`scripts/ml_dataset/z2_launch_phys_v2_dem_rebuild.sh`

Default guard:

It refuses to start until the `phys_v1` decision report watcher has completed.

Manual launch after `phys_v1`:

```bash
ssh z2 'cd /srv/data/corsewind/backfill_runner && bash scripts/ml_dataset/z2_launch_phys_v2_dem_rebuild.sh'
```

Override guard only if explicitly needed:

```bash
ssh z2 'cd /srv/data/corsewind/backfill_runner && REQUIRE_PHYS_V1_DONE=0 bash scripts/ml_dataset/z2_launch_phys_v2_dem_rebuild.sh'
```

## Signal Audit

Prepared script:

`scripts/ml_dataset/z2_phys_v2_dem_signal_audit_watcher.sh`

It checks the `phys_v1` physical signal families plus DEM static features:

- `features__spot_static_dem_reference_elevation_m`
- `features__spot_static_dem_radius_10p0km_relief_max`
- `features__spot_static_dem_sector_n_20km_barrier_max_m`
- `features__spot_static_dem_sector_s_20km_barrier_max_m`
- `features__spot_static_dem_relief_gradient_e_minus_w_m`

Expected outputs:

- `/srv/data/corsewind/ml_dataset/training_tables/phys_v2_dem_required_feature_audit.json`
- `/srv/data/corsewind/ml_dataset/training_tables/phys_v2_dem_required_feature_audit.md`

## Benchmark

The existing low-memory post-rebuild watcher can be reused with:

```bash
ML_ROOT=/srv/data/corsewind/ml_dataset \
PREFIX=residual_windsup_sst_prev_phys_v2_dem \
RUN_SUFFIX=phys_v2_dem \
STATUS=/srv/data/corsewind/ml_dataset/run_logs/phys_v2_dem_post_rebuild_lowmem.status \
bash scripts/ml_dataset/z2_regime_v1_post_rebuild_lowmem_watcher.sh
```

Expected benchmark run IDs:

- `tabular_lgbm_225k_prev_phys_v2_dem_2024_2025_to_2026_v1`
- `tabular_lgbm_calbase_phys_v2_dem_2024_to_2025h2_v1`
- `prediction_residual_calibrator_2025h2_to_2026_extratrees_autoscale_phys_v2_dem`
