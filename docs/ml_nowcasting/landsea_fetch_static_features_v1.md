# Land/Sea Maritime Fetch Static Features V1

This prepares true maritime fetch features from a raster land/sea mask.

## Source

ESA WorldCover 2021 v200, 10 m.

Downloader:

`scripts/download_esa_worldcover_tiles.py`

Required tiles for full Corsica coverage:

- `ESA_WorldCover_10m_2021_v200_N39E006_Map.tif`
- `ESA_WorldCover_10m_2021_v200_N39E009_Map.tif`
- `ESA_WorldCover_10m_2021_v200_N42E006_Map.tif`
- `ESA_WorldCover_10m_2021_v200_N42E009_Map.tif`

Current state:

The 4 Corsica tiles were downloaded locally and staged on `z2` for
reproducibility.

- local: `data/raw/landcover/esa_worldcover_v200_2021/`
- z2: `/srv/data/corsewind/backfill_runner/data/raw/landcover/esa_worldcover_v200_2021/`

## Generator

Script:

`scripts/ml_dataset/generate_landsea_fetch_static_features.py`

Default output:

`configs/ml_spot_static_features.fetch_v1.json`

This output merges on top of:

`configs/ml_spot_static_features.json`

so the final file can contain both:

- DEM relief/exposure features;
- true land/sea fetch features.

## Features

For each 8-sector direction:

- sample count / missing count;
- water share;
- land share;
- first water distance;
- first land distance;
- direct water fetch;
- coastal-snapped water fetch;
- longest water run;
- longest land run.

Cross-sector deltas are also emitted:

- north minus south fetch;
- northeast minus southwest fetch;
- east minus west fetch;
- southeast minus northwest fetch.

## Interpretation

For a forecast wind direction, the model can learn whether the upwind sector is:

- open sea;
- quickly blocked by land;
- offshore then land-interrupted;
- coast-adjacent with a short land snap before water.

This is the missing physical signal behind phrases like:

- exposure côte;
- fetch maritime;
- cross-shore vs alongshore;
- wind shadow by land.

## Operational Plan

Do not activate this during the running `phys_v1` rebuild.

Target clean rebuild:

`residual_windsup_sst_prev_phys_v2_dem`

or, if we want to distinguish DEM-only from DEM+fetch:

`residual_windsup_sst_prev_phys_v3_dem_fetch`

Prepared z2 launch/audit scripts:

- `scripts/ml_dataset/z2_launch_phys_v3_dem_fetch_rebuild.sh`
- `scripts/ml_dataset/z2_phys_v3_dem_fetch_signal_audit_watcher.sh`

Staged static file:

- local: `configs/ml_spot_static_features.fetch_v1.json`
- z2: `/srv/data/corsewind/backfill_runner/configs/ml_spot_static_features.fetch_v1.json`

Sanity-check examples:

- `la_tonnara`: west snapped water fetch `47.75 km`, east `0.0 km`.
- `balistra`: east snapped water fetch `39.25 km`, west `0.25 km`.
- `santa_manza`: east snapped water fetch `37.5 km`, west `1.5 km`.
- `lfkf`: all maritime fetch directions `0.0 km`, as expected for the inland airport station.
