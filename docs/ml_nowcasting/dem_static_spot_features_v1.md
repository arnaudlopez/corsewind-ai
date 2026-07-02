# DEM Static Spot Features V1

We now generate real DEM-derived static spot features from Copernicus GLO-30
tiles.

## Source

Local DEM tiles:

`data/raw/dem/copernicus_glo30/`

Current local coverage:

- `N41 E008`
- `N41 E009`
- `N42 E008`
- `N42 E009`
- `N43 E009`

The `N43 E009` tile was added after DNS/network recovery, so Cap Corse now has
full radial/sector sampling in the configured 20 km DEM window.

## Generator

Script:

`scripts/ml_dataset/generate_dem_spot_static_features.py`

Generated config:

`configs/ml_spot_static_features.json`

This config is compatible with `build_spot_feature_store.py`, which injects each
scalar as:

`features.spot_static_<name>`

and the residual training table keeps those columns as:

`features__spot_static_<name>`

## Current Output

Generated spots:

- `25`

Unique DEM/static feature names:

- `193`

Main feature families:

- spot/reference elevation
- nearest land elevation and distance
- radial elevation/relief stats at `1`, `2`, `5`, `10`, `20 km`
- 8-sector elevation/relief stats over `20 km`
- sector barrier max/p90/share
- nearest sector barrier distance
- DEM-derived lowland/sea share by sector
- DEM-derived open-exposure proxy by sector
- nearest 500 m mountain-barrier distance by sector
- cross-sector relief gradients

Important examples:

- La Tonnara: east relief/barrier is much stronger than west sea-side relief.
- Porticcio: east-side relief rises sharply toward the mountains.
- Figari/LFKF: strong surrounding relief signal.
- Ajaccio buoy: no DEM reference because it is offshore and farther than the
  configured nearest-land fallback.

## Limits

The current file includes DEM-derived exposure proxies, not a true vector
coastline/fetch model. It can tell the model that one sector is low/open and
another sector rapidly hits relief, but it does not yet compute exact maritime
fetch over a land/sea polygon.

Remaining static geography layer:

- coastline-derived maritime fetch;
- coastline bearing/orientation;
- cross-shore and alongshore angle for each forecast wind direction;
- explicit strait/channel geometry scores.

## Prepared Land/Sea Fetch V1

Prepared script:

`scripts/ml_dataset/generate_landsea_fetch_static_features.py`

This computes true maritime fetch from ESA WorldCover land/sea rasters by
sampling rays from each spot along the 8 cardinal/intercardinal sectors.

Generated feature family:

- `fetch_sector_<sector>_water_share`
- `fetch_sector_<sector>_land_share`
- `fetch_sector_<sector>_first_water_distance_km`
- `fetch_sector_<sector>_first_land_distance_km`
- `fetch_sector_<sector>_direct_water_fetch_km`
- `fetch_sector_<sector>_coastal_snapped_water_fetch_km`
- `fetch_sector_<sector>_longest_water_run_km`
- `fetch_sector_<sector>_longest_land_run_km`
- opposite-sector deltas such as `fetch_sector_e_minus_w_fetch_km`

This is a raster land/sea mask, not a simplified GPS proxy. It can be merged on
top of the DEM static features by passing:

```bash
python scripts/ml_dataset/generate_landsea_fetch_static_features.py \
  --base-static-features configs/ml_spot_static_features.json \
  --output configs/ml_spot_static_features.fetch_v1.json
```

To cover all Corsica, ESA WorldCover tiles still need to be available locally:

- `N42E006`
- `N42E009`
- `N45E006`
- `N45E009`

## Operational Decision

Do not copy this file to z2 as `configs/ml_spot_static_features.json` while the
current `phys_v1` rebuild is running.

Reason:

`phys_v1` monthly shards already produced before this file existed would not
contain `spot_static_*`, while later shards would. That would make the benchmark
harder to interpret.

Recommended next dataset after `phys_v1`:

`residual_windsup_sst_prev_phys_v2_dem`

It should start from a clean monthly rebuild with the DEM static config present
from month one.
