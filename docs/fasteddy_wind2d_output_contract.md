# FastEddy To Wind2D Output Contract

## Scope

This contract defines what the FastEddy post-processing step must publish for Wind2D and future Beacon Live integration.

Machine-readable contract:

```text
benchmarks/fasteddy/wind2d_output_contract.json
```

## Required Display Fields

At the selected display height, normally 1 m, 2 m or 10 m:

```text
speed_ms
direction_from_deg
u_ms
v_ms
w_ms
vertical_motion_class
confidence
```

Rules:

- `speed_ms = sqrt(u_ms^2 + v_ms^2)`.
- `direction_from_deg` must use the same meteorological convention as AROME/WindNinja layers.
- `w_ms` must be geometric vertical velocity in m/s.
- Data tiles must preserve values for live recolorization.
- Color tiles are visualization products only.

## Required Metadata

Every published layer must include:

```text
source_model = FastEddy
source_arome_run_time_utc
valid_time_utc
zone_id
display_height_m
grid_resolution_m
bbox_wgs84
z0m_lookup_version
icbc_manifest
solver_status
```

## Publication Gate

Do not publish if:

- FastEddy solver status is `fail`;
- display-height interpolation has more than 1% non-finite values over land/nearshore cells;
- AROME run time or valid time is missing;
- direction convention cannot be verified;
- data tiles are missing.

## Integration Target

Expected future locations:

```text
visualizations/wind2d/fasteddy-data/<step>/
visualizations/wind2d/fasteddy-tiles/<step>/
data/processed/diagnostics/fasteddy_<zone>_<step>.json
```

The Wind2D UI should treat FastEddy as a separate selectable layer from AROME and WindNinja until enough validation exists to blend products.
