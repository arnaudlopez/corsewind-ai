# ML Storage And Backfill Plan

## Current Mode

The project is currently in `backfill_only` mode.

Continuous collection, watchers, and recurring jobs stay disabled until the
historical dataset foundation is built and reviewed. The immediate goal is to
populate past observations and past forecast features on z2, then assemble
training tables from that historical corpus.

## Storage Rule

Large historical backfills must not write to the repository disk. Set
`ML_DATASET_ROOT` to a large external volume before running any source that can
download NetCDF, GRIB, CSV archives, or multi-year API responses.

```bash
export ML_DATASET_ROOT=/Volumes/<large-disk>/corsewind/ml_dataset
python3 scripts/ml_dataset/storage_preflight.py \
  --ml-root "$ML_DATASET_ROOT" \
  --min-free-gb 250 \
  --create
```

For the remote z2 machine, first verify the SSH storage path:

```bash
python3 scripts/ml_dataset/remote_storage_preflight.py \
  --host z2 \
  --remote-path /srv/data/corsewind/ml_dataset \
  --min-free-gb 250 \
  --create
```

Validated z2 target:

```text
z2:/srv/data/corsewind/ml_dataset
free space checked: 869 GiB
```

Current local machine check on 2026-06-23: the system data volume has less than
1 GiB free, so it is not acceptable for backfills.

## Source Order

The canonical source registry is `configs/ml_backfill_sources.json`.

1. Météo-France DPClim observations: historical labels and climatology features.
2. Open-Meteo Historical Forecast: recent multi-year forecast features at our
   spot coordinates.
3. Open-Meteo Previous Runs: horizon-specific forecast-error learning.
4. Météo-France AROME vertical profiles: pressure-level air-column features.
5. Copernicus Marine SST: observed sea-surface temperature.
6. EUMETSAT MTG/LSA SAF products: cloud type, land surface temperature and
   instability indices.
7. MeteoNet: older research pretraining dataset.
8. Météo-France PNT 14-day retention: recent native GRIB bridge.

## Dataset Layout

All source collectors should write below `ML_DATASET_ROOT`:

```text
observations/meteo_france_climatology/
open_meteo/historical_forecast/
open_meteo/previous_runs/
research/meteonet/
meteo_france_pnt_14d/
meteo_france_nwp/vertical_profiles/
copernicus_marine/sst_samples/
eumetsat/cloud_type_samples/
eumetsat/land_surface_temperature_samples/
eumetsat/global_instability_indices_samples/
feature_store/
source_inventories/
```

## z2 ML Runner

The heavy collectors run through the Docker image
`corsewind-ml-dataset-runner:latest` on z2. It is built from
`Dockerfile.ml-dataset-runner` and includes the base project requirements plus
`requirements-ml-dataset.txt`.

Build command:

```bash
cd /srv/data/corsewind/backfill_runner
docker build -f Dockerfile.ml-dataset-runner \
  -t corsewind-ml-dataset-runner:latest .
```

Run pattern:

```bash
docker run --rm \
  -e ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
  -v /srv/data/corsewind/backfill_runner:/work \
  -v /srv/data/corsewind/ml_dataset:/srv/data/corsewind/ml_dataset \
  -w /work \
  corsewind-ml-dataset-runner:latest \
  python <collector.py> <args>
```

First smoke-test command for forecast backfill:

```bash
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/collect_open_meteo_historical_forecast.py \
  --start-date 2025-06-15 \
  --end-date 2025-06-15 \
  --spot-id balistra
```

Smoke-test result on z2:

- `meteofrance_arome_france`: 24 rows for `balistra` on 2025-06-15, complete
  wind, pressure, cloud, radiation, CAPE, precipitation fields.
- `meteofrance_arome_france_hd`: 24 rows for the same spot/day, complete wind
  and temperature fields, but pressure/radiation/cloud-total fields are missing.

Use `meteofrance_arome_france` as the primary rich-feature backfill model, and
optionally add `meteofrance_arome_france_hd` for wind-focused longer history.

Current historical forecast backfill launched on z2:

```bash
cd /srv/data/corsewind/backfill_runner
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/collect_open_meteo_historical_forecast.py \
  --start-date 2024-01-02 \
  --end-date 2026-06-23 \
  --include-context-spots \
  --model meteofrance_arome_france \
  --max-days-per-request 31 \
  --request-sleep-sec 0.2
```

The first full run completed with intermittent Open-Meteo API timeouts. A
repair pass is running with smaller seven-day chunks and skip-existing enabled:

```bash
cd /srv/data/corsewind/backfill_runner
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/collect_open_meteo_historical_forecast.py \
  --start-date 2024-01-02 \
  --end-date 2026-06-23 \
  --include-context-spots \
  --model meteofrance_arome_france \
  --max-days-per-request 7 \
  --request-sleep-sec 0.5 \
  --timeout-sec 90
```

Coverage can be audited with:

```bash
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/audit_open_meteo_coverage.py \
  --start-date 2024-01-02 \
  --end-date 2026-06-23 \
  --include-context-spots
```

Latest repair audit snapshot:

- Expected rows: 542,400 (`25 spots x 904 days x 24 hours`).
- Observed rows: 542,232.
- Missing rows: 168.
- Complete spots: 19 / 25.
- A final one-day-chunk repair pass left only 168 missing rows; this is good
  enough for the first training-table pilots.

First smoke-test command for DPClim observation backfill:

```bash
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/collect_meteo_france_dpclim_backfill.py \
  --frequency hourly \
  --station-id 20004003 \
  --start-datetime 2025-06-15T00:00:00Z \
  --end-datetime 2025-06-15T23:59:59Z
```

Smoke-test result on z2:

- 24 hourly rows for station `20004003` / spot `la_parata`.
- Correct UTC timestamps from `2025-06-15T00:00:00Z` to
  `2025-06-15T23:00:00Z`.
- Wind, gust, direction, temperature, humidity, station pressure and sea-level
  pressure fields were populated.

Current DPClim hourly backfill launched on z2 for the Météo-France spot
stations:

```bash
cd /srv/data/corsewind/backfill_runner
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/collect_meteo_france_dpclim_backfill.py \
  --frequency hourly \
  --station-id 20107001 \
  --station-id 20004003 \
  --station-id 20114002 \
  --station-id 20004002 \
  --station-id 20342001 \
  --station-id 20093002 \
  --station-id 20041001 \
  --start-datetime 2024-01-02T00:00:00Z \
  --end-datetime 2026-06-23T23:59:59Z \
  --max-days-per-order 31 \
  --request-sleep-sec 1.2
```

Hourly DPClim result on z2:

- 151,872 normalized rows.
- 904 dates from `2024-01-02` to `2026-06-23`.
- 7 Météo-France spot stations.
- No command errors.

DPClim station metadata backfill result on z2:

- 40 station info summaries after adding all DPClim hourly wind stations in
  Corse.
- Raw and normalized station metadata are stored under
  `observations/meteo_france_climatology/station_info/`.
- `configs/ml_context_stations.json` is the contextual station registry. It
  stores each station GPS position, altitude, open/public status, parameter
  groups, nearest ML spot, nearest Beacon spot and context role.
- Current registry: 40 DPClim hourly wind stations in Corse.
- Context roles: coastal official context, inland thermal context, mountain
  relief context and regional official context.
- The 7 original Météo-France spot stations remain target-capable stations.
- The 33 additional stations are context stations by default; they should be
  used as official nearby/coastal/inland/relief features, not direct windsurf
  targets unless explicitly promoted later.

Current DPClim contextual hourly backfill launched on z2:

```bash
cd /srv/data/corsewind/backfill_runner
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/collect_meteo_france_dpclim_backfill.py \
  --frequency hourly \
  --station-id <33-context-stations> \
  --start-datetime 2024-01-02T00:00:00Z \
  --end-datetime 2026-06-23T23:59:59Z \
  --max-days-per-order 31 \
  --request-sleep-sec 1.2
```

Monitor it with:

```bash
tail -f /srv/data/corsewind/backfill_runner/logs/dpclim_context_hourly_backfill.log
```

DPClim contextual hourly result on z2:

- 678,528 hourly observation rows in `dpclim_station_hourly`.
- 40 DPClim hourly wind stations in Corse are registered, including 7 direct
  target stations and 33 contextual coastal, inland, relief and regional
  stations.
- Some expected 404 responses remain for station/period combinations where
  Météo-France has no hourly data.

DPClim 6-minute caveat:

The `infrahoraire-6m` endpoint was tested, but it returns precipitation field
`RR6` for the tested stations, not wind. It must not be used as a wind target.
It may be revisited later as fine precipitation context after mapping `RR6`.

Important distinction:

- DPClim historical `commande-station/infrahoraire-6m` is documented in the
  local swagger as a precipitation climatology endpoint, and our backfill
  confirmed it is not a historical 6-minute wind source.
- DPObs / DPPaquetObs real-time and recent endpoints expose 6-minute wind
  fields (`ff`, `dd`, `raf10`, `ddraf10`) and were tested successfully during
  the API inventory phase. They are useful for live collection and short recent
  catch-up, but they are not the same thing as a multi-year historical
  6-minute wind backfill.

The stopped test command was:

```bash
cd /srv/data/corsewind/backfill_runner
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/collect_meteo_france_dpclim_backfill.py \
  --frequency 6min \
  --station-id 20107001 \
  --station-id 20004003 \
  --station-id 20114002 \
  --station-id 20004002 \
  --station-id 20342001 \
  --station-id 20093002 \
  --station-id 20041001 \
  --start-datetime 2024-01-02T00:00:00Z \
  --end-datetime 2026-06-23T23:59:59Z \
  --max-days-per-order 7 \
  --request-sleep-sec 1.2
```

Open-Meteo Previous Runs is implemented as a P1 source for horizon-specific
error learning:

```bash
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
python3 scripts/ml_dataset/collect_open_meteo_previous_runs.py \
  --start-date 2025-06-15 \
  --end-date 2025-06-15 \
  --spot-id balistra \
  --model best_match \
  --variables wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,pressure_msl \
  --lead-days 1,2
```

## AROME Vertical Profiles

The vertical profile collector is implemented and smoke-tested on z2.

Validated smoke test:

- Product: Météo-France AROME WCS 0.025.
- Run: `2026-06-24T09:00:00Z`.
- Valid times: 5 lead times from H+0 to H+4.
- First spot: `balistra`, then expanded to the 20 ML spots in
  `configs/ml_spots.json`.
- Pressure levels: `1000`, `950`, `925`, `900`, `850` hPa.
- Features: temperature, relative humidity, vertical pressure velocity,
  geopotential height and pseudo-adiabatic potential temperature.
- Result: 125 GeoTIFF slices downloaded and 5 spot-profile rows written under
  `meteo_france_nwp/vertical_profiles/`.
- Expanded result: 100 profile rows for 20 ML spots x 5 lead times.

Example derived features now available per spot/time:

- `geopotential_thickness_1000_850_m`.
- `temperature_lapse_rate_1000_850_c_per_km`.
- `relative_humidity_mean_1000_850_pct`.
- `vertical_velocity_pressure_850_pa_s`.
- `low_level_inversion_strength_c`.

This is the primary dataset block for the "air column" thermal-energy signal.

## Copernicus Marine SST

The SST collector is implemented and smoke-tested on z2.

Validated smoke test:

- Dataset:
  `cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m`.
- Variable: `analysed_sst`.
- Window: `2026-06-22T12:00:00` to `2026-06-22T15:00:00`.
- First spot: `balistra`, then expanded to the 20 ML spots.
- Result: 4 rows, 4 valid SST values written under
  `copernicus_marine/sst_samples/date=2026-06-22/`.
- Expanded result: 80 valid SST rows for 20 ML spots x 4 hours.
- Controlled windsurf-window backfill for `2026-06-22T10:00:00Z` to
  `18:00:00Z`: 180 valid SST rows for 20 ML spots x 9 hourly timestamps.
- The controlled run used `scripts/ml_dataset/run_copernicus_sst_backfill.py`
  and `--delete-raw-after-sample`; only the original smoke-test NetCDF files
  remain under `copernicus_marine/raw/sst/`.

Credential note:

- The earlier Copernicus account worked for this test.
- The later credential pair was rejected by Copernicus Marine.

SST is required for observed land-sea thermal contrast when combined with
EUMETSAT land surface temperature.

## EUMETSAT Satellite Products

The generic EUMETSAT spot-product collector is implemented and smoke-tested on
z2 for the three selected products.

Validated smoke tests on `2026-06-24` at `balistra`, then expanded to the 20
ML spots:

- `cloud_type` / `EO:EUM:DAT:0680`: sampled `cloud_type`, `cloud_phase`,
  `quality_overall_processing`, `quality_illumination` and
  `quality_nwp_parameters`.
- `land_surface_temperature` / `EO:EUM:DAT:1088`: sampled `LST`,
  `LST_uncertainty` and `QFLAGS`.
- `global_instability_indices` / `EO:EUM:DAT:0683`: sampled `k_index`,
  `lifted_index`, precipitable-water layers and `percent_cloud_free`.
- Expanded result: 20 rows per product for the selected product time.

Historical-depth caveat:

- Quarterly and monthly availability probes were run on z2 with
  `scripts/ml_dataset/inventory_eumetsat_availability.py`.
- `global_instability_indices` has products from `2025-02` onward in the
  monthly probe.
- `cloud_type` has products from `2026-01` onward in the monthly probe.
- `land_surface_temperature` has products from `2026-02` onward in the monthly
  probe.
- A `cloud_type` query for `2024-06-24T10:00:00Z` to `12:00:00Z` returned zero
  products for collection `EO:EUM:DAT:0680`.
- Before a large satellite backfill, split the products by their real archive
  depth and keep raw NetCDF cleanup enabled.

Pilot backfill result:

- Window: `2026-06-23T10:00:00Z` to `12:00:00Z`.
- Spots: 20 ML spots.
- `cloud_type`: 14 products, 280 sampled rows.
- `land_surface_temperature`: 12 products, 240 sampled rows.
- `global_instability_indices`: 14 products, 280 sampled rows.
- `--delete-raw-after-sample` removed downloaded NetCDF files after sampling;
  retained raw EUMETSAT smoke-test files are only about 33 MiB.

Controlled one-day windsurf-window backfill:

- Window: `2026-06-22T10:00:00Z` to `18:00:00Z`, split into 2-hour chunks.
- Runner: `scripts/ml_dataset/run_eumetsat_spot_backfill.py`.
- Commands: 12 / 12 succeeded.
- `cloud_type`: 1,000 rows for `2026-06-22`.
- `land_surface_temperature`: 960 rows for `2026-06-22`.
- `global_instability_indices`: 1,000 rows for `2026-06-22`.
- Raw EUMETSAT directory after the run: about 33 MiB, confirming cleanup works.

## Feature Store Pilot

The first joined training-table pilot was built on z2 with:

```bash
cd /srv/data/corsewind/backfill_runner
docker run --rm \
  -e ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
  -v /srv/data/corsewind/backfill_runner:/work \
  -v /srv/data/corsewind/ml_dataset:/srv/data/corsewind/ml_dataset \
  -w /work \
  corsewind-ml-dataset-runner:latest \
  python scripts/ml_dataset/build_spot_feature_store.py \
    --start-datetime 2026-06-22T10:00:00Z \
    --end-datetime 2026-06-22T18:00:00Z \
    --output-root /srv/data/corsewind/ml_dataset/feature_store/pilot_20260622 \
    --schema-doc /srv/data/corsewind/ml_dataset/feature_store/pilot_20260622/feature_store_schema.md
```

Pilot result:

- 63 training rows: 7 target-capable official stations x 9 hourly target times.
- 974 feature columns after adding Météo-France DPClim context-station
  features.
- 29 Open-Meteo / `meteofrance_arome_france` feature columns, including wind,
  gust, pressure, cloud cover, radiation, CAPE, precipitation and derived
  `wind_u_10m` / `wind_v_10m`.
- 782 context-station feature columns. The builder selects nearest, coastal,
  inland, relief and regional Météo-France stations from
  `configs/ml_context_stations.json`, then adds a GPS-distance fallback with
  the closest available DPClim stations in Corse. It also always tries to add
  one global coastal station, one global relief/mountain station and one global
  inland station near the target spot. It attaches only observations available
  before the target time to avoid future leakage.
- `context_agg_*` is the preferred training surface for context stations. It
  summarizes nearby, coastal, inland, relief and all-context groups using
  counts, distances, age, altitude, wind, gust, wind `u/v`, temperature,
  humidity, pressure and deltas versus the latest real target-spot observation.
- Raw per-station `context_*` columns remain useful for audit/debug and feature
  research, but should not be blindly passed to the first production model.
- Source coverage: 63 / 63 rows with Open-Meteo forecast, SST, context
  stations, Cloud Type and
  Global Instability Indices; 56 / 63 rows with Land Surface Temperature;
  495 context-station slot observations attached across the 63 rows, including
  nearby, coastal and relief/mountain signals.
- Output:
  `/srv/data/corsewind/ml_dataset/feature_store/pilot_20260622/spot_forecast_15min.jsonl`.

Important caveat:

- The pilot is hourly because the currently available historical target labels
  are DPClim hourly rows. The 15-minute grain is already supported in the
  schema; true 15-minute supervised rows require a target observation source at
  15-minute or 6-minute wind cadence.
- DPClim 6-minute rows currently contain precipitation context, not wind. The
  builder now searches backward for the latest observation row with actual
  values so those sparse 6-minute rows do not mask the latest hourly wind and
  temperature values.

Smoke-test result:

- 48 rows for `balistra` on 2025-06-15 (`24h x 2 lead-day offsets`).
- `best_match` returned populated day-1/day-2 data.
- Météo-France AROME previous-day variables returned nulls on the tested sample,
  so Previous Runs should initially be treated as global/seamless context, not
  as a replacement for native AROME archives.

## MeteoNet Research Dataset

MeteoNet is accepted as a research / pretraining source, not as the main recent
spot-level training source.

Official scope:

- areas: North-West and South-East France;
- years: 2016, 2017, 2018;
- ground observations: 6-minute station observations with wind direction
  `dd`, wind speed `ff`, precipitation, humidity, dew point, temperature and
  sea-level pressure;
- weather models: daily forecast model files with 2D and 3D parameters;
- radar, satellite and masks are also available.

References:

- `https://meteofrance.github.io/meteonet/english/data/summary/`
- `https://meteofrance.github.io/meteonet/english/data/ground-observations/`
- public data root: `https://meteonet.umr-cnrm.fr/dataset/data/`

Inventory on z2:

```bash
cd /srv/data/corsewind/backfill_runner
docker run --rm \
  -e ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
  -v /srv/data/corsewind/backfill_runner:/work \
  -v /srv/data/corsewind/ml_dataset:/srv/data/corsewind/ml_dataset \
  -w /work \
  corsewind-ml-dataset-runner:latest \
  python scripts/ml_dataset/inventory_meteonet_dataset.py \
    --zones SE \
    --output-json /srv/data/corsewind/ml_dataset/source_inventories/meteonet_inventory.json \
    --output-md /srv/data/corsewind/ml_dataset/source_inventories/meteonet_inventory.md
```

SE inventory result:

- `SE/ground_stations`: 3 files, 2016-2018, about 2.14 GiB. This is the first
  MeteoNet block to download because it gives true 6-minute wind observations.
- `SE/weather_models/2D_parameters`: about 21.06 GiB.
- `SE/weather_models/3D_parameters`: about 7.33 GiB.
- Radar/satellite blocks are available but should be downloaded later only if
  we decide to train spatial nowcasting/pretraining models.

Download priority:

1. `SE/ground_stations` for 6-minute observation pretraining.
2. `SE/weather_models/2D_parameters` and `SE/weather_models/3D_parameters` for
   learning observation-model error relationships.
3. Radar/satellite only after the tabular station/model baseline is useful.

Download result on z2:

- `SE_ground_stations_2016.tar.gz`: downloaded, complete.
- `SE_ground_stations_2017.tar.gz`: downloaded, complete.
- `SE_ground_stations_2018.tar.gz`: downloaded, complete.
- Raw path:
  `/srv/data/corsewind/ml_dataset/research/meteonet/raw/SE/ground_stations/`.

Fast station-coverage probe:

- `2016`: 483 stations seen in first timestamp, 32 stations in Corsica bbox.
- `2017`: 482 stations seen in first timestamp, 36 stations in Corsica bbox.
- `2018`: 482 stations seen in first timestamp, 40 stations in Corsica bbox.

This confirms MeteoNet SE is useful for Corsica 6-minute wind pretraining.

Normalization result on z2:

```bash
cd /srv/data/corsewind/backfill_runner
docker run --rm \
  -e ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
  -v /srv/data/corsewind/backfill_runner:/work \
  -v /srv/data/corsewind/ml_dataset:/srv/data/corsewind/ml_dataset \
  -w /work \
  corsewind-ml-dataset-runner:latest \
  python scripts/ml_dataset/normalize_meteonet_ground_stations.py \
    --input-root /srv/data/corsewind/ml_dataset/research/meteonet/raw/SE/ground_stations \
    --output-root /srv/data/corsewind/ml_dataset/research/meteonet/normalized/ground_stations \
    --profile /srv/data/corsewind/ml_dataset/research/meteonet/normalized/ground_stations/profile.json \
    --zone SE
```

- Normalized Corsica-bbox rows: 9,934,994.
- Stations: 51 unique Corsica-bbox stations across 2016-2018.
- Time range: `2016-01-01T00:00:00Z` to `2018-12-31T23:54:00Z`.
- Normalized size: about 4.8 GiB.
- Output:
  `/srv/data/corsewind/ml_dataset/research/meteonet/normalized/ground_stations/`.
- Station registry:
  `/srv/data/corsewind/ml_dataset/research/meteonet/normalized/ground_stations/zone=SE/stations.json`.

Annual rows:

- 2016: 3,044,773 rows, 39 stations, 2,049,217 non-null wind-speed rows.
- 2017: 3,344,351 rows, 42 stations, 2,098,756 non-null wind-speed rows.
- 2018: 3,545,870 rows, 50 stations, 2,265,931 non-null wind-speed rows.

Small documentation artifacts can remain in `docs/ml_nowcasting/`; large raw
and normalized datasets must stay under `ML_DATASET_ROOT`.
