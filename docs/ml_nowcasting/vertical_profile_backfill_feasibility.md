# Vertical Profile Backfill Feasibility

Date: 2026-06-27

## Decision

Use Open-Meteo Historical Forecast pressure-level variables as the immediate
historical vertical-profile backfill for the 2024-2026 training window.

Keep the native Meteo-France AROME WCS vertical-profile collector for forward
collection and recent native samples, but do not rely on it yet for multi-year
historical backfill. The local z2 inventory currently proves only recent native
vertical rows, from 2026-06-24 to 2026-06-26.

## Why

The RMSE 0.9 audit shows that the hard thermal regimes still lack vertical
air-column signal. The current best validated model is still at RMSE 1.269403.
Post-hoc specialists and shallow ensembles did not generalize. The missing
signal is therefore more likely data/physics than another residual layer.

Open-Meteo Historical Forecast is usable now because:

- it exposes pressure-level hourly variables;
- it supports Météo-France AROME France;
- AROME France historical forecast coverage starts on 2024-01-02 in the
  Open-Meteo model table;
- it works through the existing point collector with only a `--hourly` override.

Official references:

- https://open-meteo.com/en/docs/historical-forecast-api
- https://open-meteo.com/en/docs
- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
- https://meteofrance.github.io/meteonet/english/data/weather-models/

## Smoke Test

Command run on z2:

```bash
cd /srv/data/corsewind/backfill_runner
SMOKE=/srv/data/corsewind/ml_dataset/source_inventories/open_meteo_pressure_level_smoke
rm -rf "$SMOKE"
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
  /home/z2/corsewind-ml-smoke/.venv/bin/python \
  scripts/ml_dataset/collect_open_meteo_historical_forecast.py \
  --output-root "$SMOKE" \
  --start-date 2025-06-15 \
  --end-date 2025-06-15 \
  --model meteofrance_arome_france \
  --spot-id balistra \
  --max-days-per-request 1 \
  --request-sleep-sec 0 \
  --timeout-sec 60 \
  --hourly wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_1000hPa,temperature_950hPa,temperature_925hPa,temperature_900hPa,temperature_850hPa,relative_humidity_1000hPa,relative_humidity_950hPa,relative_humidity_925hPa,relative_humidity_900hPa,relative_humidity_850hPa,geopotential_height_1000hPa,geopotential_height_950hPa,geopotential_height_925hPa,geopotential_height_900hPa,geopotential_height_850hPa,wind_speed_1000hPa,wind_speed_950hPa,wind_speed_925hPa,wind_speed_900hPa,wind_speed_850hPa,wind_direction_1000hPa,wind_direction_950hPa,wind_direction_925hPa,wind_direction_900hPa,wind_direction_850hPa
```

Result:

- model: `meteofrance_arome_france`
- spot: `balistra`
- day: `2025-06-15`
- rows: `24`
- time range: `2025-06-15T00:00:00Z` to `2025-06-15T23:00:00Z`
- all requested pressure-level fields returned `24/24` non-null values.

Returned pressure-level families:

- temperature at 1000/950/925/900/850 hPa;
- relative humidity at 1000/950/925/900/850 hPa;
- geopotential height at 1000/950/925/900/850 hPa;
- wind speed at 1000/950/925/900/850 hPa;
- wind direction at 1000/950/925/900/850 hPa.

## Implemented Integration

The existing Open-Meteo collector already writes arbitrary hourly variables into
the `features` map. The spot feature store already flattens these variables
under:

```text
model_open_meteo_meteofrance_arome_france_*
```

The residual training table now derives explicit vertical features when the
pressure-level fields are present:

- `open_meteo_vertical_geopotential_thickness_1000_850_m`
- `open_meteo_vertical_temperature_lapse_rate_1000_850_c_per_km`
- `open_meteo_vertical_temperature_delta_1000_850_c`
- `open_meteo_vertical_temperature_delta_1000_950_c`
- `open_meteo_vertical_temperature_delta_950_850_c`
- `open_meteo_vertical_relative_humidity_mean_1000_850_pct`
- `open_meteo_vertical_relative_humidity_delta_1000_850_pct`
- `open_meteo_vertical_wind_shear_speed_1000_850_ms`
- `open_meteo_vertical_wind_shear_direction_1000_850_deg`
- `open_meteo_vertical_low_level_inversion_strength_c`

The training backfill runner now accepts:

```text
--open-meteo-hourly <comma-separated variables>
```

so the pressure-level backfill can run through the normal pipeline.

## Backfill Command

Recommended first full vertical Open-Meteo backfill:

```bash
ssh z2 'cd /srv/data/corsewind/backfill_runner && \
ML_DATASET_ROOT=/srv/data/corsewind/ml_dataset \
/home/z2/corsewind-ml-smoke/.venv/bin/python \
scripts/ml_dataset/run_training_backfill_pipeline.py \
  --ml-root /srv/data/corsewind/ml_dataset \
  --start-date 2024-01-02 \
  --end-date 2026-06-23 \
  --chunk-days 7 \
  --collect-open-meteo \
  --include-context-spots \
  --open-meteo-model meteofrance_arome_france \
  --open-meteo-max-days-per-request 7 \
  --open-meteo-request-sleep-sec 0.2 \
  --open-meteo-timeout-sec 90 \
  --no-open-meteo-skip-existing-complete \
  --open-meteo-hourly wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,shortwave_radiation,direct_radiation,diffuse_radiation,cape,lifted_index,boundary_layer_height,precipitation,rain,showers,temperature_1000hPa,temperature_950hPa,temperature_925hPa,temperature_900hPa,temperature_850hPa,relative_humidity_1000hPa,relative_humidity_950hPa,relative_humidity_925hPa,relative_humidity_900hPa,relative_humidity_850hPa,geopotential_height_1000hPa,geopotential_height_950hPa,geopotential_height_925hPa,geopotential_height_900hPa,geopotential_height_850hPa,wind_speed_1000hPa,wind_speed_950hPa,wind_speed_925hPa,wind_speed_900hPa,wind_speed_850hPa,wind_direction_1000hPa,wind_direction_950hPa,wind_direction_925hPa,wind_direction_900hPa,wind_direction_850hPa \
  --continue-on-error'
```

After this backfill:

1. rebuild the feature store for 2024-01-02 to 2026-06-23;
2. rebuild residual training tables;
3. rerun the 2025 -> 2026 short-horizon benchmark;
4. rerun feature-family coverage audit and RMSE09 reduction audit.

## Alternatives

ERA5 pressure levels are a robust fallback because they provide hourly pressure
level reanalysis over a very long period, but their grid is too coarse to
replace AROME for Corsican coastal thermal effects. Use ERA5 as synoptic
stability context if Open-Meteo AROME pressure-level backfill later proves
insufficient.

MeteoNet 3D is useful for pretraining/research because it includes 3D weather
model files, but its 2016-2018 window does not align directly with the current
2024-2026 target training/evaluation stack.
