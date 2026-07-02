# Physical Signal Selection

Date: 2026-06-28

Purpose:
Choose the missing physical signals that should drive the next dataset
iteration for the RMSE 0.9 objective.

Reference diagnostic:

- `docs/ml_nowcasting/scientific_error_diagnostic_report.md`
- Current champion RMSE: `1.268019`
- Main hard regimes: La Tonnara, Santa Manza, lead `+45/+60 min`,
  high wind, thermal onset/decay.

## Decision

Do not add every available meteorological variable. The next dataset should
prioritize the signals that explain local thermal dynamics, high-wind
underprediction, and horizon degradation.

The selected P0 physical signals are:

1. Low-level vertical stability and mixing.
2. Land-sea and air-sea thermal contrast.
3. Coast-inland-relief pressure and temperature gradients.
4. Upwind station propagation and mountain/coastal exchange.
5. Wind-direction conditional spot exposure.
6. Forecast evolution and run-to-run instability.

These six families are the highest-probability missing information. They are
more important than adding another generic ML model on the current table.

## P0 Signal 1 - Low-Level Vertical Stability And Mixing

Why:
Thermal wind depends on whether the lower atmosphere can mix, accelerate, or
stay capped by an inversion. The current champion has no vertical profile
features, while hard errors are strongest at `+45/+60 min`, where regime
evolution matters.

Use:

- pressure levels: `1000`, `950`, `925`, `900`, `850 hPa`
- temperature at each level
- relative humidity at each level
- wind speed and direction at each level
- geopotential height at each level
- if available: vertical velocity and boundary-layer height

Derived features:

- `vertical_temperature_delta_1000_850_c`
- `vertical_lapse_rate_1000_850_c_per_km`
- `vertical_low_level_inversion_strength_c`
- `vertical_relative_humidity_mean_1000_850_pct`
- `vertical_relative_humidity_delta_1000_850_pct`
- `vertical_wind_shear_speed_1000_850_ms`
- `vertical_wind_shear_direction_1000_850_deg`
- `vertical_geopotential_thickness_1000_850_m`
- `vertical_mixing_potential_index`

Decision:
Use Open-Meteo Historical Forecast AROME pressure levels for historical
backfill, and keep native Météo-France vertical profiles for forward/recent
collection.

## P0 Signal 2 - Land-Sea And Air-Sea Thermal Contrast

Why:
This is the core thermal-breeze driver. SST alone is not enough. We need the
temperature contrast between sea, coastal air, inland air, relief air, and
actual heated land surface where available.

Use:

- Copernicus SST
- coastal station air temperature
- inland station air temperature
- relief/mountain station air temperature
- NWP 2 m temperature
- NWP radiation and cloud cover
- EUMETSAT Land Surface Temperature only when actual value coverage is good

Derived features:

- `thermal_air_minus_sst_c`
- `thermal_inland_minus_coastal_temperature_c`
- `thermal_relief_minus_coastal_temperature_c`
- `thermal_land_minus_sst_c`
- `thermal_land_minus_air_c`
- `thermal_recent_heating_rate_c_per_h`
- `thermal_shortwave_x_air_sea_delta`
- `thermal_clear_sky_heating_index`
- `thermal_low_cloud_suppression_index`

Decision:
Treat `thermal_air_minus_sst_c`, inland/coastal/relief deltas, and NWP
radiation/cloud proxies as historical P0. Treat satellite LST as forward P1
until historical coverage is proven; current historical coverage is too sparse
to rely on it.

## P0 Signal 3 - Coast-Inland-Relief Pressure And Temperature Gradients

Why:
The user hypothesis is physically sound: mountain/sea exchange and local
pressure gradients likely explain part of the Corsican thermal timing. The
champion has weak pressure observation coverage and no explicit pressure
gradient family.

Use:

- coastal station pressure
- nearest inland station pressure
- relief/mountain station pressure
- NWP surface pressure and MSLP at spot
- NWP pressure at small land/sea offset points if available

Derived features:

- `gradient_inland_minus_coastal_pressure_hpa`
- `gradient_relief_minus_coastal_pressure_hpa`
- `gradient_coastal_pressure_tendency_1h_hpa`
- `gradient_inland_pressure_tendency_1h_hpa`
- `gradient_relief_pressure_tendency_1h_hpa`
- `gradient_pressure_vector_u_hpa_per_km`
- `gradient_pressure_vector_v_hpa_per_km`
- `gradient_pressure_aligned_with_forecast_wind`
- `gradient_temperature_aligned_with_forecast_wind`

Decision:
Make pressure/temperature gradients explicit. Do not rely on the model to infer
them from independent station columns.

## P0 Signal 4 - Upwind Propagation And Mountain/Coastal Exchange

Why:
The scientific papers we use as a reference show that neighboring stations are
often more valuable than more complex model architecture. Our own diagnostic
also shows context/inland wind features correlated with absolute error. But
generic nearest-station context is weaker than direction-aware propagation.

Use station groups:

- nearest coastal station
- target-like coastal station
- inland station
- relief/mountain station
- all stations within radius

Derived features:

- `upwind_weighted_wind_mean_ms`
- `upwind_weighted_gust_ms`
- `upwind_weighted_temperature_c`
- `upwind_weighted_pressure_hpa`
- `upwind_weighted_wind_direction_deg`
- `upwind_weight_sum`
- `upwind_station_count`
- `upwind_mean_age_minutes`
- `upwind_travel_time_minutes`
- `upwind_wind_ramp_15m_ms`
- `upwind_wind_ramp_60m_ms`
- `relief_to_coast_wind_exchange_index`
- `coast_to_relief_pressure_exchange_index`

Decision:
Keep and harden the existing upwind feature family, then evaluate it on the
critical spots separately. It must be available before target time only.

## P0 Signal 5 - Wind-Direction Conditional Spot Exposure

Why:
A spot is not just a GPS point. La Tonnara, Santa Manza, Balistra, Porticcio,
and Cap Corse react differently depending on wind direction, coastline shape,
relief, channeling, and lee effects. A `spot_id` can learn average bias, but it
cannot represent direction-specific exposure unless we give it geometry.

Use static geometry:

- coastline bearing sectors around each spot
- open-sea fetch by wind-direction sector
- distance to coast along forecast wind direction
- distance to relief along forecast wind direction
- local terrain elevation gradient by sector
- valley/channel alignment score
- lee/shelter score by wind sector

Derived features:

- `spot_fetch_km_for_forecast_wind_dir`
- `spot_open_sea_exposure_for_forecast_wind_dir`
- `spot_relief_blocking_for_forecast_wind_dir`
- `spot_channeling_score_for_forecast_wind_dir`
- `spot_cross_shore_angle_deg`
- `spot_alongshore_angle_deg`
- `spot_thermal_breeze_alignment_score`
- `spot_venturi_sector_score`

Decision:
Add this family as P0 even though it is static-derived. It is a physical signal
because it tells the model when the same synoptic wind should accelerate,
decelerate, or rotate at a specific spot.

## P0 Signal 6 - Forecast Evolution And Run-To-Run Instability

Why:
For `+45/+60 min`, the model needs to know whether the NWP expects acceleration,
rotation, pressure fall, cloud arrival, or a weakening thermal. Recent forecast
runs also tell us whether the NWP itself is stable or correcting quickly.

Use:

- current forecast at issue time
- target-horizon forecast
- previous forecast runs for the same valid time
- NWP wind, gust, direction, pressure, temperature, radiation, cloud, CAPE

Derived features:

- `nwp_horizon_wind_ramp_ms`
- `nwp_horizon_gust_ramp_ms`
- `nwp_horizon_wind_direction_delta_deg`
- `nwp_horizon_pressure_ramp_hpa`
- `nwp_horizon_temperature_ramp_c`
- `nwp_horizon_shortwave_ramp`
- `nwp_horizon_low_cloud_ramp_pct`
- `previous_run_wind_delta_ms`
- `previous_run_direction_delta_deg`
- `previous_run_pressure_delta_hpa`
- `previous_run_consistency_score`

Decision:
Keep this in P0. It is not a pure observation signal, but it is physically
meaningful and directly targets the +45/+60 min degradation.

## P1 Signals

These are useful, but should not block the next training dataset.

### Satellite Cloud/LST/Instability

Use EUMETSAT Cloud Type, LST, and Global Instability Indices for forward
collection and recent validation. Do not depend on them for the 2024-2026
historical target unless coverage becomes much stronger.

Priority:

- low marine cloud / clear sky state
- cloud transition/ramp
- LST quality-filtered land heating
- lifted index / K-index / precipitable water where available

### Humidity And Marine Layer

Useful for detecting low cloud, fog, inversion, and suppressed thermal days.
Much of this is already included in vertical stability if pressure-level
relative humidity is backfilled.

Priority:

- dewpoint depression
- coastal humidity vs inland humidity
- low-level RH mean/delta
- low-cloud suppression index

### Gust/Turbulence/Mixing State

Useful for rafales and for confidence intervals, but less central than wind
mean RMSE.

Priority:

- gust factor
- gust/wind spread
- vertical wind shear
- boundary-layer height
- turbulence proxy from NWP

## P2 / Not Now

These should not be the next focus:

- Radar precipitation as a core signal: useful for squalls and rain regimes,
  but not the main thermal-windsurf failure mode.
- Raw SST alone: already present and high coverage; the missing piece is the
  delta with land/air and the temporal tendency.
- More generic station columns without direction/age/source weighting.
- Another generic global model benchmark before the P0 signals are present.
- EUMETSAT LST as a historical backbone until value coverage is proven.

## Expected Impact

Expected highest-impact combinations:

1. Vertical stability + thermal contrast:
   should help distinguish true thermal days from false warm days.

2. Pressure gradient + mountain/coastal exchange:
   should help La Tonnara, Santa Manza, and Balistra timing errors.

3. Upwind propagation + forecast evolution:
   should help `+45/+60 min` extrapolation and fast corrections.

4. Spot exposure + forecast wind direction:
   should reduce spot-specific amplitude errors and high-wind compression.

## Acceptance Criteria

The next dataset iteration should not be judged only globally. It must report:

- global RMSE/MAE/bias;
- La Tonnara RMSE;
- Santa Manza RMSE;
- critical spots RMSE;
- lead `+45/+60 min` RMSE;
- actual wind `>= 8 m/s` RMSE and bias;
- actual wind `0-2 m/s` RMSE and bias;
- top 5% SSE share;
- route/oracle gain after the new signals are added.

The P0 signal work is successful only if it reduces the hard-regime error, even
if the first global score is noisy.
