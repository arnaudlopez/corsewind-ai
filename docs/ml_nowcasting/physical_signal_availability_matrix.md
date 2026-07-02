# Physical Signal Availability Matrix

Date: 2026-06-28

Question:
For the selected physical signals, do we already have the information, can we
compute it from current data, or do we need additional sources/backfill?

Reference:

- signal selection: `docs/ml_nowcasting/physical_signal_selection.md`
- diagnostic report: `docs/ml_nowcasting/scientific_error_diagnostic_report.md`

Remote audit source:

- host: `z2`
- table prefix audited:
  `/srv/data/corsewind/ml_dataset/training_tables/residual_windsup_sst_prev_regime_v1_YYYY_MM/training_rows.parquet`
- months: `2024-01` to `2026-06`
- files: `30`
- rows: `1,528,776`

Important distinction:

- The champion model artifact does not contain all newer regime features.
- The newer `regime_v1` training tables contain many thermal/upwind/previous-run
  features, but not vertical-profile derived features and not static
  spot-exposure geometry.

## Summary

| Signal family | In current tables? | Can compute from current data? | Need external/source work? | Decision |
| --- | --- | --- | --- | --- |
| Low-level vertical stability | Not in training table | Yes, from Open-Meteo pressure-level raw files | No new access for Open-Meteo; rebuild/derive needed | P0 next implementation |
| Air-sea / coast-relief thermal contrast | Mostly yes | Yes | True LST needs EUMETSAT forward/history | Use now, improve LST later |
| Pressure gradients coast-inland-relief | Schema yes, useful values mostly missing | Partly from NWP; station pressure sparse | Need better pressure source/offset sampling | P0 source+feature work |
| Upwind propagation | Yes | Yes | Better coverage/station registry only | Keep and harden |
| Spot exposure by wind direction | No | Partly, from static geo layers | Need coastline/DEM/fetch preprocessing | P0 static feature build |
| Forecast evolution/run-to-run | Yes | Yes | No new source for current Open-Meteo path | Keep and extend |
| EUMETSAT LST/cloud/instability | Sparse/recent | Partly | Needs historical proof or forward only | P1, not blocking |

## Audit Results From Current `regime_v1` Tables

| Feature | Status | Non-null rows | Coverage |
| --- | --- | ---: | ---: |
| `features__sst_c` | present | 1,521,923 | 99.552% |
| `features__thermal_air_minus_sst_c` | present | 1,518,902 | 99.354% |
| `features__thermal_land_minus_sst_c` | present but sparse | 2,863 | 0.187% |
| `features__thermal_land_minus_air_c` | present but sparse | 2,325 | 0.152% |
| `features__thermal_inland_minus_coastal_temperature_c` | present | 1,525,914 | 99.813% |
| `features__thermal_relief_minus_coastal_temperature_c` | present | 1,525,593 | 99.792% |
| `features__thermal_inland_minus_coastal_pressure_hpa` | present but empty | 0 | 0.000% |
| `features__thermal_relief_minus_coastal_pressure_hpa` | present but empty | 0 | 0.000% |
| `features__thermal_recent_heating_rate_c_per_h` | present but partial | 249,338 | 16.310% |
| `features__thermal_recent_pressure_tendency_hpa_per_h` | present but partial | 214,088 | 14.004% |
| `features__context_agg_all_upwind_weighted_wind_mean_ms_mean` | present | 1,370,998 | 89.679% |
| `features__context_agg_coastal_upwind_weighted_wind_mean_ms_mean` | present | 1,346,189 | 88.057% |
| `features__context_agg_inland_upwind_weighted_wind_mean_ms_mean` | present | 693,116 | 45.338% |
| `features__context_agg_relief_upwind_weighted_wind_mean_ms_mean` | present | 706,735 | 46.229% |
| `features__context_global_relief_1_wind_mean_ms` | present | 1,523,162 | 99.633% |
| `features__context_global_relief_1_temperature_c` | present | 1,525,537 | 99.788% |
| `features__context_global_relief_1_pressure_hpa` | present but empty | 0 | 0.000% |
| `features__previous_run_open_meteo_best_match_day1_wind_speed_10m` | present | 1,500,219 | 98.132% |
| `features__previous_run_open_meteo_best_match_day2_wind_speed_10m` | present | 1,498,600 | 98.026% |
| `features__nwp_horizon_wind_ramp_ms` | present | 1,525,343 | 99.775% |
| `features__nwp_horizon_gust_ramp_ms` | present | 1,525,343 | 99.775% |
| `features__nwp_horizon_pressure_msl_ramp_hpa` | present | 1,525,343 | 99.775% |
| `features__nwp_horizon_surface_pressure_ramp_hpa` | present | 1,525,343 | 99.775% |
| `features__nwp_horizon_shortwave_ramp` | present | 1,525,343 | 99.775% |
| `features__nwp_horizon_cloud_cover_ramp_pct` | present | 1,525,343 | 99.775% |
| `features__eumetsat_land_surface_temperature_LST_c` | present but sparse | 1,660 | 0.109% |
| `features__open_meteo_vertical_geopotential_thickness_1000_850_m` | absent | 0 | 0.000% |
| `features__vertical_arome_geopotential_thickness_1000_850_m` | absent | 0 | 0.000% |
| `features__spot_fetch_km_for_forecast_wind_dir` | absent | 0 | 0.000% |

## P0 Signal 1 - Low-Level Vertical Stability

Current state:

- Not present in current training tables as derived features.
- Native Météo-France vertical profile collector exists, but current native data
  is recent-only and too short for training.
- Open-Meteo historical forecast raw files already contain pressure-level
  variables for `meteofrance_arome_france`.

Evidence:

- `904/904` audited Open-Meteo daily forecast files contained
  `temperature_1000hPa`.
- Sample files also contain temperature, relative humidity, wind speed, wind
  direction, and geopotential height at `1000/950/925/900/850 hPa`.

Can compute now?

Yes. We can compute:

- lapse rate;
- inversion strength;
- 1000-850 hPa thickness;
- low-level humidity mean/delta;
- wind shear speed/direction;
- mixing/stability index.

Work needed:

- Rebuild feature store/training table so these raw Open-Meteo pressure-level
  fields become derived `features__open_meteo_vertical_*` columns.
- Keep Météo-France native WCS vertical profiles for forward collection.

No new access is needed for the Open-Meteo historical route.

## P0 Signal 2 - Thermal Contrast

Current state:

- SST is present and strong.
- Air-SST is already present with high coverage.
- Inland/coastal and relief/coastal temperature deltas are present with high
  coverage.
- True land-surface temperature exists only sparsely.

Can compute now?

Yes for the historical backbone:

- `air - SST`;
- `inland air - coastal air`;
- `relief air - coastal air`;
- NWP/radiation/cloud-based heating proxies.

Partially for heating tendencies:

- current `thermal_recent_heating_rate_c_per_h` coverage is only `16.310%`.
- this should be improved by computing an NWP-based heating tendency, not only
  an observation-dependent tendency.

Need source work?

- True `land - SST` and `land - air` require reliable land surface temperature.
- Current EUMETSAT LST value coverage is only `0.109%`; it should be treated as
  forward/P1 unless historical coverage is proven.

## P0 Signal 3 - Pressure Gradients

Current state:

- NWP pressure and pressure ramps are present with high coverage.
- Some station pressure exists in earlier/champion artifacts, but the current
  `regime_v1` explicit inland/relief pressure delta features are empty.
- `context_global_relief_1_pressure_hpa` is present but empty.

Can compute now?

Partly:

- from NWP MSLP/surface pressure at the spot;
- from NWP pressure ramps;
- from existing station pressure where available.

Not enough yet for the desired physical feature:

- coast-inland-relief station pressure gradient is not currently usable;
- relief station pressure coverage appears absent in the current table.

Need source/feature work:

1. Add NWP offset-point sampling around each spot:
   sea-side point, coastal point, inland point, relief/upwind point.
2. Compute pressure vector/gradient from those NWP points.
3. Continue using station pressure when available, but do not depend on it as
   the only pressure-gradient source.

This is source work inside Open-Meteo/Météo-France NWP sampling, not necessarily
a new API provider.

## P0 Signal 4 - Upwind Propagation

Current state:

- Already present.
- All/coastal upwind coverage is strong.
- Inland/relief upwind coverage is moderate.
- Relief global wind and temperature are now strong after the active-station
  relief fix.

Can compute now?

Yes. We can compute and use:

- upwind weighted wind/gust/temperature;
- upwind station count;
- upwind weight sum;
- upwind age;
- upwind tendency/ramp;
- coastal/inland/relief exchange indices.

Need source work?

Mostly no. Needed work is quality/coverage hardening:

- make station selection window-aware everywhere;
- keep active relief stations;
- verify no target leakage;
- evaluate hard spots separately.

## P0 Signal 5 - Directional Spot Exposure

Current state:

- Not present.
- No `fetch`, `exposure`, `channel`, or `relief_blocking` columns are present in
  current training tables.

Can compute now?

Partly. We already have:

- spot GPS positions;
- forecast wind direction;
- station/spot geometry.

But to compute useful physical exposure, we need static geographic layers:

- coastline / land-sea mask;
- DEM/elevation around the spot;
- possibly rough terrain sectors or slope/aspect.

Need source work:

- Add a static preprocessing step using a coastline source and a DEM source.
- Candidate sources: OpenStreetMap/Natural Earth coastline for fetch; Copernicus
  DEM, SRTM, or IGN/RGE ALTI for relief/elevation if available.

This is not time-series backfill; it is a one-time static feature build.

## P0 Signal 6 - Forecast Evolution And Run-To-Run Instability

Current state:

- Already present and well covered.
- NWP horizon ramps are around `99.775%`.
- Previous-run Open-Meteo day1/day2 wind speed is around `98%`.

Can compute now?

Yes. We can extend from current data:

- wind/gust ramps;
- pressure ramps;
- temperature/radiation/cloud ramps;
- previous-run wind/pressure/direction deltas;
- forecast consistency score.

Need source work?

No new source for the Open-Meteo path. If later we want native Météo-France
previous AROME runs, that is separate, but not required for the next iteration.

## P1 Satellite Signals

Current state:

- EUMETSAT collections exist and are integrated structurally.
- Actual historical values are too sparse for training.
- LST value coverage in current table is `0.109%`.

Can compute now?

Only for recent/forward periods.

Need source work?

Yes if we want historical use:

- prove historical availability/download depth;
- backfill LST/cloud/instability;
- quality-filter values and rebuild tables.

Decision:

Do not block the next model on EUMETSAT historical coverage. Keep it as forward
collection and future feature validation.

## Concrete Next Work

1. Integrate Open-Meteo pressure-level raw variables into derived vertical
   training features.
2. Add NWP offset-point sampling for pressure/temperature gradients around each
   spot.
3. Add static spot-exposure geometry from coastline and DEM.
4. Harden existing upwind/thermal features and evaluate them on hard-regime
   metrics, not only global RMSE.
5. Treat EUMETSAT LST/cloud/instability as forward-only until historical
   coverage is proven.
