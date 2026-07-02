# Copernicus Marine Product Inventory

Generated at: `2026-06-23T14:15:52.340209Z`

## Decision Summary

| Priority | Decision | Feature | Dataset | Target variables | Notes |
| --- | --- | --- | --- | --- | --- |
| `P1` | `integrate_now` | `sea_surface_temperature` | `cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m` | `analysed_sst` OK | Hourly Mediterranean subskin SST, useful for land-sea thermal contrast. |
| `P2` | `test_after_sst` | `ocean_current` | `cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m` | `uo` OK, `vo` OK | Hourly 2D surface current forecast. Indirect for wind, useful for marine context and validation. |
| `P2` | `test_after_sst` | `mixed_layer` | `cmems_mod_med_phy-mld_anfc_4.2km-2D_PT1H-m` | `mlotst` OK | Mixed layer depth may help characterize coastal water inertia, but impact on 15 min wind is uncertain. |
| `P2` | `test_after_sst` | `waves` | `cmems_mod_med_wav_anfc_4.2km_PT1H-i` | `VHM0` OK, `VMDR` OK, `VTPK` OK | Hourly wave forecast can help explain sea state and observation quality, not the primary thermal driver. |
| `P3` | `backtest_only` | `satellite_sea_wind` | `cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H` | `eastward_wind` OK, `northward_wind` OK | Hourly gridded sea-surface wind, too coarse for spot correction but useful as an independent large-scale check. |
| `P3` | `backtest_only` | `sar_sea_wind` | `cmems_obs-wind_med_phy_nrt_l3-s1a-sar-asc-0.01deg_P1D-i` | `wind_speed` OK, `wind_to_dir` OK, `eastward_wind` OK, `northward_wind` OK | High-resolution SAR sea wind is episodic, so useful for audits/backtests more than operational nowcast. |

## Dataset Details

### `cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m`

- product: `SST_MED_PHY_SUBSKIN_L4_NRT_010_036`
- version: `202105`
- services: `files`, `geoseries`, `timeseries`, `wmts`
- released: `None`
- arco updated: `2026-06-23T11:40:02.945Z`

| Variable | Standard name | Units | BBox | Coordinates |
| --- | --- | --- | --- | --- |
| `analysed_sst` | `sea_surface_subskin_temperature` | `kelvin` | `[-18.125, 30.25, 36.25, 46.0]` | time: step=3600000.0, latitude: step=0.0625, longitude: step=0.0625 |
| `analysis_error` | `None` | `percentage` | `[-18.125, 30.25, 36.25, 46.0]` | time: step=3600000.0, latitude: step=0.0625, longitude: step=0.0625 |
| `mask` | `None` | `1` | `[-18.125, 30.25, 36.25, 46.0]` | time: step=3600000.0, latitude: step=0.0625, longitude: step=0.0625 |
| `sea_ice_fraction` | `sea_ice_area_fraction` | `1` | `[-18.125, 30.25, 36.25, 46.0]` | time: step=3600000.0, latitude: step=0.0625, longitude: step=0.0625 |

### `cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m`

- product: `MEDSEA_ANALYSISFORECAST_PHY_006_013`
- version: `202511`
- services: `files`, `geoseries`, `timeseries`, `wmts`
- released: `2025-11-25T13:00:00.000Z`
- arco updated: `2026-06-23T12:14:19.775Z`

| Variable | Standard name | Units | BBox | Coordinates |
| --- | --- | --- | --- | --- |
| `uo` | `eastward_sea_water_velocity` | `m s-1` | `[-17.29166603088379, 30.1875, 36.29166793823242, 45.97916793823242]` | latitude: step=0.04166667002172143, longitude: step=0.041666668644218384, time: step=3600000.0 |
| `vo` | `northward_sea_water_velocity` | `m s-1` | `[-17.29166603088379, 30.1875, 36.29166793823242, 45.97916793823242]` | latitude: step=0.04166667002172143, longitude: step=0.041666668644218384, time: step=3600000.0 |

### `cmems_mod_med_phy-mld_anfc_4.2km-2D_PT1H-m`

- product: `MEDSEA_ANALYSISFORECAST_PHY_006_013`
- version: `202511`
- services: `files`, `geoseries`, `timeseries`, `wmts`
- released: `2025-11-25T13:00:00.000Z`
- arco updated: `2026-06-23T12:17:22.509Z`

| Variable | Standard name | Units | BBox | Coordinates |
| --- | --- | --- | --- | --- |
| `mlotst` | `ocean_mixed_layer_thickness_defined_by_sigma_theta` | `m` | `[-17.29166603088379, 30.1875, 36.29166793823242, 45.97916793823242]` | latitude: step=0.04166667002172143, longitude: step=0.041666668644218384, time: step=3600000.0 |

### `cmems_mod_med_wav_anfc_4.2km_PT1H-i`

- product: `MEDSEA_ANALYSISFORECAST_WAV_006_017`
- version: `202311`
- services: `files`, `geoseries`, `timeseries`, `wmts`
- released: `2023-11-30T11:00:00.000Z`
- arco updated: `2026-06-23T01:56:52.376Z`

| Variable | Standard name | Units | BBox | Coordinates |
| --- | --- | --- | --- | --- |
| `VCMX` | `sea_surface_wave_maximum_height` | `m` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VHM0` | `sea_surface_wave_significant_height` | `m` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VHM0_SW1` | `sea_surface_primary_swell_wave_significant_height` | `m` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VHM0_SW2` | `sea_surface_secondary_swell_wave_significant_height` | `m` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VHM0_WW` | `sea_surface_wind_wave_significant_height` | `m` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VMDR` | `sea_surface_wave_from_direction` | `degree` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VMDR_SW1` | `sea_surface_primary_swell_wave_from_direction` | `degree` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VMDR_SW2` | `sea_surface_secondary_swell_wave_from_direction` | `degree` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VMDR_WW` | `sea_surface_wind_wave_from_direction` | `degree` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VMXL` | `sea_surface_wave_maximum_crest_height` | `m` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VPED` | `sea_surface_wave_from_direction_at_variance_spectral_density_maximum` | `degree` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VSDX` | `sea_surface_wave_stokes_drift_x_velocity` | `m/s` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VSDY` | `sea_surface_wave_stokes_drift_y_velocity` | `m/s` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VTM01_SW1` | `sea_surface_primary_swell_wave_mean_period` | `s` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VTM01_SW2` | `sea_surface_secondary_swell_wave_mean_period` | `s` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VTM01_WW` | `sea_surface_wind_wave_mean_period` | `s` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VTM02` | `sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment` | `s` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VTM10` | `sea_surface_wave_mean_period_from_variance_spectral_density_inverse_frequency_moment` | `s` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |
| `VTPK` | `sea_surface_wave_period_at_variance_spectral_density_maximum` | `s` | `[-18.125, 30.1875, 36.29166793823242, 45.97916793823242]` | time: step=3600000.0, latitude: step=0.04166667002172143, longitude: step=0.04166666764030048 |

### `cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H`

- product: `WIND_GLO_PHY_L4_NRT_012_004`
- version: `202207`
- services: `files`, `geoseries`, `timeseries`, `wmts`
- released: `None`
- arco updated: `2026-06-22T14:41:48.110Z`

| Variable | Standard name | Units | BBox | Coordinates |
| --- | --- | --- | --- | --- |
| `air_density` | `air_density` | `kg m-3` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `eastward_stress` | `surface_downward_eastward_stress` | `N m-2` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `eastward_stress_bias` | `surface_downward_eastward_stress_bias` | `N m-2` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `eastward_stress_sdd` | `surface_downward_eastward_stress_standard_deviation_of_differences` | `N m-2` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `eastward_wind` | `eastward_wind` | `m s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `eastward_wind_bias` | `eastward_wind_bias` | `m s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `eastward_wind_sdd` | `eastward_wind_standard_deviation_of_differences` | `m s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `northward_stress` | `surface_downward_northward_stress` | `N m-2` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `northward_stress_bias` | `surface_downward_northward_stress_bias` | `N m-2` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `northward_stress_sdd` | `surface_downward_northward_stress_standard_deviation_of_differences` | `N m-2` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `northward_wind` | `northward_wind` | `m s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `northward_wind_bias` | `northward_wind_bias` | `m s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `northward_wind_sdd` | `northward_wind_standard_deviation_of_differences` | `m s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `number_of_observations` | `number_of_observations` | `1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `number_of_observations_divcurl` | `number_of_observations` | `1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `stress_curl` | `vertical_component_of_surface_downward_stress_curl` | `N m-3` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `stress_curl_bias` | `vertical_component_of_surface_downward_stress_curl_bias` | `N m-3` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `stress_curl_dv` | `vertical_component_of_surface_downward_stress_curl_difference_of_variances` | `N2 m-6` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `stress_divergence` | `divergence_of_surface_downward_stress` | `N m-3` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `stress_divergence_bias` | `divergence_of_surface_downward_stress_bias` | `N m-3` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `stress_divergence_dv` | `divergence_of_surface_downward_stress_difference_of_variances` | `N2 m-6` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `wind_curl` | `atmosphere_relative_vorticity` | `s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `wind_curl_bias` | `atmosphere_relative_vorticity_bias` | `s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `wind_curl_dv` | `atmosphere_relative_vorticity_difference_of_variances` | `s-2` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `wind_divergence` | `divergence_of_wind` | `s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `wind_divergence_bias` | `divergence_of_wind_bias` | `s-1` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |
| `wind_divergence_dv` | `divergence_of_wind_difference_of_variances` | `s-2` | `[-179.9375, -89.9375, 179.9375, 89.9375]` | time: step=3600000.0, latitude: step=0.125, longitude: step=0.125 |

### `cmems_obs-wind_med_phy_nrt_l3-s1a-sar-asc-0.01deg_P1D-i`

- product: `WIND_MED_PHY_HR_L3_NRT_012_104`
- version: `202506`
- services: `files`, `geoseries`, `timeseries`, `wmts`
- released: `2025-06-25T13:00:00.000Z`
- arco updated: `2026-06-23T07:58:09.924Z`

| Variable | Standard name | Units | BBox | Coordinates |
| --- | --- | --- | --- | --- |
| `eastward_model_wind` | `eastward_wind` | `m s-1` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `eastward_wind` | `eastward_wind` | `m s-1` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `measurement_time` | `time` | `seconds since 1990-01-01` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `model_wind_speed` | `wind_speed` | `m s-1` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `model_wind_to_dir` | `wind_to_direction` | `degree` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `northward_model_wind` | `northward_wind` | `m s-1` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `northward_wind` | `northward_wind` | `m s-1` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `number_of_pixel` | `wind_speed_number_of_observation` | `` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `product_info` | `status_flag` | `` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `quality_level` | `aggregate_quality_flag` | `` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `rejection_flag` | `aggregate_quality_flag` | `` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `wind_speed` | `wind_speed` | `m s-1` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
| `wind_to_dir` | `wind_to_direction` | `degree` | `[-5.604000091552734, 30.125999450683594, 36.80400085449219, 46.00199890136719]` | time: step=86400000.0, latitude: step=0.011999999584794856, longitude: step=0.012000000267698053 |
