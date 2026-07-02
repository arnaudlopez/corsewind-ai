# EUMETSAT Product Inventory

Generated at: `2026-06-23T15:20:28.135789Z`

## Decision Summary

| Priority | Decision | Feature | Collection | Target features | Notes |
| --- | --- | --- | --- | --- | --- |
| `P1` | `integrate_after_access` | `cloud_mask` | `EO:EUM:DAT:0678` | `cloud_fraction_satellite`, `clear_sky_fraction`, `dust_or_ash_flag` | Best first satellite feature for thermal days: tells whether the ground can actually heat. |
| `P1` | `integrated_spot_sampler` | `cloud_type` | `EO:EUM:DAT:0680` | `cloud_type_dominant`, `low_cloud_fraction`, `high_cloud_fraction` | Distinguishes low marine cloud, high cloud, and convective/cloud regimes. |
| `P2` | `test_after_cloud_mask` | `cloud_top` | `EO:EUM:DAT:0681` | `cloud_top_height_m`, `cloud_top_temperature_c` | Useful for convection and cloud vertical development, less direct for pure thermal sea breeze. |
| `P2` | `test_after_cloud_mask` | `optimal_cloud_analysis` | `EO:EUM:DAT:0684` | `cloud_phase`, `cloud_optical_thickness`, `cloud_effective_radius` | Richer cloud microphysics; probably too much for V1 but valuable for ablation tests. |
| `P2` | `integrated_spot_sampler` | `land_surface_temperature` | `EO:EUM:DAT:1088` | `land_surface_temperature_c`, `land_minus_sea_surface_temperature_c` | Potentially excellent proxy for actual ground heating, complementing air temperature stations. |
| `P2` | `test_for_convection` | `precipitation_rate` | `EO:EUM:DAT:1086` | `satellite_precip_rate_mm_h`, `convective_precip_flag` | Helps exclude disturbed/convection days where thermal wind behaves differently. |
| `P2` | `integrated_spot_sampler` | `global_instability_indices` | `EO:EUM:DAT:0683` | `satellite_instability_index`, `convective_potential_flag` | Potentially useful to separate clean thermal days from unstable convective regimes. |
| `P2` | `test_for_convection` | `lightning` | `EO:EUM:DAT:0691` | `lightning_flash_count_nearby`, `lightning_detected_radius` | Useful disturbed-day flag; not a normal thermal driver but important for exclusion and gust risk. |
| `P3` | `context_or_backtest` | `atmospheric_motion_vectors` | `EO:EUM:DAT:0676` | `upper_cloud_motion_wind`, `midlevel_flow_context` | Large-scale/aloft wind context; probably weaker than NWP wind fields for V1. |
| `P3` | `duplicate_check` | `mtg_sea_surface_temperature` | `EO:EUM:DAT:0694` | `mtg_sst_c` | Could cross-check Copernicus SST, but Copernicus remains the cleaner SST source for V1. |
| `P3` | `historical_or_backtest` | `surface_radiation` | `EO:EUM:DAT:0863` | `surface_solar_radiation`, `daily_solar_energy` | Very useful for historical solar context, but less likely to be a low-latency nowcast feed. |
| `P3` | `fallback_legacy` | `msg_cloud_mask` | `EO:EUM:DAT:MSG:CLM` | `cloud_fraction_satellite_legacy` | MSG/SEVIRI fallback with long archive and 15 min cadence if MTG is awkward to process. |

## Access Model

- Public catalogue metadata is available through `https://api.eumetsat.int/data/browse/1.0.0/collections/<collection>?format=json`.
- Product download requires an EUMETSAT account plus `EUMETSAT_CONSUMER_KEY` and `EUMETSAT_CONSUMER_SECRET`.
- Operational download/prototyping should use `eumdac`; spatial tailoring will likely need Data Tailor for MTG products.

## Dataset Details

### `EO:EUM:DAT:0678`

- title: `Cloud Mask (netCDF) - MTG - 0 degree`
- date: `2025-01-27/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0678

The central aim of the cloud mask (CLM) product is to identify cloudy and cloud free FCI Level 1c pixels with high confidence. The product also provides information on the presence of snow/sea ice, volcanic ash and dust...

### `EO:EUM:DAT:0680`

- title: `Cloud Type - MTG - 0 degree`
- date: `2025-12-10/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0680

The Cloud Type (CT) product provides a detailed cloud analysis for all pixels identified as cloudy in a scene. It contains information about the classification of the cloudy pixels, discriminating between low-level, med...

### `EO:EUM:DAT:0681`

- title: `Cloud Top Temperature and Height - MTG - 0 degree`
- date: `2025-12-10/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0681

The Cloud Top Temperature and Height product contains cloud-top temperature and height (expressed as both geometric height and pressure) for all pixels identified as cloudy. This encompasses opaque and semi-transparent...

### `EO:EUM:DAT:0684`

- title: `Optimal Cloud Analysis - MTG - 0 degree`
- date: `2025-01-27/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0684

The Optimal Cloud Analysis (OCA) product uses an optimal estimation retrieval scheme to retrieve cloud properties (phase, height and microphysical properties) from visible, near-infrared and thermal infrared FCI channel...

### `EO:EUM:DAT:1088`

- title: `Land Surface Temperature - MTG`
- date: `2025-08-14/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A1088

Land Surface Temperature (LST) refers to the radiative skin temperature over land. It plays a key role in the physics of the land surface, as it is involved in the exchange of energy and water with the atmosphere. Accur...

### `EO:EUM:DAT:1086`

- title: `Precipitation rate at ground by blended FCI IR / LEO MW precipitation - MTG - 0 Degree`
- date: `2025-08-14/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A1086

An instantaneous precipitation map is generated by IR images from the MTG FCI “calibrated” by precipitation estimates from MW radiometers on board LEO satellites, processed soon after each acquisition of a new image fro...

### `EO:EUM:DAT:0683`

- title: `Global Instability Indices - MTG - 0 degree`
- date: `2025-01-27/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0683

The Global Instability Index (GII) product provides information about instability of the atmosphere and thus can identify regions of convective potential. GII is a segmented product that uses an optimal estimation schem...

### `EO:EUM:DAT:0691`

- title: `LI Lightning Flashes - MTG - 0 degree`
- date: `2024-07-04/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0691

LI Level 2 Lightning Flashes (LFL) contains LI flashes. The definition of a flash is shared by LI and GLM; collections of groups that are correlated in space and time within the two windows of 330 milliseconds (temporal...

### `EO:EUM:DAT:0676`

- title: `Atmospheric Motion Vectors (netCDF) - MTG - 0 degree`
- date: `2025-01-27/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0676

The Atmospheric Motion Vector (AMV) product is realised by tracking clouds or water vapour features in consecutive FCI satellite images based on feature tracking between each pair of consecutive repeat cycles, leading t...

### `EO:EUM:DAT:0694`

- title: `FCI Level 3 Sea Surface Temperature - MTG`
- date: `2025-08-20/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0694

Level 3 hourly sub-skin Sea Surface Temperature derived from Meteosat at 0° longitude, covering 60S-60N and 60W-60E and re-projected on a 0.05° regular grid, in GHRSST compliant netCDF format.

### `EO:EUM:DAT:0863`

- title: `Surface Radiation Data Set - Heliosat (SARAH) - Edition 3`
- date: `1983-01-01/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3A0863

The third edition of the Surface Solar Radiation Data Set - Heliosat (SARAH-3) is a satellite-based climate data record of the solar surface irradiance (SIS), the surface direct irradiance ((direct horizontal and direct...

### `EO:EUM:DAT:MSG:CLM`

- title: `Cloud Mask - MSG - 0 degree`
- date: `2004-01-29/`
- rights: `NoConditions`
- catalogue: https://data.eumetsat.int/product/EO%3AEUM%3ADAT%3AMSG%3ACLM

The Cloud Mask product describes the scene type (either 'clear' or 'cloudy') on a pixel level. Each pixel is classified as one of the following four types: clear sky over water, clear sky over land, cloud, or not proces...
