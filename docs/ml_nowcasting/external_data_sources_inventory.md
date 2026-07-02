# External Data Sources Inventory

Date d'analyse : 2026-06-23.

Objectif : identifier les sources les plus fiables et les plus pertinentes pour
ameliorer les previsions de vent sur les spots corses, avec un focus particulier
sur les journees thermiques.

## Synthese priorisee

| Priorite | Source | Statut local | Interet thermique | Decision |
| --- | --- | --- | --- | --- |
| P0 | Observations Beacon Live | OK | tres fort | source cible et contexte local |
| P0 | Meteo-France observations 6 min / horaire | OK API | tres fort | integrer en premier |
| P0 | AROME / AROME-PI WCS | OK API | tres fort | etendre au-dela du vent |
| P1 | Rayonnement/ensoleillement Meteo-France | OK si station disponible | tres fort | verifier couverture par station |
| P1 | Pression/tendance/gradients SYNOP | OK API | fort | integrer rapidement |
| P1 | Copernicus Marine SST | OK credentials testes | tres fort | SST integree, autres champs a tester |
| P1 | Topographie IGN RGE ALTI / API alti | ouvert | fort | integrer features statiques |
| P2 | EUMETSAT cloud mask/type | OK credentials testes | tres fort | Cloud Mask MTG collecte, etendre aux produits P1/P2 |
| P2 | Radar Meteo-France | OK avec token radar | moyen/fort perturbe | integrer comme contexte pluie |
| P2 | ECMWF Open Data | pas de cle requise | moyen | contexte synoptique large |
| P3 | ERA5 / CDS | acces a creer | backtest fort | utile pour historique long |

## 1. Observations locales Beacon Live

Statut : deja accessible localement.

Source projet :

```text
/Users/arnaud/Documents/beacon-live-app
```

Donnees deja importees :

- 25 points/stations ;
- 20 points utilisables ML ;
- resolutions : 6 min, 15 min, 1 h ;
- sources : WindsUp, Meteo-France, Wunderground, eSurfmar, CANDHIS, OWM.

Interet :

- cible principale du modele : vent moyen et rafales observees ;
- contexte temps reel proche des spots ;
- possibilite de construire des signaux de declenchement thermique :
  acceleration du vent, rotation de direction, retard entre spots, coherence
  entre spots voisins.

Manques :

- qualite variable selon source ;
- pas toujours temperature/pression/rayonnement ;
- historique a consolider hors snapshot live.

Decision :

```text
source cible principale
```

## 2. Meteo-France observations in situ

Statut : API testee OK avec la cle locale Meteo-France.

Documentation officielle :

```text
https://confluence-meteofrance.atlassian.net/wiki/spaces/OpenDataMeteoFrance/overview
https://donneespubliques.meteofrance.fr/
```

Endpoints identifies :

```text
DPPaquetObs v2: https://public-api.meteofrance.fr/public/DPPaquetObs/v2
DPObs v2:       https://public-api.meteofrance.fr/public/DPObs/v2
```

Donnees disponibles :

- vent moyen `ff` ;
- direction `dd` ;
- rafales `raf10`, `raf`, `rafper` ;
- temperature `t`, point de rosee `td`, humidite `u` ;
- pression station `pres`, pression mer `pmer` ;
- precipitation `rr_per`, `rr1`, `rr3`, `rr6`, `rr12`, `rr24` ;
- visibilite `vv` ;
- nebulosite `n` et temps present via SYNOP ;
- ensoleillement `insolh` ;
- rayonnement global `ray_glo01` ;
- temperatures sol `t_10`, `t_20`, `t_50`, `t_100` quand disponibles.

Granularite :

- infra-horaire 6 min ;
- horaire ;
- SYNOP enrichi ;
- bouees.

Interet thermique :

- tres fort pour temperature, pression, humidite, rayonnement ;
- permet de construire les gradients cote/interieur ;
- permet de verifier la montee thermique reelle ;
- permet de detecter les contextes non thermiques.

Acces requis :

```text
METEOFRANCE_API_KEY
```

Decision :

```text
integrer en P0/P1
```

## 3. AROME / AROME-PI Meteo-France WCS

Statut : API testee OK avec la cle locale Meteo-France.

Endpoints deja utilises :

```text
https://public-api.meteofrance.fr/public/arome/1.0
https://public-api.meteofrance.fr/public/aromepi/1.0
```

Services WCS :

```text
MF-NWP-HIGHRES-AROME-001-FRANCE-WCS
MF-NWP-HIGHRES-AROME-0025-FRANCE-WCS
MF-NWP-HIGHRES-AROMEPI-001-FRANCE-WCS
MF-NWP-HIGHRES-AROMEPI-0025-FRANCE-WCS
```

Inventaire capabilities observe le 2026-06-23 :

| Service | Coverages | Donnees thermiques utiles observees |
| --- | ---: | --- |
| AROME 0.01 | 5909 | vent, rafales, pression, humidite, pluie, nuages, rayonnement, PBL height |
| AROME 0.025 | 8640 | vent, rafales, pression, humidite, pluie, nuages, rayonnement, PBL height |
| AROME-PI 0.01 | 6867 | vent, rafales 15 min, pression, humidite, pluie, low cloud |
| AROME-PI 0.025 | 10682 | vent, rafales 15 min, pression, humidite, pluie, nuages/fog, rayonnement |

Variables observees importantes :

```text
WIND_SPEED
U_COMPONENT_OF_WIND
V_COMPONENT_OF_WIND
WIND_SPEED_GUST / WIND_SPEED_GUST_15MIN
PRESSURE
RELATIVE_HUMIDITY
TOTAL_WATER_PRECIPITATION
LOW_CLOUD_COVER
SHORT_WAVE_RADIATION_FLUX_CLEAR_SKY
NET_SHORT_WAVE_RADIATION_CLEAR_SKY
PLANETARY_BOUNDARY_LAYER_HEIGHT
```

Interet thermique :

- AROME/AROME-PI restent la base de forecast ;
- les champs rayonnement, cloud, pression et PBL sont plus importants que
  prevu pour notre cas ;
- il faut extraire ces variables aux spots, pas seulement vent/rafales.

Acces requis :

```text
METEOFRANCE_API_KEY
```

Libs/format :

- WCS ;
- GeoTIFF ou GRIB ;
- besoin de `rasterio`/GDAL pour extraction robuste.

Decision :

```text
P0 : etendre le sampler modele a temperature/pression/humidite/rayonnement/cloud/PBL
```

## 4. Radar Meteo-France

Statut : OK avec token Meteo-France disposant de l'abonnement radar.

Documentation officielle :

```text
https://www.data.gouv.fr/dataservices/api-donnees-radar
```

Endpoint :

```text
https://public-api.meteofrance.fr/public/DPPaquetRadar/v1
```

Donnees disponibles :

- mosaique precipitation 5 min ;
- HDF5 `IPRN20` ;
- grille 500 m ;
- `ACRR` : precipitation accumulee ;
- `QIND` : qualite ;
- radars individuels Ajaccio `37` et Aleria `61` en BUFR.

Interet thermique :

- moyen en thermique pur ;
- fort pour separer les journees perturbees ;
- utile pour averses, grains, fronts, rafales descendantes, convection.

Acces requis :

```text
METEOFRANCE_API_KEY avec abonnement DPPaquetRadar
```

Libs/format :

- `gzip` + `tar` ;
- HDF5 : besoin de `h5py` ;
- projection : besoin de `pyproj`.

Decision :

```text
P2 : contexte pluie/convection, pas source principale thermique
```

## 5. Copernicus Marine SST et donnees oceaniques

Statut local :

- credentials testes OK le 2026-06-23 ;
- pas de credentials persistants dans `.env` ;
- collecteur SST implemente dans
  `scripts/ml_dataset/collect_copernicus_marine_sst.py` ;
- inventaire produits implemente dans
  `scripts/ml_dataset/inventory_copernicus_marine_products.py`.

Documentation officielle :

```text
https://data.marine.copernicus.eu/products
https://toolbox-docs.marine.copernicus.eu/
```

Produit integre :

```text
SST_MED_PHY_SUBSKIN_L4_NRT_010_036
dataset: cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m
variable: analysed_sst
```

Donnees disponibles :

- temperature de surface mer ;
- produit subskin horaire Mediterranee a 0.0625 deg ;
- format NetCDF via `geoseries` ;
- sample spot produit sous
  `data/processed/ml_dataset/copernicus_marine/sst_samples`.

Autres produits accessibles identifies :

| Priorite | Famille | Dataset | Variables | Decision |
| --- | --- | --- | --- | --- |
| P2 | courants surface | `cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m` | `uo`, `vo` | tester apres SST |
| P2 | mixed layer | `cmems_mod_med_phy-mld_anfc_4.2km-2D_PT1H-m` | `mlotst` | tester apres SST |
| P2 | vagues | `cmems_mod_med_wav_anfc_4.2km_PT1H-i` | `VHM0`, `VMDR`, `VTPK` | contexte mer/qualite obs |
| P3 | vent mer L4 global | `cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H` | `eastward_wind`, `northward_wind` | backtest/check large echelle |
| P3 | vent SAR Mediterranee | `cmems_obs-wind_med_phy_nrt_l3-s1a-sar-asc-0.01deg_P1D-i` | `wind_speed`, `wind_to_dir` | episodique, audit/backtest |

Interet thermique :

- tres fort ;
- permet le contraste terre/mer ;
- aide a distinguer journee a fort potentiel thermique et journee mer trop
  chaude/froide ;
- variable quasi indispensable pour le moteur thermique.

Acces teste, a configurer en production :

```text
COPERNICUSMARINE_SERVICE_USERNAME
COPERNICUSMARINE_SERVICE_PASSWORD
```

Lib a installer :

```text
copernicusmarine
xarray
netCDF4 ou zarr
```

Decision :

```text
P1 : SST integree maintenant ; autres champs oceaniques a tester apres validation du gain
```

## 6. EUMETSAT cloud mask / cloud type

Statut local :

- catalogue public teste OK le 2026-06-23 ;
- credentials de telechargement testes OK le 2026-06-23 ;
- credentials non persistants dans `.env` ;
- inventaire produits implemente dans
  `scripts/ml_dataset/inventory_eumetsat_products.py` ;
- scan catalogue global implemente dans
  `scripts/ml_dataset/inventory_eumetsat_catalog_keywords.py` ;
- collecteur Cloud Mask MTG implemente dans
  `scripts/ml_dataset/collect_eumetsat_cloud_mask.py` ;
- collecteur generique Cloud Type / Land Surface Temperature /
  Global Instability Indices implemente dans
  `scripts/ml_dataset/collect_eumetsat_spot_product.py` ;
- doc generee :
  `docs/ml_nowcasting/eumetsat_product_inventory.md` ;
- scan global genere :
  `docs/ml_nowcasting/eumetsat_catalog_keyword_inventory.md`.

Documentation officielle :

```text
https://user.eumetsat.int/data-access/data-store
https://user.eumetsat.int/resources/user-guides/eumetsat-data-access-client-eumdac-guide
https://user.eumetsat.int/resources/user-guides/mtg-fci-l2-clm-and-cla-data-guide
```

Produits candidats identifies :

| Priorite | Collection | Produit | Variables/features visees | Decision |
| --- | --- | --- | --- | --- |
| P1 | `EO:EUM:DAT:0678` | Cloud Mask netCDF MTG | `cloud_fraction_satellite`, `clear_sky_fraction` | premier prototype |
| P1 | `EO:EUM:DAT:0680` | Cloud Type MTG | type nuage dominant, low/high cloud fraction | collecteur integre |
| P2 | `EO:EUM:DAT:0681` | Cloud Top Temperature/Height | hauteur/sommet nuage | convection/perturbations |
| P2 | `EO:EUM:DAT:0684` | Optimal Cloud Analysis | phase, optical thickness | ablation tests |
| P2 | `EO:EUM:DAT:1088` | Land Surface Temperature MTG | temperature surface terre | collecteur integre |
| P2 | `EO:EUM:DAT:1086` | Precipitation rate MTG | pluie satellite/convection | perturbations |
| P2 | `EO:EUM:DAT:0683` | Global Instability Indices MTG | instabilite/convection | collecteur integre |
| P2 | `EO:EUM:DAT:0691` | Lightning Flashes MTG | eclairs proches | flag risque rafales/convection |
| P3 | `EO:EUM:DAT:0676` | Atmospheric Motion Vectors MTG | vent haut/moyen niveau | contexte large |
| P3 | `EO:EUM:DAT:0694` | FCI SST MTG | temperature mer satellite | duplicat/check Copernicus |
| P3 | `EO:EUM:DAT:0863` | SARAH-3 surface radiation | rayonnement historique | backtests |
| P3 | `EO:EUM:DAT:MSG:CLM` | MSG Cloud Mask | cloud fraction legacy | fallback long historique |

Donnees disponibles :

- pixel clair/nuageux ;
- extraction testee sur 6 produits MTG NRT, 25 spots, 150 lignes ;
- cadence observee : produit toutes les 10 minutes environ ;
- type de nuage ;
- hauteur/sommet de nuage ;
- qualite ;
- temperature radiative de surface terrestre avec `EO:EUM:DAT:1088` ;
- precipitation satellite avec `EO:EUM:DAT:1086` ;
- tres utile pour savoir si le sol chauffe vraiment.

Interet thermique :

- tres fort ;
- source la plus directe pour qualifier ciel clair/couvert a l'echelle du spot ;
- complete `insolh` et `ray_glo01`, qui ne sont disponibles que sur certaines
  stations.

Acces teste, a configurer en production :

```text
EUMETSAT_CONSUMER_KEY
EUMETSAT_CONSUMER_SECRET
```

Lib a installer :

```text
eumdac
satpy ou xarray selon format
```

Decision :

```text
P1/P2 : Cloud Mask collecte rapide ; Cloud Type, Land Surface Temperature et GII integres en collecte thermique/convection
```

## 7. IGN topographie

Statut : acces ouvert, pas de compte necessaire pour l'API altimetrique de base.

Documentation officielle :

```text
https://geoplateforme.pages.gpf-tech.ign.fr/altimetrie/api-rest-calcul-altimetrique/api/project.html
https://www.data.gouv.fr/datasets/rge-alti-r
```

Donnees disponibles :

- altitude point ;
- profil altimetrique ;
- RGE ALTI 1 m / 5 m en telechargement ;
- MNT pour calcul de pente, exposition, masque relief, couloirs.

Interet thermique :

- fort ;
- indispensable pour apprendre les biais par spot ;
- permet de deriver :
  - distance a la cote ;
  - orientation baie ;
  - relief amont ;
  - pente et exposition ;
  - ouverture aux secteurs thermiques.

Acces requis :

```text
aucun pour API publique de base
```

Limites :

- API alti utile pour points/profils ;
- pour features avancees, telecharger les tuiles RGE ALTI Corse est preferable.

Decision :

```text
P1 : integrer features statiques spot
```

## 8. ECMWF Open Data

Statut local : pas de lib installee ; pas de cle necessaire pour open data.

Documentation officielle :

```text
https://www.ecmwf.int/en/forecasts/datasets/open-data
https://data.ecmwf.int/
```

Donnees disponibles :

- IFS/AIFS open data ;
- GRIB2 ;
- resolution open data typiquement 0.25 deg ;
- rolling archive recente ;
- contexte synoptique large.

Interet thermique :

- moyen a fort pour contexte large ;
- pas assez fin pour corriger directement une baie ;
- utile pour vent synoptique, pression large echelle, gradients regionaux,
  regimes de masse d'air.

Acces requis :

```text
aucun pour open data
```

Lib a installer :

```text
ecmwf-opendata
cfgrib
eccodes
```

Decision :

```text
P2 : ajouter contexte synoptique si AROME ne suffit pas
```

## 9. ERA5 / Copernicus Climate Data Store

Statut local : pas de credentials trouves dans `.env`.

Documentation officielle :

```text
https://cds.climate.copernicus.eu/
```

Donnees disponibles :

- reanalyse horaire ;
- historique long ;
- temperature, pression, vent, rayonnement, nuages, precipitation ;
- utile pour entrainement historique ou typologie de regimes.

Interet thermique :

- faible pour nowcast temps reel ;
- fort pour backtesting et climatologie ;
- resolution trop grossiere pour spot, mais utile pour etiqueter les regimes
  synoptiques.

Acces a creer :

```text
Compte CDS
CDSAPI_URL
CDSAPI_KEY
```

Lib a installer :

```text
cdsapi
xarray
netCDF4
```

Decision :

```text
P3 : utile plus tard pour historique long
```

## Variables prioritaires a produire

### Thermique

```text
temp_coast_c
temp_inland_c
temp_inland_minus_coast_c
temp_rise_since_08_local_c
temp_rise_1h_c
sst_nearest_c
land_minus_sea_temp_c
pressure_msl_hpa
pressure_tendency_1h_hpa
pressure_gradient_corse_hpa
solar_observed_jm2
sunshine_minutes
cloud_fraction_satellite
low_cloud_cover_nwp
pbl_height_m
synoptic_wind_10m_ms
synoptic_wind_925_850_ms_if_available
```

### Vent cible et nowcast

```text
wind_obs_ms
gust_obs_ms
wind_dir_obs_deg
wind_obs_trend_30min_ms
gust_max_30min_ms
wind_dir_rotation_30min_deg
neighbor_spot_wind_delta_ms
nwp_wind_bias_latest_ms
nwp_dir_bias_latest_deg
```

### Perturbations

```text
radar_precip_5min_mm
radar_precip_15min_mm
radar_rain_detected_10km
radar_distance_to_rain_km
radar_upwind_rain_detected
synop_weather_code
visibility_m
```

## Acces et configuration

1. Copernicus Marine
   - statut : credentials testes OK, non persistants dans le repo ;
   - objectif : SST Mediterranee puis courants/mixed layer/vagues en tests ;
   - variables `.env` : `COPERNICUSMARINE_SERVICE_USERNAME`,
     `COPERNICUSMARINE_SERVICE_PASSWORD`.

2. EUMETSAT
   - statut : consumer key/secret testes OK, non persistants dans le repo ;
   - objectif : Cloud Mask MTG collecte, puis cloud type et land surface
     temperature ;
   - variables `.env` : `EUMETSAT_CONSUMER_KEY`,
     `EUMETSAT_CONSUMER_SECRET`.

3. CDS/ERA5, plus tard
   - objectif : historique long/backtests ;
   - variables `.env` : `CDSAPI_URL`, `CDSAPI_KEY`.

## Libs techniques a ajouter

Pour l'extraction geospatiale et satellite :

```text
h5py
pyproj
rasterio
xarray
netCDF4
zarr
copernicusmarine
eumdac
ecmwf-opendata
cfgrib
eccodes
cdsapi
```

On n'a pas besoin de tout installer tout de suite. Premier lot recommande :

```text
h5py
pyproj
rasterio
xarray
netCDF4
copernicusmarine
eumdac
```

## Decision operationnelle

Le chemin le plus rentable est :

1. etendre Meteo-France observations + AROME/AROME-PI aux variables thermiques ;
2. construire gradients cote/interieur et pression Corse ;
3. ajouter SST Copernicus Marine ;
4. ajouter EUMETSAT Cloud Mask puis Cloud Type / Land Surface Temperature ;
5. ajouter topographie IGN statique ;
6. ajouter radar pluie/convection ;
7. ajouter ECMWF/ERA5 pour contexte large et historique.

Pour les vraies journees thermiques, le plus critique est :

```text
rayonnement reel + ciel clair + contraste terre/mer + gradient pression local
+ vent synoptique compatible + historique vent spot
```
