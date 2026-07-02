# Missing Access API Call Structures

Date : 2026-06-23.

Objectif : preparer les appels API pour les sources utiles mais pas encore
configurees durablement localement.

Statut local actuel :

```text
COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD : testes OK, non persistants
EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET     : testes OK, non persistants
CDSAPI_URL / CDSAPI_KEY                              : manquants
```

## 1. Copernicus Marine SST

Documentation officielle :

```text
https://toolbox-docs.marine.copernicus.eu/
https://toolbox-docs.marine.copernicus.eu/en/stable/usage/subset-usage.html
```

But pour CorseWind :

- recuperer la temperature de surface mer autour de la Corse ;
- construire `sst_nearest_c` ;
- construire `land_minus_sea_temp_c`.
- alimenter le dataset via `scripts/ml_dataset/collect_copernicus_marine_sst.py`.

Produits candidats :

```text
SST_MED_PHY_SUBSKIN_L4_NRT_010_036
SST_MED_SST_L4_NRT_OBSERVATIONS_010_004
```

Test reel effectue le 2026-06-23 :

```text
Credentials Copernicus Marine valides.
Product ID: SST_MED_PHY_SUBSKIN_L4_NRT_010_036
Dataset ID: cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m
```

Structure observee via `describe` :

```text
variable: analysed_sst
standard_name: sea_surface_subskin_temperature
units: kelvin
temporal step: 1 h
latitude step: 0.0625 deg
longitude step: 0.0625 deg
bbox produit: [-18.125, 30.25, 36.25, 46.0]
time coverage observee: 2019-01-01T00:00:00Z -> 2026-06-22T23:00:00Z
services utiles: arco-geo-series / geoseries, arco-time-series / timeseries
```

Acces teste, a configurer en production :

```text
COPERNICUSMARINE_SERVICE_USERNAME
COPERNICUSMARINE_SERVICE_PASSWORD
```

Installation probable :

```bash
pip install copernicusmarine xarray netCDF4
```

Structure CLI :

```bash
copernicusmarine subset \
  --dataset-id cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m \
  --variable analysed_sst \
  --minimum-longitude 7.5 \
  --maximum-longitude 10.2 \
  --minimum-latitude 41.0 \
  --maximum-latitude 43.3 \
  --start-datetime 2026-06-22T12:00:00 \
  --end-datetime 2026-06-22T15:00:00 \
  --output-directory data/raw/copernicus_marine/sst \
  --output-filename sst_corse_20260622_12_15.nc \
  --file-format netcdf \
  --service geoseries
```

Structure Python probable :

```python
import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_obs-sst_med_phy-sst_nrt_diurnal-oi-0.0625deg_PT1H-m",
    variables=["analysed_sst"],
    minimum_longitude=7.5,
    maximum_longitude=10.2,
    minimum_latitude=41.0,
    maximum_latitude=43.3,
    start_datetime="2026-06-22T12:00:00",
    end_datetime="2026-06-22T15:00:00",
    output_directory="data/raw/copernicus_marine/sst",
    output_filename="sst_corse_20260622_12_15.nc",
    file_format="netcdf",
    service="geoseries",
)
```

Mini subset telecharge et inspecte :

```text
file: tmp/copernicusmarine_sst_subset/sst_corse_20260622_12_15.nc
size: 40084 bytes
dims:
  time: 4
  latitude: 37
  longitude: 44
coords:
  time: 2026-06-22T12:00, 13:00, 14:00, 15:00 UTC
  latitude: 41.0 -> 43.25
  longitude: 7.5 -> 10.1875
data_vars:
  analysed_sst(time, latitude, longitude), float64, kelvin
range observed on subset:
  297.79 K -> 302.16 K
  24.64 C -> 29.01 C
```

Collecte integree le 2026-06-23 :

```text
script: scripts/ml_dataset/collect_copernicus_marine_sst.py
fenetre test: 2026-06-22T08:00:00Z -> 2026-06-22T17:00:00Z
spots: 25
rows: 250
valid_sst_rows: 250
range sampled: 24.51 C -> 27.74 C
output raw: data/processed/ml_dataset/copernicus_marine/raw/sst/
output samples: data/processed/ml_dataset/copernicus_marine/sst_samples/
```

Inventaire produits associe :

```text
script: scripts/ml_dataset/inventory_copernicus_marine_products.py
doc: docs/ml_nowcasting/copernicus_marine_product_inventory.md
```

Points a confirmer avant production :

- strategie operationnelle : telecharger par jour complet ou par fenetre courte ;
- choix `geoseries` vs `timeseries` selon usage ;
- latence operationnelle exacte en production ;
- extraction spot : nearest pixel ou moyenne rayon autour du spot ;
- conservation long terme : raw NetCDF ou samples par spot seulement.
- rotation du mot de passe partage en clair avant stockage en environnement.

## 2. EUMETSAT Cloud Mask / Cloud Type

Documentation officielle :

```text
https://user.eumetsat.int/data-access/data-store
https://user.eumetsat.int/resources/user-guides/eumetsat-data-access-client-eumdac-guide
https://user.eumetsat.int/resources/user-guides/mtg-data-access-guide
```

But pour CorseWind :

- recuperer cloud mask / cloud type autour de la Corse ;
- construire `cloud_fraction_satellite` ;
- construire `cloud_type_dominant` ;
- eventuellement `cloud_top_height` ou `cloud_top_temperature`.

Produits candidats :

```text
EO:EUM:DAT:0678      Cloud Mask (netCDF) - MTG - 0 degree
EO:EUM:DAT:0680      Cloud Type - MTG - 0 degree
EO:EUM:DAT:0681      Cloud Top Temperature and Height - MTG - 0 degree
EO:EUM:DAT:0684      Optimal Cloud Analysis - MTG - 0 degree
EO:EUM:DAT:1088      Land Surface Temperature - MTG
EO:EUM:DAT:1086      Precipitation rate at ground by blended FCI IR / LEO MW
EO:EUM:DAT:0683      Global Instability Indices - MTG - 0 degree
EO:EUM:DAT:0691      LI Lightning Flashes - MTG - 0 degree
EO:EUM:DAT:0676      Atmospheric Motion Vectors (netCDF) - MTG - 0 degree
EO:EUM:DAT:0694      FCI Level 3 Sea Surface Temperature - MTG
EO:EUM:DAT:0863      SARAH-3 Surface Radiation Data Set
EO:EUM:DAT:MSG:CLM   Cloud Mask - MSG - 0 degree, fallback historique
```

Inventaire public effectue le 2026-06-23 :

```text
script: scripts/ml_dataset/inventory_eumetsat_products.py
doc: docs/ml_nowcasting/eumetsat_product_inventory.md
metadata API: https://api.eumetsat.int/data/browse/1.0.0/collections/<collection>?format=json
status: 12 collections trouvees, 0 erreur
```

Acces teste, a configurer en production :

```text
EUMETSAT_CONSUMER_KEY
EUMETSAT_CONSUMER_SECRET
```

Premier test acces effectue le 2026-06-23 :

```text
eumdac installe temporairement dans tmp/eumdac_test_pkgs
collection EO:EUM:DAT:0678 connue et metadata catalogue publique OK
credentials fournis comme username/password testes via eumdac.AccessToken(cache=False)
resultat API token: 401 Unauthorized
interpretation: ce ne sont probablement pas les Consumer Key / Consumer Secret requis par l'API Data Store
```

Deuxieme test acces effectue le 2026-06-23 :

```text
Consumer Key / Consumer Secret valides.
Token eumdac cree avec cache=False.
Collection testee: EO:EUM:DAT:0678
Titre collection: Cloud Mask (netCDF) - MTG - 0 degree
Recherche bbox Corse OK:
  bbox 7.5,41.0,10.2,43.3
  fenetre test: 2026-06-21T14:46Z -> 2026-06-23T13:46Z
  produits trouves: au moins 20
Produit recent teste:
  W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-2-CLM--FD--x-x---x_C_EUMT_20260623143802_L2PF_OPE_20260623142000_20260623143000_N__O_0087_0000
  sensing_start: 2026-06-23T14:20:00Z
  sensing_end: 2026-06-23T14:30:00Z
  timeliness: NRT
  quality: NOMINAL
  entree NetCDF telechargee dans tmp/eumetsat_cloudmask_test/
  taille fichier: 4.4 MB
```

Structure NetCDF observee :

```text
dims:
  number_of_rows: 5568
  number_of_columns: 5568
coords:
  x, y en projection geostationnaire MTG
projection:
  mtg_geos_projection
variables principales:
  cloud_state
  quality_illumination
  quality_nwp_parameters
  quality_MTG_parameters
  quality_overall_processing
  product_quality
  product_completeness
  product_timeliness
```

Test geolocalisation spots :

```text
conversion WGS84 -> MTG geostationary OK avec pyproj
exemples:
  lfkj pixel distance ~0.49 km
  lfkf pixel distance ~0.78 km
  cap_corse pixel distance ~0.92 km
  balistra pixel distance ~1.30 km
```

Action requise avant production :

```text
Rotater les valeurs EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET partagees
en clair, puis les stocker uniquement en environnement de production.
```

Installation probable :

```bash
pip install eumdac
```

Structure CLI attendue :

```bash
eumdac search \
  --collection EO:EUM:DAT:0678 \
  --start 2026-06-23T10:00:00Z \
  --end 2026-06-23T10:30:00Z \
  --bbox 7.5 41.0 10.2 43.3
```

Puis telechargement d'un produit :

```bash
eumdac download \
  --collection EO:EUM:DAT:0678 \
  --product <product_id> \
  --output-dir data/raw/eumetsat/cloud
```

Structure Python probable :

```python
import eumdac

credentials = (
    os.environ["EUMETSAT_CONSUMER_KEY"],
    os.environ["EUMETSAT_CONSUMER_SECRET"],
)
token = eumdac.AccessToken(credentials)
datastore = eumdac.DataStore(token)

collection = datastore.get_collection("EO:EUM:DAT:0678")
products = collection.search(
    dtstart="2026-06-23T10:00:00Z",
    dtend="2026-06-23T10:30:00Z",
    bbox=[7.5, 41.0, 10.2, 43.3],
)
```

Points a confirmer apres login :

- mapping officiel des valeurs `cloud_state` ;
- possibilite de subset spatial avant telechargement via Data Tailor ;
- latence mediane par heure de journee ;
- taille produit pour un usage operationnel sur plusieurs pas de temps ;
- lecture NetCDF/Satpy et extraction d'une fenetre Corse ;
- variables/classes exactes a mapper en `cloud_fraction_satellite`.

## 3. CDS / ERA5

Documentation officielle :

```text
https://cds.climate.copernicus.eu/how-to-api
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
```

But pour CorseWind :

- pas de nowcast operationnel ;
- construire historique long / climatologie / regimes synoptiques ;
- completer quand les observations locales historiques sont trop courtes.

Acces a creer :

```text
Compte Copernicus Climate Data Store
CDSAPI_URL
CDSAPI_KEY
```

Installation probable :

```bash
pip install cdsapi xarray netCDF4
```

Structure Python :

```python
import cdsapi

client = cdsapi.Client()
client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
            "surface_solar_radiation_downwards",
            "total_cloud_cover",
        ],
        "year": ["2025"],
        "month": ["06"],
        "day": ["01", "02"],
        "time": ["00:00", "01:00", "02:00"],
        "data_format": "netcdf",
        "area": [43.3, 7.5, 41.0, 10.2],
    },
    "data/raw/cds/era5_corse_20250601_20250602.nc",
)
```

Points a confirmer apres login :

- nouvelle syntaxe exacte selon l'interface CDS courante ;
- licences produits a accepter dans le compte ;
- quotas / temps de preparation ;
- pertinence des variables single-level vs pressure-levels.

## 4. ECMWF Open Data

Statut : pas un acces manquant strict, car l'open data ne demande pas forcement
de cle. Mais la librairie n'est pas installee localement.

Documentation officielle :

```text
https://data.ecmwf.int/
https://pypi.org/project/ecmwf-opendata/
```

But pour CorseWind :

- contexte synoptique large ;
- comparaison AROME vs IFS/AIFS ;
- pas prioritaire pour le thermique local.

Installation probable :

```bash
pip install ecmwf-opendata cfgrib eccodes
```

Structure Python :

```python
from ecmwf.opendata import Client

client = Client(source="ecmwf")
client.retrieve(
    date="20260623",
    time=0,
    step=[0, 3, 6, 9, 12],
    type="fc",
    param=["10u", "10v", "2t", "msl"],
    target="data/raw/ecmwf/open_data_20260623_00.grib2",
)
```

Points a confirmer :

- parametres exacts voulus ;
- resolution disponible ;
- methode de subset Corse pour eviter les gros fichiers ;
- valeur ajoutee par rapport a AROME/AROME-PI.

## Priorite d'acces

1. Copernicus Marine : prioritaire pour SST et contraste terre/mer.
2. EUMETSAT : prioritaire ensuite pour ciel clair / nuages reels.
3. CDS/ERA5 : utile plus tard pour historique long.
4. ECMWF Open Data : optionnel pour contexte synoptique large.
