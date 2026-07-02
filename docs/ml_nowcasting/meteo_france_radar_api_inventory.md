# Meteo-France Radar API Inventory

Date d'exploration : 2026-06-23.

Swagger source :

```text
/Users/arnaud/Downloads/Package_Radar_swagger.json
```

API :

```text
DPPaquetRadar v1: https://public-api.meteofrance.fr/public/DPPaquetRadar/v1
```

## Verdict

Les donnees radar sont exploitables pour le dataset CorseWind.ai, mais elles ne
doivent pas etre traitees comme une observation directe du vent au spot.

Elles apportent surtout du contexte meteorologique haute frequence :

- presence de pluie au spot ;
- intensite et evolution de la pluie autour du spot ;
- front pluvieux ou cellule convective proche ;
- regimes ou les observations de vent sont perturbees par averses, grains,
  rafales descendantes ou bascules de direction ;
- indicateur negatif pour les journees thermiques propres.

Pour l'objectif windsurf, le radar peut aider le modele a comprendre pourquoi
un vent observe ou prevu se casse, tourne, devient rafaleux, ou cesse de suivre
la logique thermique classique. Il ne remplace pas les observations de vent,
les modeles NWP, la pression, la temperature ou le rayonnement.

## Acces API

La cle initialement presente dans `.env` n'avait pas les droits radar :

```text
403 Resource forbidden
```

Le token API teste ensuite contient bien les abonnements suivants :

```text
DonneesPubliquesPaquetRadar /public/DPPaquetRadar/v1
DonneesPubliquesRadar       /public/DPRadar/v1
```

Les endpoints DPPaquetRadar fonctionnent avec un header :

```text
apikey: <token>
```

Attention : si un token est colle dans une conversation ou un ticket, il doit
etre considere comme expose et remplace.

## Endpoints DPPaquetRadar

| Endpoint | Statut | Interet dataset |
| --- | --- | --- |
| `GET /stations` | OK | Liste JSON des radars disponibles |
| `GET /liste-stations` | OK | Liste CSV avec produits disponibles par radar |
| `GET /mosaique/paquet` | OK | Source prioritaire : mosaigue precipitation 5 min |
| `GET /station/paquet?id_station=37` | OK | Radar Ajaccio, donnees individuelles BUFR |
| `GET /station/paquet?id_station=61` | OK | Radar Aleria, donnees individuelles BUFR |

Note importante : le Swagger indique un parametre `station`, mais l'API reelle
attend `id_station` sur `/station/paquet`.

## Radars utiles pour la Corse

| Id | Nom | Usage |
| --- | --- | --- |
| `37` | RADAR AJACCIO | radar individuel ou source de la mosaique |
| `61` | RADAR ALERIA | radar individuel ou source de la mosaique |

Dans la mosaique nationale, les sources internes confirment notamment :

```text
NOD:fraja, PLC:Ajaccio, WMO:07760
NOD:frale, PLC:Aleria,  WMO:07774
```

## Format des paquets

### `/mosaique/paquet`

Le endpoint renvoie un `application/gzip`.

Le contenu decompresse est une archive `tar`, pas un JSON direct. Un paquet
recent contenait 33 fichiers, dont :

- fichiers HDF5 de mosaique ;
- fichiers BUFR compresses ;
- trois pas de temps a 5 minutes couvrant le dernier quart d'heure.

Les fichiers HDF5 les plus utiles pour la Corse sont les `IPRN20` :

```text
T_IPRN20_C_LFPW_YYYYMMDDHHMMSS.h5
```

Caracteristiques observees :

| Champ | Valeur observee |
| --- | --- |
| convention | `ODIM_H5/V2_3` |
| produit | composite precipitation |
| grille | `3472 x 3472` |
| resolution | `500 m x 500 m` |
| projection | stereographique |
| quantite principale | `ACRR` |
| quantite qualite | `QIND` |
| pas temporel | 5 min |
| start/end | exemple `10:10:00Z -> 10:15:00Z` |

Attributs utiles du HDF5 :

```text
/dataset1/data1/data
  quantity = ACRR
  gain = 0.01
  offset = 0
  nodata = 65535
  undetect = 65534

/dataset1/data1/quality1/data
  quantity = QIND
  gain = 0.01
  offset = 0
  nodata = 255
  undetect = 254

/where
  projdef = +proj=stere +lat_0=90 +lon_0=0 +lat_ts=45 ...
  xscale = 500
  yscale = 500
  xsize = 3472
  ysize = 3472
```

Interpretation dataset :

- `ACRR` est la lame d'eau accumulee sur le pas du produit ;
- valeur normalisee = `raw * gain + offset`, sauf `nodata` et `undetect` ;
- `QIND` donne un indicateur de qualite ou confiance du pixel.

### `/station/paquet`

Les paquets individuels Ajaccio et Aleria sont aussi des `gzip` contenant une
archive `tar`.

Pour `id_station=37` et `id_station=61`, l'API renvoie des fichiers comme :

```text
T_PAGA37_C_EODC_YYYYMMDDHHMMSS.bufr.gz
T_PAMA37_C_LFPW_YYYYMMDDHHMMSS.bufr.gz
T_IPSR37_C_LFPW_YYYYMMDDHHMMSS.bufr.gz
```

Ces produits sont plus riches, mais en BUFR. Ils demandent un decodeur
specialise, typiquement `eccodes`/`bufr_dump` ou une bibliotheque Python
compatible.

Decision : ne pas les mettre dans le chemin critique du premier dataset.

## Features par spot

Pour chaque spot GPS, on peut echantillonner la mosaique HDF5 `IPRN20`.

Features recommandees :

```text
radar_precip_5min_mm_nearest
radar_precip_5min_mm_mean_1km
radar_precip_5min_mm_max_1km
radar_precip_5min_mm_mean_3km
radar_precip_5min_mm_max_3km
radar_precip_5min_mm_mean_10km
radar_precip_5min_mm_max_10km
radar_precip_15min_mm
radar_precip_30min_mm
radar_precip_60min_mm
radar_quality_nearest
radar_quality_mean_3km
radar_rain_detected_3km
radar_rain_detected_10km
radar_distance_to_rain_km
radar_direction_to_rain_deg
radar_upwind_rain_detected
```

La feature la plus importante pour le vent n'est pas seulement "il pleut ici".
Il faut aussi savoir si une zone de pluie ou convection est proche et arrive
sur le spot. Pour cela, on calcule :

- le maximum de pluie dans un rayon 10-20 km ;
- la distance au premier pixel pluvieux significatif ;
- la direction de ce pixel par rapport au spot ;
- si ce pixel est dans le secteur amont du vent actuel ou prevu ;
- l'evolution entre les trois images 5 min du dernier quart d'heure.

## Alignement temporel

Le radar est disponible toutes les 5 minutes.

Le dataset CorseWind.ai a des observations a 6 min, 15 min et 1 h. Strategie :

- conserver le radar en natif 5 min dans une table brute ;
- pour les features 6 min, prendre le dernier radar disponible strictement
  anterieur ou egal a l'observation ;
- pour les features 15 min, cumuler ou agreger les trois images radar 5 min ;
- pour les features 1 h, cumuler ou agreger les douze images radar 5 min ;
- ne jamais utiliser une image radar posterieure au timestamp de prediction.

## Stockage recommande

Raw archive :

```text
data/processed/ml_dataset/radar/raw/source=dppaqradar/date=YYYY-MM-DD/
```

Extraction spot :

```text
data/processed/ml_dataset/radar/spot_samples/date=YYYY-MM-DD/*.parquet
```

Schema spot sample :

```text
spot_id
valid_time_utc
radar_product
radar_file
precip_5min_mm_nearest
precip_5min_mm_mean_1km
precip_5min_mm_max_1km
precip_5min_mm_mean_3km
precip_5min_mm_max_3km
precip_5min_mm_mean_10km
precip_5min_mm_max_10km
quality_nearest
quality_mean_3km
rain_detected_3km
rain_detected_10km
distance_to_rain_km
direction_to_rain_deg
source_fetched_at_utc
```

## Implementation proposee

Phase 1 : mosaique HDF5.

1. Ajouter un collecteur `DPPaquetRadar /mosaique/paquet`.
2. Archiver le gzip/tar brut ou les HDF5 extraits.
3. Ne garder pour la Corse que les fichiers `IPRN20`.
4. Lire les HDF5 avec `h5py` ou un outil equivalent.
5. Reprojeter les spots GPS avec `pyproj`.
6. Convertir lat/lon en pixel grille.
7. Echantillonner `ACRR` et `QIND` autour du spot.
8. Produire des features rolling 15/30/60 min.

Phase 2 : radar individuel BUFR.

1. Installer `eccodes`.
2. Decoder `PAG`, `PAM`, `IPSR` pour Ajaccio et Aleria.
3. Evaluer si les variables supplementaires ameliorent les scores.
4. Garder seulement si le gain modele justifie la complexite.

## Decision

Oui, on ajoute le radar au blueprint dataset, mais comme source de contexte.

Priorite :

1. observations vent/rafales/temp/pression/rayonnement ;
2. NWP brut multi-modeles ;
3. radar mosaique precipitation 5 min ;
4. SYNOP/bouees enrichies ;
5. radar individuel BUFR si utile apres benchmark.

Le radar peut ameliorer les previsions dans les situations perturbees :
averses, grains, convection, fronts, bascules rapides, rafales anormales.

Il aura probablement peu d'impact sur les journees thermiques seches et
stables, mais il aidera le modele a identifier justement ces journees propres
par absence de pluie/convection autour du spot.
