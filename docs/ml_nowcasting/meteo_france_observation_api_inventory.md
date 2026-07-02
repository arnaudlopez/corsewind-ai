# Meteo-France Observation API Inventory

Date: 2026-06-23

## Sources inspectees

Swagger locaux fournis :

```text
/Users/arnaud/Downloads/Package_Observations_swagger (2).json
/Users/arnaud/Downloads/Donnees_d'observation_swagger (3).json
```

Endpoints racines :

```text
DPPaquetObs v2: https://public-api.meteofrance.fr/public/DPPaquetObs/v2
DPObs v2:       https://public-api.meteofrance.fr/public/DPObs/v2
```

Les Swagger ne contiennent pas de schemas de champs detailles. L'inventaire
ci-dessous vient donc d'appels reels aux endpoints avec la cle Meteo-France du
projet, sans exposer la cle.

## Conclusion courte

Pour CorseWind ML, les endpoints les plus utiles sont :

1. `DPPaquetObs /paquet/stations/infrahoraire-6m`
   - meilleur endpoint pour archiver toutes les stations a un instant 6 min ;
   - permet ensuite de filtrer les stations corses.
2. `DPPaquetObs /paquet/infrahoraire-6m`
   - meilleur endpoint pour recuperer les dernieres 24 h d'une station precise.
3. `DPPaquetObs /paquet/stations/horaire`
   - meilleur endpoint bulk horaire toutes stations.
4. `DPObs /synop`
   - plus riche pour nebulosite, temps present, couches nuageuses, pluie 1/3/6/12/24 h.
5. `DPObs /bouees`
   - essentiel pour mer/houle/temp mer, notamment bouee Ajaccio.

## Endpoints disponibles

### DPPaquetObs

```text
GET /liste-stations
GET /paquet/infrahoraire-6m
GET /paquet/horaire
GET /paquet/stations/infrahoraire-6m
GET /paquet/stations/horaire
```

Usage recommande :

- `paquet/stations/infrahoraire-6m` pour ingestion bulk temps reel ;
- `paquet/infrahoraire-6m?id_station=...` pour bootstrap ou rattrapage 24 h ;
- `paquet/stations/horaire` pour enrichissement horaire bulk.

### DPObs

```text
GET /station/infrahoraire-6m
GET /station/horaire
GET /liste-stations
GET /liste-stations-synop
GET /synop
GET /liste-bouees
GET /bouees
```

Usage recommande :

- `station/infrahoraire-6m` et `station/horaire` pour interrogation ponctuelle ;
- `synop` pour metriques atmospheriques plus riches ;
- `bouees` pour mer/houle/temp mer.

## Stations corses

`/liste-stations` retourne environ 2151 stations nationales. Filtrage Corse
par lat/lon donne 55 stations environ.

Stations deja pertinentes pour Beacon Live / CorseWind :

| Id station | OMM | Nom | Lat | Lon | Alt m | Pack |
|---|---:|---|---:|---:|---:|---|
| 20004002 | 07761 | AJACCIO | 41.918000 | 8.792667 | 5 | RADOME |
| 20004003 | 07752 | AJACCIO-PARATA | 41.908333 | 8.618167 | 124 | ETENDU |
| 20041001 | 07770 | CAP PERTUSATO | 41.374833 | 9.178333 | 107 | RADOME |
| 20050001 | 07754 | CALVI | 42.529500 | 8.791500 | 57 | RADOME |
| 20093002 | 07753 | ILE ROUSSE | 42.633333 | 8.922500 | 140 | RADOME |
| 20107001 | 07785 | CAP CORSE | 43.003833 | 9.359500 | 72 | RADOME |
| 20114002 | 07780 | FIGARI | 41.505167 | 9.103667 | 20 | RADOME |
| 20148001 | 07790 | BASTIA | 42.540667 | 9.485167 | 10 | RADOME |
| 20342001 | 07765 | SOLENZARA | 41.921833 | 9.400833 | n/a | n/a |

Le bulk 6 min a retourne 50 lignes corses sur l'instant teste.
Le bulk horaire a retourne 52 lignes corses sur l'instant teste.

## Metriques infrahoraire 6 min

Endpoints testes :

```text
DPObs /station/infrahoraire-6m
DPPaquetObs /paquet/infrahoraire-6m
DPPaquetObs /paquet/stations/infrahoraire-6m
```

Champs observes :

| Champ | Sens probable | Unite API | Normalisation ML |
|---|---|---:|---|
| `lat` | latitude | deg | deg |
| `lon` | longitude | deg | deg |
| `geo_id_insee` | id station | texte | texte |
| `reference_time` | reference produit | ISO UTC | datetime UTC |
| `insert_time` | insertion API | ISO UTC | datetime UTC |
| `validity_time` | temps observation | ISO UTC | datetime UTC |
| `dd` | direction vent moyen | deg | deg |
| `ff` | vitesse vent moyen | m/s | m/s |
| `ddraf10` | direction rafale 10 min | deg | deg |
| `raf10` | rafale 10 min | m/s | m/s |
| `t` | temperature air | K | C |
| `td` | point de rosee | K | C |
| `u` | humidite relative | % | % |
| `rr_per` | precipitation periode | mm probable | mm |
| `pres` | pression station | Pa | hPa |
| `pmer` | pression niveau mer | Pa | hPa |
| `vv` | visibilite | m | m |
| `etat_sol` | etat du sol | code | code |
| `sss` | neige au sol | cm probable | cm |
| `insolh` | duree ensoleillement sur periode | min | min |
| `ray_glo01` | rayonnement global sur periode | J/m2 probable | J/m2 ou Wh/m2 derive |
| `t_10` | temperature sol -10 cm | K | C |
| `t_20` | temperature sol -20 cm | K | C |
| `t_50` | temperature sol -50 cm | K | C |
| `t_100` | temperature sol -100 cm | K | C |

Interet ML fort :

- `ff`, `raf10`, `dd`, `ddraf10` comme cible et contexte ;
- `t`, `td`, `u`, `pres`, `pmer` pour brise/gradient ;
- `insolh`, `ray_glo01` pour thermique ;
- `rr_per`, `vv`, `etat_sol` pour regimes perturbes ;
- temperatures sol si disponibles pour inertie thermique.

## Metriques horaires

Endpoints testes :

```text
DPObs /station/horaire
DPPaquetObs /paquet/stations/horaire
```

Champs observes :

| Champ | Sens probable | Unite API | Normalisation ML |
|---|---|---:|---|
| `dd` | direction vent moyen | deg | deg |
| `ff` | vitesse vent moyen | m/s | m/s |
| `dxy` | direction du vent max horaire | deg | deg |
| `fxy` | vitesse du vent max horaire | m/s | m/s |
| `ddraf` | direction rafale horaire | deg | deg |
| `raf` | rafale horaire | m/s | m/s |
| `t` | temperature air instantanee | K | C |
| `tx` | temperature max horaire | K | C |
| `tn` | temperature min horaire | K | C |
| `td` | point de rosee | K | C |
| `u` | humidite relative | % | % |
| `ux` | humidite max | % | % |
| `un` | humidite min | % | % |
| `rr1` | precipitation 1 h | mm | mm |
| `pres` | pression station | Pa | hPa |
| `pmer` | pression niveau mer | Pa | hPa |
| `vv` | visibilite | m | m |
| `n` | nebulosite totale | octas/code | code |
| `etat_sol` | etat du sol | code | code |
| `sss` | neige au sol | cm probable | cm |
| `insolh` | duree ensoleillement horaire | min | min |
| `ray_glo01` | rayonnement global horaire | J/m2 probable | J/m2 ou Wh/m2 derive |
| `t_10`, `t_20`, `t_50`, `t_100` | temperatures sol | K | C |

Interet ML :

- tres utile pour agregats 1 h et verification de tendance ;
- moins reactif que le 6 min, mais plus riche sur min/max.

## Metriques SYNOP

Endpoint teste :

```text
DPObs /synop?id_station=07753,07754,07761,07765,07770,07775,07780,07785,07790
```

Stations SYNOP corses observees :

```text
07753 ILE ROUSSE
07754 CALVI
07761 AJACCIO
07765 SOLENZARA
07770 CAP PERTUSATO
07775 ALISTRO
07780 FIGARI
07785 CAP CORSE
07790 BASTIA
```

Champs observes :

| Famille | Champs |
|---|---|
| Identite | `geo_id_wmo`, `geo_id_wigos`, `name`, `lat`, `lon` |
| Temps | `reference_time`, `insert_time`, `validity_time` |
| Vent | `dd`, `ff`, `raf10`, `rafper`, `per` |
| Temperature/humidite | `t`, `td`, `u`, `tminsol`, `tn12`, `tn24`, `tx12`, `tx24` |
| Pression | `pres`, `pmer`, `tend`, `cod_tend`, `tend24`, `niv_bar`, `geop` |
| Pluie | `rr1`, `rr3`, `rr6`, `rr12`, `rr24` |
| Visibilite/temps present | `vv`, `ww`, `w1`, `w2` |
| Nebulosite | `n`, `nbas`, `hbas`, `cl`, `cm`, `ch` |
| Nuages couches | `nnuage1..4`, `ctype1..4`, `hnuage1..4` |
| Sol/neige | `etat_sol`, `ht_neige`, `ssfrai`, `perssfrai` |
| Phenomenes speciaux | `phenspe1..4`, `sw`, `tw` |

Interet ML fort :

- `tend`, `cod_tend`, `tend24` pour evolution de pression ;
- `rr1/3/6/12/24` pour regimes de pluie ;
- `n`, `nbas`, `hbas`, `nnuage*`, `hnuage*`, `ww` pour couverture nuageuse
  et etat du ciel ;
- `rafper` peut completer les rafales.

## Metriques bouees

Endpoint teste :

```text
DPObs /bouees?id_bouees=6101031
```

Bouee Ajaccio observee :

```text
6101031 BOUEE_AJACCIO
```

Champs observes :

| Champ | Sens probable | Unite API | Normalisation ML |
|---|---|---:|---|
| `dd` | direction vent | deg | deg |
| `ff` | vent moyen | m/s | m/s |
| `rafper` | rafale periode | m/s | m/s |
| `t` | temperature air | K | C |
| `td` | point de rosee | K | C |
| `u` | humidite relative | % | % |
| `pmer` | pression mer | Pa | hPa |
| `tmer` | temperature mer | K | C |
| `haut_vag` | hauteur vague | m | m |
| `per_moy_vag` | periode moyenne vagues | s | s |
| `dir_vag` | direction vagues | deg | deg |
| `per` | periode/code | code ou s selon doc | a verifier |

Interet ML :

- temperature mer pour contraste terre/mer ;
- houle comme contexte produit ;
- vent large en mer comme contexte pour golfe/littoral.

## Strategie d'ingestion recommandee

### Temps reel / collecte continue

Toutes les 6 minutes :

```text
GET /DPPaquetObs/v2/paquet/stations/infrahoraire-6m?date=<YYYY-MM-DDTHH:mm:00Z>&format=json
```

Puis :

- filtrer lat/lon Corse ;
- archiver JSONL normalise ;
- mettre a jour observations 15 min.

Toutes les heures :

```text
GET /DPPaquetObs/v2/paquet/stations/horaire?date=<YYYY-MM-DDTHH:00:00Z>&format=json
GET /DPObs/v2/synop?format=json&id_station=<ids_synop_corse>
GET /DPObs/v2/bouees?format=json&id_bouees=6101031
```

### Bootstrap par station

Pour une station precise :

```text
GET /DPPaquetObs/v2/paquet/infrahoraire-6m?id_station=<id>&format=json
```

Ce endpoint retourne jusqu'a environ 24 h de donnees 6 min pour la station.

## Normalisation dataset

Unites a appliquer :

| API | Dataset ML |
|---|---|
| temperature K | deg C |
| pression Pa | hPa |
| vent m/s | m/s |
| direction deg | deg + composantes sin/cos derivees |
| rayonnement J/m2 probable | garder brut + convertir Wh/m2 si confirme |
| precipitation mm | mm |

Noms dataset proposes :

```text
wind_mean_ms
wind_gust_10m_ms
wind_dir_deg
wind_gust_dir_deg
temperature_c
dewpoint_c
humidity_pct
station_pressure_hpa
sea_level_pressure_hpa
precip_period_mm
visibility_m
sunshine_minutes
global_radiation_j_m2
soil_temperature_10cm_c
soil_temperature_20cm_c
soil_temperature_50cm_c
soil_temperature_100cm_c
cloud_cover_code
present_weather_code
wave_height_m
wave_period_s
wave_direction_deg
sea_temperature_c
```

## Priorites CorseWind

Priorite 1, a integrer tout de suite :

- `ff`, `raf10`/`raf`, `dd`, `ddraf10`/`ddraf` ;
- `t`, `td`, `u` ;
- `pres`, `pmer` ;
- `rr_per`/`rr1` ;
- `insolh`, `ray_glo01` quand disponibles.

Priorite 2 :

- SYNOP `n`, `nbas`, `hbas`, `ww`, `rr3/6/12/24`, `tend`, `tend24` ;
- bouee `tmer`, `haut_vag`, `per_moy_vag`, `dir_vag`.

Priorite 3 :

- sol/neige ;
- couches nuageuses detaillees ;
- phenomenes speciaux SYNOP.

## Decision

Le moteur d'agregation observations Meteo-France doit utiliser DPPaquetObs en
collecte principale et DPObs en enrichissement :

```text
DPPaquetObs = base haute frequence / bulk stations
DPObs SYNOP = enrichissement atmosphere/nuages/pluie
DPObs bouees = enrichissement mer/houle/temp mer
```
