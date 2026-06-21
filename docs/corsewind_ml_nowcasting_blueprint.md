# CorseWind ML Nowcasting Blueprint

## Statut du document

Ce document est le rail de reference pour construire la couche ML de
CorseWind.ai. Il decrit ce que nous voulons predire, quelles donnees utiliser,
comment entrainer et evaluer les modeles, et comment les integrer au moteur
operationnel existant.

Il est volontairement oriente decision windsurf. Le but n'est pas de refaire un
modele meteo global, mais de produire une prevision locale plus utile que les
modeles bruts pour choisir le bon spot, le bon horaire, et le bon niveau de
risque.

References principales :

- Etude SAPHIR : "Local wind speed forecasting at short time horizons based on
  Numerical Weather Prediction and observations from surrounding stations",
  arXiv 2503.18797v2, https://arxiv.org/html/2503.18797v2
- Code SAPHIR publie sur Zenodo : https://doi.org/10.5281/zenodo.15222910
- Chronos-2 : https://huggingface.co/amazon/chronos-2
- IBM Granite TTM r2 :
  https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2

## Objectif produit

CorseWind doit repondre a une question simple :

```text
Est-ce que ce spot sera navigable, quand, avec quelle force, quelle direction,
quelle regularite, et quel niveau de confiance ?
```

Le systeme doit etre particulierement bon sur la fenetre windsurf :

```text
11h-17h locale
```

mais il ne doit pas etre entraine uniquement sur cette fenetre. Le cycle complet
de la journee contient des signaux utiles : nuit precedente, matinee, montee en
temperature, bascule de direction, evolution pression, mise en place ou echec
du thermique.

## Hypothese centrale

La prevision brute AROME, AROME-PI, MOLOCH, ICON-2I ou autre modele NWP apporte
la dynamique atmospherique generale. Les observations locales apportent les
effets fins que les modeles resolvent mal :

- relief ;
- exposition de cote ;
- effet Venturi ;
- deventes ;
- acceleration thermique ;
- biais local d'un spot ;
- erreur systematique selon la direction ;
- demarrage ou effondrement rapide du vent.

La couche ML doit donc agir comme un modele de correction locale et de
nowcasting :

```text
observations recentes + previsions NWP + contexte spot
        -> vent reel attendu au spot
```

## Comparaison avec l'etude SAPHIR

L'etude SAPHIR est notre base scientifique la plus proche.

### Ce que SAPHIR fait

SAPHIR predit le vent de surface local a horizons courts, typiquement 1 a 6 h.
Le modele combine :

- observations de stations ;
- predictions AROME ;
- predictions ARPEGE ;
- informations temporelles ;
- position et altitude station.

Leur architecture hybride fusionne plusieurs branches :

- une branche temporelle pour les observations recentes ;
- des branches spatiales/convolutionnelles pour les sous-grilles NWP autour de
  la station ;
- une sortie deterministe ou probabiliste.

Ils comparent le modele a :

- persistance ;
- AROME brut ;
- ARPEGE brut ;
- regression lineaire ;
- modeles utilisant seulement une partie des entrees.

Resultat cle : le modele hybride bat les baselines, avec un gain fort face a
AROME brut, et le fine-tuning par station corse ameliore encore les resultats.
L'article indique aussi que l'inference est legere : leur chaine operationnelle
horaire complete peut tourner en moins de quelques secondes sur laptop.

### Ce que CorseWind reprend

Nous reprenons :

- l'approche hybride NWP + observations ;
- la prevision courte echeance ;
- la logique station cible + stations voisines ;
- la comparaison systematique aux baselines ;
- la validation temporelle stricte ;
- la prediction probabiliste ;
- le fine-tuning local par station/spot ;
- les scores de depassement de seuils.

### Ce que CorseWind change

CorseWind n'a pas le meme objectif metier.

```text
SAPHIR :
  prevision scientifique du vent local 1-6 h

CorseWind :
  decision windsurf locale, frequente, orientee 11h-17h
```

Differences assumees :

- pas de temps cible 15 min au lieu d'une evaluation uniquement horaire ;
- recalcul operationnel toutes les 8 a 15 min ;
- importance forte de la fenetre 11h-17h ;
- ajout de variables soleil/pression/tendances si disponibles ;
- prediction des rafales et de la direction, pas seulement vitesse moyenne ;
- sortie metier navigable / limite / pas navigable ;
- optimisation autour des seuils windsurf, pas seulement vents extremes meteo ;
- integration au moteur existant AROME-PI, AROME, MOLOCH, ICON-2I, WindNinja.

## Produit attendu

Pour chaque spot ou station cible, produire :

```text
vent moyen prevu
rafale prevue
direction prevue
quantiles P10/P50/P90
probabilite de depassement de seuils
score session
timestamp du run
sources utilisees
niveau de confiance
```

Horizons prioritaires :

```text
+15 min
+30 min
+45 min
+1 h
+2 h
+3 h
+6 h
jusqu'a la fin de la fenetre 11h-17h
```

Horizons secondaires :

```text
demain 11h, 13h, 15h, 17h
```

## Donnees d'entree

### Observations brutes

Priorite haute :

- vent moyen ;
- rafales ;
- direction ;
- temperature ;
- pression ;
- humidite ;
- ensoleillement ou rayonnement solaire constate ;
- pluie ;
- qualite capteur ;
- indicateur donnee manquante.

Les observations doivent etre historisees au pas natif disponible. Les sources
connues peuvent arriver a plusieurs resolutions :

```text
6 min
15 min
1 h
```

La regle de depart est :

- conserver le brut a sa resolution native ;
- produire une vue operationnelle principale au pas 15 min ;
- agregger le 6 min vers 15 min avec des statistiques utiles ;
- injecter les sources horaires comme covariables avec leur age et leur lead
  time ;
- ne jamais melanger deux resolutions sans garder la provenance.

Le pas 15 min reste le pas de publication prioritaire, mais le 6 min est
precieux pour detecter les transitions rapides : demarrage thermique, rotation
de direction, rafales, molles, chute brutale du vent.

### Strategie multi-resolution

Pour chaque fenetre 15 min, les observations 6 min doivent produire plusieurs
features, pas seulement une moyenne :

- derniere valeur disponible ;
- moyenne 15 min ;
- minimum / maximum 15 min ;
- rafale max 15 min ;
- ecart-type 15 min ;
- tendance entre debut et fin de fenetre ;
- age de la derniere observation ;
- nombre de points disponibles.

Exemple :

```text
observations 6 min dans [12:00, 12:15]
  -> vent_last_15m
  -> vent_mean_15m
  -> gust_max_15m
  -> wind_std_15m
  -> direction_rotation_15m
  -> obs_count_15m
```

Les sources 1 h doivent etre alignees sur la grille 15 min sans inventer une
precision artificielle. Elles peuvent etre :

- forward-fill avec variable `age_minutes` ;
- interpolees si la variable est continue et que la source le permet ;
- conservees comme "dernier run connu" + "prochaine echeance connue".

Le modele doit recevoir explicitement :

```text
source_resolution_minutes
source_age_minutes
lead_time_minutes
is_interpolated
is_forward_filled
```

Objectif : permettre au modele de comprendre qu'une observation 6 min recente
n'a pas la meme valeur informationnelle qu'une prevision horaire vieille de 45
min.

### Observations derivees

Les modeles de series temporelles peuvent apprendre des variations seuls, mais
il faut leur donner des indices meteorologiques utiles pour reduire la quantite
de donnees necessaire et ameliorer la generalisation.

Features candidates :

- vent moyen 30 min, 1 h, 3 h ;
- rafale max 30 min, 1 h, 3 h ;
- variation vent 30 min, 1 h, 3 h ;
- rotation de direction 30 min, 1 h, 3 h ;
- temperature actuelle ;
- variation temperature 1 h, 3 h, depuis 8 h locale ;
- pression actuelle ;
- variation pression 1 h, 3 h, 6 h ;
- humidite actuelle et tendance ;
- ensoleillement cumule depuis 8 h locale ;
- pluie cumulee 1 h, 3 h ;
- ecart entre observation et NWP brut au dernier pas disponible ;
- indicateur "thermique potentiel" ;
- indicateur "vent synoptique dominant".

Important : une feature est autorisee uniquement si elle est disponible au
moment de la prediction. Aucune observation future ne doit entrer dans les
features.

### Previsions NWP

Sources existantes ou candidates :

- AROME ;
- AROME-PI ;
- MOLOCH ;
- ICON-2I ;
- ARPEGE / ECMWF si disponible plus tard ;
- WindNinja 50 m derive d'AROME pour les echeances session.

Variables candidates :

- vent u/v 10 m ;
- vitesse 10 m ;
- direction 10 m ;
- rafales si disponibles ;
- temperature 2 m ;
- pression mer ou surface ;
- humidite ;
- pluie ;
- nebulosite ;
- rayonnement solaire ;
- CAPE / convection si disponible ;
- heure du run ;
- lead time ;
- age effectif de la prevision.

Pour chaque spot/station, il faut stocker :

- valeur interpolee au point ;
- moyenne autour du spot ;
- gradient local si disponible ;
- comparaison entre plusieurs modeles ;
- ecart entre NWP et derniere observation.

### Metadonnees spot

Variables fixes ou quasi fixes :

- latitude ;
- longitude ;
- altitude ;
- distance a la mer ;
- orientation de la cote ;
- exposition aux directions dominantes ;
- type de spot : baie, cap, plaine, vallee, relief ;
- rugosite locale si disponible ;
- groupe de spots comparables ;
- seuils windsurf propres au spot.

## Cibles a predire

### Regression

Predire :

- vitesse moyenne ;
- rafale ;
- direction ;
- eventuellement composantes u/v plutot que direction directe.

La direction doit etre traitee avec prudence. Pour eviter les discontinuites
0/360 degres, preferer :

```text
u = vitesse * cos(direction)
v = vitesse * sin(direction)
```

puis reconstruire vitesse/direction en sortie.

### Probabiliste

Predire des quantiles :

```text
P10
P50
P90
```

Ces quantiles servent a afficher :

- scenario prudent ;
- scenario median ;
- scenario optimiste ;
- incertitude ;
- risque de rafales ou molles.

### Classification metier

Predire aussi :

- probabilite vent > 12 noeuds ;
- probabilite vent > 15 noeuds ;
- probabilite vent > 18 noeuds ;
- probabilite vent > 22 noeuds ;
- probabilite vent trop fort ;
- probabilite direction compatible ;
- score "navigable" ;
- score "irregulier / rafaleux".

Les seuils doivent etre configurables par spot et par pratique.

## Fenetre windsurf

Le modele doit voir toute la journee, mais la loss et les metriques doivent
favoriser la fenetre utile.

Regle de depart :

```text
poids erreur 11h-17h = 3
poids erreur hors fenetre = 1
poids erreur autour des seuils navigables = 2 a 4
```

Exemples de priorites :

- une erreur a 14h compte plus qu'une erreur a 3h ;
- une erreur qui transforme "navigable" en "pas navigable" compte beaucoup ;
- une erreur de timing de montee du vent compte plus qu'une petite erreur
  nocturne ;
- un mauvais sens de direction sur un spot expose compte beaucoup.

## Modeles a tester

### Baseline 0 : climatologie locale

Reference minimale :

```text
vent moyen historique par spot, mois, heure, direction synoptique
```

### Baseline 1 : persistance

Reference nowcasting :

```text
vent futur = dernier vent observe
```

Variantes :

- persistance simple ;
- persistance tendance 1 h ;
- moyenne mobile 30 min / 1 h.

### Baseline 2 : NWP brut

Reference modele :

```text
AROME/AROME-PI/MOLOCH/ICON-2I interpole au spot
```

Il faut mesurer les biais par :

- spot ;
- heure ;
- saison ;
- direction ;
- force du vent ;
- horizon.

### Baseline 3 : modele tabulaire

Premier modele ML recommande :

```text
LightGBM ou XGBoost
```

Pourquoi :

- rapide ;
- robuste ;
- excellent sur features derivees ;
- interpretable ;
- bon pour etablir l'importance des variables.

Ce modele sert aussi de garde-fou. Si un modele complexe ne bat pas ce baseline,
il n'est pas pret.

### Modele 4 : IBM TTM

IBM Granite TTM r2 est un candidat leger pour series temporelles.

Usage prevu :

- fine-tuning local ;
- inference frequente ;
- benchmark par spot ;
- execution CPU ou petit GPU ;
- pas de temps 15 min.

Role probable :

```text
modele operationnel leger pour prevision rapide
```

### Modele 5 : Chronos-2

Chronos-2 est un candidat fondation model plus general.

Usage prevu :

- zero-shot initial ;
- covariables passees et futures ;
- quantiles probabilistes ;
- comparaison avec TTM et tabulaire ;
- eventuel fine-tuning si utile.

Role probable :

```text
modele probabiliste riche ou modele de comparaison haut niveau
```

### Modele 6 : hybride maison inspire SAPHIR

Si les resultats justifient un modele dedie, construire une architecture locale :

- branche observations station cible ;
- branche stations voisines ;
- branche NWP interpolee ou grille locale ;
- branche metadonnees spot ;
- fusion dense ;
- sorties regression + quantiles + classification.

Ce modele est plus couteux a maintenir. Il ne doit venir qu'apres les baselines.

## Strategie d'entrainement

### Split temporel

Ne jamais splitter ligne par ligne au hasard. Les donnees proches dans le temps
sont tres correlees.

Regle :

```text
train       = periodes anciennes
validation  = periodes completes separees
test        = saison ou mois recents jamais vus
```

Variantes utiles :

- split par annees ;
- split par saisons ;
- test special sur episodes forts ;
- test special sur mois d'ete.

### Donnees manquantes

Chaque variable doit avoir :

- valeur ;
- indicateur "missing" ;
- age de la derniere observation si forward-fill ;
- source.

Ne pas supprimer trop agressivement les lignes. En operationnel, les trous
arriveront. Le modele doit apprendre a fonctionner avec des donnees imparfaites.

### Normalisation

Normaliser avec statistiques calculees sur le train uniquement :

- moyenne/ecart-type par variable ;
- eventuellement par station ;
- encodage cyclique heure/jour.

Encodages temporels :

```text
sin(hour)
cos(hour)
sin(day_of_year)
cos(day_of_year)
is_weekend si utile plus tard
```

### Fine-tuning local

Ordre recommande :

1. modele global Corse ou toutes stations disponibles ;
2. fine-tuning par station cible ;
3. fine-tuning par groupe de spots similaires ;
4. calibration probabiliste par spot.

Cette logique reprend l'esprit SAPHIR : entrainer globalement, puis adapter aux
stations corses pour gagner en precision locale sans surapprendre trop vite.

## Evaluation

### Metriques continues

Calculer :

- MAE vent moyen ;
- RMSE vent moyen ;
- MAE rafale ;
- erreur directionnelle ;
- MAE par horizon ;
- MAE par heure locale ;
- MAE 11h-17h ;
- MAE hors 11h-17h.

### Metriques seuils windsurf

Calculer pour chaque seuil :

- precision ;
- recall ;
- F1 ;
- Brier score ;
- CSI ;
- PSS ;
- taux de faux "bonne session" ;
- taux de sessions manquees.

Seuils initiaux :

```text
12 noeuds
15 noeuds
18 noeuds
22 noeuds
```

Ces seuils seront ensuite adaptes par spot.

### Metriques timing

Mesurer :

- erreur d'heure de montee au-dessus de 12/15 noeuds ;
- erreur d'heure de chute ;
- duree de fenetre navigable prevue vs observee ;
- erreur sur le meilleur creneau de la journee.

Ces metriques sont critiques pour le produit. Une faible RMSE ne suffit pas.

### Benchmarks obligatoires

Aucun modele ML ne doit etre accepte sans comparaison a :

- climatologie locale ;
- persistance ;
- NWP brut ;
- LightGBM/XGBoost.

Un modele complexe est accepte seulement s'il ameliore clairement :

- 11h-17h ;
- les seuils windsurf ;
- le timing de session ;
- la calibration de l'incertitude.

## Boucle operationnelle

La couche ML doit s'integrer au moteur existant sans bloquer les couches brutes.

Cycle cible :

```text
toutes les 8 a 15 min :
  1. lire dernieres observations
  2. controler qualite
  3. aligner au pas 15 min
  4. charger derniers champs NWP disponibles
  5. construire features par spot
  6. lancer inference batch
  7. publier predictions + incertitude + score
```

Le modele doit rester charge en memoire pour eviter le cout de cold start.

Sorties candidates :

```text
data/processed/ml_nowcasting/predictions_latest.json
data/processed/ml_nowcasting/predictions_<run_id>.json
data/processed/ml_nowcasting/diagnostics_latest.json
visualizations/wind2d/ml-nowcast-corsica-latest.json
```

Le contrat exact de sortie sera defini quand le premier prototype sera pret.

## Controle qualite

Regles minimales :

- vent negatif impossible ;
- direction dans 0-360 ;
- rafale >= vent moyen, sauf tolerance capteur ;
- saut de vitesse impossible marque suspect ;
- station muette marquee stale ;
- observation trop ancienne exclue ou penalisee ;
- NWP absent degrade vers modele sans NWP ;
- modele absent degrade vers NWP brut ou persistance.

Chaque prediction doit contenir :

- model_version ;
- features_version ;
- sources ;
- generated_at_utc ;
- valid_time_utc ;
- target_spot ;
- missing_features_count ;
- quality_flags.

## Architecture de donnees proposee

```text
data/raw/
  observations/
  nwp/

data/processed/
  observations_15min/
  spot_features/
  ml_training/
  ml_nowcasting/
  diagnostics/

models/
  ml_nowcasting/
    baseline_xgboost/
    ttm/
    chronos/

reports/
  ml_nowcasting/
```

Ne pas committer les donnees brutes, les checkpoints lourds ou les secrets.

## Roadmap

### Phase 1 : dataset et baselines

Objectif : verifier que le signal existe.

Taches :

- choisir 3 a 5 stations/spots pilotes ;
- construire dataset 15 min ;
- integrer observations + NWP brut ;
- calculer features derivees ;
- entrainer persistance, NWP brut, LightGBM/XGBoost ;
- produire rapport 11h-17h.

Critere de succes :

```text
LightGBM/XGBoost bat NWP brut et persistance sur 11h-17h.
```

### Phase 2 : fondation models

Objectif : tester si TTM/Chronos apportent un gain.

Taches :

- formatter dataset pour TTM ;
- benchmark TTM zero-shot/fine-tune ;
- benchmark Chronos-2 zero-shot ;
- comparer aux baselines ;
- mesurer latence inference.

Critere de succes :

```text
TTM ou Chronos ameliore les seuils windsurf ou l'incertitude sans latence
operationnelle excessive.
```

### Phase 3 : probabiliste et score session

Objectif : passer de "prevoir le vent" a "decider la session".

Taches :

- produire quantiles ;
- calibrer probabilites ;
- definir score navigabilite ;
- evaluer faux positifs/faux negatifs de sessions ;
- ajouter explications simples : source du signal, confiance, risque.

Critere de succes :

```text
Le score session reduit les mauvaises recommandations par rapport au NWP brut.
```

### Phase 4 : integration operationnelle

Objectif : brancher le ML au moteur CorseWind.

Taches :

- service inference charge en memoire ;
- sortie JSON stable ;
- diagnostics ;
- fallback ;
- integration Wind2D ;
- monitoring latence et qualite.

Critere de succes :

```text
Une nouvelle observation met a jour les predictions en moins de 2 minutes.
```

### Phase 5 : modele hybride dedie

Objectif : decider si un modele inspire SAPHIR vaut le cout.

Taches :

- construire architecture multi-branches ;
- comparer a TTM/Chronos/LightGBM ;
- tester fine-tuning par station ;
- tester grille NWP locale autour du spot ;
- documenter gain vs complexite.

Critere de succes :

```text
Gain clair sur 11h-17h, seuils windsurf et timing, superieur au cout de
maintenance.
```

## Decisions de depart

1. Entrainer sur toute la journee, mais ponderer 11h-17h.
2. Utiliser les observations brutes et les features derivees.
3. Toujours comparer au NWP brut et a la persistance.
4. Commencer avec LightGBM/XGBoost avant les modeles complexes.
5. Tester TTM comme modele leger et Chronos-2 comme modele probabiliste riche.
6. Optimiser le produit sur la decision de session, pas seulement la RMSE.
7. Garder des fallbacks operationnels simples.
8. Eviter toute fuite temporelle dans les features et les splits.

## Questions ouvertes

- Quelles observations sont reellement disponibles en temps reel par station ?
- Dispose-t-on de rayonnement solaire observe ou seulement d'ensoleillement
  estime ?
- Quels spots pilotes choisir pour couvrir des comportements differents ?
- Quels seuils windsurf par spot : 12/15/18 noeuds suffisent-ils ?
- Faut-il predire directement les spots sans station ou passer par stations
  proches + correction spot ?
- Quelle source NWP sert de reference principale : AROME-PI, AROME, MOLOCH,
  ICON-2I, WindNinja ?
- Quel niveau d'explicabilite afficher a l'utilisateur final ?

## Regle de gouvernance

Toute nouvelle idee de modele doit passer par cette question :

```text
Est-ce que cela ameliore concretement la decision windsurf sur 11h-17h,
face a persistance, NWP brut et baseline tabulaire ?
```

Si la reponse n'est pas mesurable, l'idee reste experimentale et ne doit pas
entrer dans le chemin operationnel.
