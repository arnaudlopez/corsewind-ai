# Z2 GPU Smoke Test - ML Nowcasting

Date: 2026-06-23

## Machine

```text
ssh z2
hostname: Z2
OS: Debian, Linux 6.12.94+deb13-amd64
GPU: NVIDIA Quadro P2000
VRAM: 5120 MiB
driver: 550.163.01
RAM: 15 GiB
disk free: 332 GiB
```

## Environnements

`uv` a ete installe en local utilisateur :

```text
/home/z2/.local/bin/uv
uv 0.11.23
```

Deux environnements ont ete crees :

```text
/home/z2/corsewind-ml-smoke/.venv
  Python 3.11.15
  torch 2.6.0+cu124
  chronos-forecasting 2.3.0
  transformers 5.12.1

/home/z2/corsewind-ml-smoke/.venv-ttm
  Python 3.11.15
  torch 2.6.0+cu124
  granite-tsfm 0.2.22
  transformers 4.57.6
```

Raison des deux environnements : `granite-tsfm` v0.2.22 ne chargeait pas TTM
avec `transformers 5.12.1` (`all_tied_weights_keys`). Le chemin TTM fonctionne
avec `transformers<5`.

## PyTorch CUDA

Resultat :

```json
{
  "torch_version": "2.6.0+cu124",
  "torch_cuda_version": "12.4",
  "cuda_available": true,
  "device_count": 1,
  "device_name": "Quadro P2000",
  "total_memory_mib": 5043.4,
  "compute_capability": "6.1"
}
```

Test tenseur GPU :

```json
{
  "matmul_shape": "4096x4096",
  "elapsed_ms": 90.708,
  "allocated_mib": 200.1,
  "reserved_mib": 212.0
}
```

Conclusion : PyTorch CUDA fonctionne correctement sur `z2`.

## Chronos-2

Modele :

```text
amazon/chronos-2
```

Test :

```text
contexte: 96 pas de 15 min
horizon: 24 pas de 15 min
covariables futures synthetiques: temperature, pression
sorties: predictions + quantiles 0.1 / 0.5 / 0.9
dtype: float32
device: cuda
```

Smoke test initial :

```json
{
  "n_series": 5,
  "context_len": 96,
  "prediction_len": 24,
  "load_seconds": 1.048,
  "inference_seconds": 0.219,
  "rows_out": 120,
  "gpu_memory_reserved_mib_after_infer": 484.0
}
```

Benchmark `predict_df` :

| Series | Context | Horizon | Mean ms | Min ms | Max ms | Rows out | Reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96 | 24 | 23.126 | 22.826 | 23.636 | 24 | 480.0 |
| 10 | 96 | 24 | 72.252 | 70.274 | 74.262 | 240 | 490.0 |
| 50 | 96 | 24 | 215.942 | 213.910 | 217.528 | 1200 | 542.0 |
| 100 | 96 | 24 | 452.766 | 451.988 | 453.256 | 2400 | 638.0 |

Conclusion : Chronos-2 est utilisable sur la Quadro P2000 pour des batchs
modestes a moyens. Les temps observes sont tres compatibles avec une mise a
jour operationnelle toutes les 8 a 15 min.

## IBM Granite TTM r2

Modele :

```text
ibm-granite/granite-timeseries-ttm-r2
revision selectionnee: 512-96-ft-l1-r2.1
```

Test :

```text
contexte: 512 pas
horizon: 96 pas
freq: 15min
batchs synthetiques
device: cuda
```

Smoke test initial :

```json
{
  "load_seconds": 0.52,
  "inference_seconds": 0.1184,
  "batch": 5,
  "context_length": 512,
  "prediction_length": 96,
  "prediction_shape": [5, 96, 1],
  "gpu_memory_reserved_mib": 26.0
}
```

Benchmark tensor forward :

| Batch | Context | Horizon | Mean ms | Min ms | Max ms | Reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 96 | 3.445 | 3.417 | 3.546 | 26.0 |
| 10 | 512 | 96 | 3.578 | 3.541 | 3.831 | 26.0 |
| 50 | 512 | 96 | 4.300 | 4.276 | 4.340 | 32.0 |
| 100 | 512 | 96 | 7.033 | 6.518 | 7.317 | 36.0 |

Conclusion : TTM est extremement leger sur `z2`. Il est largement compatible
avec une boucle nowcasting frequente, meme avec beaucoup de stations/spots.

## Conclusions

`z2` est validee comme machine de test GPU pour la couche ML CorseWind.

Resultat principal :

```text
TTM: tres large marge pour inference 15 min
Chronos-2: faisable sur batchs modestes/moyens avec quantiles
```

Le goulot probable ne sera pas l'inference GPU, mais :

- preparation des donnees ;
- alignement multi-resolution 6 min / 15 min / 1 h ;
- interpolation NWP vers spots/stations ;
- controle qualite observations ;
- packaging d'un dataset pilote realiste.

## Prochaines etapes

1. Construire un dataset pilote avec 3 a 5 spots/stations.
2. Rejouer le benchmark avec vraies features 15 min.
3. Ajouter baselines persistance et NWP brut.
4. Tester TTM en fine-tuning court.
5. Tester Chronos-2 avec covariables futures issues d'AROME-PI.
6. Mesurer MAE, seuils windsurf et timing de session 11h-17h.
