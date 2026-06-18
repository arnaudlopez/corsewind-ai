# QES-Winds GPU benchmark

This benchmark compares the current CorseWind WindNinja 50 m pipeline with QES-Winds on two focused test zones:

- Ajaccio bay;
- Bonifacio.

QES-Winds is not vendored in this repository. It is an external GPL-3.0 project and must be installed separately from <https://github.com/UtahEFD/QES-Public>. The public QES README describes QES as C++17/CUDA software requiring an NVIDIA GPU with compute capability 7.0 or higher for GPU acceleration. QES-Winds is a 3D diagnostic, mass-conserving wind solver based on a variational adjustment and SOR solution of a Poisson equation.

## Benchmark Goal

The benchmark answers practical product questions:

- Can QES-Winds run Ajaccio/Bonifacio domains faster than the current WindNinja Docker path?
- Does it expose 3D fields that are useful for near-surface windsurf interpretation?
- Are acceleration/devente corridors materially different from WindNinja?
- Is the install/runtime complexity acceptable for a future production worker?

## Prepare cases

Requires generated AROME and Copernicus DEM data:

```bash
python3 scripts/prepare_qes_winds_benchmark.py \
  --config benchmarks/qes_winds/benchmark_config.json \
  --lead-hour 0 \
  --horizontal-resolution-m 50
```

This writes comparable WindNinja and QES inputs under:

```text
data/processed/benchmarks/qes_winds/
```

## Run benchmark

WindNinja only:

```bash
python3 scripts/run_qes_winds_benchmark.py --engine windninja
```

QES-Winds GPU:

```bash
QES_WINDS_BIN=/path/to/QES-Public/build/qesWinds/qesWinds \
python3 scripts/run_qes_winds_benchmark.py --engine qes
```

Both:

```bash
QES_WINDS_BIN=/path/to/qesWinds \
python3 scripts/run_qes_winds_benchmark.py --engine both
```

If `nvidia-smi` or the QES binary is missing, QES cases are marked as skipped rather than failing the whole benchmark.

## Compare outputs

```bash
python3 scripts/compare_wind_benchmark_outputs.py
```

The comparator reads:

- WindNinja ASCII speed/direction grids;
- QES NetCDF `*_windsOut.nc` outputs when `netCDF4` is installed.

Install the optional comparator dependency with:

```bash
pip install -r requirements-benchmark.txt
```

## Fairness Notes

This first benchmark is intentionally conservative:

- WindNinja uses the same gridded AROME speed/direction forcing as the production pipeline.
- QES-Winds is initialized from representative AROME sensor samples because QES' public examples use sensor-based wind inputs for this path.
- Both engines use the same DEM sample, horizontal resolution, domain bounds, and target comparison height.

The first result should be treated as a feasibility benchmark, not final model validation. If QES is promising, the next step is a stricter forcing adapter or direct gridded initialization path for QES.
