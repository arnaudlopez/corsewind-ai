# FastEddy GPU benchmark

This benchmark adds FastEddy as an early candidate beside WindNinja and QES-Winds.

FastEddy is an NSF NCAR resident-GPU Large Eddy Simulation model for atmospheric boundary layer flow. It is not vendored in this repository. Install it separately from <https://github.com/NCAR/FastEddy-model>. The official documentation describes a build requiring a C compiler, MPI and CUDA, with the executable produced in `SRC/FEMAIN/FastEddy`.

## Why Test It Now

FastEddy is much more complex than the current WindNinja pipeline, but it is one of the few open GPU-first atmospheric microscale models that could reveal useful 3D coastal flow structure. We want to know early whether it is promising before over-investing in another solver path.

## Benchmark Level

The first CorseWind FastEddy benchmark is a smoke/feasibility benchmark:

- same Ajaccio and Bonifacio test zones as the QES benchmark;
- Copernicus DEM sampled at 50 m;
- AROME mean `u/v` injected as idealized geostrophic forcing;
- terrain binary written in FastEddy-compatible topography format;
- optional GIS NetCDF and GeoSpec/SimGrid config files prepared for the real-data workflow;
- short runtime by default, intended to reveal build/runtime/output viability.

This is not yet a final AROME-to-FastEddy production coupling. The stricter version would require generating proper WRF-like or FastEddy parent boundary conditions.

## Prepare Cases

Requires generated AROME and Copernicus DEM data:

```bash
python3 scripts/prepare_fasteddy_benchmark.py \
  --config benchmarks/fasteddy/benchmark_config.json \
  --lead-hour 0 \
  --horizontal-resolution-m 50
```

This writes:

```text
data/processed/benchmarks/fasteddy/
```

## Run FastEddy

Single-rank smoke run:

```bash
FASTEDDY_BIN=/path/to/FastEddy \
python3 scripts/run_fasteddy_benchmark.py
```

Dry-run:

```bash
FASTEDDY_BIN=/path/to/FastEddy \
python3 scripts/run_fasteddy_benchmark.py --dry-run --allow-no-gpu
```

Multi-rank run:

```bash
FASTEDDY_BIN=/path/to/FastEddy \
python3 scripts/run_fasteddy_benchmark.py --mpirun-bin mpirun
```

The rank count is configured in `benchmarks/fasteddy/benchmark_config.json`.

## Compare Outputs

```bash
pip install -r requirements-benchmark.txt
python3 scripts/compare_fasteddy_benchmark_outputs.py
```

The comparator looks for FastEddy output files under each case `output/` directory and summarizes `u/v` wind speed if NetCDF output is available.

## Decision Criteria

FastEddy should stay in the roadmap only if the first GPU machine test answers yes to most of these:

- build can be automated without heavy HPC-specific assumptions;
- Ajaccio/Bonifacio smoke cases run without numerical instability;
- output can be converted to the Wind2D raster contract;
- runtime is plausible for selected spot windows, even if not for all Corsica hours;
- vertical velocity/turbulence information provides visible value over WindNinja/QES.
