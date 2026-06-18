# CorseWind.ai

Autonomous forecast engine and 2D viewer for high-resolution wind visualization over Corsica.

The current public scope is intentionally narrow:

- fetch the latest useful Meteo-France AROME forecast steps;
- downscale selected session hours with WindNinja over Corsica at 50 m / 10 m output height;
- publish progressive Wind2D tile manifests as each forecast hour finishes;
- expose generated outputs for later Beacon Live integration.

Generated weather data, WindNinja cases, raster tiles, reports, and local secrets are not committed.

## Repository Layout

```text
scripts/
  run_forecast_update_engine.py        # polling/orchestration entrypoint
  build_arome_corsica_wind_layer.py    # Meteo-France AROME refresh
  prepare_corsica_windninja_tiles.py   # 50 m WindNinja tile case generation
  run_corsica_windninja_batch.py       # parallel WindNinja batch runner
  run_windninja_cases_docker.py        # Docker/Katana WindNinja launcher
  build_corsica_windninja_raster_tiles.py
  serve_wind2d_compressed.py

visualizations/wind2d/
  index.html
  wind2d.js
  wind2d.css

docs/
  forecast_update_engine.md
  qes_winds_benchmark.md
  fasteddy_benchmark.md
  fasteddy_arome_coupling_poc.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env`:

```bash
METEOFRANCE_API_KEY=...
CORSEWIND_HOST_ROOT=/absolute/path/to/CorseWind.ai
```

The engine launches WindNinja through Docker, so Docker must be running and the host must be able to pull/use the WindNinja/Katana image configured by the scripts.

## Run One Forecast Cycle

Dry-run:

```bash
python3 scripts/run_forecast_update_engine.py --once --dry-run
```

Production-like single cycle for the remaining useful windsurf window:

```bash
python3 scripts/run_forecast_update_engine.py \
  --once \
  --force \
  --session-days today \
  --session-past-tolerance-hours 0 \
  --windninja-parallel 6 \
  --windninja-runtime-min 60
```

Daemon mode:

```bash
python3 scripts/run_forecast_update_engine.py \
  --poll-interval-sec 900 \
  --windninja-parallel 6 \
  --windninja-runtime-min 60
```

## Docker

```bash
docker compose -f docker-compose.forecast-engine.yml up --build
```

The compose file mounts the repository and `/var/run/docker.sock` so the engine container can launch WindNinja containers on the host Docker daemon.

## Wind2D Viewer

```bash
python3 scripts/serve_wind2d_compressed.py --port 8769
```

Open:

```text
http://127.0.0.1:8769/visualizations/wind2d/
```

The viewer expects generated files such as:

```text
visualizations/wind2d/arome-corsica-latest.json
visualizations/wind2d/windninja-corsica-data-50m/manifest.json
visualizations/wind2d/windninja-corsica-tiles-50m/manifest.json
```

## Documentation

See `docs/forecast_update_engine.md` for the full update process, selected forecast hours, generated artifacts, and Beacon Live handoff contract.

See `docs/qes_winds_benchmark.md` for the optional QES-Winds GPU benchmark against the current WindNinja 50 m pipeline on Ajaccio and Bonifacio test zones.

See `docs/fasteddy_benchmark.md` for the optional FastEddy GPU LES smoke benchmark on the same Ajaccio and Bonifacio zones.

See `docs/fasteddy_arome_coupling_poc.md` for the AROME 3D inventory and parent-data POC required before FastEddy can be evaluated as a serious forecast engine.

See `docs/fasteddy_prod_like_pipeline.md` for the prod-like FastEddy real-case package generator, GeoSpec/SimGrid/ICBC handoff, and remaining production gates.
