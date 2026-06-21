# CorseWind.ai

Autonomous forecast engine and 2D viewer for high-resolution wind visualization over Corsica.

The current public scope is intentionally narrow:

- fetch the latest useful Meteo-France AROME forecast steps;
- optionally normalize a CNR-ISAC / MeteoHub MOLOCH 1.2 km wind bundle for display;
- optionally normalize an ItaliaMeteo / MeteoHub ICON-2I 2.2 km wind bundle for comparison;
- downscale selected session hours with WindNinja over Corsica at 50 m / 10 m output height;
- publish progressive Wind2D tile manifests as each forecast hour finishes;
- expose generated outputs for later Beacon Live integration.

Generated weather data, WindNinja cases, raster tiles, reports, and local secrets are not committed.

## Repository Layout

```text
scripts/
  run_forecast_update_engine.py        # polling/orchestration entrypoint
  build_arome_corsica_wind_layer.py    # Meteo-France AROME refresh
  build_moloch_corsica_wind_layer.py   # optional MOLOCH 1.2 km layer normalization
  build_icon2i_corsica_wind_layer.py   # optional ICON-2I 2.2 km layer normalization
  meteohub_opendata_client.py          # MeteoHub public bundle discovery helper
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
MOLOCH_SOURCE_URL=...
ICON2I_SOURCE_URL=...
CORSEWIND_HOST_ROOT=/absolute/path/to/CorseWind.ai
```

The engine launches WindNinja through Docker, so Docker must be running and the host must be able to pull/use the WindNinja/Katana image configured by the scripts.

MeteoHub GRIB decoding is optional. Install the extra GRIB/NetCDF stack only on hosts that will build MOLOCH or ICON-2I layers:

```bash
pip install -r requirements-moloch.txt
```

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
  --enable-moloch \
  --enable-icon2i \
  --session-days today \
  --session-past-tolerance-hours 0 \
  --windninja-parallel 6 \
  --windninja-runtime-min 60
```

When MOLOCH or ICON-2I are enabled without explicit `--*-lead-hours`, the
engine publishes every lead hour available in the MeteoHub source bundle.

Daemon mode:

```bash
python3 scripts/run_forecast_update_engine.py \
  --arome-poll-interval-sec 900 \
  --aromepi-poll-interval-sec 300 \
  --aromepi-stale-poll-interval-sec 60 \
  --aromepi-freshness-target-sec 900 \
  --aromepi-horizon-hours 24 \
  --aromepi-request-sleep-sec 1.3 \
  --fast-window-poll-interval-sec 60 \
  --enable-moloch \
  --moloch-poll-interval-sec 1800 \
  --enable-icon2i \
  --icon2i-poll-interval-sec 1800 \
  --windninja-parallel 6 \
  --windninja-runtime-min 60
```

The daemon tracks source state independently under `models.arome`,
`models.aromepi`, `models.moloch`, and `models.icon2i`. Wind2D JSON layers are
refreshed per source when due. Model layers publish all available forecast data
by default, except AROME-PI which publishes the next 24 hours at 15-minute
steps. WindNinja 50 m keeps its own session-hour selection and is rebuilt only
when the main AROME forcing run changes, or when `--force` is explicitly passed.
Each source also records `publication_history` so the scheduler can learn
real publication delays and switch to 60-second polling inside expected fast
publication windows.
MOLOCH and ICON-2I use MeteoHub bundle discovery when no explicit source URL is
provided, so their availability can be observed automatically.

## Docker

```bash
docker compose -f docker-compose.forecast-engine.yml up --build
```

The compose file mounts the repository and `/var/run/docker.sock` so the engine container can launch WindNinja containers on the host Docker daemon.

For Portainer Git stacks, set these environment variables in the stack:

```bash
METEOFRANCE_API_KEY=...
CORSEWIND_HOST_ROOT=/host/path/to/Portainer/checkout/CorseWind.ai
# Optional, disables WindNinja execution and WindNinja tile/data generation:
WINDNINJA_ENABLED=false
# Optional, starts the Wind2D web viewer service:
COMPOSE_PROFILES=wind2d-web
WIND2D_WEB_PORT=8769
```

`CORSEWIND_HOST_ROOT` must be the host-side checkout path used by Portainer. It cannot be `/app`, because child WindNinja containers are launched by the host Docker daemon and need host paths for bind mounts.

The compose file does not require a committed `.env` file. Locally, Docker Compose still reads `.env` automatically for variable interpolation. In Portainer, define the same variables in the stack environment.

By default the compose stack only starts the forecast engine. To expose Wind2D from the same stack, enable the `wind2d-web` profile:

```bash
COMPOSE_PROFILES=wind2d-web docker compose -f docker-compose.forecast-engine.yml up --build
```

The viewer is then available on:

```text
http://<host>:8769/visualizations/wind2d/
```

Use `WIND2D_WEB_PORT` to change the host port.

Generated data and diagnostics are stored in the mounted repository tree, especially `data/processed/`, `visualizations/wind2d/`, `reports/`, and `tmp/`. On pull/redeploy, the old container receives `SIGTERM`, the engine releases its lock, and the new container resumes from `data/processed/diagnostics/forecast_update_engine_state.json`.

When `WINDNINJA_ENABLED=false`, the engine still refreshes AROME, AROME-PI, MOLOCH, ICON-2I and compressed Wind2D model JSON files, but it skips WindNinja cases and WindNinja raster/data tile generation. The skipped AROME run is not marked as completed by WindNinja, so re-enabling `WINDNINJA_ENABLED=true` lets the engine compute the latest pending AROME forcing run.

The container exposes a healthcheck through:

```bash
python scripts/check_forecast_engine_health.py --max-status-age-sec 7200
```

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
visualizations/wind2d/aromepi-corsica-latest.json.gz
visualizations/wind2d/moloch-corsica-latest.json
visualizations/wind2d/icon2i-corsica-latest.json
visualizations/wind2d/windninja-corsica-data-50m/manifest.json
visualizations/wind2d/windninja-corsica-tiles-50m/manifest.json
```

The forecast engine writes `.json.gz` companions for the raw model payloads.
Wind2D loads those compressed payloads first and falls back to plain `.json`
if a compressed file is missing or unsupported.

## Documentation

See `docs/forecast_update_engine.md` for the full update process, selected forecast hours, generated artifacts, and Beacon Live handoff contract.

See `docs/corsewind_ml_nowcasting_blueprint.md` for the ML nowcasting blueprint: SAPHIR-inspired methodology, observations/NWP features, windsurf-focused metrics, model candidates, and operational roadmap.

See `docs/qes_winds_benchmark.md` for the optional QES-Winds GPU benchmark against the current WindNinja 50 m pipeline on Ajaccio and Bonifacio test zones.

See `docs/fasteddy_benchmark.md` for the optional FastEddy GPU LES smoke benchmark on the same Ajaccio and Bonifacio zones.

See `docs/fasteddy_arome_coupling_poc.md` for the AROME 3D inventory and parent-data POC required before FastEddy can be evaluated as a serious forecast engine.

See `docs/fasteddy_prod_like_pipeline.md` for the prod-like FastEddy real-case package generator, GeoSpec/SimGrid/ICBC handoff, and remaining production gates.

See `docs/fasteddy_icbc_contract.md`, `docs/fasteddy_wind2d_output_contract.md`, and `docs/fasteddy_linux_gpu_runbook.md` for the adapter contract, output contract, and GPU-machine run procedure.
