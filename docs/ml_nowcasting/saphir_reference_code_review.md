# SAPHIR / Baggio et al. Reference Code Review

Date: 2026-06-30

## Source

- Paper reference: `arXiv:2503.18797v2`
- Zenodo concept DOI: `10.5281/zenodo.15222910`
- Correct code/data record inspected: Zenodo record `20327672`, version `v 2.0`
- Downloaded archive on z2: `/srv/data/corsewind/reference/saphir_predict/v2/Zenodo2026.zip`
- Verified MD5: `61add72af3e68f51f3df835cd22164e3`
- Extracted code/configs: `/srv/data/corsewind/reference/saphir_predict/v2/extracted_code/Zenodo_BM_2026`

Note: an older v1 archive was initially downloaded at `/srv/data/corsewind/reference/saphir_predict/spred.tar.gz`. The v2 archive above is the one to use for reference.

Detailed follow-up analysis: `docs/ml_nowcasting/saphir_zenodo_v2_deep_analysis.md`.

## Important Files

- `ARCHITECTURE.md`: high-level workflow and package architecture.
- `script_examples/loop_build_dics.py`: dataset/dictionary construction parameters.
- `script_examples/benchmark_wind_Corsica.py`: Corsica wind benchmark entrypoint.
- `script_examples/learn_all.py`: simple deterministic training example.
- `src/saphir_predict/dataproc/dic_from_saphir_V2.py`: SAPHIR NetCDF to HDF5 dictionary builder.
- `src/saphir_predict/data_generators/partition_generation.py`: train/validation/test partition helpers.
- `src/saphir_predict/data_generators/data_generation_pytorch.py`: PyTorch dataset and dataloader.
- `src/saphir_predict/models/neural_network_utils_pytorch.py`: deterministic multi-source neural architecture.
- `src/saphir_predict/models/neural_network_utils_pytorch_proba.py`: probabilistic wind loss and distribution utilities.
- `src/saphir_predict/storage/setup_files/setup_wind.yaml`: wind hyperparameters.

## Dataset Construction Observed In V2

The wind dictionary builder uses:

- Period: `2016-01-01` to `2018-12-31`
- Region: Corsica region code `20`
- Native sample cadence for learning examples: `time_step = 60` minutes
- Forecast horizons: `[1, 2, 3, 4, 5, 6]` hours
- Observation past window: `past = 6`, which produces 7 station timesteps including current time
- Station context: target station plus `nNeigh = 10` nearest neighboring stations
- Output target: `wind_speed_station`
- NWP sources: AROME and ARPEGE
- Static context: hour, day of year, lat, lon, altitude, plus relative neighbor geometry

Wind speed/direction are converted to U/V components before normalization:

- Station fields after conversion: `wind_eastward_`, `wind_northward_`, `temperature_`
- AROME fields after conversion: `relative_humidity_`, `wind_eastward_U`, `wind_northward_V`, `temperature_2m_`, `sealevel_pressure_`
- ARPEGE fields after conversion: `wind_eastward_U`, `wind_northward_V`, `temperature_`, `pressure_`

Example Ajaccio dictionary metadata from `features_20004002.pkl`:

- Station tensor normalization shape: `(7, 33)`
- AROME tensor normalization shape: `(6, 11, 11, 5)`
- ARPEGE tensor normalization shape: `(6, 7, 5, 5, 4)`

Interpretation:

- `station`: 7 timesteps x 33 channels = 11 stations x 3 variables
- `arome`: 6 forecast horizons x 11x11 spatial grid x 5 variables
- `arpege`: 6 forecast horizons x 7 pressure/vertical levels x 5x5 grid x 4 variables

## Split And Leakage Notes

The v2 code builds fixed global train/validation/test key lists:

- `id_training_full.pkl.gz`
- `id_validation_full.pkl`
- `id_test_full.pkl`

Samples are string keys like `station_yyyymmdd_index`. The model intersects each experiment key list with those fixed partitions.

This is useful for reproducibility, but CorseWind should not blindly copy the random/day split. For our live forecasting objective, we should keep a strict time-forward validation/test split so future dates never influence training, normalization, calibration, or feature selection.

## Architecture Notes

The neural model is a hybrid multi-source architecture:

- Station time series: LSTM with local station context and neighbor context.
- AROME fields: 2D convolution per horizon/time slice, then LSTM over horizons.
- ARPEGE fields: 3D/vertical-spatial encoder, then temporal aggregation.
- Constants/context: encoded and concatenated with source encodings.
- Final aggregation: dense block, then deterministic or probabilistic head.

The probabilistic wind model uses a Rice-like likelihood with Gauss-Hermite quadrature, then derives means/quantiles/probabilities for thresholds. This matches our target direction: predict a distribution, not just a mean.

## Lessons For CorseWind

What we should reuse conceptually:

- Multi-source samples, not a flat single-row-only table.
- Station target plus selected context stations.
- Explicit current/recent station history.
- U/V wind representation for station and NWP winds.
- Spatial AROME grids around the spot, not only nearest-point NWP.
- Vertical ARPEGE/pressure context around the point.
- Per-source normalization computed only from training keys.
- Probabilistic outputs and threshold metrics.

What we should adapt:

- Cadence: SAPHIR uses hourly examples; CorseWind needs 15 min and possibly 6 min observation updates.
- Horizons: SAPHIR uses 1-6 h; CorseWind needs +15/+30/+45/+60 min plus 2-6 h.
- Splits: use strict chronological splits for our operational setup.
- Inputs: include our live residual features, AROME-PI, spot static terrain/fetch context, and Beacon/Winds-Up observations.
- Targets: train both wind mean and gusts when gust observations are available.

## Next Engineering Step

Build a `saphir_style_sequence_dataset` for CorseWind:

1. Generate samples keyed by `spot_id`, `target_time`, and `lead_minutes`.
2. Include target spot observations over the last 1-6 h at native resolution when available.
3. Include context stations with distance, bearing, upwind score, altitude delta, freshness, and missingness masks.
4. Include AROME/AROME-PI grids around the spot for each lead time.
5. Include vertical pressure-level/context features when available.
6. Normalize using train-only statistics.
7. Benchmark:
   - current LightGBM residual champion,
   - SAPHIR-style neural deterministic model,
   - SAPHIR-style probabilistic model,
   - sequence foundation models only after the dataset is shaped correctly.
