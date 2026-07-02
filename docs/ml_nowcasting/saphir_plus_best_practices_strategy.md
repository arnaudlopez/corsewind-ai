# CorseWind Beyond SAPHIR Strategy

Date: 2026-06-30

## Purpose

This note compares the SAPHIR/Baggio design with current best practices for:

- deep learning on structured time-series/weather inputs;
- tabular models and tabular foundation models;
- probabilistic nowcasting and model evaluation.

The goal is not to clone SAPHIR. The goal is to keep the scientific ideas that
are strong, reject what is weak for our use case, and design a CorseWind
experiment that can plausibly beat both our current champion and a direct
SAPHIR-style replica.

## Executive Takeaway

SAPHIR gives us the right *weather learning framing*:

```text
NWP prior + recent observations + neighbor station dynamics -> local correction
```

But SAPHIR is not necessarily the best *modern ML implementation* for
CorseWind. Their architecture is sensible, but conservative: LSTM branches,
Conv2D/Conv3D NWP encoders, hourly cadence, nearest-neighbor station context,
and direct wind distribution prediction.

For CorseWind, the best strategy is a hybrid:

```text
structured deep model for temporal/spatial propagation
+ strong tabular residual models for wide heterogeneous features
+ regime/router layer for thermal, strong wind, spot/lead specialists
+ probabilistic/conformal calibration
```

This is more robust than betting everything on one neural architecture.

## What SAPHIR Gets Right

### 1. Correct Problem Framing

SAPHIR does not replace AROME/ARPEGE. It corrects the local forecast using
observations and context. This matches best practice for short-range weather
ML: NWP gives the synoptic and mesoscale prior; ML learns local errors.

For CorseWind, this remains non-negotiable. A pure ML model from observations
alone is unlikely to beat a correction model at +1h and beyond.

### 2. Structured Inputs

SAPHIR keeps source geometry:

- station history as a sequence;
- target station plus neighbor station histories;
- AROME as horizon x grid x variables;
- ARPEGE as horizon x vertical levels x grid x variables;
- compact context constants.

This is better than flattening everything at the beginning. Deep models need
structure to be useful.

### 3. Neighbor Histories

Their strongest design choice is the station tensor:

```text
target recent history + 10 neighbor recent histories
```

This is exactly the kind of signal that can capture propagation, delay,
upwind/downwind movement, and local onset.

Our current V1 missed this in full form. We have neighbor snapshots and context
features, but not enough neighbor time history.

### 4. U/V Wind Representation

Using wind components avoids angular discontinuity around 0/360 degrees. This
is best practice for wind direction in both tabular and neural models.

### 5. Multi-Horizon Output

SAPHIR predicts a vector of horizons in one forward pass. This forces the model
to learn a coherent trajectory, not four independent rows.

For us, this matters for navigability windows: the shape of the coming hour is
as important as the point estimate at +30 minutes.

### 6. Probabilistic Output

For windsurf, P50 alone is insufficient. We need:

- P10/P50/P90;
- probability of crossing 15 kt, 20 kt, gust thresholds;
- uncertainty and session confidence.

SAPHIR's probabilistic head is conceptually aligned with this.

## What SAPHIR Does Not Automatically Solve

### 1. The Architecture Is Not The Modern Ceiling

SAPHIR uses LSTM + Conv2D/Conv3D branches. That is reasonable, but not
necessarily state-of-the-art for our data.

Alternatives worth testing:

- temporal convolution / TCN for short local histories;
- Transformer-style attention across station histories;
- graph attention over station network;
- retrieval-augmented tabular/neural models;
- mixture-of-experts by regime, spot, and lead.

We should treat SAPHIR as a strong baseline, not as the final architecture.

### 2. Hourly Cadence Is Too Coarse For Us

SAPHIR's examples are hourly with horizons +1h to +6h. CorseWind has 6-minute
and 15-minute observations and cares deeply about +15/+30/+45/+60 minutes.

That changes the best architecture:

- persistence and recent error matter more;
- neighbor propagation delays are shorter;
- freshness and missingness are critical features;
- a model must update every 6-15 minutes without heavy retraining.

### 3. Nearest Neighbors Are Not Always Best Neighbors

SAPHIR selects nearest neighbors within a distance rule. For Corsica wind,
better context station selection should be dynamic:

- nearest station;
- upwind station under current NWP wind direction;
- coastal reference station;
- mountain/relief station;
- same-exposure station;
- thermal proxy station.

The right neighbor at 11:30 may not be the right neighbor at 15:30.

### 4. Compact Static Context Can Underfit Corsica

SAPHIR's `const` is compact and clean, but it ignores many signals that are
important for us:

- coastline orientation;
- maritime fetch;
- relief corridor;
- valley/venturi exposure;
- distance to sea and mountain;
- land-sea thermal contrast;
- sector-specific exposure.

We should not blindly reduce to 47 constants. The better approach is compact
but physically chosen static features, not thousands of noisy columns.

### 5. Exact Reproduction Is Blocked

The Zenodo V2 archive contains Ajaccio data and code, but not the partition
files used by the scripts. We can learn the design, but cannot fully reproduce
their benchmark split byte-for-byte from the archive alone.

## Additional 2024-2026 Literature Signals

The pasted review in the Codex attachment adds several useful recent papers.
They do not change the direction, but they sharpen the experiment design.

### Iwase/Takenawa 2026: LightGBM Is Not A Weak Baseline

`Improvements to the post-processing of weather forecasts using machine
learning and feature selection` is very relevant to our current question.

The paper uses JMA Mesoscale Model data around 18 Japanese sites, including
plains, mountains, and islands. It predicts precipitation, temperature, and wind
speed. The most important lessons for CorseWind:

- local NWP post-processing with LightGBM is a serious baseline;
- surrounding-grid NWP features matter;
- correlation/importance feature selection is part of the method, not an
  afterthought;
- results must be analysed by location and lead time;
- tested neural baselines did not automatically beat LightGBM.

This directly supports our current stance:

```text
do not demote HGB/LightGBM/Ridge until the structured neural model beats them
on exactly the same samples
```

It also suggests a strong tabular V2 baseline:

```text
surface NWP grid/offset features
+ pressure-level features
+ lead/month/hour
+ correlation pruning
+ per-target/per-lead LightGBM
```

### Dual-Resolution Wind Ensemble Post-Processing

Baran/Lakatos 2025/2026 compares raw and post-processed ECMWF wind-speed
ensembles at different resolutions.

Key lesson for CorseWind:

- probabilistic post-processing improves calibration and point accuracy;
- high spatial resolution can be more valuable than simply adding more ensemble
  members;
- high-resolution members can improve lower-resolution ensembles.

This maps cleanly to our stack:

```text
AROME-PI / AROME high-resolution features stay central
global models are context, not replacement
probabilistic calibration is a core product requirement
```

### AIFS-CRPS: Proper Probabilistic Loss Matters

AIFS-CRPS trains an AI weather ensemble with a CRPS-based objective. The key
lesson is not that we should use AIFS directly. The lesson is that minimizing
only RMSE/MSE is not the modern probabilistic forecasting endpoint.

For CorseWind neural models, this strengthens the case for testing:

- quantile loss;
- approximate CRPS;
- Gaussian/lognormal/Rice-like distribution heads;
- weighted losses around windsurf thresholds.

But it also warns us that uncertainty can be miscalibrated, so the final layer
still needs split/conformal calibration.

### GenCast / FGN / WeatherNext-Like Models

The useful signal is philosophical:

- forecast scenarios, not a single trajectory;
- optimize probabilistic skill, not only point RMSE;
- evaluate calibration and extremes.

For our operational nowcast, these models are too global/coarse to replace the
local correction engine. They can become synoptic/context features later.

### Aardvark Weather

Aardvark is conceptually important because it uses raw observations, encodes an
initial state, processes it, and decodes local forecasts. For CorseWind, the
practical near-term translation is:

```text
observation encoder
+ NWP encoder
+ temporal processor
+ spot decoder
```

This supports the `corsewind_saphir_dictionary_v2` direction, especially the
idea of a shared model with spot-specific decoders or calibration layers.

### Aurora / Large Earth-System Foundation Models

Aurora validates the broader idea of fine-tuning a large Earth-system model,
but it is too heavy for the immediate CorseWind loop. If we use foundation
models soon, Chronos/TimesFM/Moirai/TabPFN-style experiments are more
operationally realistic.

## Best Practices For Tabular Models

### 1. Keep GBDT As A First-Class Baseline

Modern tabular benchmarks still show that tree ensembles are extremely strong
on ordinary tabular data, especially when:

- feature count is high;
- data is heterogeneous;
- missingness patterns matter;
- nonlinear interactions are sparse;
- sample size is medium rather than huge.

For CorseWind, this means LightGBM/CatBoost/HGB/Ridge are not "old baselines".
They are production-grade competitors.

### 2. Use Tabular Models Where They Are Naturally Strong

Tabular models are excellent for:

- residual correction over NWP;
- static terrain/fetch features;
- spot/lead categorical effects;
- missingness/freshness indicators;
- regime flags;
- calibrated blend/router decisions;
- final residual calibration on top of neural outputs.

They are less natural for raw station-time tensors or NWP grids unless we
hand-engineer a lot of summary features.

### 3. Avoid Feature Explosion Without Data Scale

Our phys_v3 experience already showed the danger: more physical columns do not
guarantee better RMSE. Wide feature sets can dilute signal, increase variance,
and favor models that overfit local correlations.

Best practice is:

- add feature families behind ablation gates;
- track coverage and leakage;
- keep a compact champion feature set;
- add specialist features only where they improve spot/lead/regime metrics.

### 4. Test TabPFN/Tabular Foundation Models As Specialists

TabPFN-style models are especially interesting when tables are small or medium.
They should be tested, but with caution:

- they may be limited by row/feature count;
- they are not designed to ingest native time/weather tensors;
- they can be useful as a calibration or specialist model on compact tables;
- they may be useful for per-spot or per-regime small-data learning.

For CorseWind, TabPFN should not replace the whole pipeline. It should compete
on compact residual tables:

```text
NWP baseline + recent error summaries + compact static + regime flags
```

## Best Practices For Deep Learning

### 1. Give Deep Learning The Structure It Needs

Deep learning becomes useful when it receives structured signals that trees do
not naturally exploit:

- station histories;
- neighbor histories;
- spatial NWP patches;
- vertical atmospheric profiles;
- multi-horizon trajectories;
- learned embeddings by spot/station.

If we flatten early, we make the neural model fight the tabular model on the
tabular model's home field.

### 2. Use Multimodal Encoders, Not One Giant MLP

A good CorseWind neural model should have separate branches:

| Branch | Input | Suggested encoder |
| --- | --- | --- |
| target history | target obs and errors | GRU/TCN |
| neighbor histories | K stations x T x vars | station attention or graph attention |
| NWP surface | H x offsets/grid x vars | Conv/attention over horizon + space |
| vertical profile | H x pressure levels x vars | small Conv1D/attention |
| static context | compact terrain/fetch | MLP |
| lead/horizon | horizon vector | direct multi-output head |

This is SAPHIR-like, but modernized.

### 3. Predict Multi-Horizon Trajectories

Instead of independent rows:

```text
(sample, lead) -> target
```

prefer:

```text
sample -> [target_15, target_30, target_45, target_60]
```

This makes the model learn ramps, onset, decay, and consistent uncertainty.

### 4. Decide Direct Target Vs Residual Empirically

SAPHIR predicts direct wind speed. Our tabular champion predicts residual
corrections.

Best practice is not dogmatic. We should compare:

- direct wind/gust target;
- residual target: observation - NWP;
- two-head model: direct target plus residual target;
- error-propagation target: current observed error -> future error.

For short-term NWP correction, residual/error evolution often has the strongest
physical meaning.

### 5. Use Probabilistic Training, Then Calibrate

Probabilistic heads are useful, but raw neural uncertainty is often
miscalibrated. Best practice is:

- train quantiles or distribution parameters;
- evaluate pinball loss, CRPS-like metrics, Brier score;
- calibrate intervals with conformal or split calibration on recent data;
- report threshold probabilities, not only RMSE.

For windsurf decisions, a slightly worse RMSE but much better threshold
calibration can be more valuable.

### 6. Do Not Overtrust Foundation Time-Series Models

Chronos, TimesFM, and Moirai are valuable baselines because they are pretrained
probabilistic forecasters. But for CorseWind:

- they mostly see univariate or generic multivariate sequences;
- they do not naturally know local topography, NWP errors, and station roles;
- zero-shot performance is not enough for a high-precision local nowcast.

Use them as:

- sequence baselines;
- residual-sequence forecasters;
- uncertainty/ensemble members;
- possible teachers or feature generators.

Do not assume they will beat a well-built hybrid NWP correction model.

## Proposed "Better Than SAPHIR" Architecture

### Stage A: Strong Tabular Production Baseline

Keep and improve the tabular residual champion:

```text
features:
  NWP baseline
  recent observed error summaries
  compact static terrain/fetch
  weather regime flags
  context station snapshots
  missingness/freshness
models:
  LightGBM / CatBoost / HGB / Ridge
outputs:
  wind_mean residual
  gust residual
```

This remains the production floor. Any deep model must beat it on the same
time-forward split.

### Stage B: SAPHIR-Dictionary V2 Deep Model

Build a faithful structured dataset:

```text
sample = spot_id + issue_time
station_tensor = target + neighbor histories
nwp_surface_tensor = horizon x offsets/grid x vars
vertical_tensor = horizon x levels x vars
const = compact local + neighbor + terrain/fetch context
target = [15, 30, 45, 60] wind/gust
```

Train a modernized model:

```text
station sequence encoder
+ neighbor attention/graph encoder
+ NWP horizon-space encoder
+ vertical profile encoder
+ static MLP
-> multi-horizon residual/direct/probabilistic heads
```

### Stage C: Router / Specialist Layer

Instead of forcing one global model to win everywhere, train a selector:

```text
if thermal regime and coastal spot:
    use thermal specialist / blend weight
elif strong synoptic wind:
    use NWP-heavy correction
elif lead <= 30:
    use persistence/error-propagation-heavy correction
else:
    use NWP + context correction
```

The router can be tabular and should be evaluated by:

- global RMSE;
- spot/lead RMSE;
- thermal day RMSE;
- onset timing error;
- threshold Brier score.

### Stage D: Probabilistic Calibration

Final product outputs:

```text
P10/P50/P90 wind mean
P10/P50/P90 gust
P(wind > 15 kt)
P(wind > 20 kt)
P(gust > threshold)
session confidence
```

Use calibration on time-forward validation, and keep a live recalibration layer
for recent residual drift.

## What To Test Next

Priority order:

1. Build `corsewind_saphir_dictionary_v2`.
2. Add true context station histories.
3. Train a small multi-horizon neural model on the medium sample set.
4. Compare against flat Ridge/HGB on exactly the same samples.
5. Add a tabular blend/calibrator:
   - raw NWP;
   - tabular residual;
   - neural residual;
   - persistence/error-propagation.
6. Test TabPFN/TabPFN-2.5 on compact residual tables, not on the full raw
   tensor dataset.
7. Test Chronos/TimesFM/Moirai on residual sequences as baselines or ensemble
   members.
8. Add an Iwase/Takenawa-style LightGBM baseline with correlation-pruned
   surrounding-grid/offset features.
9. Only scale to full data once the medium benchmark beats the tabular floor.

## Anti-Patterns To Avoid

- Copying SAPHIR exactly without adapting cadence and operational split.
- Adding more static columns without ablation proof.
- Judging a model by global RMSE only.
- Letting eval data leak into normalization, feature selection, or calibration.
- Comparing models on different sample sets.
- Training one giant neural model before validating the data representation.
- Replacing NWP instead of correcting it.
- Using foundation models as magic baselines without local calibration.

## Acceptance Gates

A new approach is worth scaling only if it passes these gates:

1. Same-sample comparison beats current flat Ridge/HGB on wind mean RMSE.
2. It does not worsen gust RMSE materially.
3. It improves at least one hard regime:
   - thermal onset;
   - +45/+60 min;
   - high wind;
   - difficult spots.
4. It has no leakage under audit.
5. It improves or preserves threshold calibration.

## Final Position

The strongest path is not "deep learning instead of tabular" or "SAPHIR instead
of our current model".

The strongest path is:

```text
SAPHIR data structure
+ modernized deep encoders for dynamic propagation
+ tabular residual specialists for heterogeneous local corrections
+ probabilistic/conformal calibration for decision quality
```

This is the most credible way to do better than SAPHIR for CorseWind.

## References

- SAPHIR/Baggio reference: https://arxiv.org/html/2503.18797v2
- Iwase/Takenawa 2026 NWP post-processing and feature selection:
  https://arxiv.org/html/2604.19340v1
- Baran/Lakatos dual-resolution wind ensemble post-processing:
  https://arxiv.org/abs/2506.15578
- AIFS-CRPS:
  https://www.nature.com/articles/s44387-026-00073-7
- Aardvark Weather:
  https://www.nature.com/articles/s41586-025-08897-0
- GenCast:
  https://www.nature.com/articles/s41586-024-08252-9
- Aurora:
  https://www.nature.com/articles/s41586-025-09005-y
- Grinsztajn et al., tree models vs deep learning on tabular data:
  https://arxiv.org/abs/2207.08815
- TabR retrieval-augmented tabular deep learning:
  https://arxiv.org/abs/2307.14338
- TabPFN Nature paper:
  https://www.nature.com/articles/s41586-024-08328-6
- TabPFN-2.5 report:
  https://arxiv.org/abs/2511.08667
- Chronos:
  https://arxiv.org/abs/2403.07815
- TimesFM:
  https://proceedings.mlr.press/v235/das24c.html
- Moirai:
  https://arxiv.org/abs/2402.02592
- Moirai-MoE:
  https://arxiv.org/abs/2410.10469
