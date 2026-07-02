# Reliable Champion Search - 2026-07-01

## Goal

Find a candidate that reliably beats the current CorseWind champions, without temporal leakage, on same-key samples:

- key: `spot_id + issue_time_utc + target_time_utc + lead_time_minutes`
- wind mean champion: `scale070`
- wind mean official gate: RMSE `1.268019`, MAE `0.930465`
- gust champion: `new_scale070_gust_recipe`
- gust official gate: RMSE `1.484221`, MAE `1.073906`

A candidate can be promoted only if it beats the champion on validation, holdout, and full same-key, and if full same-key RMSE beats the official gate.

## Experiments

### v_next versus champion, cautious static blends

Report:

- `docs/ml_nowcasting/vnext_champion_blend_results_2026h1_v1.md`
- `docs/ml_nowcasting/vnext_champion_blend_decision_2026_07_01.md`

Result:

- wind same-key champion RMSE: `1.298842`
- wind v_next RMSE: `1.344099`
- wind best static blend: `blend_all_a0.2_clip1`, RMSE `1.292466`
- gust same-key champion RMSE: `1.538418`
- gust v_next RMSE: `1.592368`
- gust best static blend: `blend_all_a0.2_clip2`, RMSE `1.527820`

Decision: useful signal exists, but static blends do not beat official gates.

### Trust correction, HGB split March

Reports:

- `docs/ml_nowcasting/vnext_trust_correction_hgb_split_mar_v2.json`
- `docs/ml_nowcasting/vnext_trust_correction_hgb_split_mar_v3.json`

Split:

- train: before `2026-03-01`
- validation: `2026-03-01` to `2026-04-01`
- holdout: after `2026-04-01`

Selected-by-validation result:

- wind selected `residual_hist_gradient_boosting_scale0.5_clip1.5`
- wind selected full RMSE: `1.236669`
- wind selected validation delta: `-0.049813`
- wind selected holdout delta: `+0.007757`
- decision: do not promote selected-by-validation candidate, because holdout is slightly worse

Strict candidate found in v3:

- wind candidate `residual_hist_gradient_boosting_scale0.3_clip2`
- validation RMSE: `1.307673`
- holdout RMSE: `1.193597`
- full RMSE: `1.253970`
- full MAE: `0.934700`
- beats champion on validation, holdout, and full RMSE
- beats official full RMSE gate `1.268019`

Current status: first reliable wind-mean candidate found on one split. Needs confirmation on another temporal split before promotion.

Gust status:

- selected HGB candidate improves full RMSE strongly, but still fails strict promotion due holdout/official-gate criteria.
- no strict gust candidate yet.

## Next Checks

1. Run HGB split April with candidate table:
   - train before `2026-04-01`
   - validation April
   - holdout May-June
2. Check whether the conservative residual family still appears as strict candidate.
3. If confirmed, package wind candidate as `scale070_vnext_trust_residual_hgb_scale030_clip2` or equivalent.
4. Continue gust search separately, likely with a more conservative or gust-specific objective.

## Follow-up Results

### HGB split April, full v_next features

Report:

- `docs/ml_nowcasting/vnext_trust_correction_hgb_split_apr_v2.json`

Result:

- wind mean: no strict candidate. Conservative candidates beat validation and full, but holdout is slightly worse by roughly `+0.0035 m/s` for `scale0.3_clip2`.
- gust: strict candidates exist. Best strict candidate:
  - `residual_hist_gradient_boosting_scale0.3_clip2`
  - validation RMSE `1.349527`
  - holdout RMSE `1.369171`
  - full RMSE `1.475337`
  - official gate `1.484221`

### HGB split April, wind fine scales

Report:

- `docs/ml_nowcasting/vnext_trust_correction_hgb_split_apr_wind_fine_v1.json`

Result:

- no strict wind candidate.
- the failure is very small: `scale0.18_clip2` gets full RMSE `1.267033`, under the official gate, but holdout remains `+0.000675 m/s` worse than champion.

### HGB split April, horizon-gated candidates

Report:

- `docs/ml_nowcasting/vnext_trust_correction_hgb_split_apr_wind_leadgate_v1.json`

Result:

- no strict wind candidate.
- lead gates did not change top candidates, because the evaluated samples are already effectively within the tested short lead range.

### HGB prediction-only mode

Patch:

- `scripts/ml_dataset/train_vnext_trust_correction.py` now supports `--feature-mode prediction_only`.

Reports:

- `docs/ml_nowcasting/vnext_trust_correction_hgb_split_mar_predonly_v1.json`
- `docs/ml_nowcasting/vnext_trust_correction_hgb_split_apr_predonly_v1.json`

Result:

- prediction-only reduces overfit and makes the correction more stable.
- March wind mean strict candidate:
  - `residual_hist_gradient_boosting_scale0.5_clip2`
  - validation RMSE `1.297834`
  - holdout RMSE `1.190722`
  - full RMSE `1.261666`
  - official gate `1.268019`
- April wind mean still misses strict promotion by a small holdout margin:
  - `residual_hist_gradient_boosting_scale0.5_clip2`
  - validation RMSE `1.151897`
  - holdout RMSE `1.208142`
  - full RMSE `1.266037`
  - holdout delta `+0.002459 m/s`
- April gust nearly passes the official gate in prediction-only mode:
  - `residual_hist_gradient_boosting_scale0.5_clip2`
  - validation RMSE `1.343737`
  - holdout RMSE `1.368488`
  - full RMSE `1.484729`
  - official gate `1.484221`

Current interpretation:

- The useful signal is real.
- The safest architecture is not the full 1400-feature trust model; it is a constrained residual corrector around the champion.
- Wind mean has a strict candidate on March and near-strict on April.
- Gust has a strict candidate on April and near-strict on March/April depending on feature mode.
- A third temporal split is needed before promotion.

### HGB prediction-only split May with saved artifacts

Report:

- `docs/ml_nowcasting/vnext_trust_correction_hgb_split_may_predonly_artifact_v1.json`

Remote artifacts:

- wind mean model: `/srv/data/corsewind/ml_dataset/benchmarks/vnext_trust_correction_2026h1_hgb_split_may_predonly_artifact_v1/wind_mean_hist_gradient_boosting_best_strict/residual_model.joblib`
- wind mean metadata: `/srv/data/corsewind/ml_dataset/benchmarks/vnext_trust_correction_2026h1_hgb_split_may_predonly_artifact_v1/wind_mean_hist_gradient_boosting_best_strict/metadata.json`
- gust model: `/srv/data/corsewind/ml_dataset/benchmarks/vnext_trust_correction_2026h1_hgb_split_may_predonly_artifact_v1/gust_hist_gradient_boosting_best_strict/residual_model.joblib`
- gust metadata: `/srv/data/corsewind/ml_dataset/benchmarks/vnext_trust_correction_2026h1_hgb_split_may_predonly_artifact_v1/gust_hist_gradient_boosting_best_strict/metadata.json`

Both saved artifacts use:

- feature mode: `prediction_only`
- candidate: `residual_hist_gradient_boosting_scale0.5_clip2`
- candidate type: `residual_regressor`
- correction: `champion + 0.5 * clip(predicted_residual_vs_champion, -2.0, +2.0)`

Wind mean result on split May:

- champion validation RMSE: `1.238669`
- candidate validation RMSE: `1.230983`
- champion holdout RMSE: `1.156616`
- candidate holdout RMSE: `1.151990`
- champion full same-key RMSE: `1.298842`
- candidate full same-key RMSE: `1.260096`
- official RMSE gate: `1.268019`
- decision: strict reliable candidate on this latest split

Gust result on split May:

- champion validation RMSE: `1.402789`
- candidate validation RMSE: `1.389041`
- champion holdout RMSE: `1.324122`
- candidate holdout RMSE: `1.310620`
- champion full same-key RMSE: `1.538418`
- candidate full same-key RMSE: `1.479031`
- official RMSE gate: `1.484221`
- decision: strict reliable candidate on this latest split

Cross-split caveat:

- March: wind mean is strict in prediction-only mode; gust is just above the official gate.
- April: gust is strict with full features; wind mean is near-strict but misses holdout by `+0.002459 m/s` in prediction-only mode.
- May: both wind mean and gust are strict in prediction-only mode.

Promotion decision:

- Promote this as a saved challenger/shadow champion, not as an irreversible replacement yet.
- It is good enough to run in shadow/live inference and compare against fresh observations.
- Direct champion replacement should wait for either:
  - fresh live holdout confirming the May behavior, or
  - a rolling evaluation policy that accepts the April `+0.002459 m/s` miss as measurement-noise-level tolerance.

Implementation changes:

- `scripts/ml_dataset/train_vnext_trust_correction.py`
  - added `--feature-mode prediction_only`
  - added `lightgbm` as an optional model family for benchmarking
  - added `--residual-max-lead-minutes` benchmark candidates
  - added `--save-best-strict-artifact` to persist the best strict residual challenger
