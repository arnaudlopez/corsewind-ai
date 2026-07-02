# v_next Champion Blend Benchmark

Generated: `2026-07-01T19:56:05.257743Z`
Selection split: `2026-04-01T00:00:00Z`

## Summary

| Target | Rows | Champion RMSE | v_next RMSE | Oracle RMSE | Selected holdout RMSE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wind_mean` | 24276 | 1.298842 | 1.344099 | 1.174058 | 1.188602 | `do_not_promote` |
| `gust` | 24276 | 1.538418 | 1.592368 | 1.374317 | 1.361503 | `do_not_promote` |

## Wind Mean

- rows: `24276`
- calibration rows before split: `11425`
- holdout rows after split: `12851`
- max absolute actual diff between champion and v_next rows: `0.0`

Baseline metrics:

| Prediction | RMSE | MAE | Bias | Count |
| --- | ---: | ---: | ---: | ---: |
| `champion` | 1.298842 | 0.967489 | 0.044222 | 24276 |
| `raw` | 2.161885 | 1.651248 | 0.328109 | 24276 |
| `vnext` | 1.344099 | 0.999217 | 0.143419 | 24276 |
| `pair oracle, not deployable` | 1.174058 | 0.838035 | 0.068534 | 24276 |

Selected by calibration:

- variant: `blend_all_a0.2_clip1`
- champion calibration RMSE: `1.407252`
- calibration RMSE: `1.400116`
- champion holdout RMSE: `1.194227`
- holdout RMSE: `1.188602`
- holdout gain vs champion: `0.471016%`
- full eval RMSE: `1.292466`
- promotion verdict: `do_not_promote`

Best full-evaluation variants:

| Variant | RMSE | MAE | Bias | Gate share | Alpha | Clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `blend_all_a0.2_clip1` | 1.292466 | 0.963399 | 0.059532 | 1.0 | 0.2 | 1.0 |
| `blend_all_a0.2_clip2` | 1.292495 | 0.963948 | 0.062552 | 1.0 | 0.2 | 2.0 |
| `blend_all_a0.2_clip0.75` | 1.29271 | 0.963339 | 0.057854 | 1.0 | 0.2 | 0.75 |
| `blend_all_a0.2_clip0.5` | 1.293389 | 0.963612 | 0.055086 | 1.0 | 0.2 | 0.5 |
| `blend_all_a0.15_clip2` | 1.293497 | 0.964386 | 0.05797 | 1.0 | 0.15 | 2.0 |
| `blend_all_a0.15_clip1` | 1.293636 | 0.964072 | 0.055704 | 1.0 | 0.15 | 1.0 |
| `blend_all_a0.15_clip0.75` | 1.293888 | 0.964077 | 0.054446 | 1.0 | 0.15 | 0.75 |
| `blend_hotspots_a0.2_clip2` | 1.294306 | 0.964292 | 0.048337 | 0.43714 | 0.2 | 2.0 |
| `blend_vnext_below_champion_a0.2_clip2` | 1.294487 | 0.963288 | 0.022314 | 0.433762 | 0.2 | 2.0 |
| `blend_all_a0.15_clip0.5` | 1.294496 | 0.964346 | 0.05237 | 1.0 | 0.15 | 0.5 |

Best holdout variants:

| Variant | RMSE | MAE | Bias | Gate share | Alpha | Clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `blend_all_a0.2_clip2` | 1.188516 | 0.910736 | 0.052625 | 1.0 | 0.2 | 2.0 |
| `blend_all_a0.2_clip1` | 1.188602 | 0.910718 | 0.051472 | 1.0 | 0.2 | 1.0 |
| `blend_all_a0.2_clip0.75` | 1.18867 | 0.9107 | 0.050785 | 1.0 | 0.2 | 0.75 |
| `blend_all_a0.2_clip0.5` | 1.188959 | 0.91088 | 0.049851 | 1.0 | 0.2 | 0.5 |
| `blend_all_a0.15_clip2` | 1.189555 | 0.911546 | 0.051693 | 1.0 | 0.15 | 2.0 |
| `blend_all_a0.15_clip1` | 1.18968 | 0.911588 | 0.050828 | 1.0 | 0.15 | 1.0 |
| `blend_all_a0.15_clip0.75` | 1.189763 | 0.911623 | 0.050313 | 1.0 | 0.15 | 0.75 |
| `blend_all_a0.15_clip0.5` | 1.190042 | 0.911797 | 0.049612 | 1.0 | 0.15 | 0.5 |
| `blend_vnext_below_champion_a0.2_clip2` | 1.190228 | 0.911514 | 0.025409 | 0.502996 | 0.2 | 2.0 |
| `blend_vnext_below_champion_a0.2_clip1` | 1.1903 | 0.911591 | 0.025538 | 0.502996 | 0.2 | 1.0 |

## Gust

- rows: `24276`
- calibration rows before split: `11425`
- holdout rows after split: `12851`
- max absolute actual diff between champion and v_next rows: `0.0`

Baseline metrics:

| Prediction | RMSE | MAE | Bias | Count |
| --- | ---: | ---: | ---: | ---: |
| `champion` | 1.538418 | 1.124678 | 0.081652 | 24276 |
| `raw` | 3.918143 | 3.01038 | 2.577239 | 24276 |
| `vnext` | 1.592368 | 1.164728 | 0.16129 | 24276 |
| `pair oracle, not deployable` | 1.374317 | 0.963252 | 0.087785 | 24276 |

Selected by calibration:

- variant: `blend_all_a0.2_clip2`
- champion calibration RMSE: `1.706548`
- calibration RMSE: `1.695511`
- champion holdout RMSE: `1.371749`
- holdout RMSE: `1.361503`
- holdout gain vs champion: `0.74693%`
- full eval RMSE: `1.52782`
- promotion verdict: `do_not_promote`

Best full-evaluation variants:

| Variant | RMSE | MAE | Bias | Gate share | Alpha | Clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `blend_all_a0.2_clip2` | 1.52782 | 1.12052 | 0.094554 | 1.0 | 0.2 | 2.0 |
| `blend_all_a0.2_clip1` | 1.528472 | 1.119596 | 0.09118 | 1.0 | 0.2 | 1.0 |
| `blend_all_a0.2_clip0.75` | 1.52916 | 1.119463 | 0.08931 | 1.0 | 0.2 | 0.75 |
| `blend_raw_above_champion_1kt_a0.2_clip2` | 1.529715 | 1.122087 | 0.097435 | 0.808782 | 0.2 | 2.0 |
| `blend_all_a0.15_clip2` | 1.529788 | 1.121017 | 0.091329 | 1.0 | 0.15 | 2.0 |
| `blend_champion_ge_12kt_a0.2_clip2` | 1.530178 | 1.119934 | 0.079227 | 0.422269 | 0.2 | 2.0 |
| `blend_hotspots_a0.2_clip2` | 1.53018 | 1.119659 | 0.081003 | 0.43714 | 0.2 | 2.0 |
| `blend_raw_above_champion_1kt_a0.2_clip1` | 1.53019 | 1.121076 | 0.09402 | 0.808782 | 0.2 | 1.0 |
| `blend_vnext_below_champion_a0.2_clip2` | 1.530271 | 1.118124 | 0.050371 | 0.474337 | 0.2 | 2.0 |
| `blend_all_a0.15_clip1` | 1.530474 | 1.120401 | 0.088798 | 1.0 | 0.15 | 1.0 |

Best holdout variants:

| Variant | RMSE | MAE | Bias | Gate share | Alpha | Clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `blend_all_a0.2_clip2` | 1.361503 | 1.033419 | 0.098301 | 1.0 | 0.2 | 2.0 |
| `blend_all_a0.2_clip1` | 1.36165 | 1.033328 | 0.097631 | 1.0 | 0.2 | 1.0 |
| `blend_all_a0.2_clip0.75` | 1.362059 | 1.03351 | 0.096992 | 1.0 | 0.2 | 0.75 |
| `blend_all_a0.2_clip0.5` | 1.363282 | 1.034201 | 0.096554 | 1.0 | 0.2 | 0.5 |
| `blend_raw_above_champion_1kt_a0.2_clip2` | 1.363417 | 1.034946 | 0.101434 | 0.779861 | 0.2 | 2.0 |
| `blend_raw_above_champion_1kt_a0.2_clip1` | 1.363474 | 1.034807 | 0.100775 | 0.779861 | 0.2 | 1.0 |
| `blend_all_a0.15_clip2` | 1.363573 | 1.034927 | 0.098651 | 1.0 | 0.15 | 2.0 |
| `blend_raw_above_champion_1kt_a0.2_clip0.75` | 1.363728 | 1.034895 | 0.100129 | 0.779861 | 0.2 | 0.75 |
| `blend_all_a0.15_clip1` | 1.36375 | 1.034888 | 0.098149 | 1.0 | 0.15 | 1.0 |
| `blend_vnext_below_champion_a0.2_clip2` | 1.363789 | 1.034164 | 0.067264 | 0.525562 | 0.2 | 2.0 |
