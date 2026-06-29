# PyTorch AD/DA v2.36 Narrowed ZB vs ZD CPU Precheck

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/60

## Motivation

v2.35 showed that the visibility-first two-head family is promising but not
stable enough for baseline replacement. v2.36 narrows the comparison to the
current single-head baseline `ZU`, fixed state-downweighted two-head `ZB`, and
state-head-annealed two-head `ZD`.

## Source State

- Config: `configs/experiments/minigrid-torch-adda-v45.json`
- Local runner: current Mac
- Torch: `2.12.1`
- Device: `cpu`
- Seeds: `4001,4002,4003,4004,4005`

## CPU Sweep

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v45.json --output-dir .tmp/local-v60-zb-zd-narrow --seeds 4001,4002,4003,4004,4005 --device cpu`

Summary artifact:

`.tmp/local-v60-zb-zd-narrow/20260629T161618Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZB_torch_gotoobj_two_head_state010_visibility065` | 1 | 0.160 | 0.100 | 0.105 | 0.085 | 0.460 | 0.210 | 0.230 |
| `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal` | 1 | 0.260 | 0.200 | 0.157 | 0.146 | 0.590 | 0.280 | 0.360 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 3 | 0.320 | 0.250 | 0.192 | 0.149 | 0.540 | 0.320 | 0.350 |

Per-seed winners:

- `4001`: `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `4002`: `ZB_torch_gotoobj_two_head_state010_visibility065`
- `4003`: `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `4004`: `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `4005`: `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal`

## Decision

Do not replace the `ZU` baseline with either `ZB` or `ZD`.

`ZD` is clearly the better two-head protocol in this narrowed gate: it beat
`ZB` on success, return, target-visible, target-center, and target-near
averages. However, `ZU` still won three of five seeds and led mean
final-window success, median success, mean return, median return, and
target-center. `ZD` kept a small edge on target-visible and target-near, so it
remains useful as a diagnostic candidate, but it is not strong enough to justify
baseline replacement or CUDA escalation from this CPU gate alone.

The next branch should treat `ZU` as the baseline to beat and search for a
mission-preservation-first objective that keeps the `ZD` visibility/near
advantage without losing `ZU`'s success and return.
