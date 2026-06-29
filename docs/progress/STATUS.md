# baby-model Status

Updated: 2026-06-29 JST

## Proven

- Local repository initialized at `/Users/s30519/work/baby-model`.
- GitHub repository created as private:
  `https://github.com/Nicolas0315/baby-model`.
- v0 research scaffold is designed around AD-first / DA-delayed learning.
- `./scripts/verify.sh` passed on the current Mac.
- Local tmux session `baby-model` exists with:
  - window `1`: completed finite local loop.
  - window `2`: monitor window.
- Three local loop runs completed:
  - `runs/local-loop/20260628T022557Z/summary.md`
  - `runs/local-loop/20260628T022603Z/summary.md`
  - `runs/local-loop/20260628T022609Z/summary.md`
- Initial v0 result: `B_encoder_first` won all three local-loop seeds by
  last-window success. `C_baby_surprise` underperformed, likely because raw
  surprise reward encourages over-exploration in this small grid.
- v0.1 local tmux loop completed in `baby-model-v01`:
  - `runs/local-loop/20260628T024012Z/summary.md`
  - `runs/local-loop/20260628T024018Z/summary.md`
  - `runs/local-loop/20260628T024023Z/summary.md`
- v0.1 result: `D_baby_progress` improved over `C_baby_surprise`, but
  `B_encoder_first` still won all three seeds by last-window success.
- Fleet verify run completed for all four configured worker classes using
  `git archive | ssh | tmux`; each worker returned `verify ok` and `exit=0`.
  Exact host-level evidence is kept outside this repository in local docs.
- Fleet read-only check from the current Mac confirmed one macOS worker and
  three Windows/WSL workers are reachable enough for future approved lanes.
  Exact hostnames and GPU inventory are kept outside this repository in local
  docs evidence.
- `agmsg` is joined locally for turn-based progress delivery. Team details stay
  out of committed docs.
- GitHub Actions `verify` succeeded for commit
  `764de19117cae1cad927a305c3bbc074a889e00a` on 2026-06-29.
- GitHub issues:
  - Fleet worker protocol: `https://github.com/Nicolas0315/baby-model/issues/1` closed
  - MiniGrid/BabyAI migration: `https://github.com/Nicolas0315/baby-model/issues/2` closed
  - CI billing/spending-limit blocker: `https://github.com/Nicolas0315/baby-model/issues/3` resolved
  - v0.1 prediction-improvement reward: `https://github.com/Nicolas0315/baby-model/issues/4` closed
  - v0.2 progress-reward tuning/richer task: `https://github.com/Nicolas0315/baby-model/issues/5` closed
  - v0.3 annealed or auxiliary progress reward: `https://github.com/Nicolas0315/baby-model/issues/6` closed
  - harder MiniGrid/BabyAI trained run: `https://github.com/Nicolas0315/baby-model/issues/7` closed
  - v0.4 MiniGrid curriculum to BabyAI Unlock: `https://github.com/Nicolas0315/baby-model/issues/8` closed
  - v0.5 function approximation pilot: `https://github.com/Nicolas0315/baby-model/issues/9` closed
  - v0.6 multi-seed linear BabyAI Unlock sweep: `https://github.com/Nicolas0315/baby-model/issues/10` closed
  - v0.7 optional neural encoder pilot: `https://github.com/Nicolas0315/baby-model/issues/11` closed
  - v0.8 optional deep-learning/GPU lane: `https://github.com/Nicolas0315/baby-model/issues/12` closed
  - v0.9 full-fleet GPU-ready PyTorch lane: `https://github.com/Nicolas0315/baby-model/issues/13`
  - v1.0 multi-seed GPU PyTorch sweep: `https://github.com/Nicolas0315/baby-model/issues/14` closed
  - v1.1 revise PyTorch AD/DA conditions: `https://github.com/Nicolas0315/baby-model/issues/15` closed
  - v1.2 longer-window PyTorch AD/DA training design: `https://github.com/Nicolas0315/baby-model/issues/16` closed
  - v1.3 predictive representation objective: `https://github.com/Nicolas0315/baby-model/issues/17` closed
  - v1.4 predictive target and curriculum redesign: `https://github.com/Nicolas0315/baby-model/issues/18` open
- GitHub tracking:
  - milestone: `v0.1 Baby AD/DA`
  - labels: `experiment`, `fleet`, `ci-blocker`, `research-framework`
- v0.2 beta sweep implementation landed on `main`:
  - richer sparse-reward BabyGrid with obstacles and multiple toys
  - `configs/experiments/v02-sweep.json`
  - `scripts/run_beta_sweep.sh`
  - `scripts/launch_tmux_sweep.sh`
- `./scripts/verify.sh` passed after adding the v0.2 sweep smoke test.
- Local tmux sweep `baby-model-sweep` completed:
  - `runs/sweeps/20260628T030149Z/summary.md`
- v0.2 result: `B_encoder_first` still won all five seeds by last-window
  success. The smallest tested progress beta, `0.05`, was closest; larger beta
  values over-weighted intrinsic reward and reduced sparse-goal success.
- Fleet v0.2 sweep completed on all four configured worker classes using
  `git archive | ssh | tmux`; all workers produced the same aggregate winner,
  `B_encoder_first`. Exact host-level evidence is kept outside this repository
  in local docs.
- v0.3 annealed/gated/auxiliary progress controls are implemented locally:
  - `configs/experiments/v03-sweep.json`
  - `intrinsic_schedule`, `intrinsic_gate`, and `intrinsic_target` condition fields
  - `./scripts/verify.sh` includes v0.3 config and two-seed sweep smoke tests
- Local tmux sweep `baby-model-v03-aux` completed after implementing a real
  auxiliary action-value path:
  - `runs/v03-sweeps/20260629T005034Z/summary.md`
- v0.3 result: `B_encoder_first` still won by mean last-window success.
  `E_progress_anneal` was the best intrinsic variant; `G_progress_aux`
  underperformed direct reward shaping in this tiny grid.
- Fleet v0.3 sweep completed on all four configured worker classes using
  `git archive | ssh | tmux`; all workers produced the same aggregate winner,
  `B_encoder_first`, and the same condition table as the local run. Exact
  host-level evidence is kept outside this repository in local docs.
- MiniGrid/BabyAI optional verifier is implemented:
  - `baby_model/minigrid_probe.py`
  - `baby_model/minigrid_experiment.py`
  - `configs/experiments/minigrid-smoke.json`
  - `scripts/verify_minigrid.sh`
- Local optional venv verification passed with `minigrid==3.1.0` and
  `gymnasium==1.3.0`.
- MiniGrid trained smoke on `MiniGrid-Empty-5x5-v0` completed; `A_end_to_end`
  won with `success_last=1.000`.
- Fleet MiniGrid optional verification completed on all four configured worker
  classes using `git archive | ssh | tmux`; all workers produced the same
  trained smoke table. Exact host-level evidence is kept outside this
  repository in local docs.
- Harder BabyAI trained config for issue #7 is implemented:
  - `configs/experiments/minigrid-babyai-unlock.json`
  - `docs/experiments/minigrid-babyai-unlock.md`
  - optional verifier support via `MINIGRID_EXTRA_CONFIG`
- Local hard-task result on `BabyAI-Unlock-v0`: `A_end_to_end` won by
  last-window success (`0.050`). `B_encoder_first` had higher all-window
  success (`0.050`) but no last-window success, and `E_progress_anneal` did not
  reach external reward.
- Fleet hard-task replication completed on all four configured worker classes
  at commit `b89b50faa7fa50d805d5247372a0c5c5697a3e56`; all workers produced
  the same `BabyAI-Unlock-v0` extra table. Exact host-level evidence is kept
  outside this repository in local docs.
- MiniGrid curriculum runner is implemented locally:
  - `baby_model/minigrid_curriculum.py`
  - `configs/experiments/minigrid-curriculum-unlock.json`
  - optional verifier support via `MINIGRID_CURRICULUM_CONFIG`
- Local curriculum result on `BabyAI-Unlock-v0` final stage: `A_hard_only`
  still won by final-stage last-window success (`0.050`). The curriculum
  conditions did not preserve final-stage success, so this is another negative
  result for the current tabular implementation.
- Fleet curriculum replication completed on all four configured worker classes
  at commit `976591913b649e50b2455e0dbf44b39b8a4e1c9e`; all workers produced
  the same final-stage table. Exact host-level evidence is kept outside this
  repository in local docs.
- Linear function-approximation pilot for issue #9 is implemented locally:
  - `baby_model/minigrid_linear.py`
  - `configs/experiments/minigrid-linear-unlock.json`
  - optional verifier support via `MINIGRID_LINEAR_CONFIG`
- Local linear result on `BabyAI-Unlock-v0`: `B_linear_encoder_first` won by
  last-window success (`0.050`). This is the first local hard-task run where an
  encoder-first variant retained last-window success.
- Fleet linear replication completed on all four configured worker classes at
  commit `4077130a2c1d2dcca4dfa39e690f27a77edbf557`; all workers produced the
  same linear table. Exact host-level evidence is kept outside this repository
  in local docs.
- Multi-seed linear sweep for issue #10 is implemented locally:
  - `baby_model/minigrid_linear_sweep.py`
  - optional verifier support via `MINIGRID_LINEAR_SWEEP_CONFIG`
- Local multi-seed result over seeds `401,402,403`: `A_linear_hard_only` won
  by win count (`2/3`) after tying `B_linear_encoder_first` on mean and median
  last-window success (`0.050`). The issue #9 single-seed positive signal is
  therefore not yet robust enough to treat as a conclusion.
- Fleet multi-seed linear sweep replication completed on all four configured
  worker classes at commit `69f111719c9829128b78a8a0a5a97367dc6c19db`; all
  workers produced the same aggregate table. Exact host-level evidence is kept
  outside this repository in local docs.
- CPU-safe neural encoder pilot for issue #11 is implemented locally:
  - `baby_model/minigrid_neural.py`
  - `configs/experiments/minigrid-neural-unlock.json`
  - optional verifier support via `MINIGRID_NEURAL_CONFIG`
- Local neural result on `BabyAI-Unlock-v0`: all conditions had
  `success_last=0.000`; `A_neural_hard_only` won only by tie order.
- Fleet neural replication completed on all four configured worker classes at
  commit `c08b295014ec24489585b6a3798b9834c6c1597e`; all workers produced the
  same neural table as local verification. Exact host-level evidence is kept
  outside this repository in local docs.
- PyTorch was selected as the optional deep-learning/GPU framework for issue
  #12 from current official docs. The runner/config/verifier hook are
  implemented without adding project dependencies:
  - `baby_model/minigrid_torch.py`
  - `configs/experiments/minigrid-torch-unlock-smoke.json`
  - optional verifier support via `MINIGRID_TORCH_CONFIG`
  - docs in `docs/experiments/minigrid-torch-lane.md`
- Local optional PyTorch CPU smoke passed in an isolated venv with
  `torch==2.12.1` and `minigrid==3.1.0`. On `BabyAI-Unlock-v0`,
  `A_torch_hard_only` won by last-window success (`0.083`), while
  `B_torch_encoder_first` and `E_torch_progress` had `success_last=0.000`.
- Local optional PyTorch `auto` smoke selected MPS on the current Mac and
  completed, but all three conditions had `success_last=0.000`.
- Bounded issue #12 PyTorch fleet smoke is partially proven at commit
  `c009eda8ceea8b0e96f62ce24df2e4f00ea67e80`:
  - one remote macOS worker completed on MPS;
  - one remote Windows/WSL worker completed on CUDA;
  - one additional remote Windows/WSL worker completed only as CPU fallback
    after CUDA driver/wheel mismatch;
  - one Windows/WSL worker did not complete dependency installation before the
    clean stop point.
  Exact host-level evidence is kept outside this repository in local docs.
- Fleet MiniGrid/PyTorch setup is hardened locally:
  - `scripts/setup_minigrid_env.sh` supports `venv` and `uv` setup backends,
    Python selection, official PyTorch wheel index selection, and optional CPU
    fallback after CUDA failure.
  - `scripts/fleet_archive_run.sh` forwards those controls to tmux workers.
  - `./scripts/verify.sh` now covers shell syntax and setup dry-run checks
    without installing optional dependencies.
  - Local setup smoke passed with `uv`, Python 3.12, and `minigrid==3.1.0`.
- Follow-up issue #12 blocker-worker rerun at commit
  `c25dc648a938629938a809de382e293541e407e3` completed:
  - the newer-GPU Windows/WSL worker still failed CUDA due driver/wheel
    mismatch, then completed scripted CPU fallback with `torch==2.12.1+cu132`;
  - the remaining Windows/WSL worker did not finish the cu126 CUDA wheel install
    within a bounded smoke window, then completed CPU-only PyTorch smoke with
    `torch==2.12.1+cpu`;
  - both follow-up summaries matched the local CPU result:
    `A_torch_hard_only success_last=0.083`, `B_torch_encoder_first=0.000`,
    `E_torch_progress=0.000`.
- Issue #13 GPU compatibility planning is implemented locally:
  - `baby_model/gpu_compat.py`
  - `configs/fleet/gpu-wheel-policy.json`
  - `docs/experiments/gpu-compatibility-plan.md`
  The policy keeps repo-visible worker names anonymous and classifies each GPU
  worker into CUDA candidate vs. CPU-fallback-required using current driver,
  CUDA UMD, and PyTorch wheel family constraints.
- Issue #13 strict `cu132` CUDA smoke:
  - `gpu-worker-a` completed with `torch==2.12.1+cu132`, `device=cuda`, and
    `A_torch_hard_only success_last=0.083`.
  - `gpu-worker-b` remains driver/wheel blocked for CUDA 13 and was not rerun
    as strict GPU in this turn.
  - `gpu-worker-c` completed strict CUDA smoke with `torch==2.12.1+cu132` after
    switching Torch setup to `MINIGRID_TORCH_INSTALLER=pip`. Setup proved
    `torch_cuda_available=True` before training; the smoke used `device=cuda`
    and matched the worker-a table: `A_torch_hard_only success_last=0.083`,
    `B_torch_encoder_first=0.000`, `E_torch_progress=0.000`. Host-level
    evidence is kept outside this repository.
- PyTorch multi-seed sweep support is implemented locally:
  - `baby_model/minigrid_torch_sweep.py`
  - optional verifier support via `MINIGRID_TORCH_SWEEP_CONFIG`
  - docs in `docs/experiments/minigrid-torch-sweep.md`
  - tracking issue: `https://github.com/Nicolas0315/baby-model/issues/14`
- Issue #14 three-seed CUDA sweep completed on both CUDA-proven worker classes
  at commit `8bd7583c3a351f6d81a1b2e6c28fdf997039102f`. Both workers produced
  the same aggregate result: `A_torch_hard_only` won all three seeds with
  `mean_success_last=0.028`; `B_torch_encoder_first` and `E_torch_progress`
  stayed at `0.000`. This is a robust negative result for the current PyTorch
  AD/DA variants.
- Issue #15 v1.1 AD/DA condition set is implemented:
  - `configs/experiments/minigrid-torch-adda-v11.json`
  - `docs/experiments/minigrid-torch-adda-v11.md`
  The design shortens decoder delay and moves prediction progress into an
  auxiliary action-selection head instead of the main reward target.
- A bounded v1.1 CUDA smoke completed on `gpu-worker-c` at commit
  `3fe12c716fe125577414b9a57da7709f31ed312c`. The run was executable on
  `torch==2.12.1+cu132` and `device=cuda`, but all four conditions had
  `success_last=0.000`; this is negative evidence for escalating v1.1 to a
  multi-seed sweep.
- Issue #16 v1.2 longer-window condition set is implemented:
  - `configs/experiments/minigrid-torch-adda-v12.json`
  - `docs/experiments/minigrid-torch-adda-v12.md`
  The design extends all conditions to 48 episodes while keeping a long-window
  hard-only baseline.
- A bounded v1.2 CUDA smoke completed on `gpu-worker-c` at commit
  `b739545f58e55ed40e044b1a5cd5b3b4083f0dd9`. The run was executable on
  `torch==2.12.1+cu132` and `device=cuda`. `I_torch_long_delay` and
  `K_torch_long_coarse_aux` each reached one all-window success
  (`success_all=0.021`), but all four conditions had `success_last=0.000`.
  This is not promising enough to escalate v1.2 to a multi-seed GPU sweep.
- Issue #17 v1.3 predictive representation objective is implemented:
  - `configs/experiments/minigrid-torch-adda-v13.json`
  - `docs/experiments/minigrid-torch-adda-v13.md`
  - PyTorch DQN now supports a `next_feature` prediction head attached to the
    shared hidden representation.
- Local v1.3 CPU smoke passed in the existing optional PyTorch venv.
- A bounded v1.3 CUDA smoke completed on `gpu-worker-c` at commit
  `fd3ef5eb3fcb833d78f7c68652709d0d75567e0e`. The run was executable on
  `torch==2.12.1+cu132` and `device=cuda`; the predictive conditions each ran
  `5760` representation updates, but all conditions had
  `success_last=0.000`. This is not promising enough to escalate v1.3 to a
  multi-seed GPU sweep.
- Issue #18 v1.4 task-signal predictive objective is implemented:
  - `configs/experiments/minigrid-torch-adda-v14.json`
  - `docs/experiments/minigrid-torch-adda-v14.md`
  - PyTorch DQN now supports a `next_task_signal` representation objective.
- Local v1.4 CPU smoke passed in the existing optional PyTorch venv.
- A bounded v1.4 CUDA smoke completed on `gpu-worker-c` at commit
  `6105475cc025eead8669c8abb967bb79161bdf3a`. The run was executable on
  `torch==2.12.1+cu132` and `device=cuda`; both task-signal predictive
  conditions ran `5760` representation updates, but both stayed at
  `success_last=0.000`. `A_torch_hard_only_long` won with
  `success_last=0.050`, so v1.4 should not be escalated to a multi-seed GPU
  sweep.

## Next

- Close issue #18 after push and green CI, then open the next experiment issue
  around a curriculum-backed PyTorch AD/DA design.

## Not Yet Proven

- Strict CUDA smoke on `gpu-worker-b`; it remains blocked by driver/wheel
  compatibility and needs an explicit external state change before rerun.
- Any robust positive AD/DA GPU training result.
- Full objective completion.
