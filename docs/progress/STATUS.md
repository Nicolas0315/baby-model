# baby-model Status

Updated: 2026-06-30 JST

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
  - v1.4 predictive target and curriculum redesign: `https://github.com/Nicolas0315/baby-model/issues/18` closed
  - v1.5 curriculum-backed PyTorch AD/DA training: `https://github.com/Nicolas0315/baby-model/issues/19` closed
  - v1.6 redesign AD/DA signal beyond curriculum task-signal: `https://github.com/Nicolas0315/baby-model/issues/20` open
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
- Issue #19 v1.5 curriculum-backed PyTorch AD/DA training is implemented:
  - `configs/experiments/minigrid-torch-adda-v15.json`
  - `docs/experiments/minigrid-torch-adda-v15.md`
  - PyTorch DQN now supports staged curriculum training with stage-level
    environment IDs, episode counts, active stage selection per condition, and
    agent carry-over across active stages.
- Local v1.5 CPU smoke passed in the existing optional PyTorch venv. On the
  final `BabyAI-Unlock-v0` stage, `Q_torch_curriculum_task_signal_aux_progress`
  beat `A_torch_hard_only_long` by last-window success (`0.050` vs `0.000`)
  and return (`0.042` vs `0.000`). This is a positive local smoke signal, but
  not yet a robust GPU result.
- A bounded v1.5 CUDA smoke completed on `gpu-worker-c` at commit
  `083069c13d40326a5312d7ffe44ea0b297fe1f2a`. The run was executable on
  `torch==2.12.1+cu132` and `device=cuda`. `Q_torch_curriculum_task_signal_aux_progress`
  tied the hard-only baseline on final-stage last-window success (`0.050`) and
  all-window success (`0.021`), but had lower final-stage return (`0.042` vs
  `0.045`). This does not meet the v1.5 escalation rule, so do not run a
  multi-seed GPU sweep for this condition family.
- Issue #20 v1.6 action-prior AD/DA signal is implemented:
  - `configs/experiments/minigrid-torch-adda-v16.json`
  - `docs/experiments/minigrid-torch-adda-v16.md`
  - PyTorch DQN now supports an `action_prior` representation objective and an
    optional `action_prior_weight` bonus mixed into action selection after the
    decoder delay.
- Local v1.6 CPU smoke passed in the existing optional PyTorch venv after
  correcting the MiniGrid view-coordinate used by the action-prior label. The
  action-prior family did not beat `A_torch_hard_only_long` on final-window
  success; `S_torch_action_prior_policy_mix` had one all-window success
  (`success_all=0.021`) but `success_last=0.000` and `return_last=0.000`.
- A bounded v1.6 CUDA smoke completed on `gpu-worker-c` at commit
  `ff9ef00afb7c6d79b8f580db4cde8cc8c8cab0fc`. The run was executable on
  `torch==2.12.1+cu132` and `device=cuda`. `R_torch_action_prior_delay` beat
  the hard-only baseline on final-stage last-window success (`0.050` vs
  `0.000`) and return (`0.044` vs `0.000`). This met the v1.6 escalation rule
  and triggered the multi-seed CUDA sweep below.
- The v1.6 three-seed CUDA sweep completed on `gpu-worker-c` at commit
  `b66e1c591dea7afdddfeb362e49303de9b051e5b` with seeds `1201,1202,1203`.
  `R_torch_action_prior_delay` and `S_torch_action_prior_policy_mix` tied on
  mean final-window success (`0.017`), while `S` had slightly higher mean
  final-window return (`0.016` vs `0.015`). All conditions had median
  final-window success and median return of `0.000`, and each condition won one
  seed. Treat v1.6 as non-robust evidence; do not escalate the current
  action-prior design further without changing the signal or task family.
- Issue #22 v1.7 controllability AD/DA signal is implemented:
  - `configs/experiments/minigrid-torch-adda-v17.json`
  - `docs/experiments/minigrid-torch-adda-v17.md`
  - PyTorch DQN now supports a `controllability` representation objective that
    predicts whether the chosen action changes the sparse observation signature.
- Local v1.7 CPU smoke passed in the existing optional PyTorch venv with
  `torch==2.12.1` and `device=cpu`. `T_torch_controllability_delay` beat
  `A_torch_hard_only_long` on final-stage last-window success (`0.050` vs
  `0.000`) and return (`0.048` vs `0.000`) with non-zero representation updates,
  meeting the v1.7 CUDA escalation rule.
- A bounded v1.7 CUDA smoke completed on `gpu-worker-c` at commit
  `556308831bafe60c99f5767e4e2c9a1b2199702f` with `torch==2.12.1+cu132` and
  `device=cuda`. It reproduced the local positive signal:
  `T_torch_controllability_delay` beat `A_torch_hard_only_long` on final-stage
  last-window success (`0.050` vs `0.000`) and return (`0.048` vs `0.000`).
- The v1.7 three-seed CUDA sweep completed on `gpu-worker-c` at the same commit
  with seeds `1301,1302,1303`. `T_torch_controllability_delay` led by mean
  final-window success (`0.017` vs `0.000`) and return (`0.016` vs `0.000`),
  but median final-window success and median return were both `0.000`, and
  `A_torch_hard_only_long` had the higher seed win count (`2` vs `1`). Treat
  v1.7 as weak positive but not robust enough to escalate without changing the
  task ladder or representation target.
- Issue #23 v1.8 dense controllability ladder is implemented:
  - `configs/experiments/minigrid-torch-adda-v18.json`
  - `docs/experiments/minigrid-torch-adda-v18.md`
  - The config compares the v1.7-style sparse controllability ladder against a
    denser door/key/local-unlock ladder before final `BabyAI-Unlock-v0`.
- Local v1.8 CPU smoke passed in the existing optional PyTorch venv with
  `torch==2.12.1` and `device=cpu`, but did not meet the CUDA escalation rule.
  `A_torch_hard_only_long` won final-stage last-window success (`0.050`) and
  return (`0.042`), while both dense-ladder controllability conditions stayed
  at `0.000` success and return. Treat v1.8 as negative ladder evidence for the
  current controllability target.
- Issue #24 v1.9 affordance-progress representation target is implemented:
  - `configs/experiments/minigrid-torch-adda-v19.json`
  - `docs/experiments/minigrid-torch-adda-v19.md`
  - PyTorch DQN now supports an `affordance_progress` representation objective
    that predicts a structured local key/door/goal/front-cell affordance vector
    for the next observation.
- Local v1.9 CPU smoke passed in the existing optional PyTorch venv with
  `torch==2.12.1` and `device=cpu`, but did not meet the CUDA escalation rule.
  The v1.7-style `T_torch_controllability_delay` baseline won final-stage
  last-window success (`0.050`) and return (`0.045`), while both
  affordance-progress conditions stayed at `0.000` success and return. Treat
  v1.9 as negative evidence for this affordance-progress target.
- Issue #25 v2.0 transition-group representation target is implemented:
  - `configs/experiments/minigrid-torch-adda-v20.json`
  - `docs/experiments/minigrid-torch-adda-v20.md`
  - PyTorch DQN now supports a `transition_group` representation objective that
    predicts which semantic affordance-progress groups changed from current to
    next observation after an action.
- Local v2.0 CPU smoke passed in the existing optional PyTorch venv with
  `torch==2.12.1` and `device=cpu`, but did not meet the CUDA escalation rule.
  Both transition-group conditions tied the baselines on final-stage
  last-window success (`0.000`) and did not improve final-stage return
  (`0.000`). Treat v2.0 as negative evidence for a plain transition change-mask
  target.
- Issue #26 v2.1 explicit subgoal-progress representation target is
  implemented:
  - `configs/experiments/minigrid-torch-adda-v21.json`
  - `docs/experiments/minigrid-torch-adda-v21.md`
  - PyTorch DQN now supports a `subgoal_progress` representation objective that
    predicts a 10-bit key-door-goal progress event vector from current and next
    observations.
- Local v2.1 CPU smoke passed in the existing optional PyTorch venv with
  `torch==2.12.1` and `device=cpu`, but did not meet the CUDA escalation rule.
  All final-stage last-window success and return values were `0.000`; both
  subgoal-progress conditions ran non-zero representation updates but did not
  improve final-stage reward. Treat v2.1 as negative evidence for a sparse
  explicit subgoal event target under the current ladder.
- Issue #27 v2.2 state-plus-delta representation target is implemented:
  - `configs/experiments/minigrid-torch-adda-v22.json`
  - `docs/experiments/minigrid-torch-adda-v22.md`
  - PyTorch DQN now supports a `state_plus_delta` representation objective that
    predicts current affordance state, next affordance state, transition mask,
    and subgoal-progress events as one 58-dimensional target.
- Local v2.2 CPU smoke passed in the existing optional PyTorch venv with
  `torch==2.12.1` and `device=cpu`, but did not meet the CUDA escalation rule.
  `ZD_torch_state_plus_delta_delay` reached one all-window final-stage success
  (`success_all=0.021`), but both state-plus-delta conditions had final-stage
  last-window success and return of `0.000`. Treat v2.2 as weak but
  non-escalating target evidence.
- Issue #28 v2.3 dense key-door ladder is implemented:
  - `configs/experiments/minigrid-torch-adda-v23.json`
  - `docs/experiments/minigrid-torch-adda-v23.md`
  - The config evaluates `state_plus_delta` on a denser key-door ladder with
    door navigation, door opening, DoorKey, UnlockLocal, and UnlockPickup stages
    before final `BabyAI-Unlock-v0`.
- Local v2.3 CPU smoke passed in the existing optional PyTorch venv with
  `torch==2.12.1` and `device=cpu`, but did not meet the CUDA escalation rule.
  `ZF_torch_dense_keydoor_state_plus_delta_delay` reached one all-window
  final-stage success (`success_all=0.021`), but both dense conditions had
  final-stage last-window success and return of `0.000`. Treat v2.3 as negative
  ladder evidence: intermediate tasks are learnable, but they did not transfer
  to stable final unlock success under the current PyTorch DQN setup.
- Issue #29 v2.4 two-phase frozen-encoder protocol is implemented:
  - `configs/experiments/minigrid-torch-adda-v24.json`
  - `docs/experiments/minigrid-torch-adda-v24.md`
  - PyTorch DQN now supports `freeze_encoder_after_delay` and
    `stop_representation_after_delay`, so a condition can run AD-only
    representation learning first, then freeze the shared encoder and train the
    DA decoder/head.
- Local v2.4 CPU smoke passed in the existing optional PyTorch venv with
  `torch==2.12.1` and `device=cpu`, but did not meet the CUDA escalation rule.
  `ZH_torch_two_phase_state_plus_delta_frozen` correctly froze the encoder and
  ran `8480` representation updates before the DA phase, but final-stage
  last-window success and return were both `0.000`. Treat v2.4 as negative
  evidence for the current PyTorch DQN family.
- Issue #30 v2.5 task-family change to BabyAI GoToObj is implemented:
  - `configs/experiments/minigrid-torch-adda-v25.json`
  - `docs/experiments/minigrid-torch-adda-v25.md`
  - local CPU smoke passed with `torch==2.12.1` and `device=cpu`;
    `T_torch_gotoobj_controllability_delay` beat the hard-only baseline on
    final-stage last-window success (`0.550` vs `0.250`) and return (`0.331`
    vs `0.128`), meeting the CUDA escalation rule.
  - bounded CUDA smoke completed on `gpu-worker-c` with `torch==2.12.1+cu132`
    and `device=cuda`; it reproduced the same positive signal for
    `T_torch_gotoobj_controllability_delay`.
  - three-seed CUDA sweep completed on `gpu-worker-c` with seeds
    `2101,2102,2103`; `ZI_torch_gotoobj_state_plus_delta_delay` won by mean
    final-stage last-window success (`0.533` vs hard-only `0.450`) and mean
    return (`0.342` vs hard-only `0.280`).
  - Per-seed winners split one win each across `T`, hard-only, and `ZI`. Treat
    v2.5 as positive task-family evidence, but not a stable single-condition
    winner.
- Issue #31 v2.6 matched GoToObj representation isolation is implemented:
  - `configs/experiments/minigrid-torch-adda-v26.json`
  - `docs/experiments/minigrid-torch-adda-v26.md`
  - local CPU smoke passed with `torch==2.12.1` and `device=cpu`;
    `ZM_torch_gotoobj_state_plus_delta_matched_delay` tied the matched
    no-representation curriculum on final-stage last-window success (`0.650`)
    while improving return (`0.461` vs `0.366`), meeting the CUDA escalation
    rule.
  - bounded CUDA smoke completed on `gpu-worker-c` with `torch==2.12.1+cu132`
    and `device=cuda`; it reproduced the same result.
  - Treat v2.6 as evidence that the GoToObj curriculum explains most of the
    success lift, while state-plus-delta remains the best representation
    candidate by return.
- Issue #32 v2.7 multi-seed matched GoToObj sweep is complete:
  - CUDA sweep ran on `gpu-worker-c` with seeds `2201,2202,2203`,
    `torch==2.12.1+cu132`, and `device=cuda`.
  - `ZL_torch_gotoobj_controllability_matched_delay` won by mean final-stage
    last-window success (`0.533`) and seed wins (`2`).
    `ZL_torch_gotoobj_controllability_matched_delay` and
    `ZM_torch_gotoobj_state_plus_delta_matched_delay` both rounded to
    `0.365` mean return, above `ZK_torch_gotoobj_curriculum_no_repr_delay`
    at `0.348`.
  - Neither representation condition met the #32 decision rule because both
    lost median final-window success to `ZK_torch_gotoobj_curriculum_no_repr_delay`
    (`ZK` `0.650`, `ZL` `0.450`, `ZM` `0.500`).
  - Treat the GoToObj curriculum effect as dominant under the current DQN
    auxiliary-head family.
- Issue #33 v2.8 non-DQN supervised representation diagnostic is implemented:
  - `baby_model/minigrid_repr_probe.py`
  - `configs/experiments/minigrid-repr-probe-v28.json`
  - `docs/experiments/minigrid-repr-probe-v28.md`
  - local CPU smoke collected `986` GoToObj-family random-policy transitions.
  - Best fixed feature set was `raw_current`, with mean decision-label accuracy
    `0.606` and mean lift `0.093`, but the decision rule failed because
    `mission_color` accuracy was `0.553` and `changed` lift was `-0.010`.
  - Treat fixed current-observation features as insufficient for the full probe
    target set; redesign toward stronger transition labels or a trained
    predictive encoder before returning to RL/CUDA.
- Issue #34 v2.9 trained non-DQN predictive encoder probe is implemented:
  - `baby_model/minigrid_repr_probe.py`
  - `configs/experiments/minigrid-repr-probe-v29.json`
  - `docs/experiments/minigrid-repr-probe-v29.md`
  - local CPU smoke collected `821` GoToObj-family random-policy transitions.
  - The predictive encoder trained on `changed` reached held-out accuracy
    `0.945` and held-out lift `0.445` over majority.
  - `predictive_encoder` met the #34 relative decision rule: it improved
    `changed` lift over `raw_current` by `0.415`, while mission-object
    accuracy dropped only `0.006` and mission-color accuracy dropped only
    `0.024`, both within the allowed `0.050` maximum drop.
  - Treat v2.9 as positive non-DQN evidence that a trained predictive signal
    can add transition information without erasing the mission signal.
- Issue #35 v2.10 richer non-DQN predictive encoder probe is implemented:
  - `baby_model/minigrid_repr_probe.py`
  - `configs/experiments/minigrid-repr-probe-v30.json`
  - `docs/experiments/minigrid-repr-probe-v30.md`
  - local CPU smoke collected `936` GoToObj-family random-policy transitions.
  - `predictive_changed` reproduced a useful held-out `changed` signal
    (`test_accuracy=0.904`, `test_lift=0.214`).
  - `predictive_next_signature` failed as a richer objective
    (`test_accuracy=0.037`, `test_lift=-0.075`) and did not beat
    `predictive_changed` on `next_signature_bucket` probe lift
    (`0.139` vs `0.150`).
  - Treat v2.10 as negative evidence for a plain next-signature-bucket
    predictive objective under the current random-policy dataset.
- Issue #36 v2.11 scripted-policy non-DQN representation probe is implemented:
  - `baby_model/minigrid_repr_probe.py`
  - `configs/experiments/minigrid-repr-probe-v31.json`
  - `docs/experiments/minigrid-repr-probe-v31.md`
  - local CPU smoke collected `964` GoToObj-family scripted-policy transitions
    after correcting MiniGrid image-axis handling to `image[x][y][channel]`.
  - `predictive_changed` reached held-out `changed` accuracy `1.000`, but
    held-out lift was only `0.083` because the scripted dataset made `changed`
    highly imbalanced.
  - The `changed` probe did not meet the v2.11 rule: `predictive_changed`
    lift was `-0.010`, below the v2.10 random-policy external baseline
    `0.209`.
  - The scripted dataset still improved fixed `next_signature_bucket`
    separability: `raw_current` lift rose to `0.646`, versus v2.10
    random-policy `0.134`.
  - Treat v2.11 as negative for the current scripted `changed` encoder, but
    positive evidence that scripted collection produces a more structured
    dataset for richer transition labels.
- Issue #37 v2.12 scripted-policy transition-label probe is implemented:
  - `baby_model/minigrid_repr_probe.py`
  - `configs/experiments/minigrid-repr-probe-v32.json`
  - `docs/experiments/minigrid-repr-probe-v32.md`
  - local CPU smoke collected `781` GoToObj-family scripted-policy transitions.
  - `relative_to_baseline` decision logic now honors the configured
    `transition_label`, allowing `next_signature_bucket` to be evaluated
    without a new decision mode.
  - `predictive_next_signature` reached held-out `next_signature_bucket`
    accuracy `0.282` and lift `0.115`, but the downstream representation probe
    matched `raw_current` exactly on `next_signature_bucket` lift (`0.462` vs
    `0.462`), so the lift delta was `0.000` against the required `0.010`.
  - Treat v2.12 as negative for bucketed next-signature prediction with raw
    passthrough. The next non-DQN probe should either remove raw passthrough for
    a purer representation diagnostic or replace hashed buckets with a semantic
    object/color transition label.
- Issue #38 v2.13 pure scripted representation probe is implemented:
  - `configs/experiments/minigrid-repr-probe-v33.json`
  - `docs/experiments/minigrid-repr-probe-v33.md`
  - local CPU smoke collected `904` GoToObj-family scripted-policy transitions.
  - `predictive_next_signature` and `predictive_next_signature_pure` reached
    the same held-out `next_signature_bucket` classifier result:
    accuracy `0.361`, lift `0.239`.
  - The pure learned representation did not improve downstream transition
    probing: `predictive_next_signature_pure` `next_signature_bucket` lift was
    `0.389`, below both `raw_current` and raw-passthrough
    `predictive_next_signature` at `0.500`.
  - Mission-object and mission-color accuracy also dropped too far in the pure
    representation (`-0.089` and `-0.128` vs `raw_current`), so v2.13 is
    negative under the documented rule.
  - Treat hashed next-signature prediction as the wrong next diagnostic for
    this non-DQN lane. The next probe should add a semantic object/color
    transition label on the corrected scripted-policy dataset.
- Issue #39 v2.14 semantic object-color transition label probe is implemented:
  - `baby_model/minigrid_repr_probe.py`
  - `configs/experiments/minigrid-repr-probe-v34.json`
  - `docs/experiments/minigrid-repr-probe-v34.md`
  - local CPU smoke collected `905` GoToObj-family scripted-policy transitions.
  - Added `target_visibility_transition`, an interpretable semantic label such
    as `absent->center_near`, derived from the mission target object/color and
    corrected MiniGrid `image[x][y][channel]` relation to the agent.
  - `predictive_target_visibility` reached held-out semantic-label accuracy
    `0.663` and lift `0.376`.
  - Downstream probe preserved mission-object and mission-color accuracy, and
    improved `target_visibility_transition` lift from `0.387` to `0.392`, but
    the lift delta was only `0.0055`, below the documented `0.010` threshold.
  - Treat v2.14 as a negative near-miss. The semantic target is learnable and
    directionally useful, but not yet strong enough to return to RL/CUDA.
- Issue #40 v2.15 multi-seed semantic transition probe is implemented:
  - `baby_model/minigrid_repr_probe_sweep.py`
  - `configs/experiments/minigrid-repr-probe-v35.json`
  - `docs/experiments/minigrid-repr-probe-v35.md`
  - local CPU sweep ran seeds `2901,2902,2903`.
  - Mean `target_visibility_transition` lift delta over `raw_current` was
    `0.045099`, above the `0.010` threshold.
  - All three seeds had non-negative semantic lift delta (`0.006`, `0.130`,
    `0.000`), and mission-object/color accuracy never dropped below the
    `-0.050` gate.
  - The smallest per-seed semantic-label test set had `131` examples, above
    the sweep's `10` example gate.
  - Treat v2.15 as positive non-DQN evidence that the semantic transition label
    captures a stable representation signal under scripted-policy collection.
- Issue #41 v2.16 semantic transition signal RL integration design is
  implemented:
  - `baby_model/minigrid_torch.py`
  - `configs/experiments/minigrid-torch-adda-v36.json`
  - `docs/experiments/minigrid-torch-adda-v36.md`
  - Added `target_visibility_transition` as a PyTorch RL auxiliary
    representation objective using a 49-way before/after target-relation
    one-hot vector.
  - Local CPU smoke on the bounded GoToObj matched curriculum selected
    `ZN_torch_gotoobj_target_visibility_matched_delay` as
    `winner_last_window`.
  - `ZN` beat the matched no-representation curriculum on final-stage
    last-window success (`0.400` vs `0.100`) and return (`0.274` vs `0.031`)
    with `1997` representation updates.
  - v2.16 uses final-stage external success/return as the mission-preservation
    proxy because the RL runner does not directly score mission-object/color
    probe accuracy.
  - A bounded CUDA smoke on `gpu-worker-c` at commit
    `8f628a562324fdcd7ea19209edf665a0fb027f0b` reproduced the same winner with
    `torch==2.12.1+cu132`, `torch_cuda_available=True`, and `device=cuda`.
  - CUDA `ZN` again beat the matched no-representation curriculum on
    final-stage last-window success (`0.400` vs `0.100`) and return (`0.274`
    vs `0.031`) with `1997` representation updates.
  - Treat v2.16 as positive single-seed CUDA replication evidence, not yet
    multi-seed proof.
- Issue #43 v2.18 CUDA multi-seed semantic-transition sweep is complete:
  - The sweep ran on `gpu-worker-c` with seeds `3001,3002,3003`,
    `torch==2.12.1+cu132`, and `devices=cuda`.
  - `ZN_torch_gotoobj_target_visibility_matched_delay` beat the matched
    no-representation curriculum on mean final-stage last-window success
    (`0.283` vs `0.100`) and mean return (`0.185` vs `0.059`), with equal-or-
    better median success (`0.300` vs `0.100`).
  - `ZN` did not become a clear single-condition winner against
    `ZM_torch_gotoobj_state_plus_delta_matched_delay`: both had
    `0.283` mean final-window success and `0.300` median final-window success;
    `ZN` had higher mean return (`0.185` vs `0.158`), but `ZM` won more seeds
    (`2` vs `1`).
  - Treat v2.18 as positive representation-vs-no-repr multi-seed CUDA evidence,
    not proof that semantic transition is the stable best representation target.
- Issue #44 v2.19 combined state-plus-semantic objective is implemented:
  - `baby_model/minigrid_torch.py`
  - `configs/experiments/minigrid-torch-adda-v37.json`
  - `docs/experiments/minigrid-torch-adda-v37.md`
  - Added `state_plus_target_visibility`, a 107-dimensional representation
    target concatenating `state_plus_delta` and `target_visibility_transition`.
  - Local CPU smoke on the bounded GoToObj matched curriculum selected
    `ZO_torch_gotoobj_state_plus_target_visibility_delay` as
    `winner_last_window`.
  - `ZO` beat `ZM` and `ZN` on final-stage last-window success (`0.600` vs
    `0.300` and `0.100`) and return (`0.376` vs `0.157` and `0.062`) with
    `1760` representation updates.
  - `ZO` also beat the matched no-representation curriculum (`0.600` vs
    `0.350` success_last; `0.376` vs `0.219` return_last).
  - Treat v2.19 as positive CPU-first evidence for combining state-plus-delta
    and semantic target-visibility signals, not yet CUDA or multi-seed proof.
- Issue #45 v2.20 CUDA replication for the combined objective is complete:
  - A bounded CUDA smoke on `gpu-worker-c` at commit
    `b94f765331d807f89fde8f119e02461641e9218d` reproduced the same winner with
    `torch==2.12.1+cu132`, `torch_cuda_available=True`, and `device=cuda`.
  - CUDA `ZO_torch_gotoobj_state_plus_target_visibility_delay` beat `ZM` and
    `ZN` on final-stage last-window success (`0.600` vs `0.300` and `0.100`)
    and return (`0.376` vs `0.157` and `0.062`) with `1760` representation
    updates.
  - CUDA `ZO` also beat the matched no-representation curriculum (`0.600` vs
    `0.350` success_last; `0.376` vs `0.219` return_last).
  - Treat v2.20 as positive single-seed CUDA replication evidence, not yet
    multi-seed proof.
- Issue #46 v2.21 CUDA multi-seed sweep for the combined objective is complete:
  - The sweep ran on `gpu-worker-c` with seeds `3101,3102,3103`,
    `torch==2.12.1+cu132`, and `devices=cuda`.
  - `ZO_torch_gotoobj_state_plus_target_visibility_delay` won one seed, but
    did not pass the multi-seed decision rule.
  - `ZO` lost mean final-stage last-window success to both `ZM` and `ZN`
    (`0.350` vs `0.450` and `0.417`) and lost mean return to both (`0.208` vs
    `0.275` and `0.289`).
  - Treat v2.21 as negative stability evidence for the simple combined target:
    the positive single-seed CPU/CUDA result did not generalize across this
    three-seed CUDA sweep.
- Issue #47 v2.22 direct mission-preservation probe is implemented:
  - `baby_model/minigrid_torch.py`
  - `baby_model/minigrid_torch_sweep.py`
  - `docs/experiments/minigrid-torch-mission-probe-v38.md`
  - Added final-observation mission-target probe fields to PyTorch RL metrics
    and summary tables: target visible, center, and near rates over the final
    window.
  - Added the same probe fields to PyTorch sweep aggregates so multi-seed runs
    can report mean mission-target retention.
  - Local CPU smoke on the v37 GoToObj matched curriculum completed with
    `torch==2.12.1` and `device=cpu`.
  - In that smoke, `ZO_torch_gotoobj_state_plus_target_visibility_delay` had
    the best reward metrics and the best probe metrics:
    `success_last=0.450`, `return_last=0.295`,
    `target_visible_last=0.600`, `target_center_last=0.500`,
    `target_near_last=0.500`.
  - The probe did not reveal a hidden mission-preservation winner that
    success/return missed, so no CUDA issue is opened from v2.22 alone.
- Issue #48 v2.23 mission-conditioned target redesign has a first CPU gate:
  - `baby_model/minigrid_torch.py`
  - `configs/experiments/minigrid-torch-adda-v38.json`
  - `docs/experiments/minigrid-torch-adda-v38.md`
  - Added `mission_target_transition`, which keeps the same relation-transition
    label space as `target_visibility_transition` but returns a zero vector
    when the mission target is unknown or both current and next relation are
    `absent`.
  - Added `state_plus_mission_target`, which concatenates `state_plus_delta`
    with the mission-conditioned semantic transition target.
  - Local CPU smoke on seed `3301` completed with `torch==2.12.1` and
    `device=cpu`.
  - The new candidates were negative on this CPU gate:
    `ZP_torch_gotoobj_mission_target_visibility_delay` and
    `ZQ_torch_gotoobj_state_plus_mission_target_delay` both had
    `success_last=0.000` and `return_last=0.000`.
  - Do not escalate v2.23 to CUDA from this result.
- Issue #48 v2.24 state-plus-target-visibility beta sweep passed a CPU gate:
  - `configs/experiments/minigrid-torch-adda-v39.json`
  - `docs/experiments/minigrid-torch-adda-v39.md`
  - Tested the combined `state_plus_target_visibility` objective at
    `representation_beta` values `0.1`, `0.3`, and `0.5`, while retaining
    `ZK`, `ZM`, and `ZN` baselines.
  - Local CPU smoke on seed `3401` completed with `torch==2.12.1` and
    `device=cpu`.
  - `ZR_torch_gotoobj_state_plus_target_visibility_b010` won the CPU gate:
    `success_last=0.450`, `return_last=0.247`,
    `target_visible_last=0.900`, `target_center_last=0.550`,
    `target_near_last=0.700`.
  - This is enough to justify bounded CUDA replication for the beta `0.1`
    combined objective, not enough to claim stability.
- Issue #49 v2.25 CUDA replication for the beta `0.1` combined objective is
  complete:
  - A bounded CUDA smoke ran on `gpu-worker-c` at commit
    `cabadbebd418872ecc6c839a56fdd3e0a703b5c3`.
  - Setup proved `torch==2.12.1+cu132`, `torch_cuda_available=True`,
    `torch_cuda_device_count=1`, and `device=cuda`.
  - CUDA reproduced the same seed-`3401` table as the CPU gate.
  - `ZR_torch_gotoobj_state_plus_target_visibility_b010` again won:
    `success_last=0.450`, `return_last=0.247`,
    `target_visible_last=0.900`, `target_center_last=0.550`,
    `target_near_last=0.700`.
  - Treat v2.25 as positive single-seed CUDA replication evidence, not
    multi-seed stability proof.
- Issue #50 v2.26 CUDA multi-seed sweep for the beta `0.1` combined objective
  is complete:
  - The sweep ran on `gpu-worker-c` at commit
    `e722e6c1ed43e40a2ae5810022add8be1aa60066` with seeds
    `3401,3402,3403`.
  - Setup proved `torch==2.12.1+cu132`, `torch_cuda_available=True`,
    `torch_cuda_device_count=1`, and `devices=cuda`.
  - `ZR_torch_gotoobj_state_plus_target_visibility_b010` won two of three
    seeds and led the aggregate table:
    `mean_success_last=0.350`, `median_success_last=0.350`,
    `mean_return_last=0.182`, `median_return_last=0.204`,
    `target_visible_last=0.700`, `target_center_last=0.383`,
    `target_near_last=0.483`.
  - Treat `ZR` as the current strongest representation-driven AD/DA candidate
    and the next baseline to beat.
- Issue #51 v2.27 tight beta-neighborhood CPU gate is implemented:
  - `configs/experiments/minigrid-torch-adda-v40.json`
  - `docs/experiments/minigrid-torch-adda-v40.md`
  - Tested `state_plus_target_visibility` beta values `0.05`, `0.075`, `0.1`,
    `0.125`, and `0.15` against the `ZK` no-representation curriculum.
  - Local CPU smoke on seed `3501` completed with `torch==2.12.1` and
    `device=cpu`.
  - `ZT_torch_gotoobj_state_plus_target_visibility_b005` won the CPU gate:
    `success_last=0.650`, `return_last=0.472`,
    `target_visible_last=0.850`, `target_center_last=0.650`,
    `target_near_last=0.700`.
  - `ZT` beat the current `ZR` beta `0.1` baseline on final-window success,
    return, and all mission-preservation probe columns in this CPU smoke.
  - This justifies bounded CUDA replication for beta `0.05`, not a baseline
    replacement until CUDA evidence exists.
- Issue #52 v2.28 CUDA replication for the beta `0.05` combined objective is
  complete:
  - A bounded CUDA smoke ran on `gpu-worker-c` at commit
    `ebfcab3299cfe48e0d989ec168c31b80d56465d9`.
  - Setup proved `torch==2.12.1+cu132`, `torch_cuda_available=True`,
    `torch_cuda_device_count=1`, and `device=cuda`.
  - CUDA reproduced the same seed-`3501` table as the CPU gate.
  - `ZT_torch_gotoobj_state_plus_target_visibility_b005` again won:
    `success_last=0.650`, `return_last=0.472`,
    `target_visible_last=0.850`, `target_center_last=0.650`,
    `target_near_last=0.700`.
  - Treat v2.28 as positive single-seed CUDA replication evidence, not
    multi-seed stability proof.
- Issue #53 v2.29 CUDA multi-seed sweep for the beta-neighborhood candidates
  is complete:
  - The sweep ran on `gpu-worker-c` at commit
    `3a9b729050600246c85c9bbd73013a0c46425754` with seeds
    `3501,3502,3503`.
  - Setup proved `torch==2.12.1+cu132`, `torch_cuda_available=True`,
    `torch_cuda_device_count=1`, and `devices=cuda`.
  - `ZT_torch_gotoobj_state_plus_target_visibility_b005` beat the prior `ZR`
    baseline, but `ZU_torch_gotoobj_state_plus_target_visibility_b0075` was the
    actual multi-seed winner.
  - `ZU` won two of three seeds and led the aggregate table:
    `mean_success_last=0.550`, `median_success_last=0.550`,
    `mean_return_last=0.372`, `median_return_last=0.360`,
    `target_visible_last=0.700`, `target_center_last=0.567`,
    `target_near_last=0.583`.
  - Treat `ZU` beta `0.075` as the current strongest beta-neighborhood
    candidate and the next baseline to beat.
- Issue #54 v2.30 longer-horizon ZU probe gate has a CPU result:
  - `configs/experiments/minigrid-torch-adda-v41.json`
  - `docs/experiments/minigrid-torch-adda-v41.md`
  - Doubled the short beta-sweep stage lengths and compared the no-repr
    curriculum against beta `0.05`, `0.075`, and `0.1` combined objectives.
  - Local CPU smoke on seed `3601` completed with `torch==2.12.1` and
    `device=cpu`.
  - The result was mixed, not a CUDA escalation signal:
    `ZK_torch_gotoobj_curriculum_no_repr_delay_long` won final-window success
    (`0.550` vs `ZU` at `0.500`), while `ZU` beat `ZK` on return
    (`0.367` vs `0.262`) and `target_near_last` (`0.600` vs `0.550`).
  - Do not open a CUDA follow-up from v2.30 alone.
- Issue #55 v2.31 true two-head objective is implemented and has a CPU result:
  - `configs/experiments/minigrid-torch-adda-v42.json`
  - `docs/experiments/minigrid-torch-adda-v42.md`
  - Added `state_delta_and_target_visibility`, a true two-head objective with
    separate `state_plus_delta` and `target_visibility_transition` predictors
    plus explicit per-head config weights.
  - Local CPU smoke on seed `3701` completed with `torch==2.12.1` and
    `device=cpu`.
  - The result was negative for CUDA escalation: the existing single combined
    `ZU_torch_gotoobj_state_plus_target_visibility_b0075` baseline won
    final-window success (`0.500`) and return (`0.367`), while the two-head
    variants reached only `0.200`/`0.300` success and `0.156`/`0.151` return.
  - Do not launch CUDA for v2.31.
- Issue #56 v2.32 two-head diagnostics and gating is implemented and has a CPU
  result:
  - `configs/experiments/minigrid-torch-adda-v43.json`
  - `docs/experiments/minigrid-torch-adda-v43.md`
  - Added per-head diagnostics for two-head representation learning:
    state-delta raw loss, target-visibility raw loss, and effective per-head
    beta values.
  - Added `representation_schedule=linear_anneal` with per-head beta end
    values for the two-head objective.
  - Local CPU smoke on seed `3801` completed with `torch==2.12.1` and
    `device=cpu`.
  - The result was negative for CUDA escalation: `ZU` still won final-window
    success (`0.400`), while constant two-head reached `0.300`, anneal-to-zero
    reached `0.100`, and AD-only stop reached `0.200`.
  - The useful diagnostic is that active two-head state-delta raw loss was about
    2x target-visibility raw loss (`0.0326` vs `0.0157`), suggesting equal
    head pressure is the wrong protocol.
  - Do not launch CUDA for v2.32.

## Next

- Start a visibility-first or state-head-downweighted branch. The next protocol
  should not apply equal pressure to state-delta and target-visibility heads;
  use the v2.32 diagnostics to bias toward target-visibility or decouple the
  representation probe from DQN training.

## Not Yet Proven

- Strict CUDA smoke on `gpu-worker-b`; it remains blocked by driver/wheel
  compatibility and needs an explicit external state change before rerun.
- A broader stability claim beyond the bounded three-seed CUDA sweep.
- A redesigned objective that beats the no-representation curriculum and the
  current best representation baselines under the mission-preservation probe.
  v2.23's single-head mission-conditioned target did not pass this gate, while
  v2.24 beta `0.1` passed CPU, CUDA replication, and bounded three-seed CUDA
  gates. v2.29 beta `0.075` is now the strongest beta-neighborhood candidate
  from a bounded three-seed CUDA sweep, but v2.30 did not preserve that edge
  cleanly under a longer CPU horizon and v2.31's naive two-head split
  underperformed the single combined `ZU` baseline on CPU. v2.32's diagnostics
  and anneal/gate variants also underperformed `ZU`, but identified state-head
  loss pressure as the likely two-head failure mode.
- Full objective completion.
