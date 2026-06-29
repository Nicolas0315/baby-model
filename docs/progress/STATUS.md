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
  - v0.5 function approximation pilot: `https://github.com/Nicolas0315/baby-model/issues/9`
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

## Next

- Run a multi-seed linear sweep or a stronger optional neural encoder lane.

## Not Yet Proven

- Any GPU training result.
- Full objective completion.
