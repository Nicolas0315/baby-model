# baby-model Status

Updated: 2026-06-28 JST

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
- Fleet read-only check from the current Mac confirmed one macOS worker and
  three Windows/WSL workers are reachable enough for future approved lanes.
  Exact hostnames and GPU inventory are kept outside this repository in local
  docs evidence.
- `agmsg` is joined locally for turn-based progress delivery. Team details stay
  out of committed docs.
- GitHub Actions workflow exists, but remote CI is currently blocked by GitHub
  account billing/spending-limit state before runner startup. Local verifier is
  the authoritative verification until that account-level blocker is cleared.
- GitHub issues:
  - Fleet worker protocol: `https://github.com/Nicolas0315/baby-model/issues/1`
  - MiniGrid/BabyAI migration: `https://github.com/Nicolas0315/baby-model/issues/2`
  - CI billing/spending-limit blocker: `https://github.com/Nicolas0315/baby-model/issues/3`
  - v0.1 prediction-improvement reward: `https://github.com/Nicolas0315/baby-model/issues/4` closed
  - v0.2 progress-reward tuning/richer task: `https://github.com/Nicolas0315/baby-model/issues/5`
- GitHub tracking:
  - milestone: `v0.1 Baby AD/DA`
  - labels: `experiment`, `fleet`, `ci-blocker`, `research-framework`

## Next

- Create GitHub repository and push after local verification.
- Open GitHub issues for MiniGrid/BabyAI migration and fleet worker lanes.
- Tune `D_baby_progress` beta and test a richer sparse-reward task where
  intrinsic reward has room to help.
- Prepare remote worker checkouts after the GitHub repository is available.
- Clear GitHub Actions billing/spending-limit blocker and rerun `verify`.

## Not Yet Proven

- Remote checkout and remote tmux worker loops.
- MiniGrid/BabyAI dependency install.
- Any GPU training result.
- GitHub-hosted CI execution.
- Full objective completion.
