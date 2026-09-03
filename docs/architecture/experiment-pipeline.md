# Experiment Pipeline: Control Flow and Artifact Model

Date: 2026-08-23 JST

Reference for what actually executes in the PyTorch AD/DA lane and what is
persisted afterwards. Written from the code at `56f0c3d` plus the v2.46 run
artifacts, not from the protocol documents.

Both Mermaid diagrams below were rendered with `mmdc` 11.16.0 to confirm they
parse.

## Sequence: one sweep

```mermaid
sequenceDiagram
    autonumber
    participant CLI as minigrid_torch_sweep.main
    participant SW as run_minigrid_torch_sweep
    participant SU as run_minigrid_torch_suite
    participant CC as run_minigrid_torch_curriculum_condition
    participant ST as _run_minigrid_torch_stage
    participant ENV as gymnasium / minigrid
    participant AG as TorchDQNAgent
    participant HO as run_greedy_holdout
    participant STAT as baby_model.stats

    CLI->>SW: config JSON, seeds, --device
    loop for each seed
        SW->>SU: run_minigrid_torch_suite(config, seed)
        SU->>SU: parse_minigrid_torch_config
        SU->>SU: torch.manual_seed(seed) then select_torch_device
        loop for each condition (config order)
            SU->>CC: stages, active_stages, condition, device
            loop for each active stage
                CC->>ST: stage, agent (carried over), global_episode
                ST->>ENV: gym.make(stage.env_id)
                Note over ST,ENV: the env truncates at its own max_steps<br/>stage.max_steps is only a ceiling
                ST->>AG: construct on first stage only (CPU init, then .to(device))
                loop for each episode in stage
                    ST->>ENV: reset(seed = cond.seed*100000 + stage_index*1000 + episode)
                    loop for each step until done
                        ST->>AG: choose(features, force_random = global_episode < decoder_delay)
                        AG-->>ST: action (epsilon-greedy, epsilon fixed at 0.2)
                        ST->>ENV: step(action)
                        ST->>AG: update_representation(...) if representation active
                        ST->>AG: update(...) only when not force_random
                    end
                    ST->>ST: append EpisodeMetrics (in memory only)
                end
                ST-->>CC: stage summary — aggregates only, per-episode rows discarded
            end
            CC->>HO: if holdout_episodes > 0
            HO->>ENV: gym.make(final stage env_id)
            HO->>AG: set epsilon to 0, no update calls
            Note over HO,ENV: reset(seed = cond.seed*100000 + 99000 + i)<br/>disjoint from the training range
            HO-->>CC: holdout_success_rate / mean_return / mean_steps
            CC-->>SU: condition result
        end
        SU-->>SW: suite report (winner_last_window)
    end
    SW->>SW: aggregate_torch_reports
    SW->>STAT: analyze_report (paired sign-flip vs baseline_condition)
    STAT-->>SW: statistics block
    SW->>CLI: metrics.json + summary.md + latest symlink
```

Three properties of this flow drive most of the verification findings:

- **The agent is created once per condition and carried across stages.** The
  curriculum is a single continuous training run, so `decoder_delay_episodes` is
  counted on a *global* episode index, not per stage.
- **Randomness is entirely CPU-side.** `torch.manual_seed(seed)` seeds weight
  initialization, which happens before `.to(device)`; there is no dropout and no
  CUDA RNG draw anywhere. Action selection and replay sampling both come from
  one `random.Random(seed)` instance inside the agent. Nothing in the loop
  consumes a device-dependent random stream.
- **`EpisodeMetrics` never leaves memory.** Only stage-level aggregates are
  written, so no artifact contains a learning curve.

## Entity model: what a run persists

```mermaid
erDiagram
    CONFIG_FILE ||--o{ STAGE_SPEC : "stages[]"
    CONFIG_FILE ||--o{ CONDITION_SPEC : "conditions[]"
    CONFIG_FILE ||--|| AGENT_SPEC : "agent"
    CONFIG_FILE ||--o| ENVIRONMENT_SPEC : "environment"

    SWEEP_REPORT ||--o{ SUITE_RUN : "runs[]  (one per seed)"
    SWEEP_REPORT ||--o{ AGGREGATE_ROW : "aggregate[]  (one per condition)"
    SWEEP_REPORT ||--|| STATISTICS : "statistics"
    SWEEP_REPORT ||--|| SUMMARY_MD : "rendered as"

    SUITE_RUN ||--o{ CONDITION_RESULT : "results[]"
    SUITE_RUN ||--|| FRAMEWORK : "framework"

    CONDITION_RESULT ||--o{ STAGE_RESULT : "stage_results[]"
    CONDITION_RESULT ||--|| STAGE_RESULT : "final_stage"
    CONDITION_RESULT ||--o| HOLDOUT : "holdout_* keys"
    CONDITION_RESULT }o--|| CONDITION_SPEC : "instantiates"
    STAGE_RESULT }o--|| STAGE_SPEC : "instantiates"

    STATISTICS ||--o{ METRIC_BLOCK : "metrics{}"
    METRIC_BLOCK ||--o{ CONDITION_STATS : "per condition"
    CONDITION_STATS ||--o| PAIRED_COMPARISON : "vs_baseline"

    STAGE_RESULT ||..o{ EPISODE_METRICS : "NOT PERSISTED"

    CONFIG_FILE {
        string hypothesis
        string baseline_condition "optional, defaults to conditions[0]"
        int holdout_episodes "0 = off"
    }
    STAGE_SPEC {
        string name PK
        string env_id
        int max_steps "ceiling only; env truncation may bind first"
        int episodes
    }
    CONDITION_SPEC {
        string name PK
        string encoder_mode
        string representation_objective
        float representation_beta
        int decoder_delay_episodes "counted globally across stages"
        bool stop_representation_after_delay
        bool freeze_encoder_after_delay
    }
    AGENT_SPEC {
        int feature_dim "hashed bucket count"
        int hidden_dim
        float epsilon "constant; never annealed"
        float learning_rate
        float gamma
        int batch_size
        int replay_capacity
        int target_sync_updates
        string device
    }
    SUITE_RUN {
        int seed PK
        string winner_last_window
    }
    FRAMEWORK {
        string version
        string device
        bool cuda_available
        string gpu_name "MISSING: not recorded"
        string driver_version "MISSING: not recorded"
        string host "MISSING: not recorded"
    }
    CONDITION_RESULT {
        string name PK
        int seed PK
        float success_rate_all
        float success_rate_last_window "decision metric through v2.45"
        float mean_return_last_window
        float mission_target_visible_rate_last_window
        int representation_updates
        int updates "cumulative, not per stage"
        int parameter_count
    }
    STAGE_RESULT {
        string stage PK
        string env_id
        int episodes
        float success_rate_all
        float success_rate_last_window "last 20 episodes of this stage"
        int updates "cumulative agent total, not stage-local"
    }
    HOLDOUT {
        string holdout_env_id
        int holdout_episodes
        float holdout_success_rate
        float holdout_mean_return
        float holdout_mean_steps
    }
    STATISTICS {
        string baseline
        int seeds
    }
    PAIRED_COMPARISON {
        float mean_diff
        float ci95_low
        float ci95_high
        float p_two_sided "exact floor = 2 / 2**n"
        int wins
        int losses
        int ties
    }
    EPISODE_METRICS {
        bool success
        int steps
        float external_return
        float intrinsic_return
        int unique_features
    }
    AGGREGATE_ROW {
        string name PK
        int win_count
        float mean_success_rate_last_window
        float mean_holdout_success_rate
    }
```

Two schema facts worth stating plainly, both confirmed against the v2.46
artifacts:

- `updates` on a `STAGE_RESULT` is the agent's **cumulative** counter at the end
  of that stage, not the number of updates performed in it. Reading the
  per-stage column as stage-local overstates late stages.
- `FRAMEWORK` records the torch version and the device string but **not** the
  GPU model, driver version, or host. Every worker/GPU claim in
  `docs/experiments/` was transcribed by hand into prose and cannot be checked
  against the artifact.
