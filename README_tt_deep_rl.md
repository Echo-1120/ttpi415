# TT + Deep RL Research Starter

This folder turns your three staged directions into one shared prototype:

1. `TT as value function approximator`
   - run with `--actor-arch mlp --critic-arch tt`
2. `TT for policy compression`
   - run with `--actor-arch tt --critic-arch mlp`
3. `Hybrid TT + neural network architecture`
   - run with `--actor-arch hybrid --critic-arch hybrid`

## Why this prototype is shaped this way

Your seed papers already occupy part of the space:

- `TTPI` uses TT to approximate the state-value and advantage function, then uses TTGO for policy retrieval in hybrid control.
- `TEQL` uses a low-rank CP tensor Q-function plus uncertainty-aware exploration.
- `ACTeN` shows tensor networks can serve as both actor and critic approximators in a model-free actor-critic pipeline.

That means a paper is unlikely to be novel if it only says:

- "we replaced an MLP critic with TT", or
- "we compressed a policy with TT and kept reward roughly unchanged".

The more defensible novelty targets are:

1. **Stage 1: TT critic is only a baseline unless you add one structural mechanism**
   - Examples: adaptive TT rank, uncertainty-aware TT critic, continuous-state no-discretization TT critic, or explicit factorized observation priors.
2. **Stage 2: TT policy compression becomes publishable when compression is part of learning, not just post-hoc pruning**
   - Examples: train-time rank scheduling, latency-aware objectives, robustness under low parameter budgets, or deployment-constrained RL.
3. **Stage 3: hybrid TT + NN is the strongest main direction**
   - The cleanest claim is: the NN path handles local nonlinearities, the TT path captures factorized global structure, and the combination improves the sample-efficiency / parameter-efficiency frontier.

## Suggested novelty bar

If you want "enough innovation" relative to the current literature, a practical bar is:

- At least **one algorithmic idea**, not just a new backbone.
- At least **one setting where TT structure is genuinely useful**, not just CartPole.
- At least **one Pareto-style claim**:
  - better return under same parameter budget, or
  - better sample efficiency under same wall-clock budget, or
  - lower latency / memory under matched return.
- At least **three ablations**:
  - MLP only
  - TT only
  - Hybrid
- At least **one structure-sensitive task family** where factorized state or action design matters.

## Minimal run commands

Use the `pytorch-cpu-mac` conda environment on this machine.

```bash
conda run -n pytorch-cpu-mac python ./train_ppo.py \
  --env-id CartPole-v1 \
  --actor-arch mlp \
  --critic-arch tt \
  --total-timesteps 512 \
  --rollout-steps 128
```

```bash
conda run -n pytorch-cpu-mac python ./train_ppo.py \
  --env-id CartPole-v1 \
  --actor-arch tt \
  --critic-arch mlp \
  --total-timesteps 512 \
  --rollout-steps 128
```

```bash
conda run -n pytorch-cpu-mac python ./train_ppo.py \
  --env-id CartPole-v1 \
  --actor-arch hybrid \
  --critic-arch hybrid \
  --total-timesteps 512 \
  --rollout-steps 128
```

## Next steps I recommend

1. Start with `CartPole-v1` only to validate optimization and logging.
2. Move to a factorized or higher-dimensional task where TT should help.
3. Add a research feature before scaling experiments:
   - adaptive rank
   - uncertainty bonus from TT residuals
   - gated hybrid routing
   - budget-aware objective
4. Compare against matched-parameter MLP baselines, not just default MLP sizes.
