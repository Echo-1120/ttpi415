from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from tt_deep_rl.networks import ModelConfig, QNetwork


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CartPole rollout transitions for structure diagnostics."
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument(
        "--policy",
        choices=("random", "dqn_greedy", "dqn_epsilon_greedy"),
        default="random",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--critic-arch", choices=("mlp", "tt", "hybrid"), default="mlp")
    parser.add_argument("--hidden-dims", default="128,128")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--activation", choices=("tanh", "relu", "gelu"), default="relu")
    parser.add_argument("--tt-rank", type=int, default=4)
    parser.add_argument("--tt-order", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-file", default="cartpole_rollouts.npz")
    return parser.parse_args()


def load_q_network(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    device: str,
    model_config: ModelConfig,
) -> QNetwork:
    if not checkpoint_path:
        raise ValueError("A checkpoint path is required for non-random rollout policies.")

    network = QNetwork(obs_dim=obs_dim, action_dim=action_dim, config=model_config).to(device)
    raw_checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict: dict[str, Any] | None = None
    if isinstance(raw_checkpoint, dict):
        if all(isinstance(value, torch.Tensor) for value in raw_checkpoint.values()):
            state_dict = raw_checkpoint
        else:
            for key in ("online_q", "online_q_state_dict", "model_state_dict", "state_dict"):
                candidate = raw_checkpoint.get(key)
                if isinstance(candidate, dict) and all(isinstance(value, torch.Tensor) for value in candidate.values()):
                    state_dict = candidate
                    break

    if state_dict is None:
        raise ValueError(f"Could not find a usable state_dict in checkpoint: {checkpoint_path}")

    network.load_state_dict(state_dict)
    network.eval()
    return network


def choose_action(
    obs: np.ndarray,
    env: gym.Env,
    policy: str,
    q_network: QNetwork | None,
    epsilon: float,
    device: torch.device,
) -> tuple[int, float]:
    if policy == "random":
        return int(env.action_space.sample()), float("nan")

    assert q_network is not None
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(obs_tensor).squeeze(0)
    state_value = float(q_values.max().item())

    if policy == "dqn_epsilon_greedy" and np.random.random() < epsilon:
        return int(env.action_space.sample()), state_value
    return int(q_values.argmax().item()), state_value


def compute_state_value(
    obs: np.ndarray,
    q_network: QNetwork | None,
    device: torch.device,
) -> float:
    if q_network is None:
        return float("nan")
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(obs_tensor).squeeze(0)
    return float(q_values.max().item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    initial_obs, _ = env.reset(seed=args.seed)
    obs_dim = int(np.prod(initial_obs.shape))
    action_dim = int(env.action_space.n)

    model_config = ModelConfig(
        critic_arch=args.critic_arch,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        latent_dim=args.latent_dim,
        activation=args.activation,
        tt_rank=args.tt_rank,
        tt_order=args.tt_order,
    )

    q_network = None
    if args.policy != "random":
        q_network = load_q_network(
            checkpoint_path=args.checkpoint,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=args.device,
            model_config=model_config,
        )

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[bool] = []
    state_values: list[float] = []
    next_state_values: list[float] = []

    obs = initial_obs
    episode_return = 0.0
    episode_returns: list[float] = []

    for _ in range(args.num_steps):
        action, state_value = choose_action(
            obs=obs,
            env=env,
            policy=args.policy,
            q_network=q_network,
            epsilon=args.epsilon,
            device=device,
        )
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_value = compute_state_value(next_obs, q_network, device)

        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        next_observations.append(np.asarray(next_obs, dtype=np.float32))
        dones.append(done)
        state_values.append(state_value)
        next_state_values.append(next_state_value)

        episode_return += float(reward)
        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    env.close()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        state_values=np.asarray(state_values, dtype=np.float32),
        next_state_values=np.asarray(next_state_values, dtype=np.float32),
    )

    summary = {
        "env_id": args.env_id,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "policy": args.policy,
        "checkpoint": args.checkpoint,
        "epsilon": args.epsilon if args.policy == "dqn_epsilon_greedy" else None,
        "episodes_finished": len(episode_returns),
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "last_episode_return": float(episode_returns[-1]) if episode_returns else 0.0,
        "output_file": str(output_path),
    }

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
