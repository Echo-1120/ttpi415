from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from tt_deep_rl.networks import ActorCritic, ModelConfig, QNetwork


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CartPole rollout transitions for structure diagnostics."
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument(
        "--policy",
        choices=("random", "dqn_greedy", "dqn_epsilon_greedy", "ppo_greedy", "ppo_stochastic"),
        default="random",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
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


def _model_config_from_payload(
    raw_checkpoint: Any,
    fallback_config: ModelConfig,
) -> ModelConfig:
    if not isinstance(raw_checkpoint, dict):
        return fallback_config

    payload = raw_checkpoint.get("model_config")
    if not isinstance(payload, dict):
        return fallback_config

    return ModelConfig(
        actor_arch=payload.get("actor_arch", fallback_config.actor_arch),
        critic_arch=payload.get("critic_arch", fallback_config.critic_arch),
        hidden_dims=tuple(payload.get("hidden_dims", fallback_config.hidden_dims)),
        latent_dim=int(payload.get("latent_dim", fallback_config.latent_dim)),
        activation=payload.get("activation", fallback_config.activation),
        tt_rank=int(payload.get("tt_rank", fallback_config.tt_rank)),
        tt_order=int(payload.get("tt_order", fallback_config.tt_order)),
    )


def load_ppo_agent(
    checkpoint_path: str,
    obs_dim: int,
    action_space: gym.Space,
    device: str,
    fallback_config: ModelConfig,
) -> tuple[ActorCritic, ModelConfig, dict[str, Any]]:
    if not checkpoint_path:
        raise ValueError("A checkpoint path is required for PPO rollout export.")

    raw_checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = _model_config_from_payload(raw_checkpoint, fallback_config)
    agent = ActorCritic(obs_dim=obs_dim, action_space=action_space, config=model_config).to(device)

    state_dict: dict[str, Any] | None = None
    if isinstance(raw_checkpoint, dict):
        for key in ("actor_critic_state_dict", "agent_state_dict", "model_state_dict", "state_dict"):
            candidate = raw_checkpoint.get(key)
            if isinstance(candidate, dict) and all(isinstance(value, torch.Tensor) for value in candidate.values()):
                state_dict = candidate
                break
        if state_dict is None and all(isinstance(value, torch.Tensor) for value in raw_checkpoint.values()):
            state_dict = raw_checkpoint

    if state_dict is None:
        raise ValueError(f"Could not find a usable PPO state_dict in checkpoint: {checkpoint_path}")

    agent.load_state_dict(state_dict)
    agent.eval()
    ppo_config = raw_checkpoint.get("ppo_config", {}) if isinstance(raw_checkpoint, dict) else {}
    return agent, model_config, ppo_config


def choose_action(
    obs: np.ndarray,
    env: gym.Env,
    policy: str,
    q_network: QNetwork | None,
    ppo_agent: ActorCritic | None,
    epsilon: float,
    device: torch.device,
) -> tuple[int, float]:
    if policy == "random":
        return int(env.action_space.sample()), float("nan")

    if policy in {"ppo_greedy", "ppo_stochastic"}:
        assert ppo_agent is not None
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            distribution = ppo_agent.policy.distribution(obs_tensor)
            value = ppo_agent.values(obs_tensor).squeeze(0)
        if not ppo_agent.is_discrete:
            raise ValueError("PPO rollout export currently supports only discrete action spaces.")
        if policy == "ppo_stochastic":
            action = int(distribution.sample().item())
        else:
            action = int(torch.argmax(distribution.logits, dim=-1).item())
        return action, float(value.item())

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
    ppo_agent: ActorCritic | None,
    device: torch.device,
) -> float:
    if ppo_agent is not None:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            value = ppo_agent.values(obs_tensor).squeeze(0)
        return float(value.item())
    if q_network is None:
        return float("nan")
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(obs_tensor).squeeze(0)
    return float(q_values.max().item())


def compute_gae_advantages(
    rewards: np.ndarray,
    dones: np.ndarray,
    state_values: np.ndarray,
    next_state_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_advantage = 0.0

    for index in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - float(dones[index])
        delta = (
            float(rewards[index])
            + gamma * float(next_state_values[index]) * next_non_terminal
            - float(state_values[index])
        )
        running_advantage = delta + gamma * gae_lambda * next_non_terminal * running_advantage
        advantages[index] = running_advantage
        returns[index] = running_advantage + float(state_values[index])

    return advantages, returns


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
        actor_arch=args.critic_arch,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        latent_dim=args.latent_dim,
        activation=args.activation,
        tt_rank=args.tt_rank,
        tt_order=args.tt_order,
    )

    q_network = None
    ppo_agent = None
    ppo_config_payload: dict[str, Any] = {}
    if args.policy in {"dqn_greedy", "dqn_epsilon_greedy"}:
        q_network = load_q_network(
            checkpoint_path=args.checkpoint,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=args.device,
            model_config=model_config,
        )
    elif args.policy in {"ppo_greedy", "ppo_stochastic"}:
        ppo_agent, model_config, ppo_config_payload = load_ppo_agent(
            checkpoint_path=args.checkpoint,
            obs_dim=obs_dim,
            action_space=env.action_space,
            device=args.device,
            fallback_config=model_config,
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
            ppo_agent=ppo_agent,
            epsilon=args.epsilon,
            device=device,
        )
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_value = compute_state_value(next_obs, q_network, ppo_agent, device)

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
    state_values_array = np.asarray(state_values, dtype=np.float32)
    next_state_values_array = np.asarray(next_state_values, dtype=np.float32)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    dones_array = np.asarray(dones, dtype=np.bool_)

    advantages_array = np.full_like(rewards_array, np.nan, dtype=np.float32)
    returns_array = np.full_like(rewards_array, np.nan, dtype=np.float32)
    effective_gamma = args.gamma
    effective_gae_lambda = args.gae_lambda
    if args.policy in {"ppo_greedy", "ppo_stochastic"}:
        if effective_gamma is None:
            effective_gamma = float(ppo_config_payload.get("gamma", 0.99))
        if effective_gae_lambda is None:
            effective_gae_lambda = float(ppo_config_payload.get("gae_lambda", 0.95))
        advantages_array, returns_array = compute_gae_advantages(
            rewards=rewards_array,
            dones=dones_array.astype(np.float32),
            state_values=state_values_array,
            next_state_values=next_state_values_array,
            gamma=effective_gamma,
            gae_lambda=effective_gae_lambda,
        )

    np.savez_compressed(
        output_path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=rewards_array,
        next_observations=np.asarray(next_observations, dtype=np.float32),
        dones=dones_array,
        state_values=state_values_array,
        next_state_values=next_state_values_array,
        advantages=advantages_array,
        returns=returns_array,
    )

    summary = {
        "env_id": args.env_id,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "policy": args.policy,
        "checkpoint": args.checkpoint,
        "epsilon": args.epsilon if args.policy == "dqn_epsilon_greedy" else None,
        "gamma": effective_gamma if args.policy in {"ppo_greedy", "ppo_stochastic"} else None,
        "gae_lambda": effective_gae_lambda if args.policy in {"ppo_greedy", "ppo_stochastic"} else None,
        "model_config": asdict(model_config),
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
