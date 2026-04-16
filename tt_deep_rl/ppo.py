from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from tt_deep_rl.networks import ActorCritic, ModelConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    seed: int = 7
    total_timesteps: int = 4096
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"


class SimpleAdam:
    """Tiny Adam optimizer to avoid environment-specific torch.optim issues."""

    def __init__(
        self,
        parameters,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.first_moment = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.second_moment = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.steps = 0

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()

    def step(self) -> None:
        self.steps += 1
        bias_correction1 = 1.0 - self.beta1 ** self.steps
        bias_correction2 = 1.0 - self.beta2 ** self.steps

        for parameter, first_moment, second_moment in zip(
            self.parameters,
            self.first_moment,
            self.second_moment,
        ):
            if parameter.grad is None:
                continue

            gradient = parameter.grad
            first_moment.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            second_moment.mul_(self.beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - self.beta2,
            )

            first_unbiased = first_moment / bias_correction1
            second_unbiased = second_moment / bias_correction2
            denominator = second_unbiased.sqrt().add_(self.eps)
            parameter.data.addcdiv_(first_unbiased, denominator, value=-self.lr)


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int, action_dim: int, is_discrete: bool) -> None:
        self.rollout_steps = rollout_steps
        self.is_discrete = is_discrete
        self.observations = torch.zeros(rollout_steps, obs_dim, dtype=torch.float32)
        action_dtype = torch.long if is_discrete else torch.float32
        self.actions = torch.zeros(rollout_steps, action_dim, dtype=action_dtype)
        self.log_probs = torch.zeros(rollout_steps, dtype=torch.float32)
        self.rewards = torch.zeros(rollout_steps, dtype=torch.float32)
        self.dones = torch.zeros(rollout_steps, dtype=torch.float32)
        self.values = torch.zeros(rollout_steps, dtype=torch.float32)
        self.advantages = torch.zeros(rollout_steps, dtype=torch.float32)
        self.returns = torch.zeros(rollout_steps, dtype=torch.float32)
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        index = self.position
        self.observations[index] = torch.as_tensor(observation, dtype=torch.float32)
        if self.is_discrete:
            self.actions[index, 0] = action.squeeze().long()
        else:
            self.actions[index] = action.squeeze().float()
        self.log_probs[index] = log_prob.squeeze().float()
        self.rewards[index] = float(reward)
        self.dones[index] = float(done)
        self.values[index] = value.squeeze().float()
        self.position += 1

    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        advantage = torch.tensor(0.0)
        last_value = last_value.squeeze().float()
        for index in reversed(range(self.rollout_steps)):
            if index == self.rollout_steps - 1:
                next_non_terminal = 1.0 - self.dones[index]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[index]
                next_value = self.values[index + 1]
            delta = self.rewards[index] + gamma * next_value * next_non_terminal - self.values[index]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[index] = advantage
        self.returns = self.advantages + self.values

    def batches(self, minibatch_size: int):
        indices = torch.randperm(self.rollout_steps)
        for start in range(0, self.rollout_steps, minibatch_size):
            yield indices[start : start + minibatch_size]


class PPOTrainer:
    def __init__(self, config: PPOConfig, model_config: ModelConfig) -> None:
        self.config = config
        self.model_config = model_config
        set_seed(config.seed)

        self.env = gym.make(config.env_id)
        self.env.action_space.seed(config.seed)
        self.env.observation_space.seed(config.seed)

        obs_sample, _ = self.env.reset(seed=config.seed)
        self.obs_dim = int(np.prod(obs_sample.shape))
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = 1
        else:
            self.action_dim = int(np.prod(self.env.action_space.shape))

        self.device = torch.device(config.device)
        self.agent = ActorCritic(self.obs_dim, self.env.action_space, model_config).to(self.device)
        self.optimizer = SimpleAdam(self.agent.parameters(), lr=config.learning_rate)
        self.buffer = RolloutBuffer(
            rollout_steps=config.rollout_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            is_discrete=self.agent.is_discrete,
        )

    def train(
        self,
        rollout_callback: Callable[[RolloutBuffer, torch.Tensor], None] | None = None,
    ) -> dict[str, Any]:
        observation, _ = self.env.reset(seed=self.config.seed)
        episode_return = 0.0
        episode_returns: list[float] = []
        update_metrics: list[dict[str, float]] = []

        for _ in range(0, self.config.total_timesteps, self.config.rollout_steps):
            self.buffer.position = 0
            for _ in range(self.config.rollout_steps):
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = self.agent.act(obs_tensor)
                env_action = self.agent.environment_action(action)
                next_observation, reward, terminated, truncated, _ = self.env.step(env_action)
                done = terminated or truncated

                self.buffer.add(
                    observation=observation,
                    action=action.cpu(),
                    log_prob=log_prob.cpu(),
                    reward=float(reward),
                    done=done,
                    value=value.cpu(),
                )

                episode_return += float(reward)
                if done:
                    episode_returns.append(episode_return)
                    episode_return = 0.0
                    observation, _ = self.env.reset()
                else:
                    observation = next_observation

            with torch.no_grad():
                last_value = self.agent.values(
                    torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                ).cpu()
            self.buffer.compute_returns(
                last_value=last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            if rollout_callback is not None:
                rollout_callback(self.buffer, last_value.clone())
            update_metrics.append(self._update())

        summary = {
            "config": asdict(self.config),
            "model_config": asdict(self.model_config),
            "episodes_finished": len(episode_returns),
            "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "last_episode_return": float(episode_returns[-1]) if episode_returns else 0.0,
            "update_metrics": update_metrics,
            "actor_stats": self.agent.policy.compression_stats(),
            "critic_stats": self.agent.value_function.compression_stats(),
        }
        self.env.close()
        return summary

    def _update(self) -> dict[str, float]:
        observations = self.buffer.observations.to(self.device)
        actions = self.buffer.actions.to(self.device)
        old_log_probs = self.buffer.log_probs.to(self.device)
        advantages = self.buffer.advantages.to(self.device)
        returns = self.buffer.returns.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.agent.is_discrete:
            actions_for_eval = actions.squeeze(-1).long()
        else:
            actions_for_eval = actions

        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.config.update_epochs):
            for batch_indices in self.buffer.batches(self.config.minibatch_size):
                batch_obs = observations[batch_indices]
                batch_actions = actions_for_eval[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                new_log_probs, entropy, values = self.agent.evaluate_actions(batch_obs, batch_actions)
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = log_ratio.exp()

                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_bonus.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }


def save_summary(summary: dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
