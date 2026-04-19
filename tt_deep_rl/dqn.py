from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from tt_deep_rl.networks import ModelConfig, QNetwork


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DQNConfig:
    env_id: str = "CartPole-v1"
    seed: int = 7
    total_timesteps: int = 100000
    buffer_size: int = 50000
    batch_size: int = 128
    learning_starts: int = 5000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    target_update_freq: int = 1000
    train_freq: int = 4
    gradient_steps: int = 1
    eval_freq: int = 5000
    eval_episodes: int = 10
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append((obs.copy(), action, reward, next_obs.copy(), done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNTrainer:
    def __init__(self, config: DQNConfig, model_config: ModelConfig) -> None:
        self.config = config
        self.model_config = model_config
        set_seed(config.seed)

        self.env = gym.make(config.env_id)
        self.env.action_space.seed(config.seed)
        self.env.observation_space.seed(config.seed)

        obs_sample, _ = self.env.reset(seed=config.seed)
        self.obs_dim = int(np.prod(obs_sample.shape))
        self.action_dim = int(self.env.action_space.n)

        self.device = torch.device(config.device)
        self.online_q = QNetwork(self.obs_dim, self.action_dim, model_config).to(self.device)
        self.target_q = QNetwork(self.obs_dim, self.action_dim, model_config).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())

        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.buffer_size, self.obs_dim)

        self.epsilon = config.epsilon_start
        self.gradient_step_counter = 0

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return int(self.env.action_space.sample())
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_q(obs_t)
        return int(q_values.argmax(dim=1).item())

    def current_epsilon(self, env_step: int) -> float:
        if env_step < self.config.learning_starts:
            return self.config.epsilon_start
        frac = min(
            1.0,
            (env_step - self.config.learning_starts) / self.config.epsilon_decay_steps,
        )
        epsilon = self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )
        return float(max(self.config.epsilon_end, epsilon))

    def train_one_step(self) -> dict[str, float]:
        min_replay_size = max(self.config.batch_size, self.config.learning_starts)
        if len(self.replay) < min_replay_size:
            return {"q_loss": 0.0, "epsilon": self.epsilon}

        obs, action, reward, next_obs, done = self.replay.sample(self.config.batch_size)
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_q_online = self.online_q(next_obs)
            next_action = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_q(next_obs).gather(1, next_action).squeeze(1)
            target = reward + self.config.gamma * (1 - done) * next_q_target

        current_q = self.online_q(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_q.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.gradient_step_counter += 1
        if self.gradient_step_counter % self.config.target_update_freq == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())

        return {"q_loss": float(loss.item()), "epsilon": float(self.epsilon)}

    def evaluate(self) -> float:
        eval_env = gym.make(self.config.env_id)
        eval_returns: list[float] = []

        for episode_index in range(self.config.eval_episodes):
            obs, _ = eval_env.reset(seed=self.config.seed + 1000 + episode_index)
            done = False
            episode_return = 0.0

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = int(self.online_q(obs_t).argmax(dim=1).item())
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_return += float(reward)
                obs = next_obs

            eval_returns.append(episode_return)

        eval_env.close()
        return float(np.mean(eval_returns)) if eval_returns else 0.0

    def train(self) -> dict[str, Any]:
        obs, _ = self.env.reset(seed=self.config.seed)
        episode_return = 0.0
        episode_returns: list[float] = []
        metrics: list[dict[str, float]] = []
        latest_eval_avg_return = 0.0

        for step in range(self.config.total_timesteps):
            self.epsilon = self.current_epsilon(step)
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay.add(obs, action, reward, next_obs, done)
            episode_return += float(reward)

            if done:
                episode_returns.append(episode_return)
                obs, _ = self.env.reset()
                episode_return = 0.0
            else:
                obs = next_obs

            loss_dict = {"q_loss": 0.0, "epsilon": float(self.epsilon)}
            if step >= self.config.learning_starts and step % self.config.train_freq == 0:
                q_losses = []
                for _ in range(self.config.gradient_steps):
                    train_metrics = self.train_one_step()
                    q_losses.append(train_metrics["q_loss"])
                loss_dict = {
                    "q_loss": float(np.mean(q_losses)) if q_losses else 0.0,
                    "epsilon": float(self.epsilon),
                }

            if step > 0 and step % self.config.eval_freq == 0:
                latest_eval_avg_return = self.evaluate()

            if step % 100 == 0:
                metrics.append(
                    {
                        "step": step,
                        "avg_return": float(np.mean(episode_returns[-10:])) if episode_returns else 0.0,
                        "eval_avg_return": float(latest_eval_avg_return),
                        **loss_dict,
                    }
                )

            if step % 1000 == 0:
                print(
                    f"Step {step:6d} | "
                    f"Train Avg Return: {metrics[-1]['avg_return']:.2f} | "
                    f"Eval Avg Return: {latest_eval_avg_return:.2f} | "
                    f"epsilon: {self.epsilon:.3f}"
                )

        self.env.close()
        return {
            "final_avg_return": float(np.mean(episode_returns[-20:])) if episode_returns else 0.0,
            "final_eval_avg_return": float(latest_eval_avg_return),
            "episodes_finished": len(episode_returns),
            "metrics": metrics,
            "compression": self.online_q.compression_stats(),
        }
