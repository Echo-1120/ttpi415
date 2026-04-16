from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tt_deep_rl.networks import ModelConfig, QNetwork

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@dataclass
class DQNConfig:
    env_id: str = "CartPole-v1"
    seed: int = 7
    total_timesteps: int = 20000          # CartPole 通常需要更多步数
    buffer_size: int = 10000
    batch_size: int = 64
    # DQNConfig
    learning_starts: int = 2000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    learning_rate: float = 3e-4
    gamma: float = 0.99
    target_update_freq: int = 500         # 每多少 step 更新 target net
    epsilon_decay_steps: int = 20000        # 从 epsilon_start 线性衰减到 epsilon_end 的步数
    epsilon_decay: float = 0.995
    device: str = "cpu"

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim

    def add(self, obs, action, reward, next_obs, done):
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

    def __len__(self):
        return len(self.buffer)

class DQNTrainer:
    def __init__(self, config: DQNConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        set_seed(config.seed)

        self.env = gym.make(config.env_id)
        self.env.action_space.seed(config.seed)
        self.env.observation_space.seed(config.seed)

        obs_sample, _ = self.env.reset(seed=config.seed)
        self.obs_dim = int(np.prod(obs_sample.shape))
        self.action_dim = int(self.env.action_space.n)   # CartPole-v1 = 2

        self.device = torch.device(config.device)

        # online net + target net（都支持 TT）
        self.online_q = QNetwork(self.obs_dim, self.action_dim, model_config).to(self.device)
        self.target_q = QNetwork(self.obs_dim, self.action_dim, model_config).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())

        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.buffer_size, self.obs_dim)

        self.epsilon = config.epsilon_start
        self.step_counter = 0

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_q(obs_t)
        return q_values.argmax(dim=1).item()

    def train_one_step(self) -> dict[str, float]:
        if len(self.replay) < max(self.config.batch_size, self.config.learning_starts):
            return {"q_loss": 0.0, "epsilon": self.epsilon}

        obs, action, reward, next_obs, done = self.replay.sample(self.config.batch_size)
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # Double DQN TD target
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

        # 更新 target net
        self.step_counter += 1
        if self.step_counter % self.config.target_update_freq == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())


        return {"q_loss": loss.item(), "epsilon": self.epsilon}

    def current_epsilon(self, env_step: int) -> float:
        frac = min(1.0, env_step / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def train(self) -> dict[str, Any]:
        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_returns = []
        metrics = []

        for step in range(self.config.total_timesteps):
            self.epsilon = self.current_epsilon(step)
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay.add(obs, action, reward, next_obs, done)
            episode_return += reward

            if done:
                episode_returns.append(episode_return)
                obs, _ = self.env.reset()
                episode_return = 0.0
            else:
                obs = next_obs

            loss_dict = self.train_one_step()
            if step % 100 == 0:
                metrics.append({
                    "step": step,
                    "avg_return": np.mean(episode_returns[-10:]) if episode_returns else 0.0,
                    **loss_dict
                })

            if step % 1000 == 0:
                print(f"Step {step:5d} | Avg Return: {metrics[-1]['avg_return']:.2f} | ε: {self.epsilon:.3f}")

        return {
            "final_avg_return": float(np.mean(episode_returns[-20:])) if episode_returns else 0.0,
            "metrics": metrics,
            "compression": self.online_q.compression_stats()
        
        }