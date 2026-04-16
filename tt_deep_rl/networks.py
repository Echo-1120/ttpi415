from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from tt_deep_rl.tt_layers import TTMLP


def _activation_from_name(name: str) -> type[nn.Module]:
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    return nn.Tanh


class MLPBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        dims = (input_dim,) + hidden_dims + (output_dim,)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != output_dim:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def compression_stats(self) -> dict[str, float]:
        params = sum(parameter.numel() for parameter in self.parameters())
        return {"module_params": float(params)}


class TTBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: type[nn.Module],
        tt_rank: int,
        tt_order: int,
    ) -> None:
        super().__init__()
        self.net = TTMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            tt_rank=tt_rank,
            tt_order=tt_order,
            activation=activation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def compression_stats(self) -> dict[str, float]:
        return self.net.compression_stats()


class HybridBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: type[nn.Module],
        tt_rank: int,
        tt_order: int,
    ) -> None:
        super().__init__()
        latent_dim = hidden_dims[-1] if hidden_dims else output_dim
        self.mlp_path = MLPBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            activation=activation,
        )
        self.tt_path = TTBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            tt_rank=tt_rank,
            tt_order=tt_order,
        )
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim * 2, output_dim),
            activation(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mlp_features = self.mlp_path(inputs)
        tt_features = self.tt_path(inputs)
        fused = torch.cat([mlp_features, tt_features], dim=-1)
        return self.fuse(fused)

    def compression_stats(self) -> dict[str, float]:
        stats = self.tt_path.compression_stats()
        stats["module_params"] = float(sum(parameter.numel() for parameter in self.parameters()))
        return stats


def build_backbone(
    arch: str,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation_name: str,
    tt_rank: int,
    tt_order: int,
) -> nn.Module:
    activation = _activation_from_name(activation_name)
    if arch == "tt":
        return TTBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            tt_rank=tt_rank,
            tt_order=tt_order,
        )
    if arch == "hybrid":
        return HybridBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            tt_rank=tt_rank,
            tt_order=tt_order,
        )
    return MLPBackbone(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
    )


@dataclass
class ModelConfig:
    actor_arch: str = "mlp"
    critic_arch: str = "mlp"
    hidden_dims: tuple[int, ...] = (64, 64)
    latent_dim: int = 64
    activation: str = "tanh"
    tt_rank: int = 4
    tt_order: int = 3


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_space: gym.Space,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.backbone = build_backbone(
            arch=config.actor_arch,
            input_dim=obs_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.latent_dim,
            activation_name=config.activation,
            tt_rank=config.tt_rank,
            tt_order=config.tt_order,
        )
        if self.is_discrete:
            self.action_dim = int(action_space.n)
            self.policy_head = nn.Linear(config.latent_dim, self.action_dim)
            self.log_std = None
        else:
            assert isinstance(action_space, gym.spaces.Box)
            self.action_dim = int(np.prod(action_space.shape))
            self.policy_head = nn.Linear(config.latent_dim, self.action_dim)
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def distribution(self, observations: torch.Tensor):
        features = self.backbone(observations)
        if self.is_discrete:
            logits = self.policy_head(features)
            return Categorical(logits=logits)

        mean = self.policy_head(features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def compression_stats(self) -> dict[str, float]:
        return self.backbone.compression_stats()


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = build_backbone(
            arch=config.critic_arch,
            input_dim=obs_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.latent_dim,
            activation_name=config.activation,
            tt_rank=config.tt_rank,
            tt_order=config.tt_order,
        )
        self.value_head = nn.Linear(config.latent_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.backbone(observations)
        return self.value_head(features).squeeze(-1)

    def compression_stats(self) -> dict[str, float]:
        return self.backbone.compression_stats()


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_space: gym.Space,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.policy = PolicyNetwork(obs_dim, action_space, config)
        self.value_function = ValueNetwork(obs_dim, config)
        self.is_discrete = self.policy.is_discrete
        self.action_space = action_space

    def act(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.policy.distribution(observations)
        actions = distribution.sample()
        log_prob = self._log_prob(distribution, actions)
        values = self.value_function(observations)
        return actions, log_prob, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.policy.distribution(observations)
        log_prob = self._log_prob(distribution, actions)
        entropy = self._entropy(distribution)
        values = self.value_function(observations)
        return log_prob, entropy, values

    def values(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_function(observations)

    def _log_prob(self, distribution, actions: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            return distribution.log_prob(actions.long())
        return distribution.log_prob(actions).sum(dim=-1)

    def _entropy(self, distribution) -> torch.Tensor:
        entropy = distribution.entropy()
        if self.is_discrete:
            return entropy
        return entropy.sum(dim=-1)

    def environment_action(self, actions: torch.Tensor):
        if self.is_discrete:
            return int(actions.squeeze(0).item())

        assert isinstance(self.action_space, gym.spaces.Box)
        low = torch.as_tensor(self.action_space.low, dtype=actions.dtype, device=actions.device)
        high = torch.as_tensor(self.action_space.high, dtype=actions.dtype, device=actions.device)
        clipped = torch.clamp(actions.squeeze(0), low, high)
        return clipped.cpu().numpy()

# ==================== 新增：TT 兼容的 QNetwork ====================
class QNetwork(nn.Module):
    """输出每个离散动作的 Q 值（shape: [batch, action_dim]）
    完全复用现有的 build_backbone，支持 mlp / tt / hybrid
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            arch=config.critic_arch,          # 复用 critic-arch 参数
            input_dim=obs_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.latent_dim,
            activation_name=config.activation,
            tt_rank=config.tt_rank,
            tt_order=config.tt_order,
        )
        self.q_head = nn.Linear(config.latent_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.backbone(observations)
        return self.q_head(features)

    def compression_stats(self) -> dict[str, float]:
        stats = self.backbone.compression_stats()
        stats["module_params"] = float(sum(p.numel() for p in self.parameters()))
        return stats