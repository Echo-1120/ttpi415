from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from tt_deep_rl.ppo import RolloutBuffer


DEFAULT_CARTPOLE_STATE_BINS = (8, 8, 8, 8)
DEFAULT_CARTPOLE_OBS_LOW = (-4.8, -3.0, -0.41887903, -3.5)
DEFAULT_CARTPOLE_OBS_HIGH = (4.8, 3.0, 0.41887903, 3.5)
DEFAULT_ACTION_BINS = 2
DEFAULT_TT_RANKS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class DiscretizationSpec:
    state_bins: tuple[int, ...]
    obs_low: tuple[float, ...]
    obs_high: tuple[float, ...]
    action_bins: int

    def __post_init__(self) -> None:
        if len(self.state_bins) != len(self.obs_low) or len(self.state_bins) != len(self.obs_high):
            raise ValueError("State bin counts and observation bounds must have the same dimensionality.")
        if any(bin_count <= 0 for bin_count in self.state_bins):
            raise ValueError("All state bin counts must be positive.")
        if self.action_bins <= 0:
            raise ValueError("Action bins must be positive.")
        for lower, upper in zip(self.obs_low, self.obs_high):
            if upper <= lower:
                raise ValueError("Observation upper bounds must be greater than lower bounds.")

    @property
    def state_dim(self) -> int:
        return len(self.state_bins)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.state_bins

    @property
    def state_action_shape(self) -> tuple[int, ...]:
        return self.state_bins + (self.action_bins,)

    def discretize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected observations with shape (batch, {self.state_dim}), got {tuple(observations.shape)}."
            )

        low = observations.new_tensor(self.obs_low)
        high = observations.new_tensor(self.obs_high)
        clipped = torch.clamp(observations, min=low, max=high)
        normalized = (clipped - low) / (high - low)

        bin_counts = observations.new_tensor(self.state_bins, dtype=torch.float32)
        scaled = torch.floor(normalized * bin_counts).long()
        max_indices = torch.as_tensor(self.state_bins, dtype=torch.long, device=observations.device) - 1
        return torch.minimum(torch.maximum(scaled, torch.zeros_like(scaled)), max_indices)

    def flatten_state_indices(self, state_indices: torch.Tensor) -> torch.Tensor:
        if state_indices.ndim != 2 or state_indices.shape[-1] != self.state_dim:
            raise ValueError("State indices must have shape (batch, state_dim).")

        flat = state_indices[:, 0].long()
        for dimension, bin_count in enumerate(self.state_bins[1:], start=1):
            flat = flat * bin_count + state_indices[:, dimension].long()
        return flat

    def flatten_state_action_indices(
        self,
        state_indices: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        if action_indices.ndim != 1 or action_indices.shape[0] != state_indices.shape[0]:
            raise ValueError("Action indices must have shape (batch,).")
        flat_states = self.flatten_state_indices(state_indices)
        return flat_states * self.action_bins + action_indices.long()


def default_cartpole_spec() -> DiscretizationSpec:
    return DiscretizationSpec(
        state_bins=DEFAULT_CARTPOLE_STATE_BINS,
        obs_low=DEFAULT_CARTPOLE_OBS_LOW,
        obs_high=DEFAULT_CARTPOLE_OBS_HIGH,
        action_bins=DEFAULT_ACTION_BINS,
    )


def _summarize_count_tensor(
    counts: torch.Tensor,
    target_mode: str,
    gamma: float,
    source: str,
) -> dict[str, Any]:
    observed_mask = counts > 0
    total_bins = int(counts.numel())
    visited_bins = int(observed_mask.sum().item())
    visited_counts = counts[observed_mask]
    total_samples = int(counts.sum().item())

    if visited_bins > 0:
        mean_samples = float(visited_counts.mean().item())
        max_samples = int(visited_counts.max().item())
        min_samples = int(visited_counts.min().item())
    else:
        mean_samples = 0.0
        max_samples = 0
        min_samples = 0

    return {
        "target_mode": target_mode,
        "gamma": float(gamma),
        "source": source,
        "tensor_shape": list(counts.shape),
        "total_bins": total_bins,
        "visited_bins": visited_bins,
        "visited_fraction": float(visited_bins / max(1, total_bins)),
        "total_samples": total_samples,
        "mean_samples_per_visited_bin": mean_samples,
        "max_samples_per_bin": max_samples,
        "min_samples_per_bin": min_samples,
    }


class EmpiricalQTensorBuilder:
    def __init__(
        self,
        spec: DiscretizationSpec,
        gamma: float,
        target_mode: str = "td_bootstrap",
        v_target_mode: str = "values",
    ) -> None:
        if target_mode not in {"td_bootstrap", "returns"}:
            raise ValueError("target_mode must be one of {'td_bootstrap', 'returns'}.")
        if v_target_mode not in {"values", "returns"}:
            raise ValueError("v_target_mode must be one of {'values', 'returns'}.")

        self.spec = spec
        self.gamma = gamma
        self.target_mode = target_mode
        self.v_target_mode = v_target_mode
        self.state_action_value_sums = torch.zeros(spec.state_action_shape, dtype=torch.float64)
        self.state_action_counts = torch.zeros(spec.state_action_shape, dtype=torch.float64)
        self.state_value_sums = torch.zeros(spec.state_shape, dtype=torch.float64)
        self.state_value_counts = torch.zeros(spec.state_shape, dtype=torch.float64)
        self.state_action_advantage_sums = torch.zeros(spec.state_action_shape, dtype=torch.float64)
        self.state_action_advantage_counts = torch.zeros(spec.state_action_shape, dtype=torch.float64)

    def add_rollout(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> None:
        rollout_length = buffer.position
        if rollout_length <= 0:
            return
        if not buffer.is_discrete:
            raise ValueError("CartPole diagnostics only support discrete action buffers.")

        observations = buffer.observations[:rollout_length].to(dtype=torch.float32)
        state_indices = self.spec.discretize_observations(observations)
        state_flat = self.spec.flatten_state_indices(state_indices)

        action_indices = buffer.actions[:rollout_length, 0].long()
        action_indices = torch.clamp(action_indices, min=0, max=self.spec.action_bins - 1)
        state_action_flat = self.spec.flatten_state_action_indices(state_indices, action_indices)

        q_targets = self._build_q_targets(buffer, rollout_length, last_value).to(dtype=torch.float64)
        v_targets = self._build_v_targets(buffer, rollout_length).to(dtype=torch.float64)
        a_targets = self._build_advantage_targets(buffer, rollout_length, q_targets, v_targets).to(dtype=torch.float64)
        ones = torch.ones_like(q_targets)

        self.state_action_value_sums.view(-1).index_add_(0, state_action_flat, q_targets)
        self.state_action_counts.view(-1).index_add_(0, state_action_flat, ones)
        self.state_value_sums.view(-1).index_add_(0, state_flat, v_targets)
        self.state_value_counts.view(-1).index_add_(0, state_flat, ones)
        self.state_action_advantage_sums.view(-1).index_add_(0, state_action_flat, a_targets)
        self.state_action_advantage_counts.view(-1).index_add_(0, state_action_flat, ones)

    def empirical_q_tensor(self) -> torch.Tensor:
        return _average_observed_values(self.state_action_value_sums, self.state_action_counts)

    def empirical_v_tensor(self) -> torch.Tensor:
        return _average_observed_values(self.state_value_sums, self.state_value_counts)

    def empirical_a_tensor(self) -> torch.Tensor:
        return _average_observed_values(self.state_action_advantage_sums, self.state_action_advantage_counts)

    def q_observed_mask(self) -> torch.Tensor:
        return self.state_action_counts > 0

    def q_counts(self) -> torch.Tensor:
        return self.state_action_counts

    def v_observed_mask(self) -> torch.Tensor:
        return self.state_value_counts > 0

    def v_counts(self) -> torch.Tensor:
        return self.state_value_counts

    def a_observed_mask(self) -> torch.Tensor:
        return self.state_action_advantage_counts > 0

    def a_counts(self) -> torch.Tensor:
        return self.state_action_advantage_counts

    def q_summary(self) -> dict[str, Any]:
        return _summarize_count_tensor(
            self.state_action_counts,
            self.target_mode,
            self.gamma,
            source="action_value_target",
        )

    def v_summary(self) -> dict[str, Any]:
        return _summarize_count_tensor(
            self.state_value_counts,
            self.v_target_mode,
            self.gamma,
            source="critic_value_prediction" if self.v_target_mode == "values" else "return_target",
        )

    def a_summary(self) -> dict[str, Any]:
        summary = _summarize_count_tensor(
            self.state_action_advantage_counts,
            "advantages",
            self.gamma,
            source="gae_advantage",
        )
        return summary

    def _build_q_targets(
        self,
        buffer: RolloutBuffer,
        rollout_length: int,
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_mode == "returns":
            return buffer.returns[:rollout_length].clone()

        rewards = buffer.rewards[:rollout_length]
        dones = buffer.dones[:rollout_length]
        next_values = torch.empty(rollout_length, dtype=torch.float32)
        if rollout_length > 1:
            next_values[:-1] = buffer.values[1:rollout_length]
        next_values[-1] = last_value.squeeze().float()
        return rewards + self.gamma * next_values * (1.0 - dones)

    def _build_v_targets(
        self,
        buffer: RolloutBuffer,
        rollout_length: int,
    ) -> torch.Tensor:
        if self.v_target_mode == "returns":
            v_targets = buffer.returns[:rollout_length]
        else:
            v_targets = buffer.values[:rollout_length]
        if not torch.isfinite(v_targets).all():
            raise ValueError("V targets must be finite before empirical tensor aggregation.")
        return v_targets.clone()

    def _build_advantage_targets(
        self,
        buffer: RolloutBuffer,
        rollout_length: int,
        q_targets: torch.Tensor,
        v_targets: torch.Tensor,
    ) -> torch.Tensor:
        advantages = buffer.advantages[:rollout_length]
        if not torch.isfinite(advantages).all():
            raise ValueError("Advantage targets must be finite before empirical tensor aggregation.")
        return advantages.clone()


def _average_observed_values(value_sums: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    empirical = torch.zeros_like(value_sums)
    observed = counts > 0
    empirical[observed] = value_sums[observed] / counts[observed]
    return empirical


@dataclass(frozen=True)
class TTDecomposition:
    cores: tuple[torch.Tensor, ...]
    ranks: tuple[int, ...]

    @property
    def parameter_count(self) -> int:
        return sum(core.numel() for core in self.cores)


def tt_svd(tensor: torch.Tensor, max_rank: int) -> TTDecomposition:
    if tensor.ndim < 2:
        raise ValueError("TT-SVD expects a tensor with at least two dimensions.")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive.")

    working = tensor.to(dtype=torch.float64).contiguous()
    modes = tuple(int(dim) for dim in working.shape)
    cores: list[torch.Tensor] = []
    ranks = [1]
    left_rank = 1

    for mode in modes[:-1]:
        working = working.reshape(left_rank * mode, -1)
        u, singular_values, vh = torch.linalg.svd(working, full_matrices=False)
        right_rank = max(1, min(max_rank, singular_values.numel()))
        u = u[:, :right_rank]
        singular_values = singular_values[:right_rank]
        vh = vh[:right_rank, :]
        cores.append(u.reshape(left_rank, mode, right_rank))
        working = singular_values.unsqueeze(1) * vh
        left_rank = right_rank
        ranks.append(right_rank)

    cores.append(working.reshape(left_rank, modes[-1], 1))
    ranks.append(1)
    return TTDecomposition(cores=tuple(cores), ranks=tuple(ranks))


def reconstruct_tt(decomposition: TTDecomposition) -> torch.Tensor:
    tensor = decomposition.cores[0]
    for core in decomposition.cores[1:]:
        tensor = torch.tensordot(tensor, core, dims=([-1], [0]))
    return tensor.squeeze(0).squeeze(-1)


def compute_error_metrics(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
    observed_counts: torch.Tensor | None = None,
) -> dict[str, float]:
    difference = approximation - reference
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    relative_frobenius_error = 0.0
    if reference_norm > 0.0:
        relative_frobenius_error = float(torch.linalg.vector_norm(difference).item() / reference_norm)

    metrics = {
        "relative_frobenius_error": relative_frobenius_error,
        "max_error": float(difference.abs().max().item()),
    }

    if observed_counts is not None:
        observed_mask = observed_counts > 0

    if observed_mask is not None and bool(observed_mask.any().item()):
        observed_reference = reference[observed_mask]
        observed_difference = difference[observed_mask]
        observed_norm = float(torch.linalg.vector_norm(observed_reference).item())
        observed_relative_error = 0.0
        if observed_norm > 0.0:
            observed_relative_error = float(torch.linalg.vector_norm(observed_difference).item() / observed_norm)
        metrics["observed_relative_frobenius_error"] = observed_relative_error
        metrics["observed_max_error"] = float(observed_difference.abs().max().item())

        if observed_counts is not None:
            weights = observed_counts[observed_mask].to(dtype=torch.float64)
            weight_sum = float(weights.sum().item())
            if weight_sum > 0.0:
                weighted_squared_error = weights * observed_difference.square()
                weighted_squared_reference = weights * observed_reference.square()
                observed_weighted_rmse = float(torch.sqrt(weighted_squared_error.sum() / weights.sum()).item())
                observed_weighted_relative_error = 0.0
                weighted_reference_sum = float(weighted_squared_reference.sum().item())
                if weighted_reference_sum > 0.0:
                    observed_weighted_relative_error = float(
                        torch.sqrt(weighted_squared_error.sum() / weighted_squared_reference.sum()).item()
                    )
                metrics["observed_weighted_rmse"] = observed_weighted_rmse
                metrics["observed_weighted_relative_error"] = observed_weighted_relative_error

    return metrics


def detect_rank_knee(
    rank_sweep: Sequence[dict[str, float]],
    metric_key: str = "observed_relative_frobenius_error",
) -> dict[str, Any]:
    if len(rank_sweep) < 3:
        return {"has_clear_knee": False, "suggested_rank": None, "score": 0.0, "metric_key": metric_key}

    values = [float(entry.get(metric_key, entry["relative_frobenius_error"])) for entry in rank_sweep]
    x_values = torch.tensor([math.log2(entry["tt_rank"]) for entry in rank_sweep], dtype=torch.float64)
    y_values = torch.tensor([math.log(value + 1e-12) for value in values], dtype=torch.float64)

    x_span = x_values[-1] - x_values[0]
    y_min = torch.min(y_values)
    y_span = torch.max(y_values) - y_min
    if float(x_span.item()) == 0.0 or float(y_span.item()) == 0.0:
        return {"has_clear_knee": False, "suggested_rank": None, "score": 0.0, "metric_key": metric_key}

    normalized_x = (x_values - x_values[0]) / x_span
    normalized_y = (y_values - y_min) / y_span
    chord = 1.0 - normalized_x
    distances = chord - normalized_y

    best_index = int(torch.argmax(distances).item())
    best_score = float(distances[best_index].item())
    has_clear_knee = best_index not in {0, len(rank_sweep) - 1} and best_score >= 0.08
    suggested_rank = int(rank_sweep[best_index]["tt_rank"]) if has_clear_knee else None
    return {
        "has_clear_knee": has_clear_knee,
        "suggested_rank": suggested_rank,
        "score": best_score,
        "metric_key": metric_key,
    }


def analyze_tt_rank_sweep(
    tensor: torch.Tensor,
    tt_ranks: Sequence[int],
    observed_mask: torch.Tensor | None = None,
    observed_counts: torch.Tensor | None = None,
) -> dict[str, Any]:
    reference = tensor.to(dtype=torch.float64)
    dense_parameter_count = int(reference.numel())
    rank_results: list[dict[str, Any]] = []

    for tt_rank in tt_ranks:
        decomposition = tt_svd(reference, max_rank=int(tt_rank))
        approximation = reconstruct_tt(decomposition)
        metrics = compute_error_metrics(
            reference,
            approximation,
            observed_mask=observed_mask,
            observed_counts=observed_counts,
        )
        rank_results.append(
            {
                "tt_rank": int(tt_rank),
                "relative_frobenius_error": metrics["relative_frobenius_error"],
                "max_error": metrics["max_error"],
                "observed_relative_frobenius_error": metrics.get(
                    "observed_relative_frobenius_error",
                    metrics["relative_frobenius_error"],
                ),
                "observed_max_error": metrics.get("observed_max_error", metrics["max_error"]),
                "observed_weighted_rmse": metrics.get("observed_weighted_rmse", 0.0),
                "observed_weighted_relative_error": metrics.get("observed_weighted_relative_error", 0.0),
                "parameter_count": int(decomposition.parameter_count),
                "compression_ratio": float(dense_parameter_count / max(1, decomposition.parameter_count)),
                "tt_ranks": list(decomposition.ranks),
            }
        )

    knee_metric_key = "observed_relative_frobenius_error"
    if rank_results and "observed_weighted_relative_error" in rank_results[0]:
        knee_metric_key = "observed_weighted_relative_error"

    return {
        "dense_parameter_count": dense_parameter_count,
        "rank_sweep": rank_results,
        "knee": detect_rank_knee(rank_results, metric_key=knee_metric_key),
    }
