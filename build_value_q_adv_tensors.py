from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cartpole_discretizer import (
    build_cartpole_discretizer,
    format_float_tuple,
    format_int_tuple,
    parse_float_tuple,
    parse_int_tuple,
)
from tt_deep_rl.cartpole_diagnostics import (
    DEFAULT_ACTION_BINS,
    DEFAULT_CARTPOLE_OBS_HIGH,
    DEFAULT_CARTPOLE_OBS_LOW,
    DEFAULT_CARTPOLE_STATE_BINS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build empirical V/Q/A tensors from exported CartPole rollouts."
    )
    parser.add_argument("--rollout-file", default="cartpole_rollouts.npz")
    parser.add_argument("--target-mode", choices=("returns", "td_bootstrap"), default="returns")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--state-bins", default=format_int_tuple(DEFAULT_CARTPOLE_STATE_BINS))
    parser.add_argument("--action-bins", type=int, default=DEFAULT_ACTION_BINS)
    parser.add_argument("--obs-low", default=format_float_tuple(DEFAULT_CARTPOLE_OBS_LOW))
    parser.add_argument("--obs-high", default=format_float_tuple(DEFAULT_CARTPOLE_OBS_HIGH))
    parser.add_argument("--output-file", default="cartpole_vqa_tensors.npz")
    return parser.parse_args()


def compute_discounted_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for index in reversed(range(len(rewards))):
        running_return = float(rewards[index]) + gamma * running_return * (1.0 - float(dones[index]))
        returns[index] = running_return
    return returns


def summarize_count_tensor(
    counts: torch.Tensor,
    target_mode: str,
    gamma: float,
    source: str,
) -> dict[str, Any]:
    observed_mask = counts > 0
    total_bins = int(counts.numel())
    visited_bins = int(observed_mask.sum().item())
    visited_counts = counts[observed_mask]

    return {
        "target_mode": target_mode,
        "gamma": float(gamma),
        "source": source,
        "tensor_shape": list(counts.shape),
        "total_bins": total_bins,
        "visited_bins": visited_bins,
        "visited_fraction": float(visited_bins / max(1, total_bins)),
        "total_samples": int(counts.sum().item()),
        "mean_samples_per_visited_bin": float(visited_counts.mean().item()) if visited_bins else 0.0,
        "max_samples_per_bin": int(visited_counts.max().item()) if visited_bins else 0,
        "min_samples_per_bin": int(visited_counts.min().item()) if visited_bins else 0,
    }


def average_observed_values(value_sums: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    tensor = torch.zeros_like(value_sums)
    observed_mask = counts > 0
    tensor[observed_mask] = value_sums[observed_mask] / counts[observed_mask]
    return tensor


def main() -> None:
    args = parse_args()
    rollout_data = np.load(args.rollout_file)
    observations = torch.as_tensor(rollout_data["observations"], dtype=torch.float32)
    actions = torch.as_tensor(rollout_data["actions"], dtype=torch.long)
    rewards = np.asarray(rollout_data["rewards"], dtype=np.float32)
    dones = np.asarray(rollout_data["dones"], dtype=np.float32)

    if args.target_mode == "returns":
        q_targets = torch.as_tensor(
            compute_discounted_returns(rewards, dones, gamma=args.gamma),
            dtype=torch.float64,
        )
    else:
        if "next_state_values" not in rollout_data.files:
            raise ValueError("td_bootstrap mode requires next_state_values in the rollout file.")
        next_state_values = np.asarray(rollout_data["next_state_values"], dtype=np.float32)
        if not np.isfinite(next_state_values).all():
            raise ValueError(
                "td_bootstrap mode requires finite next_state_values. "
                "Use a DQN-based rollout export or switch to target-mode=returns."
            )
        q_targets = torch.as_tensor(
            rewards + args.gamma * next_state_values * (1.0 - dones),
            dtype=torch.float64,
        )

    state_values = np.asarray(
        rollout_data["state_values"],
        dtype=np.float32,
    ) if "state_values" in rollout_data.files else np.full_like(rewards, np.nan, dtype=np.float32)
    if np.isfinite(state_values).all():
        v_targets = torch.as_tensor(state_values, dtype=torch.float64)
        advantage_source = "q_minus_state_value"
        a_targets = q_targets - v_targets
    else:
        v_targets = q_targets.clone()
        advantage_source = "q_minus_state_average"
        a_targets = None

    discretizer = build_cartpole_discretizer(
        state_bins=parse_int_tuple(args.state_bins),
        obs_low=parse_float_tuple(args.obs_low),
        obs_high=parse_float_tuple(args.obs_high),
        action_bins=args.action_bins,
    )

    state_indices = discretizer.discretize_observations(observations)
    state_flat = discretizer.flatten_state_indices(state_indices)
    state_action_flat = discretizer.flatten_state_action_indices(state_indices, actions)
    ones = torch.ones_like(q_targets)

    q_value_sums = torch.zeros(discretizer.state_action_shape, dtype=torch.float64)
    q_counts = torch.zeros(discretizer.state_action_shape, dtype=torch.float64)
    v_value_sums = torch.zeros(discretizer.state_shape, dtype=torch.float64)
    v_counts = torch.zeros(discretizer.state_shape, dtype=torch.float64)
    a_value_sums = torch.zeros(discretizer.state_action_shape, dtype=torch.float64)
    a_counts = torch.zeros(discretizer.state_action_shape, dtype=torch.float64)

    q_value_sums.view(-1).index_add_(0, state_action_flat, q_targets)
    q_counts.view(-1).index_add_(0, state_action_flat, ones)
    v_value_sums.view(-1).index_add_(0, state_flat, v_targets)
    v_counts.view(-1).index_add_(0, state_flat, ones)

    q_tensor = average_observed_values(q_value_sums, q_counts)
    v_tensor = average_observed_values(v_value_sums, v_counts)
    if a_targets is None:
        a_tensor = q_tensor - v_tensor.unsqueeze(-1)
        a_counts.copy_(q_counts)
    else:
        a_value_sums.view(-1).index_add_(0, state_action_flat, a_targets)
        a_counts.view(-1).index_add_(0, state_action_flat, ones)
        a_tensor = average_observed_values(a_value_sums, a_counts)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        q_tensor=q_tensor.cpu().numpy(),
        v_tensor=v_tensor.cpu().numpy(),
        a_tensor=a_tensor.cpu().numpy(),
        q_observed_mask=(q_counts > 0).cpu().numpy(),
        v_observed_mask=(v_counts > 0).cpu().numpy(),
        a_observed_mask=(a_counts > 0).cpu().numpy(),
        q_counts=q_counts.cpu().numpy(),
        v_counts=v_counts.cpu().numpy(),
        a_counts=a_counts.cpu().numpy(),
    )

    summary = {
        "rollout_file": args.rollout_file,
        "output_file": str(output_path),
        "target_mode": args.target_mode,
        "gamma": args.gamma,
        "discretization": {
            "state_bins": list(discretizer.state_bins),
            "obs_low": list(discretizer.obs_low),
            "obs_high": list(discretizer.obs_high),
            "action_bins": discretizer.action_bins,
        },
        "q_tensor": summarize_count_tensor(q_counts, args.target_mode, args.gamma, source="action_value_target"),
        "v_tensor": summarize_count_tensor(
            v_counts,
            args.target_mode,
            args.gamma,
            source="state_value_prediction" if np.isfinite(state_values).all() else "state_value_fallback_from_q",
        ),
        "a_tensor": {
            **summarize_count_tensor(a_counts, args.target_mode, args.gamma, source=advantage_source),
        },
    }

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
