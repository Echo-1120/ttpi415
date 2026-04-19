from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tt_deep_rl.cartpole_diagnostics import (
    DEFAULT_ACTION_BINS,
    DEFAULT_CARTPOLE_OBS_HIGH,
    DEFAULT_CARTPOLE_OBS_LOW,
    DEFAULT_CARTPOLE_STATE_BINS,
    DEFAULT_TT_RANKS,
    DiscretizationSpec,
    EmpiricalQTensorBuilder,
    analyze_tt_rank_sweep,
)
from tt_deep_rl.networks import ModelConfig
from tt_deep_rl.ppo import PPOConfig, PPOTrainer, save_summary
from tt_deep_rl.svg_plots import LineSeries, save_line_chart_svg


def _format_int_tuple(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _format_float_tuple(values: tuple[float, ...]) -> str:
    return ",".join(str(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze low-rank TT structure of empirical CartPole tensors.")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--actor-arch", choices=("mlp", "tt", "hybrid"), default="mlp")
    parser.add_argument("--critic-arch", choices=("mlp", "tt", "hybrid"), default="mlp")
    parser.add_argument("--hidden-dims", default="64,64")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--activation", choices=("tanh", "relu", "gelu"), default="relu")
    parser.add_argument("--tt-rank", type=int, default=4)
    parser.add_argument("--tt-order", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-mode", choices=("td_bootstrap", "returns"), default="td_bootstrap")
    parser.add_argument("--state-bins", default=_format_int_tuple(DEFAULT_CARTPOLE_STATE_BINS))
    parser.add_argument("--action-bins", type=int, default=DEFAULT_ACTION_BINS)
    parser.add_argument("--obs-low", default=_format_float_tuple(DEFAULT_CARTPOLE_OBS_LOW))
    parser.add_argument("--obs-high", default=_format_float_tuple(DEFAULT_CARTPOLE_OBS_HIGH))
    parser.add_argument("--tt-ranks", default=_format_int_tuple(DEFAULT_TT_RANKS))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--figure-dir", default="./cartpole_q_figures")
    return parser.parse_args()


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(token.strip()) for token in value.split(",") if token.strip())


def build_configs(args: argparse.Namespace) -> tuple[PPOConfig, ModelConfig]:
    ppo_config = PPOConfig(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )
    model_config = ModelConfig(
        actor_arch=args.actor_arch,
        critic_arch=args.critic_arch,
        hidden_dims=parse_int_tuple(args.hidden_dims),
        latent_dim=args.latent_dim,
        activation=args.activation,
        tt_rank=args.tt_rank,
        tt_order=args.tt_order,
    )
    return ppo_config, model_config


def build_discretization_spec(args: argparse.Namespace) -> DiscretizationSpec:
    state_bins = parse_int_tuple(args.state_bins)
    obs_low = parse_float_tuple(args.obs_low)
    obs_high = parse_float_tuple(args.obs_high)
    if len(state_bins) != 4:
        raise ValueError("CartPole diagnostics expect four state bin counts.")
    if len(obs_low) != 4 or len(obs_high) != 4:
        raise ValueError("CartPole diagnostics expect four observation bounds.")
    return DiscretizationSpec(
        state_bins=state_bins,
        obs_low=obs_low,
        obs_high=obs_high,
        action_bins=args.action_bins,
    )


def compute_total_updates(total_timesteps: int, rollout_steps: int) -> int:
    return len(range(0, total_timesteps, rollout_steps))


def stage_name_for_rollout(rollout_index: int, total_updates: int) -> str:
    if rollout_index < total_updates / 3:
        return "early"
    if rollout_index < 2 * total_updates / 3:
        return "middle"
    return "late"


def build_analysis_for_builder(
    builder: EmpiricalQTensorBuilder,
    tt_ranks: tuple[int, ...],
) -> dict[str, Any]:
    q_tensor = builder.empirical_q_tensor()
    v_tensor = builder.empirical_v_tensor()
    a_tensor = builder.empirical_a_tensor()

    q_analysis = analyze_tt_rank_sweep(q_tensor, tt_ranks=tt_ranks, observed_mask=builder.q_observed_mask())
    v_analysis = analyze_tt_rank_sweep(v_tensor, tt_ranks=tt_ranks, observed_mask=builder.v_observed_mask())
    a_analysis = analyze_tt_rank_sweep(a_tensor, tt_ranks=tt_ranks, observed_mask=builder.a_observed_mask())

    return {
        "q_tensor": builder.q_summary(),
        "v_tensor": builder.v_summary(),
        "a_tensor": builder.a_summary(),
        "q_tt_analysis": q_analysis,
        "v_tt_analysis": v_analysis,
        "a_tt_analysis": a_analysis,
    }


def create_figures(
    figure_dir: str,
    tt_ranks: tuple[int, ...],
    overall_analysis: dict[str, Any],
    stage_analysis: dict[str, Any],
) -> dict[str, str]:
    output_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_labels = [str(rank) for rank in tt_ranks]

    q_rank_sweep = overall_analysis["q_tt_analysis"]["rank_sweep"]
    save_line_chart_svg(
        output_path=str(output_dir / "01_error_vs_rank.svg"),
        title="CartPole Q Error vs TT Rank",
        subtitle="Relative Frobenius reconstruction error for the empirical Q tensor",
        x_labels=x_labels,
        series=[
            LineSeries(
                name="All bins",
                values=tuple(entry["relative_frobenius_error"] for entry in q_rank_sweep),
                color="#2563eb",
            ),
            LineSeries(
                name="Visited bins only",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in q_rank_sweep),
                color="#dc2626",
                dashed=True,
            ),
        ],
        x_label="TT rank",
        left_y_label="Relative Frobenius error",
    )

    save_line_chart_svg(
        output_path=str(output_dir / "02_compression_vs_performance.svg"),
        title="Compression Ratio and Error vs TT Rank",
        subtitle="Compression ratio stays high at low rank while reconstruction error falls",
        x_labels=x_labels,
        series=[
            LineSeries(
                name="Compression ratio",
                values=tuple(entry["compression_ratio"] for entry in q_rank_sweep),
                color="#0f766e",
                axis="left",
            ),
            LineSeries(
                name="Visited-bin error",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in q_rank_sweep),
                color="#b45309",
                axis="right",
                dashed=True,
            ),
        ],
        x_label="TT rank",
        left_y_label="Compression ratio",
        right_y_label="Observed relative error",
        left_scale="log10",
    )

    save_line_chart_svg(
        output_path=str(output_dir / "03_stage_rank_requirement.svg"),
        title="Rank Needs Across Training Stages",
        subtitle="Observed-bin Q reconstruction error in early, middle, and late PPO training",
        x_labels=x_labels,
        series=[
            LineSeries(
                name="Early",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in stage_analysis["early"]["q_tt_analysis"]["rank_sweep"]),
                color="#2563eb",
            ),
            LineSeries(
                name="Middle",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in stage_analysis["middle"]["q_tt_analysis"]["rank_sweep"]),
                color="#16a34a",
            ),
            LineSeries(
                name="Late",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in stage_analysis["late"]["q_tt_analysis"]["rank_sweep"]),
                color="#dc2626",
            ),
        ],
        x_label="TT rank",
        left_y_label="Observed relative error",
    )

    save_line_chart_svg(
        output_path=str(output_dir / "04_v_q_a_comparison.svg"),
        title="V, Q, and A Low-Rank Comparison",
        subtitle="Observed-bin reconstruction error for value, Q, and advantage tensors",
        x_labels=x_labels,
        series=[
            LineSeries(
                name="V(s)",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in overall_analysis["v_tt_analysis"]["rank_sweep"]),
                color="#7c3aed",
            ),
            LineSeries(
                name="Q(s,a)",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in q_rank_sweep),
                color="#2563eb",
            ),
            LineSeries(
                name="A(s,a)",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in overall_analysis["a_tt_analysis"]["rank_sweep"]),
                color="#ea580c",
            ),
        ],
        x_label="TT rank",
        left_y_label="Observed relative error",
    )

    return {
        "error_vs_rank": str(output_dir / "01_error_vs_rank.svg"),
        "compression_vs_performance": str(output_dir / "02_compression_vs_performance.svg"),
        "stage_rank_requirement": str(output_dir / "03_stage_rank_requirement.svg"),
        "v_q_a_comparison": str(output_dir / "04_v_q_a_comparison.svg"),
    }


def main() -> None:
    args = parse_args()
    ppo_config, model_config = build_configs(args)
    discretization_spec = build_discretization_spec(args)
    tt_ranks = tuple(sorted(set(parse_int_tuple(args.tt_ranks))))
    if not tt_ranks or any(rank <= 0 for rank in tt_ranks):
        raise ValueError("TT rank sweep must contain positive integers.")

    trainer = PPOTrainer(ppo_config, model_config)
    if trainer.obs_dim != 4 or not trainer.agent.is_discrete:
        raise ValueError(
            f"CartPole diagnostics expect a 4D discrete environment, got obs_dim={trainer.obs_dim}."
        )
    if getattr(trainer.env.action_space, "n", None) != discretization_spec.action_bins:
        raise ValueError(
            f"Action bins ({discretization_spec.action_bins}) must match the environment action count."
        )

    overall_builder = EmpiricalQTensorBuilder(
        spec=discretization_spec,
        gamma=ppo_config.gamma,
        target_mode=args.target_mode,
    )
    stage_builders = {
        "early": EmpiricalQTensorBuilder(discretization_spec, ppo_config.gamma, args.target_mode),
        "middle": EmpiricalQTensorBuilder(discretization_spec, ppo_config.gamma, args.target_mode),
        "late": EmpiricalQTensorBuilder(discretization_spec, ppo_config.gamma, args.target_mode),
    }
    total_updates = compute_total_updates(ppo_config.total_timesteps, ppo_config.rollout_steps)
    rollout_index = 0

    def rollout_callback(buffer, last_value) -> None:
        nonlocal rollout_index
        overall_builder.add_rollout(buffer, last_value)
        stage_key = stage_name_for_rollout(rollout_index, total_updates)
        stage_builders[stage_key].add_rollout(buffer, last_value)
        rollout_index += 1

    training_summary = trainer.train(rollout_callback=rollout_callback)
    overall_analysis = build_analysis_for_builder(overall_builder, tt_ranks)
    stage_analysis = {
        stage_name: build_analysis_for_builder(builder, tt_ranks)
        for stage_name, builder in stage_builders.items()
    }
    figure_paths = create_figures(args.figure_dir, tt_ranks, overall_analysis, stage_analysis)

    result: dict[str, Any] = {
        "training_summary": training_summary,
        "diagnostics": {
            "discretization": {
                "state_bins": list(discretization_spec.state_bins),
                "obs_low": list(discretization_spec.obs_low),
                "obs_high": list(discretization_spec.obs_high),
                "action_bins": int(discretization_spec.action_bins),
            },
            "empirical_tensor": overall_analysis["q_tensor"],
            "tt_analysis": overall_analysis["q_tt_analysis"],
            "overall_analysis": overall_analysis,
            "stage_analysis": stage_analysis,
            "figures": figure_paths,
        },
    }

    if args.output_json:
        save_summary(result, args.output_json)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
