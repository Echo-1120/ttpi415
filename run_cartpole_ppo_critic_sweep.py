from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

DEFAULT_SEEDS = (7, 11, 17, 23, 29)
DEFAULT_TIMESTEPS = (50_000, 100_000)
DEFAULT_TT_RANKS = (4, 8, 16)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise ValueError("Expected at least one hidden dimension.")
    return values


def format_tag(label: str, value: int | str | None) -> str:
    if value is None:
        return f"{label}-na"
    return f"{label}-{value}"


def run_name(env_id: str, actor_arch: str, critic_arch: str, tt_rank: int | None, total_timesteps: int, seed: int) -> str:
    env_tag = env_id.lower().replace("-", "_")
    return "_".join(
        (
            env_tag,
            "ppo",
            format_tag("actor", actor_arch),
            format_tag("critic", critic_arch),
            format_tag("rank", tt_rank),
            format_tag("steps", total_timesteps),
            format_tag("seed", seed),
        )
    )


def build_run_specs(
    env_id: str,
    seeds: tuple[int, ...],
    timesteps_list: tuple[int, ...],
    tt_ranks: tuple[int, ...],
) -> list[dict[str, int | str | None]]:
    critic_configs = [("mlp", None)]
    critic_configs.extend(("tt", rank) for rank in tt_ranks)
    critic_configs.extend(("hybrid", rank) for rank in tt_ranks)

    specs: list[dict[str, int | str | None]] = []
    for total_timesteps in timesteps_list:
        for critic_arch, tt_rank in critic_configs:
            for seed in seeds:
                specs.append(
                    {
                        "env_id": env_id,
                        "actor_arch": "mlp",
                        "critic_arch": critic_arch,
                        "tt_rank": tt_rank,
                        "total_timesteps": total_timesteps,
                        "seed": seed,
                    }
                )
    return specs


def build_trainer_configs(args: argparse.Namespace, spec: dict[str, int | str | None]) -> tuple[PPOConfig, ModelConfig]:
    from tt_deep_rl.networks import ModelConfig
    from tt_deep_rl.ppo import PPOConfig

    ppo_config = PPOConfig(
        env_id=str(spec["env_id"]),
        seed=int(spec["seed"]),
        total_timesteps=int(spec["total_timesteps"]),
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
        actor_arch="mlp",
        critic_arch=str(spec["critic_arch"]),
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        latent_dim=args.latent_dim,
        activation=args.activation,
        tt_rank=int(spec["tt_rank"] if spec["tt_rank"] is not None else args.tt_ranks[0]),
        tt_order=args.tt_order,
    )
    return ppo_config, model_config


def aggregate_group(runs: list[dict[str, object]]) -> dict[str, object]:
    metric_keys = (
        "mean_episode_return",
        "last_episode_return",
        "episodes_finished",
        "critic_module_params",
        "actor_module_params",
    )
    aggregate: dict[str, object] = {
        "env_id": runs[0]["env_id"],
        "actor_arch": runs[0]["actor_arch"],
        "critic_arch": runs[0]["critic_arch"],
        "tt_rank": runs[0]["tt_rank"],
        "total_timesteps": runs[0]["total_timesteps"],
        "num_seeds": len(runs),
        "seeds": [run["seed"] for run in runs],
    }
    for key in metric_keys:
        values = [float(run[key]) for run in runs]
        aggregate[f"{key}_mean"] = mean(values)
        aggregate[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return aggregate


def write_aggregate_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "env_id",
        "actor_arch",
        "critic_arch",
        "tt_rank",
        "total_timesteps",
        "num_seeds",
        "seeds",
        "mean_episode_return_mean",
        "mean_episode_return_std",
        "last_episode_return_mean",
        "last_episode_return_std",
        "episodes_finished_mean",
        "episodes_finished_std",
        "critic_module_params_mean",
        "critic_module_params_std",
        "actor_module_params_mean",
        "actor_module_params_std",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["seeds"] = ",".join(str(seed) for seed in row["seeds"])
            writer.writerow(csv_row)


def save_json(payload: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CartPole PPO critic-only sweeps for MLP / TT / hybrid critics."
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seeds", default="7,11,17,23,29")
    parser.add_argument("--timesteps", default="50000,100000")
    parser.add_argument("--tt-ranks", default="4,8,16")
    parser.add_argument("--hidden-dims", default="64,64")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--activation", choices=("tanh", "relu", "gelu"), default="tanh")
    parser.add_argument("--tt-order", type=int, default=3)
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
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--output-dir",
        default="./cartpole_ppo_critic_sweep",
        help="Directory used for per-run train summaries, checkpoints, and aggregate summaries.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.seeds = parse_int_tuple(args.seeds)
    args.timesteps = parse_int_tuple(args.timesteps)
    args.tt_ranks = parse_int_tuple(args.tt_ranks)
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    runs_dir = output_dir / "runs"
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    specs = build_run_specs(
        env_id=args.env_id,
        seeds=args.seeds,
        timesteps_list=args.timesteps,
        tt_ranks=args.tt_ranks,
    )

    manifest = {
        "experiment": "cartpole_ppo_critic_only",
        "env_id": args.env_id,
        "actor_arch": "mlp",
        "critic_arches": ["mlp", "tt", "hybrid"],
        "tt_ranks": list(args.tt_ranks),
        "seeds": list(args.seeds),
        "timesteps": list(args.timesteps),
        "model_defaults": {
            "hidden_dims": list(parse_hidden_dims(args.hidden_dims)),
            "latent_dim": args.latent_dim,
            "activation": args.activation,
            "tt_order": args.tt_order,
        },
        "ppo_defaults": {
            "rollout_steps": args.rollout_steps,
            "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_coef": args.clip_coef,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "device": args.device,
        },
        "num_runs": len(specs),
    }
    save_json(manifest, output_dir / "manifest.json")

    if args.dry_run:
        preview = []
        for spec in specs:
            preview.append(
                {
                    "name": run_name(
                        env_id=str(spec["env_id"]),
                        actor_arch="mlp",
                        critic_arch=str(spec["critic_arch"]),
                        tt_rank=spec["tt_rank"],
                        total_timesteps=int(spec["total_timesteps"]),
                        seed=int(spec["seed"]),
                    ),
                    **spec,
                }
            )
        print(json.dumps({"manifest": manifest, "planned_runs": preview}, indent=2))
        return

    from dataclasses import asdict

    import torch

    from tt_deep_rl.ppo import PPOTrainer

    completed_runs: list[dict[str, object]] = []
    grouped_runs: dict[tuple[str, int | None, int], list[dict[str, object]]] = {}

    for spec in specs:
        name = run_name(
            env_id=str(spec["env_id"]),
            actor_arch="mlp",
            critic_arch=str(spec["critic_arch"]),
            tt_rank=spec["tt_rank"],
            total_timesteps=int(spec["total_timesteps"]),
            seed=int(spec["seed"]),
        )
        run_output_path = runs_dir / f"{name}_train.json"
        checkpoint_path = checkpoints_dir / f"{name}.pt"

        if args.skip_existing and run_output_path.exists() and checkpoint_path.exists():
            summary = json.loads(run_output_path.read_text(encoding="utf-8"))
        else:
            ppo_config, model_config = build_trainer_configs(args, spec)
            trainer = PPOTrainer(ppo_config, model_config)
            summary = trainer.train()
            summary["checkpoint_out"] = str(checkpoint_path)
            torch.save(
                {
                    "model_family": "ppo",
                    "env_id": ppo_config.env_id,
                    "ppo_config": asdict(ppo_config),
                    "model_config": asdict(model_config),
                    "actor_critic_state_dict": trainer.agent.state_dict(),
                },
                checkpoint_path,
            )
            save_json(summary, run_output_path)

        run_record = {
            "name": name,
            "env_id": str(spec["env_id"]),
            "actor_arch": "mlp",
            "critic_arch": str(spec["critic_arch"]),
            "tt_rank": spec["tt_rank"],
            "total_timesteps": int(spec["total_timesteps"]),
            "seed": int(spec["seed"]),
            "mean_episode_return": float(summary["mean_episode_return"]),
            "last_episode_return": float(summary["last_episode_return"]),
            "episodes_finished": int(summary["episodes_finished"]),
            "critic_module_params": float(summary["critic_stats"]["module_params"]),
            "actor_module_params": float(summary["actor_stats"]["module_params"]),
            "train_summary_path": str(run_output_path),
            "checkpoint_path": str(checkpoint_path),
        }
        completed_runs.append(run_record)
        group_key = (
            str(spec["critic_arch"]),
            None if spec["tt_rank"] is None else int(spec["tt_rank"]),
            int(spec["total_timesteps"]),
        )
        grouped_runs.setdefault(group_key, []).append(run_record)

    aggregate_rows = [
        aggregate_group(group_runs)
        for _, group_runs in sorted(
            grouped_runs.items(),
            key=lambda item: (item[0][2], item[0][0], -1 if item[0][1] is None else item[0][1]),
        )
    ]

    suite_summary = {
        "manifest": manifest,
        "runs": completed_runs,
        "aggregates": aggregate_rows,
    }
    save_json(suite_summary, output_dir / "summary.json")
    write_aggregate_csv(aggregate_rows, output_dir / "summary.csv")
    print(json.dumps(suite_summary, indent=2))


if __name__ == "__main__":
    main()
