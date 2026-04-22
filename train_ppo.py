from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from tt_deep_rl.networks import ModelConfig
from tt_deep_rl.ppo import PPOConfig, PPOTrainer, save_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO with MLP / TT / hybrid backbones.")
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
    parser.add_argument("--output-json", default="")
    parser.add_argument("--checkpoint-out", default="")
    return parser.parse_args()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    return tuple(int(token) for token in value.split(",") if token)


def main() -> None:
    args = parse_args()
    hidden_dims = parse_hidden_dims(args.hidden_dims)

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
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        activation=args.activation,
        tt_rank=args.tt_rank,
        tt_order=args.tt_order,
    )

    trainer = PPOTrainer(ppo_config, model_config)
    summary = trainer.train()

    if args.checkpoint_out:
        checkpoint_path = Path(args.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_family": "ppo",
                "env_id": args.env_id,
                "ppo_config": asdict(ppo_config),
                "model_config": asdict(model_config),
                "actor_critic_state_dict": trainer.agent.state_dict(),
            },
            checkpoint_path,
        )
        summary["checkpoint_out"] = str(checkpoint_path)

    if args.output_json:
        save_summary(summary, args.output_json)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
