from __future__ import annotations

import argparse
import json

from tt_deep_rl.dqn import DQNConfig, DQNTrainer
from tt_deep_rl.networks import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DQN with MLP / TT / hybrid Q-network"
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--critic-arch",
        choices=("mlp", "tt", "hybrid"),
        default="mlp",
        dest="q_arch",
    )
    parser.add_argument("--hidden-dims", default="64,64")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--activation", choices=("tanh", "relu", "gelu"), default="relu")
    parser.add_argument("--tt-rank", type=int, default=4)
    parser.add_argument("--tt-order", type=int, default=3)

    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-freq", type=int, default=500)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="cartpole_dqn_result.json")
    return parser.parse_args()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    return tuple(int(token) for token in value.split(",") if token)


def main() -> None:
    args = parse_args()
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    dqn_config = DQNConfig(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        device=args.device,
    )

    model_config = ModelConfig(
        critic_arch=args.q_arch,
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        activation=args.activation,
        tt_rank=args.tt_rank,
        tt_order=args.tt_order,
    )

    trainer = DQNTrainer(dqn_config, model_config)
    result = trainer.train()

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nDQN training finished. Results saved to {args.output_json}")
    print(f"Q-network compression stats: {result['compression']}")


if __name__ == "__main__":
    main()
