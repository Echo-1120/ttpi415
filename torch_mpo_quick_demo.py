from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SRC = SCRIPT_DIR / "torch-mpo" / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from torch_mpo.layers.tt_linear import TTLinear


def param_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def run_demo(
    in_features: int,
    out_features: int,
    tt_rank: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)

    dense = nn.Linear(in_features, out_features, bias=True)
    tt_layer = TTLinear(
        in_features=in_features,
        out_features=out_features,
        tt_ranks=tt_rank,
        bias=True,
    )

    with torch.no_grad():
        tt_layer.from_matrix(dense.weight.data)
        if tt_layer.bias is not None:
            tt_layer.bias.copy_(dense.bias.data)

    inputs = torch.randn(batch_size, in_features)
    dense_outputs = dense(inputs)
    tt_outputs = tt_layer(inputs)

    mse = torch.mean((dense_outputs - tt_outputs) ** 2).item()
    mae = torch.mean(torch.abs(dense_outputs - tt_outputs)).item()
    max_abs = torch.max(torch.abs(dense_outputs - tt_outputs)).item()

    dense_params = param_count(dense)
    tt_params = param_count(tt_layer)

    summary = {
        "in_features": in_features,
        "out_features": out_features,
        "tt_rank": tt_rank,
        "batch_size": batch_size,
        "dense_params": dense_params,
        "tt_params": tt_params,
        "compression_ratio": dense_params / tt_params,
        "reported_tt_ratio": tt_layer.compression_ratio(),
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_abs,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick torch-mpo TTLinear demo.")
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--tt-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    summary = run_demo(
        in_features=args.in_features,
        out_features=args.out_features,
        tt_rank=args.tt_rank,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
