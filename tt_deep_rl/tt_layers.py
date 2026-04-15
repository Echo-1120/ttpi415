from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _prime_factors(value: int) -> list[int]:
    factors: list[int] = []
    n = value
    divisor = 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    if n > 1:
        factors.append(n)
    return factors


def _balanced_modes(value: int, order: int) -> tuple[int, ...]:
    padded = _next_power_of_two(max(1, value))
    modes = [1 for _ in range(order)]
    for factor in sorted(_prime_factors(padded), reverse=True):
        index = min(range(order), key=modes.__getitem__)
        modes[index] *= factor
    return tuple(modes)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@dataclass(frozen=True)
class TTShapeSpec:
    in_modes: tuple[int, ...]
    out_modes: tuple[int, ...]
    ranks: tuple[int, ...]

    @property
    def order(self) -> int:
        return len(self.in_modes)

    @property
    def padded_in_features(self) -> int:
        return math.prod(self.in_modes)

    @property
    def padded_out_features(self) -> int:
        return math.prod(self.out_modes)


def build_tt_shape_spec(
    in_features: int,
    out_features: int,
    tt_order: int,
    tt_rank: int,
) -> TTShapeSpec:
    in_modes = _balanced_modes(in_features, tt_order)
    out_modes = _balanced_modes(out_features, tt_order)
    ranks = [1]
    for _ in range(tt_order - 1):
        ranks.append(tt_rank)
    ranks.append(1)
    return TTShapeSpec(
        in_modes=in_modes,
        out_modes=out_modes,
        ranks=tuple(ranks),
    )


class TTLinear(nn.Module):
    """Tensor-train parameterized linear layer.

    The layer stores the weight in TT format and reconstructs the dense weight on
    each forward pass. This keeps the implementation concise and easy to inspect,
    which is useful for research iteration and ablation studies.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tt_rank: int = 4,
        tt_order: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shape_spec = build_tt_shape_spec(
            in_features=in_features,
            out_features=out_features,
            tt_order=tt_order,
            tt_rank=tt_rank,
        )

        cores = []
        for index in range(self.shape_spec.order):
            left_rank = self.shape_spec.ranks[index]
            right_rank = self.shape_spec.ranks[index + 1]
            out_mode = self.shape_spec.out_modes[index]
            in_mode = self.shape_spec.in_modes[index]
            core = nn.Parameter(torch.empty(left_rank, out_mode, in_mode, right_rank))
            cores.append(core)
        self.cores = nn.ParameterList(cores)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for core in self.cores:
            nn.init.normal_(core, mean=0.0, std=0.02)
        if self.bias is not None:
            fan_in = max(1, self.in_features)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def dense_weight(self) -> torch.Tensor:
        tensor = self.cores[0]
        for core in self.cores[1:]:
            tensor = torch.tensordot(tensor, core, dims=([-1], [0]))

        tensor = tensor.squeeze(0).squeeze(-1)
        permutation = list(range(0, 2 * self.shape_spec.order, 2))
        permutation += list(range(1, 2 * self.shape_spec.order, 2))
        tensor = tensor.permute(permutation).contiguous()
        weight = tensor.reshape(
            self.shape_spec.padded_out_features,
            self.shape_spec.padded_in_features,
        )
        return weight[: self.out_features, : self.in_features]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dense_weight()
        return F.linear(inputs, weight, self.bias)

    def compression_stats(self) -> dict[str, float]:
        tt_params = sum(core.numel() for core in self.cores)
        dense_params = self.in_features * self.out_features
        ratio = dense_params / max(1, tt_params)
        return {
            "dense_params": float(dense_params),
            "tt_params": float(tt_params),
            "compression_ratio": float(ratio),
        }


class TTMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        tt_rank: int = 4,
        tt_order: int = 3,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        dims = (input_dim,) + hidden_dims + (output_dim,)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(
                TTLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    tt_rank=tt_rank,
                    tt_order=tt_order,
                )
            )
            if out_dim != output_dim:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def compression_stats(self) -> dict[str, float]:
        dense_params = 0
        tt_params = 0
        for module in self.modules():
            if isinstance(module, TTLinear):
                stats = module.compression_stats()
                dense_params += int(stats["dense_params"])
                tt_params += int(stats["tt_params"])
        ratio = dense_params / max(1, tt_params)
        return {
            "dense_params": float(dense_params),
            "tt_params": float(tt_params),
            "compression_ratio": float(ratio),
            "module_params": float(_parameter_count(self)),
        }

