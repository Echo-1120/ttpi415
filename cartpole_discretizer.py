from __future__ import annotations

from tt_deep_rl.cartpole_diagnostics import (
    DEFAULT_ACTION_BINS,
    DEFAULT_CARTPOLE_OBS_HIGH,
    DEFAULT_CARTPOLE_OBS_LOW,
    DEFAULT_CARTPOLE_STATE_BINS,
    DiscretizationSpec,
)


def format_int_tuple(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def format_float_tuple(values: tuple[float, ...]) -> str:
    return ",".join(str(value) for value in values)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(token.strip()) for token in value.split(",") if token.strip())


def build_cartpole_discretizer(
    state_bins: tuple[int, ...] = DEFAULT_CARTPOLE_STATE_BINS,
    obs_low: tuple[float, ...] = DEFAULT_CARTPOLE_OBS_LOW,
    obs_high: tuple[float, ...] = DEFAULT_CARTPOLE_OBS_HIGH,
    action_bins: int = DEFAULT_ACTION_BINS,
) -> DiscretizationSpec:
    return DiscretizationSpec(
        state_bins=state_bins,
        obs_low=obs_low,
        obs_high=obs_high,
        action_bins=action_bins,
    )
