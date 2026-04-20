from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tt_deep_rl.cartpole_diagnostics import DEFAULT_TT_RANKS, analyze_tt_rank_sweep
from tt_deep_rl.svg_plots import LineSeries, save_line_chart_svg


def format_int_tuple(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TT rank analysis on empirical CartPole V/Q/A tensors."
    )
    parser.add_argument("--tensor-file", default="cartpole_vqa_tensors.npz")
    parser.add_argument("--tt-ranks", default=format_int_tuple(DEFAULT_TT_RANKS))
    parser.add_argument("--figure-dir", default="./cartpole_rank_figures")
    parser.add_argument("--output-json", default="cartpole_rank_analysis.json")
    return parser.parse_args()


def build_single_tensor_figure(
    output_path: str,
    tensor_name: str,
    rank_sweep: list[dict[str, Any]],
) -> None:
    x_labels = [str(entry["tt_rank"]) for entry in rank_sweep]
    save_line_chart_svg(
        output_path=output_path,
        title=f"{tensor_name.upper()} Tensor Error vs TT Rank",
        subtitle=f"Relative Frobenius reconstruction error for empirical {tensor_name.upper()}",
        x_labels=x_labels,
        series=[
            LineSeries(
                name="All bins",
                values=tuple(entry["relative_frobenius_error"] for entry in rank_sweep),
                color="#2563eb",
            ),
            LineSeries(
                name="Observed bins only",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in rank_sweep),
                color="#dc2626",
                dashed=True,
            ),
        ],
        x_label="TT rank",
        left_y_label="Relative Frobenius error",
    )


def build_comparison_figure(
    output_path: str,
    analyses: dict[str, dict[str, Any]],
) -> None:
    q_rank_sweep = analyses["q"]["rank_sweep"]
    x_labels = [str(entry["tt_rank"]) for entry in q_rank_sweep]
    save_line_chart_svg(
        output_path=output_path,
        title="V / Q / A Low-Rank Comparison",
        subtitle="Observed-bin reconstruction error across empirical tensors",
        x_labels=x_labels,
        series=[
            LineSeries(
                name="V(s)",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in analyses["v"]["rank_sweep"]),
                color="#7c3aed",
            ),
            LineSeries(
                name="Q(s,a)",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in analyses["q"]["rank_sweep"]),
                color="#2563eb",
            ),
            LineSeries(
                name="A(s,a)",
                values=tuple(entry["observed_relative_frobenius_error"] for entry in analyses["a"]["rank_sweep"]),
                color="#ea580c",
            ),
        ],
        x_label="TT rank",
        left_y_label="Observed relative error",
    )


def main() -> None:
    args = parse_args()
    tt_ranks = tuple(sorted(set(parse_int_tuple(args.tt_ranks))))
    if not tt_ranks or any(rank <= 0 for rank in tt_ranks):
        raise ValueError("TT rank sweep must contain positive integers.")

    tensor_file = np.load(args.tensor_file)
    analyses: dict[str, dict[str, Any]] = {}
    for tensor_name in ("q", "v", "a"):
        tensor_key = f"{tensor_name}_tensor"
        mask_key = f"{tensor_name}_observed_mask"
        if tensor_key not in tensor_file.files or mask_key not in tensor_file.files:
            continue
        tensor = torch.as_tensor(tensor_file[tensor_key], dtype=torch.float64)
        mask = torch.as_tensor(tensor_file[mask_key], dtype=torch.bool)
        analyses[tensor_name] = analyze_tt_rank_sweep(tensor, tt_ranks=tt_ranks, observed_mask=mask)

    if not analyses:
        raise ValueError(f"No tensor/mask pairs found in {args.tensor_file}")

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, str] = {}
    for tensor_name, analysis in analyses.items():
        output_path = figure_dir / f"{tensor_name}_error_vs_rank.svg"
        build_single_tensor_figure(str(output_path), tensor_name, analysis["rank_sweep"])
        figure_paths[f"{tensor_name}_error_vs_rank"] = str(output_path)

    if {"q", "v", "a"}.issubset(analyses):
        comparison_path = figure_dir / "v_q_a_comparison.svg"
        build_comparison_figure(str(comparison_path), analyses)
        figure_paths["v_q_a_comparison"] = str(comparison_path)

    result = {
        "tensor_file": args.tensor_file,
        "tt_ranks": list(tt_ranks),
        "analyses": analyses,
        "figures": figure_paths,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
