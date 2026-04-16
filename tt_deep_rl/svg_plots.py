from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LineSeries:
    name: str
    values: tuple[float, ...]
    color: str
    axis: str = "left"
    dashed: bool = False


def save_line_chart_svg(
    output_path: str,
    title: str,
    subtitle: str,
    x_labels: Sequence[str],
    series: Sequence[LineSeries],
    x_label: str,
    left_y_label: str,
    right_y_label: str | None = None,
    left_scale: str = "linear",
    right_scale: str = "linear",
) -> None:
    if not x_labels:
        raise ValueError("x_labels must not be empty.")
    if not series:
        raise ValueError("At least one series is required to draw a chart.")

    width = 960
    height = 640
    margin_left = 96
    margin_right = 96 if any(item.axis == "right" for item in series) else 48
    margin_top = 90
    margin_bottom = 96
    plot_left = margin_left
    plot_top = margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height

    left_series = [item for item in series if item.axis == "left"]
    right_series = [item for item in series if item.axis == "right"]

    left_values = _flatten_series(left_series)
    right_values = _flatten_series(right_series)
    left_min, left_max = _normalize_range(left_values, scale=left_scale)
    right_min, right_max = _normalize_range(right_values, scale=right_scale) if right_series else (0.0, 1.0)

    left_ticks = _build_ticks(left_min, left_max, scale=left_scale)
    right_ticks = _build_ticks(right_min, right_max, scale=right_scale) if right_series else []

    x_positions = _x_positions(plot_left, plot_width, len(x_labels))
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{width / 2:.1f}" y="38" text-anchor="middle" fill="#0f172a" '
        'font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700">'
        f"{escape(title)}</text>",
        f'<text x="{width / 2:.1f}" y="64" text-anchor="middle" fill="#475569" '
        'font-family="Arial, Helvetica, sans-serif" font-size="15">'
        f"{escape(subtitle)}</text>",
    ]

    for tick in left_ticks:
        y = _project_value(tick, left_min, left_max, plot_top, plot_height, left_scale)
        svg_parts.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#cbd5e1" stroke-width="1"/>')
        svg_parts.append(
            f'<text x="{plot_left - 12}" y="{y + 5:.2f}" text-anchor="end" fill="#334155" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13">'
            f"{escape(_format_tick_label(tick, left_scale))}</text>"
        )

    svg_parts.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#0f172a" stroke-width="2"/>')
    svg_parts.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#0f172a" stroke-width="2"/>')

    if right_series:
        svg_parts.append(f'<line x1="{plot_right}" y1="{plot_top}" x2="{plot_right}" y2="{plot_bottom}" stroke="#0f172a" stroke-width="2"/>')
        for tick in right_ticks:
            y = _project_value(tick, right_min, right_max, plot_top, plot_height, right_scale)
            svg_parts.append(
                f'<text x="{plot_right + 12}" y="{y + 5:.2f}" text-anchor="start" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" font-size="13">'
                f"{escape(_format_tick_label(tick, right_scale))}</text>"
            )

    for label, x in zip(x_labels, x_positions):
        svg_parts.append(f'<line x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" y2="{plot_bottom + 6}" stroke="#0f172a" stroke-width="1.5"/>')
        svg_parts.append(
            f'<text x="{x:.2f}" y="{plot_bottom + 28}" text-anchor="middle" fill="#334155" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13">'
            f"{escape(label)}</text>"
        )

    for item in series:
        scale = left_scale if item.axis == "left" else right_scale
        axis_min = left_min if item.axis == "left" else right_min
        axis_max = left_max if item.axis == "left" else right_max
        polyline_points = []
        for x, value in zip(x_positions, item.values):
            y = _project_value(value, axis_min, axis_max, plot_top, plot_height, scale)
            polyline_points.append(f"{x:.2f},{y:.2f}")
        dash = ' stroke-dasharray="10 8"' if item.dashed else ""
        svg_parts.append(
            f'<polyline points="{" ".join(polyline_points)}" fill="none" stroke="{item.color}" '
            f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"{dash}/>'
        )
        for x, value in zip(x_positions, item.values):
            y = _project_value(value, axis_min, axis_max, plot_top, plot_height, scale)
            svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{item.color}" stroke="#ffffff" stroke-width="2"/>')

    legend_x = plot_left + 8
    legend_y = 84
    for index, item in enumerate(series):
        y = legend_y + index * 24
        dash = ' stroke-dasharray="10 8"' if item.dashed else ""
        svg_parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{item.color}" stroke-width="4"{dash}/>'
        )
        svg_parts.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="4.5" fill="{item.color}" stroke="#ffffff" stroke-width="1.5"/>')
        axis_suffix = " (right axis)" if item.axis == "right" else ""
        svg_parts.append(
            f'<text x="{legend_x + 38}" y="{y + 5}" text-anchor="start" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13">'
            f"{escape(item.name + axis_suffix)}</text>"
        )

    svg_parts.append(
        f'<text x="{plot_left + plot_width / 2:.2f}" y="{height - 26}" text-anchor="middle" fill="#0f172a" '
        'font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700">'
        f"{escape(x_label)}</text>"
    )
    svg_parts.append(
        f'<text x="28" y="{plot_top + plot_height / 2:.2f}" text-anchor="middle" fill="#0f172a" '
        'font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" '
        f'transform="rotate(-90 28 {plot_top + plot_height / 2:.2f})">{escape(left_y_label)}</text>'
    )
    if right_y_label is not None and right_series:
        svg_parts.append(
            f'<text x="{width - 24}" y="{plot_top + plot_height / 2:.2f}" text-anchor="middle" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" '
            f'transform="rotate(90 {width - 24} {plot_top + plot_height / 2:.2f})">{escape(right_y_label)}</text>'
        )

    svg_parts.append("</svg>")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def _flatten_series(series: Sequence[LineSeries]) -> list[float]:
    values: list[float] = []
    for item in series:
        values.extend(float(value) for value in item.values)
    return values


def _normalize_range(values: Iterable[float], scale: str) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if scale == "log10":
        positive_values = [value for value in values if value > 0.0]
        if not positive_values:
            return 0.0, 1.0
        minimum = min(positive_values)
        maximum = max(positive_values)
    if math.isclose(minimum, maximum):
        if minimum == 0.0:
            return 0.0, 1.0
        return minimum * 0.9, maximum * 1.1
    if scale == "linear":
        padding = (maximum - minimum) * 0.08
        minimum = max(0.0, minimum - padding)
        maximum = maximum + padding
    return minimum, maximum


def _build_ticks(minimum: float, maximum: float, scale: str) -> list[float]:
    if scale == "log10":
        start = math.log10(minimum)
        stop = math.log10(maximum)
        if math.isclose(start, stop):
            return [minimum, maximum]
        tick_count = 5
        return [10 ** (start + (stop - start) * index / (tick_count - 1)) for index in range(tick_count)]
    tick_count = 5
    return [minimum + (maximum - minimum) * index / (tick_count - 1) for index in range(tick_count)]


def _project_value(
    value: float,
    minimum: float,
    maximum: float,
    plot_top: int,
    plot_height: int,
    scale: str,
) -> float:
    if scale == "log10":
        value = math.log10(max(value, 1e-12))
        minimum = math.log10(max(minimum, 1e-12))
        maximum = math.log10(max(maximum, 1e-12))
    if math.isclose(minimum, maximum):
        return plot_top + plot_height / 2
    ratio = (value - minimum) / (maximum - minimum)
    return plot_top + plot_height * (1.0 - ratio)


def _x_positions(plot_left: int, plot_width: int, count: int) -> list[float]:
    if count == 1:
        return [plot_left + plot_width / 2]
    step = plot_width / (count - 1)
    return [plot_left + index * step for index in range(count)]


def _format_tick_label(value: float, scale: str) -> str:
    if scale == "log10":
        return f"{value:.1f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"
