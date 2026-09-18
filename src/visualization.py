"""Create lightweight, dependency-free SVG report figures.

The Streamlit dashboard supplies interactive Plotly charts. SVG is used for the
batch report so the core pipeline remains portable in minimal Python setups.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import math
import pandas as pd


def _svg_bar(rows: list[tuple[str, float]], title: str, x_label: str) -> str:
    width, height, margin = 980, max(320, 95 + 42 * len(rows)), 260
    maximum = max((value for _, value in rows), default=1) or 1
    bars = []
    for index, (label, value) in enumerate(rows):
        y = 65 + index * 42; bar_width = 650 * value / maximum
        bars.append(f'<text x="{margin-10}" y="{y+18}" text-anchor="end">{escape(str(label))}</text><rect x="{margin}" y="{y}" width="{bar_width:.1f}" height="25" fill="#2563eb"/><text x="{margin+bar_width+8:.1f}" y="{y+18}">{value:,.0f}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial" font-size="13"><rect width="100%" height="100%" fill="white"/><text x="25" y="32" font-size="21" font-weight="bold">{escape(title)}</text>{"".join(bars)}<text x="{margin}" y="{height-15}" fill="#555">{escape(x_label)}</text></svg>'


def _svg_line(rows: list[tuple[str, float]], title: str) -> str:
    width, height, left, top = 1050, 430, 70, 55
    values = [value for _, value in rows]; maximum = max(values) or 1; minimum = min(values)
    scale = max(maximum - minimum, 1); plot_w, plot_h = 920, 300
    points = []
    for i, (_, value) in enumerate(rows):
        x = left + plot_w * i / max(len(rows)-1, 1); y = top + plot_h * (1 - (value-minimum)/scale); points.append(f"{x:.1f},{y:.1f}")
    tick_every = max(1, math.ceil(len(rows) / 12))
    labels = "".join(f'<text x="{left + plot_w*i/max(len(rows)-1,1):.1f}" y="390" text-anchor="middle">{escape(label)}</text>' for i, (label, _) in enumerate(rows) if i % tick_every == 0)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial" font-size="11"><rect width="100%" height="100%" fill="white"/><text x="25" y="32" font-size="21" font-weight="bold">{escape(title)}</text><line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#444"/><polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{' '.join(points)}"/>{labels}<text x="18" y="{top+10}">{maximum:,.0f}</text><text x="18" y="{top+plot_h}">{minimum:,.0f}</text></svg>'


def save_standard_figures(data: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    monthly = list(data.groupby("year_month")["revenue"].sum().items())
    category = list(data.groupby("category")["revenue"].sum().sort_values(ascending=False).items())
    profit = list(data.groupby("category")["profit"].sum().sort_values(ascending=False).items())
    (output_dir / "monthly_revenue.svg").write_text(_svg_line(monthly, "Monthly revenue trend"), encoding="utf-8")
    (output_dir / "revenue_by_category.svg").write_text(_svg_bar(category, "Revenue by category", "Revenue"), encoding="utf-8")
    (output_dir / "profit_by_category.svg").write_text(_svg_bar(profit, "Profit by category", "Profit"), encoding="utf-8")
