"""
Visualisation: render a Solution as a top-down warehouse plan.

Draws the polygon warehouse outline, ceiling-zone colour bands,
obstacles, and placed bays (coloured by type).
"""

from __future__ import annotations
import os
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from models import Solution, Warehouse, BayType


_COLOURS = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0",
]

_CEILING_COLOURS = {
    6000: "#E3F2FD", 5000: "#E3F2FD",
    4000: "#E8F5E9", 3000: "#FFF9C4",
    2000: "#FFE0B2", 1800: "#FFCCBC",
}


def _bay_colour(bay_id: str, mapping: Dict[str, str]) -> str:
    if bay_id not in mapping:
        mapping[bay_id] = _COLOURS[len(mapping) % len(_COLOURS)]
    return mapping[bay_id]


def _ceiling_bg(height_mm: float) -> str:
    for h, c in sorted(_CEILING_COLOURS.items(), reverse=True):
        if height_mm >= h:
            return c
    return "#FFCCBC"


def render(solution: Solution, warehouse: Warehouse,
           catalogue: List[BayType], path: str,
           title: str = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    _draw(ax, solution, warehouse, catalogue, title=title or solution.method)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_comparison(solutions: List[Solution], warehouse: Warehouse,
                      catalogue: List[BayType], path: str) -> None:
    n = len(solutions)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(13 * cols, 10 * rows),
                              squeeze=False)
    colour_map: Dict[str, str] = {}
    for idx, sol in enumerate(solutions):
        ax = axes[idx // cols][idx % cols]
        _draw(ax, sol, warehouse, catalogue,
              title=sol.summary(), colour_map=colour_map)
    for idx in range(len(solutions), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _draw(ax, solution: Solution, warehouse: Warehouse,
          catalogue: List[BayType], title: str,
          colour_map: Dict[str, str] = None) -> None:
    if colour_map is None:
        colour_map = {}
    by_id = {b.id: b for b in catalogue}

    x_min, y_min, x_max, y_max = warehouse.bounding_box

    # ---- ceiling zone bands (drawn first, behind everything) ----
    profile = warehouse.ceiling_profile
    for i, (y_start, h) in enumerate(profile):
        y_end = profile[i + 1][0] if i + 1 < len(profile) else y_max
        ax.axhspan(y_start, y_end, alpha=0.25, color=_ceiling_bg(h), zorder=0)
        ax.text(x_max + (x_max - x_min) * 0.02, (y_start + y_end) / 2,
                f"{h / 1000:.1f}m",
                va="center", ha="left", fontsize=7, color="#666666")

    # ---- warehouse polygon outline ----
    poly_pts = list(warehouse.polygon)
    poly_patch = MplPolygon(poly_pts, closed=True, fill=False,
                             edgecolor="black", linewidth=2.5, zorder=3)
    ax.add_patch(poly_patch)

    # ---- obstacles ----
    for o in warehouse.obstacles:
        ax.add_patch(patches.Rectangle(
            (o.x, o.y), o.width, o.depth,
            facecolor="#333333", edgecolor="black",
            hatch="///", alpha=0.8, linewidth=1, zorder=4,
        ))

    # ---- placed bays ----
    for p in solution.placements:
        c = _bay_colour(p.bay_id, colour_map)
        ax.add_patch(patches.Rectangle(
            (p.x, p.y), p.width, p.depth,
            facecolor=c, edgecolor="black",
            linewidth=0.6, alpha=0.85, zorder=5,
        ))
        if p.width >= 500 and p.depth >= 400:
            label = p.bay_id + ("↻" if p.rotated else "")
            ax.text(p.x + p.width / 2, p.y + p.depth / 2, label,
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", zorder=6)

    # ---- axes ----
    margin = (x_max - x_min) * 0.08
    ax.set_xlim(x_min - margin, x_max + margin * 3)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.3)

    # convert to metres for axis labels
    scale = 1000.0
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/scale:.1f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/scale:.1f}"))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    subtitle = (f"cap={solution.total_capacity:.0f} pallets | "
                f"cost={solution.total_cost:,.0f} | "
                f"bays={len(solution.placements)} | "
                f"util={solution.utilisation * 100:.1f}% | "
                f"t={solution.runtime_s:.2f}s")
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)

    # ---- legend ----
    handles = []
    for bay_id, colour in colour_map.items():
        bt = by_id.get(bay_id)
        if bt is None:
            continue
        cnt = sum(1 for p in solution.placements if p.bay_id == bay_id)
        handles.append(patches.Patch(
            facecolor=colour, edgecolor="black",
            label=(f"Type {bay_id} ×{cnt}  "
                   f"{bt.width/1000:.2f}×{bt.depth/1000:.2f}m  "
                   f"h={bt.height/1000:.2f}m  "
                   f"cap={bt.capacity:.0f}  "
                   f"cost={bt.cost:,.0f}"),
        ))
    if handles:
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0),
                  fontsize=7, frameon=True)
