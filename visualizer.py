"""
Visualisation: render a Solution as a top-down warehouse plan.

Produces one PNG per algorithm and a comparison grid.
"""

from __future__ import annotations
import os
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import Solution, Warehouse, BayType


# A colour-blind friendly palette
_COLOURS = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def _colour_for(bay_id: str, mapping: Dict[str, str]) -> str:
    if bay_id not in mapping:
        mapping[bay_id] = _COLOURS[len(mapping) % len(_COLOURS)]
    return mapping[bay_id]


def render(solution: Solution, warehouse: Warehouse,
           catalogue: List[BayType], path: str,
           title: str = None) -> None:
    """Render one solution to a PNG file."""
    fig, ax = plt.subplots(figsize=(12, 7))
    _draw(ax, solution, warehouse, catalogue,
          title=title or solution.method)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_comparison(solutions: List[Solution], warehouse: Warehouse,
                      catalogue: List[BayType], path: str) -> None:
    """Render a grid of all solutions side-by-side."""
    n = len(solutions)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13 * cols, 7 * rows),
                             squeeze=False)
    colour_map: Dict[str, str] = {}
    for idx, sol in enumerate(solutions):
        ax = axes[idx // cols][idx % cols]
        _draw(ax, sol, warehouse, catalogue,
              title=sol.summary(), colour_map=colour_map)
    # blank remaining axes
    for idx in range(len(solutions), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _draw(ax, solution: Solution, warehouse: Warehouse,
          catalogue: List[BayType], title: str,
          colour_map: Dict[str, str] = None) -> None:
    if colour_map is None:
        colour_map = {}
    by_id = {b.id: b for b in catalogue}

    # Warehouse outline
    ax.add_patch(patches.Rectangle((0, 0), warehouse.width, warehouse.depth,
                                    fill=False, edgecolor="black",
                                    linewidth=2.5))
    # subtle grid
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.35)

    # Obstacles
    for o in warehouse.obstacles:
        ax.add_patch(patches.Rectangle((o.x, o.y), o.width, o.depth,
                                        facecolor="#222222", edgecolor="black",
                                        hatch="///", alpha=0.85, linewidth=1))

    # Bays
    for p in solution.placements:
        c = _colour_for(p.bay_id, colour_map)
        ax.add_patch(patches.Rectangle((p.x, p.y), p.width, p.depth,
                                        facecolor=c, edgecolor="black",
                                        linewidth=0.8, alpha=0.85))
        if p.width >= 1.6 and p.depth >= 1.0:
            label = p.bay_id + ("↻" if p.rotated else "")
            ax.text(p.x + p.width / 2, p.y + p.depth / 2, label,
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")

    # Aisle shading (between rows)
    ys_used = sorted({round(p.y, 3) for p in solution.placements})
    row_tops = []
    for y in ys_used:
        row_bays = [p for p in solution.placements if abs(p.y - y) < 1e-3]
        if row_bays:
            top = y + max(p.depth for p in row_bays)
            row_tops.append((y, top))

    ax.set_xlim(-1, warehouse.width + 1)
    ax.set_ylim(-1, warehouse.depth + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("width (m)")
    ax.set_ylabel("depth (m)")

    status = "✓ feasible" if solution.feasible else "✗ INFEASIBLE"
    subtitle = (f"{status} | cost={solution.total_cost:,.0f} | "
                f"capacity={solution.total_capacity:.0f}/{solution.demand:.0f} | "
                f"bays={len(solution.placements)} | "
                f"util={solution.utilisation*100:.1f}% | "
                f"t={solution.runtime_s:.2f}s")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    # Legend
    handles = []
    for bay_id, colour in colour_map.items():
        bt = by_id.get(bay_id)
        if bt is None:
            continue
        cnt = sum(1 for p in solution.placements if p.bay_id == bay_id)
        handles.append(patches.Patch(facecolor=colour, edgecolor="black",
                                       label=f"{bay_id} (×{cnt}, "
                                             f"{bt.width}×{bt.depth}m, "
                                             f"cap={bt.capacity}, "
                                             f"€{bt.cost})"))
    if handles:
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)
