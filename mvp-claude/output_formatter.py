"""
Format a Solution as a UI-friendly JSON document.

The output is self-contained: it includes the full warehouse geometry,
ceiling zones (with colours and labels), obstacles, bay-type catalogue,
every placement with its colour, and a summary stats block.

All distances are in mm so the UI can render at any scale.
"""

from __future__ import annotations
from typing import List, Dict, Any
import datetime
from models import Warehouse, BayType, Solution


# Colour-blind-safe palette (Wong 2011) + extras
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0",
    "#00BCD4", "#FF5722", "#795548", "#607D8B", "#E91E63",
]

# Ceiling-height → background colour for the zone strip
_CEILING_COLOURS = {
    # height_mm : colour
    6000: "#E3F2FD",   # tall — light blue
    5000: "#E3F2FD",
    4000: "#E8F5E9",   # medium-tall — light green
    3000: "#FFF9C4",   # medium — light yellow
    2000: "#FFE0B2",   # low — light amber
    1800: "#FFCCBC",   # very low — light orange
}


def _ceiling_colour(height_mm: float) -> str:
    for h, c in sorted(_CEILING_COLOURS.items(), reverse=True):
        if height_mm >= h:
            return c
    return "#FFCCBC"


def _bay_colour(type_id: str, mapping: Dict[str, str]) -> str:
    if type_id not in mapping:
        mapping[type_id] = _PALETTE[len(mapping) % len(_PALETTE)]
    return mapping[type_id]


def format_solution_for_ui(solution: Solution,
                           warehouse: Warehouse,
                           catalogue: List[BayType],
                           case_name: str = "") -> Dict[str, Any]:
    by_id = {b.id: b for b in catalogue}
    colour_map: Dict[str, str] = {}

    x_min, y_min, x_max, y_max = warehouse.bounding_box

    # ---- ceiling zones ----
    zones = []
    profile = warehouse.ceiling_profile
    for i, (y_start, h) in enumerate(profile):
        y_end = profile[i + 1][0] if i + 1 < len(profile) else y_max
        zones.append({
            "y_from": y_start,
            "y_to": y_end,
            "height_mm": h,
            "height_m": round(h / 1000, 2),
            "color": _ceiling_colour(h),
            "label": f"{h / 1000:.1f}m ceiling",
        })

    # ---- bay types with colours ----
    bay_types_out = []
    for b in catalogue:
        c = _bay_colour(b.id, colour_map)
        bay_types_out.append({
            "id": b.id,
            "width_mm": b.width,
            "depth_mm": b.depth,
            "height_mm": b.height,
            "capacity": b.capacity,
            "cost": b.cost,
            "cost_per_unit": round(b.cost_per_unit, 2),
            "density_cap_per_m2": round(b.density * 1e6, 4),
            "color": c,
            "label": (f"Type {b.id}: "
                      f"{b.width / 1000:.2f}×{b.depth / 1000:.2f}m "
                      f"h={b.height / 1000:.2f}m  "
                      f"{b.capacity:.0f} pallet{'s' if b.capacity != 1 else ''}"),
        })

    # ---- placements ----
    placements_out = []
    for idx, p in enumerate(solution.placements):
        bt = by_id.get(p.bay_id)
        placements_out.append({
            "idx": idx,
            "type_id": p.bay_id,
            "x": p.x,
            "y": p.y,
            "width": p.width,
            "depth": p.depth,
            "height": p.height,
            "rotated": p.rotated,
            "capacity": bt.capacity if bt else 0,
            "cost": bt.cost if bt else 0,
            "color": colour_map.get(p.bay_id, "#888888"),
            "tooltip": (f"Type {p.bay_id} "
                        f"@ ({p.x:.0f}, {p.y:.0f}) mm "
                        f"{'[rotated]' if p.rotated else ''}"),
        })

    # ---- per-type breakdown ----
    from collections import Counter
    type_counts: Counter = Counter(p.bay_id for p in solution.placements)
    breakdown = []
    for type_id, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        bt = by_id.get(type_id)
        if bt:
            breakdown.append({
                "type_id": type_id,
                "count": count,
                "capacity": bt.capacity * count,
                "cost": bt.cost * count,
                "color": colour_map.get(type_id, "#888888"),
            })

    # ---- assemble ----
    return {
        "case": case_name,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "algorithm": solution.method,
        "solve_time_s": round(solution.runtime_s, 3),

        "warehouse": {
            "polygon": list(warehouse.polygon),
            "bounding_box": {
                "x_min": x_min, "y_min": y_min,
                "x_max": x_max, "y_max": y_max,
            },
            "area_mm2": round(warehouse.area),
            "area_m2": round(warehouse.area / 1e6, 2),
            "aisle_width_mm": warehouse.aisle_width,
        },

        "ceiling_zones": zones,

        "obstacles": [
            {"x": o.x, "y": o.y, "width": o.width, "depth": o.depth}
            for o in warehouse.obstacles
        ],

        "bay_types": bay_types_out,
        "placements": placements_out,

        "stats": {
            "total_bays": len(solution.placements),
            "total_capacity": solution.total_capacity,
            "total_cost": solution.total_cost,
            "floor_utilization_pct": round(solution.utilisation * 100, 1),
            "capacity_per_m2": round(
                solution.total_capacity / (warehouse.area / 1e6), 2
            ) if warehouse.area > 0 else 0,
            "capacity_per_cost": round(
                solution.total_capacity / solution.total_cost, 6
            ) if solution.total_cost > 0 else 0,
            "bay_type_breakdown": breakdown,
        },
    }
