"""
Load a test case from its 4 CSV files into Warehouse + List[BayType].

Expected directory layout:
    <case_dir>/
        warehouse.csv       — polygon vertices (x, y) in mm
        types_of_bays.csv   — id, width, depth, height, col4, capacity, cost
        obstacles.csv       — x, y, width, depth  (may be empty)
        ceiling.csv         — y_break, max_height_mm  (sorted ascending)
"""

from __future__ import annotations
import os
from typing import List, Tuple
from models import Warehouse, BayType, Obstacle


def _rows(filepath: str) -> List[List[str]]:
    """Read non-empty CSV rows, stripping whitespace from each cell."""
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([c.strip() for c in line.split(",")])
    return rows


def _parse_polygon(filepath: str) -> List[Tuple[float, float]]:
    return [(float(r[0]), float(r[1])) for r in _rows(filepath) if len(r) >= 2]


def _parse_bay_types(filepath: str) -> List[BayType]:
    bays = []
    for r in _rows(filepath):
        if len(r) < 7:
            continue
        bays.append(BayType(
            id=str(int(float(r[0]))),
            width=float(r[1]),
            depth=float(r[2]),
            height=float(r[3]),
            col4=float(r[4]),   # educated guess: aisle clearance mm
            capacity=float(r[5]),
            cost=float(r[6]),
        ))
    return bays


def _parse_obstacles(filepath: str) -> List[Obstacle]:
    obs = []
    for r in _rows(filepath):
        if len(r) < 4:
            continue
        obs.append(Obstacle(
            x=float(r[0]), y=float(r[1]),
            width=float(r[2]), depth=float(r[3]),
        ))
    return obs


def _parse_ceiling(filepath: str) -> List[Tuple[float, float]]:
    profile = [(float(r[0]), float(r[1])) for r in _rows(filepath) if len(r) >= 2]
    return sorted(profile, key=lambda t: t[0])


def load_case(case_dir: str) -> tuple:
    """Return (Warehouse, List[BayType]) from a case directory."""
    polygon  = _parse_polygon(  os.path.join(case_dir, "warehouse.csv"))
    bay_types = _parse_bay_types(os.path.join(case_dir, "types_of_bays.csv"))
    obstacles = _parse_obstacles(os.path.join(case_dir, "obstacles.csv"))
    ceiling   = _parse_ceiling(  os.path.join(case_dir, "ceiling.csv"))

    # col4 is constant across all bay types within a case — use it as aisle_width
    aisle_width = bay_types[0].col4 if bay_types else 500.0

    warehouse = Warehouse(
        polygon=polygon,
        ceiling_profile=ceiling,
        aisle_width=aisle_width,
        obstacles=obstacles,
        demand=float("inf"),   # no explicit demand — maximise capacity
    )
    return warehouse, bay_types
