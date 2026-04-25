"""
Data models — polygon warehouse, ceiling-height zones, bay height.

All distances in millimetres (mm). Capacity in pallets/units, cost in
arbitrary currency.

Column 4 of types_of_bays.csv is our educated guess for the minimum
aisle clearance required — it is constant per case (200 mm or 500 mm),
suggesting it is a warehouse-level parameter repeated on each row.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import math


# ---------------------------------------------------------------------------
# Polygon geometry utilities
# ---------------------------------------------------------------------------

def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Shoelace formula — always positive."""
    n = len(polygon)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += polygon[i][0] * polygon[j][1]
        a -= polygon[j][0] * polygon[i][1]
    return abs(a) / 2.0


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BayType:
    id: str
    width: float       # mm, along x when not rotated
    depth: float       # mm, along y when not rotated
    height: float      # mm, vertical — must fit under ceiling
    col4: float        # mm — educated guess: aisle clearance
    capacity: float    # pallets / storage units
    cost: float        # installation cost

    @property
    def footprint(self) -> float:
        return self.width * self.depth

    @property
    def cost_per_unit(self) -> float:
        return self.cost / self.capacity if self.capacity > 0 else float("inf")

    @property
    def density(self) -> float:
        """Capacity per mm² — primary greedy ranking key."""
        return self.capacity / self.footprint if self.footprint > 0 else 0.0


@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    depth: float

    def overlaps(self, x: float, y: float, w: float, d: float,
                 tol: float = 1.0) -> bool:
        return not (x + w <= self.x + tol or self.x + self.width <= x + tol
                    or y + d <= self.y + tol or self.y + self.depth <= y + tol)


@dataclass
class Warehouse:
    polygon: List[Tuple[float, float]]
    ceiling_profile: List[Tuple[float, float]]  # sorted [(y_break, max_height_mm), ...]
    aisle_width: float = 500.0                  # mm gap between bay rows
    obstacles: List[Obstacle] = field(default_factory=list)
    demand: float = 0.0                         # 0 or inf → maximise capacity

    # ---- bounding box ----
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        xs = [v[0] for v in self.polygon]
        ys = [v[1] for v in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def width(self) -> float:
        x0, _, x1, _ = self.bounding_box
        return x1 - x0

    @property
    def depth(self) -> float:
        _, y0, _, y1 = self.bounding_box
        return y1 - y0

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def usable_area(self) -> float:
        return self.area - sum(o.width * o.depth for o in self.obstacles)

    # ---- ceiling ----
    def ceiling_height_at(self, y: float) -> float:
        """Step-function ceiling: return the max bay height allowed at depth y."""
        if not self.ceiling_profile:
            return float("inf")
        h = self.ceiling_profile[0][1]
        for y_break, height in self.ceiling_profile:
            if y >= y_break:
                h = height
            else:
                break
        return h

    def next_tall_ceiling_y(self, y_start: float, required_height: float) -> float | None:
        """First y_break >= y_start where ceiling_height >= required_height."""
        for y_break, h in self.ceiling_profile:
            if y_break >= y_start and h >= required_height:
                return y_break
        return None

    # ---- polygon containment ----
    def contains_rect(self, x: float, y: float, w: float, d: float,
                      inset: float = 1.0) -> bool:
        """True if all 4 corners of the rectangle are inside the polygon.

        An inset of 1 mm avoids false positives when a bay edge exactly
        coincides with the polygon boundary (floating-point edge case).
        """
        return (point_in_polygon(x + inset,     y + inset,     self.polygon)
                and point_in_polygon(x + w - inset, y + inset,     self.polygon)
                and point_in_polygon(x + w - inset, y + d - inset, self.polygon)
                and point_in_polygon(x + inset,     y + d - inset, self.polygon))


# ---------------------------------------------------------------------------
# Solution dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlacedBay:
    bay_id: str
    x: float
    y: float
    width: float
    depth: float
    height: float          # bay height (mm) — needed for ceiling validation
    rotated: bool = False

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.depth

    def overlaps(self, other: "PlacedBay", tol: float = 1.0) -> bool:
        return not (self.x2 <= other.x + tol or other.x2 <= self.x + tol
                    or self.y2 <= other.y + tol or other.y2 <= self.y + tol)


@dataclass
class Solution:
    method: str
    placements: List[PlacedBay]
    total_cost: float
    total_capacity: float
    demand: float
    warehouse_area: float
    runtime_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        if self.demand <= 0 or math.isinf(self.demand):
            return len(self.placements) > 0
        return self.total_capacity >= self.demand - 1e-6

    @property
    def utilisation(self) -> float:
        used = sum(p.width * p.depth for p in self.placements)
        return used / self.warehouse_area if self.warehouse_area > 0 else 0.0

    def summary(self) -> str:
        return (f"[{self.method:28s}] "
                f"cost={self.total_cost:>10,.0f} | "
                f"cap={self.total_capacity:>6.0f} | "
                f"bays={len(self.placements):>3d} | "
                f"util={self.utilisation * 100:>5.1f}% | "
                f"t={self.runtime_s:.2f}s")
