"""
Data models for the Warehouse Optimizer.

A Warehouse has a rectangular footprint, a storage demand to cover,
and optional aisle / obstacle constraints. A BayType is a catalogue
entry with a footprint, storage capacity and cost. PlacedBay is one
concrete instance of a BayType installed at coordinates (x, y), and a
Solution is the result of an algorithm: which bays were placed, where,
total cost and total capacity covered.

Coordinate convention: origin (0, 0) at the bottom-left corner of the
warehouse. x grows to the right (width), y grows up (depth). All
distances in metres, capacities in 'units' (could be pallets or m^3).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
import json


@dataclass
class BayType:
    """A row in the bay catalogue."""
    id: str
    width: float        # along x-axis when not rotated
    depth: float        # along y-axis when not rotated
    capacity: float     # storage units this bay holds
    cost: float         # cost to install one bay of this type

    @property
    def footprint(self) -> float:
        return self.width * self.depth

    @property
    def cost_per_unit(self) -> float:
        """Lower is better — used as a greedy ranking key."""
        return self.cost / self.capacity if self.capacity > 0 else float("inf")

    @property
    def density(self) -> float:
        """Capacity per square metre — a tie-breaker for greedy."""
        return self.capacity / self.footprint if self.footprint > 0 else 0.0


@dataclass
class Obstacle:
    """A rectangular blocked region (column, door, fixed equipment)."""
    x: float
    y: float
    width: float
    depth: float

    def overlaps(self, x: float, y: float, w: float, d: float) -> bool:
        return not (x + w <= self.x or self.x + self.width <= x
                    or y + d <= self.y or self.y + self.depth <= y)


@dataclass
class Warehouse:
    width: float
    depth: float
    demand: float                       # total capacity required
    aisle_width: float = 2.5            # gap between bay rows for forklifts
    obstacles: List[Obstacle] = field(default_factory=list)

    @property
    def area(self) -> float:
        return self.width * self.depth

    @property
    def usable_area(self) -> float:
        """Footprint minus obstacles. Aisle overhead is approximated separately."""
        return self.area - sum(o.width * o.depth for o in self.obstacles)


@dataclass
class PlacedBay:
    """One installed bay at a specific (x, y) with optional 90° rotation."""
    bay_id: str
    x: float
    y: float
    width: float      # effective width after rotation
    depth: float      # effective depth after rotation
    rotated: bool = False

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.depth

    def overlaps(self, other: "PlacedBay", tol: float = 1e-6) -> bool:
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
    extra: Dict = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.total_capacity >= self.demand - 1e-6

    @property
    def utilisation(self) -> float:
        used = sum(p.width * p.depth for p in self.placements)
        return used / self.warehouse_area if self.warehouse_area > 0 else 0.0

    def summary(self) -> str:
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        return (f"[{self.method:24s}] {status} | "
                f"cost={self.total_cost:>10,.2f} | "
                f"cap={self.total_capacity:>8.1f}/{self.demand:.1f} | "
                f"bays={len(self.placements):>3d} | "
                f"util={self.utilisation*100:>5.1f}% | "
                f"t={self.runtime_s:.2f}s")


def load_instance(path: str) -> Tuple[Warehouse, List[BayType]]:
    """Load a problem instance from a JSON file.

    Schema:
    {
      "warehouse": {"width": ..., "depth": ..., "demand": ...,
                    "aisle_width": ..., "obstacles": [{"x":..., "y":..., "width":..., "depth":...}]},
      "bays": [{"id": "...", "width": ..., "depth": ..., "capacity": ..., "cost": ...}, ...]
    }
    """
    with open(path) as f:
        data = json.load(f)
    w = data["warehouse"]
    obstacles = [Obstacle(**o) for o in w.get("obstacles", [])]
    warehouse = Warehouse(
        width=w["width"], depth=w["depth"], demand=w["demand"],
        aisle_width=w.get("aisle_width", 2.5), obstacles=obstacles,
    )
    bays = [BayType(**b) for b in data["bays"]]
    return warehouse, bays


def save_solution(solution: Solution, path: str) -> None:
    with open(path, "w") as f:
        json.dump({
            "method": solution.method,
            "total_cost": solution.total_cost,
            "total_capacity": solution.total_capacity,
            "demand": solution.demand,
            "feasible": solution.feasible,
            "utilisation": solution.utilisation,
            "runtime_s": solution.runtime_s,
            "placements": [asdict(p) for p in solution.placements],
        }, f, indent=2)
