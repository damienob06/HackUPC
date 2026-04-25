"""
Solution validator.

After an algorithm produces a Solution we run an INDEPENDENT check:

    1. No bay protrudes outside the warehouse.
    2. No two bays overlap.
    3. No bay overlaps an obstacle.
    4. Total capacity meets demand.
    5. (Soft) Aisle gaps between rows are at least warehouse.aisle_width.

This is a defensive layer: if any algorithm has a bug, the validator
catches it before it becomes a wrong answer.
"""

from __future__ import annotations
from typing import List, Tuple
from models import Solution, Warehouse, BayType, PlacedBay


def validate(solution: Solution, warehouse: Warehouse,
             catalogue: List[BayType],
             tol: float = 1e-6) -> Tuple[bool, List[str]]:
    """Return (is_valid, list_of_problems)."""
    problems: List[str] = []
    by_id = {b.id: b for b in catalogue}

    # ----- 1. boundaries -----
    for i, p in enumerate(solution.placements):
        if p.x < -tol or p.y < -tol:
            problems.append(f"bay #{i} ({p.bay_id}) has negative coords "
                            f"({p.x:.3f}, {p.y:.3f})")
        if p.x + p.width > warehouse.width + tol:
            problems.append(f"bay #{i} ({p.bay_id}) exceeds warehouse width "
                            f"(x2={p.x + p.width:.3f} > {warehouse.width})")
        if p.y + p.depth > warehouse.depth + tol:
            problems.append(f"bay #{i} ({p.bay_id}) exceeds warehouse depth "
                            f"(y2={p.y + p.depth:.3f} > {warehouse.depth})")

    # ----- 2. pairwise overlaps -----
    for i, p in enumerate(solution.placements):
        for j in range(i + 1, len(solution.placements)):
            q = solution.placements[j]
            if not (p.x + p.width <= q.x + tol or q.x + q.width <= p.x + tol
                    or p.y + p.depth <= q.y + tol or q.y + q.depth <= p.y + tol):
                problems.append(f"bays #{i} ({p.bay_id}) and #{j} ({q.bay_id}) overlap")

    # ----- 3. obstacles -----
    for i, p in enumerate(solution.placements):
        for o in warehouse.obstacles:
            if not (p.x + p.width <= o.x + tol or o.x + o.width <= p.x + tol
                    or p.y + p.depth <= o.y + tol or o.y + o.depth <= p.y + tol):
                problems.append(f"bay #{i} ({p.bay_id}) collides with obstacle "
                                f"at ({o.x},{o.y})")

    # ----- 4. dimensions match catalogue -----
    for i, p in enumerate(solution.placements):
        bt = by_id.get(p.bay_id)
        if bt is None:
            problems.append(f"bay #{i} has unknown id '{p.bay_id}'")
            continue
        if p.rotated:
            if abs(p.width - bt.depth) > tol or abs(p.depth - bt.width) > tol:
                problems.append(f"bay #{i} ({p.bay_id}) rotated dims mismatch")
        else:
            if abs(p.width - bt.width) > tol or abs(p.depth - bt.depth) > tol:
                problems.append(f"bay #{i} ({p.bay_id}) dims mismatch catalogue")

    # ----- 5. capacity -----
    if solution.total_capacity < warehouse.demand - tol:
        problems.append(f"capacity {solution.total_capacity:.2f} < "
                        f"demand {warehouse.demand}")

    # ----- 6. aisle gap (soft) -----
    # group placements by their bottom-y, find gaps between consecutive rows
    if solution.placements:
        ys = sorted({round(p.y, 3) for p in solution.placements})
        for prev, curr in zip(ys, ys[1:]):
            row_top = max(p.y + p.depth for p in solution.placements
                          if abs(p.y - prev) < 1e-3)
            gap = curr - row_top
            if gap < warehouse.aisle_width - 0.05:
                problems.append(f"aisle between row y={prev:.2f} and y={curr:.2f}"
                                f" is {gap:.2f}m < {warehouse.aisle_width}m")

    return (len(problems) == 0, problems)


def print_validation(solution: Solution, warehouse: Warehouse,
                     catalogue: List[BayType]) -> bool:
    ok, issues = validate(solution, warehouse, catalogue)
    if ok:
        print(f"  ✓ {solution.method}: passes all validity checks")
        return True
    print(f"  ✗ {solution.method}: {len(issues)} issue(s):")
    for s in issues[:10]:
        print(f"      - {s}")
    if len(issues) > 10:
        print(f"      ... and {len(issues) - 10} more")
    return False
