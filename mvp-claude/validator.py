"""
Independent geometric validator.

Checks every constraint without calling back into the packer:
  1. Bay fully inside warehouse polygon
  2. Bay height fits under ceiling at its y position
  3. No two bays overlap
  4. No bay overlaps an obstacle
  5. Aisle gap >= warehouse.aisle_width between vertically adjacent bays
"""

from __future__ import annotations
from typing import List, Tuple
from models import Solution, Warehouse, BayType, PlacedBay


def validate(solution: Solution, warehouse: Warehouse,
             catalogue: List[BayType],
             tol: float = 1.0) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    by_id = {b.id: b for b in catalogue}

    for i, p in enumerate(solution.placements):
        # 1. polygon containment
        if not warehouse.contains_rect(p.x, p.y, p.width, p.depth):
            problems.append(
                f"bay #{i} (type {p.bay_id}) extends outside warehouse polygon "
                f"at ({p.x:.0f}, {p.y:.0f})"
            )

        # 2. ceiling
        ceiling = warehouse.ceiling_height_at(p.y)
        if p.height > ceiling + tol:
            problems.append(
                f"bay #{i} (type {p.bay_id}) height {p.height:.0f} mm "
                f"exceeds ceiling {ceiling:.0f} mm at y={p.y:.0f}"
            )

        # 3. catalogue dimension match
        bt = by_id.get(p.bay_id)
        if bt is None:
            problems.append(f"bay #{i} references unknown type '{p.bay_id}'")
        else:
            if p.rotated:
                ok = (abs(p.width - bt.depth) <= tol
                      and abs(p.depth - bt.width) <= tol)
            else:
                ok = (abs(p.width - bt.width) <= tol
                      and abs(p.depth - bt.depth) <= tol)
            if not ok:
                problems.append(
                    f"bay #{i} (type {p.bay_id}) dimensions "
                    f"{p.width:.0f}×{p.depth:.0f} don't match catalogue"
                )

    # 4. pairwise overlaps
    for i, p in enumerate(solution.placements):
        for j in range(i + 1, len(solution.placements)):
            q = solution.placements[j]
            if not (p.x + p.width <= q.x + tol or q.x + q.width <= p.x + tol
                    or p.y + p.depth <= q.y + tol or q.y + q.depth <= p.y + tol):
                problems.append(
                    f"bay #{i} (type {p.bay_id}) overlaps bay #{j} "
                    f"(type {q.bay_id})"
                )

    # 5. obstacle collisions
    for i, p in enumerate(solution.placements):
        for o in warehouse.obstacles:
            if not (p.x + p.width <= o.x + tol or o.x + o.width <= p.x + tol
                    or p.y + p.depth <= o.y + tol or o.y + o.depth <= p.y + tol):
                problems.append(
                    f"bay #{i} (type {p.bay_id}) collides with obstacle "
                    f"at ({o.x:.0f}, {o.y:.0f})"
                )

    # 6. aisle gap (soft check — warns, doesn't reject)
    aw = warehouse.aisle_width
    for i, p in enumerate(solution.placements):
        for j in range(i + 1, len(solution.placements)):
            q = solution.placements[j]
            x_overlap = not (p.x + p.width <= q.x + tol
                              or q.x + q.width <= p.x + tol)
            if not x_overlap:
                continue
            gap_above_p = q.y - (p.y + p.depth)
            gap_above_q = p.y - (q.y + q.depth)
            if 0 < gap_above_p < aw - 10:
                problems.append(
                    f"[aisle] bay #{i} top to bay #{j} bottom: "
                    f"gap {gap_above_p:.0f} mm < {aw:.0f} mm aisle"
                )
            elif 0 < gap_above_q < aw - 10:
                problems.append(
                    f"[aisle] bay #{j} top to bay #{i} bottom: "
                    f"gap {gap_above_q:.0f} mm < {aw:.0f} mm aisle"
                )

    return len(problems) == 0, problems


def print_validation(solution: Solution, warehouse: Warehouse,
                     catalogue: List[BayType]) -> bool:
    ok, issues = validate(solution, warehouse, catalogue)
    if ok:
        print(f"  PASS  {solution.method}")
        return True
    print(f"  FAIL  {solution.method}: {len(issues)} issue(s)")
    for s in issues[:8]:
        print(f"        - {s}")
    if len(issues) > 8:
        print(f"        ... and {len(issues) - 8} more")
    return False
