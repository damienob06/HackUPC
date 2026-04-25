"""
2D placement engine.

Given a list of bays (multi-set of BayType) we need to physically lay
them out inside the warehouse without overlap, respecting aisles and
obstacles. The algorithm here is a 'shelf packing' variant tuned for
warehouse layouts:

    * Bays are placed in horizontal rows (shelves).
    * Within a row, bays are placed left-to-right against the bottom
      edge of the row.
    * The row height equals the deepest bay in that row.
    * Between rows we leave an aisle of `warehouse.aisle_width`.
    * 90° rotation is allowed (bays can be placed either way).
    * Obstacles are respected by skipping any candidate position that
      collides with one.

This is fast (O(n log n)) and produces realistic warehouse aisle
layouts.  The result is also more predictable than a free-form
guillotine packing, which matters when you want to *verify* you fit a
candidate selection from the ILP.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from models import BayType, Warehouse, PlacedBay, Obstacle


def _fits(x: float, y: float, w: float, d: float,
          warehouse: Warehouse, placed: List[PlacedBay]) -> bool:
    """Check warehouse boundary, obstacles and existing placements."""
    # Boundary
    if x < -1e-9 or y < -1e-9:
        return False
    if x + w > warehouse.width + 1e-9 or y + d > warehouse.depth + 1e-9:
        return False
    # Obstacles
    for o in warehouse.obstacles:
        if o.overlaps(x, y, w, d):
            return False
    # Other bays
    for p in placed:
        if not (x + w <= p.x + 1e-9 or p.x2 <= x + 1e-9
                or y + d <= p.y + 1e-9 or p.y2 <= y + 1e-9):
            return False
    return True


def shelf_pack(bays_to_place: List[BayType],
               warehouse: Warehouse,
               allow_rotation: bool = True) -> Tuple[List[PlacedBay], List[BayType]]:
    """Pack bays into the warehouse using row-based shelf packing.

    Strategy:
      1. Sort bays by max(width, depth) descending — biggest first
         (First-Fit Decreasing Height, the standard 2D heuristic).
      2. For each bay, try to put it into an existing row; if it
         doesn't fit anywhere, open a new row above the previous one
         (with an aisle gap).
      3. If rotation is allowed, try both orientations and keep the
         one that fits with least wasted shelf space.

    Returns (placed_bays, unplaced_bays). If the warehouse is too small,
    `unplaced` will be non-empty.
    """
    # Sort: tallest (deepest when not rotated) first — ties on width
    items = sorted(bays_to_place,
                   key=lambda b: (max(b.width, b.depth), b.width * b.depth),
                   reverse=True)

    placed: List[PlacedBay] = []
    unplaced: List[BayType] = []

    # Each row tracked as (y0, height, x_cursor)
    rows: List[List[float]] = []   # [y0, height, x_cursor]
    next_row_y = 0.0

    for bay in items:
        # Candidate orientations: (w, d, rotated?)
        orientations = [(bay.width, bay.depth, False)]
        if allow_rotation and abs(bay.width - bay.depth) > 1e-9:
            orientations.append((bay.depth, bay.width, True))

        best_choice = None  # (row_idx_or_-1_for_new, x, y, w, d, rotated)
        best_score = float("inf")

        # Try existing rows
        for ridx, (y0, h, cursor) in enumerate(rows):
            for w, d, rot in orientations:
                if d > h + 1e-9:
                    continue                     # too tall for the shelf
                x = cursor
                if not _fits(x, y0, w, d, warehouse, placed):
                    # try sliding right past obstacles
                    x = _slide_right(x, y0, w, d, warehouse, placed)
                    if x is None:
                        continue
                if x + w > warehouse.width + 1e-9:
                    continue
                # Score: prefer tighter fit (less wasted height)
                score = (h - d) + (x - cursor) * 0.1
                if score < best_score:
                    best_score = score
                    best_choice = (ridx, x, y0, w, d, rot)

        # Try a new row (if there's room above)
        for w, d, rot in orientations:
            y0 = next_row_y
            if y0 + d > warehouse.depth + 1e-9:
                continue
            x = 0.0
            if not _fits(x, y0, w, d, warehouse, placed):
                x = _slide_right(x, y0, w, d, warehouse, placed)
                if x is None or x + w > warehouse.width + 1e-9:
                    continue
            # Slight preference for opening a new row when it's a
            # near-perfect height match for the bay.
            score = 0.5 + (x * 0.1)
            if score < best_score:
                best_score = score
                best_choice = (-1, x, y0, w, d, rot)

        if best_choice is None:
            unplaced.append(bay)
            continue

        ridx, x, y, w, d, rot = best_choice
        placed.append(PlacedBay(bay_id=bay.id, x=x, y=y,
                                width=w, depth=d, rotated=rot))
        if ridx == -1:
            # opened new row
            rows.append([y, d, x + w])
            next_row_y = y + d + warehouse.aisle_width
        else:
            # extend existing row's height if this bay is deeper
            rows[ridx][1] = max(rows[ridx][1], d)
            rows[ridx][2] = x + w
            # if extending pushed the next-row baseline, recompute
            new_top = rows[ridx][0] + rows[ridx][1] + warehouse.aisle_width
            if new_top > next_row_y:
                next_row_y = new_top

    return placed, unplaced


def _slide_right(x: float, y: float, w: float, d: float,
                 warehouse: Warehouse, placed: List[PlacedBay],
                 step: float = 0.1) -> Optional[float]:
    """Try sliding the candidate to the right until it fits or runs out
    of warehouse. Used to skip obstacles. Returns the first feasible x
    or None if no slot exists in this row at this y."""
    cur = x
    while cur + w <= warehouse.width + 1e-9:
        if _fits(cur, y, w, d, warehouse, placed):
            return cur
        # jump to the right edge of whatever blocks us
        blocker_right = cur + step
        for o in warehouse.obstacles:
            if o.overlaps(cur, y, w, d):
                blocker_right = max(blocker_right, o.x + o.width)
        for p in placed:
            if not (cur + w <= p.x + 1e-9 or p.x2 <= cur + 1e-9
                    or y + d <= p.y + 1e-9 or p.y2 <= y + 1e-9):
                blocker_right = max(blocker_right, p.x2)
        cur = blocker_right + 1e-6
    return None


def pack(bays_to_place: List[BayType],
         warehouse: Warehouse,
         allow_rotation: bool = True,
         method: str = "skyline") -> Tuple[List[PlacedBay], List[BayType]]:
    """Unified packing entry point. Choose 'skyline' (default, tighter)
    or 'shelf' (legacy, faster but wastes more space)."""
    if method == "skyline":
        from skyline import skyline_pack
        return skyline_pack(bays_to_place, warehouse, allow_rotation)
    return shelf_pack(bays_to_place, warehouse, allow_rotation)


def expand_counts_to_list(counts: dict, catalogue: List[BayType]) -> List[BayType]:
    """Turn {bay_id: n} into a flat list of BayType references."""
    by_id = {b.id: b for b in catalogue}
    flat: List[BayType] = []
    for bay_id, n in counts.items():
        if bay_id not in by_id:
            raise KeyError(f"Unknown bay id {bay_id}")
        flat.extend([by_id[bay_id]] * int(round(n)))
    return flat


def evaluate_layout(placements: List[PlacedBay],
                    catalogue: List[BayType],
                    warehouse: Warehouse) -> Tuple[float, float, bool]:
    """Compute (total_cost, total_capacity, feasible_layout) for a set
    of placements. feasible_layout means: no overlaps, all in bounds,
    no obstacle collisions."""
    by_id = {b.id: b for b in catalogue}
    cost = 0.0
    cap = 0.0
    for p in placements:
        bt = by_id[p.bay_id]
        cost += bt.cost
        cap += bt.capacity

    feasible = True
    for i, p in enumerate(placements):
        if (p.x < -1e-9 or p.y < -1e-9
                or p.x2 > warehouse.width + 1e-9
                or p.y2 > warehouse.depth + 1e-9):
            feasible = False; break
        for o in warehouse.obstacles:
            if o.overlaps(p.x, p.y, p.width, p.depth):
                feasible = False; break
        if not feasible:
            break
        for q in placements[i + 1:]:
            if p.overlaps(q):
                feasible = False; break
        if not feasible:
            break
    return cost, cap, feasible
