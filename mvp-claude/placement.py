"""
2D placement engine — polygon-aware, ceiling-height-aware.

Bays are packed in horizontal rows (shelf packing). Key differences from
a rectangular warehouse:

  * Boundary check uses polygon containment (not just bounding-box).
  * Ceiling height is checked per row: a bay with height H can only go
    in a row at y where warehouse.ceiling_height_at(y) >= H.
  * When the preferred next-row-y has too low a ceiling for a bay, the
    packer searches the ceiling profile for the next valid zone.
  * All coordinates are in mm; the starting origin is the polygon's
    bounding-box min, not necessarily (0, 0).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from models import BayType, Warehouse, PlacedBay


# ---------------------------------------------------------------------------
# Core fit check
# ---------------------------------------------------------------------------

def _fits(x: float, y: float, w: float, d: float, height: float,
          warehouse: Warehouse, placed: List[PlacedBay]) -> bool:
    """Return True if a bay at (x,y,w,d,height) is valid."""
    if not warehouse.contains_rect(x, y, w, d):
        return False
    if height > warehouse.ceiling_height_at(y) + 1.0:
        return False
    for o in warehouse.obstacles:
        if o.overlaps(x, y, w, d):
            return False
    for p in placed:
        if not (x + w <= p.x + 1.0 or p.x2 <= x + 1.0
                or y + d <= p.y + 1.0 or p.y2 <= y + 1.0):
            return False
    return True


def _slide_right(x: float, y: float, w: float, d: float, height: float,
                 warehouse: Warehouse, placed: List[PlacedBay],
                 x_max: float, step: float = 50.0) -> Optional[float]:
    """Slide the bay right until it fits or runs off the warehouse."""
    cur = x
    limit = x_max - w + 1.0
    while cur <= limit:
        if _fits(cur, y, w, d, height, warehouse, placed):
            return cur
        # Jump past whatever is blocking
        blocker = cur + step
        for o in warehouse.obstacles:
            if o.overlaps(cur, y, w, d):
                blocker = max(blocker, o.x + o.width)
        for p in placed:
            if not (cur + w <= p.x + 1.0 or p.x2 <= cur + 1.0
                    or y + d <= p.y + 1.0 or p.y2 <= y + 1.0):
                blocker = max(blocker, p.x2)
        cur = max(cur + step, blocker + 1.0)
    return None


# ---------------------------------------------------------------------------
# Shelf packer
# ---------------------------------------------------------------------------

def shelf_pack(bays_to_place: List[BayType],
               warehouse: Warehouse,
               allow_rotation: bool = True) -> Tuple[List[PlacedBay], List[BayType]]:
    """Row-based First-Fit-Decreasing shelf packer with polygon + ceiling support.

    Rows are opened bottom-up from y_min. When a bay's height exceeds the
    ceiling at the default next-row position, the packer jumps forward to
    the nearest ceiling-zone boundary where the bay fits.
    """
    items = sorted(bays_to_place,
                   key=lambda b: (max(b.width, b.depth), b.width * b.depth),
                   reverse=True)

    x_min, y_min, x_max, y_max = warehouse.bounding_box

    placed: List[PlacedBay] = []
    unplaced: List[BayType] = []
    rows: List[list] = []   # [y0, row_depth, x_cursor]
    next_row_y = y_min

    for bay in items:
        orientations = [(bay.width, bay.depth, False)]
        if allow_rotation and abs(bay.width - bay.depth) > 1.0:
            orientations.append((bay.depth, bay.width, True))

        best_choice = None
        best_score = float("inf")

        # ---- try existing rows ----
        for ridx, (y0, h, cursor) in enumerate(rows):
            for w, d, rot in orientations:
                if d > h + 1.0:
                    continue
                if bay.height > warehouse.ceiling_height_at(y0) + 1.0:
                    continue
                x = cursor
                if not _fits(x, y0, w, d, bay.height, warehouse, placed):
                    x = _slide_right(x, y0, w, d, bay.height, warehouse,
                                     placed, x_max)
                    if x is None:
                        continue
                score = (h - d) + (x - cursor) * 0.001
                if score < best_score:
                    best_score = score
                    best_choice = (ridx, x, y0, w, d, rot)

        # ---- try a new row ----
        for w, d, rot in orientations:
            y0 = next_row_y
            # If the ceiling is too low here, jump to the next valid zone
            if bay.height > warehouse.ceiling_height_at(y0) + 1.0:
                y0 = warehouse.next_tall_ceiling_y(y0, bay.height)
                if y0 is None or y0 + d > y_max + 1.0:
                    continue
            if y0 + d > y_max + 1.0:
                continue
            x = x_min
            if not _fits(x, y0, w, d, bay.height, warehouse, placed):
                x = _slide_right(x, y0, w, d, bay.height, warehouse,
                                 placed, x_max)
                if x is None:
                    continue
            score = 0.5 + (x - x_min) * 0.001
            if score < best_score:
                best_score = score
                best_choice = (-1, x, y0, w, d, rot)

        if best_choice is None:
            unplaced.append(bay)
            continue

        ridx, x, y, w, d, rot = best_choice
        placed.append(PlacedBay(
            bay_id=bay.id, x=x, y=y,
            width=w, depth=d, height=bay.height, rotated=rot,
        ))
        if ridx == -1:
            rows.append([y, d, x + w])
            next_row_y = y + d + warehouse.aisle_width
        else:
            rows[ridx][1] = max(rows[ridx][1], d)
            rows[ridx][2] = x + w
            new_top = rows[ridx][0] + rows[ridx][1] + warehouse.aisle_width
            if new_top > next_row_y:
                next_row_y = new_top

    return placed, unplaced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack(bays_to_place: List[BayType],
         warehouse: Warehouse,
         allow_rotation: bool = True,
         method: str = "shelf") -> Tuple[List[PlacedBay], List[BayType]]:
    return shelf_pack(bays_to_place, warehouse, allow_rotation)


def expand_counts_to_list(counts: dict, catalogue: List[BayType]) -> List[BayType]:
    by_id = {b.id: b for b in catalogue}
    flat: List[BayType] = []
    for bay_id, n in counts.items():
        if bay_id not in by_id:
            raise KeyError(f"Unknown bay id {bay_id!r}")
        flat.extend([by_id[bay_id]] * int(round(n)))
    return flat


def evaluate_layout(placements: List[PlacedBay],
                    catalogue: List[BayType],
                    warehouse: Warehouse):
    by_id = {b.id: b for b in catalogue}
    cost = sum(by_id[p.bay_id].cost for p in placements)
    cap  = sum(by_id[p.bay_id].capacity for p in placements)
    ok = all(
        warehouse.contains_rect(p.x, p.y, p.width, p.depth)
        and p.height <= warehouse.ceiling_height_at(p.y) + 1.0
        for p in placements
    )
    return cost, cap, ok
