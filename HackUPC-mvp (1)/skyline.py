"""
Skyline 2D packing engine.

Replaces the simple shelf packer with a proper skyline (bottom-left-fill)
algorithm. The skyline tracks the exact upper contour of occupied space
as a list of (x_start, width, y_top) segments. New bays are inserted at
the lowest valid position, obstacles are pre-raised in the skyline, and
aisles are enforced by adding aisle_width padding above each bay.

Key improvements over shelf packing:
  - Mixed-size bays pack tightly (no wasted row height)
  - Obstacles are handled natively (raised skyline segments)
  - Better utilization on irregular layouts
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from models import BayType, Warehouse, PlacedBay, Obstacle


# -----------------------------------------------------------------------
# Skyline data structure helpers
# -----------------------------------------------------------------------

def _skyline_height_at(skyline: List[list], x: float, w: float) -> Optional[float]:
    """Return max y_top across skyline segments spanning [x, x+w].
    Returns None if there's a gap (shouldn't happen with proper init)."""
    max_top = 0.0
    for sx, sw, sy in skyline:
        sx2 = sx + sw
        if sx2 <= x + 1e-9:
            continue
        if sx >= x + w - 1e-9:
            break
        max_top = max(max_top, sy)
    return max_top


def _update_skyline(skyline: List[list], x: float, w: float, new_top: float) -> List[list]:
    """Raise the skyline to new_top over the range [x, x+w]. Splits
    existing segments as needed, merges adjacent segments at same height."""
    out = []
    x2 = x + w
    for sx, sw, sy in skyline:
        sx2 = sx + sw
        if sx2 <= x + 1e-9 or sx >= x2 - 1e-9:
            # no overlap — keep as-is
            out.append([sx, sw, sy])
            continue
        # partial or full overlap
        if sx < x - 1e-9:
            out.append([sx, x - sx, sy])
        # the overlapping part gets raised
        clip_l = max(sx, x)
        clip_r = min(sx2, x2)
        out.append([clip_l, clip_r - clip_l, max(sy, new_top)])
        if sx2 > x2 + 1e-9:
            out.append([x2, sx2 - x2, sy])

    # merge adjacent segments at same height
    merged = [out[0]]
    for seg in out[1:]:
        prev = merged[-1]
        if abs(prev[0] + prev[1] - seg[0]) < 1e-9 and abs(prev[2] - seg[2]) < 1e-9:
            prev[1] += seg[1]
        else:
            merged.append(seg)
    return merged


def _init_skyline(warehouse: Warehouse) -> List[list]:
    """Flat skyline at y=0 spanning warehouse width, with obstacles raised."""
    sky = [[0.0, warehouse.width, 0.0]]
    for o in warehouse.obstacles:
        sky = _update_skyline(sky, o.x, o.width, o.y + o.depth)
    return sky


# -----------------------------------------------------------------------
# Collision check (obstacles + placed bays)
# -----------------------------------------------------------------------

def _collides(x: float, y: float, w: float, d: float,
              warehouse: Warehouse, placed: List[PlacedBay]) -> bool:
    """True if position (x,y,w,d) collides with bounds, obstacles, or existing bays."""
    if x < -1e-9 or y < -1e-9:
        return True
    if x + w > warehouse.width + 1e-9 or y + d > warehouse.depth + 1e-9:
        return True
    for o in warehouse.obstacles:
        if o.overlaps(x, y, w, d):
            return True
    for p in placed:
        if not (x + w <= p.x + 1e-9 or p.x2 <= x + 1e-9
                or y + d <= p.y + 1e-9 or p.y2 <= y + 1e-9):
            return True
    return False


# -----------------------------------------------------------------------
# Main packing function
# -----------------------------------------------------------------------

def skyline_pack(bays_to_place: List[BayType],
                 warehouse: Warehouse,
                 allow_rotation: bool = True
                 ) -> Tuple[List[PlacedBay], List[BayType]]:
    """Bottom-left-fill skyline packing.

    Aisle strategy: after placing a bay at (x, y) with depth d, the
    skyline above [x, x+w] is raised to y + d + aisle_width. This means
    the NEXT bay placed above this one will start at least aisle_width
    higher. Bays placed *beside* this one (different x range) are
    unaffected — they sit at whatever height the skyline says.

    This naturally produces row-like layouts with aisles between rows,
    but adapts to obstacles and mixed bay sizes far better than fixed
    shelf packing.
    """
    # Sort: biggest area first (FFD), break ties by max dimension
    items = sorted(bays_to_place,
                   key=lambda b: (b.width * b.depth, max(b.width, b.depth)),
                   reverse=True)

    placed: List[PlacedBay] = []
    unplaced: List[BayType] = []
    skyline = _init_skyline(warehouse)

    for bay in items:
        orientations = [(bay.width, bay.depth, False)]
        if allow_rotation and abs(bay.width - bay.depth) > 1e-9:
            orientations.append((bay.depth, bay.width, True))

        best = None  # (score_tuple, x, y, w, d, rotated)

        # Try every skyline segment start as a candidate x
        xs = sorted({seg[0] for seg in skyline})

        for x in xs:
            for w, d, rot in orientations:
                if x + w > warehouse.width + 1e-9:
                    continue
                y = _skyline_height_at(skyline, x, w)
                if y is None or y + d > warehouse.depth + 1e-9:
                    continue
                if _collides(x, y, w, d, warehouse, placed):
                    continue

                # Score: (lowest y, least wasted area below bay, leftmost x)
                waste = 0.0
                for sx, sw, sy in skyline:
                    if sx + sw <= x + 1e-9:
                        continue
                    if sx >= x + w - 1e-9:
                        break
                    ol = max(sx, x)
                    or_ = min(sx + sw, x + w)
                    waste += (or_ - ol) * max(0.0, y - sy)

                score = (y, waste, x)
                if best is None or score < best[0]:
                    best = (score, x, y, w, d, rot)

        if best is None:
            unplaced.append(bay)
            continue

        _, x, y, w, d, rot = best
        placed.append(PlacedBay(bay_id=bay.id, x=x, y=y,
                                 width=w, depth=d, rotated=rot))

        # Raise skyline: bay top + aisle, capped at warehouse depth
        new_top = min(y + d + warehouse.aisle_width, warehouse.depth)
        skyline = _update_skyline(skyline, x, w, new_top)

    return placed, unplaced
