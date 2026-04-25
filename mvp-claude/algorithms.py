"""
Optimization algorithms — maximise total stored capacity.

Because the CSV test cases carry no explicit demand, warehouse.demand is
set to float('inf') by the loader, meaning "fill the warehouse as fully
as possible."  The greedy loops already handle this correctly:
  `while capacity < inf` runs until no more bays physically fit.

Simulated annealing uses a separate energy function when demand is
infinite: energy = -total_capacity + tiny * total_cost, so it drives
toward maximum capacity and breaks ties by minimum cost.

ILP selection is kept but always falls back to greedy when demand=inf
(the capacity >= inf constraint is infeasible — that's fine).
"""

from __future__ import annotations
import math
import random
import time
from typing import Dict, List, Optional, Tuple

from models import BayType, Warehouse, PlacedBay, Solution
from placement import pack, shelf_pack, expand_counts_to_list


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _solution_from_counts(counts: Dict[str, int],
                          catalogue: List[BayType],
                          warehouse: Warehouse,
                          method: str,
                          runtime_s: float,
                          allow_rotation: bool = True,
                          extra: Optional[dict] = None) -> Solution:
    flat = expand_counts_to_list(counts, catalogue)
    placed, unplaced = pack(flat, warehouse, allow_rotation)
    by_id = {b.id: b for b in catalogue}
    cost = sum(by_id[p.bay_id].cost for p in placed)
    cap  = sum(by_id[p.bay_id].capacity for p in placed)
    return Solution(
        method=method,
        placements=placed,
        total_cost=cost,
        total_capacity=cap,
        demand=warehouse.demand,
        warehouse_area=warehouse.area,
        runtime_s=runtime_s,
        extra={"requested_counts": counts,
               "unplaced": [b.id for b in unplaced],
               **(extra or {})},
    )


def _packing_efficiency_estimate(warehouse: Warehouse,
                                  n_rows_est: int = 4) -> float:
    obstacle_area = sum(o.width * o.depth for o in warehouse.obstacles)
    aisle_area = max(0, n_rows_est) * warehouse.aisle_width * warehouse.width
    usable = max(0.0, warehouse.area - obstacle_area - aisle_area)
    return usable / warehouse.area if warehouse.area > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Greedy heuristics
# ---------------------------------------------------------------------------

def greedy_cost_efficiency(catalogue: List[BayType],
                           warehouse: Warehouse,
                           allow_rotation: bool = True) -> Solution:
    t0 = time.time()
    sorted_bays = sorted(catalogue, key=lambda b: (b.cost_per_unit, -b.density))
    counts: Dict[str, int] = {b.id: 0 for b in catalogue}
    placed: List[PlacedBay] = []
    capacity = 0.0

    for bay in sorted_bays:
        while capacity < warehouse.demand:
            tentative = dict(counts); tentative[bay.id] += 1
            flat = expand_counts_to_list(tentative, catalogue)
            new_placed, unplaced = pack(flat, warehouse, allow_rotation)
            if unplaced:
                break
            placed = new_placed
            counts[bay.id] += 1
            capacity += bay.capacity
        if capacity >= warehouse.demand:
            break

    # repair: squeeze in any type that still fits
    if capacity < warehouse.demand:
        changed = True
        while capacity < warehouse.demand and changed:
            changed = False
            for bay in sorted(catalogue, key=lambda b: (b.footprint, b.cost_per_unit)):
                tentative = dict(counts); tentative[bay.id] += 1
                flat = expand_counts_to_list(tentative, catalogue)
                new_placed, unplaced = pack(flat, warehouse, allow_rotation)
                if not unplaced:
                    placed = new_placed
                    counts[bay.id] += 1
                    capacity += bay.capacity
                    changed = True
                    break

    by_id = {b.id: b for b in catalogue}
    cost = sum(by_id[p.bay_id].cost for p in placed)
    return Solution(
        method="greedy_cost_efficiency",
        placements=placed, total_cost=cost, total_capacity=capacity,
        demand=warehouse.demand, warehouse_area=warehouse.area,
        runtime_s=time.time() - t0,
        extra={"requested_counts": counts},
    )


def greedy_density_first(catalogue: List[BayType],
                         warehouse: Warehouse,
                         allow_rotation: bool = True) -> Solution:
    """Prefer bays with the highest capacity per mm² — best for filling space."""
    t0 = time.time()
    sorted_bays = sorted(catalogue, key=lambda b: (-b.density, b.cost_per_unit))
    counts: Dict[str, int] = {b.id: 0 for b in catalogue}
    capacity = 0.0
    placed: List[PlacedBay] = []

    for bay in sorted_bays:
        while capacity < warehouse.demand:
            tentative = dict(counts); tentative[bay.id] += 1
            flat = expand_counts_to_list(tentative, catalogue)
            new_placed, unplaced = pack(flat, warehouse, allow_rotation)
            if unplaced:
                break
            placed = new_placed
            counts[bay.id] += 1
            capacity += bay.capacity
        if capacity >= warehouse.demand:
            break

    # repair
    if capacity < warehouse.demand:
        changed = True
        while capacity < warehouse.demand and changed:
            changed = False
            for bay in sorted(catalogue, key=lambda b: (b.footprint, b.cost_per_unit)):
                tentative = dict(counts); tentative[bay.id] += 1
                flat = expand_counts_to_list(tentative, catalogue)
                new_placed, unplaced = pack(flat, warehouse, allow_rotation)
                if not unplaced:
                    placed = new_placed
                    counts[bay.id] += 1
                    capacity += bay.capacity
                    changed = True
                    break

    by_id = {b.id: b for b in catalogue}
    cost = sum(by_id[p.bay_id].cost for p in placed)
    return Solution(
        method="greedy_density_first",
        placements=placed, total_cost=cost, total_capacity=capacity,
        demand=warehouse.demand, warehouse_area=warehouse.area,
        runtime_s=time.time() - t0,
        extra={"requested_counts": counts},
    )


def greedy_multistart(catalogue: List[BayType],
                      warehouse: Warehouse,
                      allow_rotation: bool = True) -> Solution:
    """Try every bay as lead + three standard orderings, return the best."""
    t0 = time.time()
    orders = [
        sorted(catalogue, key=lambda b: (-b.density, b.cost_per_unit)),
        sorted(catalogue, key=lambda b: (b.cost_per_unit, -b.density)),
        sorted(catalogue, key=lambda b: (b.cost, -b.capacity)),
    ]
    for lead in catalogue:
        rest = sorted([b for b in catalogue if b.id != lead.id],
                      key=lambda b: b.cost_per_unit)
        orders.append([lead] + rest)

    best: Optional[Solution] = None
    for order in orders:
        sol = _run_ordered_greedy(order, catalogue, warehouse, allow_rotation)
        if best is None or sol.total_capacity > best.total_capacity:
            best = sol
        elif (sol.total_capacity == best.total_capacity
              and sol.total_cost < best.total_cost):
            best = sol

    best.method = "greedy_multistart"
    best.runtime_s = time.time() - t0
    return best


def _run_ordered_greedy(order: List[BayType], catalogue: List[BayType],
                        warehouse: Warehouse,
                        allow_rotation: bool) -> Solution:
    counts: Dict[str, int] = {b.id: 0 for b in catalogue}
    placed: List[PlacedBay] = []
    capacity = 0.0

    for bay in order:
        while capacity < warehouse.demand:
            tentative = dict(counts); tentative[bay.id] += 1
            flat = expand_counts_to_list(tentative, catalogue)
            new_placed, unplaced = pack(flat, warehouse, allow_rotation)
            if unplaced:
                break
            placed = new_placed
            counts[bay.id] += 1
            capacity += bay.capacity
        if capacity >= warehouse.demand:
            break

    if capacity < warehouse.demand:
        changed = True
        while capacity < warehouse.demand and changed:
            changed = False
            for bay in sorted(catalogue, key=lambda b: (b.footprint, b.cost_per_unit)):
                tentative = dict(counts); tentative[bay.id] += 1
                flat = expand_counts_to_list(tentative, catalogue)
                new_placed, unplaced = pack(flat, warehouse, allow_rotation)
                if not unplaced:
                    placed = new_placed
                    counts[bay.id] += 1
                    capacity += bay.capacity
                    changed = True
                    break

    by_id = {b.id: b for b in catalogue}
    cost = sum(by_id[p.bay_id].cost for p in placed)
    return Solution(
        method=f"greedy({order[0].id})",
        placements=placed, total_cost=cost, total_capacity=capacity,
        demand=warehouse.demand, warehouse_area=warehouse.area,
        runtime_s=0.0, extra={"requested_counts": counts},
    )


# ---------------------------------------------------------------------------
# 2. ILP selection (falls back to greedy when demand=inf)
# ---------------------------------------------------------------------------

def ilp_select_then_pack(catalogue: List[BayType],
                         warehouse: Warehouse,
                         allow_rotation: bool = True,
                         max_retries: int = 6,
                         verbose: bool = False) -> Solution:
    if math.isinf(warehouse.demand):
        sol = greedy_density_first(catalogue, warehouse, allow_rotation)
        sol.method = "ilp_maximize_fallback_greedy"
        return sol

    try:
        import pulp
    except ImportError:
        sol = greedy_cost_efficiency(catalogue, warehouse, allow_rotation)
        sol.method = "ilp_unavailable_fallback_greedy"
        return sol

    t0 = time.time()
    eta = _packing_efficiency_estimate(warehouse, n_rows_est=4)
    last_solution: Optional[Solution] = None

    for attempt in range(max_retries):
        if eta <= 0:
            break
        prob = pulp.LpProblem("WarehouseSelect", pulp.LpMinimize)
        upper = {b.id: int(math.ceil(warehouse.demand / b.capacity)) + 5
                 for b in catalogue if b.capacity > 0}
        n_vars = {b.id: pulp.LpVariable(f"n_{b.id}", lowBound=0,
                                         upBound=upper.get(b.id, 1000),
                                         cat="Integer")
                  for b in catalogue}
        prob += pulp.lpSum(n_vars[b.id] * b.cost for b in catalogue)
        prob += (pulp.lpSum(n_vars[b.id] * b.capacity for b in catalogue)
                 >= warehouse.demand)
        prob += (pulp.lpSum(n_vars[b.id] * b.footprint for b in catalogue)
                 <= eta * warehouse.area)

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        if pulp.LpStatus[status] != "Optimal":
            eta *= 0.9
            continue

        counts = {b.id: int(round(pulp.value(n_vars[b.id]))) for b in catalogue}
        sol = _solution_from_counts(counts, catalogue, warehouse,
                                     method="ilp_select_then_pack",
                                     runtime_s=time.time() - t0,
                                     allow_rotation=allow_rotation,
                                     extra={"eta": eta, "attempt": attempt})
        last_solution = sol
        if not sol.extra.get("unplaced") and sol.feasible:
            return sol
        eta *= 0.92

    if last_solution is None:
        last_solution = greedy_cost_efficiency(catalogue, warehouse, allow_rotation)
        last_solution.method = "ilp_failed_fallback_greedy"
    last_solution.runtime_s = time.time() - t0
    return last_solution


# ---------------------------------------------------------------------------
# 3. Simulated annealing
# ---------------------------------------------------------------------------

def simulated_annealing(catalogue: List[BayType],
                        warehouse: Warehouse,
                        initial: Optional[Solution] = None,
                        iterations: int = 3000,
                        T0: float = 1.0,
                        Tmin: float = 1e-3,
                        seed: int = 42,
                        allow_rotation: bool = True) -> Solution:
    """Perturb bay counts to refine the initial solution.

    Energy function switches depending on whether we are maximising
    capacity (demand=inf) or minimising cost to meet demand.
    """
    rng = random.Random(seed)
    t0 = time.time()
    maximize_mode = math.isinf(warehouse.demand)

    if initial is None:
        initial = greedy_density_first(catalogue, warehouse, allow_rotation)

    counts: Dict[str, int] = dict(
        initial.extra.get("requested_counts", {b.id: 0 for b in catalogue})
    )

    if maximize_mode:
        PEN_INFEAS = 1e9
        cap_scale = max(b.capacity for b in catalogue) or 1.0
        cost_scale = max(b.cost for b in catalogue) or 1.0

        def energy(c: Dict[str, int]) -> Tuple[float, Solution]:
            sol = _solution_from_counts(c, catalogue, warehouse,
                                         method="sa_eval", runtime_s=0.0,
                                         allow_rotation=allow_rotation)
            unplaced = len(sol.extra.get("unplaced", []))
            # maximise capacity, break ties by cost
            e = (-sol.total_capacity / cap_scale
                 + 0.0001 * sol.total_cost / cost_scale
                 + unplaced * PEN_INFEAS)
            return e, sol
    else:
        PEN_DEMAND = max(b.cost / b.capacity for b in catalogue) * 5
        PEN_INFEAS = max(b.cost for b in catalogue) * 3

        def energy(c: Dict[str, int]) -> Tuple[float, Solution]:
            sol = _solution_from_counts(c, catalogue, warehouse,
                                         method="sa_eval", runtime_s=0.0,
                                         allow_rotation=allow_rotation)
            unplaced = len(sol.extra.get("unplaced", []))
            shortage = max(0.0, warehouse.demand - sol.total_capacity)
            e = sol.total_cost + PEN_DEMAND * shortage + PEN_INFEAS * unplaced
            return e, sol

    cur_e, cur_sol = energy(counts)
    best_e, best_sol = cur_e, cur_sol
    best_counts = dict(counts)

    T = T0
    cooling = (Tmin / T0) ** (1.0 / max(1, iterations))
    bay_ids = [b.id for b in catalogue]

    for _ in range(iterations):
        new_counts = dict(counts)
        move = rng.random()
        if move < 0.4:
            new_counts[rng.choice(bay_ids)] += 1
        elif move < 0.75:
            positives = [b for b, n in new_counts.items() if n > 0]
            if not positives:
                continue
            new_counts[rng.choice(positives)] -= 1
        else:
            positives = [b for b, n in new_counts.items() if n > 0]
            if not positives:
                continue
            a = rng.choice(positives)
            b = rng.choice(bay_ids)
            if a == b:
                continue
            new_counts[a] -= 1
            new_counts[b] += 1

        new_e, new_sol = energy(new_counts)
        delta = new_e - cur_e
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            counts = new_counts
            cur_e, cur_sol = new_e, new_sol
            if new_e < best_e:
                best_e, best_sol = new_e, new_sol
                best_counts = dict(new_counts)
        T *= cooling

    best_sol.method = "simulated_annealing"
    best_sol.runtime_s = time.time() - t0
    best_sol.extra["initial_method"] = initial.method
    best_sol.extra["iterations"] = iterations
    best_sol.extra["requested_counts"] = best_counts
    return best_sol
