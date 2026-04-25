"""
Optimization algorithms for the warehouse-bay selection problem.

We decompose the problem into:

    (S) SELECTION  — how many bays of each type to install
    (P) PLACEMENT  — where to put them in the warehouse

The selection problem is a multi-item knapsack-style ILP: minimise
total cost subject to a capacity-coverage constraint and an
area-fit constraint. The placement problem is solved separately by the
shelf-packing engine in `placement.py`. If a selection turns out not to
fit physically, we tighten the area limit and re-solve.

Implemented methods:

    1. greedy_cost_efficiency   — sort by cost/capacity, pack greedily
    2. greedy_density_first     — sort by capacity per m², then cost
    3. ilp_select_then_pack     — exact ILP for selection, retry on bad fit
    4. simulated_annealing      — perturbs counts to refine ILP solution
    5. ilp_with_aisle_model     — ILP that subtracts aisle area from the
                                  area constraint based on row count

Each function returns a `Solution` dataclass with placements and
runtime, ready for the visualizer.
"""

from __future__ import annotations
import math
import random
import time
from typing import Dict, List, Tuple, Optional

from models import BayType, Warehouse, PlacedBay, Solution
from placement import shelf_pack, expand_counts_to_list, evaluate_layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solution_from_counts(counts: Dict[str, int],
                          catalogue: List[BayType],
                          warehouse: Warehouse,
                          method: str,
                          runtime_s: float,
                          allow_rotation: bool = True,
                          extra: Optional[dict] = None) -> Solution:
    """Materialise a (counts -> Solution) by physically packing them."""
    flat = expand_counts_to_list(counts, catalogue)
    placed, unplaced = shelf_pack(flat, warehouse, allow_rotation=allow_rotation)
    by_id = {b.id: b for b in catalogue}
    cost = sum(by_id[p.bay_id].cost for p in placed)
    cap  = sum(by_id[p.bay_id].capacity for p in placed)
    sol = Solution(
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
    return sol


def _packing_efficiency_estimate(warehouse: Warehouse, n_rows_est: int = 4) -> float:
    """Approximate share of the warehouse area that's actually usable
    for bays after subtracting aisle corridors and obstacles. Used as a
    safety factor in the area constraint of the ILP."""
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
    """Pick the bay with the best cost-per-capacity, install as many as
    will fit, then move to the next type if demand still unmet.

    Includes a 'repair' phase: if standard ordering can't fill demand,
    try every bay type one-by-one to see what still fits — this rescues
    cases where the largest type filled the warehouse with no room for
    one extra big bay, but a small one still fits."""
    t0 = time.time()
    sorted_bays = sorted(catalogue, key=lambda b: (b.cost_per_unit, -b.density))
    counts: Dict[str, int] = {b.id: 0 for b in catalogue}

    placed: List[PlacedBay] = []
    capacity = 0.0

    for bay in sorted_bays:
        while capacity < warehouse.demand:
            tentative_counts = dict(counts)
            tentative_counts[bay.id] += 1
            flat = expand_counts_to_list(tentative_counts, catalogue)
            new_placed, unplaced = shelf_pack(flat, warehouse, allow_rotation=allow_rotation)
            if unplaced:
                break  # this bay type can't be added any more
            placed = new_placed
            counts[bay.id] += 1
            capacity += bay.capacity

        if capacity >= warehouse.demand:
            break

    # ---- repair phase: try ANY bay type if still short ----
    if capacity < warehouse.demand:
        repair_progress = True
        while capacity < warehouse.demand and repair_progress:
            repair_progress = False
            # try smaller bays first now (better chance of fitting)
            repair_order = sorted(catalogue,
                                   key=lambda b: (b.footprint, b.cost_per_unit))
            for bay in repair_order:
                tentative = dict(counts); tentative[bay.id] += 1
                flat = expand_counts_to_list(tentative, catalogue)
                new_placed, unplaced = shelf_pack(flat, warehouse,
                                                   allow_rotation=allow_rotation)
                if not unplaced:
                    placed = new_placed
                    counts[bay.id] += 1
                    capacity += bay.capacity
                    repair_progress = True
                    break

    cost = sum(next(b for b in catalogue if b.id == p.bay_id).cost for p in placed)
    sol = Solution(
        method="greedy_cost_efficiency",
        placements=placed, total_cost=cost, total_capacity=capacity,
        demand=warehouse.demand, warehouse_area=warehouse.area,
        runtime_s=time.time() - t0,
        extra={"requested_counts": counts},
    )
    return sol


def greedy_density_first(catalogue: List[BayType],
                         warehouse: Warehouse,
                         allow_rotation: bool = True) -> Solution:
    """Same idea but prefer space-efficient bays first (capacity / m²),
    then break ties by cost. Useful when the warehouse is small relative
    to demand. Uses the same repair phase as greedy_cost_efficiency."""
    t0 = time.time()
    sorted_bays = sorted(catalogue, key=lambda b: (-b.density, b.cost_per_unit))
    counts: Dict[str, int] = {b.id: 0 for b in catalogue}
    capacity = 0.0
    placed: List[PlacedBay] = []

    for bay in sorted_bays:
        while capacity < warehouse.demand:
            tentative = dict(counts); tentative[bay.id] += 1
            flat = expand_counts_to_list(tentative, catalogue)
            new_placed, unplaced = shelf_pack(flat, warehouse, allow_rotation=allow_rotation)
            if unplaced:
                break
            placed = new_placed
            counts[bay.id] += 1
            capacity += bay.capacity
        if capacity >= warehouse.demand:
            break

    # repair: try any bay type if still short
    if capacity < warehouse.demand:
        repair_progress = True
        while capacity < warehouse.demand and repair_progress:
            repair_progress = False
            repair_order = sorted(catalogue,
                                   key=lambda b: (b.footprint, b.cost_per_unit))
            for bay in repair_order:
                tentative = dict(counts); tentative[bay.id] += 1
                flat = expand_counts_to_list(tentative, catalogue)
                new_placed, unplaced = shelf_pack(flat, warehouse,
                                                   allow_rotation=allow_rotation)
                if not unplaced:
                    placed = new_placed
                    counts[bay.id] += 1
                    capacity += bay.capacity
                    repair_progress = True
                    break

    cost = sum(next(b for b in catalogue if b.id == p.bay_id).cost for p in placed)
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
    """Multi-start greedy: try every bay type (and every cost/density
    ordering) as the seed, run a greedy fill, return the best feasible
    solution. Much more robust than single-start greedy because it
    escapes the 'committed to a bad starting type' trap."""
    t0 = time.time()
    candidate_orders = []

    # standard orderings
    candidate_orders.append(sorted(catalogue, key=lambda b: (b.cost_per_unit, -b.density)))
    candidate_orders.append(sorted(catalogue, key=lambda b: (-b.density, b.cost_per_unit)))
    candidate_orders.append(sorted(catalogue, key=lambda b: (b.cost, -b.capacity)))

    # plus: each bay as the lead, rest by cost-eff
    for lead in catalogue:
        rest = sorted([b for b in catalogue if b.id != lead.id],
                       key=lambda b: b.cost_per_unit)
        candidate_orders.append([lead] + rest)

    best: Optional[Solution] = None
    for order in candidate_orders:
        sol = _run_ordered_greedy(order, catalogue, warehouse, allow_rotation)
        if sol.feasible and (best is None
                              or not best.feasible
                              or sol.total_cost < best.total_cost):
            best = sol
        elif best is None:
            best = sol  # at least keep something
    if best is None:
        best = _run_ordered_greedy(candidate_orders[0], catalogue,
                                    warehouse, allow_rotation)
    best.method = "greedy_multistart"
    best.runtime_s = time.time() - t0
    return best


def _run_ordered_greedy(order: List[BayType], catalogue: List[BayType],
                        warehouse: Warehouse,
                        allow_rotation: bool) -> Solution:
    """Run the greedy with a specific bay-type ordering, with a repair
    phase. Returns whatever it builds (feasible or not)."""
    counts: Dict[str, int] = {b.id: 0 for b in catalogue}
    placed: List[PlacedBay] = []
    capacity = 0.0

    for bay in order:
        while capacity < warehouse.demand:
            tentative = dict(counts); tentative[bay.id] += 1
            flat = expand_counts_to_list(tentative, catalogue)
            new_placed, unplaced = shelf_pack(flat, warehouse,
                                               allow_rotation=allow_rotation)
            if unplaced:
                break
            placed = new_placed
            counts[bay.id] += 1
            capacity += bay.capacity
        if capacity >= warehouse.demand:
            break

    # repair
    if capacity < warehouse.demand:
        repair_progress = True
        while capacity < warehouse.demand and repair_progress:
            repair_progress = False
            repair_order = sorted(catalogue,
                                   key=lambda b: (b.footprint, b.cost_per_unit))
            for bay in repair_order:
                tentative = dict(counts); tentative[bay.id] += 1
                flat = expand_counts_to_list(tentative, catalogue)
                new_placed, unplaced = shelf_pack(flat, warehouse,
                                                   allow_rotation=allow_rotation)
                if not unplaced:
                    placed = new_placed
                    counts[bay.id] += 1
                    capacity += bay.capacity
                    repair_progress = True
                    break

    cost = sum(next(b for b in catalogue if b.id == p.bay_id).cost for p in placed)
    return Solution(
        method=f"greedy_ordered({order[0].id})",
        placements=placed, total_cost=cost, total_capacity=capacity,
        demand=warehouse.demand, warehouse_area=warehouse.area,
        runtime_s=0.0,
        extra={"requested_counts": counts},
    )


# ---------------------------------------------------------------------------
# 2. Integer Linear Programming (selection)
# ---------------------------------------------------------------------------

def ilp_select_then_pack(catalogue: List[BayType],
                         warehouse: Warehouse,
                         allow_rotation: bool = True,
                         max_retries: int = 6,
                         verbose: bool = False) -> Solution:
    """Exact ILP for the SELECTION subproblem.

    minimise   sum_i n_i * cost_i
    subject to sum_i n_i * capacity_i  >= demand
               sum_i n_i * width_i * depth_i  <= eta * warehouse_area
               n_i in Z>=0

    `eta` is the fraction of the warehouse area we expect to actually
    cover with bays once aisles & obstacles are accounted for. We start
    with an estimate and tighten it if the solution doesn't physically
    fit.

    After solving, we feed the (n_i) into the shelf packer. If anything
    falls off the warehouse, we lower eta and re-solve.

    Falls back to greedy if PuLP is unavailable.
    """
    try:
        import pulp
    except ImportError:
        sol = greedy_cost_efficiency(catalogue, warehouse, allow_rotation)
        sol.method = "ilp_unavailable_fallback_greedy"
        return sol

    t0 = time.time()
    eta = _packing_efficiency_estimate(warehouse, n_rows_est=4)
    if eta <= 0:
        # warehouse too cluttered — just try greedy
        sol = greedy_cost_efficiency(catalogue, warehouse, allow_rotation)
        sol.method = "ilp_eta_zero_fallback_greedy"
        return sol

    upper_bound = {b.id: int(math.ceil(warehouse.demand / b.capacity)) + 5
                   for b in catalogue if b.capacity > 0}

    last_solution: Optional[Solution] = None
    for attempt in range(max_retries):
        prob = pulp.LpProblem("WarehouseSelect", pulp.LpMinimize)
        n_vars = {
            b.id: pulp.LpVariable(f"n_{b.id}", lowBound=0,
                                   upBound=upper_bound.get(b.id, 1000),
                                   cat="Integer")
            for b in catalogue
        }
        # objective
        prob += pulp.lpSum(n_vars[b.id] * b.cost for b in catalogue)
        # capacity
        prob += (pulp.lpSum(n_vars[b.id] * b.capacity for b in catalogue)
                 >= warehouse.demand)
        # area
        prob += (pulp.lpSum(n_vars[b.id] * b.footprint for b in catalogue)
                 <= eta * warehouse.area)

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
        status = prob.solve(solver)

        if pulp.LpStatus[status] != "Optimal":
            if verbose:
                print(f"  [ILP attempt {attempt}] eta={eta:.3f} -> {pulp.LpStatus[status]}")
            eta *= 0.9
            continue

        counts = {b.id: int(round(pulp.value(n_vars[b.id]))) for b in catalogue}
        sol = _solution_from_counts(
            counts, catalogue, warehouse,
            method=f"ilp_select_then_pack",
            runtime_s=time.time() - t0,
            allow_rotation=allow_rotation,
            extra={"eta": eta, "attempt": attempt},
        )
        last_solution = sol

        unplaced_ids = sol.extra.get("unplaced", [])
        if not unplaced_ids and sol.feasible:
            if verbose:
                print(f"  [ILP attempt {attempt}] eta={eta:.3f} -> packed OK")
            sol.runtime_s = time.time() - t0
            return sol

        if verbose:
            print(f"  [ILP attempt {attempt}] eta={eta:.3f} -> "
                  f"{len(unplaced_ids)} bays unplaced, tightening")
        eta *= 0.92  # tighten and retry

    if last_solution is None:
        last_solution = greedy_cost_efficiency(catalogue, warehouse, allow_rotation)
        last_solution.method = "ilp_failed_fallback_greedy"
    last_solution.runtime_s = time.time() - t0
    return last_solution


# ---------------------------------------------------------------------------
# 3. Simulated annealing  (refinement of an initial solution)
# ---------------------------------------------------------------------------

def simulated_annealing(catalogue: List[BayType],
                        warehouse: Warehouse,
                        initial: Optional[Solution] = None,
                        iterations: int = 4000,
                        T0: float = 1.0,
                        Tmin: float = 1e-3,
                        seed: int = 42,
                        allow_rotation: bool = True) -> Solution:
    """Refine a starting solution by perturbing the bay counts.

    Moves: +1 to a random bay type, -1 to a random bay type, swap one
    bay-type-A for bay-type-B at equal multiplicity.

    Energy = total_cost  +  PEN_DEMAND * max(0, demand - capacity)
                          +  PEN_INFEAS * unplaced_count

    Standard geometric cooling schedule. Small instances converge in <1s.
    """
    rng = random.Random(seed)
    t0 = time.time()

    if initial is None:
        initial = greedy_cost_efficiency(catalogue, warehouse, allow_rotation)

    counts: Dict[str, int] = dict(initial.extra.get("requested_counts",
                                                    {b.id: 0 for b in catalogue}))
    by_id = {b.id: b for b in catalogue}

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

    for it in range(iterations):
        new_counts = dict(counts)
        move = rng.random()
        if move < 0.4:
            # increment
            bid = rng.choice(bay_ids)
            new_counts[bid] += 1
        elif move < 0.75:
            # decrement (only if positive)
            positives = [b for b, n in new_counts.items() if n > 0]
            if not positives:
                continue
            bid = rng.choice(positives)
            new_counts[bid] -= 1
        else:
            # swap one of A for one of B
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
            if new_sol.feasible and (new_sol.total_cost < best_sol.total_cost
                                      or not best_sol.feasible):
                best_sol = new_sol
                best_e = new_e
                best_counts = dict(new_counts)
        T *= cooling

    best_sol.method = "simulated_annealing"
    best_sol.runtime_s = time.time() - t0
    best_sol.extra["initial_method"] = initial.method
    best_sol.extra["iterations"] = iterations
    best_sol.extra["requested_counts"] = best_counts
    return best_sol


# ---------------------------------------------------------------------------
# 4. ILP with aisle-aware area model
# ---------------------------------------------------------------------------

def ilp_with_aisle_model(catalogue: List[BayType],
                         warehouse: Warehouse,
                         allow_rotation: bool = True,
                         verbose: bool = False) -> Solution:
    """Variant of the ILP that explicitly models aisle area as a function
    of estimated row count.

    Estimated rows = ceil(total_depth_used / avg_bay_depth)
    Aisle area = (rows - 1) * aisle_width * warehouse.width

    Implementation note: the row count depends on the solution itself,
    so we do an iterative outer loop where we fix the row estimate,
    solve the ILP, recompute rows from the actual packing, and repeat.
    """
    try:
        import pulp
    except ImportError:
        sol = greedy_cost_efficiency(catalogue, warehouse, allow_rotation)
        sol.method = "ilp_unavailable_fallback_greedy"
        return sol

    t0 = time.time()
    avg_depth = sum(b.depth for b in catalogue) / len(catalogue)
    rows_est = max(1, int(round(warehouse.depth / (avg_depth + warehouse.aisle_width))))
    last_sol: Optional[Solution] = None

    for outer in range(5):
        eta = _packing_efficiency_estimate(warehouse, n_rows_est=rows_est)
        prob = pulp.LpProblem("WarehouseAisleAware", pulp.LpMinimize)
        n_vars = {
            b.id: pulp.LpVariable(f"n_{b.id}", lowBound=0, cat="Integer")
            for b in catalogue
        }
        prob += pulp.lpSum(n_vars[b.id] * b.cost for b in catalogue)
        prob += (pulp.lpSum(n_vars[b.id] * b.capacity for b in catalogue)
                 >= warehouse.demand)
        prob += (pulp.lpSum(n_vars[b.id] * b.footprint for b in catalogue)
                 <= eta * warehouse.area)

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            break
        counts = {b.id: int(round(pulp.value(n_vars[b.id]))) for b in catalogue}
        sol = _solution_from_counts(counts, catalogue, warehouse,
                                     method="ilp_with_aisle_model",
                                     runtime_s=time.time() - t0,
                                     allow_rotation=allow_rotation,
                                     extra={"eta": eta, "rows_est": rows_est})
        last_sol = sol

        # Recompute row count from the actual placement
        if sol.placements:
            ys = sorted({round(p.y, 2) for p in sol.placements})
            new_rows_est = max(1, len(ys))
        else:
            new_rows_est = rows_est

        unplaced = len(sol.extra.get("unplaced", []))
        if unplaced == 0 and new_rows_est == rows_est:
            if verbose:
                print(f"  [aisle ILP outer={outer}] converged: "
                      f"rows={rows_est}, eta={eta:.3f}")
            break
        rows_est = new_rows_est + (1 if unplaced > 0 else 0)
        if verbose:
            print(f"  [aisle ILP outer={outer}] rows={rows_est}, "
                  f"eta={eta:.3f}, unplaced={unplaced}")

    if last_sol is None:
        last_sol = greedy_cost_efficiency(catalogue, warehouse, allow_rotation)
        last_sol.method = "ilp_failed_fallback_greedy"
    last_sol.runtime_s = time.time() - t0
    return last_sol
