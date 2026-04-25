"""
Stress test: generate random instances and compare algorithm performance.

Useful to demonstrate that the ILP / SA reliably beat or match greedy
across a wide range of warehouse sizes and demand profiles, and to
spot edge cases where heuristics fail.

Run:
    python stress_test.py               # 10 random instances
    python stress_test.py 50            # 50 random instances
"""

from __future__ import annotations
import random
import sys
import time
from typing import List

from models import Warehouse, BayType, Obstacle
from algorithms import (
    greedy_cost_efficiency, greedy_density_first, greedy_multistart,
    ilp_select_then_pack, simulated_annealing,
)
from validator import validate


def random_instance(seed: int) -> tuple:
    rng = random.Random(seed)
    w = rng.randint(20, 60)
    d = rng.randint(15, 40)
    aisle = rng.choice([2.5, 2.8, 3.0, 3.2])

    # generate 4-7 bay types
    n_bays = rng.randint(4, 7)
    catalogue: List[BayType] = []
    for i in range(n_bays):
        bw = round(rng.uniform(1.8, 4.0), 1)
        bd = round(rng.uniform(1.0, 1.6), 1)
        cap = rng.randint(40, 350)
        # cost loosely correlated with capacity, with noise
        cost = int(cap * rng.uniform(7, 14))
        catalogue.append(BayType(id=f"B{i+1}", width=bw, depth=bd,
                                  capacity=cap, cost=cost))

    # demand sized so a fraction of warehouse fills up
    avg_density = sum(b.capacity / (b.width * b.depth) for b in catalogue) / n_bays
    target_fill = rng.uniform(0.15, 0.45)
    demand = int(avg_density * w * d * target_fill)

    # 0-3 obstacles
    obstacles = []
    for _ in range(rng.randint(0, 3)):
        ow = rng.uniform(1.5, 3.0)
        od = rng.uniform(1.5, 3.0)
        ox = rng.uniform(0, w - ow)
        oy = rng.uniform(0, d - od)
        obstacles.append(Obstacle(x=ox, y=oy, width=ow, depth=od))

    warehouse = Warehouse(width=w, depth=d, demand=demand,
                          aisle_width=aisle, obstacles=obstacles)
    return warehouse, catalogue


def main(n: int = 10) -> None:
    print(f"Running stress test on {n} random instances...\n")
    print(f"{'#':>3} {'W×D':>9} {'demand':>7} | "
          f"{'gr_cost':>10} {'gr_dens':>10} {'gr_multi':>10} {'ilp':>10} {'sa':>10} | "
          f"{'best':>10} {'gap_grcost':>11}")
    print("-" * 115)

    wins_ilp = wins_sa = wins_grcost = wins_grdens = wins_grmulti = 0
    sum_gap_grcost = sum_gap_grdens = sum_gap_grmulti = sum_gap_sa = 0.0
    valid_count = 0

    for seed in range(n):
        wh, cat = random_instance(seed)
        results = {}
        for name, fn in [
            ("gr_cost",  lambda: greedy_cost_efficiency(cat, wh)),
            ("gr_dens",  lambda: greedy_density_first(cat, wh)),
            ("gr_multi", lambda: greedy_multistart(cat, wh)),
            ("ilp",      lambda: ilp_select_then_pack(cat, wh)),
        ]:
            try:
                results[name] = fn()
            except Exception as e:
                results[name] = None
        # SA seeded from best feasible incumbent
        feasibles = [s for s in results.values() if s and s.feasible]
        seed_sol = (min(feasibles, key=lambda s: s.total_cost)
                    if feasibles else list(results.values())[0])
        try:
            results["sa"] = simulated_annealing(cat, wh, initial=seed_sol,
                                                 iterations=800, seed=seed)
        except Exception:
            results["sa"] = None

        # Validate every solution
        all_valid = True
        for name, sol in results.items():
            if sol is None or not sol.feasible:
                continue
            ok, _ = validate(sol, wh, cat)
            if not ok:
                all_valid = False
        if all_valid:
            valid_count += 1

        feasibles = [(n, s) for n, s in results.items() if s and s.feasible]
        if not feasibles:
            print(f"{seed:>3} {wh.width}×{wh.depth:<5} {wh.demand:>7} | "
                  f"NO FEASIBLE SOLUTION")
            continue

        best_name, best_sol = min(feasibles, key=lambda p: p[1].total_cost)
        if best_name == "ilp": wins_ilp += 1
        elif best_name == "sa": wins_sa += 1
        elif best_name == "gr_cost": wins_grcost += 1
        elif best_name == "gr_dens": wins_grdens += 1
        elif best_name == "gr_multi": wins_grmulti += 1

        def show(s):
            return f"{s.total_cost:>10,.0f}" if s and s.feasible else "          —"

        gap = (results["gr_cost"].total_cost - best_sol.total_cost) / best_sol.total_cost * 100 \
              if results["gr_cost"] and results["gr_cost"].feasible else 0
        sum_gap_grcost += gap
        if results["gr_dens"] and results["gr_dens"].feasible:
            sum_gap_grdens += (results["gr_dens"].total_cost - best_sol.total_cost) / best_sol.total_cost * 100
        if results["gr_multi"] and results["gr_multi"].feasible:
            sum_gap_grmulti += (results["gr_multi"].total_cost - best_sol.total_cost) / best_sol.total_cost * 100
        if results["sa"] and results["sa"].feasible:
            sum_gap_sa += (results["sa"].total_cost - best_sol.total_cost) / best_sol.total_cost * 100

        print(f"{seed:>3} {wh.width}×{wh.depth:<5} {wh.demand:>7} | "
              f"{show(results['gr_cost'])} {show(results['gr_dens'])} {show(results['gr_multi'])} "
              f"{show(results['ilp'])} {show(results['sa'])} | "
              f"{best_sol.total_cost:>10,.0f} ({best_name:<7s}) "
              f"{gap:>9.1f}%")

    print("-" * 115)
    print(f"\nWins:  ilp={wins_ilp}  sa={wins_sa}  gr_multi={wins_grmulti}  "
          f"gr_cost={wins_grcost}  gr_dens={wins_grdens}")
    print(f"Mean cost gap vs best:")
    print(f"  greedy_cost_efficiency: {sum_gap_grcost/n:.2f}%")
    print(f"  greedy_density_first:   {sum_gap_grdens/n:.2f}%")
    print(f"  greedy_multistart:      {sum_gap_grmulti/n:.2f}%")
    print(f"  simulated_annealing:    {sum_gap_sa/n:.2f}%")
    print(f"\nAll-method validation passed on {valid_count}/{n} instances")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n)
