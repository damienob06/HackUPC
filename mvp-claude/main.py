"""
Warehouse Optimizer — main entry point.

Usage:
    python main.py                       # uses bundled instance.json
    python main.py path/to/instance.json # custom instance

Runs every algorithm on the same instance, prints a comparison table,
saves per-method PNGs and a side-by-side comparison grid, and writes
the best solution to best_solution.json.
"""

from __future__ import annotations
import os
import sys
import json
from typing import List

from models import load_instance, save_solution, Solution
from algorithms import (
    greedy_cost_efficiency,
    greedy_density_first,
    greedy_multistart,
    ilp_select_then_pack,
    ilp_with_aisle_model,
    simulated_annealing,
    combined_optimizer,
)
from visualizer import render, render_comparison
from validator import print_validation


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main(instance_path: str = "instance.json",
         out_dir: str = "output") -> None:
    os.makedirs(out_dir, exist_ok=True)

    banner("LOADING INSTANCE")
    warehouse, catalogue = load_instance(instance_path)
    print(f"Warehouse:  {warehouse.width} x {warehouse.depth} m  "
          f"(area={warehouse.area:.0f} m²)")
    print(f"Demand:     {warehouse.demand} units")
    print(f"Aisle:      {warehouse.aisle_width} m")
    print(f"Obstacles:  {len(warehouse.obstacles)}")
    print(f"Bay catalogue ({len(catalogue)}):")
    for b in catalogue:
        print(f"  - {b.id:10s}  {b.width}x{b.depth}m  "
              f"cap={b.capacity:>4.0f}  cost={b.cost:>5.0f}  "
              f"€/unit={b.cost_per_unit:.2f}  cap/m²={b.density:.1f}")

    # ----------------------------------------------------------- run all
    banner("RUNNING ALGORITHMS")
    runs = []
    runs.append(("greedy_cost_efficiency",
                 lambda: greedy_cost_efficiency(catalogue, warehouse)))
    runs.append(("greedy_density_first",
                 lambda: greedy_density_first(catalogue, warehouse)))
    runs.append(("greedy_multistart",
                 lambda: greedy_multistart(catalogue, warehouse)))
    runs.append(("ilp_select_then_pack",
                 lambda: ilp_select_then_pack(catalogue, warehouse,
                                              verbose=True)))
    runs.append(("ilp_with_aisle_model",
                 lambda: ilp_with_aisle_model(catalogue, warehouse,
                                              verbose=True)))

    solutions: List[Solution] = []
    for name, fn in runs:
        print(f"\n>> {name}")
        sol = fn()
        print("  " + sol.summary())
        solutions.append(sol)

    # SA initialised from the best feasible incumbent so far
    feasible_so_far = [s for s in solutions if s.feasible]
    seed_sol = (min(feasible_so_far, key=lambda s: s.total_cost)
                if feasible_so_far else solutions[0])
    print(f"\n>> simulated_annealing (seeded from {seed_sol.method})")
    sa_sol = simulated_annealing(catalogue, warehouse,
                                 initial=seed_sol, iterations=2500)
    print("  " + sa_sol.summary())
    solutions.append(sa_sol)

    # ---------------------------------------------------------- compare
    banner("VALIDATION")
    for s in solutions:
        print_validation(s, warehouse, catalogue)

    banner("COMPARISON")
    print(f"{'method':<30s} {'feas':<6s} {'cost':>12s} "
          f"{'cap':>10s} {'bays':>6s} {'util':>7s} {'t (s)':>8s}")
    print("-" * 84)
    feasible = [s for s in solutions if s.feasible]
    best = (min(feasible, key=lambda s: s.total_cost)
            if feasible else min(solutions, key=lambda s: s.total_cost))
    for s in solutions:
        marker = "  <-- BEST" if s is best else ""
        print(f"{s.method:<30s} {'YES' if s.feasible else 'NO ':<6s} "
              f"{s.total_cost:>12,.2f} "
              f"{s.total_capacity:>10.1f} {len(s.placements):>6d} "
              f"{s.utilisation*100:>6.1f}% {s.runtime_s:>8.2f}{marker}")

    # --------------------------------------------------------- visualise
    banner("RENDERING")
    for s in solutions:
        out = os.path.join(out_dir, f"{s.method}.png")
        render(s, warehouse, catalogue, out)
        print(f"  wrote {out}")

    grid_path = os.path.join(out_dir, "comparison.png")
    render_comparison(solutions, warehouse, catalogue, grid_path)
    print(f"  wrote {grid_path}")

    # ------------------------------------------------------- save best
    best_path = os.path.join(out_dir, "best_solution.json")
    save_solution(best, best_path)
    print(f"  wrote {best_path}  (method={best.method}, "
          f"cost={best.total_cost:,.2f})")


if __name__ == "__main__":
    instance = sys.argv[1] if len(sys.argv) > 1 else "instance.json"
    main(instance)
