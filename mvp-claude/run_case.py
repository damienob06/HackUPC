"""
Solve one test case and write UI-ready outputs.

Usage:
    python run_case.py <case_dir> [output_dir]

Examples:
    python run_case.py C:/tmp/testcases/Case0
    python run_case.py C:/tmp/testcases/Case0 output/Case0
    python run_case.py C:/tmp/testcases/Case1 output/Case1

Outputs (written to output_dir):
    <CaseName>_solution.json   — UI-ready JSON with full geometry + stats
    <CaseName>_layout.png      — top-down warehouse plan
"""

from __future__ import annotations
import os
import sys
import json

from csv_loader import load_case
from algorithms import greedy_density_first, greedy_multistart, simulated_annealing
from validator import validate, print_validation
from visualizer import render
from output_formatter import format_solution_for_ui


def main(case_dir: str, out_dir: str = "output") -> None:
    case_name = os.path.basename(os.path.abspath(case_dir))
    os.makedirs(out_dir, exist_ok=True)

    # ---- load ----
    print(f"\n{'='*60}")
    print(f"  {case_name}")
    print(f"{'='*60}")
    warehouse, catalogue = load_case(case_dir)

    x0, y0, x1, y1 = warehouse.bounding_box
    print(f"  Polygon   : {len(warehouse.polygon)} vertices  "
          f"bbox [{x0:.0f},{y0:.0f}]→[{x1:.0f},{y1:.0f}] mm")
    print(f"  Area      : {warehouse.area/1e6:.2f} m²  "
          f"(usable ≈ {warehouse.usable_area/1e6:.2f} m²)")
    print(f"  Aisle     : {warehouse.aisle_width:.0f} mm  "
          f"(col 4 — educated guess)")
    print(f"  Obstacles : {len(warehouse.obstacles)}")
    print(f"  Bay types : {len(catalogue)}")
    print(f"  Ceiling   : {len(warehouse.ceiling_profile)} zone(s)")
    for yb, h in warehouse.ceiling_profile:
        print(f"              y >= {yb:.0f} mm  →  {h/1000:.1f}m ceiling")
    print(f"  Heights   : {sorted(set(b.height for b in catalogue))} mm")

    # ---- algorithms ----
    print("\n  Running optimisers...")

    print("    greedy_density_first ... ", end="", flush=True)
    s1 = greedy_density_first(catalogue, warehouse)
    print(f"cap={s1.total_capacity:.0f}  cost={s1.total_cost:,.0f}  "
          f"bays={len(s1.placements)}  [{s1.runtime_s:.2f}s]")

    print("    greedy_multistart    ... ", end="", flush=True)
    s2 = greedy_multistart(catalogue, warehouse)
    print(f"cap={s2.total_capacity:.0f}  cost={s2.total_cost:,.0f}  "
          f"bays={len(s2.placements)}  [{s2.runtime_s:.2f}s]")

    seed = max([s1, s2], key=lambda s: (s.total_capacity, -s.total_cost))
    print(f"    simulated_annealing  ... (seed: {seed.method})", end="", flush=True)
    s3 = simulated_annealing(catalogue, warehouse, initial=seed, iterations=2000)
    print(f"\n                            cap={s3.total_capacity:.0f}  "
          f"cost={s3.total_cost:,.0f}  bays={len(s3.placements)}  "
          f"[{s3.runtime_s:.2f}s]")

    best = max([s1, s2, s3],
               key=lambda s: (s.total_capacity, -s.total_cost))

    print(f"\n  BEST: {best.method}")
    print(f"        capacity={best.total_capacity:.0f} pallets | "
          f"cost={best.total_cost:,.0f} | "
          f"bays={len(best.placements)} | "
          f"util={best.utilisation*100:.1f}%")

    # ---- validate ----
    print("\n  Validation:")
    print_validation(best, warehouse, catalogue)

    # ---- write outputs ----
    ui_data = format_solution_for_ui(best, warehouse, catalogue, case_name)

    json_path = os.path.join(out_dir, f"{case_name}_solution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ui_data, f, indent=2)
    print(f"\n  Wrote: {json_path}")

    png_path = os.path.join(out_dir, f"{case_name}_layout.png")
    render(best, warehouse, catalogue, png_path,
           title=f"{case_name} — {best.method}")
    print(f"  Wrote: {png_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")
