# Mecalux Warehouse Optimizer

A Python solution for the Mecalux Warehouse Optimizer challenge: choose **which** storage bays to install and **where** to place them so demand (capacity) is met at minimum cost, on a rectangular floor with obstacles and aisle-width requirements.

The emphasis is on the **algorithms** — six different solvers, an independent geometric validator, and a stress-test harness — with simple matplotlib top-down plans for the visual output.

---

## 1. Problem formulation

Inputs (see `instance.json` for the schema):

- Warehouse: `width × depth` (metres), `demand` (capacity units required), `aisle_width` (metres between bay rows), and a list of rectangular `obstacles` (columns, machinery, etc.).
- Bay catalogue: each entry has `id`, `width`, `depth`, `capacity` (units stored), and `cost`.

Outputs:

- A set of **placed bays** — concrete `(x, y, width, depth, rotated)` rectangles inside the warehouse — whose summed capacity ≥ demand and whose total cost is minimised.

Constraints enforced (and re-checked by the independent validator):

1. Every bay lies fully inside the warehouse rectangle.
2. No two bays overlap.
3. No bay overlaps an obstacle.
4. Every bay's dimensions match the catalogue (with 90° rotation allowed).
5. Aisle-width gaps separate rows of bays (so forklifts can pass).
6. Total capacity of placed bays ≥ demand.

Objective: **minimise total bay cost** subject to the above.

---

## 2. Approach: decomposition

The problem combines a *combinatorial selection* problem (which bays, how many) with a *2-D packing* problem (where to place them). Solving them jointly is NP-hard and brittle, so the solver decomposes them:

```
SELECTION  →  PLACEMENT  →  VALIDATION
  (ILP /        (shelf-       (independent
   greedy /     packing)       geometric checks)
   SA)
```

- **Selection** picks a multiset of bay types whose summed capacity meets demand.
- **Placement** runs a deterministic 2-D shelf-packer (First-Fit Decreasing Height with rotation, obstacle-aware) to physically lay them out.
- If placement fails, the selector tightens its area budget and tries again.
- **Validation** is run as a completely separate module that re-checks every constraint geometrically — this is the safety net that catches any bug in the selectors or the packer.

---

## 3. Algorithms (`algorithms.py`)

Six solvers, all returning the same `Solution` object so they can be swapped freely:

| # | Name | Idea | Strength |
|---|------|------|----------|
| 1 | `greedy_cost_efficiency` | Sort bays by **cost per capacity unit** ascending; pack greedily; if stuck, repair phase tries other bay types. | Fast, near-optimal when the cheapest bay also packs well. |
| 2 | `greedy_density_first` | Sort by **capacity per m²** descending; pack greedily. | Wins when the warehouse is very tight on area. |
| 3 | `greedy_multistart` | Try every bay as the first pick, plus three standard orderings; return the cheapest feasible result. | Robust workhorse — feasible on every stress-test instance. |
| 4 | `ilp_select_then_pack` | **Integer Linear Program** (PuLP/CBC) minimising `Σ cost·count` subject to `Σ capacity·count ≥ demand` and `Σ area·count ≤ η · warehouse_area`. Then physically packs. Tightens η on failure. | Optimal selection given the area budget. |
| 5 | `ilp_with_aisle_model` | Same ILP but iteratively re-estimates aisle area from the observed row count after each pack. | Better on instances where aisles eat a lot of floor. |
| 6 | `simulated_annealing` | Perturbs bay counts (+1 / −1 / swap), energy = cost + heavy penalty for unmet demand or unplaceable layout, geometric cooling. Seeded from the best greedy/ILP incumbent. | Polishes selection beyond what the ILP's loose area model gives. |

The runner (`main.py`) executes all six, validates each, and reports the best feasible solution.

### Why six instead of one?

The ILP gives the cleanest *lower bound* on cost for a given area budget, but that area budget is only an *approximation* — real placement loses area to fragmentation and aisles, and on tight instances the ILP's chosen multiset may simply not fit. Multi-start greedy and SA compensate by reasoning over the actual packing. On the bundled stress test (15 random instances), SA wins 8, ILP wins 5, multi-start greedy wins 0 outright but is feasible on **all 15** — making it the safety solver. Together they always produce a feasible, validated layout.

---

## 4. Placement engine (`placement.py`)

`shelf_pack()` implements a classic **First-Fit Decreasing Height** shelf-packer adapted for warehouses:

1. Sort bays by `max(width, depth)` descending (largest first).
2. Lay them in horizontal rows; start a new row when the current one overflows the warehouse width.
3. Leave a strip of `aisle_width` between rows.
4. Try both orientations (rotation by 90°), keep the one that fits.
5. If a candidate position collides with an obstacle, slide right until it clears or the row is exhausted.

The packer is **deterministic** given the input ordering, which is what lets the validator and the stress-test harness be meaningful.

---

## 5. Validator (`validator.py`)

The validator deliberately re-implements the geometric checks from scratch (not by calling into the placer) so a bug in placement can't slip past. It checks:

- Boundary containment for every bay
- All-pairs overlap (with a small ε tolerance)
- Bay–obstacle overlap
- Catalogue dimension match (allowing 90° rotation)
- Total placed capacity ≥ demand
- Aisle gap ≥ `aisle_width` between rows

Any solution that fails is rejected from the comparison.

---

## 6. Visualisation (`visualizer.py`)

Top-down warehouse plans rendered with matplotlib:

- Each bay is a coloured rectangle labelled with its catalogue id.
- Obstacles drawn as hatched dark squares.
- Warehouse boundary, scale, aisle annotations, legend with bay counts and cost summary.
- A six-panel `comparison.png` is generated for the side-by-side view.

A colour-blind-friendly palette is used (Wong 2011).

---

## 7. How to run

```bash
# Install the one external dep (everything else is stdlib + numpy/matplotlib)
pip install pulp --break-system-packages

# Solve and visualise the bundled example
python main.py

# Or solve your own instance
python main.py path/to/your_instance.json

# Run the stress test (10 random instances by default)
python stress_test.py 15
```

Outputs land in `output/`:

- `greedy_cost_efficiency.png`, `greedy_density_first.png`, `greedy_multistart.png`,
  `ilp_select_then_pack.png`, `ilp_with_aisle_model.png`, `simulated_annealing.png`
- `comparison.png` — six-panel comparison
- `best_solution.json` — the winning solution serialised

---

## 8. Plugging in the real Mecalux data

Replace `instance.json` (or pass a different path on the CLI) with a file matching this schema:

```json
{
  "warehouse": {
    "width": 32.0,
    "depth": 22.0,
    "demand": 7500,
    "aisle_width": 2.8,
    "obstacles": [
      {"x": 5.0, "y": 8.0, "width": 1.0, "depth": 1.0}
    ]
  },
  "bays": [
    {"id": "S-Light",  "width": 2.0, "depth": 1.0, "capacity":  60, "cost":  900},
    {"id": "M-Std",    "width": 2.7, "depth": 1.1, "capacity": 120, "cost": 1700},
    {"id": "L-Pallet", "width": 3.6, "depth": 1.2, "capacity": 200, "cost": 2600}
  ]
}
```

Units are metres for lengths and arbitrary-but-consistent units for capacity and cost. No code changes are needed — every algorithm reads the same JSON.

---

## 9. Project layout

```
warehouse_optimizer/
├── README.md            ← this file
├── models.py            ← dataclasses + JSON I/O
├── placement.py         ← shelf-packing engine
├── algorithms.py        ← the six solvers
├── validator.py         ← independent geometric checks
├── visualizer.py        ← matplotlib top-down plans
├── main.py              ← runs every solver on one instance
├── stress_test.py       ← random-instance harness
├── instance.json        ← sample warehouse + bay catalogue
└── output/              ← rendered PNGs and best_solution.json
```

---

## 10. Sample result

On the bundled instance (32 × 22 m, demand 7 500, three obstacles, six bay types), all six methods produce a feasible, validated layout. The ILP, aisle-aware ILP, and simulated annealing all converge on **€66 450** (43 × XL-Drive + 1 × L-Pallet + 1 × S-Light = capacity 7 500 exactly). Cost-efficiency greedy lands within 1.6 %, density-first within 32 %.

Runtimes on this instance: greedy < 0.25 s, ILP ≈ 0.05 s, SA ≈ 1.3 s.
