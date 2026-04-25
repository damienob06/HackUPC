"""
Warehouse Optimizer — web server.

    cd mvp-claude
    py -3 server.py

Then open http://localhost:5000
"""
from __future__ import annotations
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory

from models import Warehouse, BayType, Obstacle
from algorithms import greedy_multistart
from output_formatter import format_solution_for_ui

app = Flask(__name__, static_folder="frontend", static_url_path="")

HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCE_PATH = os.path.join(HERE, "instance.json")

_cache: dict | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Load instance.json (metres) → Warehouse + catalogue (mm)
# ---------------------------------------------------------------------------

def _load_instance() -> tuple:
    with open(INSTANCE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    w = data["warehouse"]
    width_mm = w["width"] * 1000
    depth_mm = w["depth"] * 1000
    aisle_mm = w.get("aisle_width", 2.8) * 1000
    demand   = float(w.get("demand", 0))

    polygon = [
        (0.0, 0.0), (width_mm, 0.0),
        (width_mm, depth_mm), (0.0, depth_mm),
    ]

    obstacles = [
        Obstacle(
            x=o["x"] * 1000, y=o["y"] * 1000,
            width=o["width"] * 1000, depth=o["depth"] * 1000,
        )
        for o in w.get("obstacles", [])
    ]

    warehouse = Warehouse(
        polygon=polygon,
        ceiling_profile=[(0.0, 5000.0)],
        aisle_width=aisle_mm,
        obstacles=obstacles,
        demand=demand,
    )

    catalogue = [
        BayType(
            id=b["id"],
            width=b["width"] * 1000,
            depth=b["depth"] * 1000,
            height=3000.0,
            col4=aisle_mm,
            capacity=float(b["capacity"]),
            cost=float(b["cost"]),
        )
        for b in data["bays"]
    ]
    return warehouse, catalogue


def _run_solver(algorithm: str = "greedy_multistart") -> dict:
    warehouse, catalogue = _load_instance()
    if algorithm == "greedy_multistart":
        from algorithms import greedy_multistart as fn
    elif algorithm == "greedy_density_first":
        from algorithms import greedy_density_first as fn
    elif algorithm == "greedy_cost_efficiency":
        from algorithms import greedy_cost_efficiency as fn
    elif algorithm == "simulated_annealing":
        from algorithms import simulated_annealing
        seed = greedy_multistart(catalogue, warehouse)
        solution = simulated_annealing(catalogue, warehouse, initial=seed, iterations=2000)
        return format_solution_for_ui(solution, warehouse, catalogue, "default")
    else:
        fn = greedy_multistart

    solution = fn(catalogue, warehouse)
    return format_solution_for_ui(solution, warehouse, catalogue, "default")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/solution")
def get_solution():
    global _cache
    with _lock:
        if _cache is None:
            _cache = _run_solver()
    return jsonify(_cache)


@app.route("/api/solve", methods=["POST"])
def solve():
    global _cache
    body = request.get_json(silent=True) or {}
    algorithm = body.get("algorithm", "greedy_multistart")
    result = _run_solver(algorithm)
    with _lock:
        _cache = result
    return jsonify(result)


@app.route("/api/algorithms")
def list_algorithms():
    return jsonify([
        {"id": "greedy_cost_efficiency", "label": "Greedy — Cost Efficiency", "fast": True},
        {"id": "greedy_density_first",   "label": "Greedy — Density First",   "fast": True},
        {"id": "greedy_multistart",      "label": "Greedy — Multistart",       "fast": True},
        {"id": "simulated_annealing",    "label": "Simulated Annealing",       "fast": False},
    ])


if __name__ == "__main__":
    print("Warehouse Optimizer  ->  http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)
