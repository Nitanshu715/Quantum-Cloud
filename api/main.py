from fastapi import FastAPI
from pydantic import BaseModel

from qcrs.problem import make_small_problem
from qcrs.classical_solver import (
    GreedySolver,
    SimulatedAnnealingSolver,
    BruteForceSolver,
)

# Try importing QAOA safely (since Qiskit may fail sometimes)
try:
    from qcrs.qaoa_solver import QAOASolver
    QAOA_AVAILABLE = True
except:
    QAOA_AVAILABLE = False

app = FastAPI(
    title="Quantum Cloud Resource Scheduler API",
    description="QUBO-based scheduling using classical and quantum solvers",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Quantum Cloud Scheduler API running",
        "endpoints": [
            "/run",
            "/compare",
            "/qaoa",
            "/health",
            "/docs"
        ]
    }


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# RUN GREEDY (DEFAULT)
# ─────────────────────────────────────────────
@app.get("/run")
def run_scheduler():
    problem = make_small_problem()

    solver = GreedySolver(problem)
    _, metrics = solver.solve()

    return {
        "solver": "greedy",
        "energy": metrics["energy"],
        "feasible": metrics["feasible"]
    }


# ─────────────────────────────────────────────
# COMPARE ALL CLASSICAL SOLVERS
# ─────────────────────────────────────────────
@app.get("/compare")
def compare_solvers():
    problem = make_small_problem()

    g = GreedySolver(problem).solve()[1]
    sa = SimulatedAnnealingSolver(problem).solve()[1]
    bf = BruteForceSolver(problem).solve()[1]

    return {
        "greedy": {
            "energy": g["energy"],
            "feasible": g["feasible"]
        },
        "simulated_annealing": {
            "energy": sa["energy"],
            "feasible": sa["feasible"]
        },
        "brute_force": {
            "energy": bf["energy"],
            "feasible": bf["feasible"],
            "optimal": True
        }
    }


# ─────────────────────────────────────────────
# QAOA (SAFE)
# ─────────────────────────────────────────────
@app.get("/qaoa")
def run_qaoa():
    if not QAOA_AVAILABLE:
        return {
            "status": "skipped",
            "reason": "QAOA/Qiskit not available"
        }

    problem = make_small_problem()
    solver = QAOASolver(problem)

    _, metrics = solver.solve()

    return {
        "solver": "qaoa",
        "energy": metrics["energy"],
        "feasible": metrics["feasible"]
    }
