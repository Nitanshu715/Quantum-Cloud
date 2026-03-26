from fastapi import FastAPI
from qcrs.problem import make_small_problem
from qcrs.classical_solver import GreedySolver

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Quantum Cloud Scheduler API running"}

@app.get("/run")
def run_scheduler():
    problem = make_small_problem()
    solver = GreedySolver(problem)

    x, metrics = solver.solve()

    return {
        "energy": metrics["energy"],
        "feasible": metrics["feasible"]
    }