"""
dashboard/qaoa_worker.py — runs QAOA in its own process.

qiskit-aer's native C++ backend can crash the whole interpreter (a
segfault, not a catchable Python exception) under constrained/sandboxed
containers. Running it as a subprocess means a crash there can't take
the Streamlit app down with it -- the parent just sees a failed
subprocess and reports it cleanly.

Usage: python qaoa_worker.py < problem.json > result.json
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qcrs import Job, Node, SchedulingProblem, QAOASolver  # noqa: E402


def main():
    payload = json.load(sys.stdin)

    jobs = [
        Job(i, cpu=float(j["cpu"]), mem=float(j["mem"]), label=j["label"])
        for i, j in enumerate(payload["jobs"])
    ]
    nodes = [
        Node(j, cpu_cap=float(n["cpu_cap"]), mem_cap=float(n["mem_cap"]),
             cost_per_unit=float(n["cost"]), label=n["label"])
        for j, n in enumerate(payload["nodes"])
    ]
    problem = SchedulingProblem(
        jobs, nodes,
        penalty_assign=payload.get("penalty_assign", 0.0),
        penalty_cpu=payload.get("penalty_cpu", 0.0),
        penalty_mem=payload.get("penalty_mem", 0.0),
    )

    solver = QAOASolver(
        problem,
        p_layers=payload.get("p_layers", 2),
        n_shots=payload.get("n_shots", 2048),
        backend=payload.get("backend", "statevector"),
        verbose=False,
    )
    x, metrics = solver.solve(n_restarts=payload.get("n_restarts", 3))

    out = {
        "x": [int(v) for v in x],
        "metrics": metrics,
        "counts": solver._result_counts,
    }
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
