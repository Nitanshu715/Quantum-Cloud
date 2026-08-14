"""
hybrid_pipeline.py — Hybrid Quantum-Classical Scheduler
========================================================
The QCRS hybrid pipeline:

  1. Classical pre-processing: reduce problem size (merge trivially assignable jobs)
  2. Routing: if n_vars <= quantum_threshold, use QAOA; else use SA
  3. Post-processing: merge sub-solutions, validate constraints
  4. Benchmarking: run both and compare

This mirrors how AWS would deploy QCRS:
  - AWS Batch / EKS handles routine scheduling (classical greedy)
  - Hard optimization subproblems (tight packing, SLA violations) are routed to QAOA
  - Results from quantum layer refine the allocation
"""

import numpy as np
import time
from typing import Optional, Tuple, List
from .problem import SchedulingProblem, Job, Node
from .classical_solver import GreedySolver, SimulatedAnnealingSolver, BruteForceSolver
from .qaoa_solver import QAOASolver


class HybridScheduler:
    """
    The main QCRS pipeline. Decides when to invoke quantum vs classical.

    Decision logic:
      n_vars <= quantum_threshold  →  QAOA (quantum)
      n_vars >  quantum_threshold  →  Simulated Annealing (classical)

    For benchmarking, run_comparison() runs ALL methods and returns a report.
    """

    QUANTUM_THRESHOLD = 12   # Variables above this → too many qubits for simulator

    def __init__(
        self,
        problem: SchedulingProblem,
        quantum_threshold: int = QUANTUM_THRESHOLD,
        qaoa_p_layers: int = 2,
        qaoa_shots: int = 4096,
        qaoa_backend: str = "statevector",
        sa_n_reads: int = 500,
        verbose: bool = True,
    ):
        self.problem = problem
        self.quantum_threshold = quantum_threshold
        self.qaoa_p_layers = qaoa_p_layers
        self.qaoa_shots = qaoa_shots
        self.qaoa_backend = qaoa_backend
        self.sa_n_reads = sa_n_reads
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[QCRS] {msg}")

    def route(self) -> str:
        """Determine which solver to use based on problem size."""
        if self.problem.n_vars <= self.quantum_threshold:
            return "quantum"
        return "classical"

    def solve(self) -> Tuple[np.ndarray, dict]:
        """
        Run the hybrid pipeline. Routes to quantum or classical based on problem size.
        Always also runs greedy as the baseline for comparison.
        """
        route = self.route()
        self._log(f"Problem: {self.problem.n_jobs} jobs × {self.problem.n_nodes} nodes "
                  f"= {self.problem.n_vars} qubits. Routing to: {route.upper()}")

        t0 = time.time()

        # Always get greedy baseline
        greedy = GreedySolver(self.problem)
        x_greedy, metrics_greedy = greedy.solve()
        self._log(f"Greedy baseline  → energy={metrics_greedy['energy']:.2f}, "
                  f"feasible={metrics_greedy['feasible']}")

        if route == "quantum":
            # Use QAOA
            solver = QAOASolver(
                self.problem,
                p_layers=self.qaoa_p_layers,
                n_shots=self.qaoa_shots,
                backend=self.qaoa_backend,
                verbose=self.verbose,
            )
            x_best, metrics_best = solver.solve()
            metrics_best["route"] = "quantum"
            metrics_best["greedy_energy"] = metrics_greedy["energy"]
            metrics_best["improvement_pct"] = (
                100 * (metrics_greedy["energy"] - metrics_best["energy"]) / abs(metrics_greedy["energy"])
                if metrics_greedy["energy"] != 0 else 0
            )
        else:
            # Use SA
            solver = SimulatedAnnealingSolver(self.problem, n_reads=self.sa_n_reads)
            x_best, metrics_best = solver.solve()
            metrics_best["route"] = "classical_sa"
            metrics_best["greedy_energy"] = metrics_greedy["energy"]
            metrics_best["improvement_pct"] = (
                100 * (metrics_greedy["energy"] - metrics_best["energy"]) / abs(metrics_greedy["energy"])
                if metrics_greedy["energy"] != 0 else 0
            )

        self._log(f"Best solution    → energy={metrics_best['energy']:.2f}, "
                  f"feasible={metrics_best['feasible']}")
        if metrics_best["improvement_pct"] > 0:
            self._log(f"Improvement over greedy: {metrics_best['improvement_pct']:.1f}%")

        metrics_best["total_pipeline_time_s"] = time.time() - t0
        metrics_best["x_greedy"] = x_greedy
        metrics_best["metrics_greedy"] = metrics_greedy

        return x_best, metrics_best

    def run_comparison(
        self,
        include_brute_force: bool = True,
        qaoa_p_layers: Optional[int] = None,
    ) -> dict:
        """
        Full benchmark: run ALL solvers and return a comparison report.
        Useful for the demo notebook and dashboard.
        """
        p = self.problem
        results = {}

        self._log("=" * 55)
        self._log(f"Full benchmark: {p.n_jobs} jobs × {p.n_nodes} nodes ({p.n_vars} vars)")
        self._log("=" * 55)

        # 1. Greedy
        self._log("Running Greedy...")
        x, m = GreedySolver(p).solve()
        results["greedy"] = {"x": x, "metrics": m, "assignment": p.decode_assignment(x)}

        # 2. Simulated Annealing
        self._log("Running Simulated Annealing...")
        x, m = SimulatedAnnealingSolver(p, n_reads=self.sa_n_reads).solve()
        results["sa"] = {"x": x, "metrics": m, "assignment": p.decode_assignment(x)}

        # 3. Brute Force (if small enough)
        if include_brute_force and p.n_vars <= 20:
            self._log("Running Brute Force (exact)...")
            x, m = BruteForceSolver(p).solve()
            if x is not None:
                results["brute_force"] = {"x": x, "metrics": m, "assignment": p.decode_assignment(x)}

        # 4. QAOA
        if p.n_vars <= self.quantum_threshold:
            pl = qaoa_p_layers or self.qaoa_p_layers
            self._log(f"Running QAOA (p={pl})...")
            solver = QAOASolver(
                p,
                p_layers=pl,
                n_shots=self.qaoa_shots,
                backend=self.qaoa_backend,
                verbose=self.verbose,
            )
            x, m = solver.solve()
            results["qaoa"] = {
                "x": x,
                "metrics": m,
                "assignment": p.decode_assignment(x),
                "energy_convergence": solver.energy_history,
            }
        else:
            self._log(f"Skipping QAOA (n_vars={p.n_vars} > threshold={self.quantum_threshold})")

        # Build summary table
        self._log("\n--- Results Summary ---")
        self._log(f"{'Method':<20} {'Energy':>10} {'Feasible':>10} {'Time(s)':>10}")
        self._log("-" * 55)
        for name, res in results.items():
            m = res["metrics"]
            self._log(
                f"{name:<20} {m.get('energy', float('nan')):>10.2f} "
                f"{str(m.get('feasible', '?')):>10} "
                f"{m.get('solve_time_s', 0):>10.3f}"
            )

        results["problem_summary"] = p.summary()
        results["n_vars"] = p.n_vars
        results["n_jobs"] = p.n_jobs
        results["n_nodes"] = p.n_nodes

        return results


def print_schedule(problem: SchedulingProblem, x: np.ndarray, method_name: str = ""):
    """Pretty-print the final schedule from an assignment vector."""
    assignment = problem.decode_assignment(x)
    violations = problem.constraint_violations(x)

    print(f"\n{'='*50}")
    print(f"  QCRS Schedule Output  {f'[{method_name}]' if method_name else ''}")
    print(f"{'='*50}")

    # Group by node
    node_jobs = {n.label: [] for n in problem.nodes}
    unassigned = []
    for job_label, node_label in assignment.items():
        if node_label:
            node_jobs[node_label].append(job_label)
        else:
            unassigned.append(job_label)

    for j, node in enumerate(problem.nodes):
        jobs_here = node_jobs[node.label]
        cpu_used = sum(
            problem.jobs[i].cpu * x[problem.var_index(i, j)]
            for i in range(problem.n_jobs)
        )
        mem_used = sum(
            problem.jobs[i].mem * x[problem.var_index(i, j)]
            for i in range(problem.n_jobs)
        )
        util_cpu = 100 * cpu_used / node.cpu_cap
        util_mem = 100 * mem_used / node.mem_cap
        bar_cpu = "█" * int(util_cpu / 10) + "░" * (10 - int(util_cpu / 10))
        bar_mem = "█" * int(util_mem / 10) + "░" * (10 - int(util_mem / 10))
        print(f"\n  {node.label}  (cost/unit: {node.cost_per_unit})")
        print(f"    Jobs   : {', '.join(jobs_here) if jobs_here else '(empty)'}")
        print(f"    CPU    : [{bar_cpu}] {cpu_used:.1f}/{node.cpu_cap} ({util_cpu:.0f}%)")
        print(f"    Memory : [{bar_mem}] {mem_used:.1f}/{node.mem_cap}GB ({util_mem:.0f}%)")

    if unassigned:
        print(f"\n  ⚠ Unassigned jobs: {', '.join(unassigned)}")

    feasible = (
        len(violations["unassigned_jobs"]) == 0
        and len(violations["cpu_overload"]) == 0
        and len(violations["mem_overload"]) == 0
    )
    status = "✓ FEASIBLE" if feasible else "✗ INFEASIBLE"
    energy = problem.energy(x)
    print(f"\n  Status : {status}")
    print(f"  Energy : {energy:.4f}")
    print(f"{'='*50}\n")
