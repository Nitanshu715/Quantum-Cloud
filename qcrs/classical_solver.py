"""
classical_solver.py — Classical Baseline Solvers
=================================================
Provides two classical approaches for comparison against QAOA:
  1. GreedySolver   — fast heuristic, O(n log n)
  2. BruteForceSolver — exact (exponential), only usable for tiny problems
  3. QUBOSimulatedAnnealing — classical SA on the QUBO, stronger baseline
"""

import numpy as np
import time
from typing import Optional, Tuple
from .problem import SchedulingProblem


class GreedySolver:
    """
    First-fit decreasing greedy: sort jobs by CPU demand descending,
    assign each to the least-loaded node that has capacity.
    Fast but not optimal.
    """

    def __init__(self, problem: SchedulingProblem):
        self.problem = problem

    def solve(self) -> Tuple[np.ndarray, dict]:
        """
        Returns:
            x       : binary assignment vector (n_vars,)
            metrics : dict with energy, solve_time, method, feasible
        """
        t0 = time.time()
        p = self.problem
        x = np.zeros(p.n_vars, dtype=int)

        # Track remaining capacity
        cpu_remaining = [n.cpu_cap for n in p.nodes]
        mem_remaining = [n.mem_cap for n in p.nodes]

        # Sort jobs by descending CPU demand
        job_order = sorted(range(p.n_jobs), key=lambda i: p.jobs[i].cpu, reverse=True)

        assigned = 0
        for i in job_order:
            job = p.jobs[i]
            # Find best-fit node (minimize wasted capacity)
            best_j = None
            best_waste = float("inf")
            for j in range(p.n_nodes):
                if cpu_remaining[j] >= job.cpu and mem_remaining[j] >= job.mem:
                    waste = (cpu_remaining[j] - job.cpu) + (mem_remaining[j] - job.mem) * 0.1
                    if waste < best_waste:
                        best_waste = waste
                        best_j = j

            if best_j is not None:
                x[p.var_index(i, best_j)] = 1
                cpu_remaining[best_j] -= job.cpu
                mem_remaining[best_j] -= job.mem
                assigned += 1

        energy = p.energy(x)
        violations = p.constraint_violations(x)
        feasible = (
            len(violations["unassigned_jobs"]) == 0
            and len(violations["cpu_overload"]) == 0
            and len(violations["mem_overload"]) == 0
        )

        metrics = {
            "method": "greedy_first_fit_decreasing",
            "energy": energy,
            "solve_time_s": time.time() - t0,
            "feasible": feasible,
            "assigned_jobs": assigned,
            "violations": violations,
        }
        return x, metrics


class BruteForceSolver:
    """
    Exhaustive search over all 2^n_vars assignments.
    Only practical for n_vars <= 20 (i.e. ~3 jobs × 6 nodes).
    Finds the globally optimal solution — the gold standard.
    """

    def __init__(self, problem: SchedulingProblem, max_vars: int = 20):
        self.problem = problem
        self.max_vars = max_vars

    def solve(self) -> Tuple[Optional[np.ndarray], dict]:
        p = self.problem
        if p.n_vars > self.max_vars:
            return None, {
                "method": "brute_force",
                "error": f"Too many variables ({p.n_vars} > {self.max_vars}). Use SA or QAOA.",
            }

        t0 = time.time()
        Q = p.build_qubo()
        best_x = None
        best_energy = float("inf")

        for bits in range(2 ** p.n_vars):
            x = np.array([(bits >> k) & 1 for k in range(p.n_vars)], dtype=int)
            e = float(x @ Q @ x)
            if e < best_energy:
                best_energy = e
                best_x = x.copy()

        violations = p.constraint_violations(best_x)
        feasible = (
            len(violations["unassigned_jobs"]) == 0
            and len(violations["cpu_overload"]) == 0
            and len(violations["mem_overload"]) == 0
        )

        metrics = {
            "method": "brute_force",
            "energy": best_energy,
            "solve_time_s": time.time() - t0,
            "feasible": feasible,
            "violations": violations,
            "states_evaluated": 2 ** p.n_vars,
        }
        return best_x, metrics


class SimulatedAnnealingSolver:
    """
    Classical simulated annealing on the QUBO.
    Strong baseline — often near-optimal even for larger problems.
    Used in the hybrid pipeline as the classical fallback.
    """

    def __init__(
        self,
        problem: SchedulingProblem,
        n_reads: int = 1000,
        T_start: float = 10.0,
        T_end: float = 0.01,
        cooling: float = 0.995,
        seed: int = 42,
    ):
        self.problem = problem
        self.n_reads = n_reads
        self.T_start = T_start
        self.T_end = T_end
        self.cooling = cooling
        self.rng = np.random.default_rng(seed)

    def _anneal(self, Q: np.ndarray) -> Tuple[np.ndarray, float, list]:
        """Single annealing run. Returns (best_x, best_energy, energy_history)."""
        n = Q.shape[0]
        x = self.rng.integers(0, 2, size=n)
        energy = float(x @ Q @ x)
        best_x = x.copy()
        best_energy = energy
        T = self.T_start
        history = [energy]

        while T > self.T_end:
            # Flip a random bit
            idx = self.rng.integers(0, n)
            x_new = x.copy()
            x_new[idx] ^= 1
            e_new = float(x_new @ Q @ x_new)
            delta = e_new - energy

            if delta < 0 or self.rng.random() < np.exp(-delta / T):
                x = x_new
                energy = e_new
                if energy < best_energy:
                    best_energy = energy
                    best_x = x.copy()

            T *= self.cooling
            history.append(energy)

        return best_x, best_energy, history

    def solve(self) -> Tuple[np.ndarray, dict]:
        t0 = time.time()
        p = self.problem
        Q = p.build_qubo()

        best_x = None
        best_energy = float("inf")
        all_histories = []

        for _ in range(self.n_reads):
            x, e, hist = self._anneal(Q)
            all_histories.append(hist)
            if e < best_energy:
                best_energy = e
                best_x = x.copy()

        violations = p.constraint_violations(best_x)
        feasible = (
            len(violations["unassigned_jobs"]) == 0
            and len(violations["cpu_overload"]) == 0
            and len(violations["mem_overload"]) == 0
        )

        metrics = {
            "method": "simulated_annealing",
            "energy": best_energy,
            "solve_time_s": time.time() - t0,
            "feasible": feasible,
            "violations": violations,
            "n_reads": self.n_reads,
            "energy_history": all_histories[0],   # first run's trajectory
        }
        return best_x, metrics
