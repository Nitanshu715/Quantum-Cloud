"""
problem.py — Workload-to-Hamiltonian Encoder
=============================================
Encodes a cloud resource scheduling problem as a QUBO (Quadratic Unconstrained
Binary Optimization) problem, which can then be mapped to a cost Hamiltonian
for QAOA.

Problem definition:
  - N jobs, each needing cpu_i CPUs and mem_i GB RAM
  - M nodes, each with capacity cpu_cap_j and mem_cap_j
  - Binary variable x_{i,j} = 1 if job i is assigned to node j
  - Objective: minimize total cost (load imbalance + constraint violations)

QUBO form:  minimize  x^T Q x
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Job:
    """Represents a cloud workload/job."""
    job_id: int
    cpu: float          # CPU cores required
    mem: float          # RAM (GB) required
    priority: float = 1.0
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"job_{self.job_id}"


@dataclass
class Node:
    """Represents a compute node / VM."""
    node_id: int
    cpu_cap: float      # Total CPU cores
    mem_cap: float      # Total RAM (GB)
    cost_per_unit: float = 1.0   # Cost coefficient
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"node_{self.node_id}"


@dataclass
class SchedulingProblem:
    """
    Container for a cloud scheduling instance.
    Builds the QUBO matrix Q used by both classical and quantum solvers.
    """
    jobs: List[Job]
    nodes: List[Node]
    penalty_assign: float = 0.0   # auto-calibrated in __post_init__ if 0
    penalty_cpu: float = 0.0      # auto-calibrated in __post_init__ if 0
    penalty_mem: float = 0.0      # auto-calibrated in __post_init__ if 0

    def __post_init__(self):
        self.n_jobs = len(self.jobs)
        self.n_nodes = len(self.nodes)
        self.n_vars = self.n_jobs * self.n_nodes

        # Auto-calibrate penalties so feasibility is always enforced.
        # Rule: penalty > max possible energy gain from violating that constraint.
        max_cost = sum(
            n.cost_per_unit * (j.cpu + j.mem * 0.1)
            for j in self.jobs for n in self.nodes
        )
        if self.penalty_assign == 0.0:
            self.penalty_assign = max(50.0, max_cost * 10)
        if self.penalty_cpu == 0.0:
            max_cpu_excess = sum(
                max(0, j.cpu - n.cpu_cap)
                for j in self.jobs for n in self.nodes
            ) + 1
            self.penalty_cpu = max(50.0, self.penalty_assign / max(1, max_cpu_excess))
        if self.penalty_mem == 0.0:
            self.penalty_mem = self.penalty_cpu * 0.5   # total binary variables

    def var_index(self, i: int, j: int) -> int:
        """Flatten (job i, node j) -> single variable index."""
        return i * self.n_nodes + j

    def build_qubo(self) -> np.ndarray:
        """
        Constructs the QUBO matrix Q of shape (n_vars, n_vars).

        Energy = sum_i penalty_assign*(1 - sum_j x_{ij})^2
               + sum_j penalty_cpu * max(0, sum_i cpu_i*x_{ij} - cpu_cap_j)^2  [relaxed]
               + sum_j penalty_mem * max(0, sum_i mem_i*x_{ij} - mem_cap_j)^2  [relaxed]
               + cost objective (load imbalance)

        For the QUBO, we expand the squared terms.
        """
        N = self.n_vars
        Q = np.zeros((N, N))

        # --- Constraint 1: each job must be assigned to exactly one node ---
        # Expand (sum_j x_ij - 1)^2:
        #   = sum_j x_ij^2 - 2 sum_j x_ij + 1 + 2 sum_{j<k} x_ij * x_ik
        # For binary x: x^2 = x, so linear term coeff = 1 - 2 = -1 per variable.
        # The constant +1 is absorbed into offset (doesn't affect optimum).
        # Penalty A must be large enough that assigning all jobs < skipping any job.
        for i in range(self.n_jobs):
            for j in range(self.n_nodes):
                v = self.var_index(i, j)
                Q[v, v] += self.penalty_assign * (-1)   # from x_ij^2 - 2*x_ij = -x_ij (binary)
            for j in range(self.n_nodes):
                for k in range(j + 1, self.n_nodes):
                    v1 = self.var_index(i, j)
                    v2 = self.var_index(i, k)
                    Q[v1, v2] += 2 * self.penalty_assign  # penalise assigning same job twice

        # --- Constraints 2 & 3: CPU and memory capacity per node ---
        # Encode as: penalty * (sum_i demand_i * x_{ij} / cap_j)^2  where the sum is
        # normalised by capacity so diagonal terms stay O(1) not O(cap^2).
        # Expanding: penalty/cap^2 * [sum_i demand_i^2 * x_{ij}^2
        #            + 2*sum_{i1<i2} demand_i1*demand_i2 * x_{i1j}*x_{i2j}]
        # This correctly penalises ANY combination that over-loads the node,
        # including 3-way or k-way co-location.
        for j, node in enumerate(self.nodes):
            cap_cpu = max(node.cpu_cap, 1e-6)
            cap_mem = max(node.mem_cap, 1e-6)
            for i1 in range(self.n_jobs):
                v1 = self.var_index(i1, j)
                # Diagonal: demand_i^2 / cap^2  (positive, adds quadratic cost to utilisation)
                Q[v1, v1] += self.penalty_cpu * (self.jobs[i1].cpu / cap_cpu) ** 2
                Q[v1, v1] += self.penalty_mem * (self.jobs[i1].mem / cap_mem) ** 2
                for i2 in range(i1 + 1, self.n_jobs):
                    v2 = self.var_index(i2, j)
                    Q[v1, v2] += 2 * self.penalty_cpu * (self.jobs[i1].cpu * self.jobs[i2].cpu) / cap_cpu ** 2
                    Q[v1, v2] += 2 * self.penalty_mem * (self.jobs[i1].mem * self.jobs[i2].mem) / cap_mem ** 2

        # --- Cost objective: penalize uneven load (imbalance) ---
        # Add a small cost proportional to node cost_per_unit * cpu usage
        for i, job in enumerate(self.jobs):
            for j, node in enumerate(self.nodes):
                v = self.var_index(i, j)
                Q[v, v] += node.cost_per_unit * (job.cpu + job.mem * 0.1)

        return Q

    def energy(self, assignment: np.ndarray) -> float:
        """
        Compute the QUBO energy for a given binary assignment vector.
        assignment: 1D numpy array of length n_vars (0 or 1)
        """
        Q = self.build_qubo()
        return float(assignment @ Q @ assignment)

    def decode_assignment(self, x: np.ndarray) -> dict:
        """
        Decode a binary vector x into a human-readable job->node mapping.
        Returns dict: {job_label: node_label or None if unassigned}
        """
        result = {}
        for i, job in enumerate(self.jobs):
            assigned_node = None
            for j, node in enumerate(self.nodes):
                if x[self.var_index(i, j)] == 1:
                    assigned_node = node.label
                    break
            result[job.label] = assigned_node
        return result

    def constraint_violations(self, x: np.ndarray) -> dict:
        """
        Check all constraints for a given assignment.
        Returns a dict with violation details.
        """
        violations = {"unassigned_jobs": [], "cpu_overload": {}, "mem_overload": {}}

        # Check each job is assigned exactly once
        for i, job in enumerate(self.jobs):
            total = sum(x[self.var_index(i, j)] for j in range(self.n_nodes))
            if total != 1:
                violations["unassigned_jobs"].append(job.label)

        # Check resource capacities
        for j, node in enumerate(self.nodes):
            cpu_used = sum(
                self.jobs[i].cpu * x[self.var_index(i, j)]
                for i in range(self.n_jobs)
            )
            mem_used = sum(
                self.jobs[i].mem * x[self.var_index(i, j)]
                for i in range(self.n_jobs)
            )
            if cpu_used > node.cpu_cap + 1e-6:
                violations["cpu_overload"][node.label] = f"{cpu_used:.1f}/{node.cpu_cap}"
            if mem_used > node.mem_cap + 1e-6:
                violations["mem_overload"][node.label] = f"{mem_used:.1f}/{node.mem_cap}"

        return violations

    def summary(self) -> str:
        lines = [
            f"SchedulingProblem: {self.n_jobs} jobs × {self.n_nodes} nodes",
            f"  Binary variables: {self.n_vars}",
            f"  QUBO matrix size: {self.n_vars}×{self.n_vars}",
            "\nJobs:",
        ]
        for job in self.jobs:
            lines.append(f"  [{job.label}] CPU={job.cpu}, MEM={job.mem}GB, priority={job.priority}")
        lines.append("\nNodes:")
        for node in self.nodes:
            lines.append(f"  [{node.label}] CPU cap={node.cpu_cap}, MEM cap={node.mem_cap}GB")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_small_problem(seed: int = 42) -> SchedulingProblem:
    """
    3 jobs × 2 nodes — small enough for exact QAOA with 6 qubits.
    Ideal for demo and testing.
    """
    rng = np.random.default_rng(seed)
    jobs = [
        Job(0, cpu=2.0, mem=4.0, label="web-api"),
        Job(1, cpu=4.0, mem=8.0, label="ml-train"),
        Job(2, cpu=1.0, mem=2.0, label="db-backup"),
    ]
    nodes = [
        Node(0, cpu_cap=6.0, mem_cap=12.0, cost_per_unit=1.0, label="node-A"),
        Node(1, cpu_cap=6.0, mem_cap=12.0, cost_per_unit=1.2, label="node-B"),
    ]
    return SchedulingProblem(jobs, nodes)


def make_medium_problem(n_jobs: int = 4, n_nodes: int = 3, seed: int = 7) -> SchedulingProblem:
    """
    Parameterized medium problem. 4×3 = 12 qubits.
    Tests the hybrid pipeline boundary.
    """
    rng = np.random.default_rng(seed)
    jobs = [
        Job(i,
            cpu=rng.uniform(1, 6),
            mem=rng.uniform(2, 16),
            priority=rng.uniform(0.5, 2.0),
            label=f"job-{i:02d}")
        for i in range(n_jobs)
    ]
    nodes = [
        Node(j,
             cpu_cap=rng.uniform(8, 16),
             mem_cap=rng.uniform(16, 32),
             cost_per_unit=rng.uniform(0.8, 1.5),
             label=f"node-{chr(65+j)}")
        for j in range(n_nodes)
    ]
    return SchedulingProblem(jobs, nodes)
