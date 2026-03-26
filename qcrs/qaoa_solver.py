"""
qaoa_solver.py — QAOA-based Quantum Solver
==========================================
Implements QAOA (Quantum Approximate Optimization Algorithm) for the
cloud scheduling QUBO problem.

Architecture:
  1. Build cost Hamiltonian H_C from QUBO matrix Q
  2. Build mixer Hamiltonian H_B (X-rotation on each qubit)
  3. Construct p-layer QAOA ansatz: alternate e^{-i*gamma*H_C} and e^{-i*beta*H_B}
  4. Optimize (gamma, beta) parameters to minimize <psi|H_C|psi>
  5. Sample the best bitstring as the solution

Requires: qiskit >= 1.0, qiskit-aer, scipy
"""

import numpy as np
import time
from typing import Tuple, List, Optional, Callable

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from scipy.optimize import minimize
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("[WARNING] Qiskit not found. QAOASolver will run in simulation-only mode.")

from .problem import SchedulingProblem


class QAOASolver:
    """
    QAOA solver for cloud scheduling QUBO problems.

    Parameters
    ----------
    problem   : SchedulingProblem
    p_layers  : int — QAOA depth (number of alternating layers). Higher = better quality, slower.
    n_shots   : int — measurement shots per circuit evaluation
    optimizer : str — classical optimizer for variational parameters ('COBYLA', 'BFGS', 'Nelder-Mead')
    backend   : 'statevector' (exact, no noise) | 'qasm' (shot-based, realistic)
    seed      : int — reproducibility
    """

    def __init__(
        self,
        problem: SchedulingProblem,
        p_layers: int = 2,
        n_shots: int = 4096,
        optimizer: str = "COBYLA",
        backend: str = "statevector",
        seed: int = 42,
        verbose: bool = True,
    ):
        self.problem = problem
        self.p = p_layers
        self.n_shots = n_shots
        self.optimizer = optimizer
        self.backend_type = backend
        self.seed = seed
        self.verbose = verbose
        self.n_qubits = problem.n_vars

        self._energy_history: List[float] = []
        self._param_history: List[np.ndarray] = []
        self._optimal_params: Optional[np.ndarray] = None
        self._optimal_circuit: Optional[object] = None
        self._result_counts: Optional[dict] = None

        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required. Run: pip install qiskit qiskit-aer")

        # Build backend
        if backend == "statevector":
            self._backend = AerSimulator(method="statevector")
        else:
            self._backend = AerSimulator(method="automatic", seed_simulator=seed)

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_qaoa_circuit(self, gammas: List[float], betas: List[float]) -> "QuantumCircuit":
        """
        Constructs the QAOA circuit for given gamma/beta angles.

        |psi_p> = prod_{k=1}^{p} [U_B(beta_k) * U_C(gamma_k)] |+>^n

        U_C(gamma) = e^{-i gamma H_C}  — applied as ZZ rotations from QUBO
        U_B(beta)  = e^{-i beta H_B}   — single-qubit X rotations (mixer)
        """
        Q = self.problem.build_qubo()
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # Initial state: equal superposition |+>^n
        qc.h(range(n))

        for layer in range(self.p):
            gamma = gammas[layer]
            beta = betas[layer]

            # --- Cost unitary U_C(gamma) ---
            # Diagonal terms: RZ rotation from Q[i,i]
            for i in range(n):
                if abs(Q[i, i]) > 1e-10:
                    qc.rz(2 * gamma * Q[i, i], i)

            # Off-diagonal terms: ZZ interaction (CNOT - RZ - CNOT)
            for i in range(n):
                for j in range(i + 1, n):
                    coeff = Q[i, j] + Q[j, i]   # symmetrize
                    if abs(coeff) > 1e-10:
                        qc.cx(i, j)
                        qc.rz(2 * gamma * coeff, j)
                        qc.cx(i, j)

            # --- Mixer unitary U_B(beta) ---
            for i in range(n):
                qc.rx(2 * beta, i)

        # Measurement
        if self.backend_type == "qasm":
            qc.measure_all()

        return qc

    def _compute_expectation(self, params: np.ndarray) -> float:
        """
        Evaluates <H_C> for a given set of QAOA parameters.
        This is the objective function minimized by the classical optimizer.
        """
        gammas = params[:self.p]
        betas = params[self.p:]
        Q = self.problem.build_qubo()
        qc = self._build_qaoa_circuit(list(gammas), list(betas))

        if self.backend_type == "statevector":
            # Exact expectation via statevector
            sv_qc = qc.copy()
            sv_qc.save_statevector()
            job = self._backend.run(transpile(sv_qc, self._backend))
            sv = job.result().get_statevector()
            expectation = self._sv_expectation(sv, Q)
        else:
            # Shot-based: measure, compute energy per bitstring
            job = self._backend.run(
                transpile(qc, self._backend),
                shots=self.n_shots,
                seed_simulator=self.seed,
            )
            counts = job.result().get_counts()
            expectation = self._counts_expectation(counts, Q)

        self._energy_history.append(float(expectation))
        self._param_history.append(params.copy())
        return float(expectation)

    def _sv_expectation(self, sv, Q: np.ndarray) -> float:
        """Exact <H_C> from statevector using diagonal QUBO."""
        n = self.n_qubits
        probs = np.abs(sv.data) ** 2
        expectation = 0.0
        for idx, prob in enumerate(probs):
            if prob < 1e-12:
                continue
            x = np.array([(idx >> k) & 1 for k in range(n)], dtype=float)
            e = float(x @ Q @ x)
            expectation += prob * e
        return expectation

    def _counts_expectation(self, counts: dict, Q: np.ndarray) -> float:
        """Shot-based <H_C> estimate from measurement counts."""
        total_shots = sum(counts.values())
        expectation = 0.0
        for bitstring, count in counts.items():
            # Qiskit returns bitstrings with qubit 0 at the right
            x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
            if len(x) != self.n_qubits:
                continue
            e = float(x @ Q @ x)
            expectation += (count / total_shots) * e
        return expectation

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _random_initial_params(self) -> np.ndarray:
        """Heuristic initialization: gammas uniform in [0, pi], betas in [0, pi/2]."""
        rng = np.random.default_rng(self.seed)
        gammas = rng.uniform(0, np.pi, size=self.p)
        betas = rng.uniform(0, np.pi / 2, size=self.p)
        return np.concatenate([gammas, betas])

    def optimize(self, n_restarts: int = 3) -> np.ndarray:
        """
        Runs the classical optimizer to find optimal QAOA parameters.
        Tries n_restarts random initializations, keeps the best.
        """
        if self.verbose:
            print(f"[QAOA] Optimizing p={self.p} layers, {self.n_qubits} qubits, "
                  f"backend={self.backend_type}")

        best_params = None
        best_energy = float("inf")

        for restart in range(n_restarts):
            x0 = self._random_initial_params()
            if self.verbose:
                print(f"  Restart {restart+1}/{n_restarts}...", end=" ", flush=True)

            self._energy_history = []
            result = minimize(
                self._compute_expectation,
                x0,
                method=self.optimizer,
                options={"maxiter": 200, "rhobeg": 0.5} if self.optimizer == "COBYLA" else {"maxiter": 200},
            )

            if self.verbose:
                print(f"energy = {result.fun:.4f}")

            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()

        self._optimal_params = best_params
        if self.verbose:
            print(f"[QAOA] Best expectation value: {best_energy:.4f}")
        return best_params

    # ------------------------------------------------------------------
    # Sampling the solution
    # ------------------------------------------------------------------

    def sample(self, params: Optional[np.ndarray] = None, n_shots: Optional[int] = None) -> dict:
        """
        Runs the optimized circuit and returns measurement counts.
        Returns {bitstring: count} dict.
        """
        if params is None:
            params = self._optimal_params
        if params is None:
            raise ValueError("Call optimize() first or provide params.")

        shots = n_shots or self.n_shots
        gammas = params[:self.p]
        betas = params[self.p:]
        qc = self._build_qaoa_circuit(list(gammas), list(betas))

        if self.backend_type == "statevector":
            # Sample from statevector
            qc_meas = qc.copy()
            qc_meas.measure_all()
        else:
            qc_meas = qc

        job = self._backend.run(
            transpile(qc_meas, self._backend),
            shots=shots,
            seed_simulator=self.seed,
        )
        counts = job.result().get_counts()
        self._result_counts = counts
        return counts

    def best_bitstring(self, counts: Optional[dict] = None) -> np.ndarray:
        """Returns the most frequent (lowest energy) bitstring as numpy array."""
        if counts is None:
            counts = self._result_counts
        if counts is None:
            raise ValueError("Run sample() first.")

        Q = self.problem.build_qubo()

        # Pick the bitstring with the lowest QUBO energy (not just most frequent)
        best_bits = None
        best_e = float("inf")
        for bitstring in counts:
            x = np.array([int(b) for b in reversed(bitstring)], dtype=int)
            if len(x) != self.n_qubits:
                continue
            e = float(x @ Q @ x)
            if e < best_e:
                best_e = e
                best_bits = x.copy()

        return best_bits

    # ------------------------------------------------------------------
    # Main solve entry point
    # ------------------------------------------------------------------

    def solve(self, n_restarts: int = 3) -> Tuple[np.ndarray, dict]:
        """
        Full QAOA solve: optimize → sample → decode best solution.

        Returns:
            x       : binary assignment vector
            metrics : solve metadata
        """
        t0 = time.time()

        params = self.optimize(n_restarts=n_restarts)
        counts = self.sample(params)
        x = self.best_bitstring(counts)

        if x is None:
            x = np.zeros(self.n_qubits, dtype=int)

        energy = self.problem.energy(x)
        violations = self.problem.constraint_violations(x)
        feasible = (
            len(violations["unassigned_jobs"]) == 0
            and len(violations["cpu_overload"]) == 0
            and len(violations["mem_overload"]) == 0
        )

        # Approximation ratio (vs best sampled energy)
        all_energies = []
        Q = self.problem.build_qubo()
        for bitstring, count in counts.items():
            xb = np.array([int(b) for b in reversed(bitstring)], dtype=int)
            if len(xb) == self.n_qubits:
                all_energies.append(float(xb @ Q @ xb))
        min_sampled = min(all_energies) if all_energies else energy

        metrics = {
            "method": f"QAOA_p{self.p}",
            "energy": energy,
            "min_sampled_energy": min_sampled,
            "solve_time_s": time.time() - t0,
            "feasible": feasible,
            "violations": violations,
            "optimal_params": params.tolist(),
            "energy_convergence": self._energy_history,
            "n_qubits": self.n_qubits,
            "p_layers": self.p,
            "n_shots": self.n_shots,
            "unique_bitstrings": len(counts),
            "top_bitstrings": sorted(counts.items(), key=lambda x: -x[1])[:5],
        }

        return x, metrics

    def get_circuit(self, params: Optional[np.ndarray] = None, draw: bool = True) -> "QuantumCircuit":
        """Returns (and optionally prints) the QAOA circuit for the optimal params."""
        if params is None:
            params = self._optimal_params
        gammas = params[:self.p]
        betas = params[self.p:]
        qc = self._build_qaoa_circuit(list(gammas), list(betas))
        if draw:
            print(qc.draw(output="text", fold=120))
        return qc

    @property
    def energy_history(self) -> List[float]:
        return self._energy_history

    @property
    def optimal_params(self) -> Optional[np.ndarray]:
        return self._optimal_params
