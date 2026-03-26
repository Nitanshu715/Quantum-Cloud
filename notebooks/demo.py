"""
notebooks/demo.py — QCRS Demo Script
=====================================
Full walkthrough of the QCRS pipeline. Can also be run as a .ipynb
(copy cell-by-cell into Jupyter, or use `jupytext` to convert).

Run: python notebooks/demo.py
"""

import sys
import os

# MUST come before ANY qcrs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Problem Definition
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  QCRS Demo: Quantum-Cloud Resource Scheduler")
print("="*60)

print("DEBUG: imports starting...")

from qcrs.problem import SchedulingProblem, Job, Node, make_small_problem, make_medium_problem
from qcrs.classical_solver import GreedySolver, SimulatedAnnealingSolver, BruteForceSolver
from qcrs.qaoa_solver import QAOASolver
from qcrs.hybrid_pipeline import HybridScheduler, print_schedule

print("DEBUG: imports successful")

print("\n[1] Building scheduling problem...")
print("DEBUG: reached section 1")
problem = make_small_problem(seed=42)
print(problem.summary())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: QUBO formulation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Building QUBO matrix...")
Q = problem.build_qubo()
print(f"    QUBO shape: {Q.shape}")
print(f"    Min Q entry: {Q.min():.3f}")
print(f"    Max Q entry: {Q.max():.3f}")
print(f"    Sparsity (zeros): {(Q == 0).sum() / Q.size * 100:.1f}%")

Q_norm = (Q - Q.min()) / (Q.max() - Q.min() + 1e-9)
Q_img = (Q_norm * 255).astype(np.uint8)

# Convert to image
img = Image.fromarray(Q_img)

# Resize for visibility (optional)
img = img.resize((300, 300))

# Save
img.save("notebooks/qubo_matrix.png")

print("    Saved: notebooks/qubo_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Classical baselines
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Running classical solvers...")

x_greedy, m_greedy = GreedySolver(problem).solve()
print(f"    Greedy  → energy={m_greedy['energy']:.4f}, feasible={m_greedy['feasible']}")
print_schedule(problem, x_greedy, "Greedy")

x_sa, m_sa = SimulatedAnnealingSolver(problem, n_reads=500, seed=42).solve()
print(f"    SA      → energy={m_sa['energy']:.4f}, feasible={m_sa['feasible']}")

x_bf, m_bf = BruteForceSolver(problem).solve()
if x_bf is not None:
    print(f"    Brute   → energy={m_bf['energy']:.4f}, feasible={m_bf['feasible']} (OPTIMAL)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: QAOA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Running QAOA...")
try:
    solver = QAOASolver(
        problem,
        p_layers=2,
        n_shots=4096,
        backend="statevector",
        verbose=True,
        seed=42,
    )
    x_qaoa, m_qaoa = solver.solve(n_restarts=3)
    print(f"\n    QAOA    → energy={m_qaoa['energy']:.4f}, feasible={m_qaoa['feasible']}")
    print_schedule(problem, x_qaoa, "QAOA p=2")

    # ── Convergence plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Convergence
    axes[0].plot(solver.energy_history, color="#a78bfa", linewidth=1.5)
    axes[0].fill_between(range(len(solver.energy_history)),
                          solver.energy_history, alpha=0.15, color="#a78bfa")
    axes[0].axhline(m_qaoa["energy"], color="#f97316", linestyle="--",
                     label=f"Final: {m_qaoa['energy']:.3f}")
    if x_bf is not None:
        axes[0].axhline(m_bf["energy"], color="#22c55e", linestyle=":",
                         label=f"Optimal: {m_bf['energy']:.3f}")
    axes[0].set_xlabel("Optimizer iteration")
    axes[0].set_ylabel("⟨H_C⟩ expectation value")
    axes[0].set_title("QAOA Energy Convergence")
    axes[0].legend()

    # Bitstring distribution
    counts = solver._result_counts
    if counts:
        sorted_items = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
        bs_labels = [s[0] for s in sorted_items]
        bs_vals = [s[1] for s in sorted_items]
        Q_mat = problem.build_qubo()
        bs_energies = []
        for bs in bs_labels:
            xb = np.array([int(b) for b in reversed(bs)], dtype=int)
            if len(xb) == problem.n_vars:
                bs_energies.append(float(xb @ Q_mat @ xb))
            else:
                bs_energies.append(0.0)
        colors = ["#f97316" if e == min(bs_energies) else "#93c5fd" for e in bs_energies]
        axes[1].bar(range(len(bs_labels)), bs_vals, color=colors, alpha=0.85)
        axes[1].set_xticks(range(len(bs_labels)))
        axes[1].set_xticklabels(bs_labels, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Count")
        axes[1].set_title("Measurement Outcomes (QAOA sampling)")

    plt.tight_layout()
    plt.savefig("notebooks/qaoa_results.png", dpi=150, bbox_inches="tight")
    print("    Saved: notebooks/qaoa_results.png")
    plt.close()

    # ── Circuit diagram ───────────────────────────────────────────────────────
    print("\n[4b] QAOA Circuit:")
    solver.get_circuit(draw=True)

except ImportError as e:
    print(f"    [SKIP] Qiskit not installed: {e}")
    m_qaoa = None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Hybrid pipeline + comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Running hybrid pipeline comparison...")
try:
    scheduler = HybridScheduler(
        problem,
        qaoa_p_layers=2,
        qaoa_backend="statevector",
        verbose=True,
    )
    results = scheduler.run_comparison(include_brute_force=True)

    # ── Energy comparison bar chart ───────────────────────────────────────────
    method_names = []
    energies = []
    colors_bar = []
    color_map = {
        "greedy": "#94a3b8",
        "sa": "#60a5fa",
        "brute_force": "#22c55e",
        "qaoa": "#a78bfa",
    }
    for name in ["greedy", "sa", "brute_force", "qaoa"]:
        if name in results and "metrics" in results[name]:
            method_names.append(name.upper())
            energies.append(results[name]["metrics"]["energy"])
            colors_bar.append(color_map[name])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(method_names, energies, color=colors_bar, width=0.55, alpha=0.9)
    ax.set_ylabel("QUBO energy (lower = better)")
    ax.set_title("Solver Comparison — Cloud Scheduling Problem")
    for bar, e in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{e:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("notebooks/solver_comparison.png", dpi=150, bbox_inches="tight")
    print("    Saved: notebooks/solver_comparison.png")
    plt.close()

except ImportError:
    print("    [SKIP] Qiskit not available.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Scaling analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Scaling: problem size vs solver time...")
import time as _time

sizes = [(2, 2), (3, 2), (3, 3), (4, 3)]
greedy_times = []
sa_times = []

for (nj, nn) in sizes:
    p = make_medium_problem(n_jobs=nj, n_nodes=nn, seed=1)
    t0 = _time.time()
    GreedySolver(p).solve()
    greedy_times.append(_time.time() - t0)
    t0 = _time.time()
    SimulatedAnnealingSolver(p, n_reads=200).solve()
    sa_times.append(_time.time() - t0)

labels_scale = [f"{nj}×{nn}\n({nj*nn} q)" for nj, nn in sizes]
fig, ax = plt.subplots(figsize=(7, 3.5))
x_pos = np.arange(len(sizes))
ax.bar(x_pos - 0.2, greedy_times, width=0.35, label="Greedy", color="#94a3b8", alpha=0.9)
ax.bar(x_pos + 0.2, sa_times, width=0.35, label="SA", color="#60a5fa", alpha=0.9)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_scale)
ax.set_ylabel("Solve time (s)")
ax.set_title("Classical Solver Scaling (jobs × nodes)")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/scaling.png", dpi=150, bbox_inches="tight")
print("    Saved: notebooks/scaling.png")
plt.close()

print("\n" + "="*60)
print("  Demo complete.")
print("  Generated plots:")
print("    notebooks/qubo_matrix.png")
print("    notebooks/qaoa_results.png")
print("    notebooks/solver_comparison.png")
print("    notebooks/scaling.png")
print("="*60)
