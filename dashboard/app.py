"""
dashboard/app.py — QCRS Interactive Dashboard
==============================================
Run with: streamlit run dashboard/app.py

Features:
  - Configure jobs/nodes interactively
  - Run QAOA, SA, Greedy with one click
  - Visualize: allocation heatmap, energy convergence, circuit diagram,
    bitstring distribution, resource utilization bars
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import json
import requests
import time
from typing import Optional

API_URL = "https://quantum-cloud.onrender.com"

# Page config
st.set_page_config(
    page_title="QCRS — Quantum-Cloud Resource Scheduler",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { font-family: 'Inter', sans-serif; }
.metric-card {
    background: #1e1e2e; border-radius: 12px; padding: 16px 20px;
    border: 1px solid #2a2a3e;
}
.section-header {
    font-size: 13px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #888; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚛  Quantum-Cloud Resource Scheduler (QCRS)")
st.caption("Hybrid QAOA + classical cloud workload allocation — powered by Qiskit & AWS Braket")

# ── Sidebar: problem configuration ───────────────────────────────────────────
with st.sidebar:
    st.header("Problem Setup")

    st.subheader("Preset")
    preset = st.selectbox(
        "Load preset",
        ["Custom", "Small (3 jobs × 2 nodes)", "Medium (4 jobs × 3 nodes)"],
        index=1,
    )

    if preset == "Small (3 jobs × 2 nodes)":
        n_jobs, n_nodes = 3, 2
    elif preset == "Medium (4 jobs × 3 nodes)":
        n_jobs, n_nodes = 4, 3
    else:
        n_jobs = st.slider("Number of jobs", 2, 5, 3)
        n_nodes = st.slider("Number of nodes", 2, 4, 2)

    st.subheader("QAOA Settings")
    p_layers = st.slider("QAOA depth (p layers)", 1, 4, 2)
    n_shots = st.select_slider("Shots", options=[512, 1024, 2048, 4096, 8192], value=4096)
    backend = st.radio("Backend", ["statevector (exact)", "qasm (shot-based)"], index=0)
    backend_key = "statevector" if "statevector" in backend else "qasm"
    n_restarts = st.slider("Optimizer restarts", 1, 5, 3)

    st.subheader("Penalties")
    pen_assign = st.slider("Assignment penalty", 1.0, 20.0, 10.0)
    pen_cpu = st.slider("CPU capacity penalty", 1.0, 15.0, 5.0)
    pen_mem = st.slider("Memory capacity penalty", 1.0, 15.0, 5.0)

    seed = st.number_input("Random seed", min_value=0, max_value=999, value=42)

# ── Import after sidebar (allows import error to surface cleanly) ─────────────
try:
    from qcrs import (
        SchedulingProblem, Job, Node,
        make_small_problem, make_medium_problem,
        GreedySolver, SimulatedAnnealingSolver,
        QAOASolver, HybridScheduler, print_schedule
    )
    QISKIT_OK = True
except ImportError as e:
    st.error(f"Import error: {e}. Run `pip install -r requirements.txt`")
    st.stop()

# ── Build problem ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_problem(n_jobs, n_nodes, seed, pen_assign, pen_cpu, pen_mem):
    rng = np.random.default_rng(seed)
    job_labels = ["web-api", "ml-train", "db-backup", "cache", "analytics", "queue"]
    node_labels = ["node-A", "node-B", "node-C", "node-D"]
    jobs = [
        Job(i,
            cpu=float(rng.uniform(1, 5)),
            mem=float(rng.uniform(2, 12)),
            label=job_labels[i % len(job_labels)])
        for i in range(n_jobs)
    ]
    nodes = [
        Node(j,
             cpu_cap=float(rng.uniform(8, 14)),
             mem_cap=float(rng.uniform(14, 28)),
             cost_per_unit=float(rng.uniform(0.8, 1.4)),
             label=node_labels[j % len(node_labels)])
        for j in range(n_nodes)
    ]
    return SchedulingProblem(
        jobs, nodes,
        penalty_assign=pen_assign,
        penalty_cpu=pen_cpu,
        penalty_mem=pen_mem,
    )

problem = build_problem(n_jobs, n_nodes, seed, pen_assign, pen_cpu, pen_mem)

# ── Problem info ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jobs", problem.n_jobs)
col2.metric("Nodes", problem.n_nodes)
col3.metric("Binary vars (qubits)", problem.n_vars)
col4.metric("QUBO matrix", f"{problem.n_vars}×{problem.n_vars}")

st.divider()

# ── Problem tables ────────────────────────────────────────────────────────────
tab_prob, tab_qubo = st.tabs(["Problem Definition", "QUBO Matrix"])

with tab_prob:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Jobs (workloads)**")
        job_df = pd.DataFrame([
            {"Job": j.label, "CPU req.": j.cpu, "RAM (GB)": j.mem}
            for j in problem.jobs
        ])
        st.dataframe(job_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Nodes (compute resources)**")
        node_df = pd.DataFrame([
            {"Node": n.label, "CPU cap.": n.cpu_cap, "RAM cap. (GB)": n.mem_cap,
             "Cost/unit": n.cost_per_unit}
            for n in problem.nodes
        ])
        st.dataframe(node_df, use_container_width=True, hide_index=True)

with tab_qubo:
    Q = problem.build_qubo()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Q, cmap="RdBu_r", aspect="auto")
    ax.set_title("QUBO Matrix Q", fontsize=12)
    ax.set_xlabel("Variable index")
    ax.set_ylabel("Variable index")
    plt.colorbar(im, ax=ax, fraction=0.04)
    labels = [f"x{i//problem.n_nodes},{i%problem.n_nodes}" for i in range(problem.n_vars)]
    ax.set_xticks(range(problem.n_vars))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(problem.n_vars))
    ax.set_yticklabels(labels, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"x_{{i,j}} = 1 if job i is assigned to node j. "
        f"Diagonal = linear terms, off-diagonal = quadratic interactions."
    )

st.divider()

# ── Run solvers ───────────────────────────────────────────────────────────────
st.subheader("Run Solvers")

col_run1, col_run2, col_run3, col_run4 = st.columns(4)
run_greedy = col_run1.button("▶ Greedy", use_container_width=True)
run_sa = col_run2.button("▶ Sim. Annealing", use_container_width=True)
run_qaoa = col_run3.button("▶ QAOA", use_container_width=True, type="primary")
run_all = col_run4.button("▶ Full Benchmark", use_container_width=True)

# Session state for results
if "results" not in st.session_state:
    st.session_state.results = {}

# ── Run logic ─────────────────────────────────────────────────────────────────
def run_solver(name, fn):
    with st.spinner(f"Running {name}..."):
        t0 = time.time()
        x, m = fn()
        m["wall_time"] = time.time() - t0
        st.session_state.results[name] = {"x": x, "metrics": m}
    st.success(f"{name} done! Energy: {m['energy']:.4f}, Feasible: {m['feasible']}")

if run_greedy:
    run_solver("Greedy", lambda: GreedySolver(problem).solve())

if run_sa:
    run_solver("SA", lambda: SimulatedAnnealingSolver(problem, n_reads=500).solve())

if run_qaoa:
    with st.spinner("Running QAOA via backend..."):
        res = requests.get(f"{API_URL}/qaoa")
        data = res.json()

        st.session_state.results["QAOA"] = {
            "x": data.get("x", []),
            "metrics": data
        }

    st.success(f"QAOA done! Energy: {data.get('energy')}, Feasible: {data.get('feasible')}")

if run_all:
    with st.spinner("Running full benchmark (Greedy + SA + QAOA)..."):
        res = requests.get(f"{API_URL}/compare")
        comparison = res.json()
    for name in ["greedy", "sa", "brute_force", "qaoa"]:
        if name in comparison:
            st.session_state.results[name.upper()] = {
                "x": comparison[name]["x"],
                "metrics": comparison[name]["metrics"],
            }
    st.success("Full benchmark complete!")

# ── Results visualisation ─────────────────────────────────────────────────────
if st.session_state.results:
    st.divider()
    st.subheader("Results")

    # Summary comparison table
    rows = []
    for name, res in st.session_state.results.items():
        m = res["metrics"]
        rows.append({
            "Method": name,
            "Energy": f"{m.get('energy', float('nan')):.4f}",
            "Feasible": "✓" if m.get("feasible") else "✗",
            "Time (s)": f"{m.get('solve_time_s', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Tabs per solver ─────────────────────────────────────────────────────
    if st.session_state.results:
        selected = st.selectbox("Inspect result:", list(st.session_state.results.keys()))
        res = st.session_state.results[selected]
        x = res["x"]
        m = res["metrics"]

        tab_alloc, tab_util, tab_conv, tab_dist = st.tabs([
            "Allocation", "Resource Utilization", "Convergence", "Bitstring Distribution"
        ])

        # ── Allocation heatmap ───────────────────────────────────────────────
        with tab_alloc:
            st.markdown("**Job → Node assignment matrix**")
            alloc = np.zeros((problem.n_jobs, problem.n_nodes))
            for i in range(problem.n_jobs):
                for j in range(problem.n_nodes):
                    alloc[i, j] = x[problem.var_index(i, j)]

            fig, ax = plt.subplots(figsize=(max(4, problem.n_nodes * 1.2), max(3, problem.n_jobs)))
            im = ax.imshow(alloc, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(problem.n_nodes))
            ax.set_xticklabels([n.label for n in problem.nodes])
            ax.set_yticks(range(problem.n_jobs))
            ax.set_yticklabels([j.label for j in problem.jobs])
            ax.set_xlabel("Node")
            ax.set_ylabel("Job")
            ax.set_title(f"Assignment Matrix — {selected}")
            for i in range(problem.n_jobs):
                for j in range(problem.n_nodes):
                    ax.text(j, i, "●" if alloc[i, j] else "○",
                            ha="center", va="center", fontsize=16,
                            color="white" if alloc[i, j] else "#ccc")
            fig.tight_layout()
            st.pyplot(fig)

            violations = problem.constraint_violations(x)
            if violations["unassigned_jobs"]:
                st.error(f"Unassigned: {violations['unassigned_jobs']}")
            if violations["cpu_overload"]:
                st.error(f"CPU overload: {violations['cpu_overload']}")
            if violations["mem_overload"]:
                st.error(f"Memory overload: {violations['mem_overload']}")
            if not any(violations.values()):
                st.success("All constraints satisfied ✓")

        # ── Resource utilization ─────────────────────────────────────────────
        with tab_util:
            fig, axes = plt.subplots(1, 2, figsize=(10, max(3, problem.n_nodes * 0.8)))
            node_labels_list = [n.label for n in problem.nodes]

            cpu_used = []
            mem_used = []
            cpu_caps = [n.cpu_cap for n in problem.nodes]
            mem_caps = [n.mem_cap for n in problem.nodes]

            for j, node in enumerate(problem.nodes):
                cpu_used.append(sum(
                    problem.jobs[i].cpu * x[problem.var_index(i, j)]
                    for i in range(problem.n_jobs)
                ))
                mem_used.append(sum(
                    problem.jobs[i].mem * x[problem.var_index(i, j)]
                    for i in range(problem.n_jobs)
                ))

            y = range(problem.n_nodes)
            axes[0].barh(y, cpu_used, color="#5b8ff9", alpha=0.8, label="Used")
            axes[0].barh(y, cpu_caps, color="#dde4f5", alpha=0.5, label="Capacity")
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(node_labels_list)
            axes[0].set_xlabel("CPU cores")
            axes[0].set_title("CPU Utilization")
            axes[0].legend(fontsize=9)

            axes[1].barh(y, mem_used, color="#61c79a", alpha=0.8, label="Used")
            axes[1].barh(y, mem_caps, color="#d5edd5", alpha=0.5, label="Capacity")
            axes[1].set_yticks(y)
            axes[1].set_yticklabels(node_labels_list)
            axes[1].set_xlabel("RAM (GB)")
            axes[1].set_title("Memory Utilization")
            axes[1].legend(fontsize=9)

            fig.suptitle(f"Resource Utilization — {selected}", fontsize=12)
            fig.tight_layout()
            st.pyplot(fig)

        # ── Convergence ──────────────────────────────────────────────────────
        with tab_conv:
            history = m.get("energy_convergence") or m.get("energy_history")
            if history:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(history, color="#a78bfa", linewidth=1.5, alpha=0.9)
                ax.fill_between(range(len(history)), history,
                                alpha=0.15, color="#a78bfa")
                ax.set_xlabel("Optimizer iteration")
                ax.set_ylabel("Expectation ⟨H_C⟩")
                ax.set_title(f"Energy Convergence — {selected}")
                ax.axhline(m["energy"], color="#f97316", linestyle="--",
                           linewidth=1, label=f"Final: {m['energy']:.3f}")
                ax.legend(fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Convergence history not available for this solver.")

        # ── Bitstring distribution ───────────────────────────────────────────
        with tab_dist:
            counts = m.get("counts") or (res.get("solver") and res["solver"]._result_counts)
            if counts:
                sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])[:20]
                labels_bs = [s[0] for s in sorted_counts]
                values_bs = [s[1] for s in sorted_counts]

                Q = problem.build_qubo()
                energies_bs = []
                for bs in labels_bs:
                    xb = np.array([int(b) for b in reversed(bs)], dtype=int)
                    if len(xb) == problem.n_vars:
                        energies_bs.append(float(xb @ Q @ xb))
                    else:
                        energies_bs.append(0.0)

                colors = ["#f97316" if e == min(energies_bs) else "#93c5fd" for e in energies_bs]

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                ax1.bar(range(len(labels_bs)), values_bs, color=colors, alpha=0.85)
                ax1.set_ylabel("Count")
                ax1.set_title(f"Top-20 measurement outcomes — {selected}")

                ax2.bar(range(len(labels_bs)), energies_bs, color=colors, alpha=0.85)
                ax2.set_ylabel("QUBO energy")
                ax2.set_xlabel("Bitstring rank")
                ax2.set_xticks(range(len(labels_bs)))
                ax2.set_xticklabels(labels_bs, rotation=45, ha="right", fontsize=7)

                legend_patches = [
                    mpatches.Patch(color="#f97316", label="Lowest energy (selected)"),
                    mpatches.Patch(color="#93c5fd", label="Other outcomes"),
                ]
                ax1.legend(handles=legend_patches, fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Bitstring distribution only available for QAOA results.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "QCRS — Quantum-Cloud Resource Scheduler | "
    "By: Nitanshu Tak"
)
