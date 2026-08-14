"""
QCRS — Quantum-Cloud Resource Scheduler
=========================================
A real tool: define your own cloud scheduling problem, then watch
Greedy, Simulated Annealing, and QAOA solve it — with results rendered
as a physical server floor and a qubit-collapse visualization instead
of tables.

Run: streamlit run dashboard/app.py
"""

import sys
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

import json
import time
import subprocess

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from viz import rack_floor, qubit_collapse, PALETTE

from qcrs import (
    SchedulingProblem, Job, Node,
    GreedySolver, SimulatedAnnealingSolver, BruteForceSolver,
)

try:
    import qiskit  # noqa: F401
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

QAOA_WORKER = os.path.join(_THIS_DIR, "qaoa_worker.py")
QAOA_TIMEOUT_S = 120


def run_qaoa_subprocess(problem_payload):
    """
    Runs QAOA in an isolated subprocess. qiskit-aer's native backend can
    segfault under constrained containers -- a subprocess crash surfaces
    as a clean non-zero return code here instead of killing the app.
    Returns (x, metrics, counts) or raises RuntimeError with a readable message.
    """
    try:
        proc = subprocess.run(
            [sys.executable, QAOA_WORKER],
            input=json.dumps(problem_payload),
            capture_output=True, text=True, timeout=QAOA_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"QAOA didn't finish within {QAOA_TIMEOUT_S}s -- try fewer qubits, shots, or restarts.")

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no error output"]
        raise RuntimeError(f"QAOA process crashed (exit {proc.returncode}): {stderr_tail[0]}")

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("QAOA process returned unreadable output.")

    return result["x"], result["metrics"], result.get("counts")


# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QCRS — Quantum-Cloud Resource Scheduler",
    page_icon="\u269b",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.qcrs-hero {{
  padding: 6px 0 18px 0;
  border-bottom: 1px solid {PALETTE['border']};
  margin-bottom: 22px;
}}
.qcrs-hero h1 {{
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 30px;
  letter-spacing: -0.01em;
  margin-bottom: 4px;
  background: linear-gradient(90deg, {PALETTE['text']}, {PALETTE['violet']} 120%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.qcrs-hero p {{
  color: {PALETTE['muted']};
  font-size: 14.5px;
  margin: 0;
}}

.qcrs-card {{
  background: {PALETTE['panel']};
  border: 1px solid {PALETTE['border']};
  border-radius: 12px;
  padding: 16px 18px;
}}
.qcrs-eyebrow {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: {PALETTE['muted']};
  margin-bottom: 10px;
}}
.qcrs-metric-row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.qcrs-metric {{
  background: {PALETTE['panel']};
  border: 1px solid {PALETTE['border']};
  border-radius: 10px;
  padding: 12px 16px;
  min-width: 130px;
  flex: 1;
}}
.qcrs-metric .label {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  color: {PALETTE['muted']};
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
.qcrs-metric .value {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 24px;
  font-weight: 600;
  color: {PALETTE['text']};
  margin-top: 2px;
}}
.qcrs-note {{
  background: {PALETTE['amber_soft']};
  border: 1px solid rgba(245,166,35,0.4);
  border-radius: 10px;
  padding: 14px 16px;
  font-size: 13.5px;
  line-height: 1.55;
  color: {PALETTE['text']};
}}
.qcrs-note b {{ color: {PALETTE['amber']}; }}

.qcrs-result-card {{
  background: {PALETTE['panel']};
  border: 1px solid {PALETTE['border']};
  border-radius: 12px;
  padding: 14px 18px;
  margin-bottom: 8px;
}}
.qcrs-result-card.qcrs-best {{ border-color: {PALETTE['teal']}; }}
.qcrs-result-name {{
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600; font-size: 15px;
}}
.qcrs-pill {{
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  padding: 2px 8px;
  border-radius: 20px;
  margin-left: 8px;
}}
.qcrs-pill.ok {{ background: rgba(45,212,191,0.15); color: {PALETTE['teal']}; border: 1px solid rgba(45,212,191,0.4); }}
.qcrs-pill.bad {{ background: rgba(241,89,107,0.15); color: {PALETTE['red']}; border: 1px solid rgba(241,89,107,0.4); }}

section[data-testid="stSidebar"] {{ border-right: 1px solid {PALETTE['border']}; }}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button p {{
  text-align: center !important;
  width: 100%;
  margin: 0 !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="qcrs-hero">
  <h1>&#9883; Quantum-Cloud Resource Scheduler</h1>
  <p>Define a real scheduling problem. Watch Greedy, Simulated Annealing, and QAOA place every job on a rack &mdash; live.</p>
</div>
""", unsafe_allow_html=True)


# ── Session state: jobs & nodes the user is building ─────────────────────
def _default_jobs():
    return [
        {"label": "web-api", "cpu": 2.0, "mem": 4.0},
        {"label": "ml-train", "cpu": 4.0, "mem": 8.0},
        {"label": "db-backup", "cpu": 1.0, "mem": 2.0},
    ]


def _default_nodes():
    return [
        {"label": "node-A", "cpu_cap": 6.0, "mem_cap": 12.0, "cost": 1.0},
        {"label": "node-B", "cpu_cap": 6.0, "mem_cap": 12.0, "cost": 1.2},
    ]


if "jobs" not in st.session_state:
    st.session_state.jobs = _default_jobs()
if "nodes" not in st.session_state:
    st.session_state.nodes = _default_nodes()
if "results" not in st.session_state:
    st.session_state.results = {}


PRESETS = {
    "Small demo (3 jobs x 2 nodes)": (_default_jobs(), _default_nodes()),
    "Busy cluster (5 jobs x 3 nodes)": (
        [
            {"label": "web-api", "cpu": 2.0, "mem": 4.0},
            {"label": "ml-train", "cpu": 5.0, "mem": 10.0},
            {"label": "db-backup", "cpu": 1.0, "mem": 3.0},
            {"label": "cache", "cpu": 1.5, "mem": 6.0},
            {"label": "analytics", "cpu": 3.0, "mem": 5.0},
        ],
        [
            {"label": "node-A", "cpu_cap": 8.0, "mem_cap": 16.0, "cost": 1.0},
            {"label": "node-B", "cpu_cap": 6.0, "mem_cap": 14.0, "cost": 1.1},
            {"label": "node-C", "cpu_cap": 5.0, "mem_cap": 12.0, "cost": 0.9},
        ],
    ),
}


# ── Sidebar: build your own problem ───────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="qcrs-eyebrow">Load a starting point</div>', unsafe_allow_html=True)
    preset = st.selectbox("Preset", ["Keep current"] + list(PRESETS.keys()), label_visibility="collapsed")
    if preset != "Keep current" and st.button("Load preset", use_container_width=True):
        jobs, nodes = PRESETS[preset]
        st.session_state.jobs = [dict(j) for j in jobs]
        st.session_state.nodes = [dict(n) for n in nodes]
        st.session_state.results = {}
        st.rerun()

    st.divider()
    st.markdown('<div class="qcrs-eyebrow">Jobs (your workloads)</div>', unsafe_allow_html=True)
    for idx, job in enumerate(st.session_state.jobs):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
            job["label"] = c1.text_input("Name", job["label"], key=f"jl{idx}")
            if c2.button("✕", key=f"jd{idx}", use_container_width=True) and len(st.session_state.jobs) > 1:
                st.session_state.jobs.pop(idx)
                st.rerun()
            c3, c4 = st.columns(2)
            job["cpu"] = c3.number_input("CPU", 0.1, 64.0, float(job["cpu"]), step=0.5, key=f"jc{idx}")
            job["mem"] = c4.number_input("RAM (GB)", 0.1, 512.0, float(job["mem"]), step=0.5, key=f"jm{idx}")
    if st.button("+ Add job", use_container_width=True):
        n = len(st.session_state.jobs)
        st.session_state.jobs.append({"label": f"job-{n+1}", "cpu": 2.0, "mem": 4.0})
        st.rerun()

    st.divider()
    st.markdown('<div class="qcrs-eyebrow">Nodes (your servers)</div>', unsafe_allow_html=True)
    for idx, node in enumerate(st.session_state.nodes):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
            node["label"] = c1.text_input("Name", node["label"], key=f"nl{idx}")
            if c2.button("✕", key=f"nd{idx}", use_container_width=True) and len(st.session_state.nodes) > 1:
                st.session_state.nodes.pop(idx)
                st.rerun()
            c3, c4, c5 = st.columns(3)
            node["cpu_cap"] = c3.number_input("CPU cap", 0.1, 256.0, float(node["cpu_cap"]), step=0.5, key=f"nc{idx}")
            node["mem_cap"] = c4.number_input("RAM cap", 0.1, 2048.0, float(node["mem_cap"]), step=0.5, key=f"nm{idx}")
            node["cost"] = c5.number_input("Cost", 0.1, 20.0, float(node["cost"]), step=0.1, key=f"ncost{idx}")
    if st.button("+ Add node", use_container_width=True):
        n = len(st.session_state.nodes)
        st.session_state.nodes.append({"label": f"node-{n+1}", "cpu_cap": 6.0, "mem_cap": 12.0, "cost": 1.0})
        st.rerun()

    st.divider()
    st.markdown('<div class="qcrs-eyebrow">QAOA settings</div>', unsafe_allow_html=True)
    p_layers = st.slider("Circuit depth (p)", 1, 4, 2)
    n_shots = st.select_slider("Shots", [512, 1024, 2048, 4096, 8192], value=2048)
    n_restarts = st.slider("Optimizer restarts", 1, 5, 3)

    st.divider()
    st.markdown('<div class="qcrs-eyebrow">Penalties</div>', unsafe_allow_html=True)
    auto_penalty = st.checkbox("Auto-calibrate (recommended)", value=True)
    if auto_penalty:
        pen_assign, pen_cpu, pen_mem = 0.0, 0.0, 0.0
        st.caption("Penalties are computed automatically so illegal placements always cost more than any legal one.")
    else:
        st.caption("⚠ Setting these too low can let the solver skip a job entirely instead of placing it.")
        pen_assign = st.slider("Assignment", 5.0, 40.0, 20.0)
        pen_cpu = st.slider("CPU capacity", 5.0, 30.0, 15.0)
        pen_mem = st.slider("Memory capacity", 5.0, 30.0, 15.0)


# ── Build the problem from session state ──────────────────────────────────
def build_problem():
    jobs = [
        Job(i, cpu=float(j["cpu"]), mem=float(j["mem"]), label=j["label"])
        for i, j in enumerate(st.session_state.jobs)
    ]
    nodes = [
        Node(j, cpu_cap=float(n["cpu_cap"]), mem_cap=float(n["mem_cap"]),
             cost_per_unit=float(n["cost"]), label=n["label"])
        for j, n in enumerate(st.session_state.nodes)
    ]
    return SchedulingProblem(
        jobs, nodes,
        penalty_assign=pen_assign, penalty_cpu=pen_cpu, penalty_mem=pen_mem,
    )


try:
    problem = build_problem()
except Exception as e:
    st.error(f"Problem definition error: {e}")
    st.stop()

if problem.n_vars > 16:
    st.warning(
        f"{problem.n_vars} binary variables — QAOA statevector simulation gets slow fast here. "
        "Keep it under ~16 (e.g. 4 jobs x 4 nodes) for QAOA to stay responsive."
    )

# ── Problem summary ────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
for col, label, value in [
    (m1, "Jobs", problem.n_jobs), (m2, "Nodes", problem.n_nodes),
    (m3, "Qubits needed", problem.n_vars), (m4, "QUBO size", f"{problem.n_vars}\u00d7{problem.n_vars}"),
]:
    col.markdown(f'<div class="qcrs-metric"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.write("")
tab_qubo, tab_help = st.tabs(["QUBO matrix", "How this works"])
with tab_qubo:
    Q = problem.build_qubo()
    labels = [f"{st.session_state.jobs[i//problem.n_nodes]['label']}\u2192{st.session_state.nodes[i%problem.n_nodes]['label']}"
              for i in range(problem.n_vars)]
    fig = go.Figure(data=go.Heatmap(
        z=Q, x=labels, y=labels, colorscale="RdBu", reversescale=True,
        zmid=0, colorbar=dict(title="", thickness=12),
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["muted"], size=10, family="IBM Plex Mono"),
        xaxis=dict(tickangle=45), yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each cell is how strongly two (job\u2192node) decisions push against or reinforce each other. Diagonal = individual decision cost.")

with tab_help:
    st.markdown(f"""
Every job/node pair becomes one binary variable (a qubit): `1` means that job runs on that node.
The problem is encoded as a QUBO matrix &mdash; a set of penalties that make illegal placements
(unassigned jobs, overloaded nodes) expensive, and a cost term that prefers cheap, balanced placements.

Three solvers then search for the lowest-cost valid placement:
- **Greedy** places each job on the first node with room &mdash; fast, but easily leaves one node overloaded and another empty.
- **Simulated Annealing** randomly reshuffles placements, gradually accepting fewer bad moves &mdash; a strong classical baseline.
- **QAOA** encodes the QUBO as a quantum circuit, puts every qubit into superposition, and repeatedly measures &mdash;
  each measurement collapses every qubit to a 0 or 1, i.e. one full placement. The lowest-energy placement seen across
  samples is returned.
""")

st.divider()

# ── Run solvers ─────────────────────────────────────────────────────────
st.markdown('<div class="qcrs-eyebrow">Run solvers</div>', unsafe_allow_html=True)
rc1, rc2, rc3, rc4 = st.columns(4)
run_greedy = rc1.button("Greedy", use_container_width=True)
run_sa = rc2.button("Simulated Annealing", use_container_width=True)
run_qaoa = rc3.button("QAOA", use_container_width=True, disabled=not QISKIT_AVAILABLE)
run_all = rc4.button("Run all three", use_container_width=True, type="primary")

if not QISKIT_AVAILABLE:
    st.info("Qiskit isn't installed in this environment, so QAOA is unavailable here — Greedy and Simulated Annealing still run normally.")


def _run(name, fn):
    t0 = time.time()
    x, metrics = fn()
    metrics.setdefault("solve_time_s", time.time() - t0)
    counts = None
    return name, x, metrics


def run_greedy_fn():
    return GreedySolver(problem).solve()


def run_sa_fn():
    n_reads = max(400, problem.n_vars * 60)
    return SimulatedAnnealingSolver(problem, n_reads=n_reads).solve()


def run_qaoa_fn():
    payload = {
        "jobs": st.session_state.jobs,
        "nodes": st.session_state.nodes,
        "penalty_assign": pen_assign, "penalty_cpu": pen_cpu, "penalty_mem": pen_mem,
        "p_layers": p_layers, "n_shots": n_shots, "n_restarts": n_restarts,
        "backend": "statevector",
    }
    x, metrics, counts = run_qaoa_subprocess(payload)
    metrics["_counts"] = counts
    return x, metrics


jobs_to_run = []
if run_greedy or run_all:
    jobs_to_run.append(("Greedy", run_greedy_fn))
if run_sa or run_all:
    jobs_to_run.append(("Simulated Annealing", run_sa_fn))
if (run_qaoa or run_all) and QISKIT_AVAILABLE:
    jobs_to_run.append(("QAOA", run_qaoa_fn))

if jobs_to_run:
    prog = st.progress(0.0, text="Solving...")
    for i, (name, fn) in enumerate(jobs_to_run):
        with st.spinner(f"Running {name}..."):
            try:
                _, x, metrics = _run(name, fn)
                st.session_state.results[name] = {"x": np.asarray(x, dtype=int).tolist(), "metrics": metrics}
            except Exception as e:
                st.error(f"{name} failed: {e}")
        prog.progress((i + 1) / len(jobs_to_run), text=f"{name} done")
    prog.empty()

# ── Results ──────────────────────────────────────────────────────────────
if st.session_state.results:
    st.divider()
    st.markdown('<div class="qcrs-eyebrow">Results</div>', unsafe_allow_html=True)

    energies = {name: r["metrics"].get("energy", float("inf")) for name, r in st.session_state.results.items()}
    best_name = min(energies, key=energies.get)

    cols = st.columns(len(st.session_state.results))
    for col, (name, res) in zip(cols, st.session_state.results.items()):
        m = res["metrics"]
        is_best = name == best_name
        feasible = m.get("feasible", False)
        pill = '<span class="qcrs-pill ok">feasible</span>' if feasible else '<span class="qcrs-pill bad">infeasible</span>'
        col.markdown(f"""
<div class="qcrs-result-card {'qcrs-best' if is_best else ''}">
  <div class="qcrs-result-name">{name}{pill}</div>
  <div style="font-family:'IBM Plex Mono',monospace; font-size:12.5px; color:{PALETTE['muted']}; margin-top:6px;">
    energy {m.get('energy', float('nan')):.2f} &middot; {m.get('solve_time_s', 0):.3f}s
  </div>
</div>
""", unsafe_allow_html=True)

    st.write("")
    selected = st.selectbox("Inspect a result on the server floor:", list(st.session_state.results.keys()))
    res = st.session_state.results[selected]
    x = res["x"]
    m = res["metrics"]

    job_labels = [j["label"] for j in st.session_state.jobs]
    job_cpu = [j["cpu"] for j in st.session_state.jobs]
    job_mem = [j["mem"] for j in st.session_state.jobs]
    node_labels = [n["label"] for n in st.session_state.nodes]
    node_cpu_cap = [n["cpu_cap"] for n in st.session_state.nodes]
    node_mem_cap = [n["mem_cap"] for n in st.session_state.nodes]

    if len(x) == problem.n_vars:
        alloc = [[int(x[problem.var_index(i, j)]) for j in range(problem.n_nodes)] for i in range(problem.n_jobs)]
    else:
        alloc = [[0] * problem.n_nodes for _ in range(problem.n_jobs)]

    tab_floor, tab_qc, tab_conv, tab_export = st.tabs(["Server floor", "Qubit collapse", "Convergence", "Export"])

    with tab_floor:
        html, h = rack_floor(node_labels, node_cpu_cap, node_mem_cap, job_labels, job_cpu, job_mem,
                              alloc, title=f"{selected} \u2014 job placement")
        components.html(html, height=h, scrolling=False)
        violations = problem.constraint_violations(np.array(x)) if len(x) == problem.n_vars else {}
        bad = any(violations.get(k) for k in ("unassigned_jobs", "cpu_overload", "mem_overload"))
        if bad:
            st.error(f"Constraint violations: {violations}")
        else:
            st.success("All constraints satisfied.")

    with tab_qc:
        if selected == "QAOA" and len(x) == problem.n_vars:
            var_labels = [f"{job_labels[i]}\u2192{node_labels[j]}" for i in range(problem.n_jobs) for j in range(problem.n_nodes)]
            bitstring = "".join(str(b) for b in x)
            html, h = qubit_collapse(bitstring, var_labels, node_labels, problem.n_nodes)
            components.html(html, height=h, scrolling=False)
            st.caption(
                "Each qubit starts in superposition (shimmering) and collapses to 0 or 1 on measurement \u2014 "
                "that's one full candidate placement. QAOA repeats this thousands of times and keeps the lowest-energy result."
            )
        else:
            st.info("Qubit collapse view is only available for QAOA results.")

    with tab_conv:
        history = m.get("energy_convergence") or m.get("energy_history")
        if history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history, mode="lines", line=dict(color=PALETTE["violet"], width=2),
                fill="tozeroy", fillcolor="rgba(124,92,252,0.12)",
            ))
            fig.add_hline(y=m.get("energy", 0), line_dash="dash", line_color=PALETTE["amber"],
                           annotation_text=f"final: {m.get('energy', 0):.2f}")
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["muted"], family="IBM Plex Mono", size=11),
                xaxis_title="iteration", yaxis_title="energy",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No convergence trace for this solver (Greedy has no iterative optimization).")

    with tab_export:
        assignment = []
        for i in range(problem.n_jobs):
            for j in range(problem.n_nodes):
                if len(x) == problem.n_vars and x[problem.var_index(i, j)] == 1:
                    assignment.append({"job": job_labels[i], "node": node_labels[j]})
        export = {"method": selected, "energy": m.get("energy"), "feasible": m.get("feasible"), "assignment": assignment}
        st.download_button(
            "Download allocation (JSON)", data=json.dumps(export, indent=2),
            file_name=f"qcrs_{selected.lower().replace(' ', '_')}_allocation.json",
            mime="application/json", use_container_width=True,
        )
        st.json(export)

    # ── Honest limits panel ─────────────────────────────────────────────
    if "QAOA" in st.session_state.results:
        st.write("")
        st.markdown(f"""
<div class="qcrs-note">
<b>What QAOA actually demonstrated here:</b> on this problem ({problem.n_vars} qubits), QAOA's result matches
the true optimum \u2014 confirmed by comparing against exact brute-force search. That's a correctness check, not a
speed advantage: at this size, brute force finds the same answer in milliseconds. This dashboard simulates
the quantum circuit on a classical computer, and that simulation is itself exponential in qubit count, so it
hits the same wall brute force does past ~20&ndash;25 qubits. QAOA's real appeal only shows up on real quantum
hardware at problem sizes classical computers can no longer brute-force or simulate.
</div>
""", unsafe_allow_html=True)

st.write("")
st.caption("QCRS \u2014 Quantum-Cloud Resource Scheduler \u00b7 by Nitanshu Tak")
