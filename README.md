<div align="center">

# ⚛ QCRS
### Quantum-Cloud Resource Scheduler

**A cloud job-scheduling optimizer that solves the same problem three ways — classical greedy, classical metaheuristic, and simulated quantum — and renders the result as a live, animated server floor instead of a table.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?style=flat-square&logo=qiskit&logoColor=white)](https://qiskit.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-black?style=flat-square)](#license)

[**▶ Live Demo**](https://quantum-cloud-resource-scheduler.streamlit.app/) · [Full Technical Guide](https://github.com/Nitanshu715/Quantum-Cloud/blob/main/QCRS_Repo_Documentation.docx) · [Report a bug](../../issues)

</div>

---

## Table of Contents

- [What is this?](#what-is-this)
- [Why it exists](#why-it-exists)
- [The three solvers](#the-three-solvers)
- [What it looks like](#what-it-looks-like)
- [Quick start](#quick-start)
- [Using it](#using-it)
- [How it actually works](#how-it-actually-works)
- [The QUBO formulation](#the-qubo-formulation)
- [The QAOA circuit](#the-qaoa-circuit)
- [Project structure](#project-structure)
- [Engineering decisions worth knowing](#engineering-decisions-worth-knowing)
- [An honest note on "quantum"](#an-honest-note-on-quantum)
- [Tech stack](#tech-stack)
- [Deployment](#deployment)
- [License](#license)

---

## What is this?

Cloud platforms constantly answer one question: **given a set of jobs that each need CPU and RAM, and a set of servers with limited capacity and different costs — which job runs on which server?**

Get it wrong and one server sits overloaded while another idles. This is a real, well-studied combinatorial optimization problem (a multi-resource, cost-weighted generalization of bin packing), and it's genuinely **NP-hard** — there's no known algorithm that finds the guaranteed-best answer in a time that scales cleanly with problem size.

QCRS lets you define this problem with your own numbers — real jobs, real servers — and then solves it three different ways so the differences are directly visible, not just asserted.

---

## Why it exists

Most "quantum computing" demos either overclaim ("quantum solves this instantly!") or under-explain ("trust me, it's quantum"). This project does neither. It:

- Solves a problem that's actually representative of real infrastructure work, not a toy puzzle invented to make quantum computing look good.
- Runs the exact same mathematical objective through three fundamentally different solving philosophies, so you can watch — not just read about — where each one wins and loses.
- States plainly, in the interface itself, exactly what the quantum solver did and did not prove at this scale (see [An honest note on "quantum"](#an-honest-note-on-quantum)).

---

## The three solvers

<table>
<tr>
<td width="33%" valign="top">

### 🟢 Greedy
**First-fit placement.**

Walks through jobs one at a time and places each on the first server with enough remaining room. Instant — effectively O(n log n) — but has no lookahead, so it routinely saturates the first server it touches while later servers sit underused.

Included specifically as a baseline: it's what "not really solving the problem" looks like, so the value of the other two is visible rather than assumed.

</td>
<td width="33%" valign="top">

### 🔵 Simulated Annealing
**Classical metaheuristic.**

Starts from a random placement and repeatedly proposes small changes, accepting improvements always and accepting worse moves with a probability that shrinks over the run (the Metropolis criterion, borrowed from metallurgical annealing). This lets it escape bad local decisions early and settle into a strong solution late.

This is a legitimate, industry-standard optimizer — not a strawman built to lose to the quantum solver.

</td>
<td width="33%" valign="top">

### 🟣 QAOA
**Simulated quantum circuit.**

The Quantum Approximate Optimization Algorithm. Every decision variable becomes a qubit, all qubits start in superposition (every possible assignment "at once"), and a parameterized circuit — tuned by a classical optimizer (COBYLA) — biases that superposition toward low-cost assignments before measurement collapses it into one concrete answer.

Run via Qiskit's `AerSimulator`, in an isolated subprocess (see below).

</td>
</tr>
</table>

All three solvers minimize the **exact same matrix** — they only differ in *how* they search. That shared objective is what makes comparing them meaningful rather than apples-to-oranges.

---

## What it looks like

Results render as physical, animated visuals instead of tables and static charts:

- 🖥️ **Server Floor** — every node renders as a physical rack; jobs animate into place with real-time CPU/RAM utilization bars, and an overloaded rack visibly flags itself.
- ⚛ **Qubit Collapse** — QAOA's qubits render as spheres shimmering in superposition, then settling into a solid violet (1) or amber (0) on measurement — the one visual in the app that's actually depicting quantum behavior, not decorating a chart with quantum-themed colors.
- 📉 **Convergence** — energy dropping across iterations for Simulated Annealing and QAOA, rendered as an interactive Plotly trace.
- 🧮 **QUBO Matrix** — the literal cost/penalty matrix being optimized, as an interactive heatmap, so the math isn't hidden behind the visuals.
- 📤 **Export** — the winning job→server assignment, downloadable as JSON.

Both animated visuals are hand-built HTML/CSS/JS components (`dashboard/viz.py`), because Streamlit's native chart widgets can't do spatial, physical animation.

---

## Quick start

```bash
git clone https://github.com/Nitanshu715/Quantum-Cloud.git
cd Quantum-Cloud

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

---

## Using it

1. **Define your problem** in the sidebar — add, edit, or remove jobs (name, CPU, RAM) and servers (name, CPU capacity, RAM capacity, cost per unit), or load one of the built-in presets to start from.
2. **Inspect the encoding** — the metrics row shows exactly how many qubits your problem needs (`jobs × nodes`), and the QUBO Matrix tab shows the real cost/penalty matrix being built from your numbers.
3. **Run solvers** — click any individual solver, or **Run all three** to compare them side by side. QAOA takes noticeably longer (10–30 seconds) because it's running an actual circuit simulation, not a shortcut.
4. **Compare results** — each solver's card shows its final energy (lower = better) and whether the result is feasible. The lowest-energy feasible result is highlighted.
5. **Inspect any result** — pick it from the dropdown, then open the Server Floor to watch the physical placement, or Qubit Collapse (QAOA only) to watch the quantum measurement resolve.
6. **Export** — download the winning allocation as JSON from the Export tab.

---

## How it actually works

### The QUBO formulation

Every `(job, node)` pair becomes one binary variable — one **qubit**:

```
x_ij ∈ {0, 1}      x_ij = 1  means job i runs on node j
```

These are assembled into a single matrix **Q**, and the entire problem becomes:

```
minimize   E(x) = xᵀ Q x
```

`Q` is built from three additive components:

**1. Assignment constraint** — every job must be placed on exactly one node:

```
A · (Σⱼ x_ij − 1)²
```

**2. Capacity constraints** — no node's CPU or RAM may be exceeded:

```
B · (Σᵢ cpuᵢ · x_ij / cap_j)²        (and the equivalent for memory)
```

**3. Cost objective** — the thing actually being minimized once feasibility is satisfied:

```
Σᵢⱼ  cost_j · (cpuᵢ + 0.1 · memᵢ) · x_ij
```

The penalty weights **A** and **B** are auto-calibrated from the problem's own numbers so that violating a constraint always costs strictly more than any possible saving — this project's own solver comparisons (against exact brute-force search) are what surfaced and fixed a real bug where an under-calibrated capacity penalty could make skipping a large job look "optimal."

**→ Full derivation, every term expanded algebraically, and a completely worked numeric example (real numbers, every matrix entry computed by hand) are in [`QCRS_Complete_Technical_Guide.md`](./QCRS_Complete_Technical_Guide.md).**

### The QAOA circuit

1. **Superposition** — every qubit starts at `|0⟩`, then a Hadamard gate puts the whole system into an equal superposition of all `2ⁿ` possible assignments simultaneously.
2. **Cost unitary** (`e^(−iγH_C)`) — phase rotations built directly from the QUBO matrix `Q`, implemented as RZ gates for single-variable terms and CNOT–RZ–CNOT chains for two-variable interaction terms.
3. **Mixer unitary** (`e^(−iβH_B)`) — spreads amplitude across candidate assignments so cost-unitary phases can interfere constructively (amplifying good assignments) and destructively (suppressing bad ones).
4. **Repeat for `p` layers** — a tunable circuit depth; more layers generally means better solution quality at the cost of a deeper circuit.
5. **Classical tuning** — the rotation angles `(γ, β)` aren't fixed; COBYLA (a derivative-free classical optimizer) iteratively adjusts them to minimize the measured expectation value of the cost Hamiltonian.
6. **Measurement** — collapses the superposition into one concrete bitstring per shot. Thousands of shots are taken; the lowest-energy bitstring seen is returned as the answer.

---

## Project structure

```
Quantum-Cloud/
├── qcrs/                       Core optimization package
│   ├── problem.py               QUBO encoder — Job, Node, SchedulingProblem, build_qubo()
│   ├── classical_solver.py      GreedySolver · SimulatedAnnealingSolver · BruteForceSolver
│   ├── qaoa_solver.py           QAOA circuit construction, COBYLA tuning, sampling
│   └── hybrid_pipeline.py       Orchestration / benchmarking utilities
│
├── dashboard/
│   ├── app.py                   Streamlit UI — problem builder, solver runner, results
│   ├── viz.py                   Custom animated visuals (server floor, qubit collapse)
│   └── qaoa_worker.py            Runs QAOA in an isolated subprocess
│
├── .streamlit/config.toml       Dark theme configuration
├── requirements.txt             All dependencies, pinned
├── README.md                    This file
└── QCRS_Complete_Technical_Guide.md   Full mathematical + architectural deep-dive
```

---

## Engineering decisions worth knowing

**QAOA runs in an isolated subprocess, not in-process.**
`qiskit-aer`'s native C++ backend can crash the Python interpreter outright under constrained environments — a segfault, not a catchable exception. `dashboard/qaoa_worker.py` runs QAOA as a separate OS process, so a crash there surfaces as a clean, reportable error instead of taking the entire app down with it. This was found and fixed by actually stress-testing the app end-to-end, not assumed in advance.

**Penalty auto-calibration is a safety guarantee, not a convenience default.**
`penalty_assign` is always re-inflated to strictly dominate the worst-case per-job capacity penalty (`penalty_assign ≥ 1.5 × (penalty_cpu + penalty_mem)`), which guarantees placing any single job anywhere is always cheaper than leaving it unassigned — closing an exploit where a large job's own utilization cost could otherwise exceed the cost of skipping it entirely.

**One process, not a split frontend/backend.**
An earlier version of this project split a Streamlit dashboard from a separate FastAPI backend; the two disagreed on response shape and produced inconsistent results depending on which code path executed. The current architecture runs everything in a single Streamlit process (with QAOA subprocess-isolated as above), eliminating that entire class of bug.

**Every solver result is cross-validated against exact brute-force search** on tractable problem sizes as part of this project's own testing — this is how the penalty-calibration bug above was actually found, not merely how it's described.

---

## An honest note on "quantum"

> QAOA's result on the problem sizes in this app matches the true optimum — confirmed against exact brute-force search. **That's a correctness check, not a demonstrated speed advantage.**
>
> This app simulates the quantum circuit on a classical computer (Qiskit's `AerSimulator`), and that simulation is itself exponential in qubit count — it costs `O(2ⁿ)`, the same scaling as brute-force search itself. At the problem sizes here (roughly 6–20 qubits), a classical computer can brute-force-check every possibility just as fast, or faster, than it can simulate the quantum circuit that's supposedly replacing that search.
>
> Real quantum advantage would require running on **actual quantum hardware**, at a problem size where classical brute force *and* classical simulation both become intractable but the physical device does not. That is explicitly outside what this project runs or claims.

This exact note appears in the app itself, immediately after QAOA finishes running — not just in this README.

---

## Tech stack

| Layer | Technology |
|---|---|
| Quantum circuit construction & simulation | Qiskit, Qiskit Aer |
| Classical optimization | NumPy, SciPy, COBYLA |
| Web interface | Streamlit |
| Interactive charts | Plotly |
| Custom animated visuals | Hand-built HTML/CSS/JS, rendered via `streamlit.components.v1` |
| Process isolation | Python `subprocess` |

---

## Deployment

Deployed on **Streamlit Community Cloud**, pointed at `dashboard/app.py`. The `.streamlit/config.toml` theme and `requirements.txt` are already configured for a zero-extra-setup deploy — connect the repo, set the main file path, done.

---

## License

MIT — see [LICENSE](./LICENSE).

<div align="center">

**Built by [Nitanshu Tak](https://github.com/Nitanshu715)** · UPES Dehradun

</div>
