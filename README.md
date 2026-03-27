<div align="center">

<img src="https://img.shields.io/badge/Quantum-Computing-6D28D9?style=for-the-badge&logo=atom&logoColor=white"/>
<img src="https://img.shields.io/badge/Qiskit-1.x-1E40AF?style=for-the-badge&logo=ibm&logoColor=white"/>
<img src="https://img.shields.io/badge/AWS-Braket-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.10+-0f766e?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>

<br/><br/>

<p align="center">
  <a href="https://the-quantum-cloud-resource-scheduler.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen?style=for-the-badge&logo=streamlit" />
  </a>
</p>

### ⚛ Quantum-Cloud Resource Scheduler

### *Hybrid QAOA + Classical Cloud Workload Allocation*

**Encodes cloud scheduling as a Hamiltonian · Solves with QAOA on Qiskit / AWS Braket · Integrates with EKS & AWS Batch**

<br/>

> **By: Nitanshu Tak**

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/Status-Active-22c55e)
![Qubits](https://img.shields.io/badge/Qubits-6--16-a78bfa)

</div>

---

## 📋 Table of Contents

- [What Is This?](#-what-is-this)
- [Why Quantum?](#-why-quantum)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [The Math: QUBO Formulation](#-the-math-qubo-formulation)
- [QAOA Circuit](#-qaoa-circuit)
- [Dashboard Guide](#-dashboard-guide)
- [Benchmark Results](#-benchmark-results)
- [Tech Stack](#-tech-stack)
- [AWS Integration](#-aws-integration)
- [Roadmap](#-roadmap)

---

## 🌌 What Is This?

Cloud providers like AWS allocate thousands of jobs to compute nodes every second. Deciding **which job runs on which server** to minimize cost, energy, and SLA violations is a combinatorial NP-hard optimization problem — classical algorithms fail at scale.

**QCRS (Quantum-Cloud Resource Scheduler)** solves this using the **Quantum Approximate Optimization Algorithm (QAOA)**:

1. **Encode** — Cloud scheduling → QUBO matrix → Cost Hamiltonian $H_C$
2. **Quantum solve** — QAOA circuit explores the full assignment space simultaneously  
3. **Measure** — Sample the lowest-energy bitstring = optimal job allocation
4. **Deploy** — Feed result to Amazon EKS / AWS Batch for real scheduling

```
Jobs + Nodes → QUBO Matrix → Hamiltonian → QAOA Circuit → Best Assignment
```

---

## ⚡ Why Quantum?

| Algorithm | Quality | Time Complexity | Notes |
|-----------|---------|-----------------|-------|
| Greedy FFD | Suboptimal | O(n log n) | Saturates nodes unevenly |
| Simulated Annealing | Near-optimal | O(n² iterations) | Strong classical baseline |
| Brute Force | **Exact** | O(2ⁿ) | Unusable at scale |
| **QAOA (ours)** | **Near-optimal** | Polynomial circuit depth | Scales with hardware improvements |

> **35% better allocation** than Greedy on our benchmark — QAOA finds the exact optimal solution on the small problem, matching Brute Force without exhaustive search.

Classical algorithms make locally optimal choices that lead to globally poor solutions. QAOA uses **quantum superposition** to simultaneously evaluate all $2^n$ possible assignments and **quantum interference** to amplify the probability of the best one.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cloud Workloads (Input Layer)                     │
│         Jobs (CPU/RAM) · SLA constraints · Cost limits              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Classical Pre-Processor                           │
│      Greedy baseline · Problem size reduction · QUBO encoding       │
└──────────────────────┬──────────────────────────┬───────────────────┘
                       │                          │
              n_vars > 12                    n_vars ≤ 12
                       │                          │
                       ▼                          ▼
          ┌────────────────────┐    ┌────────────────────────────┐
          │  Simulated         │    │  QAOA Solver               │
          │  Annealing (SA)    │    │  Qiskit · AWS Braket       │
          │  Large problems    │    │  γ, β optimized by COBYLA  │
          └─────────┬──────────┘    └──────────────┬─────────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
          ┌────────────────────────────────────────────┐
          │          Hybrid Scheduler Output            │
          │  ➜ Amazon EKS pod placement                │
          │  ➜ AWS Batch job queue decisions            │
          │  ➜ Autoscaler trigger conditions           │
          └────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Install

```bash
# 1. Clone and enter the project
git clone https://github.com/Nitanshu715/Quantum-Cloud.git
cd Quantum-Cloud

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the demo (CLI)

```bash
python notebooks/demo.py
```

Expected output:
```
==================================================
  QCRS Schedule Output  [SA — Optimal]
==================================================

  node-A  (cost/unit: 1.0)
    Jobs   : ml-train
    CPU    : [██████░░░░] 4.0/6.0 (67%)
    Memory : [██████░░░░] 8.0/12.0GB (67%)

  node-B  (cost/unit: 1.2)
    Jobs   : web-api, db-backup
    CPU    : [█████░░░░░] 3.0/6.0 (50%)
    Memory : [█████░░░░░] 6.0/12.0GB (50%)

  Status : ✓ FEASIBLE
  Energy : -352.78
==================================================
```

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
Quantum-Cloud/
│
├── qcrs/                          # Core package
│   ├── __init__.py                # Package exports
│   ├── problem.py                 # 🔵 QUBO / Hamiltonian encoder
│   │                              #    Jobs, Nodes, QUBO matrix, constraint checking
│   ├── classical_solver.py        # 🟢 Classical solvers
│   │                              #    GreedySolver, SimulatedAnnealingSolver, BruteForceSolver
│   ├── qaoa_solver.py             # 🟣 QAOA quantum solver
│   │                              #    Circuit construction, parameter optimization, sampling
│   └── hybrid_pipeline.py        # 🟠 Hybrid orchestrator
│                                  #    Routing, benchmarking, pretty-print schedule
│
├── notebooks/
│   └── demo.py                    # End-to-end CLI walkthrough + plot generation
│
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
│
├── requirements.txt               # All dependencies
└── README.md                      # This file
```

---

## 📐 The Math: QUBO Formulation

The scheduling problem is encoded as:

$$\text{minimize} \quad \mathbf{x}^T Q \mathbf{x}$$

where $x_{ij} \in \{0,1\}$ = 1 if job $i$ is assigned to node $j$.

**The QUBO matrix $Q$ encodes three things:**

**① Assignment constraint** — each job must go to exactly one node:
$$A \cdot \left(\sum_j x_{ij} - 1\right)^2 \quad \forall i$$

**② CPU capacity constraint** — no node can be overloaded:
$$B \cdot \left(\frac{\sum_i \text{cpu}_i \cdot x_{ij}}{\text{cap}_j}\right)^2 \quad \forall j$$

**③ Cost objective** — minimize total cost × utilization:
$$\sum_{i,j} \text{cost}_j \cdot (\text{cpu}_i + 0.1 \cdot \text{mem}_i) \cdot x_{ij}$$

> Penalties $A \gg B \gg \text{cost}$ ensure constraint satisfaction dominates cost minimization.

---

## ⚛ QAOA Circuit

The $p$-layer QAOA ansatz:

$$|\psi_p\rangle = \prod_{k=1}^{p} \left[ e^{-i\beta_k H_B} \cdot e^{-i\gamma_k H_C} \right] |{+}\rangle^{\otimes n}$$

**Implementation in Qiskit:**

```
|0⟩ ─ H ─┬──────── RZ(2γ Q[i,i]) ── CNOT─RZ─CNOT ──┬── RX(2β) ──┬── measure
          │         (diagonal terms)   (ZZ interactions)│             │
|0⟩ ─ H ─┘                                            └─────────────┘
          └─────── Layer 1 (cost) ────────────────┘└── Layer 1 (mixer) ──┘
                  ↑ repeated p times ↑
```

**Parameter optimization:**
- Angles $(γ_1...γ_p, β_1...β_p)$ optimized classically by **COBYLA**
- Objective: minimize $\langle \psi | H_C | \psi \rangle$ (expectation of cost Hamiltonian)
- Multiple random restarts to escape local optima

---

## 🖥 Dashboard Guide

### 1. Configure (sidebar)
Select a preset problem or customize:
- **n_jobs**: 2–5 jobs (workloads)
- **n_nodes**: 2–4 nodes (servers)
- **QAOA depth p**: 1–4 layers
- **Shots**: 512–8192 measurements
- **Backend**: `statevector` (exact) or `qasm` (shot-based)

### 2. Inspect the problem
- **Problem Definition tab**: jobs table, nodes table
- **QUBO Matrix tab**: 6×6 heatmap — red = penalty, blue = reward

### 3. Run solvers

| Button | What it runs | What to look for |
|--------|-------------|-----------------|
| ▶ Greedy | Fast baseline | node-A saturated at 100% |
| ▶ Sim. Annealing | Classical optimizer | Balanced 50–67% utilization |
| ▶ QAOA | **Quantum solver** | Matches SA energy, convergence plot |
| ▶ Full Benchmark | All methods | Energy comparison table |

### 4. Visualisation tabs

| Tab | Shows |
|-----|-------|
| **Allocation** | ●/○ heatmap: which job → which node |
| **Resource Utilization** | CPU and RAM bars per node |
| **Convergence** | Energy dropping over optimizer iterations |
| **Bitstring Distribution** | Histogram of all QAOA measurements — orange = selected |

### ✅ Project validated when:
1. Greedy energy ≈ −261, SA/Brute Force energy ≈ −352 (35% gap)
2. QAOA energy matches SA (quantum = classical optimum)
3. Bitstring distribution shows orange bar (optimal state) with highest probability
4. All methods show `Feasible: ✓`

---

## 📊 Benchmark Results

**Problem: 3 jobs × 2 nodes (6 binary variables / qubits)**

| Method | QUBO Energy | Quality | Feasible | Time |
|--------|-------------|---------|----------|------|
| Greedy FFD | −260.86 | Baseline | ✓ | < 1 ms |
| Simulated Annealing | **−352.78** | +35.2% vs Greedy | ✓ | ~50 ms |
| Brute Force (exact) | **−352.78** | Optimal | ✓ | ~5 ms |
| **QAOA p=2** | **−352.78** | **Optimal** ✨ | ✓ | ~10–30 s |

**Greedy allocation** (suboptimal):
```
node-A: [██████████] 100% CPU  [██████████] 100% RAM  ← saturated
node-B: [█░░░░░░░░░]  17% CPU  [█░░░░░░░░░]  17% RAM  ← underutilized
```

**QAOA / SA allocation** (optimal):
```
node-A: [██████░░░░]  67% CPU  [██████░░░░]  67% RAM  ← balanced
node-B: [█████░░░░░]  50% CPU  [█████░░░░░]  50% RAM  ← balanced
```

> A saturated node-A means zero headroom for load spikes, higher SLA violation risk, and wasted capacity on node-B. The quantum-optimal allocation distributes load evenly, improving fault tolerance and cost efficiency.

---

## 🛠 Tech Stack

```
Quantum Layer          Classical Layer        Infrastructure
──────────────         ───────────────        ──────────────
Qiskit 1.x             NumPy / SciPy          AWS Braket (hardware)
Qiskit Aer             Matplotlib             Amazon EKS
QAOA / VQE             Pandas                 AWS Batch
COBYLA optimizer        Streamlit              Kubernetes
```

---

## ☁ AWS Integration

**Current:** Qiskit Aer (local CPU simulation) — exact, noiseless  
**Production:** AWS Braket managed simulators or real quantum hardware

```python
# Swap Aer for Braket SV1 managed simulator:
from braket.aws import AwsDevice
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

# Or real IonQ hardware:
device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")
```

**Downstream integration:**
- QCRS output → EKS pod affinity rules
- QCRS output → AWS Batch placement policies  
- QCRS invoked when utilization imbalance > threshold (autoscaler hook)

---

## 🗺 Roadmap

- [x] QUBO encoding for cloud scheduling
- [x] QAOA solver (Qiskit, statevector + qasm backends)
- [x] Classical baseline suite (Greedy, SA, Brute Force)
- [x] Hybrid routing pipeline
- [x] Streamlit interactive dashboard
- [ ] AWS Braket adapter for real hardware execution
- [ ] Live EKS webhook integration
- [ ] Multi-objective optimization (cost + energy + SLA simultaneously)
- [ ] VQE-based approach for larger problem instances
- [ ] Noise-aware QAOA with error mitigation

---

<div align="center">

**Built for the Quantum Computing Hackathon 2025**  
*Learning, experimenting, and pushing the boundary of quantum-classical cloud infrastructure.*

<br/>

*Nitanshu Tak · Team Double Epsilons · UPES Dehradun*

</div>
