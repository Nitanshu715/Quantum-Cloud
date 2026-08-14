<div align="center">

# ⚛ QCRS
### Quantum-Cloud Resource Scheduler

**Define a real cloud-scheduling problem. Watch Greedy, Simulated Annealing, and QAOA solve it — live, on an animated server floor.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?style=flat-square&logo=qiskit&logoColor=white)](https://qiskit.org)
[![License](https://img.shields.io/badge/License-MIT-black?style=flat-square)](#license)

[**▶ Live Demo**](https://quantum-cloud-resource-scheduler.streamlit.app/) · [Full Technical Guide](./QCRS_Complete_Technical_Guide.md) · [Report a bug](../../issues)

</div>

---

## What is this?

Cloud platforms constantly answer one question: **which job runs on which server?** Get it wrong and you overload one node while another sits idle. QCRS lets you define that exact problem — your own jobs, your own servers — and watch three different solving philosophies attack it in real time:

<table>
<tr>
<td width="33%" valign="top">

### 🟢 Greedy
First-fit placement. Instant, but naive — commonly saturates one node while leaving another underused.

</td>
<td width="33%" valign="top">

### 🔵 Simulated Annealing
A classical metaheuristic that randomly perturbs, then "cools" its way to a strong, reliable solution.

</td>
<td width="33%" valign="top">

### 🟣 QAOA
A quantum circuit, simulated via Qiskit — puts every decision into superposition, then measures its way to an answer.

</td>
</tr>
</table>

All three search the **exact same mathematical objective**, so their results are directly comparable — that comparison *is* the point of this project.

---

## ✨ What it looks like

Instead of tables and matplotlib bar charts, results render as:

- 🖥️ **Server Floor** — racks as physical boxes, jobs animating into place, real-time utilization bars
- ⚛ **Qubit Collapse** — QAOA's qubits shown shimmering in superposition, then settling into a solid 0 or 1
- 📉 **Convergence** — energy dropping over iterations, live
- 📤 **Export** — download the winning allocation as JSON

---

## 🚀 Quick start

```bash
git clone https://github.com/Nitanshu715/Quantum-Cloud.git
cd Quantum-Cloud

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Add your own jobs and servers in the sidebar, hit **Run all three**, and inspect any result on the server floor.

---

## 🧠 How it actually works

Every `(job, node)` pair becomes one binary variable — one **qubit**. These are assembled into a **QUBO matrix** (Quadratic Unconstrained Binary Optimization):

```
minimize   E(x) = xᵀ Q x
```

`Q` encodes three things simultaneously: every job must be assigned exactly once, no node may be overloaded, and total cost should be minimized. All three solvers minimize this *same* matrix — they just search differently:

| | Greedy | Simulated Annealing | QAOA |
|---|:---:|:---:|:---:|
| **Approach** | First-fit heuristic | Randomized local search + cooling | Quantum superposition + measurement |
| **Speed** | Instant | Fast | Slower (circuit simulation) |
| **Guarantee** | None | Strong empirically | Matches exact optimum at these sizes |
| **Scales to** | Any size | Hundreds of variables | ~20 qubits (simulated) |

**→ For the full derivation — every formula, every penalty term, a fully worked numeric example, and the QAOA circuit explained gate-by-gate — see [`QCRS_Complete_Technical_Guide.md`](./QCRS_Complete_Technical_Guide.md).**

---

## 📁 Project structure

```
Quantum-Cloud/
├── qcrs/                       Core optimization package
│   ├── problem.py               QUBO encoder — Job, Node, SchedulingProblem
│   ├── classical_solver.py      Greedy · Simulated Annealing · Brute Force
│   └── qaoa_solver.py           QAOA circuit construction + COBYLA tuning
│
├── dashboard/
│   ├── app.py                   Streamlit UI
│   ├── viz.py                   Custom animated visuals (server floor, qubit collapse)
│   └── qaoa_worker.py            Runs QAOA in an isolated subprocess
│
├── .streamlit/config.toml       Theme
└── requirements.txt
```

---

## 🛡️ Why QAOA runs in its own process

`qiskit-aer`'s native backend can crash the interpreter outright under constrained environments — a segfault, not a catchable exception. `dashboard/qaoa_worker.py` isolates it in a subprocess, so a crash there surfaces as a clean error instead of taking the whole app down.

---

## ⚖️ An honest note on "quantum"

> QAOA's result on the problem sizes in this app matches the true optimum — confirmed against exact brute-force search. **That's a correctness check, not a demonstrated speed advantage.**
>
> This app simulates the quantum circuit on a classical computer, and that simulation is itself exponential in qubit count — it hits the same computational wall brute-force search does. Real quantum advantage only appears on real quantum hardware, at problem sizes no classical computer can brute-force or simulate.

This app says exactly this, in the interface itself, right after you run QAOA.

---

## License

MIT — see [LICENSE](./LICENSE).

<div align="center">

**Built by [Nitanshu Tak](https://github.com/Nitanshu715)** · UPES Dehradun

</div>
