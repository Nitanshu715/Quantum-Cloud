# QCRS — Quantum-Cloud Resource Scheduler

Define a cloud scheduling problem (your own jobs and nodes), then watch
Greedy, Simulated Annealing, and QAOA solve it — rendered as a live
server floor and a qubit-collapse visualization instead of tables.

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Opens at http://localhost:8501

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On share.streamlit.io, point it at `dashboard/app.py` as the main file.
3. `requirements.txt` and `.streamlit/config.toml` are already set up — no
   extra configuration needed.

## What's inside

- `qcrs/` — the QUBO encoder and three solvers (Greedy, Simulated
  Annealing, QAOA via Qiskit) — unchanged solver logic, verified against
  brute-force search.
- `dashboard/app.py` — the Streamlit app: problem builder, solver runner,
  results.
- `dashboard/viz.py` — the two custom animated visuals (server floor,
  qubit collapse), built as HTML/CSS/JS components since Streamlit's
  native charts can't do this kind of spatial animation.
- `dashboard/qaoa_worker.py` — runs QAOA in an isolated subprocess.
  qiskit-aer's native backend can crash the interpreter outright under
  constrained containers; isolating it means that failure can't take
  the rest of the app down, it just surfaces as a clean error message.

## Note on the quantum framing

QAOA's result on the included problem sizes matches the true optimum,
confirmed against exact brute-force search — that's a correctness check,
not a demonstrated speed advantage. This app simulates the quantum
circuit on a classical computer, and that simulation is itself
exponential in qubit count, so it hits the same wall brute force does
at scale. The app says this explicitly once you run QAOA.
