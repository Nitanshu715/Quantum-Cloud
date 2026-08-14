"""
dashboard/viz.py — Custom animated visual components for QCRS.

Two signature visuals, built as raw HTML/CSS/JS islands (rendered via
st.components.v1.html) because Streamlit's native chart widgets can't
do physical/spatial animation:

  1. rack_floor()   — servers as physical racks, jobs as blocks that
                       animate into place. This replaces the old
                       matplotlib allocation heatmap.
  2. qubit_collapse() — qubits rendered as spheres in shimmering
                       superposition that collapse to solid 0/1 states,
                       each one wired directly to the rack it decided.
"""

import json


PALETTE = {
    "bg": "#0B0E1A",
    "panel": "#131829",
    "panel_2": "#1A2036",
    "border": "#262E4A",
    "text": "#E8E9F3",
    "muted": "#8B92B0",
    "violet": "#7C5CFC",
    "violet_soft": "rgba(124, 92, 252, 0.18)",
    "amber": "#F5A623",
    "amber_soft": "rgba(245, 166, 35, 0.18)",
    "teal": "#2DD4BF",
    "red": "#F1596B",
}


def _font_import():
    return """
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    """


def rack_floor(node_labels, node_cpu_cap, node_mem_cap, job_labels, job_cpu, job_mem,
                alloc, title="Allocation", height=None):
    """
    Renders nodes as server racks on a floor. `alloc` is a list of lists,
    alloc[i][j] = 1 if job i placed on node j.
    Jobs animate into their assigned rack's job list, and each rack's
    fill bar animates to its true utilization.
    """
    n_nodes = len(node_labels)
    n_jobs = len(job_labels)
    if height is None:
        height = 220 + 46 * max(1, max((sum(row[j] for row in alloc) for j in range(n_nodes)), default=1))

    data = {
        "nodes": [
            {"label": node_labels[j], "cpu_cap": node_cpu_cap[j], "mem_cap": node_mem_cap[j]}
            for j in range(n_nodes)
        ],
        "jobs": [
            {"label": job_labels[i], "cpu": job_cpu[i], "mem": job_mem[i]}
            for i in range(n_jobs)
        ],
        "alloc": alloc,
    }
    data_json = json.dumps(data)

    html = f"""
<div id="rack-root"></div>
<style>
{_font_import()}
#rack-root {{
  font-family: 'Inter', sans-serif;
  color: {PALETTE['text']};
  background: transparent;
}}
.rf-title {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: {PALETTE['muted']};
  margin-bottom: 14px;
}}
.rf-floor {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
}}
.rf-rack {{
  background: linear-gradient(180deg, {PALETTE['panel_2']} 0%, {PALETTE['panel']} 100%);
  border: 1px solid {PALETTE['border']};
  border-radius: 10px;
  padding: 14px 16px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.4s ease;
}}
.rf-rack.rf-over {{
  border-color: {PALETTE['red']};
}}
.rf-rack-head {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}}
.rf-rack-name {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  color: {PALETTE['text']};
}}
.rf-rack-cap {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  color: {PALETTE['muted']};
}}
.rf-bar-track {{
  height: 6px;
  border-radius: 3px;
  background: {PALETTE['panel']};
  border: 1px solid {PALETTE['border']};
  overflow: hidden;
  margin-bottom: 3px;
}}
.rf-bar-fill {{
  height: 100%;
  width: 0%;
  border-radius: 3px;
  background: linear-gradient(90deg, {PALETTE['violet']}, {PALETTE['teal']});
  transition: width 1.1s cubic-bezier(.2,.8,.2,1);
}}
.rf-bar-fill.rf-over-fill {{
  background: linear-gradient(90deg, {PALETTE['amber']}, {PALETTE['red']});
}}
.rf-bar-label {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9.5px;
  color: {PALETTE['muted']};
  margin-bottom: 8px;
}}
.rf-jobs {{
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 16px;
}}
.rf-job {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  padding: 6px 9px;
  border-radius: 6px;
  background: {PALETTE['violet_soft']};
  border: 1px solid rgba(124,92,252,0.35);
  opacity: 0;
  transform: translateY(-8px) scale(0.96);
  animation: rf-drop 0.5s cubic-bezier(.2,.8,.2,1) forwards;
  display: flex;
  justify-content: space-between;
  gap: 8px;
}}
.rf-job-meta {{ color: {PALETTE['muted']}; }}
@keyframes rf-drop {{
  to {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
</style>
<div class="rf-title">{title}</div>
<div class="rf-floor" id="rf-floor"></div>
<script>
const DATA = {data_json};
const floor = document.getElementById('rf-floor');

DATA.nodes.forEach((node, j) => {{
  const cpuUsed = DATA.jobs.reduce((s, job, i) => s + (DATA.alloc[i][j] ? job.cpu : 0), 0);
  const memUsed = DATA.jobs.reduce((s, job, i) => s + (DATA.alloc[i][j] ? job.mem : 0), 0);
  const cpuPct = Math.min(100, (cpuUsed / node.cpu_cap) * 100);
  const memPct = Math.min(100, (memUsed / node.mem_cap) * 100);
  const over = cpuUsed > node.cpu_cap + 1e-9 || memUsed > node.mem_cap + 1e-9;

  const rack = document.createElement('div');
  rack.className = 'rf-rack' + (over ? ' rf-over' : '');

  const jobsAssigned = DATA.jobs
    .map((job, i) => ({{...job, i}}))
    .filter((job) => DATA.alloc[job.i][j] === 1);

  rack.innerHTML = `
    <div class="rf-rack-head">
      <div class="rf-rack-name">${{node.label}}</div>
      <div class="rf-rack-cap">${{node.cpu_cap}} vCPU · ${{node.mem_cap}}GB</div>
    </div>
    <div class="rf-bar-track"><div class="rf-bar-fill${{cpuPct > 100 ? ' rf-over-fill' : ''}}" data-w="${{cpuPct}}"></div></div>
    <div class="rf-bar-label">CPU ${{cpuUsed.toFixed(1)}}/${{node.cpu_cap}} (${{cpuPct.toFixed(0)}}%)</div>
    <div class="rf-bar-track"><div class="rf-bar-fill${{memPct > 100 ? ' rf-over-fill' : ''}}" data-w="${{memPct}}"></div></div>
    <div class="rf-bar-label">RAM ${{memUsed.toFixed(1)}}/${{node.mem_cap}}GB (${{memPct.toFixed(0)}}%)</div>
    <div class="rf-jobs"></div>
  `;
  floor.appendChild(rack);

  const jobsEl = rack.querySelector('.rf-jobs');
  jobsAssigned.forEach((job, idx) => {{
    const el = document.createElement('div');
    el.className = 'rf-job';
    el.style.animationDelay = (0.15 * idx) + 's';
    el.innerHTML = `<span>${{job.label}}</span><span class="rf-job-meta">${{job.cpu}}c · ${{job.mem}}GB</span>`;
    jobsEl.appendChild(el);
  }});

  requestAnimationFrame(() => {{
    rack.querySelectorAll('.rf-bar-fill').forEach((bar) => {{
      const w = bar.getAttribute('data-w');
      setTimeout(() => {{ bar.style.width = Math.min(100, w) + '%'; }}, 60);
    }});
  }});
}});
</script>
"""
    return html, int(height)


def qubit_collapse(bitstring, var_labels, node_labels, n_nodes, title="Measurement — qubits collapsing to an assignment", height=260):
    """
    Renders n_qubits spheres that start in a shimmering superposition state
    and settle into solid amber (0) / violet (1) based on `bitstring`
    (a string of 0/1, one bit per job-node variable). Each qubit is
    labeled with which (job, node) decision it encodes.
    """
    n = len(bitstring)
    qubits = [
        {"bit": int(b), "label": var_labels[k] if k < len(var_labels) else f"q{k}"}
        for k, b in enumerate(bitstring)
    ]
    data_json = json.dumps({"qubits": qubits})

    html = f"""
<div id="qc-root"></div>
<style>
{_font_import()}
#qc-root {{ font-family: 'Inter', sans-serif; color: {PALETTE['text']}; }}
.qc-title {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: {PALETTE['muted']}; margin-bottom: 16px;
}}
.qc-row {{ display: flex; flex-wrap: wrap; gap: 18px; }}
.qc-qubit {{ display: flex; flex-direction: column; align-items: center; gap: 8px; width: 78px; }}
.qc-sphere {{
  width: 46px; height: 46px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.5), {PALETTE['violet']} 45%, {PALETTE['amber']} 100%);
  background-size: 220% 220%;
  animation: qc-shimmer 1.1s ease-in-out infinite;
  box-shadow: 0 0 0 rgba(124,92,252,0);
}}
.qc-sphere.qc-settled-1 {{
  animation: none;
  background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.55), {PALETTE['violet']} 70%);
  box-shadow: 0 0 18px rgba(124,92,252,0.55);
}}
.qc-sphere.qc-settled-0 {{
  animation: none;
  background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.4), {PALETTE['amber']} 70%);
  box-shadow: 0 0 18px rgba(245,166,35,0.45);
}}
@keyframes qc-shimmer {{
  0% {{ background-position: 0% 50%; }}
  50% {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}
.qc-label {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9.5px; color: {PALETTE['muted']}; text-align: center; line-height: 1.3;
}}
.qc-state {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; color: {PALETTE['muted']}; opacity: 0; transition: opacity 0.3s;
}}
.qc-state.qc-shown {{ opacity: 1; }}
</style>
<div class="qc-title">{title}</div>
<div class="qc-row" id="qc-row"></div>
<script>
const DATA = {data_json};
const row = document.getElementById('qc-row');
DATA.qubits.forEach((q, idx) => {{
  const el = document.createElement('div');
  el.className = 'qc-qubit';
  el.innerHTML = `
    <div class="qc-sphere" id="sph-${{idx}}"></div>
    <div class="qc-state" id="st-${{idx}}">|${{q.bit}}⟩</div>
    <div class="qc-label">${{q.label}}</div>
  `;
  row.appendChild(el);
  const sphere = el.querySelector('.qc-sphere');
  const state = el.querySelector('.qc-state');
  setTimeout(() => {{
    sphere.classList.add(q.bit === 1 ? 'qc-settled-1' : 'qc-settled-0');
    state.classList.add('qc-shown');
  }}, 900 + idx * 90);
}});
</script>
"""
    return html, height
