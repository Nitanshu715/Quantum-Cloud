import streamlit as st
import requests

API_URL = "https://quantum-cloud.onrender.com"

st.set_page_config(page_title="Quantum Cloud Scheduler", layout="centered")

st.title("Quantum Cloud Resource Scheduler")

st.write("Run scheduling using classical and quantum solvers.")

# ─────────────────────────────
# RUN GREEDY
# ─────────────────────────────
if st.button("Run Greedy Scheduler"):
    res = requests.get(f"{API_URL}/run")
    data = res.json()

    st.subheader("Greedy Result")
    st.metric("Energy", data["energy"])
    st.success("Feasible" if data["feasible"] else "Not Feasible")


# ─────────────────────────────
# COMPARE SOLVERS
# ─────────────────────────────
if st.button("Compare All Solvers"):
    res = requests.get(f"{API_URL}/compare")
    data = res.json()

    st.subheader("Comparison")

    for solver, result in data.items():
        st.write(f"### {solver.upper()}")
        st.metric("Energy", result["energy"])
        st.write("Feasible:", result["feasible"])
        st.divider()


# ─────────────────────────────
# QAOA
# ─────────────────────────────
if st.button("Run QAOA"):
    res = requests.get(f"{API_URL}/qaoa")
    data = res.json()

    st.subheader("QAOA Result")

    if data.get("status") == "skipped":
        st.warning(data["reason"])
    else:
        st.metric("Energy", data["energy"])
        st.success("Feasible" if data["feasible"] else "Not Feasible")
