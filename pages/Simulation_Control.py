import streamlit as st
import subprocess
import os
import signal
import time

st.set_page_config(layout="wide", page_title="Simulation Control")
st.title("🕹️ Simulation Control Panel")

if 'simulation_process' not in st.session_state:
    st.session_state.simulation_process = None
if 'simulation_output' not in st.session_state:
    st.session_state.simulation_output = ""

def is_running():
    return st.session_state.simulation_process is not None and st.session_state.simulation_process.poll() is None

def stop_simulation():
    if is_running():
        st.info("Sending termination signal to the simulation...")
        st.session_state.simulation_process.terminate()
        try:
            st.session_state.simulation_process.wait(timeout=5)
            st.success("Simulation stopped successfully.")
        except subprocess.TimeoutExpired:
            st.session_state.simulation_process.kill()
            st.warning("Simulation forcefully terminated.")
        st.session_state.simulation_process = None
        st.rerun()

st.sidebar.header("Simulation Parameters")
num_prosumers = st.sidebar.slider("Number of Prosumers", 1, 100, 10)
num_consumers = st.sidebar.slider("Number of Consumers", 1, 200, 20)
difficulty = st.sidebar.slider("Blockchain Difficulty", 2, 5, 3)
duration = st.sidebar.slider("Simulation Duration (seconds)", 10, 300, 60)
tps = st.sidebar.slider("Ticks Per Second (TPS)", 5, 100, 20, help="Controls simulation speed. Higher is faster.")

st.header("Controls")
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Start Simulation", disabled=is_running(), type="primary"):
        command = [
            "python", "simulation.py",
            "--prosumers", str(num_prosumers),
            "--consumers", str(num_consumers),
            "--difficulty", str(difficulty),
            "--duration", str(duration),
            "--tps", str(tps) # Add the new TPS argument
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        st.session_state.simulation_process = process
        st.session_state.simulation_output = ""
        st.success(f"Simulation started! (PID: {process.pid})")
        st.rerun()

with col2:
    st.button("🛑 Stop Simulation", on_click=stop_simulation, disabled=not is_running())

st.header("Live Simulation Log")
if is_running():
    placeholder = st.empty()
    while is_running():
        log_lines = st.session_state.get("simulation_output", "").splitlines()
        with placeholder.container():
            st.text_area("Log Output", "\n".join(log_lines[-200:]), height=400, key="log_area")
        
        try:
            new_line = st.session_state.simulation_process.stdout.readline()
            if new_line:
                st.session_state.simulation_output += new_line
            elif st.session_state.simulation_process.poll() is not None:
                break # Exit loop if process ended
            time.sleep(0.1) # Small delay to prevent busy-looping
        except Exception: break
    
    # After loop finishes, do a final update
    st.info("Simulation process has finished.")
    st.session_state.simulation_process = None
    time.sleep(1)
    st.rerun()
else:
    if st.session_state.simulation_output:
        st.text_area("Final Log Output", st.session_state.simulation_output, height=400)
    else:
        st.info("Simulation is not running. Configure parameters and click 'Start Simulation'.")