import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Detection Latency")
st.title("End-to-End Detection Latency")
st.markdown("Analyzes the total time between anomaly **injection** and **detection** from the `latency_log.csv` file generated during simulation.")

LATENCY_FILE_PATH = 'latency_log.csv'

def clear_latency_log_file():
    if os.path.exists(LATENCY_FILE_PATH):
        os.remove(LATENCY_FILE_PATH)
    st.success("Cleared latency log.")

with st.sidebar:
    st.header("Log Control")
    st.button("Clear Latency Log File", on_click=clear_latency_log_file)

if st.button("Refresh Data"):
    st.cache_data.clear()

if not os.path.exists(LATENCY_FILE_PATH):
    st.warning("`latency_log.csv` not found. Run a simulation with anomaly injection to generate it.")
    st.stop()

@st.cache_data(ttl=10)
def calculate_latencies_from_log(file_path):
    if os.path.getsize(file_path) == 0: return []
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return []

    df['anomaly_id'] = df['details'].str.extract(r'id=([0-9a-f\-]+)')
    df.dropna(subset=['anomaly_id'], inplace=True)
    injections = df[df['event_type'] == 'injection'].drop_duplicates(subset=['anomaly_id'], keep='first')
    detections = df[df['event_type'] == 'detection'].drop_duplicates(subset=['anomaly_id'], keep='first')
    if injections.empty or detections.empty: return []
    merged_df = pd.merge(injections, detections, on='anomaly_id', suffixes=('_inj', '_det'))
    merged_df['latency'] = merged_df['timestamp_det'] - merged_df['timestamp_inj']
    return merged_df[merged_df['latency'] >= 0]['latency'].tolist()

latencies = calculate_latencies_from_log(LATENCY_FILE_PATH)

if not latencies:
    st.info("No completed injection-detection pairs found yet. Keep the simulation running and refresh.")
    st.stop()

st.header("Key Performance Indicators")
latencies_np = np.array(latencies)
avg_latency, median_latency, max_latency, p95 = np.mean(latencies_np), np.median(latencies_np), np.max(latencies_np), np.percentile(latencies_np, 95)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Detections Measured", len(latencies))
col2.metric("Average Latency", f"{avg_latency:.2f} s")
col3.metric("Median Latency", f"{median_latency:.2f} s")
col4.metric("95th Percentile", f"{p95:.2f} s")

st.header("Latency Visualizations")
col_hist, col_box = st.columns(2)

with col_hist:
    st.subheader("Latency Distribution")
    fig_hist = go.Figure(data=[go.Histogram(x=latencies_np)])
    fig_hist.update_layout(xaxis_title="Latency (seconds)", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_box:
    st.subheader("Latency Box Plot")
    fig_box = go.Figure(data=[go.Box(y=latencies_np, name="Latency")])
    st.plotly_chart(fig_box, use_container_width=True)