import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Latency Analysis")
st.title("⏱️ Anomaly Detection Latency Analysis")
st.markdown("Analyzes the time between anomaly injection and detection from the `latency_log.csv` file.")

LATENCY_FILE_PATH = 'latency_log.csv'

def clear_latency_log_file():
    """Deletes and re-initializes the latency log file."""
    if os.path.exists(LATENCY_FILE_PATH):
        os.remove(LATENCY_FILE_PATH)
    # The simulation will re-create it on next run
    st.success("Cleared `latency_log.csv`. The file will be recreated on the next simulation run.")

with st.sidebar:
    st.header("Log Control")
    st.button("Clear Latency Log File", on_click=clear_latency_log_file)

st.button("🔄 Refresh Data")

if not os.path.exists(LATENCY_FILE_PATH):
    st.warning("`latency_log.csv` not found. Run a simulation in `detect` mode to generate it.")
    st.stop()

@st.cache_data(ttl=10) # Cache for 10 seconds
def calculate_latencies_from_log(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        return []

    # Extract anomaly ID from the details column
    df['anomaly_id'] = df['details'].str.extract(r'id=([0-9a-f\-]+)')
    
    injections = df[df['event_type'] == 'injection'].drop_duplicates(subset=['anomaly_id'])
    detections = df[df['event_type'] == 'detection'].drop_duplicates(subset=['anomaly_id'])
    
    merged_df = pd.merge(
        injections, detections, on='anomaly_id', suffixes=('_inj', '_det')
    )
    
    merged_df['latency'] = merged_df['timestamp_det'] - merged_df['timestamp_inj']
    
    # Filter out any nonsensical latencies (e.g., negative if logs are weird)
    return merged_df[merged_df['latency'] >= 0]['latency'].tolist()

latencies = calculate_latencies_from_log(LATENCY_FILE_PATH)

if not latencies:
    st.info("No completed injection-detection pairs found yet. Keep the simulation running and refresh.")
    st.stop()

# --- KPIs and Charts ---
st.header("Key Performance Indicators")
latencies_np = np.array(latencies)
avg_latency, median_latency, min_latency, max_latency, p95 = np.mean(latencies_np), np.median(latencies_np), np.min(latencies_np), np.max(latencies_np), np.percentile(latencies_np, 95)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Detections Measured", len(latencies))
col2.metric("Average Latency", f"{avg_latency:.2f} s")
col3.metric("Median Latency", f"{median_latency:.2f} s")
col4.metric("Max Latency", f"{max_latency:.2f} s")
col5.metric("95th Percentile", f"{p95:.2f} s")

# --- Display Charts ---
st.header("Latency Visualizations")
col_hist, col_box = st.columns(2)

with col_hist:
    st.subheader("Latency Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=latencies_np, nbinsx=30, name='Latency'))
    fig_hist.add_vline(x=avg_latency, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_latency:.2f}s")
    fig_hist.update_layout(xaxis_title="Latency (seconds)", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_box:
    st.subheader("Latency Box Plot")
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=latencies_np, name="Latency", boxpoints='all', jitter=0.3, pointpos=-1.8))
    fig_box.update_layout(yaxis_title="Latency (seconds)")
    st.plotly_chart(fig_box, use_container_width=True)

# --- Raw Data Display ---
with st.expander("Show Raw Latency Data"):
    st.dataframe(pd.DataFrame(latencies, columns=["Latency (s)"]))