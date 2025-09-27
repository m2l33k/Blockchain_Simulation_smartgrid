import streamlit as st
import json
import os
import pandas as pd

st.set_page_config(layout="wide", page_title="Inference Benchmark")
st.title("Model Inference Benchmark")
st.markdown("This page displays the performance benchmarks for the anomaly detection model itself, isolated from the simulation pipeline. This measures how quickly the model can make a prediction on a single sequence of data.")

LATENCY_REPORT_PATH = 'model_latency_report.json'

if not os.path.exists(LATENCY_REPORT_PATH):
    st.error(f"`{LATENCY_REPORT_PATH}` not found. Please run the `benchmark_model.py` script first to generate the report.")
    st.code("python benchmark_model.py", language="bash")
    st.stop()

with open(LATENCY_REPORT_PATH, 'r') as f:
    results = json.load(f)

st.header("Key Performance Metrics")
st.markdown("Based on an average of 100 predictions on a sample data sequence.")

col1, col2 = st.columns(2)
col1.metric(
    "Average Inference Latency",
    f"{results['avg_latency_ms']:.2f} ms",
    help="The average time taken for the full stacked model to make one prediction. Lower is better."
)
col2.metric(
    "Model Throughput",
    f"{results['throughput_preds_per_sec']:.1f} preds/sec",
    help="How many predictions the model can make per second on a single thread. Higher is better."
)

with st.expander("Show Detailed Latency Statistics"):
    st.subheader("Latency Distribution (in milliseconds)")
    data = {
        "Metric": ["Average", "Median", "Standard Deviation", "Minimum", "Maximum"],
        "Value (ms)": [
            f"{results['avg_latency_ms']:.2f}",
            f"{results['median_latency_ms']:.2f}",
            f"{results['std_latency_ms']:.2f}",
            f"{results['min_latency_ms']:.2f}",
            f"{results['max_latency_ms']:.2f}"
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)