import streamlit as st

st.set_page_config(
    page_title="Blockchain Anomaly Detection Dashboard",
    page_icon="🔗",
    layout="wide"
)

st.title("Blockchain-based Smart Grid Anomaly Detection")
st.markdown("---")

st.header("Project Overview")
st.markdown("""
This dashboard is the interface for a sophisticated anomaly detection system designed for a blockchain-based smart grid. It leverages a stacked ensemble of machine learning models to identify unusual and potentially malicious behavior within the network.

The system combines the strengths of:
- **LSTM Autoencoders**: To learn and detect anomalies in the temporal patterns of grid transactions.
- **Isolation Forests**: To identify rare and unusual combinations of feature values in the data.
- **A Meta-Learner**: An ensemble model that intelligently combines the outputs of the base models for a final, highly accurate verdict.
""")

st.header("Application Modules")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    st.markdown("Explore detailed visualizations from the model training process, including loss curves, ROC/PR curves, and score distributions. Understand *why* the model is effective.")

    st.subheader("Inference Benchmark")
    st.markdown("Analyze the raw computational speed of the model. This page shows the isolated inference latency and throughput, answering: *'How fast is the model at making a single prediction?'*")

with col2:
    st.subheader("Detection Latency")
    st.markdown("View the end-to-end performance of the detection pipeline in a simulated environment. This measures the time from when an anomaly is injected into the network until it is detected.")

    st.subheader("System Failure Prediction")
    st.markdown("An interactive stress-testing tool. This module forecasts future grid activity and predicts the point at which increased network load might cause the anomaly detection system to fail or generate false positives.")

st.markdown("---")
st.info("Use the sidebar to navigate to the different modules of the application.")