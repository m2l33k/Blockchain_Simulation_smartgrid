# pages/3_📊_Model_Performance.py

import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Model Performance")
st.title("📊 Unsupervised Model Performance Diagnostics")
st.markdown("This page displays diagnostic plots generated during the training of the LSTM Autoencoder.")

# --- Constants ---
VIS_DIR = 'visualized_data'

# --- Display Plots ---
col1, col2 = st.columns(2)

with col1:
    st.header("Training History")
    path = os.path.join(VIS_DIR, 'unsupervised_loss_history.png')
    if os.path.exists(path):
        st.image(path, caption="Model Training & Validation Loss (MAE)")
    else:
        st.warning(f"Image not found: {path}")

with col2:
    st.header("Error Distribution")
    path = os.path.join(VIS_DIR, 'reconstruction_errors_distribution.png')
    if os.path.exists(path):
        st.image(path, caption="Distribution of Reconstruction Errors on Training Data")
    else:
        st.warning(f"Image not found: {path}")

st.header("Feature-wise Reconstruction Error")
path = os.path.join(VIS_DIR, 'feature_wise_reconstruction_error.png')
if os.path.exists(path):
    st.image(path, caption="Which features the model struggled most to reconstruct.")
else:
    st.warning(f"Image not found: {path}")

st.header("Reconstruction Examples")
col3, col4 = st.columns(2)

with col3:
    path = os.path.join(VIS_DIR, 'reconstruction_example_normal.png')
    if os.path.exists(path):
        st.image(path, caption="Example of a well-reconstructed 'Normal' sequence.")
    else:
        st.warning(f"Image not found: {path}")
        
with col4:
    path = os.path.join(VIS_DIR, 'reconstruction_example_anomaly.png')
    if os.path.exists(path):
        st.image(path, caption="Example of a poorly-reconstructed 'Anomalous' sequence.")
    else:
        st.warning(f"Image not found: {path}")