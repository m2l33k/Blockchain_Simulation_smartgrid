# pages/3_🔬_Scalability_Analysis.py (Corrected)

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import json
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import glob  # <-- FIX: ADD THIS IMPORT STATEMENT

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Grid Scalability Analysis")
st.title("🔬 Smart Grid Scalability & Stress Test")
st.markdown("""
This tool simulates the future health of the grid as the number of devices (nodes) increases. 
By projecting future transaction loads based on a target node count, we can predict the point at which normal, heavy traffic might be flagged as anomalous, indicating a potential failure point for the current detection model.
""")

# --- Constants & Paths ---
MODEL_DIR = 'saved_models'
DATA_DIR = 'data'
LOG_DIRECTORY = 'simulation_logs'

# --- Model & Training Parameters (CRITICAL: Must match training scripts) ---
AUTOENCODER_SEQ_LEN = 10
FORECAST_INPUT_LEN = 20
FORECAST_HORIZON = 10

# --- Caching Functions ---
@st.cache_resource
def load_models_and_assets():
    assets = {}
    try:
        assets['autoencoder'] = load_model(os.path.join(MODEL_DIR, 'lstm_autoencoder_model.keras'))
        assets['forecaster'] = load_model(os.path.join(MODEL_DIR, 'forecasting_model.keras'))
        assets['scaler'] = joblib.load(os.path.join(MODEL_DIR, 'data_scaler.joblib'))
        with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'r') as f:
            assets['features'] = json.load(f)
        with open(os.path.join(MODEL_DIR, 'anomaly_threshold.json'), 'r') as f:
            assets['threshold'] = json.load(f)['threshold']
        return assets
    except Exception as e:
        st.error(f"Failed to load a required model asset: {e}")
        return None

@st.cache_data
def load_historical_data_and_nodes():
    data_path = os.path.join(DATA_DIR, 'featurized_block_data.csv')
    log_files = glob.glob(os.path.join(LOG_DIRECTORY, 'simulation_run_*.log'))
    
    if not os.path.exists(data_path) or not log_files:
        return None, None
        
    df = pd.read_csv(data_path)
    df['miner_id'] = df['miner_id'].astype('category').cat.codes
    
    latest_log = max(log_files, key=os.path.getmtime)
    node_pattern = re.compile(r"Node (CON-\d+|PRO-\d+|GRID-OP-\d+) registered")
    registered_nodes = set()
    with open(latest_log, 'r') as f:
        for line in f:
            match = node_pattern.search(line)
            if match: registered_nodes.add(match.group(1))
            
    return df, list(registered_nodes)

# --- Main App Logic ---
assets = load_models_and_assets()
df_historical, historical_nodes = load_historical_data_and_nodes()

if not assets or df_historical is None or historical_nodes is None:
    st.error("Could not load all necessary data and models. Please ensure all training scripts have been run successfully and log files exist.")
    st.stop()

# --- Sidebar Configuration ---
st.sidebar.header("Scalability Simulation")

# Get historical counts
hist_consumers = len([n for n in historical_nodes if 'CON' in n])
hist_prosumers = len([n for n in historical_nodes if 'PRO' in n])
hist_total_devices = hist_consumers + hist_prosumers

st.sidebar.markdown(f"**Current Grid:** `{hist_consumers}` Consumers, `{hist_prosumers}` Prosumers")

# User input for target grid size
target_consumers = st.sidebar.number_input("Target Number of Consumers", min_value=hist_consumers, value=hist_consumers, step=10)
target_prosumers = st.sidebar.number_input("Target Number of Prosumers", min_value=hist_prosumers, value=hist_prosumers, step=5)
target_total_devices = target_consumers + target_prosumers

# Calculate the growth factor
if hist_total_devices > 0:
    growth_factor = target_total_devices / hist_total_devices
else:
    growth_factor = 1.0

st.sidebar.metric("Calculated Growth Factor", f"{growth_factor:.2f}x")

forecast_steps = st.sidebar.slider("Forecast Horizon (Blocks)", 20, 100, 50, 5)

if st.button("Run Scalability Simulation"):
    with st.spinner("Simulating future grid load and analyzing system limits..."):
        
        # 1. Autoregressive Forecasting for the "Normal" Future
        last_sequence = assets['scaler'].transform(df_historical[assets['features']].tail(FORECAST_INPUT_LEN))
        current_sequence = np.expand_dims(last_sequence, axis=0)
        all_forecasts_scaled = []
        
        for _ in range(int(np.ceil(forecast_steps / FORECAST_HORIZON))):
            predicted_chunk = assets['forecaster'].predict(current_sequence, verbose=0)[0]
            all_forecasts_scaled.extend(predicted_chunk)
            current_sequence = np.expand_dims(np.array(all_forecasts_scaled[-FORECAST_INPUT_LEN:]), axis=0)

        final_forecast_scaled = np.array(all_forecasts_scaled)[:forecast_steps]
        
        # 2. Apply Growth Factor to create the "Stressed" Future
        stressed_forecast_scaled = final_forecast_scaled.copy()
        features_to_stress = ['num_transactions', 'total_amount_transacted', 'total_energy_transacted', 'unique_senders', 'unique_recipients']
        for feature in features_to_stress:
            if feature in assets['features']:
                feature_index = assets['features'].index(feature)
                stressed_forecast_scaled[:, feature_index] *= growth_factor

        # 3. Analyze Limits using the Autoencoder
        st.header("📉 Predicted Anomaly Score Analysis")
        
        def create_autoencoder_input(data_scaled):
            sequences = [data_scaled[i:i+AUTOENCODER_SEQ_LEN] for i in range(len(data_scaled) - AUTOENCODER_SEQ_LEN + 1)]
            return np.array(sequences)

        autoencoder_input = create_autoencoder_input(stressed_forecast_scaled)
        
        if autoencoder_input.shape[0] > 0:
            reconstruction = assets['autoencoder'].predict(autoencoder_input, verbose=0)
            mae_scores = np.mean(np.abs(reconstruction - autoencoder_input), axis=(1, 2))
            
            # 4. Plot the results
            last_block_index = df_historical['block_index'].max()
            error_index = range(last_block_index + AUTOENCODER_SEQ_LEN, last_block_index + len(mae_scores) + AUTOENCODER_SEQ_LEN)

            fig_error = go.Figure()
            fig_error.add_trace(go.Scatter(x=list(error_index), y=mae_scores, mode='lines+markers', name=f'Projected Error ({target_total_devices} nodes)', line=dict(color='red')))
            fig_error.add_hline(y=assets['threshold'], line_dash="dash", line_color="orange", annotation_text=f"Anomaly Threshold ({assets['threshold']:.4f})")
            fig_error.update_layout(title="Projected Reconstruction Error for Target Grid Size", xaxis_title="Forecasted Block Index", yaxis_title="Mean Absolute Error (MAE)")
            st.plotly_chart(fig_error, use_container_width=True)

            # 5. Provide a clear verdict
            max_predicted_error = mae_scores.max()
            st.subheader("Simulation Verdict")
            
            summary_text = f"""
            - **Target Grid Size:** `{target_consumers}` Consumers & `{target_prosumers}` Prosumers (`{target_total_devices}` total devices).
            - **This represents a `{growth_factor:.2f}x` growth** in node-related activity.
            - **Anomaly Threshold:** `{assets['threshold']:.4f}`
            - **Maximum Predicted Error for this grid size:** `{max_predicted_error:.4f}`
            """
            st.markdown(summary_text)

            if max_predicted_error > assets['threshold']:
                st.error(f"""
                **CONCLUSION: 🚨 SYSTEM FAILURE PREDICTED**
                
                At this scale, the grid's normal operational behavior is so complex that it is **predicted to be flagged as anomalous**. 
                The anomaly detection model would likely generate constant false alarms, rendering it ineffective. 
                The model or the anomaly threshold needs to be retrained or adjusted before scaling the grid to this size.
                """)
            elif max_predicted_error > assets['threshold'] * 0.8:
                st.warning(f"""
                **CONCLUSION: ⚠️ AT RISK OF FAILURE**
                
                The projected error is dangerously close to the anomaly threshold. While not guaranteed to fail, the system is operating at its **absolute limit**. 
                The risk of false alarms is very high, and any unexpected surge in activity could easily trigger them. This grid size is not recommended without improving the detection model.
                """)
            else:
                st.success(f"""
                **CONCLUSION: ✅ STABLE**
                
                The smart grid is predicted to operate **stably and effectively** at the target size. 
                The projected error from normal activity remains safely below the anomaly threshold, indicating the detection model can handle this increased load.
                """)
        else:
            st.warning(f"Forecast horizon is too short to create a full sequence for anomaly analysis. Please choose a forecast of at least {AUTOENCODER_SEQ_LEN} blocks.")