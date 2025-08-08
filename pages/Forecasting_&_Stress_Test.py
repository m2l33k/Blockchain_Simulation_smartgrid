# pages/2_📈_Forecasting_&_Stress_Test.py (Corrected)

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Grid Forecasting")
st.title("📈 Grid Metric Forecasting & Stress Test")
st.markdown("""
Use this tool to forecast future grid behavior and simulate the impact of network growth.
The **Stress Test** helps predict when increased activity might cause the anomaly detection system to generate false positives.
""")

# --- Constants & Paths ---
MODEL_DIR = 'saved_models'
DATA_DIR = 'data'

# --- Model & Training Parameters (CRITICAL: Must match training scripts) ---
AUTOENCODER_SEQ_LEN = 10
FORECAST_INPUT_LEN = 20
FORECAST_HORIZON = 10

# --- Caching Functions (for performance) ---
@st.cache_resource
def load_models_and_assets():
    assets = {}
    assets['autoencoder'] = load_model(os.path.join(MODEL_DIR, 'lstm_autoencoder_model.keras'))
    assets['forecaster'] = load_model(os.path.join(MODEL_DIR, 'forecasting_model.keras'))
    assets['scaler'] = joblib.load(os.path.join(MODEL_DIR, 'data_scaler.joblib'))
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'r') as f:
        assets['features'] = json.load(f)
    with open(os.path.join(MODEL_DIR, 'anomaly_threshold.json'), 'r') as f:
        assets['threshold'] = json.load(f)['threshold']
    return assets

@st.cache_data
def load_historical_data():
    path = os.path.join(DATA_DIR, 'featurized_block_data.csv')
    df = pd.read_csv(path)
    df['miner_id'] = df['miner_id'].astype('category').cat.codes
    return df

# --- Load all necessary components ---
try:
    assets = load_models_and_assets()
    df_historical = load_historical_data()
except FileNotFoundError:
    st.error("One or more required model assets are missing. Please ensure you have run both training scripts successfully.")
    st.stop()

# --- FIX: Define the callback function for the reset button ---
def reset_parameters():
    """This function will be called when the reset button is clicked."""
    st.session_state.forecast_steps = 30
    st.session_state.growth_factor = 1.0
    # Note: We don't need st.experimental_rerun() here, Streamlit handles it.

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Forecasting Configuration")

    # Initialize session state if it doesn't exist
    if 'forecast_steps' not in st.session_state:
        st.session_state.forecast_steps = 30
    if 'growth_factor' not in st.session_state:
        st.session_state.growth_factor = 1.0

    st.slider("Number of Future Blocks to Forecast", 10, 100, key='forecast_steps')
    feature_to_plot = st.selectbox("Select Feature to Visualize", options=assets['features'], index=assets['features'].index('num_transactions'))

    st.header("Stress Test Configuration")
    st.slider("Grid Growth Factor", 1.0, 5.0, key='growth_factor', step=0.1)
    features_to_stress = st.multiselect(
        "Select Metrics to Stress",
        options=assets['features'],
        default=['num_transactions', 'total_amount_transacted', 'total_energy_transacted', 'unique_senders', 'unique_recipients']
    )

    # FIX: Use the 'on_click' parameter to call the callback function
    st.button("Reset to Defaults", on_click=reset_parameters)

# --- Main Logic ---
if st.button("Run Forecast & Analyze Grid Limits"):
    # Retrieve values from session_state for the calculation
    forecast_steps = st.session_state.forecast_steps
    growth_factor = st.session_state.growth_factor

    with st.spinner("Generating autoregressive forecast and analyzing system limits..."):
        
        # 1. Autoregressive Forecasting
        last_sequence = assets['scaler'].transform(df_historical[assets['features']].tail(FORECAST_INPUT_LEN))
        current_sequence = np.expand_dims(last_sequence, axis=0)
        all_forecasts_scaled = []
        
        for _ in range(int(np.ceil(forecast_steps / FORECAST_HORIZON))):
            predicted_chunk = assets['forecaster'].predict(current_sequence, verbose=0)[0]
            all_forecasts_scaled.extend(predicted_chunk)
            current_sequence = np.expand_dims(np.array(all_forecasts_scaled[-FORECAST_INPUT_LEN:]), axis=0)

        final_forecast_scaled = np.array(all_forecasts_scaled)[:forecast_steps]
        forecast_inversed = assets['scaler'].inverse_transform(final_forecast_scaled)
        
        last_block_index = df_historical['block_index'].max()
        forecast_index = range(last_block_index + 1, last_block_index + 1 + forecast_steps)
        df_forecast = pd.DataFrame(forecast_inversed, index=forecast_index, columns=assets['features'])
        
        df_stressed_forecast = df_forecast.copy()
        for feature in features_to_stress:
            df_stressed_forecast[feature] *= growth_factor

        # 2. Plot Forecast
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df_historical['block_index'], y=df_historical[feature_to_plot], mode='lines', name='Historical'))
        fig_forecast.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast[feature_to_plot], mode='lines', name='Forecasted (Normal)', line=dict(dash='dash')))
        if growth_factor > 1.0:
             fig_forecast.add_trace(go.Scatter(x=df_stressed_forecast.index, y=df_stressed_forecast[feature_to_plot], mode='lines', name=f'Forecasted (Stressed x{growth_factor:.1f})', line=dict(dash='dot', color='red')))
        fig_forecast.update_layout(title=f'Forecast for: {feature_to_plot}', xaxis_title='Block Index', yaxis_title='Value')
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # 3. Analyze Limits
        st.header("📉 Predicted Anomaly Score Analysis")
        stressed_forecast_scaled = assets['scaler'].transform(df_stressed_forecast)
        
        def create_autoencoder_input(data_scaled):
            sequences = [data_scaled[i:i+AUTOENCODER_SEQ_LEN] for i in range(len(data_scaled) - AUTOENCODER_SEQ_LEN + 1)]
            return np.array(sequences)

        autoencoder_input_stressed = create_autoencoder_input(stressed_forecast_scaled)
        
        if autoencoder_input_stressed.shape[0] > 0:
            recon_stressed = assets['autoencoder'].predict(autoencoder_input_stressed, verbose=0)
            mae_stressed = np.mean(np.abs(recon_stressed - autoencoder_input_stressed), axis=(1, 2))
            
            fig_error = go.Figure()
            error_index = df_forecast.index[AUTOENCODER_SEQ_LEN-1:]
            fig_error.add_trace(go.Scatter(x=error_index, y=mae_stressed, mode='lines+markers', name=f'Projected Error (Stressed x{growth_factor:.1f})', line=dict(color='red')))
            fig_error.add_hline(y=assets['threshold'], line_dash="dash", line_color="orange", annotation_text=f"Anomaly Threshold ({assets['threshold']:.4f})")
            fig_error.update_layout(title="Projected Reconstruction Error vs. Anomaly Threshold", xaxis_title="Forecasted Block Index", yaxis_title="Mean Absolute Error (MAE)")
            st.plotly_chart(fig_error, use_container_width=True)

            # 4. Provide Summary
            max_predicted_error = mae_stressed.max()
            st.subheader(f"System Health Outlook (Growth Factor: {growth_factor:.1f}x)")
            
            summary_text = f"""
            The analysis simulates a **{int((growth_factor-1)*100)}% increase** in key grid metrics over a forecast of **{forecast_steps} blocks**.
            
            - **Current Anomaly Threshold:** `{assets['threshold']:.4f}`
            - **Maximum Predicted Error under Stress:** `{max_predicted_error:.4f}`
            """
            st.markdown(summary_text)

            if max_predicted_error > assets['threshold']:
                st.error("**Conclusion: 🚨 ALERT** - The system is predicted to become **unstable**. The projected error significantly exceeds the anomaly threshold. This suggests that normal grid operations under this increased load would likely be flagged as anomalies, leading to false alarms.")
            elif max_predicted_error > assets['threshold'] * 0.8:
                st.warning("**Conclusion: ⚠️ WARNING** - The system is approaching its **operational limit**. The projected error is high and close to the anomaly threshold. The reliability of the anomaly detection model may be reduced under these conditions, with an increased risk of false positives.")
            else:
                st.success("**Conclusion: ✅ STABLE** - The system is predicted to remain **stable and effective**. The projected error stays safely below the anomaly threshold, indicating the grid can handle this level of growth without compromising the anomaly detection system.")
        else:
            st.warning(f"Forecast horizon is too short to create a full sequence for anomaly analysis. Please choose a forecast of at least {AUTOENCODER_SEQ_LEN} blocks.")