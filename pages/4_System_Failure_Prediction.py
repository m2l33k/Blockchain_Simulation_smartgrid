import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Failure Prediction")
st.title("System Failure Prediction")
st.markdown("""
This tool predicts the scalability limits of the smart grid. You can either perform a **Manual Test** for a specific grid size or use the **Auto-Finder** to see a comparative analysis of when each model component (LSTM, Isolation Forest, and the final Stacked model) would fail under increasing load.
""")

# --- Caching & Loading Functions ---
@st.cache_resource
def load_models_and_assets():
    MODEL_DIR = 'saved_models'
    try:
        assets = {}
        assets['autoencoder'] = load_model(os.path.join(MODEL_DIR, 'lstm_autoencoder_model.keras'))
        assets['forecaster'] = load_model(os.path.join(MODEL_DIR, 'forecasting_model.keras'))
        assets['iforest'] = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest_model.joblib'))
        assets['meta_learner'] = joblib.load(os.path.join(MODEL_DIR, 'meta_learner_model.joblib'))
        assets['data_scaler'] = joblib.load(os.path.join(MODEL_DIR, 'data_scaler.joblib'))
        assets['lstm_score_scaler'] = joblib.load(os.path.join(MODEL_DIR, 'lstm_score_scaler.joblib'))
        assets['if_score_scaler'] = joblib.load(os.path.join(MODEL_DIR, 'if_score_scaler.joblib'))
        with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'r') as f:
            assets['features'] = json.load(f)
        with open(os.path.join(MODEL_DIR, 'final_anomaly_threshold.json'), 'r') as f:
            assets['threshold_stacked'] = json.load(f)['threshold']
        assets['threshold_lstm'] = 0.8
        assets['threshold_iforest'] = 0.8
        return assets
    except Exception as e:
        st.error(f"Failed to load a required model asset: {e}.")
        return None

@st.cache_data
def load_historical_data():
    DATA_DIR = 'data'
    DATA_PATH = os.path.join(DATA_DIR, 'featurized_block_data.csv')
    df = pd.read_csv(DATA_PATH)
    if 'miner_id' in df.columns: df['miner_id'] = df['miner_id'].astype('category').cat.codes
    return df

# --- Core Prediction Logic ---
def run_full_analysis(growth_factor, forecast_steps, df_historical, assets):
    FORECAST_INPUT_LEN, FORECAST_HORIZON, AUTOENCODER_SEQ_LEN = 20, 10, 10
    last_sequence = assets['data_scaler'].transform(df_historical[assets['features']].tail(FORECAST_INPUT_LEN))
    current_sequence = np.expand_dims(last_sequence, axis=0)
    all_forecasts_scaled = []
    for _ in range(int(np.ceil(forecast_steps / FORECAST_HORIZON))):
        predicted_chunk = assets['forecaster'].predict(current_sequence, verbose=0)[0]
        all_forecasts_scaled.extend(predicted_chunk)
        current_sequence = np.expand_dims(np.array(all_forecasts_scaled[-FORECAST_INPUT_LEN:]), axis=0)
    final_forecast_scaled = np.array(all_forecasts_scaled)[:forecast_steps]
    stressed_forecast_scaled = final_forecast_scaled.copy()
    features_to_stress = ['num_transactions', 'total_amount_transacted', 'total_energy_transacted', 'unique_senders', 'unique_recipients']
    for feature in features_to_stress:
        if feature in assets['features']:
            feature_index = assets['features'].index(feature)
            stressed_forecast_scaled[:, feature_index] *= growth_factor
    sequences = np.array([stressed_forecast_scaled[i:i+AUTOENCODER_SEQ_LEN] for i in range(len(stressed_forecast_scaled) - AUTOENCODER_SEQ_LEN + 1)])
    if sequences.shape[0] == 0: return None
    reconstructions = assets['autoencoder'].predict(sequences, verbose=0)
    lstm_scores = np.mean(np.abs(reconstructions - sequences), axis=(1, 2))
    sequences_flat = sequences.reshape((sequences.shape[0], -1))
    if_scores = -1 * assets['iforest'].decision_function(sequences_flat)
    norm_lstm_scores = assets['lstm_score_scaler'].transform(lstm_scores.reshape(-1, 1))
    norm_if_scores = assets['if_score_scaler'].transform(if_scores.reshape(-1, 1))
    X_meta = np.hstack([norm_lstm_scores, norm_if_scores])
    final_scores = assets['meta_learner'].predict_proba(X_meta)[:, 1]
    return {
        "final_scores": final_scores,
        "lstm_scores": norm_lstm_scores.flatten(),
        "iforest_scores": norm_if_scores.flatten()
    }

# --- Initialize Session State ---
if 'manual_test_results' not in st.session_state: st.session_state.manual_test_results = None
if 'comparative_results' not in st.session_state: st.session_state.comparative_results = None
if 'comparative_verdict' not in st.session_state: st.session_state.comparative_verdict = None

# --- Load Assets ---
assets = load_models_and_assets()
if assets is None: st.stop()
df_historical = load_historical_data()

# --- Sidebar UI ---
with st.sidebar:
    st.header("Grid Configuration")
    st.subheader("Baseline Grid Size")
    base_consumers = st.number_input("Baseline Consumers", 1, value=50)
    base_prosumers = st.number_input("Baseline Prosumers", 1, value=80)
    base_total = base_consumers + base_prosumers
    
    st.subheader("Manual Test Target Size")
    target_consumers = st.number_input("Target Consumers", min_value=base_consumers, value=base_consumers * 2)
    target_prosumers = st.number_input("Target Prosumers", min_value=base_prosumers, value=base_prosumers * 2)
    
    forecast_steps = st.slider("Forecast Horizon (Blocks)", 20, 200, 50)

# --- Manual Stress Test Section ---
st.header("Manual Stress Test")
target_total = target_consumers + target_prosumers
growth_factor = target_total / base_total if base_total > 0 else 1.0
st.markdown(f"Test a specific grid size of **{target_total} total devices** ({growth_factor:.2f}x growth).")

if st.button("Run Manual Test"):
    st.session_state.comparative_results = None # Clear other results
    with st.spinner(f"Simulating grid with {target_total} devices..."):
        analysis_results = run_full_analysis(growth_factor, forecast_steps, df_historical, assets)
        st.session_state.manual_test_results = {
            "analysis": analysis_results,
            "target_total": target_total
        }

if st.session_state.manual_test_results:
    results = st.session_state.manual_test_results["analysis"]
    if results:
        st.subheader("Manual Test Results: Anomaly Score Breakdown")
        start_index = len(df_historical) + 10 - 1
        error_index = list(range(start_index, start_index + len(results["final_scores"])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=error_index, y=results["lstm_scores"], mode='lines', name='LSTM Score', line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(x=error_index, y=results["iforest_scores"], mode='lines', name='I. Forest Score', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=error_index, y=results["final_scores"], mode='lines', name='Stacked Probability', line=dict(color='red', width=3)))
        fig.add_hline(y=assets['threshold_stacked'], line_dash="dash", line_color="orange", annotation_text=f"Failure Threshold ({assets['threshold_stacked']:.4f})")
        fig.update_layout(title=f"Anomaly Score Breakdown for {st.session_state.manual_test_results['target_total']} Devices", yaxis_title="Score / Probability")
        st.plotly_chart(fig, use_container_width=True)
        
        max_prob = results["final_scores"].max()
        if max_prob > assets['threshold_stacked']:
            st.error(f"**FAILURE PREDICTED.** Max probability ({max_prob:.4f}) exceeds threshold.")
        else:
            st.success(f"**STABLE.** Max probability ({max_prob:.4f}) is below threshold.")

# --- Comparative Auto-Find Section ---
st.markdown("---")
st.header("Comparative Failure Point Analysis")
st.markdown("Automatically search for the scalability limits of each model component.")

if st.button("🚀 Find All Failure Points"):
    st.session_state.manual_test_results = None
    st.session_state.comparative_results = []
    progress_bar = st.progress(0, text="Starting search...")
    with st.spinner("Iteratively testing grid sizes..."):
        failure_points = {"lstm": -1, "iforest": -1, "stacked": -1}
        search_steps, growth_increment = 30, 0.1
        for i in range(search_steps):
            current_growth = 1.0 + (i + 1) * growth_increment
            current_devices = int(base_total * current_growth)
            analysis_results = run_full_analysis(current_growth, forecast_steps, df_historical, assets)
            if not analysis_results: continue
            
            progress_bar.progress((i + 1) / search_steps, text=f"Testing {current_devices} devices...")
            st.session_state.comparative_results.append({"devices": current_devices, **analysis_results})

            if failure_points["lstm"] == -1 and analysis_results["lstm_scores"].max() > assets["threshold_lstm"]:
                failure_points["lstm"] = current_devices
            if failure_points["iforest"] == -1 and analysis_results["iforest_scores"].max() > assets["threshold_iforest"]:
                failure_points["iforest"] = current_devices
            if failure_points["stacked"] == -1 and analysis_results["final_scores"].max() > assets["threshold_stacked"]:
                failure_points["stacked"] = current_devices
            if all(v != -1 for v in failure_points.values()): break
        st.session_state.comparative_verdict = failure_points
    progress_bar.empty()

if st.session_state.comparative_results:
    st.subheader("Comparative Stress Curve")
    results_df = pd.DataFrame([{
        "devices": r["devices"],
        "lstm_score": r["lstm_scores"].max(),
        "iforest_score": r["iforest_scores"].max(),
        "stacked_score": r["final_scores"].max()
    } for r in st.session_state.comparative_results])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df['devices'], y=results_df['lstm_score'], name='LSTM'))
    fig.add_hline(y=assets['threshold_lstm'], line_dash="dot", line_color="blue", annotation_text="LSTM Threshold")
    fig.add_trace(go.Scatter(x=results_df['devices'], y=results_df['iforest_score'], name='I. Forest'))
    fig.add_hline(y=assets['threshold_iforest'], line_dash="dot", line_color="green", annotation_text="I.Forest Threshold")
    fig.add_trace(go.Scatter(x=results_df['devices'], y=results_df['stacked_score'], name='Stacked Model', line=dict(color='red', width=4)))
    fig.add_hline(y=assets['threshold_stacked'], line_dash="dash", line_color="red", annotation_text="Stacked Threshold")
    fig.update_layout(title="Stress Curve: Model Scores vs. Grid Size", yaxis_title="Max Score / Probability")
    st.plotly_chart(fig, use_container_width=True)

    verdict = st.session_state.comparative_verdict
    if verdict:
        st.subheader("Failure Point Summary")
        summary_df = pd.DataFrame({
            "Model": ["LSTM Autoencoder", "Isolation Forest", "Stacked Meta-Learner"],
            "Predicted Failure Point (Devices)": [f"~ {v}" if v != -1 else "Did not fail" for v in verdict.values()]
        })
        st.table(summary_df)