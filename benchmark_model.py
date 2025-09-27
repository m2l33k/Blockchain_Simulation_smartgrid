import time
import timeit
import json
import joblib
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Constants & Paths ---
MODEL_DIR = 'saved_models'
SEQUENCE_LENGTH = 10 # Must match the training script

def load_all_models_and_assets():
    """Loads all necessary components for prediction."""
    print("Loading models and assets...")
    assets = {}
    assets['autoencoder'] = load_model(os.path.join(MODEL_DIR, 'lstm_autoencoder_model.keras'))
    assets['iforest'] = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest_model.joblib'))
    assets['meta_learner'] = joblib.load(os.path.join(MODEL_DIR, 'meta_learner_model.joblib'))
    assets['lstm_score_scaler'] = joblib.load(os.path.join(MODEL_DIR, 'lstm_score_scaler.joblib'))
    assets['if_score_scaler'] = joblib.load(os.path.join(MODEL_DIR, 'if_score_scaler.joblib'))
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'r') as f:
        assets['features'] = json.load(f)
    print("...loading complete.")
    return assets

def get_full_prediction(sequence, assets):
    """
    Encapsulates the entire prediction pipeline for one sequence.
    This is the function we will benchmark.
    """
    sequence_batch = np.expand_dims(sequence, axis=0)
    reconstruction = assets['autoencoder'].predict(sequence_batch, verbose=0)
    lstm_score = np.mean(np.abs(reconstruction - sequence_batch), axis=(1, 2))

    nsamples, nx, ny = sequence_batch.shape
    sequence_flat = sequence_batch.reshape((nsamples, nx * ny))
    if_score = -1 * assets['iforest'].decision_function(sequence_flat)

    norm_lstm_score = assets['lstm_score_scaler'].transform(lstm_score.reshape(-1, 1))
    norm_if_score = assets['if_score_scaler'].transform(if_score.reshape(-1, 1))
    
    X_meta = np.hstack([norm_lstm_score, norm_if_score])
    final_score = assets['meta_learner'].predict_proba(X_meta)[:, 1]
    
    return final_score

def main():
    assets = load_all_models_and_assets()
    
    num_features = len(assets['features'])
    sample_sequence = np.random.rand(SEQUENCE_LENGTH, num_features)

    print("\nPerforming warm-up prediction...")
    _ = get_full_prediction(sample_sequence, assets)
    print("...warm-up complete.")

    print("\n--- Method 1: Simple Single Prediction Timing ---")
    start_time = time.time()
    get_full_prediction(sample_sequence, assets)
    end_time = time.time()
    latency_simple = end_time - start_time
    print(f"Single prediction latency: {latency_simple * 1000:.2f} ms")

    print("\n--- Method 2: Averaging Multiple Predictions ---")
    num_runs = 100
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        get_full_prediction(sample_sequence, assets)
        end = time.time()
        latencies.append(end - start)
    
    latencies_ms = [l * 1000 for l in latencies]
    avg_latency = np.mean(latencies_ms)
    median_latency = np.median(latencies_ms)
    std_latency = np.std(latencies_ms)
    min_latency = np.min(latencies_ms)
    max_latency = np.max(latencies_ms)
    throughput = 1000 / avg_latency if avg_latency > 0 else float('inf')

    print(f"Ran {num_runs} predictions:")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  Median Latency:  {median_latency:.2f} ms")
    print(f"  Std Dev:         {std_latency:.2f} ms")
    print(f"  Min Latency:     {min_latency:.2f} ms")
    print(f"  Max Latency:     {max_latency:.2f} ms")
    print(f"  Throughput:      {throughput:.2f} predictions/sec")

    results = {
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_preds_per_sec": throughput
    }
    with open('model_latency_report.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nLatency report saved to 'model_latency_report.json'")

if __name__ == "__main__":
    main()