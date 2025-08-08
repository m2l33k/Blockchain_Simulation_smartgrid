# train_forecasting_model.py

import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = 'data'
MODEL_DIR = 'saved_models'
VIS_DIR = 'visualized_data'
DATA_PATH = os.path.join(DATA_DIR, 'featurized_block_data.csv')

# --- Model Parameters ---
# Use the past 20 blocks to predict the next 10 blocks
SEQUENCE_LENGTH = 20  # How many past steps to use as input
FORECAST_HORIZON = 10   # How many future steps to predict

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_forecast_sequences(data, input_len, output_len):
    """
    Creates input sequences (X) and their corresponding future sequences (y).
    X shape: (samples, input_len, features)
    y shape: (samples, output_len, features)
    """
    X, y = [], []
    if len(data) < input_len + output_len:
        return np.array(X), np.array(y)
        
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : (i + input_len)])
        y.append(data[(i + input_len) : (i + input_len + output_len)])
    return np.array(X), np.array(y)

def main():
    # --- 1. Setup Environment ---
    for d in [MODEL_DIR, VIS_DIR]: os.makedirs(d, exist_ok=True)
    logging.info("Starting forecasting model training...")

    # --- 2. Load and Preprocess Data ---
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data file not found at '{DATA_PATH}'. Please run data generation and preprocessing first.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # Use the same miner_id mapping and feature columns as the autoencoder for consistency
    df['miner_id'] = df['miner_id'].astype('category').cat.codes
    features_to_use = [col for col in df.columns if col not in ['block_index']]
    features_df = df[features_to_use]
    
    # --- 3. Scale and Create Sequences ---
    # NOTE: We use the *same scaler* as the autoencoder for consistency in the dashboard.
    scaler_path = os.path.join(MODEL_DIR, 'data_scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        scaled_features = scaler.transform(features_df)
        logging.info("Loaded existing data scaler for consistency.")
    else:
        logging.warning("Existing scaler not found. Fitting a new one. This might cause inconsistencies if not intended.")
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features_df)
        joblib.dump(scaler, scaler_path)
    
    X, y = create_forecast_sequences(scaled_features, SEQUENCE_LENGTH, FORECAST_HORIZON)
    
    if X.shape[0] == 0:
        logging.error(f"Not enough data to create forecast sequences (need at least {SEQUENCE_LENGTH + FORECAST_HORIZON} data points).")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 4. Build Seq2Seq LSTM Forecasting Model ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_features = input_shape[1]
    
    # Encoder
    inputs = Input(shape=input_shape)
    encoder_lstm = LSTM(128, activation='relu')
    encoder_outputs = encoder_lstm(inputs)
    
    # Repeat the context vector for each step of the forecast horizon
    decoder_repeat = RepeatVector(FORECAST_HORIZON)(encoder_outputs)
    
    # Decoder
    decoder_lstm = LSTM(128, activation='relu', return_sequences=True)(decoder_repeat)
    output = TimeDistributed(Dense(num_features))(decoder_lstm)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    
    # --- 5. Train the Model ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train, y_train, 
        epochs=150, 
        batch_size=32, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # --- 6. Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Forecasting Model Loss (MAE)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(VIS_DIR, 'forecasting_model_loss.png'))
    plt.close()
    logging.info(f"Forecasting model training history saved to '{VIS_DIR}'.")

    # --- 7. Save Model ---
    model_path = os.path.join(MODEL_DIR, 'forecasting_model.keras')
    model.save(model_path)
    logging.info(f"Forecasting model saved successfully to {model_path}")

if __name__ == "__main__":
    main()