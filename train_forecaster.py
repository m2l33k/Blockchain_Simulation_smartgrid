

import pandas as pd
import numpy as np
import os
import json
import logging
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuration ---
DATA_DIR = 'data'
MODEL_DIR = 'saved_models'
DATA_PATH = os.path.join(DATA_DIR, 'featurized_block_data.csv')
INPUT_LENGTH = 20
HORIZON = 10
EPOCHS = 100
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_forecasting_sequences(data, input_length, horizon):
    X, y = [], []
    for i in range(len(data) - input_length - horizon + 1):
        X.append(data[i:(i + input_length)])
        y.append(data[(i + input_length):(i + input_length + horizon)])
    return np.array(X), np.array(y)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logging.info("Loading and preprocessing data for forecasting...")
    try:
        df = pd.read_csv(DATA_PATH)
        feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.json')
        with open(feature_columns_path, 'r') as f:
            features_to_use = json.load(f)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required file was not found: {e}. Please run the main anomaly detection training script first.")
        return

    if 'miner_id' in features_to_use and 'miner_id' in df.columns:
        df['miner_id'] = df['miner_id'].astype('category').cat.codes
    features_df = df[features_to_use]
    
    try:
        scaler_path = os.path.join(MODEL_DIR, 'data_scaler.joblib')
        scaler = joblib.load(scaler_path)
        scaled_features = scaler.transform(features_df)
    except FileNotFoundError:
        logging.error(f"FATAL: `data_scaler.joblib` not found at '{scaler_path}'. Please run the main anomaly detection training script first.")
        return
        
    logging.info(f"Creating sequences with input length {INPUT_LENGTH} and horizon {HORIZON}...")
    X, y = create_forecasting_sequences(scaled_features, INPUT_LENGTH, HORIZON)

    if X.shape[0] == 0:
        logging.error(f"FATAL: Not enough data to create forecasting sequences. Need at least {INPUT_LENGTH + HORIZON} data points.")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info("Building the LSTM forecasting model...")
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(HORIZON * X_train.shape[2]),
        tf.keras.layers.Reshape([HORIZON, X_train.shape[2]])
    ])
    
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7)]
    
    logging.info("Starting training of the forecasting model...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
              
    model_path = os.path.join(MODEL_DIR, 'forecasting_model.keras')
    model.save(model_path)
    logging.info(f"SUCCESS: Forecasting model successfully trained and saved to '{model_path}'")

if __name__ == "__main__":
    main()