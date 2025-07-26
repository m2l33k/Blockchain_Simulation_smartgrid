import time
import logging
import threading
import numpy as np
import pandas as pd
import json
import os
from typing import Optional
from collections import deque
from tensorflow.keras.models import load_model
import joblib
from utils.latency_recorder import record_latency_event

from utils.feature_extractor import extract_features_from_block
from models.blockchain import Blockchain

logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
        self.sequence_length = 10  # Must match the sequence length used in training
        self.active = False
        self.thread: Optional[threading.Thread] = None
        
        # --- Load all assets created by the unsupervised training script ---
        self.model = self._load_asset('saved_models/lstm_autoencoder_model.keras', load_model)
        self.scaler = self._load_asset('saved_models/data_scaler.joblib', joblib.load)
        self.feature_columns = self._load_asset('saved_models/feature_columns.json', json.load)
        self.miner_id_mapping = self._load_asset('saved_models/miner_id_mapping.json', json.load)
        
        # Load the anomaly threshold from its dedicated file
        threshold_data = self._load_asset('saved_models/anomaly_threshold.json', json.load)
        # Set a default if the file is somehow missing, but log a warning
        self.anomaly_threshold = threshold_data.get('threshold', 0.5) if threshold_data else 0.5
        if not threshold_data:
            logging.warning("Could not load 'anomaly_threshold.json'. Using a default of 0.5, which may be inaccurate.")

        self.is_ready = all([self.model, self.scaler, self.feature_columns, self.miner_id_mapping])
        
        if self.is_ready:
            logging.info(f"FraudDetector (Unsupervised) initialized successfully. Anomaly threshold set to {self.anomaly_threshold:.4f}")
        else:
            logging.error("FraudDetector failed to initialize due to missing assets.")
            
        # Use a deque for an efficient rolling window of features
        self.feature_history = deque(maxlen=self.sequence_length)

    def _load_asset(self, path: str, loader_func):
        if not os.path.exists(path):
            logging.error(f"Asset not found at path: {path}")
            return None
        try:
            if loader_func == json.load:
                with open(path, 'r') as f:
                    return loader_func(f)
            else: # Covers both joblib and keras.load_model
                return loader_func(path)
        except Exception as e:
            logging.error(f"Failed to load asset from {path}: {e}", exc_info=True)
            return None

    def start(self):
        if not self.is_ready:
            logging.error("FraudDetector cannot start because it is not ready.")
            return

        self.active = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logging.info("Live Fraud Detector started monitoring the blockchain.")

    def stop(self):
        self.active = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logging.info("Live Fraud Detector stopped.")

    def _run_loop(self):
        last_processed_index = 0
        while self.active:
            with self.blockchain.lock:
                chain_length = len(self.blockchain.chain)
            
            if chain_length > last_processed_index:
                for i in range(last_processed_index + 1, chain_length):
                    self.process_new_block(i)
                last_processed_index = chain_length - 1
            
            time.sleep(1)

    def process_new_block(self, block_index: int):
        """Processes a new block, calculates its reconstruction error, and flags anomalies."""
        with self.blockchain.lock:
            # Skip genesis block
            if block_index == 0:
                return
            current_block_data = self.blockchain.chain[block_index].to_dict()
            prev_block_timestamp = self.blockchain.chain[block_index - 1].timestamp
        
        features = extract_features_from_block(current_block_data, prev_block_timestamp)
        self.feature_history.append(features)
        
        if len(self.feature_history) < self.sequence_length:
            return # Not enough data yet to form a full sequence

        # --- Prepare the live sequence for prediction ---
        live_df = pd.DataFrame(list(self.feature_history))
        
        # Convert string 'miner_id' to the numerical code the model expects
        live_df['miner_id'] = live_df['miner_id'].map(self.miner_id_mapping).fillna(-1).astype(int)
        
        # Reorder columns to match the training data exactly
        live_features_ordered = live_df[self.feature_columns]
        
        # Scale data using the loaded scaler
        scaled_live_features = self.scaler.transform(live_features_ordered)
        
        # Reshape for the LSTM model: (1, sequence_length, num_features)
        live_sequence = np.expand_dims(scaled_live_features, axis=0)
        
        # --- Get the model's reconstruction and calculate the error ---
        reconstructed_sequence = self.model.predict(live_sequence, verbose=0)
        reconstruction_error = np.mean(np.abs(live_sequence - reconstructed_sequence))
        
        # --- Compare error to the threshold ---
        is_alert = reconstruction_error > self.anomaly_threshold
        log_level = logging.WARNING if is_alert else logging.INFO
        
        logger.log(log_level, f"Detector check on Block #{block_index}: Error = {reconstruction_error:.4f} (Threshold = {self.anomaly_threshold:.4f})")
        
        if is_alert:
            alert_message = (f"!!! LIVE ALERT: Potential anomaly detected around Block #{block_index}. "
                             f"Reconstruction error ({reconstruction_error:.4f}) exceeded threshold.")
            print(f"\n🚨 {alert_message}\n")
            record_latency_event('detection', f"Block #{block_index}")