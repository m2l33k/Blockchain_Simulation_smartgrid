# fraud_detector.py (Final Corrected Version)

import time
import logging
import threading
import numpy as np
import pandas as pd
import json
import os
from typing import List, Optional
from collections import deque
from tensorflow.keras.models import load_model
import joblib

from utils.feature_extractor import extract_features_from_block
from models.blockchain import Blockchain
try:
    from train_model import TransformerEncoderBlock
except ImportError:
    TransformerEncoderBlock = None

logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
        self.sequence_length = 20
        self.active = False
        self.thread: Optional[threading.Thread] = None
        
        # Load all assets and set the readiness flag
        self.model = self._load_asset('saved_models/anomaly_detection_hybrid_model.keras', load_model)
        self.scaler = self._load_asset('saved_models/data_scaler.joblib', joblib.load)
        self.feature_columns = self._load_asset('saved_models/feature_columns.json', json.load)
        self.miner_id_mapping = self._load_asset('saved_models/miner_id_mapping.json', json.load)
        threshold_data = self._load_asset('saved_models/alert_threshold.json', json.load)
        self.alert_threshold = threshold_data.get('threshold', 0.85) if threshold_data else 0.85
        
        self.is_ready = all([self.model, self.scaler, self.feature_columns, self.miner_id_mapping])
        if self.is_ready:
            logging.info(f"FraudDetector initialized successfully. Alert threshold set to {self.alert_threshold:.4f}")
        else:
            logging.error("FraudDetector failed to initialize due to missing assets.")
            
        # Use a deque as an efficient rolling window for features
        self.feature_history = deque(maxlen=self.sequence_length)

    def _load_asset(self, path: str, loader_func):
        if not os.path.exists(path):
            logging.error(f"Asset not found at path: {path}")
            return None
        try:
            if loader_func == json.load:
                with open(path, 'r') as f:
                    return loader_func(f)
            elif loader_func == load_model:
                if TransformerEncoderBlock is None:
                    logging.warning("TransformerEncoderBlock class not imported; model loading might fail if it's a custom object.")
                # Pass the custom object so Keras knows how to load the model
                return loader_func(path, custom_objects={"TransformerEncoderBlock": TransformerEncoderBlock})
            else:
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
        """Processes a single new block, updates the feature history, and predicts."""
        with self.blockchain.lock:
            current_block_data = self.blockchain.chain[block_index].to_dict()
            prev_block_timestamp = self.blockchain.chain[block_index - 1].timestamp
        
        features = extract_features_from_block(current_block_data, prev_block_timestamp)
        self.feature_history.append(features)
        
        if len(self.feature_history) < self.sequence_length:
            return # Wait for a full sequence

        live_df = pd.DataFrame(list(self.feature_history))
        
        # --- FIX: Convert string 'miner_id' to the numerical code the model expects ---
        live_df['miner_id'] = live_df['miner_id'].map(self.miner_id_mapping).fillna(-1).astype(int)
        
        # Reorder columns to match the training data exactly
        live_features_ordered = live_df[self.feature_columns]
        
        # Scale data using the loaded scaler
        scaled_live_features = self.scaler.transform(live_features_ordered)
        
        if np.isnan(scaled_live_features).any():
            scaled_live_features = np.nan_to_num(scaled_live_features)
            
        live_sequence = np.expand_dims(scaled_live_features, axis=0)
        
        prediction_proba = self.model.predict(live_sequence, verbose=0)[0][0]
        
        is_alert = prediction_proba > self.alert_threshold
        log_level = logging.WARNING if is_alert else logging.INFO
        
        logger.log(log_level, f"Detector check on Block #{block_index}: Score = {prediction_proba:.4f} (Threshold = {self.alert_threshold:.4f})")
        
        if is_alert:
            alert_message = (f"!!! LIVE ALERT: Anomaly detected in Block #{block_index} with {prediction_proba:.2%} confidence !!!")
            print(f"\n🚨 {alert_message}\n")