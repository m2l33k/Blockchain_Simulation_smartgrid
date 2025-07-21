# fraud_detector.py

import time
import logging
import threading
import numpy as np
import pandas as pd
import json
import os
from typing import Optional

from tensorflow.keras.models import load_model
import joblib

from utils.feature_extractor import extract_features_from_block
from models.blockchain import Blockchain
from models.custom_layers import TransformerEncoderBlock

logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
        self.sequence_length = 20  # This must match the training script
        self.active = False
        self.thread: Optional[threading.Thread] = None

        # --- Load all assets ---
        custom_objects = {"TransformerEncoderBlock": TransformerEncoderBlock}
        self.model = self._load_asset('saved_models/anomaly_detection_hybrid_model.keras', load_model, custom_objects)
        self.scaler = self._load_asset('saved_models/data_scaler.joblib', joblib.load)
        self.feature_columns = self._load_asset('saved_models/feature_columns.json', json.load)
        self.miner_id_mapping = self._load_asset('saved_models/miner_id_mapping.json', json.load)
        threshold_data = self._load_asset('saved_models/alert_threshold.json', json.load)
        
        self.alert_threshold = threshold_data.get('threshold', 0.6) if threshold_data else 0.6
        
        self.is_ready = all([self.model, self.scaler, self.feature_columns, self.miner_id_mapping])
        if self.is_ready:
            logging.info(f"FraudDetector initialized successfully. Alert threshold set to {self.alert_threshold:.4f}")
        else:
            logging.error("FraudDetector failed to initialize due to missing assets.")
            
        # NOTE: The self.feature_history deque has been removed as it's no longer needed.

    def _load_asset(self, path: str, loader_func, custom_objects=None):
        """Loads a file asset (model, scaler, or JSON)."""
        if not os.path.exists(path):
            logging.error(f"Asset not found at path: {path}")
            return None
        try:
            if loader_func == json.load:
                with open(path, 'r') as f:
                    return loader_func(f)
            elif loader_func == load_model:
                return loader_func(path, custom_objects=custom_objects, safe_mode=False)
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
        """
        CORRECTED: A robust loop that processes a block only when its full
        preceding sequence is available.
        """
        # Start predicting from the first block that has a full history.
        # To predict for block at index 20, we need blocks 0-19.
        next_block_to_predict = self.sequence_length
        
        while self.active:
            with self.blockchain.lock:
                chain_length = len(self.blockchain.chain)
            
            # If there's a new block to process that has enough history, predict on it.
            if next_block_to_predict < chain_length:
                self._run_prediction_on_block(next_block_to_predict)
                next_block_to_predict += 1
            else:
                # Wait for more blocks to be mined
                time.sleep(1)

    def _run_prediction_on_block(self, target_block_index: int):
        """
        CORRECTED: Uses the 20 blocks *before* target_block_index to predict its status.
        This now matches the logic from the training script.
        """
        sequence_features = []
        with self.blockchain.lock:
            # Define the slice of the blockchain to use for the feature sequence
            start_index = target_block_index - self.sequence_length
            end_index = target_block_index # The slice is exclusive of the end index

            if start_index < 0:
                logging.warning(f"Attempted to predict on block {target_block_index} without enough history.")
                return

            # Extract features for each block in the sequence
            for i in range(start_index, end_index):
                current_block_data = self.blockchain.chain[i].to_dict()
                # Genesis block (index 0) has no predecessor
                prev_block_timestamp = self.blockchain.chain[i - 1].timestamp if i > 0 else 0
                features = extract_features_from_block(current_block_data, prev_block_timestamp)
                sequence_features.append(features)
        
        # --- Prepare data for the model (same steps as before, but on the correct sequence) ---
        live_df = pd.DataFrame(sequence_features)
        
        # Map categorical 'miner_id' to integer representation
        live_df['miner_id'] = live_df['miner_id'].map(self.miner_id_mapping).fillna(-1).astype(int)
        
        # Ensure the column order matches the training data
        try:
            live_features_ordered = live_df[self.feature_columns]
        except KeyError as e:
            logging.error(f"Mismatched columns when preparing live data for block #{target_block_index}: {e}")
            return
            
        # Scale the features using the loaded scaler
        scaled_live_features = self.scaler.transform(live_features_ordered)
        
        # REMOVED: The dangerous np.nan_to_num call is no longer here.
        # If NaNs appear, it means the feature extractor has a bug, and we want to see that error.
        
        # Reshape data into a single sequence for the model
        live_sequence = np.expand_dims(scaled_live_features, axis=0)
        
        # --- Make a prediction ---
        prediction_proba = self.model.predict(live_sequence, verbose=0)[0][0]
        
        is_alert = prediction_proba > self.alert_threshold
        log_level = logging.WARNING if is_alert else logging.INFO
        
        # Log the result for the correct target block
        logger.log(log_level, f"Detector check on Block #{target_block_index}: Score = {prediction_proba:.4f} (Threshold = {self.alert_threshold:.4f})")
        
        if is_alert:
            alert_message = (f"!!! LIVE ALERT: Anomaly detected in Block #{target_block_index} with {prediction_proba:.2%} confidence !!!")
            logger.warning(alert_message)