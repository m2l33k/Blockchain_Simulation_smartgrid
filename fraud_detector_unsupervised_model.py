# fraud_detector_unsupervised_model.py (Final Version with Heuristics)

import time
import logging
import threading
import numpy as np
import pandas as pd
import json
import os
from typing import Optional, Dict, Any
from collections import deque, Counter
from tensorflow.keras.models import load_model
import joblib

# Import the latency recording utility
from utils.latency_recorder import record_latency_event
from utils.feature_extractor import extract_features_from_block
from models.blockchain import Blockchain

logger = logging.getLogger("simulation_logger") # Use the main simulation logger

class UnsupervisedFraudDetector:
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
        self.sequence_length = 10  # Must match train_unsupervised_model.py
        self.active = False
        self.thread: Optional[threading.Thread] = None
        
        # Load all assets from the unsupervised training
        self.model = self._load_asset('saved_models/lstm_autoencoder_model.keras', load_model)
        self.scaler = self._load_asset('saved_models/data_scaler.joblib', joblib.load)
        self.feature_columns = self._load_asset('saved_models/feature_columns.json', json.load)
        self.miner_id_mapping = self._load_asset('saved_models/miner_id_mapping.json', json.load)
        
        threshold_data = self._load_asset('saved_models/anomaly_threshold.json', json.load)
        self.anomaly_threshold = threshold_data.get('threshold', 0.5) if threshold_data else 0.5

        self.is_ready = all([self.model, self.scaler, self.feature_columns, self.miner_id_mapping])
        
        if self.is_ready:
            logger.info(f"FraudDetector (Unsupervised) initialized. Threshold: {self.anomaly_threshold:.4f}")
        else:
            logger.error("FraudDetector (Unsupervised) failed to initialize due to missing assets.")
            
        self.feature_history = deque(maxlen=self.sequence_length)

    def _load_asset(self, path: str, loader_func):
        if not os.path.exists(path):
            logger.error(f"Asset not found: {path}")
            return None
        try:
            return loader_func(path) if loader_func != json.load else json.load(open(path, 'r'))
        except Exception as e:
            logger.error(f"Failed to load asset from {path}: {e}", exc_info=True)
            return None

    def start(self):
        if not self.is_ready: return
        self.active = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Unsupervised Fraud Detector started.")

    def stop(self):
        self.active = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Unsupervised Fraud Detector stopped.")

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
        with self.blockchain.lock:
            if block_index < self.sequence_length: return # Need history to form a sequence
            
            # Create a slice of the chain for the sequence
            block_sequence = self.blockchain.chain[block_index - self.sequence_length + 1 : block_index + 1]
            if len(block_sequence) < self.sequence_length: return

            # Extract features for the entire sequence
            sequence_features = []
            last_ts = self.blockchain.chain[block_index - self.sequence_length].timestamp
            for block in block_sequence:
                block_data = block.to_dict()
                features = extract_features_from_block(block_data, last_ts)
                sequence_features.append(features)
                last_ts = block.timestamp
        
        live_df = pd.DataFrame(sequence_features)
        live_df['miner_id'] = live_df['miner_id'].map(self.miner_id_mapping).fillna(-1).astype(int)
        live_features_ordered = live_df[self.feature_columns]
        
        scaled_live_features = self.scaler.transform(live_features_ordered)
        live_sequence = np.expand_dims(scaled_live_features, axis=0)
        
        reconstructed_sequence = self.model.predict(live_sequence, verbose=0)
        reconstruction_error = np.mean(np.abs(live_sequence - reconstructed_sequence))
        
        is_alert = reconstruction_error > self.anomaly_threshold
        log_level = logging.WARNING if is_alert else logging.INFO
        
        logger.log(log_level, f"Detector check on Block #{block_index}: Error = {reconstruction_error:.4f} (Threshold = {self.anomaly_threshold:.4f})")
        
        if is_alert:
            logger.warning(f"!!! LIVE ALERT (Unsupervised): Anomaly detected in Block #{block_index} !!!")
            # --- HEURISTIC ENGINE to create a specific event description ---
            self.characterize_anomaly_and_record_latency(block_index)

    def characterize_anomaly_and_record_latency(self, block_index: int):
        """Analyzes transactions in an anomalous block to create a specific event description."""
        with self.blockchain.lock:
            block_data = self.blockchain.chain[block_index].to_dict()
        
        transactions = block_data.get('transactions', [])
        if not transactions: return

        # Heuristic 1: Node Breakdown
        for tx in transactions:
            if tx['type'].startswith('alert_node_offline'):
                node_id = tx['type'].split(':')[1]
                description = f"Breakdown on {node_id}"
                record_latency_event('detection', description)
                return

        # Heuristic 2: Energy Theft
        for tx in transactions:
            if tx['type'] == 'fraudulent_payment':
                description = f"Theft from {tx['sender']} by {tx['recipient']}"
                record_latency_event('detection', description)
                return
        
        # Heuristic 3: DoS Attack (many transactions from one source)
        senders = [tx['sender'] for tx in transactions]
        if senders:
            sender_counts = Counter(senders)
            most_common_sender, count = sender_counts.most_common(1)[0]
            if count / len(transactions) > 0.5: # If one sender made >50% of txs
                description = f"DoS from {most_common_sender}"
                record_latency_event('detection', description)
                return
        
        # Heuristic 4: Coordinated Trading (back-and-forth between two nodes)
        pairs = {"-".join(sorted([tx['sender'], tx['recipient']])) for tx in transactions if tx['type'] == 'wash_trade_payment'}
        if len(pairs) == 1: # All wash trades happened between the same pair
            pair_string = list(pairs)[0]
            node_a, node_b = pair_string.split('-')
            description = f"Coordinated trading between {node_a} and {node_b}"
            record_latency_event('detection', description)
            return

        # Fallback for other anomalies like Meter Tampering
        # We can try to guess the tampered node by seeing who benefited most
        recipients = [tx['recipient'] for tx in transactions if tx['type'] == 'energy_payment']
        if recipients:
            beneficiary = Counter(recipients).most_common(1)[0][0]
            description = f"Tampering on {beneficiary}" # This is a guess
            record_latency_event('detection', description)
            return