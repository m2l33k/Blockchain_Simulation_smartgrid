# utils/feature_extractor.py

import numpy as np
import pandas as pd # Import pandas for categorical conversion
from typing import Dict

def extract_features_from_block(block_data: dict, prev_block_timestamp: float) -> dict:
    """Calculates rich, aggregated features from the transactions AND metadata of a single block."""
    
    transactions = block_data.get('transactions', [])
    num_transactions = len(transactions)
    
    features = {
        # FIX: Add block-level metadata directly here
        'block_index': block_data.get('index', 0),
        'miner_id': block_data.get('miner_id', 'unknown'),
        'num_tx': num_transactions,
        'time_since_last_block': block_data.get('timestamp', 0) - prev_block_timestamp,
        
        # Transaction-based features (default to 0)
        'avg_energy': 0, 'std_dev_energy': 0, 'total_energy': 0,
        'avg_amount': 0, 'std_dev_amount': 0, 'total_amount': 0,
        'num_offers': 0, 'num_requests': 0, 'num_deliveries': 0,
        'num_payments': 0, 'num_theft_attempts': 0, 'num_alerts': 0,
        'unique_senders': 0, 'unique_recipients': 0
    }

    if num_transactions > 0:
        energy_values = [tx.get('energy', 0) for tx in transactions]
        amount_values = [tx.get('amount', 0) for tx in transactions]
        tx_types = [tx.get('type', '') for tx in transactions]
        senders = [tx.get('sender', '') for tx in transactions]
        recipients = [tx.get('recipient', '') for tx in transactions]

        features.update({
            'avg_energy': np.mean(energy_values), 'std_dev_energy': np.std(energy_values),
            'total_energy': np.sum(energy_values), 'avg_amount': np.mean(amount_values),
            'std_dev_amount': np.std(amount_values), 'total_amount': np.sum(amount_values),
            'num_offers': tx_types.count('energy_offer'), 'num_requests': tx_types.count('energy_request'),
            'num_deliveries': tx_types.count('energy_delivery'),
            'num_payments': sum(1 for t in tx_types if t.endswith('payment')), # Catches all payment types
            'num_theft_attempts': tx_types.count('fraudulent_payment_attempt'),
            'num_alerts': sum(1 for t in tx_types if t.startswith('alert_')),
            'unique_senders': len(set(senders)), 'unique_recipients': len(set(recipients)),
        })
    return features