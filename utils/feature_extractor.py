# utils/feature_extractor.py
import numpy as np
from typing import Dict, Any

def extract_features_from_block(block_data: Dict[str, Any], prev_block_timestamp: float) -> Dict[str, Any]:
    """
    Extracts a flat dictionary of numerical features from a block's data.
    """
    transactions = block_data.get('transactions', [])
    num_tx = len(transactions)
    
    time_since_last_block = block_data.get('timestamp', 0) - prev_block_timestamp
    
    if num_tx > 0:
        tx_amounts = [tx.get('amount', 0) for tx in transactions]
        tx_energies = [tx.get('energy', 0) for tx in transactions]
        
        tx_types = [tx.get('type', 'unknown') for tx in transactions]
        type_counts = {
            'energy_payment_count': tx_types.count('energy_payment'),
            'energy_delivery_count': tx_types.count('energy_delivery'),
            'spam_ramp_up_count': tx_types.count('spam_ramp_up'),
            'spam_peak_count': tx_types.count('spam_peak'),
            'wash_trade_count': tx_types.count('wash_trade_payment'),
            'fraudulent_count': tx_types.count('fraudulent_payment'),
        }

        total_amount = sum(tx_amounts)
        avg_amount = np.mean(tx_amounts) if tx_amounts else 0
        std_amount = np.std(tx_amounts) if tx_amounts else 0
        total_energy = sum(tx_energies)
        
        unique_senders = len(set(tx.get('sender') for tx in transactions))
        unique_recipients = len(set(tx.get('recipient') for tx in transactions))
    else:
        type_counts = {k: 0 for k in ['energy_payment_count', 'energy_delivery_count', 'spam_ramp_up_count', 'spam_peak_count', 'wash_trade_count', 'fraudulent_count']}
        total_amount, avg_amount, std_amount, total_energy = 0, 0, 0, 0
        unique_senders, unique_recipients = 0, 0

    features = {
        'num_transactions': num_tx,
        'nonce': block_data.get('nonce', 0),
        'time_since_last_block': time_since_last_block,
        'miner_id': block_data.get('miner_id', 'unknown'),
        'total_amount_transacted': total_amount,
        'avg_amount_transacted': avg_amount,
        'std_dev_amount': std_amount,
        'total_energy_transacted': total_energy,
        'unique_senders': unique_senders,
        'unique_recipients': unique_recipients,
        **type_counts
    }
    
    return features