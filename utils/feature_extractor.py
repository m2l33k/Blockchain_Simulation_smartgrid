# utils/feature_extractor.py

import numpy as np

def extract_features_from_block(block: dict, last_block_timestamp: float) -> dict:
    """
    Extracts a rich set of features from a single block dictionary.
    
    Args:
        block (dict): The block data.
        last_block_timestamp (float): The timestamp of the previous block.
        
    Returns:
        dict: A dictionary of calculated features for this block.
    """
    transactions = block.get('transactions', [])
    num_transactions = len(transactions)
    current_timestamp = block.get('timestamp', last_block_timestamp)
    
    features = {
        'block_index': block.get('index'),
        'num_transactions': num_transactions,
        'nonce': block.get('nonce'),
        'time_since_last_block': current_timestamp - last_block_timestamp,
        'miner_id': block.get('miner_id', 'unknown')
    }
    
    if num_transactions > 0:
        amounts = np.array([tx.get('amount', 0) for tx in transactions])
        energies = np.array([tx.get('energy', 0) for tx in transactions])
        types = [tx.get('type', 'unknown') for tx in transactions]
        senders = [tx.get('sender') for tx in transactions]

        # Existing Features
        features['total_amount_transacted'] = np.sum(amounts)
        features['avg_amount_transacted'] = np.mean(amounts)
        features['std_dev_amount'] = np.std(amounts) if num_transactions > 1 else 0.0
        features['total_energy_transacted'] = np.sum(energies)
        features['unique_senders'] = len(set(senders))
        features['unique_recipients'] = len(set(tx.get('recipient') for tx in transactions))
        features['energy_payment_count'] = types.count('energy_payment')
        features['energy_delivery_count'] = types.count('energy_delivery')
        
        # --- NEW FEATURES TO DETECT SPECIFIC ANOMALIES ---
        
        # Feature for DoS attacks: High transaction count from a single source
        features['sender_concentration'] = features['unique_senders'] / num_transactions if num_transactions > 0 else 1.0

        # Feature for unusual transaction types (like 'alert_node_offline', 'fraudulent_payment')
        unusual_types = [t for t in types if t not in ['energy_payment', 'energy_delivery']]
        features['unusual_tx_type_count'] = len(unusual_types)

        # Feature for Coordinated Trading: High ratio of payments to deliveries
        features['payment_to_delivery_ratio'] = features['energy_payment_count'] / (features['energy_delivery_count'] + 1e-6)

    else: # Default values for blocks with no transactions
        default_keys = [
            'total_amount_transacted', 'avg_amount_transacted', 'std_dev_amount', 
            'total_energy_transacted', 'unique_senders', 'unique_recipients', 
            'energy_payment_count', 'energy_delivery_count', 'sender_concentration',
            'unusual_tx_type_count', 'payment_to_delivery_ratio'
        ]
        for key in default_keys:
            if key == 'sender_concentration':
                features[key] = 1.0 # A block with 0 txs has perfect concentration
            else:
                features[key] = 0.0
                
    return features