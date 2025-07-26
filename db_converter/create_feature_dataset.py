# create_feature_dataset.py

import mysql.connector
import pandas as pd
import json
import logging
import numpy as np
import os

# --- Configuration ---
# IMPORTANT: Update these with your MySQL connection details
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'blockchain_db'
}
TABLE_NAME = 'blocks'
OUTPUT_CSV_FILE = 'data/featurized_block_data.csv'

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features_from_db(config, output_file):
    """
    Connects to MySQL, fetches block data, calculates features for each block,
    and saves the resulting dataset to a CSV file.
    """
    processed_blocks = []
    db_connection = None

    try:
        # --- 1. Connect to the database ---
        logging.info(f"Connecting to database '{config['database']}' on host '{config['host']}'...")
        db_connection = mysql.connector.connect(**config)
        cursor = db_connection.cursor(dictionary=True)

        # --- 2. Fetch all blocks, ordered by index for correct time calculations ---
        logging.info(f"Fetching all data from the '{TABLE_NAME}' table...")
        query = f"SELECT block_index, timestamp, nonce, miner_id, transactions FROM {TABLE_NAME} ORDER BY block_index ASC"
        cursor.execute(query)
        all_blocks_raw = cursor.fetchall()

        if not all_blocks_raw:
            logging.warning("The 'blocks' table is empty. No data to process.")
            return

        logging.info(f"Found {len(all_blocks_raw)} blocks to process.")
        
        last_timestamp = None

        # --- 3. Process each block and calculate features ---
        for block_row in all_blocks_raw:
            try:
                transactions = json.loads(block_row['transactions'])
            except (json.JSONDecodeError, TypeError):
                transactions = []

            num_transactions = len(transactions)
            current_timestamp = float(block_row['timestamp'])
            
            # Calculate time since last block (handle genesis block)
            time_since_last_block = 0.0 if last_timestamp is None or block_row['block_index'] == 0 else current_timestamp - last_timestamp
            last_timestamp = current_timestamp
            
            # Initialize feature dictionary
            features = {
                'block_index': block_row['block_index'],
                'num_transactions': num_transactions,
                'nonce': block_row['nonce'],
                'time_since_last_block': time_since_last_block,
                'miner_id': block_row['miner_id']
            }
            
            # Calculate transaction-based features
            if num_transactions > 0:
                amounts = np.array([tx.get('amount', 0) for tx in transactions])
                energies = np.array([tx.get('energy', 0) for tx in transactions])
                types = [tx.get('type', 'unknown') for tx in transactions]
                senders = [tx.get('sender') for tx in transactions]

                features.update({
                    'total_amount_transacted': np.sum(amounts),
                    'avg_amount_transacted': np.mean(amounts),
                    'std_dev_amount': np.std(amounts) if num_transactions > 1 else 0.0,
                    'total_energy_transacted': np.sum(energies),
                    'unique_senders': len(set(senders)),
                    'unique_recipients': len(set(tx.get('recipient') for tx in transactions)),
                    'energy_payment_count': types.count('energy_payment'),
                    'energy_delivery_count': types.count('energy_delivery'),
                    'sender_concentration': len(set(senders)) / num_transactions,
                    'unusual_tx_type_count': len([t for t in types if t not in ['energy_payment', 'energy_delivery']]),
                    'payment_to_delivery_ratio': types.count('energy_payment') / (types.count('energy_delivery') + 1e-6)
                })
            else: # Default values for blocks with no transactions
                default_keys = [
                    'total_amount_transacted', 'avg_amount_transacted', 'std_dev_amount', 
                    'total_energy_transacted', 'unique_senders', 'unique_recipients', 
                    'energy_payment_count', 'energy_delivery_count', 'unusual_tx_type_count', 
                    'payment_to_delivery_ratio']
                for key in default_keys: features[key] = 0.0
                features['sender_concentration'] = 1.0
            
            processed_blocks.append(features)

        # --- 4. Convert to DataFrame and save to CSV ---
        if not processed_blocks:
            logging.warning("No blocks were processed. The output file will not be created.")
            return
            
        logging.info(f"Successfully calculated features for {len(processed_blocks)} blocks.")
        df = pd.DataFrame(processed_blocks)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully exported feature dataset to '{output_file}'")

    except mysql.connector.Error as err:
        logging.error(f"MySQL Error: {err}")
    finally:
        if db_connection and db_connection.is_connected():
            cursor.close()
            db_connection.close()
            logging.info("MySQL connection closed.")

if __name__ == "__main__":
    extract_features_from_db(db_config, OUTPUT_CSV_FILE)