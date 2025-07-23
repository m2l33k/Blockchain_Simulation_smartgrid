# create_feature_dataset.py

import mysql.connector
import pandas as pd
import json
import logging
import numpy as np

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


def create_feature_dataset_from_db(config, output_file):
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

        # --- 2. Fetch all blocks, ordered by index ---
        # Ordering is crucial for calculating time_since_last_block
        logging.info(f"Fetching all data from the '{TABLE_NAME}' table, ordered by index...")
        query = f"SELECT block_index, timestamp, nonce, miner_id, transactions FROM {TABLE_NAME} ORDER BY block_index ASC"
        cursor.execute(query)
        all_blocks = cursor.fetchall()

        if not all_blocks:
            logging.warning("The 'blocks' table is empty. No data to process.")
            return

        logging.info(f"Found {len(all_blocks)} blocks to process.")
        
        last_timestamp = None

        # --- 3. Process each block and calculate features ---
        for block_row in all_blocks:
            try:
                transactions_list = json.loads(block_row['transactions'])
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Block {block_row['block_index']} has malformed or empty transactions. Processing with 0 transactions.")
                transactions_list = []

            # --- Feature Calculation ---
            num_transactions = len(transactions_list)

            # Time since last block
            current_timestamp = float(block_row['timestamp'])
            if last_timestamp is None or block_row['block_index'] == 0:
                time_since_last_block = 0.0 # No delta for the first block
            else:
                time_since_last_block = current_timestamp - last_timestamp
            last_timestamp = current_timestamp
            
            # Transaction-based features
            if num_transactions > 0:
                amounts = [float(tx.get('amount', 0)) for tx in transactions_list]
                energies = [float(tx.get('energy', 0)) for tx in transactions_list]
                senders = [tx.get('sender') for tx in transactions_list]
                recipients = [tx.get('recipient') for tx in transactions_list]
                types = [tx.get('type') for tx in transactions_list]

                total_amount_transacted = sum(amounts)
                avg_amount_transacted = total_amount_transacted / num_transactions
                # Standard deviation is 0 if there's only one transaction
                std_dev_amount = np.std(amounts) if num_transactions > 1 else 0.0
                
                total_energy_transacted = sum(energies)
                unique_senders = len(set(senders))
                unique_recipients = len(set(recipients))
                
                energy_payment_count = types.count('energy_payment')
                energy_delivery_count = types.count('energy_delivery')
            else:
                # Set default values for blocks with no transactions
                total_amount_transacted = 0.0
                avg_amount_transacted = 0.0
                std_dev_amount = 0.0
                total_energy_transacted = 0.0
                unique_senders = 0
                unique_recipients = 0
                energy_payment_count = 0
                energy_delivery_count = 0

            # --- Assemble the feature dictionary for this block ---
            features = {
                'block_index': block_row['block_index'],
                'num_transactions': num_transactions,
                'nonce': block_row['nonce'],
                'time_since_last_block': time_since_last_block,
                'miner_id': block_row['miner_id'],
                'total_amount_transacted': total_amount_transacted,
                'avg_amount_transacted': avg_amount_transacted,
                'std_dev_amount': std_dev_amount,
                'total_energy_transacted': total_energy_transacted,
                'unique_senders': unique_senders,
                'unique_recipients': unique_recipients,
                'energy_payment_count': energy_payment_count,
                'energy_delivery_count': energy_delivery_count
            }
            processed_blocks.append(features)

        # --- 4. Convert to DataFrame and save to CSV ---
        if not processed_blocks:
            logging.warning("No blocks were processed. The output file will not be created.")
            return
            
        logging.info(f"Successfully calculated features for {len(processed_blocks)} blocks.")
        df = pd.DataFrame(processed_blocks)
        
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully exported feature dataset to '{output_file}'")

    except mysql.connector.Error as err:
        logging.error(f"MySQL Error: {err}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- 5. Clean up the connection ---
        if db_connection and db_connection.is_connected():
            cursor.close()
            db_connection.close()
            logging.info("MySQL connection closed.")

if __name__ == "__main__":
    create_feature_dataset_from_db(db_config, OUTPUT_CSV_FILE)