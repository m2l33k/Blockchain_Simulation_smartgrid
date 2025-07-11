# data_preprocessor.py (Final, Robust Version)

import pandas as pd
import logging
import json
import os
import time
import numpy as np

# --- Configuration ---
LOG_FILE = 'data_generation_run.log' # The log file created by 'generate' mode
OUTPUT_DATA_DIR = 'data'
OUTPUT_MODEL_DIR = 'saved_models'
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DATA_DIR, 'featurized_labeled_data.csv')
MAPPING_FILE_PATH = os.path.join(OUTPUT_MODEL_DIR, 'miner_id_mapping.json')

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Robustly import the shared feature extractor ---
try:
    from utils.feature_extractor import extract_features_from_block
except ImportError:
    logging.error("Could not import 'extract_features_from_block' from 'utils.feature_extractor'.")
    logging.error("Please ensure the 'utils/feature_extractor.py' file exists and is in your Python path.")
    exit()


def parse_log_file(log_file_path: str) -> pd.DataFrame:
    """
    Parses the structured JSON logs from the simulation, creates a featurized
    dataset, and applies labels based on anomaly injection logs.
    """
    logging.info(f"Starting to parse log file: {log_file_path}")

    if not os.path.exists(log_file_path):
        logging.error(f"Log file '{log_file_path}' not found. Please run a 'generate' simulation first.")
        return pd.DataFrame()

    events = []
    with open(log_file_path, 'r') as f:
        for line in f:
            # Chronologically capture anomaly injection markers
            if "!!! ANOMALY:" in line:
                anomaly_type = "Unknown"
                if "NODE BREAKDOWN" in line: anomaly_type = "Breakdown"
                elif "ENERGY THEFT" in line: anomaly_type = "Theft"
                elif "METER TAMPERING" in line: anomaly_type = "Tampering"
                elif "DoS attack" in line: anomaly_type = "DoS"
                events.append({'type': 'anomaly_injection', 'anomaly_type': anomaly_type})

            # Chronologically capture mined block logs
            elif "MINED_BLOCK:" in line:
                try:
                    json_str = line.split("MINED_BLOCK: ")[1]
                    block_data = json.loads(json_str)
                    events.append({'type': 'block_mined', 'data': block_data})
                except (IndexError, json.JSONDecodeError):
                    # Ignore malformed log lines to prevent crashes
                    continue

    # Process events in order to label data correctly
    labeled_data = []
    expected_anomaly = None
    last_block_timestamp = 0

    for event in events:
        if event['type'] == 'anomaly_injection':
            # When an anomaly is injected, flag the very next block as the suspect
            expected_anomaly = event['anomaly_type']
            logging.info(f"Found '{expected_anomaly}' injection. The next mined block will be labeled as anomalous.")

        elif event['type'] == 'block_mined':
            block = event['data']

            # Skip the genesis block (index 0) as it has no predecessor for time delta calculation
            if block['index'] == 0:
                last_block_timestamp = block.get('timestamp', time.time()) # Set initial timestamp
                continue

            # Ensure timestamp exists for delta calculation, using last known as a fallback
            current_timestamp = block.get('timestamp', last_block_timestamp)
            
            # Generate all features using the shared utility function
            features = extract_features_from_block(block, last_block_timestamp)
            last_block_timestamp = current_timestamp # Update for the next iteration

            # Apply the label if an anomaly was expected from the previous event
            features['is_anomaly'] = 1 if expected_anomaly else 0
            features['anomaly_type'] = expected_anomaly if expected_anomaly else 'Normal'

            labeled_data.append(features)
            # IMPORTANT: Reset the flag after using it once.
            # This ensures only the block immediately following an injection is labeled.
            expected_anomaly = None

    if not labeled_data:
        logging.warning("No block data was parsed from the log file. The output CSV will be empty.")
        return pd.DataFrame()

    logging.info(f"Successfully processed and featurized {len(labeled_data)} blocks.")
    return pd.DataFrame(labeled_data)


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    # Ensure output directories exist
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    # Parse the log file to get the raw featurized DataFrame
    df = parse_log_file(LOG_FILE)

    if df.empty:
        logging.error("Processing stopped because no data could be parsed from the log file.")
        return

    # --- CRITICAL STEP: Create and save the miner_id mapping ---
    # This mapping is essential for converting the text-based 'miner_id' into numbers
    # that the machine learning model can understand.
    logging.info("Creating and saving the 'miner_id' mapping...")
    
    # 1. Get all unique miner IDs from the training data.
    unique_miners = df['miner_id'].astype('category').cat.categories
    
    # 2. Create a dictionary mapping each string name to a unique integer code.
    miner_id_mapping = {cat: i for i, cat in enumerate(unique_miners)}
    
    # 3. Add a special code for 'unknown' miners. This is vital for the live detector,
    # in case it encounters a miner ID it has never seen during training.
    miner_id_mapping['unknown'] = -1

    # 4. Save this mapping dictionary to a JSON file so the fraud_detector can load it.
    with open(MAPPING_FILE_PATH, 'w') as f:
        json.dump(miner_id_mapping, f, indent=4)
    logging.info(f"Successfully saved miner ID mapping to {MAPPING_FILE_PATH}")

    # --- Convert categorical columns to numerical codes for training ---
    # Now that the mapping is saved, we convert the 'miner_id' column in the DataFrame
    # to its corresponding numerical code for the model training process.
    df['miner_id'] = df['miner_id'].astype('category').cat.codes

    # --- Save the final, processed dataset ---
    # This CSV file is the primary input for the train_model.py script.
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    logging.info(f"Featurized and labeled data ready for training, saved to {OUTPUT_CSV_FILE}")

    # --- Display a summary for verification ---
    print("\n--- Featurized Labeled Data Summary (First 5 Rows) ---")
    pd.set_option('display.max_columns', None)
    print(df.head())
    print("\n--- Anomaly Distribution ---")
    print(df['anomaly_type'].value_counts())
    print("\n--------------------------")
    logging.info("Preprocessing complete.")


if __name__ == "__main__":
    main()