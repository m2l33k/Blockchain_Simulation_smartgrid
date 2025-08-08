# data_preprocessor.py (Definitive Parsing Fix)

import pandas as pd
import json
import logging
import os
import time

# Ensure the feature extractor is available
try:
    from utils.feature_extractor import extract_features_from_block
except ImportError:
    logging.error("Could not import 'extract_features_from_block'. Make sure 'utils/feature_extractor.py' exists.")
    exit(1)

# --- Configuration ---
LOG_FILE_PATH = 'simulation_logs/simulation_run_2025-07-26_13-27-36.log'
OUTPUT_DIR = 'data'
MODEL_DIR = 'saved_models'
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, 'featurized_labeled_data.csv')
MAPPING_FILE_PATH = os.path.join(MODEL_DIR, 'miner_id_mapping.json')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_log_file_chronologically(log_file_path: str) -> pd.DataFrame:
    """
    Parses the simulation log file by reading events chronologically.
    This is the most reliable way to label data.
    """
    logging.info(f"Starting to parse log file chronologically: {log_file_path}")

    if not os.path.exists(log_file_path):
        logging.error(f"Log file '{log_file_path}' not found. Please run a 'generate' simulation first.")
        return pd.DataFrame()

    events = []
    with open(log_file_path, 'r') as f:
        for line in f:
            if "!!! ANOMALY:" in line:
                anomaly_type = "Unknown"
                if "NODE BREAKDOWN" in line: anomaly_type = "Breakdown"
                elif "THEFT" in line: anomaly_type = "Theft"
                elif "METER TAMPERING" in line: anomaly_type = "Tampering"
                elif "DoS ATTACK" in line or "DoS attack" in line: anomaly_type = "DoS"
                elif "COORDINATED INAUTHENTIC TRADING" in line: anomaly_type = "Coord_Trade"
                events.append({'type': 'anomaly_injection', 'anomaly_type': anomaly_type})
            
            # --- DEFINITIVE FIX ---
            # This logic is much more robust. It finds the line marker and then
            # finds the first '{' to start parsing the JSON, ignoring any
            # surrounding text or whitespace.
            elif "MINED_BLOCK:" in line:
                try:
                    start_index = line.find('{')
                    if start_index != -1:
                        json_str = line[start_index:]
                        block_data = json.loads(json_str)
                        events.append({'type': 'block_mined', 'data': block_data})
                except json.JSONDecodeError as e:
                    logging.warning(f"Found 'MINED_BLOCK:' but failed to parse JSON: {e} on line: {line.strip()}")
                    continue
    
    logging.info(f"Found {len([e for e in events if e['type'] == 'block_mined'])} blocks and {len([e for e in events if e['type'] == 'anomaly_injection'])} anomaly injections in the log.")

    labeled_data = []
    expected_anomaly = None
    last_block_timestamp = 0

    for event in events:
        if event['type'] == 'anomaly_injection':
            expected_anomaly = event['anomaly_type']
            logging.info(f"Found '{expected_anomaly}' injection. The next mined block will be labeled as anomalous.")
        
        elif event['type'] == 'block_mined':
            block = event['data']
            
            if block['index'] == 0:
                last_block_timestamp = block.get('timestamp', time.time())
                continue

            current_timestamp = block.get('timestamp', last_block_timestamp)
            features = extract_features_from_block(block, last_block_timestamp)
            last_block_timestamp = current_timestamp
            
            features['is_anomaly'] = 1 if expected_anomaly else 0
            features['anomaly_type'] = expected_anomaly if expected_anomaly else 'Normal'
            
            labeled_data.append(features)
            expected_anomaly = None

    if not labeled_data:
        logging.warning("No block data was featurized from the log file.")
        return pd.DataFrame()
    
    logging.info(f"Successfully processed and featurized {len(labeled_data)} blocks.")
    return pd.DataFrame(labeled_data)

def main():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        df = parse_log_file_chronologically(LOG_FILE_PATH)
        if df.empty:
            logging.error("Preprocessing resulted in an empty DataFrame. Aborting.")
            return

        logging.info("Creating and saving the 'miner_id' mapping...")
        unique_miners = df['miner_id'].astype('category').cat.categories
        miner_id_mapping = {cat: i for i, cat in enumerate(unique_miners)}
        miner_id_mapping['unknown'] = -1

        with open(MAPPING_FILE_PATH, 'w') as f:
            json.dump(miner_id_mapping, f, indent=4)
        logging.info(f"Successfully saved miner ID mapping to {MAPPING_FILE_PATH}")

        df['miner_id'] = df['miner_id'].astype('category').cat.codes
        
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        logging.info(f"Successfully generated and saved training data to '{OUTPUT_CSV_PATH}'")
        logging.info(f"Data shape: {df.shape}")
        
        print("\n--- Label Distribution in Generated Data ---")
        print(df['anomaly_type'].value_counts(normalize=True))
        print("------------------------------------------")

    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {e}", exc_info=True)

if __name__ == "__main__":
    main()