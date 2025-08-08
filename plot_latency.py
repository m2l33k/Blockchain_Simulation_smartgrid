
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

LOG_CSV_FILE = 'latency_log.csv'
OUTPUT_DIR = 'visualized_data'

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_and_plot_latency_from_csv(csv_path: str):
    """
    Reads a structured CSV of injection and detection events,
    calculates latency with high precision, and generates plots.
    """
    # --- 1. Load and Prepare Data with High-Precision Timestamp Handling ---
    if not os.path.exists(csv_path):
        logging.error(f"Log file not found: '{csv_path}'. Please run a 'detect' simulation to generate it.")
        return

    try:
        df = pd.read_csv(csv_path)
        
        # --- THIS IS THE FIX ---
        # Treat the 'timestamp' column as numeric (float) Unix timestamps first.
        # Then, convert them to datetime objects, preserving the high precision.
        # The 'unit="s"' tells pandas the numbers are seconds since the epoch.
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort values chronologically to ensure correct pairing
        df = df.sort_values(by='timestamp').reset_index(drop=True)

    except Exception as e:
        logging.error(f"Failed to read or parse CSV file '{csv_path}': {e}")
        return

    # Separate injections and detections
    injections = df[df['event_type'] == 'injection'].copy()
    detections = df[df['event_type'] == 'detection'].copy()

    if injections.empty or detections.empty:
        logging.warning("No injections or detections found to pair.")
        return

    # --- 2. Pair Events and Calculate Latency ---
    latencies = []
    used_detection_indices = set()
    logging.info(f"Pairing {len(injections)} injections with subsequent detections...")
    
    for _, inj_row in injections.iterrows():
        # Find the first available detection that occurred AFTER this injection
        possible_detections = detections[
            (detections['timestamp'] > inj_row['timestamp']) &
            (~detections.index.isin(used_detection_indices))
        ]
        
        if not possible_detections.empty:
            first_detection = possible_detections.iloc[0]
            latency = (first_detection['timestamp'] - inj_row['timestamp']).total_seconds()
            latencies.append({'latency': latency, 'anomaly_type': inj_row['sanitized_details']})
            used_detection_indices.add(first_detection.name)
            logging.info(f"  - Paired injection ({inj_row['sanitized_details']}) with a detection. Latency: {latency:.4f}s")
        else:
            logging.warning(f"  - Could not find a detection for injection ({inj_row['sanitized_details']})")

    if not latencies:
        logging.error("Could not pair any events. Cannot generate plots.")
        return

    latency_df = pd.DataFrame(latencies)
    
    # --- 3. Print Summary Statistics ---
    mean_latency = latency_df['latency'].mean()
    median_latency = latency_df['latency'].median()
    std_dev = latency_df['latency'].std()
    min_latency = latency_df['latency'].min()
    max_latency = latency_df['latency'].max()

    print("\n" + "="*40)
    print("      DETECTION LATENCY ANALYSIS")
    print("="*40)
    print(f"Total Paired Detections: {len(latency_df)}")
    print(f"Mean Latency:              {mean_latency:.2f} seconds")
    print(f"Median Latency:            {median_latency:.2f} seconds")
    print(f"Standard Deviation:        {std_dev:.2f} seconds")
    print(f"Min Latency:               {min_latency:.2f} seconds")
    print(f"Max Latency:               {max_latency:.2f} seconds")
    print("="*40 + "\n")

    # --- 4. Generate and Save Plots ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Histogram
    plt.figure(figsize=(12, 7))
    sns.histplot(data=latency_df, x='latency', bins=25, kde=True, color='dodgerblue')
    plt.axvline(mean_latency, color='red', linestyle='--', label=f'Mean: {mean_latency:.2f}s')
    plt.axvline(median_latency, color='green', linestyle='-', label=f'Median: {median_latency:.2f}s')
    plt.title('Distribution of Anomaly Detection Latency', fontsize=16, weight='bold')
    plt.xlabel('Latency (seconds)', fontsize=12)
    plt.ylabel('Number of Detections', fontsize=12)
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, 'latency_distribution_histogram.png')
    plt.savefig(hist_path)
    plt.close()
    logging.info(f"Latency histogram saved to {hist_path}")
    
    # Plot 2: Boxplot
    plt.figure(figsize=(14, 8))
    order = latency_df.groupby('anomaly_type')['latency'].median().sort_values().index
    sns.boxplot(data=latency_df, x='latency', y='anomaly_type', order=order, palette='viridis')
    plt.title('Detection Latency by Anomaly Type', fontsize=16, weight='bold')
    plt.xlabel('Latency (seconds)', fontsize=12)
    plt.ylabel('Anomaly Type', fontsize=12)
    plt.tight_layout()
    box_path = os.path.join(OUTPUT_DIR, 'latency_by_anomaly_type.png')
    plt.savefig(box_path)
    plt.close()
    logging.info(f"Latency boxplot saved to {box_path}")

if __name__ == "__main__":
    analyze_and_plot_latency_from_csv(LOG_CSV_FILE)