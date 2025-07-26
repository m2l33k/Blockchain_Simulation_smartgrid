# utils/latency_recorder.py

import time
import threading
import os

# Use a lock to prevent race conditions when writing to the file from different threads
LATENCY_FILE_LOCK = threading.Lock()
LATENCY_FILE_PATH = 'latency_log.csv'

def initialize_latency_log():
    """Creates the latency log file with a header if it doesn't exist."""
    if not os.path.exists(LATENCY_FILE_PATH):
        with LATENCY_FILE_LOCK:
            with open(LATENCY_FILE_PATH, 'w') as f:
                f.write("timestamp,event_type,details\n")

def record_latency_event(event_type: str, details: str):
    """
    Records a timestamped event to the latency log file in a thread-safe manner.

    Args:
        event_type (str): Either 'injection' or 'detection'.
        details (str): Information about the event (e.g., anomaly type, block index).
    """
    timestamp = time.time()
    # Sanitize details by removing commas to not break the CSV format
    sanitized_details = details.replace(',', ';')
    
    with LATENCY_FILE_LOCK:
        try:
            with open(LATENCY_FILE_PATH, 'a') as f:
                f.write(f"{timestamp},{event_type},{sanitized_details}\n")
        except Exception as e:
            print(f"Error writing to latency log: {e}")