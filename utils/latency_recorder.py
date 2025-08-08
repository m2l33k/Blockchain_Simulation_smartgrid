import time
import threading
import os

class LatencyRecorder:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # This makes the class a singleton, ensuring only one instance ever exists
        # and that the file is only initialized once per application run.
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LatencyRecorder, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # The __init__ is called every time, but our logic only runs once
        if self._initialized:
            return
        
        with self._lock:
            self.file_path = 'latency_log.csv'
            # Initialize the file with a header if it doesn't exist
            if not os.path.exists(self.file_path):
                with open(self.file_path, 'w') as f:
                    f.write("timestamp,event_type,details\n")
            self._initialized = True

    def record_event(self, event_type: str, details: str):
        """Records a timestamped event to the CSV log file in a thread-safe manner."""
        timestamp = time.time()
        # Sanitize details by removing commas to not break the CSV format
        sanitized_details = str(details).replace(',', ';')
        
        with self._lock:
            try:
                with open(self.file_path, 'a') as f:
                    f.write(f"{timestamp},{event_type},{sanitized_details}\n")
            except Exception as e:
                # Use print for critical errors as logging might not be configured
                print(f"CRITICAL ERROR writing to latency log: {e}")

# Create a single, globally accessible instance of the recorder
recorder = LatencyRecorder()

# Define simple functions that are easy to import and use this global instance
def record_latency_event(event_type: str, details: str):
    recorder.record_event(event_type, details)

def clear_latency_log():
    # This function now safely handles clearing the log
    with recorder._lock:
        if os.path.exists(recorder.file_path):
            os.remove(recorder.file_path)
        # Re-initialize it immediately with the header
        with open(recorder.file_path, 'w') as f:
            f.write("timestamp,event_type,details\n")