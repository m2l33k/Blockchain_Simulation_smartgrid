import time
import logging
import argparse
import os
import random
from typing import List

# --- Import the actual classes and utilities ---
from models.blockchain import Blockchain
from models.grid_nodes import GridOperator, Prosumer, Consumer, GridNode
from anomaly_injector import AnomalyInjector

# --- FIX: Import the correct functions from the new latency_recorder ---
# We only need `clear_latency_log` for the main script.
# The injector and detector will import `record_latency_event`.
from utils.latency_recorder import clear_latency_log

# --- FIX: Import the CORRECT class name for the detector ---
from fraud_detector_unsupervised_model import UnsupervisedFraudDetector

# --- Logging Setup ---
logger = logging.getLogger("simulation_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if logger.hasHandlers():
    logger.handlers.clear()

LOG_DIR = 'simulation_logs'
os.makedirs(LOG_DIR, exist_ok=True)
DETECT_LOG_FILE = os.path.join(LOG_DIR, f"live_detection_run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
fh = logging.FileHandler(DETECT_LOG_FILE, mode='w')
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)


class LiveDetectionSimulation:
    def __init__(self, num_prosumers: int, num_consumers: int, difficulty: int):
        logger.info("--- Initializing Live Detection Simulation ---")
        
        # --- FIX: Clear the old log file for a fresh run ---
        clear_latency_log()

        # The blockchain and nodes are initialized without a direct DB link
        # for this live detection simulation mode.
        self.blockchain = Blockchain(difficulty=difficulty)
        self.all_nodes: List[GridNode] = []
        
        self.grid_operator = GridOperator("GRID-OP-01", self.blockchain)
        
        for i in range(num_prosumers):
            node = Prosumer(node_id=f"PRO-{i:02d}", blockchain=self.blockchain, production_capacity=random.uniform(3.0, 8.0), consumption_rate=random.uniform(0.5, 1.5))
            node.connect_to_grid_operator(self.grid_operator)
            self.all_nodes.append(node)

        for i in range(num_consumers):
            node = Consumer(node_id=f"CON-{i:02d}", blockchain=self.blockchain, consumption_rate=random.uniform(1.0, 3.0))
            node.connect_to_grid_operator(self.grid_operator)
            self.all_nodes.append(node)
        
        self.anomaly_injector = AnomalyInjector(self.all_nodes, self.grid_operator, self.blockchain)
        self.fraud_detector = UnsupervisedFraudDetector(self.blockchain)
        
        logger.info("--- Live Simulation Setup Complete ---")

    def run(self, duration: int):
        logger.info(f"--- Starting Live Detection Run ({duration} seconds) ---")
        
        # Start all components (no more direct DB connections to manage here)
        # Note: You'll need to update grid_nodes to remove start/stop if you adopt the tick-based model everywhere.
        # For now, assuming they have start/stop.
        self.grid_operator.start()
        for node in self.all_nodes:
            node.start()
        self.anomaly_injector.start()
        self.fraud_detector.start()

        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user.")
        finally:
            logger.info("Shutting down simulation components...")
            self.fraud_detector.stop()
            self.anomaly_injector.stop()
            for node in self.all_nodes:
                node.stop()
            self.grid_operator.stop()
            logger.info("--- Live detection simulation finished. ---")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Run the live anomaly detection simulation.")
        # ... (add your arguments here if needed)
        parser.add_argument('--duration', type=int, default=120, help="Duration of the simulation in seconds.")
        args = parser.parse_args()

        sim = LiveDetectionSimulation(num_prosumers=10, num_consumers=20, difficulty=3)
        sim.run(args.duration)

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)

    finally:
        print("="*46)
        print("✅ Live detection simulation finished.")
        print(f"   Check the main log file: {DETECT_LOG_FILE}")
        
        print("📊 Generating latency analysis plot...")
        try:
            # This part assumes you have a separate analysis script
            from analyze_latency import analyze_log_file
            analyze_log_file('latency_log.csv')
        except Exception as e:
            logger.error(f"Failed to run latency analysis: {e}")
        print("📈 Plot generation complete.")