# main.py (Final Corrected Version)

import argparse
import time
import signal
import sys
import logging
import random
import os
# The scaler and joblib are no longer needed in main.py, they are handled by other scripts.
# import joblib 
# from sklearn.preprocessing import MinMaxScaler

# --- Centralized Logging Configuration ---
# (This section is correct and does not need changes)

# --- Import Custom Modules ---
try:
    from models.blockchain import Blockchain
    from models.grid_nodes import Prosumer, Consumer, GridOperator
    from anomaly_injector import AnomalyInjector
    from fraud_detector import FraudDetector
except ImportError as e:
    if 'anomaly_injector' in str(e): AnomalyInjector = None
    if 'fraud_detector' in str(e): FraudDetector = None
    print(f"--- Warning: Could not import a module ({e}). Some modes may be unavailable. ---")
except Exception as e:
    logging.basicConfig()
    logging.critical(f"An unexpected error occurred during imports: {e}", exc_info=True)
    sys.exit(1)


class SmartGridSimulation:
    def __init__(self, args):
        self.args = args
        self.running = False
        self.start_time = time.time()
        self.blockchain = None
        self.grid_operator = None
        self.prosumers = []
        self.consumers = []
        
        self.anomaly_injector = None
        self.fraud_detector = None
        
        self._setup_logging()
        self._init_blockchain()
        self._init_nodes()
        self._connect_nodes()
        
        if args.mode == 'generate' and AnomalyInjector:
            self._init_anomaly_injector()
        elif args.mode == 'detect' and FraudDetector:
            if AnomalyInjector: self._init_anomaly_injector()
            self._init_fraud_detector()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        logging.info("--- Simulation class initialized ---")

    def _setup_logging(self):
        log_filename = "live_detection_run.log" if self.args.mode == 'detect' else "data_generation_run.log"
        log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if root_logger.hasHandlers(): root_logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(console_handler)
        
        logging.info(f"Logging configured. Output will go to terminal and {log_filename}")

    def _init_blockchain(self):
        logging.info(f"Initializing blockchain (difficulty={self.args.difficulty})")
        self.blockchain = Blockchain(difficulty=self.args.difficulty)

    def _init_nodes(self):
        logging.info("Creating grid operator node")
        self.grid_operator = GridOperator(node_id="GRID-OP-01", wallet_balance=1_000_000.0)
        self.grid_operator.register_with_blockchain(self.blockchain)
        
        logging.info(f"Creating {self.args.prosumers} prosumer nodes")
        for i in range(self.args.prosumers):
            prosumer = Prosumer(f"PRO-{i:02d}", 100.0, random.uniform(5,15), random.uniform(3,8), self.start_time)
            prosumer.register_with_blockchain(self.blockchain)
            self.prosumers.append(prosumer)
        
        logging.info(f"Creating {self.args.consumers} consumer nodes")
        for i in range(self.args.consumers):
            consumer = Consumer(f"CON-{i:02d}", 100.0, random.uniform(5,12), self.start_time)
            consumer.register_with_blockchain(self.blockchain)
            self.consumers.append(consumer)

    def _connect_nodes(self):
        logging.info("Connecting all nodes to the main grid operator.")
        for node in self.prosumers + self.consumers:
            node.connect_to_node(self.grid_operator)
    
    def _init_anomaly_injector(self):
        logging.info("Initializing Anomaly Injector.")
        self.anomaly_injector = AnomalyInjector(self.prosumers + self.consumers, self.grid_operator)

    def _init_fraud_detector(self):
        """Initializes the fraud detector for live detection runs."""
        logging.info("Initializing Live Fraud Detector.")
        # FIX: The FraudDetector now loads its own assets. We just need to create it
        # and pass it the live blockchain instance.
        self.fraud_detector = FraudDetector(blockchain=self.blockchain)
        
        # This check is now handled inside the FraudDetector's __init__
        if not self.fraud_detector.is_ready:
            self.fraud_detector = None 

    def run(self):
        logging.info(f"--- Starting Simulation Run (Mode: {self.args.mode}) ---")
        self.running = True
        
        for node in [self.grid_operator] + self.prosumers + self.consumers:
            node.start()
        
        if self.anomaly_injector: self.anomaly_injector.start()
        if self.fraud_detector: self.fraud_detector.start()
        
        try:
            self._monitor_simulation()
        except KeyboardInterrupt:
            logging.info("Simulation interrupted by user")
        finally:
            self.stop()
    
    def _monitor_simulation(self):
        stats_interval = 15
        while self.running:
            if self.args.duration > 0 and (time.time() - self.start_time) >= self.args.duration:
                logging.info(f"Simulation duration ({self.args.duration}s) reached")
                self.running = False
                break
            if int(time.time()) % stats_interval == 0:
                self._log_statistics()
                time.sleep(1) 
            else:
                time.sleep(0.5)
    
    def _log_statistics(self):
        block_count = len(self.blockchain.chain)
        tx_count = sum(len(b.transactions) for b in self.blockchain.chain)
        energy_traded = sum(tx['energy'] for b in self.blockchain.chain for tx in b.transactions if tx['type'] == 'energy_delivery')
        total_storage = sum(node.energy_storage for node in self.prosumers + self.consumers if hasattr(node, 'energy_storage'))
        logging.info(f"STATS | Blocks: {block_count}, Txs: {tx_count}, Energy Traded: {energy_traded:.2f} kWh, Total Storage: {total_storage:.2f} kWh")

    def stop(self):
        if not self.running: return
        logging.info("--- Stopping Simulation ---")
        self.running = False
        
        if self.anomaly_injector: self.anomaly_injector.stop()
        if self.fraud_detector: self.fraud_detector.stop()
        for node in [self.grid_operator] + self.prosumers + self.consumers:
            if node: node.stop()
        
        self._log_statistics()
        if hasattr(self.blockchain, 'close_db'): self.blockchain.close_db()
        logging.info(f"Simulation completed in {time.time() - self.start_time:.2f} seconds")
    
    def _signal_handler(self, sig, frame):
        logging.warning(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Smart Grid Blockchain Simulation')
    parser.add_argument('mode', choices=['generate', 'train', 'detect'], 
                        help="Run mode: 'generate' for data, 'train' for the model, 'detect' for live detection.")
    parser.add_argument('--prosumers', type=int, default=10)
    parser.add_argument('--consumers', type=int, default=20)
    parser.add_argument('--difficulty', type=int, default=3)
    parser.add_argument('--duration', type=int, default=120)
    parser.add_argument('--db-password', type=str, default=None)
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("--- Running the training pipeline ---")
        if not os.path.exists('data/featurized_labeled_data.csv'):
            print("Labeled data not found. Run in 'generate' mode first: python main.py generate")
            sys.exit(1)
        os.system('python train_model.py')
    else:
        simulation = SmartGridSimulation(args)
        simulation.run()

if __name__ == "__main__":
    main()