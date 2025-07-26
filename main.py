
import argparse
import time
import signal
import sys
import logging
import random
import os
try:
    from models.blockchain import Blockchain
    from models.grid_nodes import Prosumer, Consumer, GridOperator, BaseNode
    from anomaly_injector import AnomalyInjector
    from fraud_detector_unsupervised_model import FraudDetector
    from utils.latency_recorder import initialize_latency_log
    logging.info("--- Custom modules imported successfully ---")
except ImportError as e:
    logging.critical(f"A required module is missing. Please check your file structure. Error: {e}", exc_info=True)
    sys.exit(1)

def setup_logging(mode: str):
    """Dynamically sets the log file name based on the simulation mode."""
    log_filename = "data_generation_run.log" if mode == 'generate' else "live_detection_run.log"
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"--- Logging configured for '{mode}' mode. Output to '{log_filename}' ---")


class SmartGridSimulation:
    def __init__(self, args):
        self.args = args
        self.running = False
        self.start_time = time.time()
        self.all_nodes: list[BaseNode] = []
        
        self._init_blockchain()
        self._init_nodes()
        self._init_components() # Handles injector and detector based on mode
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logging.info("--- Simulation class initialized ---")

    def _init_blockchain(self):
        logging.info(f"Initializing blockchain (difficulty={self.args.difficulty})")
        self.blockchain = Blockchain(difficulty=self.args.difficulty)

    def _init_nodes(self):
        """Initializes worker nodes FIRST, then the GridOperator."""
        sim_start_time = time.time()
        
        # 1. Create worker nodes
        self.prosumers = [
            Prosumer(
                node_id=f"PRO-{i:02d}", blockchain=self.blockchain,
                production_capacity=random.uniform(5.0, 15.0),
                consumption_rate=random.uniform(3.0, 8.0), sim_start_time=sim_start_time
            ) for i in range(self.args.prosumers)
        ]
        self.consumers = [
            Consumer(
                node_id=f"CON-{i:02d}", blockchain=self.blockchain,
                consumption_rate=random.uniform(5.0, 12.0), sim_start_time=sim_start_time
            ) for i in range(self.args.consumers)
        ]
        worker_nodes = self.prosumers + self.consumers

        # 2. Create GridOperator, passing it the worker nodes
        logging.info("Creating grid operator node...")
        self.grid_operator = GridOperator(
            node_id="GRID-OP-01", blockchain=self.blockchain,
            worker_nodes=worker_nodes,
            sim_start_time=sim_start_time
        )
        
        # 3. Connect workers to the operator
        for node in worker_nodes:
            node.connect_to_grid_operator(self.grid_operator)

        # 4. Create final list of all nodes
        self.all_nodes = [self.grid_operator] + worker_nodes
        logging.info(f"Total nodes initialized: {len(self.all_nodes)}")

    def _init_components(self):
        """Initializes components like injector and detector based on the run mode."""
        self.anomaly_injector = None
        self.fraud_detector = None

        if self.args.mode in ['generate', 'detect']:
            logging.info("Initializing Anomaly Injector.")
            self.anomaly_injector = AnomalyInjector(self.all_nodes, self.grid_operator, self.blockchain)

        if self.args.mode == 'detect':
            logging.info("Initializing Live Fraud Detector.")
            self.fraud_detector = FraudDetector(self.blockchain)
            initialize_latency_log()

    def run(self):
        logging.info(f"--- Starting Simulation Run (Mode: {self.args.mode}) ---")
        self.running = True
        
        for node in self.all_nodes:
            if hasattr(node, 'start'): node.start()
        
        if self.anomaly_injector: self.anomaly_injector.start()
        if self.fraud_detector: self.fraud_detector.start()
        
        try:
            self._monitor_simulation()
        except KeyboardInterrupt:
            logging.info("Simulation interrupted by user.")
        finally:
            self.stop()
    
    def _monitor_simulation(self):
        stats_interval = 10  # Log stats more frequently for better visibility
        last_stats_time = time.time()
        
        while self.running:
            if self.args.duration > 0 and (time.time() - self.start_time) >= self.args.duration:
                logging.info(f"Simulation duration ({self.args.duration}s) reached.")
                self.running = False
                break
            
            if time.time() - last_stats_time >= stats_interval:
                self._log_statistics()
                last_stats_time = time.time()
            
            time.sleep(1)
    
    def _log_statistics(self):
        if not self.blockchain: return
        with self.blockchain.lock:
            block_count = len(self.blockchain.chain)
            tx_count = sum(len(b.transactions) for b in self.blockchain.chain)
            energy_traded = sum(tx['energy'] for b in self.blockchain.chain for tx in b.transactions if tx['type'] == 'energy_delivery')
            active_grid_nodes = [n for n in (self.prosumers + self.consumers) if hasattr(n, 'energy_storage') and n.active]
            total_storage = sum(node.energy_storage for node in active_grid_nodes)
        logging.info(f"STATS | Blocks: {block_count}, Txs: {tx_count}, Energy Traded: {energy_traded:.2f} kWh, Total Storage: {total_storage:.2f} kWh")
    
    def stop(self):
        if not self.running: return
        logging.info("--- Stopping Simulation ---")
        self.running = False
        
        if self.anomaly_injector: self.anomaly_injector.stop()
        if self.fraud_detector: self.fraud_detector.stop()
            
        for node in self.all_nodes:
            if hasattr(node, 'stop'): node.stop()
        
        self._log_statistics()
        logging.info(f"Simulation completed in {time.time() - self.start_time:.2f} seconds.")
    
    def _signal_handler(self, sig, frame):
        logging.warning(f"Received signal {sig}, shutting down gracefully...")
        self.running = False

def main():
    parser = argparse.ArgumentParser(description='Smart Grid Blockchain Simulation')
    
    parser.add_argument(
        'mode',
        type=str,
        choices=['generate', 'detect'],
        help="Run mode: 'generate' for data creation, 'detect' for live anomaly detection."
    )
    parser.add_argument('--prosumers', type=int, default=10, help='Number of prosumer nodes')
    parser.add_argument('--consumers', type=int, default=20, help='Number of consumer nodes')
    parser.add_argument('--difficulty', type=int, default=3, help='Blockchain mining difficulty')
    parser.add_argument('--duration', type=int, default=300, help='Simulation duration in seconds (0 for infinite)')
    args = parser.parse_args()
    
    setup_logging(args.mode)
    
    simulation = SmartGridSimulation(args)
    simulation.run()


def run(self):
    logging.info(f"--- Starting Simulation Run (Mode: {self.args.mode}) ---")
    self.running = True
    
    for node in self.all_nodes:
        if hasattr(node, 'start'): node.start()
    
    # --- TEMPORARILY DISABLE THE INJECTOR FOR THIS RUN ---
    # if self.anomaly_injector: self.anomaly_injector.start() 
    
    if self.fraud_detector: self.fraud_detector.start()
    
    try:
        self._monitor_simulation()
    finally:
        self.stop()

if __name__ == "__main__":
    main()