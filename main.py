# Use print statements for initial diagnosis, as logging might not be configured yet.
print("--- Script starting ---")

import argparse
import time
import signal
import sys
import logging
import random
import os

# --- Centralized Logging Configuration ---
try:
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartgrid_simulation.log")
    # Using 'w' to clear the log on each run is better for debugging.
    # Change to 'a' if you want to keep a running history.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    print(f"--- Logging configured. Output will go to terminal and {log_file} ---")
except Exception as e:
    print(f"!!! CRITICAL: FAILED TO CONFIGURE LOGGING: {e} !!!")
    sys.exit(1)

# --- Now import our custom modules ---
try:
    from models.blockchain import Blockchain
    from models.grid_nodes import Prosumer, Consumer, GridOperator
    # Import the new AnomalyInjector
    from anomaly_injector import AnomalyInjector
    print("--- Custom modules imported successfully ---")
except ImportError as e:
    # Handle the case where anomaly_injector.py might not exist yet
    AnomalyInjector = None
    print(f"--- Warning: Could not import AnomalyInjector ({e}). Running without anomaly injection. ---")
except Exception as e:
    logging.critical(f"Failed to import custom modules: {e}", exc_info=True)
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
        # Add a reference for the anomaly injector
        self.anomaly_injector = None
        
        # FIX: Correctly structure the database configuration from args
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': self.args.db_password,
            'database': 'smartgrid',
            'port': 3306,
        } if self.args.db_password is not None else None # Only configure if password is provided
        
        self._init_blockchain()
        self._init_nodes()
        self._connect_nodes()
        # Initialize the anomaly injector if it was imported
        if AnomalyInjector:
            self._init_anomaly_injector()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        print("--- Simulation class initialized ---")

    def _init_blockchain(self):
        logging.info(f"Initializing blockchain (difficulty={self.args.difficulty})")
        # Pass the db_config to the Blockchain. Assumes Blockchain can handle it.
        self.blockchain = Blockchain(difficulty=self.args.difficulty, db_config=self.db_config)

    def _init_nodes(self):
        logging.info("Creating grid operator node")
        self.grid_operator = GridOperator(node_id="GRID-OP-01", wallet_balance=1_000_000.0)
        self.grid_operator.register_with_blockchain(self.blockchain)
        
        logging.info(f"Creating {self.args.prosumers} prosumer nodes")
        for i in range(self.args.prosumers):
            prosumer = Prosumer(
                node_id=f"PRO-{i:02d}",
                wallet_balance=random.uniform(80.0, 120.0),
                production_capacity=random.uniform(5.0, 15.0),
                consumption_rate=random.uniform(3.0, 8.0),
                sim_start_time=self.start_time
            )
            prosumer.register_with_blockchain(self.blockchain)
            self.prosumers.append(prosumer)
        
        logging.info(f"Creating {self.args.consumers} consumer nodes")
        for i in range(self.args.consumers):
            consumer = Consumer(
                node_id=f"CON-{i:02d}",
                wallet_balance=random.uniform(80.0, 120.0),
                consumption_rate=random.uniform(5.0, 12.0),
                sim_start_time=self.start_time
            )
            consumer.register_with_blockchain(self.blockchain)
            self.consumers.append(consumer)

    def _connect_nodes(self):
        logging.info("Connecting all nodes to the main grid operator.")
        all_nodes = self.prosumers + self.consumers
        for node in all_nodes:
            node.connect_to_node(self.grid_operator)
    
    def _init_anomaly_injector(self):
        """Initializes and configures the anomaly injector."""
        logging.info("Initializing anomaly injector.")
        all_grid_nodes = self.prosumers + self.consumers
        self.anomaly_injector = AnomalyInjector(all_grid_nodes, self.grid_operator)

    def run(self):
        logging.info("--- Starting Simulation Run ---")
        self.running = True
        
        all_nodes = [self.grid_operator] + self.prosumers + self.consumers
        for node in all_nodes:
            node.start()
        
        # Start the anomaly injector if it exists
        if self.anomaly_injector:
            self.anomaly_injector.start()
        
        try:
            self._monitor_simulation()
        except KeyboardInterrupt:
            logging.info("Simulation interrupted by user")
        finally:
            self.stop()
    
    def _monitor_simulation(self):
        stats_interval = 15
        last_stats_time = time.time()
        
        while self.running:
            if self.args.duration > 0 and (time.time() - self.start_time) >= self.args.duration:
                logging.info(f"Simulation duration ({self.args.duration}s) reached")
                self.running = False
                break
            
            if time.time() - last_stats_time >= stats_interval:
                self._log_statistics()
                last_stats_time = time.time()
            
            time.sleep(1)
    
    def _log_statistics(self):
        if not self.blockchain: return
        block_count = len(self.blockchain.chain)
        tx_count = sum(len(b.transactions) for b in self.blockchain.chain)
        energy_traded = sum(tx['energy'] for b in self.blockchain.chain for tx in b.transactions if tx['type'] == 'energy_delivery')
        total_storage = sum(node.energy_storage for node in self.prosumers + self.consumers if hasattr(node, 'energy_storage'))
        logging.info(f"STATS | Blocks: {block_count}, Txs: {tx_count}, Energy Traded: {energy_traded:.2f} kWh, Total Storage: {total_storage:.2f} kWh")
    
    def stop(self):
        if not self.running: return
        logging.info("--- Stopping Simulation ---")
        self.running = False
        
        # Stop the anomaly injector first
        if self.anomaly_injector:
            self.anomaly_injector.stop()
            
        all_nodes = [self.grid_operator] + self.prosumers + self.consumers
        for node in all_nodes:
            if node: node.stop()
        
        self._log_statistics()
        
        # Close database connection (assuming blockchain object handles it)
        if self.blockchain and hasattr(self.blockchain, 'close_db'):
            self.blockchain.close_db()
            
        logging.info(f"Simulation completed in {time.time() - self.start_time:.2f} seconds")
    
    def _signal_handler(self, sig, frame):
        logging.warning(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)

def main():
    print("--- Main function starting ---")
    parser = argparse.ArgumentParser(description='Smart Grid Blockchain Simulation')
    parser.add_argument('--prosumers', type=int, default=5, help='Number of prosumer nodes')
    parser.add_argument('--consumers', type=int, default=10, help='Number of consumer nodes')
    parser.add_argument('--difficulty', type=int, default=4, help='Blockchain mining difficulty')
    parser.add_argument('--duration', type=int, default=120, help='Simulation duration in seconds')
    # FIX: Changed default to None to make providing a password an explicit action
    parser.add_argument('--db-password', type=str, default=None, help='MySQL database password. If not provided, DB is disabled.')
    args = parser.parse_args()
    
    simulation = SmartGridSimulation(args)
    simulation.run()
    print("--- Main function finished ---")

if __name__ == "__main__":
    print("--- Script is being run directly ---")
    main()