# simulation.py (Final Corrected Version)

import time
import logging
import argparse
import random
import os
from typing import List, Optional

from models.blockchain import Blockchain
from models.grid_nodes import GridOperator, Prosumer, Consumer, BaseNode

# --- Logging Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_NAME = f"simulation_run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_DIR = os.path.join(SCRIPT_DIR, 'simulation_logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, LOG_FILE_NAME)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Simulation Class ---
class EnergyGridSimulation:
    def __init__(self, num_prosumers: int, num_consumers: int, difficulty: int, db_config: Optional[dict]):
        logger.info("--- Initializing Realistic Energy Grid Simulation ---")
        self.sim_start_time = time.time()
        self.blockchain = Blockchain(difficulty=difficulty, db_config=db_config)
        
        self.all_nodes: List[BaseNode] = []
        worker_nodes: List[BaseNode] = []

        # Create Prosumers (always residential with solar panels)
        logger.info(f"Creating {num_prosumers} prosumer nodes...")
        for i in range(num_prosumers):
            node = Prosumer(
                node_id=f"PRO-{i:02d}",
                blockchain=self.blockchain,
                production_capacity=random.uniform(3.0, 8.0),
                consumption_rate=random.uniform(0.5, 1.5),
                sim_start_time=self.sim_start_time
            )
            self.blockchain.register_node_wallet(node.node_id, random.uniform(20.0, 50.0))
            worker_nodes.append(node)

        # Create Consumers (a mix of residential and commercial for realistic patterns)
        logger.info(f"Creating {num_consumers} consumer nodes...")
        for i in range(num_consumers):
            is_commercial_node = random.random() < 0.2  # 20% chance to be a commercial node
            if is_commercial_node:
                # Commercial consumer: high, daytime usage on weekdays
                node = Consumer(
                    node_id=f"CON-{i:02d}",
                    blockchain=self.blockchain,
                    consumption_rate=random.uniform(5.0, 15.0),
                    sim_start_time=self.sim_start_time,
                    is_commercial=True
                )
            else:
                # Residential consumer: morning/evening peaks
                node = Consumer(
                    node_id=f"CON-{i:02d}",
                    blockchain=self.blockchain,
                    consumption_rate=random.uniform(1.0, 3.0),
                    sim_start_time=self.sim_start_time,
                    is_commercial=False
                )
            self.blockchain.register_node_wallet(node.node_id, random.uniform(50.0, 200.0))
            worker_nodes.append(node)
        
        # Create GridOperator, passing it the list of all other nodes and the start time
        self.grid_operator = GridOperator("GRID-OP-01", self.blockchain, worker_nodes, self.sim_start_time)
        self.blockchain.register_node_wallet(self.grid_operator.node_id, 10000.0)

        # Connect all worker nodes to the now-existing GridOperator
        for node in worker_nodes:
            if hasattr(node, 'connect_to_grid_operator'):
                node.connect_to_grid_operator(self.grid_operator)
        
        # Create the final list of all nodes
        self.all_nodes = [self.grid_operator] + worker_nodes
        
        logger.info("--- Simulation Setup Complete ---")

    def start_nodes(self):
        logger.info("Starting all network nodes...")
        for node in self.all_nodes: node.start()

    def stop_nodes(self):
        logger.info("Stopping all network nodes...")
        for node in self.all_nodes: node.stop()
    
    def _print_stats(self):
        with self.blockchain.lock:
            chain = self.blockchain.chain
            pending_txs = len(self.blockchain.current_transactions)
            total_txs = sum(len(b.transactions) for b in chain)
            energy = sum(tx['energy'] for b in chain for tx in b.transactions if tx['type'] == 'energy_delivery')
        logger.info(f"STATS | Blocks: {len(chain)} | Txs Confirmed: {total_txs} | Pending: {pending_txs} | Energy Traded: {energy:.2f} kWh")

    def run(self, duration: int):
        logger.info(f"--- Starting Simulation Run ({duration} seconds) ---")
        self.start_nodes()
        start_time = time.time()
        last_stats_time = start_time
        try:
            while time.time() - start_time < duration:
                if time.time() - last_stats_time >= 10:
                    self._print_stats()
                    last_stats_time = time.time()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user.")
        finally:
            logger.info("Shutting down...")
            self.stop_nodes()
            self._print_stats()
            self.blockchain.close()
            logger.info("--- Simulation Finished ---")

# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Run a realistic blockchain energy grid simulation.")
    parser.add_argument('--prosumers', type=int, default=10, help="Number of prosumer nodes.")
    parser.add_argument('--consumers', type=int, default=20, help="Number of consumer nodes.")
    parser.add_argument('--difficulty', type=int, default=3, help="Blockchain mining difficulty.")
    parser.add_argument('--duration', type=int, default=600, help="Duration of the simulation in seconds.")
    args = parser.parse_args()

    db_config = {
        'database': os.getenv('DB_NAME', 'blockchain_db'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '3306')
    }
    
    if db_config.get('password') is None:
        logger.warning("DB_PASSWORD environment variable not set. Database functionality disabled.")
        db_config = None
    
    sim = EnergyGridSimulation(args.prosumers, args.consumers, args.difficulty, db_config)
    sim.run(args.duration)

if __name__ == "__main__":
    main()