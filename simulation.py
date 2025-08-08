import time
import logging
import argparse
import os
import threading
import random
from typing import Dict, Optional, List, Any

# --- Import all necessary components ---
# This structure assumes your models and utils are in their respective directories
try:
    from models.blockchain import Blockchain
    from models.grid_nodes import GridOperator, Prosumer, Consumer, GridNode
    from utils.db_utils import DatabaseManager
except ImportError as e:
    # This provides a clear error if the file structure is wrong
    print(f"FATAL ERROR: A required module could not be imported. Please check your project structure.")
    print(f"Details: {e}")
    exit()

# --- Logging Setup ---
# A dedicated logger is used to avoid conflicts with other libraries or Streamlit's root logger
logger = logging.getLogger("simulation_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Clear any handlers from previous runs to prevent duplicate logs in the console
if logger.hasHandlers():
    logger.handlers.clear()

# Setup handlers for both file and console output
LOG_DIR = 'simulation_logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"simulation_run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
fh = logging.FileHandler(LOG_FILE, mode='w')
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)


class EnergyGridSimulation:
    """A high-speed, tick-based simulation engine for a blockchain-based smart grid."""
    def __init__(self, num_prosumers: int, num_consumers: int, difficulty: int, db_config: Optional[dict]):
        logger.info("--- Initializing High-Performance Energy Grid Simulation ---")
        
        # Initialize the Database Manager first
        self.db_manager = None
        if db_config:
            self.db_manager = DatabaseManager(**db_config)
            # If the DB setup failed, disable it for this run to prevent crashes
            if not getattr(self.db_manager, 'setup_complete', False):
                logger.error("Database setup failed. Disabling DB features for this run.")
                self.db_manager = None

        # Pass the created manager instance (or None) to the Blockchain
        self.blockchain = Blockchain(difficulty=difficulty, db_manager=self.db_manager)
        
        self.all_nodes: Dict[str, Any] = {}
        
        self.grid_operator = GridOperator("GRID-OP-01", self.blockchain)
        self.all_nodes[self.grid_operator.node_id] = self.grid_operator
        self.blockchain.register_node_wallet(self.grid_operator.node_id, 100000.0)
        if self.db_manager:
            self.db_manager.save_node("GRID-OP-01", "Operator", 100000.0)

        for i in range(num_prosumers):
            node_id = f"PRO-{i:02d}"
            balance = random.uniform(20.0, 50.0)
            node = Prosumer(node_id, self.blockchain, random.uniform(3.0, 8.0), random.uniform(0.5, 1.5))
            self.all_nodes[node_id] = node
            self.blockchain.register_node_wallet(node_id, balance)
            node.connect_to_grid_operator(self.grid_operator)
            if self.db_manager:
                self.db_manager.save_node(node_id, "Prosumer", balance, node.energy_storage)

        for i in range(num_consumers):
            node_id = f"CON-{i:02d}"
            balance = random.uniform(50.0, 200.0)
            node = Consumer(node_id, self.blockchain, random.uniform(1.0, 3.0))
            self.all_nodes[node_id] = node
            self.blockchain.register_node_wallet(node_id, balance)
            node.connect_to_grid_operator(self.grid_operator)
            if self.db_manager:
                self.db_manager.save_node(node_id, "Consumer", balance, node.energy_storage)
            
        self.worker_nodes: List[GridNode] = [n for n in self.all_nodes.values() if isinstance(n, (Prosumer, Consumer))]
        
        self.is_running = False
        self.mining_thread = None
        self.cumulative_energy_traded = 0.0
        self.cumulative_price_x_energy = 0.0
        logger.info("--- Simulation Setup Complete ---")

    def _mining_loop(self):
        """Dedicated background thread for mining blocks."""
        while self.is_running:
            if not self.blockchain.get_pending_transactions():
                time.sleep(0.05)
                continue
            new_block = self.blockchain.create_new_block(self.grid_operator.node_id)
            if new_block:
                self._process_block_for_nodes(new_block.to_dict())

    def _process_block_for_nodes(self, block: Dict[str, Any]):
        """Centrally notify all relevant nodes about transactions in a new block."""
        for tx in block.get('transactions', []):
            if tx.get('sender') in self.all_nodes:
                self.all_nodes[tx['sender']].process_transaction(tx)
            if tx.get('recipient') in self.all_nodes:
                self.all_nodes[tx['recipient']].process_transaction(tx)

    def _print_and_save_stats(self):
        """Calculates, prints, and saves key simulation metrics."""
        with self.blockchain.lock:
            block_count = len(self.blockchain.chain)
            tx_count = sum(len(b.transactions) for b in self.blockchain.chain)
        
        avg_price = self.cumulative_price_x_energy / self.cumulative_energy_traded if self.cumulative_energy_traded > 0 else 0
        
        logger.info(f"STATS | Blocks: {block_count} | Txs: {tx_count} | Energy Traded: {self.cumulative_energy_traded:.2f} kWh | Avg Price: ${avg_price:.4f}")
        
        if self.db_manager:
            stats_data = {
                'timestamp': time.time(),
                'total_energy_traded': self.cumulative_energy_traded,
                'avg_energy_price': avg_price,
                'block_count': block_count,
                'transaction_count': tx_count
            }
            self.db_manager.save_stats(stats_data)
            for node in self.worker_nodes:
                self.db_manager.save_node(node.node_id, type(node).__name__, self.blockchain.get_balance(node.node_id), node.energy_storage)

    def run(self, duration: int, ticks_per_second: int):
        """Main simulation loop."""
        logger.info(f"--- Starting Simulation Run ({duration}s, {ticks_per_second} TPS) ---")
        self.is_running = True
        
        self.mining_thread = threading.Thread(target=self._mining_loop, daemon=True)
        self.mining_thread.start()

        start_time = time.time()
        last_stats_time = start_time
        
        simulated_time_hours = 7.0  # Start simulation day at 7 AM
        time_step_hours = 10 / 60.0 # Each tick simulates 10 minutes of grid time

        try:
            while time.time() - start_time < duration:
                tick_start_time = time.time()
                hour_of_day = simulated_time_hours % 24
                
                # Update all nodes to generate energy data and market orders
                for node in self.worker_nodes:
                    node.update(hour_of_day, time_step_hours)
                
                # Grid operator clears the market, creating transactions and returning stats
                energy_tick, price_tick = self.grid_operator.clear_market()
                if energy_tick > 0:
                    self.cumulative_energy_traded += energy_tick
                    self.cumulative_price_x_energy += energy_tick * price_tick
                
                simulated_time_hours += time_step_hours
                
                if time.time() - last_stats_time >= 5:
                    self._print_and_save_stats()
                    last_stats_time = time.time()
                
                # Regulate speed to match the target TPS
                tick_duration = time.time() - tick_start_time
                sleep_time = (1.0 / ticks_per_second) - tick_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user.")
        finally:
            self.is_running = False
            logger.info("Shutting down...")
            if self.mining_thread:
                self.mining_thread.join(timeout=2.0)
            
            self._print_and_save_stats() # Save final stats
            self.blockchain.close()
            logger.info("--- Simulation Finished ---")

def main():
    """Parses arguments and runs the simulation."""
    parser = argparse.ArgumentParser(description="Run a high-speed blockchain energy grid simulation.")
    parser.add_argument('--prosumers', type=int, default=10, help="Number of prosumer nodes.")
    parser.add_argument('--consumers', type=int, default=20, help="Number of consumer nodes.")
    parser.add_argument('--difficulty', type=int, default=3, help="Blockchain mining difficulty.")
    parser.add_argument('--duration', type=int, default=60, help="Duration of the simulation in seconds.")
    parser.add_argument('--tps', type=int, default=20, help="Target Ticks Per Second for the simulation speed.")
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--db-port', type=str, default='3306')
    parser.add_argument('--db-user', type=str, default='root')
    parser.add_argument('--db-password', type=str, default=None)
    parser.add_argument('--db-name', type=str, default='blockchain_db')
    
    args = parser.parse_args()
    
    db_config = None
    if args.db_password is not None:
        db_config = {
            'database': args.db_name,
            'user': args.db_user,
            'password': args.db_password,
            'host': args.db_host,
            'port': args.db_port
        }
        logger.info("Database credentials provided via arguments. Database saving is ENABLED.")
    else:
        logger.warning("Database password not provided via --db-password argument. Database functionality is DISABLED.")
    
    sim = EnergyGridSimulation(args.prosumers, args.consumers, args.difficulty, db_config)
    sim.run(args.duration, args.tps)

if __name__ == "__main__":
    main()