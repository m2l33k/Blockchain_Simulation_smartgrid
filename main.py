# main.py (Corrected with Proper Node List Management)

from __future__ import annotations
import argparse
import time
import signal
import sys
import logging
import random
import os
import json

# --- Configure Logging ---
try:
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("--- Logging configured successfully ---")
except Exception as e:
    print(f"!!! CRITICAL: FAILED TO CONFIGURE LOGGING: {e} !!!")
    sys.exit(1)

# --- Import Custom Modules ---
try:
    from models.blockchain import Blockchain
    from models.grid_nodes import Prosumer, Consumer, GridOperator, BaseNode
    # Make sure to use the corrected anomaly_injector from the previous step
    from anomaly_injector import AnomalyInjector
    logging.info("--- Custom modules imported successfully ---")
except ImportError as e:
    logging.critical(f"A required module is missing. Please check your file structure. Error: {e}", exc_info=True)
    sys.exit(1)


class SmartGridSimulation:
    def __init__(self, args):
        self.args = args
        self.running = False
        self.start_time = time.time()
        
        # --- FIX: Initialize lists here ---
        self.all_nodes: list[BaseNode] = []
        
        self._init_blockchain()
        self._init_nodes()
        self._init_anomaly_injector()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logging.info("--- Simulation class initialized ---")

    def _init_blockchain(self):
        logging.info(f"Initializing blockchain (difficulty={self.args.difficulty})")
        self.blockchain = Blockchain(difficulty=self.args.difficulty)

    def _init_nodes(self):
        """
        --- FIX: Initializes all nodes and populates a single `self.all_nodes` list ---
        """
        sim_start_time = time.time()
        
        logging.info("Creating grid operator node...")
        self.grid_operator = GridOperator(node_id="GRID-OP-01", blockchain=self.blockchain)
        self.all_nodes.append(self.grid_operator)
        
        # Keep separate lists for easy access if needed, but also add to the main list
        self.prosumers = []
        logging.info(f"Creating {self.args.prosumers} prosumer nodes...")
        for i in range(self.args.prosumers):
            prosumer = Prosumer(
                node_id=f"PRO-{i:02d}",
                blockchain=self.blockchain,
                production_capacity=random.uniform(5.0, 15.0),
                consumption_rate=random.uniform(3.0, 8.0),
                sim_start_time=sim_start_time
            )
            prosumer.connect_to_grid_operator(self.grid_operator)
            self.prosumers.append(prosumer)
            self.all_nodes.append(prosumer)
        
        self.consumers = []
        logging.info(f"Creating {self.args.consumers} consumer nodes...")
        for i in range(self.args.consumers):
            consumer = Consumer(
                node_id=f"CON-{i:02d}",
                blockchain=self.blockchain,
                consumption_rate=random.uniform(5.0, 12.0),
                sim_start_time=sim_start_time
            )
            consumer.connect_to_grid_operator(self.grid_operator)
            self.consumers.append(consumer)
            self.all_nodes.append(consumer)

        logging.info(f"Total nodes initialized: {len(self.all_nodes)}")

    def _init_anomaly_injector(self):
        logging.info("Initializing anomaly injector.")
        # --- FIX: Now correctly uses the `self.all_nodes` attribute ---
        self.anomaly_injector = AnomalyInjector(self.all_nodes, self.grid_operator, self.blockchain)

    def run(self):
        logging.info("--- Starting Simulation Run ---")
        self.running = True
        
        for node in self.all_nodes:
            if hasattr(node, 'start'):
                node.start()
        
        if self.anomaly_injector:
            self.anomaly_injector.start()
        
        try:
            self._monitor_simulation()
        except KeyboardInterrupt:
            logging.info("Simulation interrupted by user.")
        finally:
            self.stop()
    
    def _monitor_simulation(self):
        stats_interval = 15
        last_stats_time = time.time()
        
        while self.running:
            if self.args.duration > 0 and (time.time() - self.start_time) >= self.args.duration:
                logging.info(f"Simulation duration ({self.args.duration}s) reached. Stopping.")
                self.running = False
                break

            if self.blockchain.get_pending_transactions():
                logging.info(f"Main loop found {len(self.blockchain.get_pending_transactions())} pending transactions. Initiating mining...")
                self.blockchain.create_new_block(miner_id="GRID-OP-01")
            
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
            # FIX: Use the prosumer/consumer lists for specific stats
            total_storage = sum(node.energy_storage for node in (self.prosumers + self.consumers) if hasattr(node, 'energy_storage'))
        logging.info(f"STATS | Blocks: {block_count}, Txs: {tx_count}, Energy Traded: {energy_traded:.2f} kWh, Total Storage: {total_storage:.2f} kWh")
    
    def stop(self):
        if not self.running: return
        logging.info("--- Stopping Simulation ---")
        self.running = False
        
        if self.anomaly_injector:
            self.anomaly_injector.stop()
            
        for node in self.all_nodes:
            if hasattr(node, 'stop'):
                node.stop()
        
        self._log_statistics()
        
        logging.info(f"Simulation completed in {time.time() - self.start_time:.2f} seconds.")
    
    def _signal_handler(self, sig, frame):
        logging.warning(f"Received signal {sig}, shutting down gracefully...")
        self.running = False

def main():
    parser = argparse.ArgumentParser(description='Smart Grid Blockchain Simulation Log Generator')
    parser.add_argument('--prosumers', type=int, default=10, help='Number of prosumer nodes')
    parser.add_argument('--consumers', type=int, default=20, help='Number of consumer nodes')
    parser.add_argument('--difficulty', type=int, default=3, help='Blockchain mining difficulty')
    parser.add_argument('--duration', type=int, default=120, help='Simulation duration in seconds (0 for infinite)')
    args = parser.parse_args()
    
    simulation = SmartGridSimulation(args)
    simulation.run()
    logging.info("--- Main log generation finished ---")

if __name__ == "__main__":
    main()