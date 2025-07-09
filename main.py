#!/usr/bin/env python3
import argparse
import time
import signal
import sys
import logging
import random
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from models.blockchain import Blockchain
from models.grid_nodes import Prosumer, Consumer, GridOperator

# Try to import database utilities, but make it optional
try:
    from utils.db_utils import DatabaseManager, MYSQL_AVAILABLE
except ImportError:
    # If the module exists but mysql.connector is not available, MYSQL_AVAILABLE will be False
    # If the entire module fails to import, we need to define MYSQL_AVAILABLE here
    MYSQL_AVAILABLE = False

# Configure logging with better file handling
try:
    # Get absolute path to log file
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartgrid_simulation.log")
    
    # Create file handler with absolute path
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get our specific logger
    logger = logging.getLogger('smart_grid_simulation')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
except Exception as e:
    # Fallback to basic logging if file logging fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('smart_grid_simulation')
    logger.error(f"Failed to initialize file logging: {e}")
    logger.warning("Continuing with console logging only")

class SmartGridSimulation:
    def __init__(self, args):
        """Initialize the Smart Grid Blockchain Simulation with command-line arguments."""
        self.args = args
        self.running = False
        self.blockchain = None
        self.grid_operator = None
        self.prosumers = []
        self.consumers = []
        self.db_manager = None
        self.start_time = None
        self.total_energy_traded = 0
        
        # Initialize components
        self._init_blockchain()
        self._init_nodes()
        self._init_database()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_blockchain(self):
        """Initialize the blockchain with specified difficulty and GPU mode."""
        logger.info(f"Initializing blockchain (difficulty={self.args.difficulty}, GPU={self.args.use_gpu})")
        self.blockchain = Blockchain(difficulty=self.args.difficulty, use_gpu=self.args.use_gpu)

    def _init_nodes(self):
        """Initialize grid nodes (prosumers, consumers, grid operator)."""
        # Create the grid operator
        logger.info("Creating grid operator node")
        self.grid_operator = GridOperator(wallet_balance=1000.0)
        self.grid_operator.register_with_blockchain(self.blockchain)
        
        # Create prosumers
        logger.info(f"Creating {self.args.prosumers} prosumer nodes")
        for i in range(self.args.prosumers):
            # Random production and consumption rates
            prod_capacity = random.uniform(5.0, 15.0)  # kWh per hour
            cons_rate = random.uniform(3.0, 8.0)       # kWh per hour
            
            prosumer = Prosumer(
                wallet_balance=100.0,
                production_capacity=prod_capacity,
                consumption_rate=cons_rate
            )
            prosumer.register_with_blockchain(self.blockchain)
            self.prosumers.append(prosumer)
        
        # Create consumers
        logger.info(f"Creating {self.args.consumers} consumer nodes")
        for i in range(self.args.consumers):
            # Random consumption rate
            cons_rate = random.uniform(5.0, 12.0)  # kWh per hour
            
            consumer = Consumer(
                wallet_balance=100.0,
                consumption_rate=cons_rate
            )
            consumer.register_with_blockchain(self.blockchain)
            self.consumers.append(consumer)
    
    def _init_database(self):
        """Initialize database connection if MySQL details are provided."""
        if self.args.db_host:
            if not MYSQL_AVAILABLE:
                logger.warning("MySQL database support is not available. Install mysql-connector-python package.")
                logger.warning("The simulation will run without database storage.")
                return
                
            logger.info(f"Connecting to MySQL database at {self.args.db_host}")
            try:
                self.db_manager = DatabaseManager(
                    host=self.args.db_host,
                    user=self.args.db_user,
                    password=self.args.db_password,
                    database=self.args.db_name,
                    port=self.args.db_port,
                    create_if_not_exists=True
                )
                logger.info("Database connection established")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_manager = None
    
    def run(self):
        """Run the smart grid blockchain simulation."""
        logger.info("Starting smart grid blockchain simulation")
        self.running = True
        self.start_time = time.time()
        
        # Start the grid operator
        self.grid_operator.start()
        
        # Start all prosumers
        for prosumer in self.prosumers:
            prosumer.start()
            # Small delay to prevent all nodes starting at exactly the same time
            time.sleep(0.1)
        
        # Start all consumers
        for consumer in self.consumers:
            consumer.start()
            # Small delay
            time.sleep(0.1)
        
        # Monitor the simulation
        try:
            self._monitor_simulation()
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self.stop()
    
    def _monitor_simulation(self):
        """Monitor the simulation progress and collect statistics."""
        stats_interval = 10  # How often to log stats (seconds)
        last_stats_time = time.time()
        
        while self.running:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Check if simulation duration is reached
            if self.args.duration > 0 and elapsed_time >= self.args.duration:
                logger.info(f"Simulation duration ({self.args.duration}s) reached")
                self.running = False
                break
            
            # Log statistics periodically
            if current_time - last_stats_time >= stats_interval:
                self._log_statistics()
                last_stats_time = current_time
            
            # Save blockchain state to database periodically
            if self.db_manager and random.random() < 0.1:  # 10% chance each cycle
                self._save_to_database()
            
            # Small sleep to prevent high CPU usage
            time.sleep(1)
    
    def _log_statistics(self):
        """Log current simulation statistics."""
        # Calculate basic stats
        block_count = len(self.blockchain.chain)
        tx_count = sum(len(block.transactions) for block in self.blockchain.chain)
        
        # Calculate energy trading stats
        energy_traded = 0.0
        energy_prices = []
        trades_executed = 0
        
        # Look through all blocks for energy trades
        for block in self.blockchain.chain:
            for tx in block.transactions:
                # Count actual energy deliveries
                if tx['type'] == 'energy_delivery' and tx['energy'] > 0:
                    energy_traded += tx['energy']
                    trades_executed += 1
                
                # Collect prices from energy payments
                if tx['type'] == 'energy_payment' and tx['amount'] > 0:
                    energy_prices.append(tx['amount'])
        
        # Also check pending transactions that haven't been mined yet
        for tx in self.blockchain.current_transactions:
            if tx['type'] == 'energy_delivery' and tx['energy'] > 0:
                energy_traded += tx['energy']
                trades_executed += 1
            
            if tx['type'] == 'energy_payment' and tx['amount'] > 0:
                energy_prices.append(tx['amount'])
        
        avg_price = sum(energy_prices) / len(energy_prices) if energy_prices else 0
        
        # Log the stats
        logger.info(f"Simulation stats: {block_count} blocks, {tx_count} transactions")
        logger.info(f"Energy trades executed: {trades_executed}, Total energy: {energy_traded:.2f} kWh")
        logger.info(f"Energy trading average price: ${avg_price:.4f}/kWh")
        
        # Save to database if available
        if self.db_manager:
            stats = {
                'timestamp': time.time(),
                'total_energy_traded': energy_traded,
                'avg_energy_price': avg_price,
                'block_count': block_count,
                'transaction_count': tx_count
            }
            self.db_manager.save_stats(stats)
    
    def _save_to_database(self):
        """Save blockchain state to database."""
        if not self.db_manager or not MYSQL_AVAILABLE:
            return
            
        # Save recent blocks that might not be in the database yet
        for block in self.blockchain.chain[-5:]:  # Save last 5 blocks
            self.db_manager.save_block(block.to_dict())
        
        # Save node states
        self.db_manager.save_node(
            self.grid_operator.node_id, 
            'grid_operator',
            self.grid_operator.wallet_balance,
            self.grid_operator.grid_balance
        )
        
        for prosumer in self.prosumers:
            self.db_manager.save_node(
                prosumer.node_id,
                'prosumer',
                prosumer.wallet_balance,
                prosumer.energy_surplus
            )
            
        for consumer in self.consumers:
            self.db_manager.save_node(
                consumer.node_id,
                'consumer',
                consumer.wallet_balance,
                -consumer.consumption_rate  # Negative to indicate consumption
            )
    
    def stop(self):
        """Stop the simulation and clean up resources."""
        logger.info("Stopping simulation...")
        self.running = False
        
        # Stop all nodes
        self.grid_operator.stop()
        
        for prosumer in self.prosumers:
            prosumer.stop()
            
        for consumer in self.consumers:
            consumer.stop()
        
        # Final statistics
        self._log_statistics()
        
        # Close database connection
        if self.db_manager:
            self.db_manager.close()
        
        # Calculate total simulation time
        total_time = time.time() - self.start_time
        logger.info(f"Simulation completed in {total_time:.2f} seconds")
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)


def parse_arguments():
    """Parse command-line arguments for the simulation."""
    parser = argparse.ArgumentParser(
        description='Smart Grid Blockchain Simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Simulation parameters
    parser.add_argument('--prosumers', type=int, default=3,
                        help='Number of prosumer nodes (produce and consume energy)')
    parser.add_argument('--consumers', type=int, default=5,
                        help='Number of consumer nodes (only consume energy)')
    parser.add_argument('--operators', type=int, default=1,
                        help='Number of grid operator nodes (fixed at 1 for now)')
    parser.add_argument('--difficulty', type=int, default=4,
                        help='Blockchain mining difficulty (number of leading zeros)')
    parser.add_argument('--duration', type=int, default=300,
                        help='Simulation duration in seconds (0 for infinite)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for mining if available')
    
    # Database parameters
    parser.add_argument('--db-host', type=str, default='',
                        help='MySQL database host')
    parser.add_argument('--db-port', type=int, default=3306,
                        help='MySQL database port')
    parser.add_argument('--db-user', type=str, default='root',
                        help='MySQL database user')
    parser.add_argument('--db-password', type=str, default='',
                        help='MySQL database password')
    parser.add_argument('--db-name', type=str, default='smartgrid',
                        help='MySQL database name')
    
    return parser.parse_args()


def main():
    """Main entry point for the simulation."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create and run the simulation
    simulation = SmartGridSimulation(args)
    simulation.run()


if __name__ == "__main__":
    main()