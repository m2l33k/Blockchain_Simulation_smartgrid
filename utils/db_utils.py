import logging
from typing import Dict, Any, Union
from datetime import datetime

logger = logging.getLogger("simulation_logger")

try:
    import mysql.connector
    import json
    MYSQL_AVAILABLE = True
except ImportError:
    logger.warning("mysql-connector-python not installed. DB features disabled. Run: pip install mysql-connector-python")
    MYSQL_AVAILABLE = False

class DatabaseManager:
    def __init__(self, host: str, user: str, password: str, database: str, port: str, create_if_not_exists: bool = True):
        self.conn = None
        self.setup_complete = False
        if not MYSQL_AVAILABLE: return
            
        self.config = {'host': host, 'user': user, 'password': password, 'port': port}
        self.database = database
        self.cursor = None
        
        if self._connect() and create_if_not_exists:
            self._setup_database()
    
    def _connect(self) -> bool:
        try:
            self.conn = mysql.connector.connect(**self.config, database=self.database, connect_timeout=10)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.database}")
            return True
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                try:
                    self.conn = mysql.connector.connect(**self.config)
                    self.cursor = self.conn.cursor()
                    logger.warning(f"Database '{self.database}' not found. Attempting to create it.")
                    return True
                except mysql.connector.Error as e:
                    logger.error(f"FATAL: Failed to connect to MySQL server: {e}")
                    return False
            else:
                logger.error(f"FATAL: Failed to connect to database: {err}")
                return False
    
    def _setup_database(self):
        try:
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            self.cursor.execute(f"USE {self.database}")
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    block_index INT NOT NULL UNIQUE,
                    hash VARCHAR(64) NOT NULL UNIQUE,
                    previous_hash VARCHAR(64) NOT NULL,
                    timestamp DOUBLE NOT NULL,
                    nonce INT NOT NULL,
                    miner_id VARCHAR(255),
                    transactions JSON
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    tx_id VARCHAR(64) NOT NULL UNIQUE,
                    block_index INT,
                    sender VARCHAR(128),
                    recipient VARCHAR(128),
                    amount FLOAT,
                    energy FLOAT,
                    transaction_type VARCHAR(32),
                    timestamp DOUBLE
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    node_id VARCHAR(128) NOT NULL UNIQUE,
                    node_type VARCHAR(32) NOT NULL,
                    wallet_balance FLOAT NOT NULL,
                    energy_balance FLOAT DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_stats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DOUBLE NOT NULL,
                    total_energy_traded FLOAT NOT NULL,
                    avg_energy_price FLOAT NOT NULL,
                    block_count INT NOT NULL,
                    transaction_count INT NOT NULL
                )
            """)
            
            self.conn.commit()
            self.setup_complete = True
            logger.info("Database schema verified/created successfully.")
        except mysql.connector.Error as err:
            logger.error(f"FATAL: Error setting up database schema: {err}")
            self.conn.rollback()
            self.setup_complete = False

    def save_block(self, block: Dict[str, Any]):
        if not self.setup_complete: return
        try:
            block_sql = "INSERT INTO blocks (block_index, hash, previous_hash, timestamp, nonce, miner_id, transactions) VALUES (%s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE hash=VALUES(hash), nonce=VALUES(nonce)"
            self.cursor.execute(block_sql, (block['index'], block['hash'], block['previous_hash'], block['timestamp'], block['nonce'], block['miner_id'], json.dumps(block['transactions'])))
            
            tx_sql = "INSERT INTO transactions (tx_id, block_index, sender, recipient, amount, energy, transaction_type, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE block_index=VALUES(block_index)"
            for tx in block.get('transactions', []):
                self.cursor.execute(tx_sql, (tx['tx_id'], block['index'], tx['sender'], tx['recipient'], tx['amount'], tx['energy'], tx['type'], tx['timestamp']))
            
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Error saving block to database: {err}")
            self.conn.rollback()

    def save_node(self, node_id: str, node_type: str, wallet_balance: float, energy_balance: float = 0.0):
        if not self.setup_complete: return
        try:
            node_sql = "INSERT INTO nodes (node_id, node_type, wallet_balance, energy_balance) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE wallet_balance = VALUES(wallet_balance), energy_balance = VALUES(energy_balance)"
            self.cursor.execute(node_sql, (node_id, node_type, wallet_balance, energy_balance))
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Error saving node to database: {err}")
            self.conn.rollback()

    def save_stats(self, stats: Dict[str, Union[float, int]]):
        if not self.setup_complete: return
        try:
            stats_sql = "INSERT INTO simulation_stats (timestamp, total_energy_traded, avg_energy_price, block_count, transaction_count) VALUES (%s, %s, %s, %s, %s)"
            self.cursor.execute(stats_sql, (stats['timestamp'], stats['total_energy_traded'], stats['avg_energy_price'], stats['block_count'], stats['transaction_count']))
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Error saving stats to database: {err}")
            self.conn.rollback()
    
    def close(self):
        if self.conn and self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
            logger.info("Database connection closed")