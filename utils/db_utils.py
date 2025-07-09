import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('db_utils')

# Try to import mysql.connector, but make it optional
MYSQL_AVAILABLE = False
try:
    import mysql.connector
    import json
    MYSQL_AVAILABLE = True
except ImportError:
    logger.warning("mysql-connector-python not installed. Database functionality will be disabled.")
    logger.warning("To enable database support, install: pip install mysql-connector-python")

class DatabaseManager:
    def __init__(self, host: str = 'localhost', user: str = 'root', 
                 password: str = '', database: str = 'smartgrid',
                 port: int = 3306, create_if_not_exists: bool = True):
        """
        Initialize database connection for the smart grid blockchain.
        
        Args:
            host: MySQL host
            user: MySQL username
            password: MySQL password
            database: MySQL database name
            port: MySQL port
            create_if_not_exists: If True, create the database and tables if they don't exist
        """
        if not MYSQL_AVAILABLE:
            logger.warning("Database functionality is disabled. Install mysql-connector-python to enable.")
            return
            
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'port': port
        }
        self.database = database
        self.conn = None
        self.cursor = None
        
        # Try to connect to the database
        self._connect()
        
        # Create database and tables if needed
        if create_if_not_exists:
            self._setup_database()
    
    def _connect(self) -> bool:
        """Connect to the MySQL database"""
        if not MYSQL_AVAILABLE:
            return False
            
        try:
            # First try to connect to the specified database
            self.conn = mysql.connector.connect(
                **self.config,
                database=self.database
            )
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.database}")
            return True
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                # Database doesn't exist, connect without specifying a database
                try:
                    self.conn = mysql.connector.connect(**self.config)
                    self.cursor = self.conn.cursor()
                    logger.info("Connected to MySQL server without database")
                    return True
                except mysql.connector.Error as err:
                    logger.error(f"Failed to connect to MySQL: {err}")
                    return False
            else:
                logger.error(f"Failed to connect to database: {err}")
                return False
    
    def _setup_database(self) -> None:
        """Set up the database and required tables if they don't exist"""
        if not self.conn:
            if not self._connect():
                return
        
        try:
            # Create database if it doesn't exist
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            self.cursor.execute(f"USE {self.database}")
            logger.info(f"Using database: {self.database}")
            
            # Create blocks table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    block_index INT NOT NULL,
                    block_hash VARCHAR(64) NOT NULL,
                    previous_hash VARCHAR(64) NOT NULL,
                    timestamp DOUBLE NOT NULL,
                    nonce INT NOT NULL,
                    transactions JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create transactions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    block_index INT NOT NULL,
                    sender VARCHAR(128) NOT NULL,
                    recipient VARCHAR(128) NOT NULL,
                    amount FLOAT NOT NULL,
                    energy FLOAT NOT NULL,
                    transaction_type VARCHAR(32) NOT NULL,
                    timestamp DOUBLE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create nodes table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    node_id VARCHAR(128) NOT NULL,
                    node_type VARCHAR(32) NOT NULL,
                    wallet_balance FLOAT NOT NULL,
                    energy_balance FLOAT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            
            # Create simulation_stats table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_stats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DOUBLE NOT NULL,
                    total_energy_traded FLOAT NOT NULL,
                    avg_energy_price FLOAT NOT NULL,
                    block_count INT NOT NULL,
                    transaction_count INT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.commit()
            logger.info("Database tables created or already exist")
        
        except mysql.connector.Error as err:
            logger.error(f"Error setting up database: {err}")
    
    def save_block(self, block: Dict[str, Any]) -> bool:
        """Save a block to the database"""
        if not MYSQL_AVAILABLE or not self.conn:
            return False
        
        try:
            # Insert block into blocks table
            self.cursor.execute("""
                INSERT INTO blocks 
                (block_index, block_hash, previous_hash, timestamp, nonce, transactions)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                block['index'],
                block['hash'],
                block['previous_hash'],
                block['timestamp'],
                block['nonce'],
                json.dumps(block['transactions'])
            ))
            
            # Insert all transactions into transactions table
            for tx in block['transactions']:
                self.cursor.execute("""
                    INSERT INTO transactions
                    (block_index, sender, recipient, amount, energy, transaction_type, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    block['index'],
                    tx['sender'],
                    tx['recipient'],
                    tx['amount'],
                    tx['energy'],
                    tx['type'],
                    tx['timestamp']
                ))
            
            self.conn.commit()
            logger.debug(f"Saved block {block['index']} to database")
            return True
            
        except Exception as err:
            logger.error(f"Error saving block to database: {err}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def save_node(self, node_id: str, node_type: str, wallet_balance: float, energy_balance: float = 0) -> bool:
        """Save node information to the database"""
        if not MYSQL_AVAILABLE or not self.conn:
            return False
        
        try:
            # Insert or update node in nodes table
            self.cursor.execute("""
                INSERT INTO nodes (node_id, node_type, wallet_balance, energy_balance)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    wallet_balance = %s,
                    energy_balance = %s
            """, (
                node_id, node_type, wallet_balance, energy_balance,
                wallet_balance, energy_balance
            ))
            
            self.conn.commit()
            logger.debug(f"Saved node {node_id} to database")
            return True
            
        except Exception as err:
            logger.error(f"Error saving node to database: {err}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def save_stats(self, stats: Dict[str, Union[float, int]]) -> bool:
        """Save simulation statistics to the database"""
        if not MYSQL_AVAILABLE or not self.conn:
            return False
        
        try:
            self.cursor.execute("""
                INSERT INTO simulation_stats
                (timestamp, total_energy_traded, avg_energy_price, block_count, transaction_count)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                stats['timestamp'],
                stats['total_energy_traded'],
                stats['avg_energy_price'],
                stats['block_count'],
                stats['transaction_count']
            ))
            
            self.conn.commit()
            logger.debug("Saved simulation stats to database")
            return True
            
        except Exception as err:
            logger.error(f"Error saving stats to database: {err}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def get_blocks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent blocks from the database"""
        if not MYSQL_AVAILABLE or not self.conn:
            return []
        
        try:
            self.cursor.execute("""
                SELECT block_index, block_hash, previous_hash, timestamp, nonce, transactions
                FROM blocks
                ORDER BY block_index DESC
                LIMIT %s
            """, (limit,))
            
            result = []
            for (block_index, block_hash, previous_hash, timestamp, nonce, transactions) in self.cursor:
                result.append({
                    'index': block_index,
                    'hash': block_hash,
                    'previous_hash': previous_hash,
                    'timestamp': timestamp,
                    'nonce': nonce,
                    'transactions': json.loads(transactions)
                })
            
            return result
            
        except mysql.connector.Error as err:
            logger.error(f"Error retrieving blocks from database: {err}")
            return []
    
    def get_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent transactions from the database"""
        if not MYSQL_AVAILABLE or not self.conn:
            return []
        
        try:
            self.cursor.execute("""
                SELECT block_index, sender, recipient, amount, energy, transaction_type, timestamp
                FROM transactions
                ORDER BY id DESC
                LIMIT %s
            """, (limit,))
            
            result = []
            for (block_index, sender, recipient, amount, energy, transaction_type, timestamp) in self.cursor:
                result.append({
                    'block_index': block_index,
                    'sender': sender,
                    'recipient': recipient,
                    'amount': amount,
                    'energy': energy,
                    'type': transaction_type,
                    'timestamp': timestamp
                })
            
            return result
            
        except mysql.connector.Error as err:
            logger.error(f"Error retrieving transactions from database: {err}")
            return []
    
    def get_energy_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get energy trading statistics for the specified time period"""
        if not MYSQL_AVAILABLE or not self.conn:
            return {}
        
        try:
            # Calculate timestamp threshold
            current_time = datetime.now().timestamp()
            threshold = current_time - (hours * 3600)
            
            # Query total energy traded
            self.cursor.execute("""
                SELECT SUM(energy) as total_energy, AVG(amount/energy) as avg_price
                FROM transactions
                WHERE timestamp > %s AND transaction_type IN ('energy_delivery', 'energy_offer')
                AND energy > 0
            """, (threshold,))
            
            total_energy, avg_price = self.cursor.fetchone()
            
            # Query transaction counts by type
            self.cursor.execute("""
                SELECT transaction_type, COUNT(*) as count
                FROM transactions
                WHERE timestamp > %s
                GROUP BY transaction_type
            """, (threshold,))
            
            tx_counts = {}
            for (tx_type, count) in self.cursor:
                tx_counts[tx_type] = count
            
            return {
                'period_hours': hours,
                'total_energy_traded': float(total_energy) if total_energy else 0,
                'avg_energy_price': float(avg_price) if avg_price else 0,
                'transaction_counts': tx_counts
            }
            
        except mysql.connector.Error as err:
            logger.error(f"Error retrieving energy stats from database: {err}")
            return {}
    
    def close(self) -> None:
        """Close the database connection"""
        if not MYSQL_AVAILABLE:
            return
            
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")