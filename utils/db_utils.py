# utils/db_utils.py (MySQL Version)

import mysql.connector
from mysql.connector import Error as MySQLError
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages the connection and operations with the MySQL database."""

    def __init__(self, db_config: Dict[str, Any]):
        self.conn = None
        try:
            self.conn = mysql.connector.connect(**db_config)
            self.cursor = self.conn.cursor(dictionary=True)
            logger.info("Successfully connected to MySQL database.")
            self._create_tables()
        except MySQLError as e:
            logger.error(f"Could not connect to MySQL database: {e}")
            self.conn = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during database initialization: {e}")
            self.conn = None

    def _create_tables(self):
        """Creates the 'blocks' table, dropping the old one if it exists."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS blocks (
            block_index INT PRIMARY KEY,
            hash TEXT UNIQUE NOT NULL,
            previous_hash TEXT NOT NULL,
            timestamp DOUBLE NOT NULL,
            nonce INT NOT NULL,
            miner_id VARCHAR(255),
            transactions JSON
        );
        """
        try:
            self.cursor.execute("DROP TABLE IF EXISTS blocks;")
            self.cursor.execute(create_table_query)
            self.conn.commit()
            logger.info("Table 'blocks' is ready (recreated for this run).")
        except MySQLError as e:
            logger.error(f"Failed to create table 'blocks': {e}")
            self.conn.rollback()

    def save_block(self, block_data: Dict[str, Any]):
        """Saves a single block to the database."""
        if not self.conn:
            return
            
        insert_query = """
        INSERT INTO blocks (block_index, hash, previous_hash, timestamp, nonce, miner_id, transactions)
        VALUES (%(block_index)s, %(hash)s, %(previous_hash)s, %(timestamp)s, %(nonce)s, %(miner_id)s, %(transactions)s)
        ON DUPLICATE KEY UPDATE hash=VALUES(hash);
        """
        try:
            block_data_copy = block_data.copy()
            block_data_copy['transactions'] = json.dumps(block_data_copy['transactions'])
            
            self.cursor.execute(insert_query, block_data_copy)
            self.conn.commit()
        except MySQLError as e:
            logger.error(f"Failed to insert block {block_data.get('block_index')}: {e}")
            self.conn.rollback()
            
    def close(self):
        """Closes the database connection."""
        if self.conn and self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
            logger.info("MySQL database connection closed.")