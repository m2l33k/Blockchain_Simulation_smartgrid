# models/blockchain.py (Final Corrected Version)

from __future__ import annotations
import hashlib
import json
import time
import uuid
import logging
import threading
from typing import List, Dict, Any, Optional

try:
    from utils.db_utils import DatabaseManager
except ImportError:
    DatabaseManager = None
    logging.warning("utils/db_utils.py not found. Database features will be disabled.")

logger = logging.getLogger(__name__)

class Block:
    def __init__(self, index: int, timestamp: float, transactions: List[Dict[str, Any]],
                 previous_hash: str, miner_id: str = "", nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.transactions = sorted(transactions, key=lambda t: t.get('tx_id', ''))
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.miner_id = miner_id
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_content = {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'miner_id': self.miner_id
        }
        block_string = json.dumps(block_content, sort_keys=True, default=str).encode()
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {'index': self.index, 'hash': self.hash, 'previous_hash': self.previous_hash,
                'timestamp': self.timestamp, 'nonce': self.nonce, 'miner_id': self.miner_id,
                'transactions': self.transactions}

class Blockchain:
    def __init__(self, difficulty: int = 4, db_config: Optional[Dict[str, Any]] = None):
        self.chain: List[Block] = []
        self.current_transactions: List[Dict[str, Any]] = []
        self.difficulty = difficulty
        self.lock = threading.Lock()
        self.wallets: Dict[str, float] = {}
        
        self.db_manager = None
        if db_config and DatabaseManager:
            try:
                self.db_manager = DatabaseManager(db_config)
            except Exception as e:
                logger.error(f"Blockchain: Failed to instantiate DatabaseManager: {e}")
                self.db_manager = None

        self.create_genesis_block()

    def register_node_wallet(self, node_id: str, starting_balance: float):
        with self.lock:
            if node_id not in self.wallets:
                self.wallets[node_id] = starting_balance
                logger.info(f"Wallet for {node_id} registered with starting balance of ${starting_balance:.2f}")

    def create_genesis_block(self):
        if not self.chain:
            logger.info("Mining Genesis Block (Block #0)...")
            genesis_block = Block(index=0, timestamp=time.time(), transactions=[], previous_hash="0", miner_id="genesis")
            self.proof_of_work(genesis_block)
            self.add_block(genesis_block, is_genesis=True)

    def proof_of_work(self, block: Block):
        target = "0" * self.difficulty
        block.nonce = 0
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.calculate_hash()
        logger.info(f"Block #{block.index} Mined! Nonce: {block.nonce}")

    def add_block(self, block: Block, is_genesis: bool = False) -> bool:
        with self.lock:
            if not is_genesis:
                last_block = self.chain[-1]
                if block.previous_hash != last_block.hash:
                    logger.error(f"Failed to add Block #{block.index}: Invalid previous_hash.")
                    return False

            for tx in block.transactions:
                sender, recipient, amount = tx.get('sender'), tx.get('recipient'), tx.get('amount')
                if tx.get('type') == 'energy_payment' and self.wallets.get(sender, 0) >= amount:
                    self.wallets[sender] -= amount
                    self.wallets[recipient] = self.wallets.get(recipient, 0) + amount
            
            if block.miner_id != "genesis":
                self.wallets[block.miner_id] = self.wallets.get(block.miner_id, 0) + 1.0

            self.chain.append(block)
            block_dict = block.to_dict()

        db_block_dict = block_dict.copy()
        db_block_dict['block_index'] = db_block_dict.pop('index')
        logger.info(f'MINED_BLOCK: {json.dumps(db_block_dict)}')
        if self.db_manager:
            self.db_manager.save_block(db_block_dict)
            
        return True

    def create_new_block(self, miner_id: str) -> Optional[Block]:
        with self.lock:
            if not self.current_transactions: return None
            transactions_for_block = list(self.current_transactions)
            self.current_transactions.clear()
            previous_block = self.chain[-1]
        
        new_block = Block(index=previous_block.index + 1, timestamp=time.time(), transactions=transactions_for_block, previous_hash=previous_block.hash, miner_id=miner_id)
        self.proof_of_work(new_block)
        
        if self.add_block(new_block): return new_block
        else:
            with self.lock: self.current_transactions.extend(transactions_for_block)
            return None

    def new_transaction(self, sender: str, recipient: str, amount: float,
                       energy: float = 0, transaction_type: str = "financial") -> str:
        with self.lock:
            if "payment" in transaction_type and self.wallets.get(sender, 0) < amount: return ""
            tx_id = uuid.uuid4().hex
            self.current_transactions.append({'sender': sender, 'recipient': recipient, 'amount': amount, 'energy': energy, 'type': transaction_type, 'tx_id': tx_id, 'timestamp': time.time()})
            return tx_id

    def close(self):
        if self.db_manager:
            self.db_manager.close()