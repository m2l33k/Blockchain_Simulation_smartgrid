from __future__ import annotations
import hashlib
import json
import time
import uuid
import logging
import threading
from typing import List, Dict, Any, Optional

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
        block_string = json.dumps({
            "index": self.index, "timestamp": self.timestamp,
            "transactions": self.transactions, "previous_hash": self.previous_hash,
            "nonce": self.nonce, "miner_id": self.miner_id
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the block to a dictionary, perfect for JSON logging."""
        return {
            "index": self.index, "timestamp": self.timestamp,
            "transactions": self.transactions, "previous_hash": self.previous_hash,
            "hash": self.hash, "nonce": self.nonce, "miner_id": self.miner_id
        }

class Blockchain:
    def __init__(self, difficulty: int = 4, db_config=None): # Added db_config for your future use
        self.chain: List[Block] = []
        self.current_transactions: List[Dict[str, Any]] = []
        self.nodes: set[BaseNode] = set()
        self.difficulty = difficulty
        self.lock = threading.Lock()
        self.create_genesis_block()
    
    def create_genesis_block(self):
        genesis_block = Block(0, time.time(), [], "0", "genesis")
        self.proof_of_work(genesis_block)
        # FIX: Log genesis block in the same structured format
        logger.info(f"MINED_BLOCK: {json.dumps(genesis_block.to_dict())}")
        self.chain.append(genesis_block)
    
    def proof_of_work(self, block: Block):
        target = "0" * self.difficulty
        while block.hash[:self.difficulty] != target:
            block.nonce += 1
            block.hash = block.calculate_hash()
    
    def add_block(self, block: Block) -> bool:
        with self.lock:
            if not self.validate_block(block): return False
            self.chain.append(block)
            # FIX: Log the entire block as a structured JSON object
            # This is the key change for reliable parsing.
            logger.info(f"MINED_BLOCK: {json.dumps(block.to_dict())}")

        for node in list(self.nodes):
            for tx in block.transactions:
                if hasattr(node, 'process_transaction'):
                    node.process_transaction(tx)
        return True
    
    def create_new_block(self, miner_id: str) -> Optional[Block]:
        with self.lock:
            if not self.current_transactions: return None
            transactions_to_mine = list(self.current_transactions)
            self.current_transactions = []
        
        new_block = Block(len(self.chain), time.time(), transactions_to_mine, self.chain[-1].hash, miner_id)
        self.proof_of_work(new_block)
        
        if self.add_block(new_block):
            return new_block
        else: 
            with self.lock:
                self.current_transactions.extend(transactions_to_mine)
            return None
    
    def new_transaction(self, sender: str, recipient: str, amount: float, 
                       energy: float = 0, transaction_type: str = "financial") -> str:
        with self.lock:
            tx_id = uuid.uuid4().hex
            transaction = {
                'sender': sender, 'recipient': recipient, 'amount': amount,
                'energy': energy, 'type': transaction_type, 'tx_id': tx_id
            }
            self.current_transactions.append(transaction)
        return tx_id
    
    def get_pending_transactions(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.current_transactions)

    def validate_block(self, block: Block) -> bool:
        last_block = self.chain[-1]
        if block.index != last_block.index + 1: return False
        if block.previous_hash != last_block.hash: return False
        if block.hash[:self.difficulty] != "0" * self.difficulty: return False
        if block.hash != block.calculate_hash(): return False
        return True