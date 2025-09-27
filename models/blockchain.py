import hashlib
import json
import time
import uuid
import logging
import threading
from typing import List, Dict, Any, Optional
from queue import Queue

logger = logging.getLogger("simulation_logger")

try:
    from utils.db_utils import DatabaseManager
except ImportError:
    DatabaseManager = None

class Block:
    def __init__(self, index: int, timestamp: float, transactions: List[Dict[str, Any]], previous_hash: str, miner_id: str = "", nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.transactions = sorted(transactions, key=lambda t: t.get('tx_id', ''))
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.miner_id = miner_id
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_dict = self.to_dict(include_hash=False)
        block_string = json.dumps(block_dict, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self, include_hash=True) -> Dict[str, Any]:
        data = {
            'index': self.index, 'previous_hash': self.previous_hash, 'timestamp': self.timestamp,
            'nonce': self.nonce, 'miner_id': self.miner_id, 'transactions': self.transactions
        }
        if include_hash:
            data['hash'] = self.hash
        return data

class Blockchain:
    def __init__(self, difficulty: int = 3, db_write_queue: Optional[Queue] = None):
        self.chain: List[Block] = []
        self.current_transactions: List[Dict[str, Any]] = []
        self.difficulty = difficulty
        self.lock = threading.Lock()
        self.wallets: Dict[str, float] = {}
        self.db_write_queue = db_write_queue
        self.create_genesis_block()

    def create_genesis_block(self):
        with self.lock:
            if not self.chain:
                genesis_block = Block(0, time.time(), [], "0", "genesis")
                genesis_block.hash = genesis_block.calculate_hash()
                self.chain.append(genesis_block)
                if self.db_write_queue:
                    self.db_write_queue.put({'type': 'block', 'data': genesis_block.to_dict()})

    def add_block(self, block: Block) -> bool:
        with self.lock:
            if self.chain and block.previous_hash != self.chain[-1].hash: return False
            for tx in block.transactions:
                if tx.get('type') == 'energy_payment':
                    sender, recipient, amount = tx['sender'], tx['recipient'], tx['amount']
                    if self.wallets.get(sender, 0) >= amount:
                        self.wallets[sender] -= amount
                        self.wallets[recipient] = self.wallets.get(recipient, 0) + amount
            if block.miner_id != "genesis": self.wallets[block.miner_id] = self.wallets.get(block.miner_id, 0) + 1.0
            
            self.chain.append(block)
            logger.info(f'MINED_BLOCK: {json.dumps(block.to_dict())}')
            if self.db_write_queue:
                self.db_write_queue.put({'type': 'block', 'data': block.to_dict()})
        return True
    
    def get_balance(self, node_id: str) -> float:
        with self.lock: return self.wallets.get(node_id, 0.0)
            
    def register_node_wallet(self, node_id: str, starting_balance: float):
        with self.lock:
            if node_id not in self.wallets: self.wallets[node_id] = starting_balance
    
    def proof_of_work(self, block: Block):
        target = "0" * self.difficulty
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.calculate_hash()
    
    def create_new_block(self, miner_id: str) -> Optional[Block]:
        with self.lock:
            if not self.current_transactions: return None
            transactions_for_block = list(self.current_transactions)
            self.current_transactions.clear()
            previous_block = self.chain[-1]
        new_block = Block(previous_block.index + 1, time.time(), transactions_for_block, previous_block.hash, miner_id)
        self.proof_of_work(new_block)
        if self.add_block(new_block): return new_block
        else:
            with self.lock: self.current_transactions.extend(transactions_for_block)
            return None

    def new_transaction(self, sender: str, recipient: str, amount: float, energy: float, transaction_type: str, metadata: Optional[Dict] = None) -> str:
        """Creates a new transaction, optionally including metadata."""
        with self.lock:
            if "payment" in transaction_type:
                if self.wallets.get(sender, 0) < amount:
                    return ""
            tx_id = uuid.uuid4().hex
            transaction = {
                'sender': sender, 'recipient': recipient, 'amount': amount,
                'energy': energy, 'type': transaction_type, 'tx_id': tx_id,
                'timestamp': time.time()
            }
            if metadata:
                transaction.update(metadata)
            
            self.current_transactions.append(transaction)
            return tx_id

            
    def get_pending_transactions(self) -> List[Dict[str, Any]]:
        with self.lock: return list(self.current_transactions)