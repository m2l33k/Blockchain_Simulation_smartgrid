from __future__ import annotations # MUST BE THE VERY FIRST LINE
import hashlib
import json
import time
import uuid
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger('blockchain')

class Block:
    """A block in the blockchain"""
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
        """Calculate the hash of this block"""
        block_string = json.dumps({
            "index": self.index, "timestamp": self.timestamp,
            "transactions": self.transactions, "previous_hash": self.previous_hash,
            "nonce": self.nonce, "miner_id": self.miner_id
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    """A blockchain implementation for a decentralized smart grid system"""
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.current_transactions: List[Dict[str, Any]] = []
        self.nodes: set[BaseNode] = set()
        self.difficulty = difficulty
        self.create_genesis_block()
    
    def create_genesis_block(self):
        genesis_block = Block(index=0, timestamp=time.time(), transactions=[], previous_hash="0", miner_id="genesis")
        self.proof_of_work(genesis_block)
        self.chain.append(genesis_block)
        logger.info(f"Genesis block created: {genesis_block.hash[:12]}")
    
    def proof_of_work(self, block: Block):
        target = "0" * self.difficulty
        while block.hash[:self.difficulty] != target:
            block.nonce += 1
            block.hash = block.calculate_hash()
    
    def add_block(self, block: Block) -> bool:
        if block.previous_hash != self.chain[-1].hash or block.index != len(self.chain): return False
        if block.hash[:self.difficulty] != "0" * self.difficulty: return False
        
        self.chain.append(block)
        logger.info(f"Block #{block.index} added. Miner: {block.miner_id}, Txs: {len(block.transactions)}")
        
        # Process transactions within the block
        for tx in block.transactions:
            for node in self.nodes:
                if hasattr(node, 'process_transaction'):
                    node.process_transaction(tx)

        # Notify nodes of the new block
        for node in list(self.nodes):
            if hasattr(node, 'receive_block') and node.node_id != block.miner_id:
                node.receive_block(block)
        return True
    
    def create_new_block(self, miner_id: str) -> Optional[Block]:
        if not self.current_transactions: return None
        
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), time.time(), list(self.current_transactions), previous_block.hash, miner_id)
        
        self.proof_of_work(new_block)
        
        if self.add_block(new_block):
            self.current_transactions = []
            return new_block
        return None
    
    def new_transaction(self, sender: str, recipient: str, amount: float, 
                       energy: float = 0, transaction_type: str = "financial") -> str:
        tx_id = uuid.uuid4().hex
        transaction = {
            'sender': sender, 'recipient': recipient, 'amount': amount,
            'energy': energy, 'type': transaction_type, 'tx_id': tx_id
        }
        self.current_transactions.append(transaction)
        
        for node in list(self.nodes):
            if hasattr(node, 'on_new_transaction'): node.on_new_transaction(transaction)
        return tx_id
    
    def get_pending_transactions(self) -> List[Dict[str, Any]]:
        return self.current_transactions

    def validate_block(self, block: Block) -> bool:
        return (block.previous_hash == self.chain[-1].hash and
                block.index == len(self.chain) and
                block.hash[:self.difficulty] == "0" * self.difficulty and
                block.hash == block.calculate_hash())

# This line ensures the module can be run standalone for testing without error
if __name__ == '__main__':
    print(f"models.blockchain.py is syntactically correct and can be imported.")