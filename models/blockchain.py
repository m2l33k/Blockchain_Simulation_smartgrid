import hashlib
import json
import time
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional

try:
    import numpy as np
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('blockchain')

class Block:
    def __init__(self, index: int, timestamp: float, transactions: List[Dict[str, Any]], 
                 previous_hash: str, nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate the hash of the block."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "nonce": self.nonce,
        }
    
    def __repr__(self) -> str:
        return f"Block(index={self.index}, hash={self.hash[:10]}...)"


class Blockchain:
    def __init__(self, difficulty: int = 4, use_gpu: bool = False):
        self.chain: List[Block] = []
        self.current_transactions: List[Dict[str, Any]] = []
        self.nodes = set()  # This will store node objects instead of just addresses
        self.difficulty = difficulty
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Create the genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        """Create the first block in the chain."""
        genesis_block = Block(0, time.time(), [], "0")
        self.proof_of_work(genesis_block)
        self.chain.append(genesis_block)
        logger.info(f"Genesis block created: {genesis_block.hash}")
    
    def proof_of_work(self, block: Block) -> None:
        """
        Proof of Work algorithm:
        - Find a number (nonce) such that hash(block) contains leading zeros equal to the difficulty
        """
        if self.use_gpu and GPU_AVAILABLE:
            self._gpu_mine(block)
        else:
            self._cpu_mine(block)
    
    def _cpu_mine(self, block: Block) -> None:
        """CPU-based mining implementation."""
        target = "0" * self.difficulty
        
        while block.hash[:self.difficulty] != target:
            block.nonce += 1
            block.hash = block.calculate_hash()
    
    def _gpu_mine(self, block: Block) -> None:
        """GPU-accelerated mining implementation using CUDA."""
        try:
            target = "0" * self.difficulty
            batch_size = 100000  # Process in batches
            
            while block.hash[:self.difficulty] != target:
                # Generate a batch of nonces
                start_nonce = block.nonce
                end_nonce = start_nonce + batch_size
                
                # Create base block string without nonce
                base_block = {
                    "index": block.index,
                    "timestamp": block.timestamp,
                    "transactions": block.transactions,
                    "previous_hash": block.previous_hash,
                }
                base_string = json.dumps(base_block, sort_keys=True)
                
                # Find hash using GPU
                found, nonce, hash_val = self._gpu_find_hash(base_string, start_nonce, end_nonce, target)
                
                if found:
                    block.nonce = nonce
                    block.hash = hash_val
                    break
                else:
                    block.nonce = end_nonce
                    
        except Exception as e:
            logger.warning(f"GPU mining failed: {e}. Falling back to CPU mining.")
            self._cpu_mine(block)
    
    def _gpu_find_hash(self, base_string: str, start_nonce: int, end_nonce: int, target: str) -> tuple:
        """
        Use GPU to find a valid hash. Returns (found, nonce, hash_value).
        This is a simplified version; a real implementation would use CUDA kernels.
        """
        nonces = cp.arange(start_nonce, end_nonce, dtype=cp.int32)
        
        # In a real implementation, this would be a CUDA kernel
        for nonce in range(start_nonce, end_nonce):
            block_string = f'{base_string},"nonce":{nonce}' + "}"
            hash_val = hashlib.sha256(block_string.encode()).hexdigest()
            
            if hash_val[:self.difficulty] == target:
                return True, nonce, hash_val
                
        return False, end_nonce, ""
    
    def add_block(self, block: Block) -> bool:
        """
        Add a new block to the chain after verification.
        Returns: True if block was added, False otherwise
        """
        # Check if block index is valid
        if block.index != len(self.chain):
            logger.warning(f"Invalid block index: {block.index}, expected {len(self.chain)}")
            return False
            
        # Check if previous hash matches the hash of the last block in the chain
        if block.previous_hash != self.chain[-1].hash:
            logger.warning("Invalid previous hash")
            return False
            
        # Verify the proof of work
        if block.hash[:self.difficulty] != "0" * self.difficulty:
            logger.warning("Invalid proof of work")
            return False
            
        # Verify the hash is correct
        if block.hash != block.calculate_hash():
            logger.warning("Block hash verification failed")
            return False
        
        self.chain.append(block)
        logger.info(f"Block #{block.index} added to the blockchain")
        return True
    
    def create_new_block(self) -> Block:
        """
        Create a new block in the blockchain
        """
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.current_transactions.copy(),
            previous_hash=previous_block.hash
        )
        
        # Reset the current list of transactions
        self.current_transactions = []
        
        # Find the proof of work for the new block
        self.proof_of_work(new_block)
        
        # Add the new block to the chain
        self.add_block(new_block)
        
        return new_block
    
    def new_transaction(self, sender: str, recipient: str, amount: float, 
                       energy: float = 0, transaction_type: str = "financial") -> int:
        """
        Creates a new transaction to go into the next mined Block
        
        :param sender: Address of the Sender
        :param recipient: Address of the Recipient
        :param amount: Amount of currency
        :param energy: Amount of energy (for smart grid transactions)
        :param transaction_type: Type of transaction (financial, energy, data)
        :return: The index of the Block that will hold this transaction
        """
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
            'energy': energy,
            'type': transaction_type,
            'timestamp': time.time()
        }
        
        self.current_transactions.append(transaction)
        logger.debug(f"New transaction: {sender} -> {recipient}, {amount} units, {energy} kWh, type: {transaction_type}")
        
        # Process the transaction if it's an energy transaction
        if transaction_type in ['energy_offer', 'energy_request', 'energy_delivery', 'energy_payment']:
            # If the recipient is GRID_OPERATOR, pass the transaction to all grid operators
            for node in self.nodes:
                if hasattr(node, 'process_transaction') and transaction['recipient'] == node.node_id:
                    node.process_transaction(transaction)
                    break
        
        return len(self.chain)
    
    def get_chain(self) -> List[Dict[str, Any]]:
        """Return the full blockchain as a list of dictionaries"""
        return [block.to_dict() for block in self.chain]
    
    def is_chain_valid(self) -> bool:
        """
        Determine if the blockchain is valid
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check if the current block's hash is correct
            if current.hash != current.calculate_hash():
                logger.error(f"Block #{current.index} has invalid hash")
                return False
            
            # Check if the previous hash reference is correct
            if current.previous_hash != previous.hash:
                logger.error(f"Block #{current.index} has invalid previous hash reference")
                return False
            
            # Check if the proof of work is valid
            if current.hash[:self.difficulty] != "0" * self.difficulty:
                logger.error(f"Block #{current.index} has invalid proof of work")
                return False
        
        return True

    def register_node(self, node):
        """Add a new node to the list of nodes in the network"""
        self.nodes.add(node)
        return self
    
    def __repr__(self) -> str:
        return f"Blockchain(blocks={len(self.chain)}, difficulty={self.difficulty})"