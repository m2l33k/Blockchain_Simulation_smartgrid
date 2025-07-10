from __future__ import annotations
import time
import uuid
import random
import logging
import threading
from typing import Dict, List, Any, Optional

# Get the logger instance for this module
logger = logging.getLogger(__name__)

SIMULATION_SPEED_FACTOR = 360

class SmartMeter:
    def __init__(self, owner_id: str, prod_cap: float, cons_rate: float, sim_start_time: float):
        self.owner_id = owner_id
        self.production_capacity = prod_cap
        self.consumption_rate = cons_rate
        # FIX: The simulation start time is now passed in, not defined globally.
        self.sim_start_time = sim_start_time
        self.last_reading_time = time.time()
        self.owner_node: Optional[GridNode] = None

    def connect_to_node(self, node: GridNode):
        self.owner_node = node

    def read_meter(self) -> Dict[str, Any]:
        current_time = time.time()
        simulated_hours = (current_time - self.last_reading_time) * SIMULATION_SPEED_FACTOR / 3600.0
        if simulated_hours <= 0: return {'net_energy': 0, 'hour': 0}

        simulated_seconds_elapsed = (current_time - self.sim_start_time) * SIMULATION_SPEED_FACTOR
        hour_of_day = (7 + (simulated_seconds_elapsed / 3600)) % 24

        solar_factor = max(0, 1 - abs(hour_of_day - 14) / 7) * random.uniform(0.7, 1.0)
        production = self.production_capacity * solar_factor * simulated_hours

        peak_factor = 0.5 + 0.5 * max(max(0, 1 - abs(hour_of_day - 8)/3), max(0, 1 - abs(hour_of_day - 19)/4))
        consumption = self.consumption_rate * peak_factor * random.uniform(0.8, 1.2) * simulated_hours
        
        self.last_reading_time = current_time
        net_energy = production - consumption
        
        if self.owner_node: self.owner_node.process_meter_reading({'net_energy': net_energy, 'hour': hour_of_day})
        return {'net_energy': net_energy, 'hour': hour_of_day}

# --- Forward declaration for type hints ---
class BaseNode: pass
class Blockchain: pass
class Block: pass
class GridOperator(BaseNode): pass

# --- BaseNode Class ---
class BaseNode:
    def __init__(self, node_id: str, wallet_balance: float):
        self.node_id = node_id
        self.wallet_balance = wallet_balance
        self.blockchain: Optional[Blockchain] = None
        self.can_mine = False
        self.mining_thread: Optional[threading.Thread] = None
        self.mining_active = False
        self.active = True
        
        # Track whether we need to save to database
        self._last_db_update = time.time()
        self._db_update_interval = 10  # seconds

    def register_with_blockchain(self, blockchain: Blockchain):
        self.blockchain = blockchain
        blockchain.nodes.add(self)
        logger.info(f"Node {self.node_id} registered.")
        
        # Initial save to database
        self._save_to_db()

    def create_transaction(self, recipient: str, amount: float, energy: float, tx_type: str) -> str:
        if not self.blockchain: return ""
        if tx_type.endswith("payment") and amount > self.wallet_balance: return ""
        tx_id = self.blockchain.new_transaction(self.node_id, recipient, amount, energy, tx_type)
        if tx_type.endswith("payment"): self.wallet_balance -= amount
        
        # Save state to database when wallet balance changes
        self._save_to_db()
        
        return tx_id
    
    def _save_to_db(self):
        """Save node data to database"""
        current_time = time.time()
        if current_time - self._last_db_update < self._db_update_interval:
            return  # Don't update too frequently
            
        self._last_db_update = current_time
        
        if hasattr(self, 'energy_storage'):
            energy_balance = getattr(self, 'energy_storage', 0.0)
        else:
            energy_balance = 0.0
            
        if self.blockchain and hasattr(self.blockchain, 'db_manager') and self.blockchain.db_manager:
            node_type = self.__class__.__name__.lower()
            self.blockchain.db_manager.save_node(
                self.node_id, 
                node_type, 
                self.wallet_balance, 
                energy_balance
            )

    def start_mining(self):
        if self.can_mine and not self.mining_active:
            self.mining_active = True
            self.mining_thread = threading.Thread(target=self._mining_worker, daemon=True)
            self.mining_thread.start()

    def stop_mining(self):
        if self.mining_active:
            self.mining_active = False
            if self.mining_thread and self.mining_thread.is_alive(): self.mining_thread.join(timeout=1.0)
            self.mining_thread = None

    def _mining_worker(self):
        while self.active and self.mining_active:
            if self.blockchain and self.blockchain.get_pending_transactions():
                block = self.blockchain.create_new_block(miner_id=self.node_id)
                if block: self.wallet_balance += 1.0
            else: time.sleep(2)

    def process_transaction(self, transaction: Dict[str, Any]):
        if transaction['recipient'] == self.node_id:
            if transaction['type'] == 'energy_payment': 
                self.wallet_balance += transaction['amount']
                # Save state after receiving payment
                self._save_to_db()
            elif transaction['type'] == 'energy_delivery' and hasattr(self, 'energy_storage'):
                self.energy_storage = min(self.max_storage, self.energy_storage + transaction['energy'] * self.storage_efficiency)
                logger.info(f"Node {self.node_id} received {transaction['energy']:.2f} kWh, storage is now {self.energy_storage:.2f} kWh.")
                self.create_transaction(transaction['sender'], transaction['amount'], 0, 'energy_payment')
                # State will be saved by create_transaction

# --- GridOperator Class ---
class GridOperator(BaseNode):
    def __init__(self, node_id: str, wallet_balance: float):
        super().__init__(node_id, wallet_balance)
        self.can_mine = True
        self.base_price = 0.15
        self.order_book = {'offers': [], 'requests': []}
        self.lock = threading.Lock()
        self.update_thread: Optional[threading.Thread] = None

    def start(self):
        logger.info(f"GridOperator {self.node_id} started.")
        self.start_mining()
        self.update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.update_thread.start()

    def stop(self):
        self.active = False
        self.stop_mining()
        if self.update_thread and self.update_thread.is_alive(): self.update_thread.join(timeout=1.0)
        logger.info(f"GridOperator {self.node_id} stopped.")

    def submit_offer(self, offer: Dict[str, Any]):
        with self.lock: self.order_book['offers'].append(offer)

    def submit_request(self, request: Dict[str, Any]):
        with self.lock: self.order_book['requests'].append(request)

    def _monitoring_loop(self):
        while self.active:
            self._clear_market()
            time.sleep(5)
    
    def _clear_market(self):
        with self.lock:
            offers, requests = self.order_book['offers'], self.order_book['requests']
            if not offers or not requests: return

            offers.sort(key=lambda o: o['price'])
            requests.sort(key=lambda r: r['price'], reverse=True)

            trades_executed = 0
            total_energy_traded = 0
            total_value = 0
            
            for req in list(requests):
                for offer in list(offers):
                    if offer['price'] <= req['price']:
                        trade_energy = min(offer['energy'], req['energy'])
                        if trade_energy < 0.01: continue
                        trade_amount = trade_energy * offer['price']
                        
                        self.create_transaction(req['sender_id'], trade_amount, trade_energy, 'energy_delivery')
                        trades_executed += 1
                        total_energy_traded += trade_energy
                        total_value += trade_amount
                        
                        offer['energy'] -= trade_energy
                        req['energy'] -= trade_energy
            
            if trades_executed > 0: 
                logger.info(f"GridOperator cleared {trades_executed} trades.")
                
                # Save market statistics to database
                if self.blockchain and hasattr(self.blockchain, 'db_manager') and self.blockchain.db_manager:
                    avg_price = total_value / total_energy_traded if total_energy_traded > 0 else 0
                    block_count = len(self.blockchain.chain) if self.blockchain else 0
                    
                    stats = {
                        'timestamp': time.time(),
                        'total_energy_traded': total_energy_traded,
                        'avg_energy_price': avg_price,
                        'block_count': block_count,
                        'transaction_count': trades_executed
                    }
                    
                    self.blockchain.save_simulation_stats(stats)
                
            self.order_book['offers'] = [o for o in offers if o['energy'] > 0.01]
            self.order_book['requests'] = [r for r in requests if r['energy'] > 0.01]

    def get_market_price(self) -> float:
        with self.lock:
            valid_offers = [o for o in self.order_book['offers'] if o['energy'] > 0]
            if not valid_offers: return self.base_price * 1.1
            return min(o['price'] for o in valid_offers)

# --- GridNode Class ---
class GridNode(BaseNode):
    def __init__(self, node_id: str, wallet_balance: float, is_producer: bool, prod_cap: float, cons_rate: float, sim_start_time: float):
        super().__init__(node_id, wallet_balance)
        self.is_producer = is_producer
        self.smart_meter = SmartMeter(node_id, prod_cap, cons_rate, sim_start_time)
        self.smart_meter.connect_to_node(self)
        self.energy_storage = 0.0
        self.max_storage = 10.0
        self.storage_efficiency = 0.95
        self.grid_operator: Optional[GridOperator] = None
        self.update_thread: Optional[threading.Thread] = None

    def start(self):
        logger.info(f"Node {self.node_id} started.")
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def stop(self):
        self.active = False
        self.stop_mining()
        if self.update_thread and self.update_thread.is_alive(): self.update_thread.join(timeout=1.0)
        logger.info(f"Node {self.node_id} stopped.")

    def connect_to_node(self, other_node: BaseNode):
        if isinstance(other_node, GridOperator): self.grid_operator = other_node

    def _update_loop(self):
        time.sleep(random.uniform(1, 5))
        while self.active:
            self.update_energy_status()
            time.sleep(random.uniform(5, 10))

    def update_energy_status(self):
        reading = self.smart_meter.read_meter()
        net_energy, hour = reading['net_energy'], reading.get('hour', 12)

        if net_energy > 0: # Surplus
            to_store = min(net_energy, self.max_storage - self.energy_storage)
            if to_store > 0: self.energy_storage += to_store * self.storage_efficiency
            remaining_excess = net_energy - (to_store / (self.storage_efficiency or 1))
            if remaining_excess > 0.1: self._offer_excess_energy(remaining_excess)
        elif net_energy < 0: # Deficit
            from_storage = min(self.energy_storage, abs(net_energy) / (self.storage_efficiency or 1))
            self.energy_storage -= from_storage
            remaining_deficit = abs(net_energy) - (from_storage * self.storage_efficiency)
            if remaining_deficit > 0.1: self._request_needed_energy(remaining_deficit)
            
        # Save node data to database periodically
        self._save_to_db()

    def _offer_excess_energy(self, amount: float):
        if self.grid_operator:
            price = self.grid_operator.get_market_price() * random.uniform(0.95, 1.05)
            self.grid_operator.submit_offer({'sender_id': self.node_id, 'energy': amount, 'price': price})

    def _request_needed_energy(self, amount: float):
        if self.grid_operator:
            price = self.grid_operator.get_market_price() * random.uniform(0.98, 1.1)
            self.grid_operator.submit_request({'sender_id': self.node_id, 'energy': amount, 'price': price})

    def process_meter_reading(self, reading: Dict[str, Any]): pass

# --- Prosumer and Consumer Classes ---
class Prosumer(GridNode):
    def __init__(self, node_id: str, wallet_balance: float, production_capacity: float, consumption_rate: float, sim_start_time: float):
        super().__init__(node_id, wallet_balance, True, production_capacity, consumption_rate, sim_start_time)
        self.can_mine = True
        self.max_storage = 15.0
        self.energy_storage = random.uniform(5.0, 10.0)

class Consumer(GridNode):
    def __init__(self, node_id: str, wallet_balance: float, consumption_rate: float, sim_start_time: float):
        super().__init__(node_id, wallet_balance, False, 0.0, consumption_rate, sim_start_time)
        self.max_storage = 5.0