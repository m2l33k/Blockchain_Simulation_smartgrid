# models/grid_nodes.py (Final Corrected Version with Realistic Logic)

from __future__ import annotations
import time
import uuid
import random
import logging
import threading
from typing import Dict, Any, Optional, List

from models.blockchain import Blockchain

logger = logging.getLogger(__name__)
SIMULATION_SPEED_FACTOR = 360

class SmartMeter:
    def __init__(self, owner_id: str, prod_cap: float, cons_rate: float, sim_start_time: float, is_commercial: bool = False):
        self.owner_id = owner_id
        self.production_capacity = prod_cap
        self.base_consumption_rate = cons_rate
        self.sim_start_time = sim_start_time
        self.last_reading_time = sim_start_time
        self.is_commercial = is_commercial

    # --- REALISTIC CONSUMPTION LOGIC ---
    def get_consumption_profile_multiplier(self, hour: float, day_of_week: int) -> float:
        """Returns a multiplier based on typical human activity patterns."""
        is_weekend = day_of_week >= 5  # Saturday=5, Sunday=6

        if self.is_commercial:  # Business/Factory Profile
            if is_weekend: return random.uniform(0.1, 0.2)
            if 9 <= hour <= 17: return random.uniform(1.2, 1.5)      # High usage during work hours
            elif 7 <= hour < 9 or 17 < hour <= 19: return random.uniform(0.8, 1.0) # Shoulder hours
            else: return random.uniform(0.1, 0.3)                   # Low overnight usage
        else:  # Residential/Home Profile
            if is_weekend:
                if 10 <= hour <= 22: return random.uniform(1.1, 1.4) # Active during the day
                else: return random.uniform(0.4, 0.7)                # Early morning / late night
            else:  # Weekday
                if 7 <= hour <= 9 or 18 <= hour <= 22: return random.uniform(1.3, 1.6) # Morning and evening peaks
                elif 9 < hour < 18: return random.uniform(0.3, 0.5)                   # Low usage when people are at work
                else: return random.uniform(0.2, 0.4)                                  # Overnight

    def read_meter(self) -> Dict[str, Any]:
        current_time = time.time()
        hours_passed = (current_time - self.last_reading_time) * SIMULATION_SPEED_FACTOR / 3600.0
        if hours_passed <= 0: return {'net_energy': 0, 'hour': 0}

        total_sim_seconds = (current_time - self.sim_start_time) * SIMULATION_SPEED_FACTOR
        day_of_week = int((total_sim_seconds / 86400)) % 7
        hour_of_day = (total_sim_seconds / 3600) % 24

        # --- REALISTIC SOLAR PRODUCTION ---
        solar_angle_factor = max(0, -((hour_of_day - 13) ** 2) / 30 + 1)
        weather_factor = random.uniform(0.6, 1.0) # Simulate cloud cover
        production = self.production_capacity * solar_angle_factor * weather_factor * hours_passed

        consumption_multiplier = self.get_consumption_profile_multiplier(hour_of_day, day_of_week)
        consumption = self.base_consumption_rate * consumption_multiplier * random.uniform(0.9, 1.1) * hours_passed

        self.last_reading_time = current_time
        return {'net_energy': production - consumption, 'hour': hour_of_day}

class BaseNode:
    # ... (This class is correct and unchanged) ...
    def __init__(self, node_id: str, blockchain: Blockchain):
        self.node_id, self.blockchain, self.active = node_id, blockchain, True
        logger.info(f"Node {self.node_id} registered.")
    def start(self): self.active = True; logger.info(f"Node {self.node_id} started.")
    def stop(self): self.active = False; logger.info(f"Node {self.node_id} stopped.")
    def process_new_block(self, block: Dict[str, Any]): pass

class GridOperator(BaseNode):
    def __init__(self, node_id: str, blockchain: Blockchain, worker_nodes: List[BaseNode], sim_start_time: float):
        super().__init__(node_id, blockchain)
        self.base_price, self.order_book, self.lock = 0.15, {'offers': [], 'requests': []}, threading.Lock()
        self.update_thread: Optional[threading.Thread] = None
        self.worker_nodes = worker_nodes
        self.sim_start_time = sim_start_time

    def start(self):
        super().start()
        self.update_thread = threading.Thread(target=self._run_simulation_heartbeat)
        self.update_thread.start()
        
    def stop(self):
        super().stop()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)

    def submit_offer(self, offer: Dict[str, Any]):
        with self.lock: self.order_book['offers'].append(offer)

    def submit_request(self, request: Dict[str, Any]):
        with self.lock: self.order_book['requests'].append(request)

    def _run_simulation_heartbeat(self):
        while self.active:
            time.sleep(5)
            self._clear_market()
            
            if self.blockchain.current_transactions:
                new_block = self.blockchain.create_new_block(self.node_id)
                if new_block:
                    for node in [self] + self.worker_nodes:
                        if node.active:
                            node.process_new_block(new_block.to_dict())

    def _clear_market(self):
        # ... (This function is correct and unchanged) ...
        with self.lock:
            if not self.order_book['offers'] or not self.order_book['requests']: return
            
            offers = sorted(self.order_book['offers'], key=lambda o: o['price'])
            requests = sorted(self.order_book['requests'], key=lambda r: r['price'], reverse=True)
            trades_executed = 0
            for req in list(requests):
                for offer in list(offers):
                    if offer['price'] <= req['price']:
                        trade_energy = min(offer['energy'], req['energy'])
                        if trade_energy < 0.01: continue
                        
                        trade_amount = trade_energy * offer['price']
                        self.blockchain.new_transaction(sender=req['sender_id'], recipient=offer['sender_id'], amount=trade_amount, transaction_type='energy_payment')
                        self.blockchain.new_transaction(sender=self.node_id, recipient=req['sender_id'], amount=trade_amount, energy=trade_energy, transaction_type='energy_delivery')
                        
                        trades_executed += 1; offer['energy'] -= trade_energy; req['energy'] -= trade_energy
                        
            if trades_executed > 0: 
                logger.info(f"GridOperator cleared {trades_executed} trades, creating {trades_executed * 2} pending transactions.")

            self.order_book['offers'] = [o for o in offers if o['energy'] > 0.01]
            self.order_book['requests'] = [r for r in requests if r['energy'] > 0.01]

    # --- DYNAMIC MARKET PRICING ---
    def get_market_price(self) -> float:
        with self.lock:
            total_supply = sum(o['energy'] for o in self.order_book['offers'])
            total_demand = sum(r['energy'] for r in self.order_book['requests'])
            demand_pressure_factor = min(2.0, total_demand / (total_supply + 0.1))
            total_sim_seconds = (time.time() - self.sim_start_time) * SIMULATION_SPEED_FACTOR
            hour_of_day = (total_sim_seconds / 3600) % 24
            is_peak_hour = (7 <= hour_of_day <= 9) or (18 <= hour_of_day <= 21)
            peak_price_multiplier = 1.4 if is_peak_hour else 1.0
            return self.base_price * peak_price_multiplier * (1 + demand_pressure_factor * 0.2)

class GridNode(BaseNode):
    def __init__(self, node_id: str, blockchain: Blockchain, is_producer: bool, 
                 prod_cap: float, cons_rate: float, sim_start_time: float, 
                 is_commercial: bool = False):
        super().__init__(node_id, blockchain)
        self.is_producer = is_producer
        self.smart_meter = SmartMeter(node_id, prod_cap, cons_rate, sim_start_time, is_commercial)
        self.energy_storage, self.max_storage, self.storage_efficiency = 0.0, 10.0, 0.95
        self.grid_operator: Optional[GridOperator] = None
        self.update_thread: Optional[threading.Thread] = None
    def start(self):
        if not self.active: return
        super().start()
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.start()
    def stop(self):
        super().stop()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
    def connect_to_grid_operator(self, grid_operator: GridOperator): self.grid_operator = grid_operator
    def _update_loop(self):
        time.sleep(random.uniform(1, 5))
        while self.active: self._process_meter_reading(self.smart_meter.read_meter()); time.sleep(random.uniform(4, 8))
    def _process_meter_reading(self, reading: Dict[str, Any]):
        # ... (This function is now correct) ...
        net_energy = reading.get('net_energy', 0)
        if net_energy > 0:
            to_store = min(net_energy, self.max_storage - self.energy_storage)
            self.energy_storage += to_store * self.storage_efficiency
            remaining_to_sell = net_energy - to_store
            if remaining_to_sell > 0.1: self._offer_excess_energy(remaining_to_sell)
        elif net_energy < 0:
            from_storage = min(self.energy_storage, abs(net_energy))
            self.energy_storage -= from_storage
            remaining_to_buy = abs(net_energy) - from_storage
            if remaining_to_buy > 0.1: self._request_needed_energy(remaining_to_buy)

    def _offer_excess_energy(self, amount: float):
        if self.grid_operator: self.grid_operator.submit_offer({'sender_id': self.node_id, 'energy': amount, 'price': self.grid_operator.get_market_price() * random.uniform(0.95, 1.0)})
    def _request_needed_energy(self, amount: float):
        if self.grid_operator: self.grid_operator.submit_request({'sender_id': self.node_id, 'energy': amount, 'price': self.grid_operator.get_market_price() * random.uniform(1.0, 1.05)})
    def process_new_block(self, block: Dict[str, Any]):
        for tx in block.get('transactions', []):
            if tx['recipient'] == self.node_id:
                if tx['type'] == 'energy_payment': logger.info(f"Node {self.node_id} received payment of ${tx['amount']:.2f}")
                elif tx['type'] == 'energy_delivery': self.energy_storage = min(self.max_storage, self.energy_storage + tx['energy']); logger.info(f"Node {self.node_id} received {tx['energy']:.2f} kWh, storage is now {self.energy_storage:.2f} kWh.")

class Prosumer(GridNode):
    def __init__(self, node_id: str, blockchain: Blockchain, production_capacity: float, consumption_rate: float, sim_start_time: float):
        super().__init__(node_id, blockchain, True, production_capacity, consumption_rate, sim_start_time, is_commercial=False)
        self.max_storage = 15.0; self.energy_storage = random.uniform(5.0, 10.0)
class Consumer(GridNode):
    def __init__(self, node_id: str, blockchain: Blockchain, consumption_rate: float, sim_start_time: float, is_commercial: bool = False):
        super().__init__(node_id, blockchain, False, 0.0, consumption_rate, sim_start_time, is_commercial)
        self.max_storage = 5.0

class TelemetryNode(BaseNode):
    # This class is intentionally simple for this version.
    def __init__(self, node_id: str, blockchain: Blockchain):
        super().__init__(node_id, blockchain)
    def start(self):
        super().start()
    def stop(self):
        super().stop()