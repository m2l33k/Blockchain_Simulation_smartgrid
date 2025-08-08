import time
import random
import logging
from typing import Dict, Any, Optional, Tuple

class Blockchain: pass

logger = logging.getLogger("simulation_logger")

class SmartMeter:
    def __init__(self, owner_id: str, prod_cap: float, cons_rate: float):
        self.owner_id = owner_id
        self.production_capacity = prod_cap
        self.consumption_rate = cons_rate

    def read_meter(self, hour_of_day: float, time_step_hours: float) -> float:
        solar_factor = max(0, 1 - abs(hour_of_day - 14) / 7) * random.uniform(0.8, 1.0)
        production = self.production_capacity * solar_factor * time_step_hours
        
        peak_factor = 0.5 + 0.5 * max(max(0, 1 - abs(hour_of_day - 8)/3), max(0, 1 - abs(hour_of_day - 19)/4))
        consumption = self.consumption_rate * peak_factor * random.uniform(0.9, 1.1) * time_step_hours
        
        return production - consumption

class BaseNode:
    def __init__(self, node_id: str, blockchain: Blockchain):
        self.node_id = node_id
        self.blockchain = blockchain
        logger.debug(f"Node {node_id} component initialized.")

    def process_transaction(self, tx: Dict[str, Any]):
        pass

class GridOperator(BaseNode):
    def __init__(self, node_id: str, blockchain: Blockchain):
        super().__init__(node_id, blockchain)
        self.base_price = 0.15
        self.order_book = {'offers': [], 'requests': []}

    def submit_offer(self, offer: Dict[str, Any]): self.order_book['offers'].append(offer)
    def submit_request(self, request: Dict[str, Any]): self.order_book['requests'].append(request)

    def get_market_price(self, hour_of_day: float) -> float:
        total_offered = sum(o['energy'] for o in self.order_book['offers']) + 1e-6
        total_requested = sum(r['energy'] for r in self.order_book['requests']) + 1e-6
        supply_demand_ratio = total_offered / total_requested
        price_multiplier = max(0.5, min(2.0, 1.0 / supply_demand_ratio))
        peak_hour_multiplier = 1.0 + 0.75 * (max(0, 1 - abs(hour_of_day - 8)/3) + max(0, 1 - abs(hour_of_day - 19)/4))
        return self.base_price * price_multiplier * peak_hour_multiplier

    def clear_market(self) -> Tuple[float, float]:
        if not self.order_book['offers'] or not self.order_book['requests']: return 0.0, 0.0
        
        offers = sorted(self.order_book['offers'], key=lambda o: o['price'])
        requests = sorted(self.order_book['requests'], key=lambda r: r['price'], reverse=True)
        total_energy_tick, total_value_tick = 0.0, 0.0
        
        for req in list(requests):
            for offer in list(offers):
                if offer['price'] <= req['price']:
                    trade_energy = min(offer['energy'], req['energy'])
                    if trade_energy < 0.01: continue
                    
                    trade_amount = trade_energy * offer['price']
                    self.blockchain.new_transaction(req['sender_id'], offer['sender_id'], trade_amount, 0, 'energy_payment')
                    self.blockchain.new_transaction(self.node_id, req['sender_id'], 0, trade_energy, 'energy_delivery')
                    
                    total_energy_tick += trade_energy
                    total_value_tick += trade_amount
                    offer['energy'] -= trade_energy
                    req['energy'] -= trade_energy
        
        avg_price = total_value_tick / total_energy_tick if total_energy_tick > 0 else self.get_market_price(12)
        self.order_book['offers'] = [o for o in offers if o['energy'] > 0.01]
        self.order_book['requests'] = [r for r in requests if r['energy'] > 0.01]
        return total_energy_tick, avg_price

class GridNode(BaseNode):
    def __init__(self, node_id: str, blockchain: Blockchain, prod_cap: float, cons_rate: float):
        super().__init__(node_id, blockchain)
        self.smart_meter = SmartMeter(node_id, prod_cap, cons_rate)
        self.energy_storage = 0.0
        self.max_storage = 10.0
        self.grid_operator: Optional[GridOperator] = None

    def connect_to_grid_operator(self, grid_operator: GridOperator): self.grid_operator = grid_operator

    def update(self, hour_of_day: float, time_step_hours: float):
        net_energy = self.smart_meter.read_meter(hour_of_day, time_step_hours)
        if net_energy > 0:
            to_store = min(net_energy, self.max_storage - self.energy_storage)
            self.energy_storage += to_store
            if (remaining := net_energy - to_store) > 0.01: self._offer_energy(remaining, hour_of_day)
        elif net_energy < 0:
            from_storage = min(self.energy_storage, abs(net_energy))
            self.energy_storage -= from_storage
            if (remaining := abs(net_energy) - from_storage) > 0.01: self._request_energy(remaining, hour_of_day)

    def _offer_energy(self, amount: float, hour_of_day: float):
        urgency_discount = 0.8 if self.energy_storage / self.max_storage > 0.9 else 0.98
        offer_price = self.grid_operator.get_market_price(hour_of_day) * urgency_discount
        self.grid_operator.submit_offer({'sender_id': self.node_id, 'energy': amount, 'price': offer_price})

    def _request_energy(self, amount: float, hour_of_day: float):
        urgency_premium = 1.25 if self.energy_storage / self.max_storage < 0.1 else 1.05
        request_price = self.grid_operator.get_market_price(hour_of_day) * urgency_premium
        self.grid_operator.submit_request({'sender_id': self.node_id, 'energy': amount, 'price': request_price})

    def process_transaction(self, tx: Dict[str, Any]):
        if tx['recipient'] == self.node_id and tx['type'] == 'energy_delivery':
            self.energy_storage = min(self.max_storage, self.energy_storage + tx['energy'])

class Prosumer(GridNode):
    def __init__(self, node_id: str, blockchain: Blockchain, production_capacity: float, consumption_rate: float):
        super().__init__(node_id, blockchain, production_capacity, consumption_rate)
        self.max_storage = 15.0; self.energy_storage = random.uniform(5.0, 10.0)

class Consumer(GridNode):
    def __init__(self, node_id: str, blockchain: Blockchain, consumption_rate: float):
        super().__init__(node_id, blockchain, 0.0, consumption_rate)
        self.max_storage = 5.0