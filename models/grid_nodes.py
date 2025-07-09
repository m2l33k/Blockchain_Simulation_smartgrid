import time
import uuid
import random
import logging
from typing import Dict, List, Any, Optional
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('grid_nodes')

class BaseNode:
    def __init__(self, node_id: Optional[str] = None, wallet_balance: float = 100.0):
        self.node_id = node_id if node_id else str(uuid.uuid4())
        self.wallet_balance = wallet_balance
        self.blockchain = None  # Will be set when registered with network
        self.active = False
        self.thread = None
    
    def register_with_blockchain(self, blockchain):
        """Register this node with the blockchain network"""
        self.blockchain = blockchain
        # Register this node with the blockchain so it can receive transaction notifications
        blockchain.register_node(self)
        return self
    
    def start(self):
        """Start the node's activities in a separate thread"""
        if self.thread is None:
            self.active = True
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"{self.__class__.__name__} {self.node_id[:8]} started")
    
    def stop(self):
        """Stop the node's activities"""
        self.active = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
            logger.info(f"{self.__class__.__name__} {self.node_id[:8]} stopped")
    
    def _run(self):
        """Main activity loop - to be implemented by subclasses"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.node_id[:8]})"


class Prosumer(BaseNode):
    """A node that both produces and consumes energy"""
    def __init__(self, node_id=None, wallet_balance=100.0, 
                 production_capacity=10.0, consumption_rate=5.0):
        super().__init__(node_id, wallet_balance)
        self.production_capacity = production_capacity  # kWh per hour
        self.consumption_rate = consumption_rate  # kWh per hour
        self.energy_surplus = 0.0
        self.energy_price = random.uniform(0.10, 0.20)  # $ per kWh
    
    def _run(self):
        """Simulate energy production and consumption"""
        while self.active:
            # Simulate energy production with some randomness (e.g., solar panel output varies)
            production = self.production_capacity * random.uniform(0.5, 1.0)
            
            # Simulate energy consumption with some randomness
            consumption = self.consumption_rate * random.uniform(0.7, 1.2)
            
            # Calculate surplus or deficit
            self.energy_surplus = production - consumption
            
            # If we have surplus energy, try to sell it
            if self.energy_surplus > 0 and self.blockchain:
                self._offer_energy()
                
            # If we have energy deficit, try to buy energy
            elif self.energy_surplus < 0 and self.blockchain:
                self._request_energy()
            
            # Adjust price based on market conditions
            self._adjust_energy_price()
            
            # Wait a bit before next cycle
            time.sleep(random.uniform(5, 15))  # Simulated time
    
    def _offer_energy(self):
        """Offer surplus energy on the market"""
        if not self.blockchain:
            return
            
        # Create a transaction to offer energy
        grid_operator = self._find_grid_operator()
        if grid_operator:
            # Transaction with energy marketplace
            self.blockchain.new_transaction(
                sender=self.node_id,
                recipient=grid_operator,
                amount=self.energy_surplus * self.energy_price,
                energy=self.energy_surplus,
                transaction_type="energy_offer"
            )
            logger.info(f"Prosumer {self.node_id[:8]} offered {self.energy_surplus:.2f} kWh at ${self.energy_price:.2f}/kWh")
    
    def _request_energy(self):
        """Request energy from the market"""
        if not self.blockchain:
            return
            
        # Create a transaction to request energy
        grid_operator = self._find_grid_operator()
        if grid_operator:
            energy_needed = abs(self.energy_surplus)
            cost = energy_needed * self.energy_price * 1.1  # Willing to pay 10% premium
            
            if cost <= self.wallet_balance:
                self.blockchain.new_transaction(
                    sender=self.node_id,
                    recipient=grid_operator,
                    amount=cost,
                    energy=energy_needed,
                    transaction_type="energy_request"
                )
                self.wallet_balance -= cost
                logger.info(f"Prosumer {self.node_id[:8]} requested {energy_needed:.2f} kWh at ${self.energy_price * 1.1:.2f}/kWh")
    
    def _adjust_energy_price(self):
        """Adjust energy price based on surplus/deficit"""
        if self.energy_surplus > 0:
            # Decrease price slightly if we have surplus
            self.energy_price = max(0.08, self.energy_price * random.uniform(0.95, 0.99))
        else:
            # Increase price slightly if we have deficit
            self.energy_price = min(0.30, self.energy_price * random.uniform(1.01, 1.05))
    
    def _find_grid_operator(self):
        """Find a grid operator in the network - simplified version"""
        # In a real implementation, this would look up an actual grid operator
        # For this simulation, we'll assume there's a known grid operator
        return "GRID_OPERATOR"


class Consumer(BaseNode):
    """A node that only consumes energy"""
    def __init__(self, node_id=None, wallet_balance=100.0, consumption_rate=7.0):
        super().__init__(node_id, wallet_balance)
        self.consumption_rate = consumption_rate  # kWh per hour
        self.max_price = random.uniform(0.15, 0.25)  # Maximum price willing to pay per kWh
    
    def _run(self):
        """Simulate energy consumption"""
        while self.active:
            # Simulate energy consumption with some randomness
            consumption = self.consumption_rate * random.uniform(0.8, 1.2)
            
            # Request energy from the grid
            if self.blockchain:
                self._request_energy(consumption)
            
            # Adjust max price based on market and personal factors
            self._adjust_max_price()
            
            # Wait a bit before next cycle
            time.sleep(random.uniform(10, 20))  # Simulated time
    
    def _request_energy(self, amount):
        """Request energy from the market"""
        if not self.blockchain:
            return
            
        grid_operator = self._find_grid_operator()
        if grid_operator:
            # Calculate cost based on current max price
            cost = amount * self.max_price
            
            if cost <= self.wallet_balance:
                self.blockchain.new_transaction(
                    sender=self.node_id,
                    recipient=grid_operator,
                    amount=cost,
                    energy=amount,
                    transaction_type="energy_request"
                )
                self.wallet_balance -= cost
                logger.info(f"Consumer {self.node_id[:8]} requested {amount:.2f} kWh at ${self.max_price:.2f}/kWh")
    
    def _adjust_max_price(self):
        """Adjust the maximum price willing to pay based on various factors"""
        # Simple adjustment based on random market pressure
        market_pressure = random.uniform(-0.02, 0.02)
        self.max_price = max(0.10, min(0.35, self.max_price * (1 + market_pressure)))
    
    def _find_grid_operator(self):
        """Find a grid operator in the network - simplified version"""
        return "GRID_OPERATOR"


class GridOperator(BaseNode):
    """A node that manages the energy grid and facilitates energy trading"""
    def __init__(self, node_id="GRID_OPERATOR", wallet_balance=1000.0):
        super().__init__(node_id, wallet_balance)
        self.energy_offers: List[Dict[str, Any]] = []
        self.energy_requests: List[Dict[str, Any]] = []
        self.base_energy_price = 0.15  # $ per kWh
        self.grid_balance = 0.0  # kWh available in the grid
        self.last_match_attempt = 0  # Track when we last tried to match trades
    
    def _run(self):
        """Simulate grid operations"""
        while self.active:
            # Process pending energy offers and requests
            if self.blockchain:
                # Attempt to match trades more frequently
                current_time = time.time()
                if current_time - self.last_match_attempt >= 1:  # Check every second instead of waiting
                    self._match_energy_trades()
                    self.last_match_attempt = current_time
            
            # Adjust base energy price based on supply and demand
            self._adjust_base_price()
            
            # Create a new block periodically - increased probability
            if self.blockchain and random.random() < 0.5:  # 50% chance each cycle (up from 30%)
                new_block = self.blockchain.create_new_block()
                logger.info(f"Grid Operator created new block: {new_block}")
            
            # Wait a bit before next cycle - shorter wait time
            time.sleep(random.uniform(1, 3))  # Reduced from 3-8 seconds
    
    def process_transaction(self, transaction):
        """Process an energy transaction"""
        if transaction['type'] == 'energy_offer':
            self.energy_offers.append({
                'sender': transaction['sender'],
                'energy': transaction['energy'],
                'price': transaction['amount'] / transaction['energy'] if transaction['energy'] > 0 else 0,
                'timestamp': transaction['timestamp']
            })
            self.grid_balance += transaction['energy']
            logger.info(f"Grid Operator received energy offer: {transaction['energy']:.2f} kWh")
        
        elif transaction['type'] == 'energy_request':
            self.energy_requests.append({
                'sender': transaction['sender'],
                'energy': transaction['energy'],
                'price': transaction['amount'] / transaction['energy'] if transaction['energy'] > 0 else 0,
                'timestamp': transaction['timestamp']
            })
            logger.info(f"Grid Operator received energy request: {transaction['energy']:.2f} kWh")
    
    def _match_energy_trades(self):
        """Match energy offers with energy requests"""
        # Early return if no offers or requests
        if not self.energy_offers or not self.energy_requests:
            return
            
        # Sort offers by price (ascending)
        sorted_offers = sorted(self.energy_offers, key=lambda x: x['price'])
        
        # Sort requests by price (descending)
        sorted_requests = sorted(self.energy_requests, key=lambda x: x['price'], reverse=True)
        
        # Try to match offers with requests - more flexible matching
        matches_made = False
        for request in sorted_requests[:]:
            # For short simulations, be more lenient with matching
            for offer in sorted_offers[:]:
                # Accept partial matches too
                amount_to_trade = min(offer['energy'], request['energy'])
                if amount_to_trade > 0:
                    # Create a copy of the offer/request with adjusted energy
                    modified_offer = offer.copy()
                    modified_request = request.copy()
                    modified_offer['energy'] = amount_to_trade
                    modified_request['energy'] = amount_to_trade
                    
                    # Execute the trade
                    self._execute_trade(modified_offer, modified_request)
                    matches_made = True
                    
                    # Adjust the remaining energy
                    offer['energy'] -= amount_to_trade
                    request['energy'] -= amount_to_trade
                    
                    # Remove if fully satisfied
                    if offer['energy'] <= 0.01:
                        sorted_offers.remove(offer)
                    if request['energy'] <= 0.01:
                        sorted_requests.remove(request)
                        break
        
        # Update our lists
        self.energy_offers = [o for o in sorted_offers if o['energy'] > 0.01]
        self.energy_requests = [r for r in sorted_requests if r['energy'] > 0.01]
        
        # Clean up old offers and requests (older than 5 minutes - reduced from 10)
        current_time = time.time()
        self.energy_offers = [o for o in self.energy_offers if current_time - o['timestamp'] < 300]
        self.energy_requests = [r for r in self.energy_requests if current_time - r['timestamp'] < 300]
        
        # Log status
        if matches_made:
            logger.info(f"Grid balance after matching: {self.grid_balance:.2f} kWh")
    
    def _execute_trade(self, offer, request):
        """Execute an energy trade between a prosumer and consumer"""
        if not self.blockchain:
            return
            
        # Calculate the trade price (average of offer and request)
        trade_price = (offer['price'] + request['price']) / 2
        total_amount = request['energy'] * trade_price
        
        # Create a transaction from grid operator to consumer
        self.blockchain.new_transaction(
            sender=self.node_id,
            recipient=request['sender'],
            amount=0,
            energy=request['energy'],
            transaction_type="energy_delivery"
        )
        
        # Create a transaction from grid operator to prosumer (payment)
        self.blockchain.new_transaction(
            sender=self.node_id,
            recipient=offer['sender'],
            amount=total_amount,
            energy=0,
            transaction_type="energy_payment"
        )
        
        # Update grid balance
        self.grid_balance -= request['energy']
        
        logger.info(f"Grid Operator executed trade: {request['energy']:.2f} kWh at ${trade_price:.2f}/kWh")
    
    def _adjust_base_price(self):
        """Adjust the base energy price based on grid balance"""
        if self.grid_balance > 20:
            # Surplus of energy, decrease price
            self.base_energy_price = max(0.10, self.base_energy_price * 0.98)
        elif self.grid_balance < 5:
            # Shortage of energy, increase price
            self.base_energy_price = min(0.30, self.base_energy_price * 1.02)
        
        # Small random fluctuation
        self.base_energy_price *= random.uniform(0.995, 1.005)