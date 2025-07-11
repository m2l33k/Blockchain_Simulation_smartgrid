# anomaly_injector.py

import random
import time
import logging
import threading
from typing import List

# Import the node classes for type hinting and checking
from models.grid_nodes import BaseNode, GridNode, Prosumer, Consumer, GridOperator

logger = logging.getLogger(__name__)

class AnomalyInjector:
    def __init__(self, all_nodes: List[BaseNode], grid_operator: GridOperator):
        self.all_nodes = [node for node in all_nodes if isinstance(node, GridNode)]
        self.grid_operator = grid_operator
        self.active = False
        self.thread: threading.Thread = None
        
        self.anomalies = [
            self.inject_node_breakdown,
            self.inject_energy_theft,
            self.inject_meter_tampering,
            self.inject_dos_attack,
        ]

    def start(self, interval_seconds: int = 10):
        if not self.all_nodes:
            logger.warning("AnomalyInjector: No nodes to inject anomalies into. Stopping.")
            return

        self.active = True
        self.thread = threading.Thread(target=self._run_loop, args=(interval_seconds,), daemon=True)
        self.thread.start()
        logger.info("Anomaly Injector started.")

    def stop(self):
        self.active = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Anomaly Injector stopped.")

    def _run_loop(self, interval: int):
        while self.active:
            sleep_time = random.uniform(interval * 0.8, interval * 1.2)
            time.sleep(sleep_time)
            
            if not self.active: break

            chosen_anomaly = random.choice(self.anomalies)
            chosen_anomaly()

    def inject_node_breakdown(self):
        target_node = random.choice(self.all_nodes)
        if not target_node.active: return

        logger.warning(f"!!! ANOMALY: Injecting NODE BREAKDOWN on {target_node.node_id} !!!")
        alert_message = f"ALERT: Node {target_node.node_id} has become unresponsive and is offline."
        
        target_node.stop()
        
        self.grid_operator.create_transaction(
            "GRID_SECURITY", 0, 0, f"alert_node_offline:{target_node.node_id}"
        )
        print(f"\n🚨 {alert_message}\n")

    def inject_energy_theft(self):
        malicious_node = random.choice(self.all_nodes)
        # To avoid errors, ensure there's at least one other node to be a victim
        possible_victims = [n for n in self.all_nodes if n != malicious_node]
        if not possible_victims: return
        victim_node = random.choice(possible_victims)
        
        theft_amount = random.uniform(10, 20)
        
        logger.warning(f"!!! ANOMALY: {malicious_node.node_id} attempting ENERGY THEFT from {victim_node.node_id} !!!")
        alert_message = f"ALERT: Node {malicious_node.node_id} attempted a fraudulent transaction."
        
        if victim_node.blockchain:
            victim_node.blockchain.new_transaction(
                sender=victim_node.node_id,
                recipient=malicious_node.node_id,
                amount=theft_amount,
                energy=0,
                transaction_type="fraudulent_payment_attempt"
            )
        print(f"\n🚨 {alert_message}\n")

    def inject_meter_tampering(self):
        """
        Anomaly: A node's meter starts under-reporting consumption or over-reporting production.
        """
        target_node = random.choice(self.all_nodes)
        
        # Avoid re-tampering an already tampered meter
        if hasattr(target_node.smart_meter, 'is_tampered'):
            return

        logger.warning(f"!!! ANOMALY: Injecting METER TAMPERING on {target_node.node_id} !!!")
        alert_message = f"ALERT: Suspicious readings detected from meter of {target_node.node_id}. Possible tampering."
        
        # Get the actual SmartMeter instance we are going to tamper with
        meter_to_tamper = target_node.smart_meter
        # Save a reference to the original, untampered method
        original_read_meter = meter_to_tamper.read_meter
        
        def tampered_read_meter(*args, **kwargs):
            """This function will replace the original read_meter method."""
            # Call the original method to get the true reading
            true_reading = original_read_meter(*args, **kwargs)
            true_net_energy = true_reading['net_energy']

            # Simulate tampering
            if true_net_energy < 0:
                # If the node is consuming, report that it's consuming LESS
                # A smaller negative number (closer to zero) means less consumption
                tampered_net_energy = true_net_energy * random.uniform(0.1, 0.5) # Report 10-50% of consumption
            else:
                # If the node is producing, report that it's producing MORE
                tampered_net_energy = true_net_energy * random.uniform(1.2, 1.5) # Report 20-50% more production
            
            logger.debug(f"TAMPERING {target_node.node_id}: True Net: {true_net_energy:.2f}, Reported Net: {tampered_net_energy:.2f}")
            
            # Return the faked reading
            true_reading['net_energy'] = tampered_net_energy
            return true_reading
            
        # Replace the meter's method with our tampered version
        meter_to_tamper.read_meter = tampered_read_meter
        # Add a flag so we know this meter has been tampered with
        meter_to_tamper.is_tampered = True
        
        print(f"\n🚨 {alert_message}\n")

    def inject_dos_attack(self):
        """
        Anomaly: A node spams the grid operator with tiny, pointless requests.
        """
        attacking_node = random.choice(self.all_nodes)
        
        logger.warning(f"!!! ANOMALY: {attacking_node.node_id} launching DoS attack on Grid Operator !!!")
        alert_message = f"ALERT: Grid Operator is being spammed with requests from {attacking_node.node_id}."

        for i in range(50):
            # Use the direct submission method on the operator
            if attacking_node.grid_operator:
                 attacking_node.grid_operator.submit_request({
                     'sender_id': attacking_node.node_id, 
                     'energy': 0.01, 
                     'price': 999 # A high price to ensure it doesn't get matched
                 })
            time.sleep(0.02)
        
        print(f"\n🚨 {alert_message}\n")