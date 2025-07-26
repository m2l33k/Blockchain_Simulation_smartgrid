# anomaly_injector.py (Corrected for New Architecture and Latency Recording)

import random
import time
import logging
import threading
from typing import List, Dict, Callable, Set

from models.grid_nodes import BaseNode, GridNode, GridOperator
from models.blockchain import Blockchain

# --- ADD THIS IMPORT ---
# Import the latency recording utility
from utils.latency_recorder import record_latency_event

logger = logging.getLogger(__name__)

class AnomalyInjector:
    def __init__(self, all_nodes: List[BaseNode], grid_operator: GridOperator, blockchain: Blockchain):
        self.grid_nodes: List[GridNode] = [node for node in all_nodes if isinstance(node, GridNode)]
        self.grid_operator: GridOperator = grid_operator
        self.blockchain: Blockchain = blockchain
        self.active: bool = False
        self.thread: threading.Thread = None

        # --- STATE TRACKING ---
        self.nodes_under_anomaly: Set[str] = set()
        
        # --- ANOMALY DEFINITIONS ---
        self.anomaly_methods: Dict[str, Callable] = {
            'multi_stage_dos': self.inject_multi_stage_dos,
            'energy_theft': self.inject_energy_theft,
            'persistent_meter_tampering': self.inject_persistent_meter_tampering,
            'coordinated_trading': self.inject_coordinated_inauthentic_trading,
            'node_breakdown': self.inject_node_breakdown,
        }
        self.anomaly_weights: Dict[str, float] = {
            'multi_stage_dos': 0.25,
            'energy_theft': 0.20,
            'persistent_meter_tampering': 0.30,
            'coordinated_trading': 0.15,
            'node_breakdown': 0.10,
        }

    def start(self, interval_ms: int = 10000, injection_probability: float = 0.25):
        if not self.grid_nodes:
            logger.warning("AnomalyInjector: No GridNodes to target. Stopping.")
            return
        self.active = True
        self.thread = threading.Thread(target=self._run_loop, args=(interval_ms, injection_probability), daemon=True)
        self.thread.start()
        logger.info("Enhanced Anomaly Injector started.")

    def stop(self):
        self.active = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Anomaly Injector stopped.")

    def _run_loop(self, interval_ms: int, injection_probability: float):
        interval_seconds = interval_ms / 1000.0
        while self.active:
            time.sleep(interval_seconds)
            if not self.active: break
            
            if random.random() < injection_probability:
                chosen_anomaly = random.choices(list(self.anomaly_methods.keys()), weights=list(self.anomaly_weights.values()), k=1)[0]
                try:
                    self.anomaly_methods[chosen_anomaly]()
                except Exception as e:
                    logger.error(f"Error injecting anomaly '{chosen_anomaly}': {e}", exc_info=True)

    def get_random_nodes(self, count: int, exclude_nodes: List[GridNode] = None) -> List[GridNode]:
        exclude_ids = set(n.node_id for n in exclude_nodes) if exclude_nodes else set()
        exclude_ids.update(self.nodes_under_anomaly) 
        
        possible_targets = [n for n in self.grid_nodes if n.active and n.node_id not in exclude_ids]
        
        if len(possible_targets) < count:
            return [] 
        
        return random.sample(possible_targets, k=count)

    def inject_node_breakdown(self):
        nodes_to_affect = self.get_random_nodes(1)
        if not nodes_to_affect: return
        target_node = nodes_to_affect[0]
        
        target_node.active = False
        self.nodes_under_anomaly.add(target_node.node_id)
        
        log_message = f"Injecting NODE BREAKDOWN on {target_node.node_id}. Node is now permanently offline."
        logger.warning(f"!!! ANOMALY: {log_message}")
        
        # --- ADD LATENCY RECORDING ---
        record_latency_event('injection', f"Breakdown on {target_node.node_id}")
        
        print(f"\n🚨 ALERT: Node {target_node.node_id} has gone offline.\n")

    def inject_energy_theft(self):
        nodes = self.get_random_nodes(2)
        if not nodes: return
        malicious_node, victim_node = nodes[0], nodes[1]
        
        theft_amount = random.uniform(15, 30)
        
        log_message = f"{malicious_node.node_id} attempting THEFT of ${theft_amount:.2f} from {victim_node.node_id}"
        logger.warning(f"!!! ANOMALY: {log_message} !!!")
        
        # --- ADD LATENCY RECORDING ---
        record_latency_event('injection', f"Theft from {victim_node.node_id} by {malicious_node.node_id}")
        
        self.blockchain.new_transaction(
            sender=victim_node.node_id,
            recipient=malicious_node.node_id,
            amount=theft_amount,
            energy=0,
            transaction_type="fraudulent_payment"
        )
        print(f"\n🚨 ALERT: Fraudulent high-value transaction submitted by {malicious_node.node_id}.\n")

    def inject_persistent_meter_tampering(self):
        nodes_to_affect = self.get_random_nodes(1)
        if not nodes_to_affect: return
        target_node = nodes_to_affect[0]

        log_message = f"Injecting PERSISTENT METER TAMPERING on {target_node.node_id}"
        logger.warning(f"!!! ANOMALY: {log_message} !!!")
        
        # --- ADD LATENCY RECORDING ---
        record_latency_event('injection', f"Tampering on {target_node.node_id}")
        
        self.nodes_under_anomaly.add(target_node.node_id)
        
        tamper_thread = threading.Thread(target=self._meter_tamper_worker, args=(target_node,), daemon=True)
        tamper_thread.start()

    def _meter_tamper_worker(self, target_node: GridNode):
        meter = target_node.smart_meter
        original_read_method = meter.read_meter
        multiplier = random.choice([0.2, 1.8])

        def tampered_read_meter(*args, **kwargs):
            reading = original_read_method(*args, **kwargs)
            reading['net_energy'] *= multiplier
            return reading
        
        meter.read_meter = tampered_read_meter
        print(f"\n🚨 ALERT: Meter for {target_node.node_id} shows signs of tampering.\n")
        
        tamper_duration = random.uniform(30, 90)
        time.sleep(tamper_duration)
        
        meter.read_meter = original_read_method
        self.nodes_under_anomaly.remove(target_node.node_id)
        logger.warning(f"--- ANOMALY END: Meter tampering for {target_node.node_id} has ended. ---")

    def inject_multi_stage_dos(self):
        nodes_to_affect = self.get_random_nodes(1)
        if not nodes_to_affect: return
        attacker = nodes_to_affect[0]

        log_message = f"{attacker.node_id} beginning a MULTI-STAGE DoS ATTACK"
        logger.warning(f"!!! ANOMALY: {log_message} !!!")
        
        # --- ADD LATENCY RECORDING ---
        record_latency_event('injection', f"DoS from {attacker.node_id}")
        
        self.nodes_under_anomaly.add(attacker.node_id)
        
        dos_thread = threading.Thread(target=self._dos_worker, args=(attacker,), daemon=True)
        dos_thread.start()

    def _dos_worker(self, attacker: GridNode):
        print(f"\n📈 RAMP-UP: Suspiciously high traffic from {attacker.node_id}.\n")
        for _ in range(random.randint(20, 40)):
            if not self.active: break
            self.blockchain.new_transaction(attacker.node_id, "dos_target", 0.01, 0.01, "spam_ramp_up")
            time.sleep(random.uniform(0.05, 0.1))

        print(f"\n🚨 ALERT: Full DoS spam attack initiated by {attacker.node_id}.\n")
        for _ in range(random.randint(100, 200)):
            if not self.active: break
            self.blockchain.new_transaction(attacker.node_id, "dos_target", 0.001, 0.001, "spam_peak")
            time.sleep(0.005)

        time.sleep(random.uniform(5, 10))
        self.nodes_under_anomaly.remove(attacker.node_id)
        logger.warning(f"--- ANOMALY END: DoS attack from {attacker.node_id} has ceased. ---")

    def inject_coordinated_inauthentic_trading(self):
        colluders = self.get_random_nodes(2)
        if not colluders: return
        
        node_a, node_b = colluders[0], colluders[1]
        
        log_message = f"Injecting COORDINATED INAUTHENTIC TRADING between {node_a.node_id} and {node_b.node_id}"
        logger.warning(f"!!! ANOMALY: {log_message} !!!")
        
        # --- ADD LATENCY RECORDING ---
        record_latency_event('injection', f"Coordinated trading between {node_a.node_id} and {node_b.node_id}")
        
        print(f"\n🎭 ALERT: Unusual coordinated trading activity detected between nodes.\n")
        
        self.nodes_under_anomaly.add(node_a.node_id)
        self.nodes_under_anomaly.add(node_b.node_id)
        
        trade_thread = threading.Thread(target=self._wash_trading_worker, args=(node_a, node_b), daemon=True)
        trade_thread.start()
        
    def _wash_trading_worker(self, node_a: GridNode, node_b: GridNode):
        for _ in range(random.randint(5, 15)):
            if not self.active: break
            
            trade_amount_ab = random.uniform(1, 5)
            self.blockchain.new_transaction(node_a.node_id, node_b.node_id, trade_amount_ab, 0, "wash_trade_payment")
            time.sleep(random.uniform(0.1, 0.5))
            
            trade_amount_ba = random.uniform(1, 5)
            self.blockchain.new_transaction(node_b.node_id, node_a.node_id, trade_amount_ba, 0, "wash_trade_payment")
            time.sleep(random.uniform(0.1, 0.5))
            
        self.nodes_under_anomaly.remove(node_a.node_id)
        self.nodes_under_anomaly.remove(node_b.node_id)
        logger.warning(f"--- ANOMALY END: Coordinated trading between {node_a.node_id} and {node_b.node_id} has ended. ---")