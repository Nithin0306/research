import random
import numpy as np
from collections import defaultdict

# --- Simulation Parameters ---
random.seed(42)
np.random.seed(42)
num_nodes = 50
area_size = 100
rounds = 50
init_energy = 2

# Energy + Timing Model
tx_energy_per_bit = 50e-9
rx_energy_per_bit = 50e-9
fs_energy = 10e-12
mp_energy = 0.0013e-12
packet_size = 4000  # bits
idle_energy = 0.0001
time_per_bit = 1e-6  # 1 µs = 1e-6 sec
deadline = 0.1  # Real-time deadline (100 ms)

# Base station location
bs_x, bs_y = area_size / 2, 110

class Node:
    def _init_(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = init_energy
        self.is_CH = False
        self.cluster_head = None
        self.prev_ch_id = None
        self.alive = True
        self.distance_to_bs = np.hypot(self.x - bs_x, self.y - bs_y)

    def is_alive(self):
        return self.alive and self.energy > 0

    def distance_to(self, other):
        return np.hypot(self.x - other.x, self.y - other.y)

    def consume_tx(self, dist):
        d_threshold = 75
        if dist < d_threshold:
            energy = tx_energy_per_bit * packet_size + fs_energy * packet_size * dist ** 2
        else:
            energy = tx_energy_per_bit * packet_size + mp_energy * packet_size * dist ** 4
        self.energy -= energy
        if self.energy <= 0:
            self.alive = False
        return packet_size * time_per_bit  # delay

    def consume_rx(self):
        energy = rx_energy_per_bit * packet_size
        self.energy -= energy
        if self.energy <= 0:
            self.alive = False
        return packet_size * time_per_bit  # delay

    def consume_idle(self):
        self.energy -= idle_energy
        if self.energy <= 0:
            self.alive = False

# Initialize nodes
nodes = [Node(i, random.uniform(0, area_size), random.uniform(0, area_size)) for i in range(num_nodes)]

# Initialize stats
stats = {
    'alive_nodes': [],
    'total_energy': [],
    'chs_per_round': [],
    'delivered_packets': [],
    'dropped_packets': [],
    'cluster_switches': [],
    'load_distribution': [],
    'round_delay': [],
    'met_deadline_packets': [],
    'missed_deadline_packets': [],
    'real_time_ratio': [],
    'pdr_over_time': [],
    'first_death': None,
    'last_death': None
}

# Simulation loop
for r in range(rounds):
    alive_nodes = [n for n in nodes if n.is_alive()]
    if not alive_nodes:
        stats['last_death'] = r + 1
        break

    # Idle energy drain
    for n in alive_nodes:
        n.is_CH = False
        n.cluster_head = None
        n.consume_idle()

    # CH election
    CHs = []
    for node in alive_nodes:
        ch_prob = min(1.0, 0.1 * (node.energy / init_energy))
        if random.random() < ch_prob:
            node.is_CH = True
            CHs.append(node)

    if not CHs:
        CHs = sorted(alive_nodes, key=lambda n: n.energy, reverse=True)[:1]
        for ch in CHs:
            ch.is_CH = True

    cluster_members = defaultdict(list)
    delivered, dropped, switches = 0, 0, 0
    met_deadline, missed_deadline = 0, 0
    load_map = defaultdict(int)
    round_delay = 0.0

    for node in alive_nodes:
        if not node.is_CH:
            closest_ch = min(CHs, key=lambda ch: node.distance_to(ch))
            dist = node.distance_to(closest_ch)
            node.cluster_head = closest_ch

            # CH switching
            if node.prev_ch_id is not None and node.prev_ch_id != closest_ch.id:
                switches += 1
            node.prev_ch_id = closest_ch.id

            # Transmission delay and check for deadline
            tx1 = node.consume_tx(dist)
            if node.is_alive():
                rx = closest_ch.consume_rx()
                if closest_ch.is_alive():
                    cluster_members[closest_ch.id].append(node)
                    tx2 = closest_ch.consume_tx(closest_ch.distance_to_bs)
                    delay = tx1 + rx + tx2
                    delivered += 1
                    round_delay += delay
                    if delay <= deadline:
                        met_deadline += 1
                    else:
                        missed_deadline += 1
                else:
                    dropped += 1
            else:
                dropped += 1

    for ch_id, members in cluster_members.items():
        load_map[ch_id] = len(members)

    # Record statistics
    stats['alive_nodes'].append(sum(n.is_alive() for n in nodes))
    stats['total_energy'].append(sum(n.energy for n in nodes if n.is_alive()))
    stats['chs_per_round'].append(len(CHs))
    stats['delivered_packets'].append(delivered)
    stats['dropped_packets'].append(dropped)
    stats['cluster_switches'].append(switches)
    stats['load_distribution'].append(np.std(list(load_map.values())) if load_map else 0)
    stats['round_delay'].append(round_delay)
    stats['met_deadline_packets'].append(met_deadline)
    stats['missed_deadline_packets'].append(missed_deadline)

    if delivered > 0:
        stats['real_time_ratio'].append(met_deadline / delivered)
    else:
        stats['real_time_ratio'].append(0)

    total_tx = delivered + dropped
    stats['pdr_over_time'].append(delivered / total_tx if total_tx > 0 else 0)

    if stats['alive_nodes'][-1] < num_nodes and stats['first_death'] is None:
        stats['first_death'] = r + 1
    if stats['alive_nodes'][-1] == 0 and stats['last_death'] is None:
        stats['last_death'] = r + 1

# --- Summary Printout ---
print("\n=== Time-Based HEED with Real-Time Responsiveness ===")
print(f"Rounds simulated: {len(stats['alive_nodes'])}")
print(f"First node death at round: {stats['first_death'] or 'N/A'}")
print(f"Last node death at round: {stats['last_death'] or 'N/A'}")
print(f"Final alive nodes: {stats['alive_nodes'][-1]}")
print(f"Average CHs per round: {np.mean(stats['chs_per_round']):.2f}")
print(f"Average Delivered Packets: {np.mean(stats['delivered_packets']):.2f}")
print(f"Average Dropped Packets: {np.mean(stats['dropped_packets']):.2f}")
print(f"Average PDR: {np.mean(stats['pdr_over_time']):.4f}")
print(f"Average Cluster Switches: {np.mean(stats['cluster_switches']):.2f}")
print(f"Average Load Std Dev: {np.mean(stats['load_distribution']):.4f}")
print(f"Average Delay per Round (s): {np.mean(stats['round_delay']):.6f}")
print(f"Average Real-Time Deadline Ratio: {np.mean(stats['real_time_ratio']):.4f}")
