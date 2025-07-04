import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

# --- Simulation Parameters ---
random.seed(42)
np.random.seed(42)
num_nodes = 50
area_size = 100
p = 0.1  # CH election probability
rounds = 50
init_energy = 2  # Joules

# Enhanced Energy Model
tx_energy_per_bit = 50e-9  # 50 nJ/bit
rx_energy_per_bit = 50e-9  # 50 nJ/bit
fs_energy = 10e-12  # Free space model 10 pJ/bit/m^2
mp_energy = 0.0013e-12  # Multi-path model 1.3 pJ/bit/m^4
data_packet_size = 4000  # bits
ctrl_packet_size = 200  # bits (for CH announcements)
idle_energy_per_round = 0.0001  # Joules
radio_speed = 2e5  # 200 m/ms (signal propagation speed)

# Base Station Location
bs_x, bs_y = area_size / 2, 110

class Node:
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = init_energy
        self.is_CH = False
        self.cluster_head = None
        self.alive = True
        self.distance_to_bs = np.hypot(self.x - bs_x, self.y - bs_y)
        self.data_packets_sent = 0
        self.ctrl_packets_sent = 0
        self.packets_received = 0
        self.ch_history = 0  # Times this node has been CH
        self.last_CH_round = -10  # Round when last CH

    def is_alive(self):
        return self.alive and self.energy > 0

    def distance_to(self, other):
        return np.hypot(self.x - other.x, self.y - other.y)

    def consume_tx(self, distance, packet_type='data'):
        """Calculate transmission energy based on packet type and distance"""
        pkt_size = data_packet_size if packet_type == 'data' else ctrl_packet_size
        d_threshold = 75  # meters (free space vs multi-path threshold)
        
        if distance < d_threshold:
            energy = (tx_energy_per_bit + fs_energy * distance**2) * pkt_size
        else:
            energy = (tx_energy_per_bit + mp_energy * distance**4) * pkt_size
            
        self.energy -= energy
        if packet_type == 'data':
            self.data_packets_sent += 1
        else:
            self.ctrl_packets_sent += 1
            
        if self.energy <= 0:
            self.alive = False

    def consume_rx(self, packet_type='data'):
        """Calculate reception energy"""
        pkt_size = data_packet_size if packet_type == 'data' else ctrl_packet_size
        energy = rx_energy_per_bit * pkt_size
        self.energy -= energy
        self.packets_received += 1
        if self.energy <= 0:
            self.alive = False

    def consume_idle(self):
        """Energy consumed while idle"""
        self.energy -= idle_energy_per_round
        if self.energy <= 0:
            self.alive = False

# Initialize nodes and statistics
nodes = [Node(i, random.uniform(0, area_size), random.uniform(0, area_size)) for i in range(num_nodes)]
stats = {
    'alive_nodes': [],
    'total_energy': [],
    'chs_per_round': [],
    'pdr': [],  # Packet Delivery Ratio
    'avg_delay': [],
    'throughput': [],
    'energy_efficiency': [],  # bits/Joule
    'cluster_balance': [],  # Std dev of cluster sizes
    'first_death': None,
    'last_death': None,
    'control_overhead': []
}

for r in range(rounds):
    plt.figure(figsize=(12, 10))
    plt.title(f"LEACH Round {r+1}\nAlive: {sum(n.is_alive() for n in nodes)}/{num_nodes}")
    plt.xlim(0, area_size)
    plt.ylim(0, area_size + 20)
    plt.gca().set_aspect('equal')

    # Reset states and consume idle energy
    alive_nodes = [n for n in nodes if n.is_alive()]
    if not alive_nodes:
        stats['last_death'] = r + 1
        break

    for node in alive_nodes:
        node.is_CH = False
        node.cluster_head = None
        node.consume_idle()

    # Enhanced CH selection with cooldown period
    CHs = []
    for node in alive_nodes:
        # Nodes can't be CH more than once every 1/p rounds
        if (r - node.last_CH_round) < (1/p):
            continue
            
        threshold = p * (node.energy / init_energy) * (1 / (1 + node.ch_history))
        if random.random() < threshold:
            node.is_CH = True
            node.ch_history += 1
            node.last_CH_round = r
            CHs.append(node)

    # Fallback: Select 5% of alive nodes as CH if none elected
    if not CHs:
        num_ch = max(1, int(0.05 * len(alive_nodes)))
        CHs = sorted(alive_nodes, key=lambda n: n.energy, reverse=True)[:num_ch]
        for ch in CHs:
            ch.is_CH = True
            ch.ch_history += 1
            ch.last_CH_round = r

    # Cluster formation
    cluster_members = defaultdict(list)
    total_data_tx = 0
    total_data_rx = 0
    total_ctrl_tx = 0
    delays = []
    
    for node in alive_nodes:
        if not node.is_CH:
            # Find closest CH
            closest_ch = min(CHs, key=lambda ch: node.distance_to(ch))
            dist = node.distance_to(closest_ch)
            
            # Energy for cluster formation (control packets)
            node.consume_tx(dist, 'ctrl')
            closest_ch.consume_rx('ctrl')
            total_ctrl_tx += 1
            
            # Add to cluster
            node.cluster_head = closest_ch
            cluster_members[closest_ch.id].append(node)
            
            # Calculate data transmission delay
            delays.append(dist / radio_speed)

    # Data transmission phase
    for ch in CHs:
        if not ch.is_alive():
            continue
            
        # Receive data from members
        for member in cluster_members.get(ch.id, []):
            if member.is_alive():
                member.consume_tx(member.distance_to(ch), 'data')
                ch.consume_rx('data')
                total_data_tx += 1
                total_data_rx += 1
        
        # Transmit aggregated data to BS
        ch.consume_tx(ch.distance_to_bs, 'data')
        total_data_tx += 1

    # Calculate statistics
    alive_count = sum(n.is_alive() for n in nodes)
    stats['alive_nodes'].append(alive_count)
    stats['total_energy'].append(sum(n.energy for n in nodes if n.is_alive()))
    stats['chs_per_round'].append(len(CHs))
    
    # Network metrics
    pdr = total_data_rx / total_data_tx if total_data_tx > 0 else 0
    stats['pdr'].append(pdr)
    stats['avg_delay'].append(np.mean(delays) if delays else 0)
    stats['throughput'].append(total_data_rx)
    stats['energy_efficiency'].append(total_data_rx * data_packet_size / stats['total_energy'][-1] if stats['total_energy'][-1] > 0 else 0)
    stats['control_overhead'].append(total_ctrl_tx / (total_data_tx + 1e-10))  # Prevent division by zero
    
    # Cluster balance metric
    if CHs:
        cluster_sizes = [len(cluster_members[ch.id]) for ch in CHs]
        stats['cluster_balance'].append(np.std(cluster_sizes))
    else:
        stats['cluster_balance'].append(0)
    
    # Track first and last deaths
    if alive_count < num_nodes and stats['first_death'] is None:
        stats['first_death'] = r + 1

    # Visualization
    colors = plt.cm.tab20.colors
    for i, ch in enumerate(CHs):
        if not ch.is_alive():
            continue
            
        color = colors[i % len(colors)]
        plt.plot(ch.x, ch.y, 'o', markersize=12, markeredgecolor='k',
                color=color, label=f'CH {i+1} (E={ch.energy:.2f}J)')
        
        # Draw connection to BS
        plt.plot([ch.x, bs_x], [ch.y, bs_y], '--', color=color, alpha=0.7)
        
        # Draw connections to members
        for member in cluster_members.get(ch.id, []):
            if member.is_alive():
                plt.plot([member.x, ch.x], [member.y, ch.y], ':', color=color, alpha=0.5)
                plt.plot(member.x, member.y, 'o', color=color, markersize=8, alpha=0.7)

    # Plot dead nodes
    dead_x = [n.x for n in nodes if not n.is_alive()]
    dead_y = [n.y for n in nodes if not n.is_alive()]
    if dead_x:
        plt.plot(dead_x, dead_y, 'kx', markersize=10, label='Dead nodes')

    # Plot base station
    plt.plot(bs_x, bs_y, 'ks', markersize=16, label='Base Station')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Comprehensive Statistics Plotting
plt.figure(figsize=(18, 12))

metrics = [
    ('alive_nodes', 'b', 'Alive Nodes', 'Count'),
    ('total_energy', 'r', 'Total Energy', 'Joules'),
    ('chs_per_round', 'g', 'Cluster Heads', 'Count'),
    ('pdr', 'm', 'Packet Delivery Ratio', 'Ratio'),
    ('avg_delay', 'c', 'Average Delay', 'ms'),
    ('throughput', 'y', 'Throughput', 'Packets'),
    ('energy_efficiency', 'k', 'Energy Efficiency', 'bits/Joule'),
    ('cluster_balance', 'purple', 'Cluster Balance', 'Std Dev'),
    ('control_overhead', 'orange', 'Control Overhead', 'Ratio')
]

for i, (metric, color, title, ylabel) in enumerate(metrics[:6]):
    plt.subplot(2, 3, i+1)
    plt.plot(stats[metric], f'{color}-o')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.grid(True)

plt.tight_layout()
plt.show()

# Print comprehensive summary
print("\n=== Enhanced LEACH Simulation Results ===")
print(f"Simulation Rounds: {len(stats['alive_nodes'])}")
print(f"First Node Death: Round {stats['first_death'] or 'N/A'}")
print(f"Last Node Death: Round {stats['last_death'] or 'N/A'}")
print(f"\nNetwork Lifetime Metrics:")
print(f"Average PDR: {np.mean(stats['pdr']):.4f}")
print(f"Average Delay: {np.mean(stats['avg_delay']):.6f} ms")
print(f"Average Throughput: {np.mean(stats['throughput']):.2f} packets/round")
print(f"\nEnergy Metrics:")
print(f"Initial Total Energy: {num_nodes * init_energy:.1f} J")
print(f"Final Total Energy: {stats['total_energy'][-1]:.4f} J")
print(f"Energy Efficiency: {np.mean(stats['energy_efficiency']):.2f} bits/Joule")
print(f"\nCluster Metrics:")
print(f"Average CHs per Round: {np.mean(stats['chs_per_round']):.2f}")
print(f"Average Cluster Balance: {np.mean(stats['cluster_balance']):.2f} std dev")
print(f"Control Overhead: {np.mean(stats['control_overhead']):.4f} ctrl/data ratio")
