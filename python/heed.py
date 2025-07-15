#!/usr/bin/env python3

# Python-only HEED algorithm with dataset analysis

import sys
import os
import random
import math
from collections import defaultdict
from datetime import datetime
import csv

# --- Simulation Parameters ---
RANDOM_SEED = 42
AREA_SIZE = 100
INIT_ENERGY = 2.0

# Energy + Timing Model
TX_ENERGY_PER_BIT = 50e-9
RX_ENERGY_PER_BIT = 50e-9
FS_ENERGY = 10e-12
MP_ENERGY = 0.0013e-12
PACKET_SIZE = 4000  # bits
IDLE_ENERGY = 0.0001
TIME_PER_BIT = 1e-6  # 1 µs = 1e-6 sec
DEADLINE = 0.1  # Real-time deadline (100 ms)

# Base station location
BS_X, BS_Y = AREA_SIZE / 2, 110

# Default dataset path (modify this to your actual dataset path)
DEFAULT_DATASET = "sensor_data.csv"

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.sensor_data = {}

    def load_dataset(self):
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

            with open(self.file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        dt_str = f"{row['date_d_m_y']} {row['time']}"
                        dt = datetime.strptime(dt_str, '%d/%m/%Y %H:%M:%S')
                        record = {
                            'sort_id': int(row['sort_id']),
                            'datetime': dt,
                            'sensor_id': int(row['sensor_id']),
                            'sensor_type': row['sensor_type'],
                            'temp_C': float(row['temp_C']),
                            'hpa_div_4': float(row['hpa_div_4']),
                            'batterylevel': float(row['batterylevel']),
                            'sensor_cycle': int(row['sensor_cycle'])
                        }
                        self.data.append(record)
                        self.sensor_data.setdefault(record['sensor_id'], []).append(record)
                    except (ValueError, KeyError):
                        continue

            self.data.sort(key=lambda x: x['datetime'])
            for sensor_id in self.sensor_data:
                self.sensor_data[sensor_id].sort(key=lambda x: x['datetime'])
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def get_time_windows(self, window_size_minutes=60):
        if not self.data:
            return []

        time_windows = []
        current_window = []
        window_start = self.data[0]['datetime']

        for record in self.data:
            time_diff = (record['datetime'] - window_start).total_seconds() / 60
            if time_diff <= window_size_minutes:
                current_window.append(record)
            else:
                if current_window:
                    time_windows.append(current_window)
                current_window = [record]
                window_start = record['datetime']

        if current_window:
            time_windows.append(current_window)

        return time_windows

class Node:
    def __init__(self, node_id, x, y, initial_battery=100):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = INIT_ENERGY
        self.battery_level = initial_battery
        self.is_CH = False
        self.cluster_head = None
        self.prev_ch_id = None
        self.alive = True
        self.distance_to_bs = math.sqrt((self.x - BS_X)**2 + (self.y - BS_Y)**2)

    def is_alive(self):
        return self.alive and self.energy > 0 and self.battery_level > 0

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def update_battery(self, new_level):
        self.battery_level = max(0, new_level)
        self.energy = INIT_ENERGY * (self.battery_level / 100.0)
        if self.battery_level <= 0:
            self.alive = False

    def consume_tx(self, dist):
        d_threshold = 75
        if dist < d_threshold:
            energy = TX_ENERGY_PER_BIT * PACKET_SIZE + FS_ENERGY * PACKET_SIZE * dist ** 2
        else:
            energy = TX_ENERGY_PER_BIT * PACKET_SIZE + MP_ENERGY * PACKET_SIZE * dist ** 4
        self.energy -= energy
        self.battery_level -= (energy / INIT_ENERGY) * 5
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False
        return PACKET_SIZE * TIME_PER_BIT

    def consume_rx(self):
        energy = RX_ENERGY_PER_BIT * PACKET_SIZE
        self.energy -= energy
        self.battery_level -= (energy / INIT_ENERGY) * 2
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False
        return PACKET_SIZE * TIME_PER_BIT

    def consume_idle(self):
        self.energy -= IDLE_ENERGY
        self.battery_level -= 0.01
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False

class HEEDSimulation:
    def __init__(self, dataset_path):
        self.dataset_loader = DatasetLoader(dataset_path)
        self.nodes = {}
        self.stats = {
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

    def run(self):
        if not self.dataset_loader.load_dataset():
            return None

        for i, sensor_id in enumerate(self.dataset_loader.sensor_data.keys()):
            first_record = self.dataset_loader.sensor_data[sensor_id][0]
            x, y = random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)
            node = Node(sensor_id, x, y, first_record['batterylevel'])
            self.nodes[sensor_id] = node

        time_windows = self.dataset_loader.get_time_windows()
        random.seed(RANDOM_SEED)
        round_num = 0
        num_nodes = len(self.nodes)

        for window in time_windows:
            round_num += 1
            for r in window:
                sensor_id = r['sensor_id']
                battery_vals = [d['batterylevel'] for d in window if d['sensor_id'] == sensor_id]
                avg_battery = sum(battery_vals) / len(battery_vals)
                if sensor_id in self.nodes:
                    self.nodes[sensor_id].update_battery(avg_battery)

            alive_nodes = [n for n in self.nodes.values() if n.is_alive()]
            if not alive_nodes:
                self.stats['last_death'] = round_num
                break

            for n in alive_nodes:
                n.is_CH = False
                n.cluster_head = None
                n.consume_idle()

            CHs = []
            for node in alive_nodes:
                prob = min(1.0, 0.1 * (node.energy / INIT_ENERGY) * (node.battery_level / 100.0))
                if random.random() < prob:
                    node.is_CH = True
                    CHs.append(node)
            if not CHs:
                best_node = max(alive_nodes, key=lambda n: n.energy * n.battery_level)
                best_node.is_CH = True
                CHs.append(best_node)

            delivered, dropped, switches = 0, 0, 0
            met_deadline, missed_deadline = 0, 0
            round_delay, cluster_members = 0.0, defaultdict(list)

            for r in window:
                sensor_id = r['sensor_id']
                if sensor_id not in self.nodes:
                    dropped += 1
                    continue
                node = self.nodes[sensor_id]
                if not node.is_alive():
                    dropped += 1
                    continue
                if not node.is_CH:
                    closest_ch = min(CHs, key=lambda ch: node.distance_to(ch))
                    dist = node.distance_to(closest_ch)
                    node.cluster_head = closest_ch
                    if node.prev_ch_id is not None and node.prev_ch_id != closest_ch.id:
                        switches += 1
                    node.prev_ch_id = closest_ch.id
                    tx1 = node.consume_tx(dist)
                    if node.is_alive():
                        rx = closest_ch.consume_rx()
                        if closest_ch.is_alive():
                            cluster_members[closest_ch.id].append(node)
                            tx2 = closest_ch.consume_tx(closest_ch.distance_to_bs)
                            delay = tx1 + rx + tx2
                            delivered += 1
                            round_delay += delay
                            if delay <= DEADLINE:
                                met_deadline += 1
                            else:
                                missed_deadline += 1
                        else:
                            dropped += 1
                    else:
                        dropped += 1
                else:
                    tx = node.consume_tx(node.distance_to_bs)
                    if node.is_alive():
                        delivered += 1
                        round_delay += tx
                        if tx <= DEADLINE:
                            met_deadline += 1
                        else:
                            missed_deadline += 1
                    else:
                        dropped += 1

            self.stats['alive_nodes'].append(sum(n.is_alive() for n in self.nodes.values()))
            self.stats['total_energy'].append(sum(n.energy for n in self.nodes.values() if n.is_alive()))
            self.stats['chs_per_round'].append(len(CHs))
            self.stats['delivered_packets'].append(delivered)
            self.stats['dropped_packets'].append(dropped)
            self.stats['cluster_switches'].append(switches)
            self.stats['load_distribution'].append(math.sqrt(sum((len(m) - len(cluster_members))**2 for m in cluster_members.values()) / len(cluster_members)) if cluster_members else 0)
            self.stats['round_delay'].append(round_delay)
            self.stats['met_deadline_packets'].append(met_deadline)
            self.stats['missed_deadline_packets'].append(missed_deadline)
            self.stats['real_time_ratio'].append(met_deadline / delivered if delivered else 0)
            self.stats['pdr_over_time'].append(delivered / (delivered + dropped) if delivered + dropped else 0)
            if self.stats['alive_nodes'][-1] < num_nodes and self.stats['first_death'] is None:
                self.stats['first_death'] = round_num
            if self.stats['alive_nodes'][-1] == 0 and self.stats['last_death'] is None:
                self.stats['last_death'] = round_num

        return self.stats, round_num, num_nodes

def main():
    print("\nHEED Simulation - Wireless Sensor Network")
    print("----------------------------------------")
    
    # Check if default dataset exists
    if os.path.exists(DEFAULT_DATASET):
        use_default = input(f"Default dataset found at '{DEFAULT_DATASET}'. Use this? (y/n): ").strip().lower()
        if use_default == 'y':
            file_path = DEFAULT_DATASET
        else:
            file_path = input("Enter path to your dataset file: ").strip()
    else:
        print(f"Default dataset not found at '{DEFAULT_DATASET}'")
        file_path = input("Enter path to your dataset file: ").strip()
    
    if not file_path:
        print("Error: No dataset path given.")
        return

    print("\nRunning simulation...")
    sim = HEEDSimulation(file_path)
    result = sim.run()
    if result is None:
        print("Simulation failed.")
        return

    stats, rounds, nodes = result
    print("\nSimulation Summary")
    print("-----------------")
    print(f"Nodes: {nodes}, Rounds: {rounds}")
    print(f"First death: {stats['first_death']}, Last death: {stats['last_death']}")
    print(f"Avg CHs/Round: {sum(stats['chs_per_round']) / rounds:.2f}")
    print(f"Avg Delivered: {sum(stats['delivered_packets']) / rounds:.2f}")
    print(f"Avg Dropped: {sum(stats['dropped_packets']) / rounds:.2f}")
    print(f"Avg PDR: {sum(stats['pdr_over_time']) / rounds:.4f}")
    print(f"Avg Deadline Ratio: {sum(stats['real_time_ratio']) / rounds:.4f}")

if __name__ == "__main__":
    main()