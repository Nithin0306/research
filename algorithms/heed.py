#!/usr/bin/env python3

# NS-3 compatible HEED algorithm with dataset analysis
# Modified for ns-3 environment with dataset integration

import sys
import os
import random
import math
from collections import defaultdict
from datetime import datetime
import csv

# Import ns-3 modules
try:
    import ns.core
    import ns.network
    import ns.internet
    import ns.wifi
    import ns.mobility
    import ns.applications
    import ns.netanim
    from ns.core import *
    from ns.network import *
    from ns.internet import *
    from ns.wifi import *
    from ns.mobility import *
    from ns.applications import *
except ImportError:
    print("Warning: ns-3 modules not found. Running in standalone mode.")
    # Create dummy ns module for standalone testing
    class DummyNS:
        class core:
            @staticmethod
            def Simulator_Schedule(time, callback, *args):
                pass
            @staticmethod
            def Simulator_Run():
                pass
            @staticmethod
            def Simulator_Destroy():
                pass
        class network:
            pass
    ns = DummyNS()

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

class DatasetLoader:
    """Efficient dataset loader for HEED simulation"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.sensor_data = {}
        self.time_windows = []
        
    def load_dataset(self):
        """Load and preprocess the dataset"""
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
            
            print(f"Loading dataset from: {self.file_path}")
            
            with open(self.file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    try:
                        # Parse datetime
                        dt_str = f"{row['date_d_m_y']} {row['time']}"
                        dt = datetime.strptime(dt_str, '%d/%m/%Y %H:%M:%S')
                        
                        # Create record
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
                        
                        # Group by sensor_id
                        if record['sensor_id'] not in self.sensor_data:
                            self.sensor_data[record['sensor_id']] = []
                        self.sensor_data[record['sensor_id']].append(record)
                        
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Skipping invalid row - {e}")
                        continue
            
            # Sort data by datetime
            self.data.sort(key=lambda x: x['datetime'])
            
            # Sort sensor data
            for sensor_id in self.sensor_data:
                self.sensor_data[sensor_id].sort(key=lambda x: x['datetime'])
            
            print(f"Dataset loaded: {len(self.data)} records from {len(self.sensor_data)} sensors")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def get_time_windows(self, window_size_minutes=60):
        """Divide dataset into time windows for round-based simulation"""
        if not self.data:
            return []
        
        time_windows = []
        current_window = []
        window_start = self.data[0]['datetime']
        
        for record in self.data:
            # Check if record falls within current window
            time_diff = (record['datetime'] - window_start).total_seconds() / 60
            
            if time_diff <= window_size_minutes:
                current_window.append(record)
            else:
                # Start new window
                if current_window:
                    time_windows.append(current_window)
                current_window = [record]
                window_start = record['datetime']
        
        # Add last window
        if current_window:
            time_windows.append(current_window)
        
        print(f"Created {len(time_windows)} time windows")
        return time_windows

class Node:
    """Enhanced Node class for ns-3 compatibility"""
    
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
        self.data_records = []
        self.ns3_node = None  # ns-3 node reference
        
    def is_alive(self):
        return self.alive and self.energy > 0 and self.battery_level > 0
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def update_battery(self, new_level):
        """Update battery level and scale energy accordingly"""
        self.battery_level = max(0, new_level)
        # Scale energy based on battery level
        self.energy = INIT_ENERGY * (self.battery_level / 100.0)
        if self.battery_level <= 0:
            self.alive = False
    
    def consume_tx(self, dist):
        """Consume transmission energy"""
        d_threshold = 75
        if dist < d_threshold:
            energy = TX_ENERGY_PER_BIT * PACKET_SIZE + FS_ENERGY * PACKET_SIZE * dist ** 2
        else:
            energy = TX_ENERGY_PER_BIT * PACKET_SIZE + MP_ENERGY * PACKET_SIZE * dist ** 4
        
        self.energy -= energy
        # Also reduce battery level proportionally
        battery_drain = (energy / INIT_ENERGY) * 5  # Adjust scaling factor as needed
        self.battery_level -= battery_drain
        
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False
        return PACKET_SIZE * TIME_PER_BIT  # delay
    
    def consume_rx(self):
        """Consume reception energy"""
        energy = RX_ENERGY_PER_BIT * PACKET_SIZE
        self.energy -= energy
        battery_drain = (energy / INIT_ENERGY) * 2  # Less drain for receiving
        self.battery_level -= battery_drain
        
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False
        return PACKET_SIZE * TIME_PER_BIT  # delay
    
    def consume_idle(self):
        """Consume idle energy"""
        self.energy -= IDLE_ENERGY
        self.battery_level -= 0.01  # Small idle drain
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False

class NS3HEEDSimulation:
    """NS-3 compatible HEED simulation with dataset integration"""
    
    def __init__(self, dataset_path):
        self.dataset_loader = DatasetLoader(dataset_path)
        self.nodes = {}
        self.ns3_nodes = None
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
        
    def initialize_nodes_from_dataset(self):
        """Initialize nodes based on dataset sensor IDs"""
        sensor_ids = list(self.dataset_loader.sensor_data.keys())
        
        for i, sensor_id in enumerate(sensor_ids):
            # Get initial battery level from first record
            first_record = self.dataset_loader.sensor_data[sensor_id][0]
            initial_battery = first_record['batterylevel']
            
            # Generate positions (you can modify this based on your topology)
            x = random.uniform(0, AREA_SIZE)
            y = random.uniform(0, AREA_SIZE)
            
            node = Node(sensor_id, x, y, initial_battery)
            self.nodes[sensor_id] = node
        
        print(f"Initialized {len(self.nodes)} nodes from dataset")
        return len(self.nodes)
    
    def setup_ns3_topology(self):
        """Setup ns-3 network topology"""
        try:
            # Create nodes
            nodes = ns.network.NodeContainer()
            nodes.Create(len(self.nodes))
            self.ns3_nodes = nodes
            
            # Setup WiFi
            wifi = ns.wifi.WifiHelper()
            wifi.SetStandard(ns.wifi.WIFI_PHY_STANDARD_80211b)
            
            # Configure mobility
            mobility = ns.mobility.MobilityHelper()
            mobility.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                                        "X", ns.core.StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"),
                                        "Y", ns.core.StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"))
            mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
            mobility.Install(nodes)
            
            # Link ns-3 nodes to our Node objects
            for i, (sensor_id, node) in enumerate(self.nodes.items()):
                if i < nodes.GetN():
                    node.ns3_node = nodes.Get(i)
                    
            print("NS-3 topology setup complete")
            return True
            
        except Exception as e:
            print(f"NS-3 setup error: {e}")
            return False
    
    def run_simulation(self):
        """Run the HEED simulation with dataset"""
        # Load dataset
        if not self.dataset_loader.load_dataset():
            return None
        
        # Initialize nodes
        num_nodes = self.initialize_nodes_from_dataset()
        
        # Setup ns-3 (optional)
        self.setup_ns3_topology()
        
        # Get time windows
        time_windows = self.dataset_loader.get_time_windows()
        
        if not time_windows:
            print("No time windows found in dataset")
            return None
        
        # Set random seed
        random.seed(RANDOM_SEED)
        
        # Simulation loop
        round_num = 0
        
        for window_data in time_windows:
            round_num += 1
            
            # Update node battery levels from current window data
            window_sensors = defaultdict(list)
            for record in window_data:
                window_sensors[record['sensor_id']].append(record)
            
            for sensor_id, records in window_sensors.items():
                if sensor_id in self.nodes:
                    # Use average battery level for the window
                    avg_battery = sum(r['batterylevel'] for r in records) / len(records)
                    self.nodes[sensor_id].update_battery(avg_battery)
            
            # Check alive nodes
            alive_nodes = [n for n in self.nodes.values() if n.is_alive()]
            if not alive_nodes:
                self.stats['last_death'] = round_num
                break
            
            # Idle energy consumption
            for node in alive_nodes:
                node.is_CH = False
                node.cluster_head = None
                node.consume_idle()
            
            # Cluster Head Election
            CHs = self.elect_cluster_heads(alive_nodes)
            
            # Data transmission and analysis
            self.process_data_transmission(window_data, CHs, round_num, num_nodes)
            
            # Schedule next round in ns-3 (if available)
            try:
                ns.core.Simulator.Schedule(ns.core.Seconds(1.0), self.process_round, round_num)
            except:
                pass  # Continue without ns-3 scheduling
        
        return self.stats, round_num, num_nodes
    
    def elect_cluster_heads(self, alive_nodes):
        """HEED Cluster Head Election Algorithm"""
        CHs = []
        
        for node in alive_nodes:
            # Calculate CH probability based on energy and battery level
            energy_factor = node.energy / INIT_ENERGY
            battery_factor = node.battery_level / 100.0
            
            # HEED probability function
            ch_prob = min(1.0, 0.1 * energy_factor * battery_factor)
            
            if random.random() < ch_prob:
                node.is_CH = True
                CHs.append(node)
        
        # Ensure at least one CH exists
        if not CHs:
            # Select node with highest energy*battery product
            best_node = max(alive_nodes, key=lambda n: n.energy * n.battery_level)
            best_node.is_CH = True
            CHs.append(best_node)
        
        return CHs
    
    def process_data_transmission(self, window_data, CHs, round_num, num_nodes):
        """Process data transmission for current round"""
        cluster_members = defaultdict(list)
        delivered, dropped, switches = 0, 0, 0
        met_deadline, missed_deadline = 0, 0
        load_map = defaultdict(int)
        round_delay = 0.0
        
        # Process each data record in the window
        for record in window_data:
            sensor_id = record['sensor_id']
            
            if sensor_id not in self.nodes:
                dropped += 1
                continue
            
            node = self.nodes[sensor_id]
            
            if not node.is_alive():
                dropped += 1
                continue
            
            if not node.is_CH:
                # Node is not a CH, find closest CH
                if not CHs:
                    dropped += 1
                    continue
                
                closest_ch = min(CHs, key=lambda ch: node.distance_to(ch))
                dist = node.distance_to(closest_ch)
                node.cluster_head = closest_ch
                
                # Check for CH switching
                if node.prev_ch_id is not None and node.prev_ch_id != closest_ch.id:
                    switches += 1
                node.prev_ch_id = closest_ch.id
                
                # Transmission to CH
                tx1 = node.consume_tx(dist)
                if node.is_alive():
                    rx = closest_ch.consume_rx()
                    if closest_ch.is_alive():
                        cluster_members[closest_ch.id].append(node)
                        # CH forwards to BS
                        tx2 = closest_ch.consume_tx(closest_ch.distance_to_bs)
                        total_delay = tx1 + rx + tx2
                        delivered += 1
                        round_delay += total_delay
                        
                        # Check deadline
                        if total_delay <= DEADLINE:
                            met_deadline += 1
                        else:
                            missed_deadline += 1
                    else:
                        dropped += 1
                else:
                    dropped += 1
            else:
                # Node is a CH, transmit directly to BS
                tx_delay = node.consume_tx(node.distance_to_bs)
                if node.is_alive():
                    delivered += 1
                    round_delay += tx_delay
                    if tx_delay <= DEADLINE:
                        met_deadline += 1
                    else:
                        missed_deadline += 1
                else:
                    dropped += 1
        
        # Calculate load distribution
        for ch_id, members in cluster_members.items():
            load_map[ch_id] = len(members)
        
        # Record statistics
        self.record_round_stats(CHs, delivered, dropped, switches, met_deadline, 
                               missed_deadline, load_map, round_delay, round_num, num_nodes)
    
    def record_round_stats(self, CHs, delivered, dropped, switches, met_deadline, 
                          missed_deadline, load_map, round_delay, round_num, num_nodes):
        """Record statistics for current round"""
        alive_count = sum(1 for n in self.nodes.values() if n.is_alive())
        
        self.stats['alive_nodes'].append(alive_count)
        self.stats['total_energy'].append(sum(n.energy for n in self.nodes.values() if n.is_alive()))
        self.stats['chs_per_round'].append(len(CHs))
        self.stats['delivered_packets'].append(delivered)
        self.stats['dropped_packets'].append(dropped)
        self.stats['cluster_switches'].append(switches)
        
        # Load distribution standard deviation
        if load_map:
            loads = list(load_map.values())
            mean_load = sum(loads) / len(loads)
            variance = sum((x - mean_load)**2 for x in loads) / len(loads)
            std_dev = math.sqrt(variance)
            self.stats['load_distribution'].append(std_dev)
        else:
            self.stats['load_distribution'].append(0)
        
        self.stats['round_delay'].append(round_delay)
        self.stats['met_deadline_packets'].append(met_deadline)
        self.stats['missed_deadline_packets'].append(missed_deadline)
        
        # Calculate ratios
        if delivered > 0:
            self.stats['real_time_ratio'].append(met_deadline / delivered)
        else:
            self.stats['real_time_ratio'].append(0)
        
        total_tx = delivered + dropped
        self.stats['pdr_over_time'].append(delivered / total_tx if total_tx > 0 else 0)
        
        # Track first and last death
        if alive_count < num_nodes and self.stats['first_death'] is None:
            self.stats['first_death'] = round_num
        if alive_count == 0 and self.stats['last_death'] is None:
            self.stats['last_death'] = round_num
    
    def process_round(self, round_num):
        """NS-3 scheduled round processing"""
        # This method can be called by ns-3 scheduler
        pass
    
    def print_results(self, stats, total_rounds, num_nodes):
        """Print simulation results"""
        print("\n" + "="*60)
        print("NS-3 HEED Algorithm Dataset Analysis Results")
        print("="*60)
        print(f"Total nodes analyzed: {num_nodes}")
        print(f"Rounds simulated: {total_rounds}")
        print(f"First node death at round: {stats['first_death'] or 'N/A'}")
        print(f"Last node death at round: {stats['last_death'] or 'N/A'}")
        print(f"Final alive nodes: {stats['alive_nodes'][-1] if stats['alive_nodes'] else 0}")
        
        if stats['chs_per_round']:
            print(f"Average CHs per round: {sum(stats['chs_per_round'])/len(stats['chs_per_round']):.2f}")
        if stats['delivered_packets']:
            print(f"Average Delivered Packets: {sum(stats['delivered_packets'])/len(stats['delivered_packets']):.2f}")
        if stats['dropped_packets']:
            print(f"Average Dropped Packets: {sum(stats['dropped_packets'])/len(stats['dropped_packets']):.2f}")
        if stats['pdr_over_time']:
            print(f"Average PDR: {sum(stats['pdr_over_time'])/len(stats['pdr_over_time']):.4f}")
        if stats['cluster_switches']:
            print(f"Average Cluster Switches: {sum(stats['cluster_switches'])/len(stats['cluster_switches']):.2f}")
        if stats['load_distribution']:
            print(f"Average Load Std Dev: {sum(stats['load_distribution'])/len(stats['load_distribution']):.4f}")
        if stats['round_delay']:
            print(f"Average Delay per Round (s): {sum(stats['round_delay'])/len(stats['round_delay']):.6f}")
        if stats['real_time_ratio']:
            print(f"Average Real-Time Deadline Ratio: {sum(stats['real_time_ratio'])/len(stats['real_time_ratio']):.4f}")
        
        print(f"\n{'='*30} Dataset-Specific Metrics {'='*30}")
        print(f"Total Delivered Packets: {sum(stats['delivered_packets'])}")
        print(f"Total Dropped Packets: {sum(stats['dropped_packets'])}")
        print(f"Total Packets Meeting Deadline: {sum(stats['met_deadline_packets'])}")
        print(f"Total Packets Missing Deadline: {sum(stats['missed_deadline_packets'])}")
        
        total_packets = sum(stats['delivered_packets']) + sum(stats['dropped_packets'])
        if total_packets > 0:
            print(f"Overall PDR: {sum(stats['delivered_packets']) / total_packets:.4f}")
        
        total_delivered = sum(stats['delivered_packets'])
        if total_delivered > 0:
            print(f"Overall Real-Time Ratio: {sum(stats['met_deadline_packets']) / total_delivered:.4f}")

def main():
    """Main function to run the simulation"""
    # Get dataset file path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to your dataset file: ").strip()
    
    if not file_path:
        print("Error: No dataset file path provided")
        return
    
    # Create and run simulation
    simulation = NS3HEEDSimulation(file_path)
    
    try:
        # Run ns-3 simulation
        ns.core.Simulator.Run()
        ns.core.Simulator.Destroy()
    except:
        print("Running without ns-3 scheduler...")
    
    # Run the HEED simulation
    result = simulation.run_simulation()
    
    if result is None:
        print("Simulation failed due to dataset loading error.")
        return
    
    stats, total_rounds, num_nodes = result
    
    # Print results
    simulation.print_results(stats, total_rounds, num_nodes)

if __name__ == "__main__":
    main()
