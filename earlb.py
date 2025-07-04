#!/usr/bin/env python3

import sys
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import time

# NS-3 Python bindings import
try:
    import ns3
    from ns3 import core, network, internet, applications, mobility, wifi, csma
    NS3_AVAILABLE = True
    print("NS-3 modules imported successfully")
except ImportError as e:
    print(f"NS-3 import error: {e}")
    print("Please ensure NS-3 is properly installed with Python bindings")
    sys.exit(1)

# Simulation Parameters
SIMULATION_SEED = 42
AREA_SIZE = 100  # Network area in meters
INIT_ENERGY = 2.0  # Initial energy in Joules
BASE_STATION_POS = (AREA_SIZE / 2, 110)  # Base station position

# Energy Model Parameters
TX_ENERGY_PER_BIT = 50e-9  # Transmission energy per bit (J/bit)
RX_ENERGY_PER_BIT = 50e-9  # Reception energy per bit (J/bit)
FS_ENERGY = 10e-12  # Free space energy coefficient
MP_ENERGY = 0.0013e-12  # Multipath energy coefficient
PACKET_SIZE = 4000  # Packet size in bits
IDLE_ENERGY = 0.0001  # Idle energy consumption (J/s)
TIME_PER_BIT = 1e-6  # Time per bit transmission (seconds)
DEADLINE = 0.1  # Real-time deadline (100 ms)

class NS3WSNNode:
    """Enhanced WSN Node class for NS-3 integration"""
    
    def __init__(self, node_id, x, y, initial_battery=100, sensor_type=None):
        self.id = node_id
        self.x = float(x)
        self.y = float(y)
        self.energy = INIT_ENERGY
        self.battery_level = float(initial_battery)
        self.sensor_type = sensor_type
        self.is_CH = False
        self.cluster_head = None
        self.prev_ch_id = None
        self.alive = True
        self.distance_to_bs = np.sqrt((self.x - BASE_STATION_POS[0])**2 + (self.y - BASE_STATION_POS[1])**2)
        
        # Data tracking
        self.temperature = 0.0
        self.pressure = 0.0
        self.sensor_cycle = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.data_records = []
        
        # NS-3 specific
        self.ns3_node = None
        self.ns3_device = None
        
    def is_alive(self):
        """Check if node is alive"""
        return self.alive and self.energy > 0 and self.battery_level > 0
    
    def distance_to(self, other_node):
        """Calculate distance to another node"""
        return np.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)
    
    def update_battery(self, new_level):
        """Update battery level and corresponding energy"""
        self.battery_level = max(0.0, float(new_level))
        self.energy = INIT_ENERGY * (self.battery_level / 100.0)
        if self.battery_level <= 0:
            self.alive = False
    
    def update_sensor_data(self, temp_c, hpa_div_4, sensor_cycle):
        """Update sensor readings"""
        self.temperature = float(temp_c)
        self.pressure = float(hpa_div_4)
        self.sensor_cycle = int(sensor_cycle)
    
    def consume_transmission_energy(self, distance):
        """Calculate and consume energy for transmission"""
        if not self.is_alive():
            return 0.0
        
        # Energy consumption model
        d_threshold = 75  # Distance threshold for energy model
        if distance < d_threshold:
            energy = TX_ENERGY_PER_BIT * PACKET_SIZE + FS_ENERGY * PACKET_SIZE * (distance ** 2)
        else:
            energy = TX_ENERGY_PER_BIT * PACKET_SIZE + MP_ENERGY * PACKET_SIZE * (distance ** 4)
        
        # Apply energy consumption
        self.energy -= energy
        battery_drain = (energy / INIT_ENERGY) * 5.0  # Battery drain factor
        self.battery_level -= battery_drain
        
        # Check if node dies
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False
            
        return PACKET_SIZE * TIME_PER_BIT  # Return transmission delay
    
    def consume_reception_energy(self):
        """Calculate and consume energy for reception"""
        if not self.is_alive():
            return 0.0
        
        energy = RX_ENERGY_PER_BIT * PACKET_SIZE
        self.energy -= energy
        battery_drain = (energy / INIT_ENERGY) * 2.0  # Battery drain factor
        self.battery_level -= battery_drain
        
        # Check if node dies
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False
            
        return PACKET_SIZE * TIME_PER_BIT  # Return reception delay
    
    def consume_idle_energy(self):
        """Consume idle energy"""
        if not self.is_alive():
            return
        
        self.energy -= IDLE_ENERGY
        self.battery_level -= 0.01  # Small battery drain for idle
        
        if self.energy <= 0 or self.battery_level <= 0:
            self.alive = False

class NS3HEEDSimulation:
    """NS-3 integrated HEED simulation with dataset analysis"""
    
    def __init__(self):
        self.nodes = {}
        self.round_num = 0
        self.total_transmissions = 0
        self.ns3_nodes = None
        self.ns3_devices = None
        
        # Initialize statistics tracking
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
        
        # Initialize NS-3 environment
        self.setup_ns3_environment()
    
    def setup_ns3_environment(self):
        """Initialize NS-3 simulation environment"""
        try:
            # Set random seed for reproducibility
            ns3.core.RngSeedManager.SetSeed(SIMULATION_SEED)
            ns3.core.RngSeedManager.SetRun(1)
            
            # Enable logging (optional)
            ns3.core.LogComponentEnable("UdpEchoClientApplication", ns3.core.LOG_LEVEL_INFO)
            ns3.core.LogComponentEnable("UdpEchoServerApplication", ns3.core.LOG_LEVEL_INFO)
            
            print("NS-3 environment initialized successfully")
            
        except Exception as e:
            print(f"Error setting up NS-3 environment: {e}")
            raise
    
    def load_and_validate_dataset(self, file_path):
        """Load and validate the dataset"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            # Load dataset
            print(f"Loading dataset from: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Dataset loaded: {len(df)} records")
            
            # Validate required columns
            required_cols = ['sort_id', 'date_d_m_y', 'time', 'sensor_id', 'sensor_type', 
                           'temp_C', 'hpa_div_4', 'batterylevel', 'sensor_cycle']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Data preprocessing
            print("Preprocessing dataset...")
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['date_d_m_y'] + ' ' + df['time'], 
                                         format='%d/%m/%Y %H:%M:%S', errors='coerce')
            
            # Remove invalid datetime entries
            initial_count = len(df)
            df = df.dropna(subset=['datetime'])
            print(f"Removed {initial_count - len(df)} records with invalid datetime")
            
            # Sort by datetime and sensor_id
            df = df.sort_values(['datetime', 'sensor_id'])
            
            # Clean numeric columns
            numeric_cols = ['temp_C', 'hpa_div_4', 'batterylevel', 'sensor_cycle']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with invalid numeric data
            initial_count = len(df)
            df = df.dropna(subset=numeric_cols)
            print(f"Removed {initial_count - len(df)} records with invalid numeric data")
            
            # Validate battery levels (should be 0-100)
            df = df[(df['batterylevel'] >= 0) & (df['batterylevel'] <= 100)]
            
            # Get unique sensors
            unique_sensors = df['sensor_id'].unique()
            print(f"Found {len(unique_sensors)} unique sensors")
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def initialize_ns3_nodes(self, df):
        """Initialize NS-3 nodes from dataset"""
        try:
            # Get unique sensors and their initial data
            sensor_info = df.groupby('sensor_id').agg({
                'sensor_type': 'first',
                'batterylevel': 'first',
                'temp_C': 'first',
                'hpa_div_4': 'first',
                'sensor_cycle': 'first'
            }).reset_index()
            
            num_nodes = len(sensor_info)
            print(f"Initializing {num_nodes} NS-3 nodes...")
            
            # Create NS-3 node container
            self.ns3_nodes = ns3.network.NodeContainer()
            self.ns3_nodes.Create(num_nodes)
            
            # Create mobility model
            mobility = ns3.mobility.MobilityHelper()
            mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
            
            # Grid-based topology for better network structure
            grid_size = int(np.ceil(np.sqrt(num_nodes)))
            cell_size = AREA_SIZE / grid_size
            
            # Initialize nodes
            for i, row in sensor_info.iterrows():
                sensor_id = row['sensor_id']
                sensor_type = row['sensor_type']
                initial_battery = row['batterylevel']
                
                # Calculate position in grid with randomization
                grid_x = i % grid_size
                grid_y = i // grid_size
                
                x = grid_x * cell_size + random.uniform(5, cell_size - 5)
                y = grid_y * cell_size + random.uniform(5, cell_size - 5)
                
                # Ensure within bounds
                x = max(5, min(x, AREA_SIZE - 5))
                y = max(5, min(y, AREA_SIZE - 5))
                
                # Create WSN node
                node = NS3WSNNode(sensor_id, x, y, initial_battery, sensor_type)
                node.update_sensor_data(row['temp_C'], row['hpa_div_4'], row['sensor_cycle'])
                
                # Set NS-3 node reference
                node.ns3_node = self.ns3_nodes.Get(i)
                
                # Set position
                pos = ns3.mobility.Vector(x, y, 0.0)
                node.ns3_node.GetObject(ns3.mobility.MobilityModel.GetTypeId()).SetPosition(pos)
                
                self.nodes[sensor_id] = node
                
                if i < 10:  # Print first 10 nodes
                    print(f"Node {sensor_id}: Pos({x:.1f},{y:.1f}), Type:{sensor_type}, Battery:{initial_battery}%")
            
            # Install mobility
            mobility.Install(self.ns3_nodes)
            
            print(f"Successfully initialized {len(self.nodes)} nodes")
            return True
            
        except Exception as e:
            print(f"Error initializing NS-3 nodes: {e}")
            return False
    
    def create_time_windows(self, df, window_minutes=30):
        """Create time windows for simulation rounds"""
        df['time_window'] = df['datetime'].dt.floor(f'{window_minutes}min')
        time_windows = list(df.groupby('time_window'))
        print(f"Created {len(time_windows)} time windows of {window_minutes} minutes each")
        return time_windows
    
    def heed_cluster_formation(self, alive_nodes):
        """HEED clustering algorithm implementation"""
        cluster_heads = []
        
        # Reset CH status
        for node in alive_nodes:
            node.is_CH = False
        
        # HEED CH election
        for node in alive_nodes:
            # Calculate factors for CH probability
            energy_factor = node.energy / INIT_ENERGY
            battery_factor = node.battery_level / 100.0
            cycle_factor = max(0.1, 1.0 - (node.sensor_cycle / 1000.0))
            
            # HEED probability calculation
            ch_probability = min(1.0, 0.08 * energy_factor * battery_factor * cycle_factor)
            
            # CH selection
            if random.random() < ch_probability:
                node.is_CH = True
                cluster_heads.append(node)
        
        # Ensure minimum number of CHs
        if not cluster_heads:
            # Select best nodes as CHs
            alive_nodes.sort(key=lambda n: (n.energy * n.battery_level * (1000 - n.sensor_cycle)), reverse=True)
            num_chs = max(1, len(alive_nodes) // 10)
            cluster_heads = alive_nodes[:num_chs]
            for ch in cluster_heads:
                ch.is_CH = True
        
        return cluster_heads
    
    def simulate_transmission(self, sender, receiver):
        """Simulate packet transmission between nodes"""
        if not sender.is_alive() or not receiver.is_alive():
            return False, 0.0
        
        # Calculate distance
        distance = sender.distance_to(receiver)
        
        # Transmission energy and delay
        tx_delay = sender.consume_transmission_energy(distance)
        rx_delay = receiver.consume_reception_energy()
        
        # Update counters
        sender.packets_sent += 1
        
        # Check if transmission successful
        if sender.is_alive() and receiver.is_alive():
            receiver.packets_received += 1
            return True, tx_delay + rx_delay
        
        return False, 0.0
    
    def simulate_round(self, window_data, cluster_heads):
        """Simulate one communication round"""
        # Initialize round metrics
        delivered_packets = 0
        dropped_packets = 0
        cluster_switches = 0
        met_deadline = 0
        missed_deadline = 0
        total_delay = 0.0
        load_distribution = defaultdict(int)
        
        # Process each data record in the window
        for _, record in window_data.iterrows():
            sensor_id = record['sensor_id']
            self.total_transmissions += 1
            
            # Check if node exists and is alive
            if sensor_id not in self.nodes:
                dropped_packets += 1
                continue
            
            node = self.nodes[sensor_id]
            if not node.is_alive():
                dropped_packets += 1
                continue
            
            # Update node with current data
            node.update_sensor_data(record['temp_C'], record['hpa_div_4'], record['sensor_cycle'])
            
            # Handle CH nodes
            if node.is_CH:
                # Direct transmission to base station
                bs_distance = node.distance_to_bs
                delay = node.consume_transmission_energy(bs_distance)
                
                if node.is_alive():
                    delivered_packets += 1
                    total_delay += delay
                    load_distribution[node.id] += 1
                    
                    # Check deadline
                    if delay <= DEADLINE:
                        met_deadline += 1
                    else:
                        missed_deadline += 1
                else:
                    dropped_packets += 1
            
            # Handle non-CH nodes
            else:
                # Find alive cluster heads
                alive_chs = [ch for ch in cluster_heads if ch.is_alive()]
                if not alive_chs:
                    dropped_packets += 1
                    continue
                
                # Find closest CH
                closest_ch = min(alive_chs, key=lambda ch: node.distance_to(ch))
                
                # Track cluster switching
                if node.prev_ch_id is not None and node.prev_ch_id != closest_ch.id:
                    cluster_switches += 1
                node.prev_ch_id = closest_ch.id
                
                # Transmit to CH
                success, delay1 = self.simulate_transmission(node, closest_ch)
                
                if success:
                    load_distribution[closest_ch.id] += 1
                    
                    # CH forwards to base station
                    bs_distance = closest_ch.distance_to_bs
                    delay2 = closest_ch.consume_transmission_energy(bs_distance)
                    
                    if closest_ch.is_alive():
                        total_round_delay = delay1 + delay2
                        delivered_packets += 1
                        total_delay += total_round_delay
                        
                        # Check deadline
                        if total_round_delay <= DEADLINE:
                            met_deadline += 1
                        else:
                            missed_deadline += 1
                    else:
                        dropped_packets += 1
                else:
                    dropped_packets += 1
        
        # Calculate load distribution standard deviation
        load_values = list(load_distribution.values()) if load_distribution else [0]
        load_std = np.std(load_values)
        
        return {
            'delivered': delivered_packets,
            'dropped': dropped_packets,
            'switches': cluster_switches,
            'met_deadline': met_deadline,
            'missed_deadline': missed_deadline,
            'total_delay': total_delay,
            'load_std': load_std
        }
    
    def update_statistics(self, round_results, cluster_heads):
        """Update simulation statistics"""
        # Node statistics
        alive_count = sum(1 for node in self.nodes.values() if node.is_alive())
        total_energy = sum(node.energy for node in self.nodes.values() if node.is_alive())
        
        # Update arrays
        self.stats['alive_nodes'].append(alive_count)
        self.stats['total_energy'].append(total_energy)
        self.stats['chs_per_round'].append(len(cluster_heads))
        self.stats['delivered_packets'].append(round_results['delivered'])
        self.stats['dropped_packets'].append(round_results['dropped'])
        self.stats['cluster_switches'].append(round_results['switches'])
        self.stats['round_delay'].append(round_results['total_delay'])
        self.stats['met_deadline_packets'].append(round_results['met_deadline'])
        self.stats['missed_deadline_packets'].append(round_results['missed_deadline'])
        self.stats['load_distribution'].append(round_results['load_std'])
        
        # Calculate ratios
        total_packets = round_results['delivered'] + round_results['dropped']
        if total_packets > 0:
            pdr = round_results['delivered'] / total_packets
            self.stats['pdr_over_time'].append(pdr)
        else:
            self.stats['pdr_over_time'].append(0.0)
        
        if round_results['delivered'] > 0:
            rt_ratio = round_results['met_deadline'] / round_results['delivered']
            self.stats['real_time_ratio'].append(rt_ratio)
        else:
            self.stats['real_time_ratio'].append(0.0)
        
        # Track node deaths
        total_nodes = len(self.nodes)
        if alive_count < total_nodes and self.stats['first_death'] is None:
            self.stats['first_death'] = self.round_num
        if alive_count == 0 and self.stats['last_death'] is None:
            self.stats['last_death'] = self.round_num
    
    def run_simulation(self, dataset_path):
        """Main simulation execution"""
        print("=== Starting NS-3 HEED Simulation with Dataset ===")
        
        # Load and validate dataset
        df = self.load_and_validate_dataset(dataset_path)
        if df is None:
            print("Failed to load dataset. Exiting.")
            return None
        
        # Initialize NS-3 nodes
        if not self.initialize_ns3_nodes(df):
            print("Failed to initialize NS-3 nodes. Exiting.")
            return None
        
        # Create time windows
        time_windows = self.create_time_windows(df, window_minutes=30)
        
        print(f"Starting simulation with {len(self.nodes)} nodes and {len(time_windows)} time windows...")
        
        # Main simulation loop
        for window_idx, (time_window, window_data) in enumerate(time_windows):
            self.round_num += 1
            
            # Progress reporting
            if self.round_num % 10 == 0:
                print(f"Processing round {self.round_num}/{len(time_windows)}...")
            
            # Update battery levels from current window
            for sensor_id, sensor_data in window_data.groupby('sensor_id'):
                if sensor_id in self.nodes:
                    node = self.nodes[sensor_id]
                    # Use most recent battery level
                    latest_battery = sensor_data['batterylevel'].iloc[-1]
                    node.update_battery(latest_battery)
            
            # Get alive nodes
            alive_nodes = [node for node in self.nodes.values() if node.is_alive()]
            
            # Check for network death
            if not alive_nodes:
                print(f"All nodes died at round {self.round_num}")
                self.stats['last_death'] = self.round_num
                break
            
            # Apply idle energy consumption
            for node in alive_nodes:
                node.consume_idle_energy()
            
            # HEED cluster formation
            cluster_heads = self.heed_cluster_formation(alive_nodes)
            
            # Simulate communication round
            round_results = self.simulate_round(window_data, cluster_heads)
            
            # Update statistics
            self.update_statistics(round_results, cluster_heads)
            
            # Periodic progress report
            if self.round_num % 50 == 0:
                alive_count = len(alive_nodes)
                print(f"Round {self.round_num}: Alive={alive_count}, CHs={len(cluster_heads)}, "
                      f"Delivered={round_results['delivered']}, Dropped={round_results['dropped']}")
        
        print(f"Simulation completed after {self.round_num} rounds")
        
        # Return results
        return {
            'stats': self.stats,
            'total_rounds': self.round_num,
            'total_nodes': len(self.nodes),
            'final_alive_nodes': sum(1 for node in self.nodes.values() if node.is_alive())
        }
    
    def print_comprehensive_results(self, results):
        """Print detailed simulation results"""
        stats = results['stats']
        total_rounds = results['total_rounds']
        total_nodes = results['total_nodes']
        final_alive = results['final_alive_nodes']
        
        print("\n" + "="*80)
        print("NS-3 HEED SIMULATION RESULTS")
        print("="*80)
        
        # Basic simulation info
        print(f"Simulation Overview:")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Simulation rounds: {total_rounds}")
        print(f"  Final alive nodes: {final_alive}")
        print(f"  Network lifetime: {stats['last_death'] or 'Network still alive'}")
        
        # Performance metrics
        print(f"\nNetwork Performance:")
        delivered_total = sum(stats['delivered_packets'])
        dropped_total = sum(stats['dropped_packets'])
        total_packets = delivered_total + dropped_total
        
        if total_packets > 0:
            overall_pdr = delivered_total / total_packets
            print(f"  Total packets transmitted: {total_packets}")
            print(f"  Packets delivered: {delivered_total}")
            print(f"  Packets dropped: {dropped_total}")
            print(f"  Overall PDR: {overall_pdr:.4f}")
        
        if stats['delivered_packets']:
            print(f"  Average packets delivered per round: {np.mean(stats['delivered_packets']):.2f}")
            print(f"  Average packets dropped per round: {np.mean(stats['dropped_packets']):.2f}")
        
        # Clustering metrics
        if stats['chs_per_round']:
            print(f"\nClustering Performance:")
            print(f"  Average CHs per round: {np.mean(stats['chs_per_round']):.2f}")
            print(f"  Average cluster switches per round: {np.mean(stats['cluster_switches']):.2f}")
            print(f"  Average load distribution (std): {np.mean(stats['load_distribution']):.4f}")
        
        # Real-time performance
        if stats['real_time_ratio']:
            print(f"\nReal-time Performance:")
            print(f"  Average real-time ratio: {np.mean(stats['real_time_ratio']):.4f}")
            total_met = sum(stats['met_deadline_packets'])
            total_missed = sum(stats['missed_deadline_packets'])
            if total_met + total_missed > 0:
                overall_rt_ratio = total_met / (total_met + total_missed)
                print(f"  Overall real-time ratio: {overall_rt_ratio:.4f}")
        
        # Energy analysis
        if stats['total_energy']:
            initial_energy = total_nodes * INIT_ENERGY
            final_energy = stats['total_energy'][-1] if stats['total_energy'] else 0
            energy_consumed = initial_energy - final_energy
            
            print(f"\nEnergy Analysis:")
            print(f"  Initial total energy: {initial_energy:.4f} J")
            print(f"  Final total energy: {final_energy:.4f} J")
            print(f"  Energy consumed: {energy_consumed:.4f} J")
            if initial_energy > 0:
                efficiency = final_energy / initial_energy
                print(f"  Energy efficiency: {efficiency:.4f}")
        
        # Node lifetime analysis
        if stats['first_death'] is not None:
            print(f"\nNode Lifetime:")
            print(f"  First node death: Round {stats['first_death']}")
            if stats['last_death'] is not None:
                print(f"  Last node death: Round {stats['last_death']}")
                print(f"  Network lifetime: {stats['last_death']} rounds")
        
        print("="*80)

def main():
    """Main execution function"""
    print("=== NS-3 HEED Algorithm Dataset Analysis ===")
    print("This simulation analyzes WSN performance using the HEED clustering algorithm")
    print("with NS-3 network simulator integration.\n")
    
    # Get dataset path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("Enter the path to your dataset file: ").strip()
    
    # Clean path
    dataset_path = dataset_path.strip('"\'')
    
    # Validate file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return
    
    try:
        # Create and run simulation
        print("Initializing NS-3 HEED simulation...")
        simulation = NS3HEEDSimulation()
        
        # Run simulation
        results = simulation.run_simulation(dataset_path)
        
        if results is None:
            print("Simulation failed. Please check the dataset format and try again.")
            return
        
        # Print results
        simulation.print_comprehensive_results(results)
        
        print(f"\nSimulation completed successfully!")
        print(f"Dataset: {dataset_path}")
        print(f"Total simulation time: {results['total_rounds']} rounds")
        
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
