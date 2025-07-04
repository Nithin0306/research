#!/usr/bin/env python3

import ns3
import sys
import os
import csv
import math
import random
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# NS-3 specific imports
from ns3 import core, network, internet, mobility, energy, applications, wifi, propagation

class NS3WirelessSensorNetworkSimulator:
    def __init__(self, file_path: str, area_size: int = 100, p: float = 0.1,
                 init_energy: float = 2.0, freq: str = 'H'):
        # Initialize NS-3 parameters
        self.area_size = area_size
        self.p = p
        self.init_energy = init_energy
        self.freq = freq
        
        # Energy model parameters
        self.tx_energy_per_bit = 50e-9
        self.rx_energy_per_bit = 50e-9
        self.fs_energy = 10e-12
        self.mp_energy = 0.0013e-12
        self.data_packet_size = 4000
        self.ctrl_packet_size = 200
        self.idle_energy_per_round = 0.0001
        self.radio_speed = 2e5
        
        # Base station location
        self.bs_x, self.bs_y = area_size / 2, 110
        
        # NS-3 containers
        self.nodes = network.NodeContainer()
        self.devices = None
        self.interfaces = None
        
        # Load and preprocess data
        self.df = self._load_dataset(file_path)
        self.node_list = self._initialize_nodes()
        self.num_nodes = len(self.node_list)
        self.rounds = self._get_total_rounds()
        
        # Statistics storage
        self.stats = {
            'alive_nodes': [], 'total_energy': [], 'chs_per_round': [],
            'pdr': [], 'avg_delay': [], 'throughput': [], 'energy_efficiency': [],
            'cluster_balance': [], 'first_death': None, 'last_death': None,
            'control_overhead': []
        }
        
        # Initialize NS-3 simulation
        self._setup_ns3_simulation()

    def _load_dataset(self, file_path: str) -> List[Dict]:
        """Load and preprocess the sensor dataset with proper field mapping."""
        try:
            dataset = []
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Validate required columns
                required_cols = ['sensor_id', 'date_d_m_y', 'time', 'batterylevel']
                fieldnames = reader.fieldnames
                missing_cols = [col for col in required_cols if col not in fieldnames]
                
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                print(f"Dataset columns found: {fieldnames}")
                
                for row in reader:
                    try:
                        # Parse datetime
                        date_str = f"{row['date_d_m_y']} {row['time']}"
                        dt = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
                        
                        # Convert battery level to energy (assuming batterylevel is in percentage)
                        energy = float(row['batterylevel']) / 100.0 * self.init_energy
                        
                        record = {
                            'sort_id': int(row.get('sort_id', 0)),
                            'datetime': dt,
                            'sensor_id': int(row['sensor_id']),
                            'sensor_type': row.get('sensor_type', 'default'),
                            'temp_C': float(row.get('temp_C', 0)),
                            'hpa_div_4': float(row.get('hpa_div_4', 0)),
                            'pressure': float(row.get('hpa_div_4', 0)) * 4,  # Convert back to hPa
                            'batterylevel': float(row['batterylevel']),
                            'sensor_cycle': int(row.get('sensor_cycle', 0)),
                            'energy': energy,
                            'active': float(row['batterylevel']) > 0
                        }
                        dataset.append(record)
                        
                    except (ValueError, KeyError) as e:
                        print(f"Skipping invalid row: {e}")
                        continue
            
            # Sort by datetime and create rounds
            dataset.sort(key=lambda x: x['datetime'])
            
            # Create rounds based on time frequency
            if dataset:
                start_time = dataset[0]['datetime']
                for record in dataset:
                    if self.freq == 'H':
                        round_num = int((record['datetime'] - start_time).total_seconds() / 3600)
                    elif self.freq == 'D':
                        round_num = (record['datetime'] - start_time).days
                    else:  # Default to hourly
                        round_num = int((record['datetime'] - start_time).total_seconds() / 3600)
                    
                    record['round'] = round_num
            
            print(f"Dataset loaded successfully with {len(dataset)} records")
            unique_sensors = set(record['sensor_id'] for record in dataset)
            unique_rounds = set(record['round'] for record in dataset)
            print(f"Unique sensors: {len(unique_sensors)}")
            print(f"Total rounds: {len(unique_rounds)}")
            
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def _get_total_rounds(self) -> int:
        """Get total number of rounds from dataset."""
        if not self.df:
            return 0
        return max(record['round'] for record in self.df) + 1

    def _setup_ns3_simulation(self) -> None:
        """Setup NS-3 simulation environment."""
        # Enable logging
        core.LogComponentEnable("UdpEchoClientApplication", core.LOG_LEVEL_INFO)
        core.LogComponentEnable("UdpEchoServerApplication", core.LOG_LEVEL_INFO)
        
        # Create nodes
        self.nodes.Create(self.num_nodes + 1)  # +1 for base station
        
        # Setup mobility model
        mobility = mobility.MobilityHelper()
        positionAlloc = mobility.CreateObjectByTypeId("ns3::ListPositionAllocator")
        
        # Add node positions
        for node_info in self.node_list:
            positionAlloc.Add(core.Vector(node_info['x'], node_info['y'], 0.0))
        
        # Add base station position
        positionAlloc.Add(core.Vector(self.bs_x, self.bs_y, 0.0))
        
        mobility.SetPositionAllocator(positionAlloc)
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mobility.Install(self.nodes)
        
        # Setup WiFi
        wifi_helper = wifi.WifiHelper()
        wifi_helper.SetStandard(wifi.WIFI_PHY_STANDARD_80211b)
        
        # WiFi MAC and PHY
        wifi_mac = wifi.WifiMacHelper()
        wifi_mac.SetType("ns3::AdhocWifiMac")
        
        wifi_phy = wifi.YansWifiPhyHelper()
        wifi_channel = wifi.YansWifiChannelHelper()
        wifi_channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel")
        wifi_channel.AddPropagationLoss("ns3::FriisPropagationLossModel")
        wifi_phy.SetChannel(wifi_channel.Create())
        
        self.devices = wifi_helper.Install(wifi_phy, wifi_mac, self.nodes)
        
        # Setup Internet stack
        internet_stack = internet.InternetStackHelper()
        internet_stack.Install(self.nodes)
        
        # Assign IP addresses
        ipv4 = internet.Ipv4AddressHelper()
        ipv4.SetBase("10.1.1.0", "255.255.255.0")
        self.interfaces = ipv4.Assign(self.devices)
        
        # Setup energy model
        self._setup_energy_model()

    def _setup_energy_model(self) -> None:
        """Setup energy model for sensor nodes."""
        # Basic energy source
        energy_source_helper = energy.BasicEnergySourceHelper()
        energy_source_helper.Set("BasicEnergySourceInitialEnergyJ", 
                                core.DoubleValue(self.init_energy))
        
        # Install energy source on nodes (excluding base station)
        for i in range(self.num_nodes):
            energy_source_helper.Install(self.nodes.Get(i))

    class SensorNode:
        """Represents a sensor node in the network."""
        def __init__(self, node_id: int, x: float, y: float, initial_energy: float, 
                     sensor_type: str = 'default'):
            self.id = node_id
            self.x = x
            self.y = y
            self.energy = initial_energy
            self.initial_energy = initial_energy
            self.is_CH = False
            self.cluster_head = None
            self.alive = True
            self.distance_to_bs = None
            self.data_packets_sent = 0
            self.ctrl_packets_sent = 0
            self.packets_received = 0
            self.ch_history = 0
            self.last_CH_round = -10
            self.lifetime = 0
            self.cluster_member_count = 0
            self.sensor_type = sensor_type
            self.real_energy_history = []

        def is_alive(self) -> bool:
            return self.alive and self.energy > 0

        def distance_to(self, other) -> float:
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

        def update_real_energy(self, real_energy: float) -> None:
            """Update node energy based on real dataset values."""
            self.real_energy_history.append(self.energy)
            self.energy = real_energy
            if self.energy <= 0:
                self.alive = False

        def consume_tx(self, distance: float, packet_type: str = 'data') -> None:
            """Calculate transmission energy consumption."""
            pkt_size = 4000 if packet_type == 'data' else 200
            d_threshold = 75
            
            if distance < d_threshold:
                energy = (50e-9 + 10e-12 * distance**2) * pkt_size
            else:
                energy = (50e-9 + 0.0013e-12 * distance**4) * pkt_size
            
            self.energy -= energy
            
            if packet_type == 'data':
                self.data_packets_sent += 1
            else:
                self.ctrl_packets_sent += 1
            
            if self.energy <= 0:
                self.alive = False

        def consume_rx(self, packet_type: str = 'data') -> None:
            """Calculate reception energy consumption."""
            pkt_size = 4000 if packet_type == 'data' else 200
            energy = 50e-9 * pkt_size
            self.energy -= energy
            self.packets_received += 1
            
            if self.energy <= 0:
                self.alive = False

        def consume_idle(self) -> None:
            """Calculate idle energy consumption."""
            self.energy -= 0.0001
            if self.energy <= 0:
                self.alive = False

        def mark_participation(self, was_in_cluster: bool) -> None:
            """Track node participation metrics."""
            if self.is_alive():
                self.lifetime += 1
                if was_in_cluster:
                    self.cluster_member_count += 1

    def _initialize_nodes(self) -> List[SensorNode]:
        """Initialize sensor nodes from the dataset."""
        nodes = []
        
        # Get unique sensors and their initial data
        unique_sensors = set(record['sensor_id'] for record in self.df)
        unique_sensors = sorted(list(unique_sensors))
        
        # Generate positions for each sensor
        random.seed(42)  # For reproducible results
        
        for sensor_id in unique_sensors:
            # Generate random position
            x = random.uniform(0, self.area_size)
            y = random.uniform(0, self.area_size)
            
            # Get initial energy and sensor type from first record
            sensor_records = [r for r in self.df if r['sensor_id'] == sensor_id]
            if sensor_records:
                first_record = sensor_records[0]
                initial_energy = first_record['energy']
                sensor_type = first_record['sensor_type']
            else:
                initial_energy = self.init_energy
                sensor_type = 'default'
            
            node = self.SensorNode(sensor_id, x, y, initial_energy, sensor_type)
            node.distance_to_bs = math.sqrt((x - self.bs_x)**2 + (y - self.bs_y)**2)
            nodes.append(node)
        
        return nodes

    def _get_round_data(self, round_num: int) -> Dict:
        """Get sensor data for a specific round."""
        round_records = [r for r in self.df if r['round'] == round_num]
        
        sensor_metrics = {}
        for record in round_records:
            sensor_id = record['sensor_id']
            sensor_metrics[sensor_id] = {
                'energy': record['energy'],
                'batterylevel': record['batterylevel'],
                'active': record['active'],
                'temperature': record['temp_C'],
                'pressure': record['pressure'],
                'cycle': record['sensor_cycle']
            }
        
        return sensor_metrics

    def _elect_cluster_heads(self, alive_nodes: List[SensorNode], round_num: int, 
                           sensor_metrics: Dict) -> List[SensorNode]:
        """Perform cluster head election with enhanced threshold calculation using real data."""
        CHs = []
        
        for node in alive_nodes:
            # Update node energy from real data
            if node.id in sensor_metrics:
                node.update_real_energy(sensor_metrics[node.id]['energy'])
            
            # Skip if node is not alive after energy update
            if not node.is_alive():
                continue
            
            # Enforce cooldown period
            if (round_num - node.last_CH_round) < (1 / self.p):
                continue
            
            # Enhanced threshold calculation incorporating real energy data
            energy_ratio = node.energy / node.initial_energy if node.initial_energy > 0 else 0
            threshold = (
                self.p * energy_ratio * 
                (1 / (1 + node.ch_history * self.p))
            )
            
            if random.random() < threshold:
                node.is_CH = True
                node.ch_history += 1
                node.last_CH_round = round_num
                CHs.append(node)
        
        # Fallback mechanism if no CHs elected
        if not CHs:
            eligible_nodes = [n for n in alive_nodes if n.is_alive()]
            if eligible_nodes:
                num_ch = max(1, int(0.05 * len(eligible_nodes)))
                CHs = sorted(eligible_nodes, key=lambda n: n.energy, reverse=True)[:num_ch]
                for ch in CHs:
                    ch.is_CH = True
                    ch.ch_history += 1
                    ch.last_CH_round = round_num
        
        return CHs

    def _form_clusters(self, alive_nodes: List[SensorNode], CHs: List[SensorNode]) -> Tuple[
        Dict[int, List[SensorNode]], int, int, int, List[float]]:
        """Form clusters and calculate communication metrics."""
        cluster_members = defaultdict(list)
        total_data_tx = total_data_rx = total_ctrl_tx = 0
        delays = []
        
        for node in alive_nodes:
            if not node.is_CH and CHs:
                closest_ch = min(CHs, key=lambda ch: node.distance_to(ch))
                dist = node.distance_to(closest_ch)
                
                # Control packet exchange
                node.consume_tx(dist, 'ctrl')
                closest_ch.consume_rx('ctrl')
                total_ctrl_tx += 1
                
                # Cluster formation
                node.cluster_head = closest_ch
                cluster_members[closest_ch.id].append(node)
                delays.append(dist / self.radio_speed)
        
        return cluster_members, total_data_tx, total_data_rx, total_ctrl_tx, delays

    def _data_transmission(self, CHs: List[SensorNode], cluster_members: Dict[int, List[SensorNode]],
                          total_data_tx: int, total_data_rx: int) -> Tuple[int, int]:
        """Handle intra-cluster and CH-to-BS data transmission."""
        for ch in CHs:
            if not ch.is_alive():
                continue
            
            # Intra-cluster communication
            for member in cluster_members.get(ch.id, []):
                if member.is_alive():
                    member.consume_tx(member.distance_to(ch), 'data')
                    ch.consume_rx('data')
                    total_data_tx += 1
                    total_data_rx += 1
            
            # CH to BS communication
            ch.consume_tx(ch.distance_to_bs, 'data')
            total_data_tx += 1
        
        return total_data_tx, total_data_rx

    def _update_stats(self, round_num: int, alive_count: int, CHs: List[SensorNode],
                     cluster_members: Dict[int, List[SensorNode]], total_data_tx: int,
                     total_data_rx: int, total_ctrl_tx: int, delays: List[float]) -> None:
        """Update simulation statistics for the current round."""
        self.stats['alive_nodes'].append(alive_count)
        self.stats['total_energy'].append(sum(n.energy for n in self.node_list if n.is_alive()))
        self.stats['chs_per_round'].append(len(CHs))
        
        # Network performance metrics
        self.stats['pdr'].append(
            total_data_rx / total_data_tx if total_data_tx > 0 else 0
        )
        self.stats['avg_delay'].append(sum(delays) / len(delays) if delays else 0)
        self.stats['throughput'].append(total_data_rx)
        self.stats['energy_efficiency'].append(
            total_data_rx * self.data_packet_size / self.stats['total_energy'][-1]
            if self.stats['total_energy'][-1] > 0 else 0
        )
        self.stats['control_overhead'].append(
            total_ctrl_tx / (total_data_tx + 1e-10)
        )
        
        # Cluster metrics
        if CHs:
            cluster_sizes = [len(cluster_members[ch.id]) for ch in CHs]
            self.stats['cluster_balance'].append(
                math.sqrt(sum((x - sum(cluster_sizes)/len(cluster_sizes))**2 for x in cluster_sizes) / len(cluster_sizes))
            )
        else:
            self.stats['cluster_balance'].append(0)
        
        # Track first and last deaths
        if alive_count < self.num_nodes and self.stats['first_death'] is None:
            self.stats['first_death'] = round_num + 1

    def run_simulation(self) -> None:
        """Run the complete simulation using real dataset."""
        print("Starting NS-3 LEACH simulation with real dataset...")
        
        for round_num in range(self.rounds):
            # Get real sensor data for this round
            sensor_metrics = self._get_round_data(round_num)
            
            # Get alive nodes that are active in this round
            alive_nodes = [n for n in self.node_list if n.is_alive() and n.id in sensor_metrics
                          and sensor_metrics[n.id]['active']]
            
            if not alive_nodes:
                self.stats['last_death'] = round_num + 1
                print(f"All nodes dead at round {round_num + 1}")
                break
            
            # Reset node states and consume idle energy
            for node in alive_nodes:
                node.is_CH = False
                node.cluster_head = None
                node.consume_idle()
            
            # Cluster head election using real data
            CHs = self._elect_cluster_heads(alive_nodes, round_num, sensor_metrics)
            
            # Cluster formation
            cluster_members, total_data_tx, total_data_rx, total_ctrl_tx, delays = \
                self._form_clusters(alive_nodes, CHs)
            
            # Data transmission phase
            total_data_tx, total_data_rx = self._data_transmission(
                CHs, cluster_members, total_data_tx, total_data_rx
            )
            
            # Update node participation metrics
            for node in alive_nodes:
                node.mark_participation(node.cluster_head is not None)
            
            # Update statistics
            alive_count = sum(n.is_alive() for n in self.node_list)
            self._update_stats(
                round_num, alive_count, CHs, cluster_members,
                total_data_tx, total_data_rx, total_ctrl_tx, delays
            )
            
            # Progress update
            if round_num % 50 == 0:
                print(f"Round {round_num + 1}/{self.rounds} completed. Alive nodes: {alive_count}")

    def generate_summary(self) -> None:
        """Generate and print simulation summary."""
        # Node-level metrics
        node_data = []
        for node in self.node_list:
            stability_ratio = (
                node.cluster_member_count / node.lifetime
                if node.lifetime > 0 else 0
            )
            node_data.append([
                node.id, node.sensor_type, node.lifetime, node.ch_history,
                node.data_packets_sent, node.packets_received,
                f"{stability_ratio:.3f}", f"{node.energy:.3f}"
            ])
        
        print("\n" + "="*80)
        print("NS-3 LEACH ALGORITHM PERFORMANCE ANALYSIS")
        print("="*80)
        print(f"Dataset Analysis Complete!")
        print(f"Total Nodes Analyzed: {self.num_nodes}")
        print(f"Total Rounds Simulated: {len(self.stats['alive_nodes'])}")
        
        print("\n=== Node Performance Metrics ===")
        headers = ['Node ID', 'Sensor Type', 'Lifetime (Rounds)', 'CH Count',
                  'Data Sent', 'Data Received', 'Stability Ratio', 'Final Energy']
        
        # Sort by lifetime (descending)
        node_data.sort(key=lambda x: x[2], reverse=True)
        
        # Print headers
        print(" | ".join(f"{h:>15}" for h in headers))
        print("-" * (len(headers) * 18))
        
        # Print node data
        for row in node_data:
            print(" | ".join(f"{str(val):>15}" for val in row))
        
        # Network-level summary
        print("\n=== Network Lifetime Summary ===")
        print(f"Total Rounds: {len(self.stats['alive_nodes'])}")
        print(f"First Node Death: Round {self.stats['first_death'] or 'N/A'}")
        print(f"Last Node Death: Round {self.stats['last_death'] or 'N/A'}")
        print(f"Average Alive Nodes: {sum(self.stats['alive_nodes'])/len(self.stats['alive_nodes']):.2f}")
        print(f"Average Cluster Heads: {sum(self.stats['chs_per_round'])/len(self.stats['chs_per_round']):.2f}")
        
        print("\n=== Performance Metrics ===")
        print(f"Average PDR: {sum(self.stats['pdr'])/len(self.stats['pdr']):.4f}")
        print(f"Average Throughput: {sum(self.stats['throughput'])/len(self.stats['throughput']):.2f} packets/round")
        print(f"Average Delay: {sum(self.stats['avg_delay'])/len(self.stats['avg_delay']):.6f} seconds")
        print(f"Energy Efficiency: {sum(self.stats['energy_efficiency'])/len(self.stats['energy_efficiency']):.2f} bits/Joule")
        print(f"Control Overhead: {sum(self.stats['control_overhead'])/len(self.stats['control_overhead']):.4f} ctrl/data ratio")
        print(f"Cluster Balance (Std Dev): {sum(self.stats['cluster_balance'])/len(self.stats['cluster_balance']):.3f}")
        
        # Save results to file
        results_file = 'ns3_leach_analysis_results.csv'
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(node_data)
        
        print(f"\nResults saved to: {results_file}")

def main():
    """Main function to run the NS-3 LEACH simulation."""
    # Replace 'sensor_data.csv' with your actual dataset file path
    dataset_path = "sensor_data.csv"
    
    try:
        # Initialize the NS-3 simulator
        simulator = NS3WirelessSensorNetworkSimulator(
            file_path=dataset_path,
            area_size=100,
            p=0.1,
            init_energy=2.0,
            freq='H'  # Hourly rounds - adjust based on your data frequency
        )
        
        # Run simulation
        simulator.run_simulation()
        
        # Generate summary
        simulator.generate_summary()
        
        print("\nNS-3 LEACH simulation completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found.")
        print("Please ensure your dataset file is in the same directory or provide the correct path.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your dataset format and ensure all required columns are present.")

if __name__ == "__main__":
    main()
